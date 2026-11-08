"""Per-op profiler for the Soup-vs-Unsloth case-1 diagnostic (soup-8a-prof).  NOT part of the TCO record.

Attached from tco_meter._attach() when TCO_PROFILE=1.  It never changes what the trainer computes; it
only (a) wraps compute_loss / training_step with timers, (b) registers forward hooks on the decoder
body and on every decoder layer, (c) runs torch.profiler over a fixed window of steps, (d) reads the
CUDA allocator counters at phase boundaries, (e) takes one allocator snapshot in one extra step.

Step timeline (global step numbers are 1-based, as in the TCO meter):
    steps 1..W            warm-up (W = 5, unprofiled; allocator marks are taken)
    steps W+1..W+N        profiled window (N = 3): torch.profiler + record_function ranges
    step  W+N+1           one extra step with torch.cuda.memory._record_memory_history (stacks) -> snapshot summary
    then training is stopped (control.should_training_stop)

record_function ranges (all prefixed TCO::), used by analyze.py to attribute GPU kernels to phases:
    TCO::step                      whole step as the TCO meter times it (on_step_begin .. on_step_end)
    TCO::prep fwd_body head_ce backward clip optimizer post     sequential phases inside a step
    TCO::fwd_layer_<i>             decoder layer i, first forward
    TCO::recompute_layer_<i>       decoder layer i entered again while in_backward is set = the
                                   gradient-checkpoint recompute (layer.forward is wrapped, so every call counts)
    TCO::soup_rewind_row_losses    Soup's per-row loss recorder (Soup arm only)
"""
from __future__ import annotations

import collections
import gzip
import json
import os
import shutil
import time
import traceback

MIB = 1024.0 * 1024.0
N_PROF = int(os.environ.get("TCO_PROFILE_STEPS", "3"))
DO_SNAPSHOT = os.environ.get("TCO_PROFILE_SNAPSHOT", "1") != "0"


def _write(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=1, default=str)
    os.replace(tmp, path)


def find_body(model):
    """The decoder stack module (has .embed_tokens and a .layers ModuleList): Qwen2Model."""
    import torch
    for module in model.modules():
        layers = getattr(module, "layers", None)
        if isinstance(layers, torch.nn.ModuleList) and hasattr(module, "embed_tokens"):
            return module
    return None


def param_table(model) -> dict:
    """Bytes of every parameter / buffer / bnb quant-state tensor, grouped by (category, dtype)."""
    import torch
    groups = collections.defaultdict(lambda: [0, 0])   # key -> [n_tensors, bytes]

    def cat(name: str, tensor) -> str:
        low = name.lower()
        if "lora_a" in low or "lora_b" in low:
            return "lora_AB"
        if "embed_tokens" in low:
            return "embed_tokens"
        if "lm_head" in low:
            return "lm_head"
        if "norm" in low:
            return "norm"
        if str(tensor.dtype) == "torch.uint8":
            return "nf4_packed_weight"
        if "bias" in low:
            return "bias"
        return "other"

    seen = set()
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        if id(tensor) in seen:
            continue
        seen.add(id(tensor))
        key = f"{cat(name, tensor)} [{str(tensor.dtype).replace('torch.', '')}]"
        groups[key][0] += 1
        groups[key][1] += tensor.numel() * tensor.element_size()
        qs = getattr(tensor, "quant_state", None)
        if qs is not None:
            for sub in (qs, getattr(qs, "state2", None)):
                if sub is None:
                    continue
                for attr in ("absmax", "code", "offset"):
                    val = getattr(sub, attr, None)
                    if isinstance(val, torch.Tensor) and id(val) not in seen:
                        seen.add(id(val))
                        k2 = f"bnb_quant_state.{attr} [{str(val.dtype).replace('torch.', '')}]"
                        groups[k2][0] += 1
                        groups[k2][1] += val.numel() * val.element_size()
    rows = sorted(({"group": k, "tensors": v[0], "mib": round(v[1] / MIB, 1)} for k, v in groups.items()),
                  key=lambda r: -r["mib"])
    return {"total_mib": round(sum(r["mib"] for r in rows), 1), "groups": rows}


def snapshot_summary(snap: dict, base_alloc_mib: float) -> dict:
    """Replay the allocator's device trace of ONE step: live blocks at the allocated-bytes peak, grouped by
    the Python stack that allocated them.  Blocks allocated before recording started are not in the trace;
    they are reported as base_alloc_mib (weights, optimizer states, ...)."""
    traces = snap.get("device_traces") or []
    events = traces[0] if traces else []
    live, cur, best, best_idx = {}, 0, 0, -1
    for idx, ev in enumerate(events):
        act = ev.get("action")
        if act == "alloc":
            live[ev["addr"]] = ev
            cur += ev["size"]
            if cur > best:
                best, best_idx = cur, idx
        elif act == "free_requested":
            old = live.pop(ev["addr"], None)
            if old is not None:
                cur -= old["size"]
    # replay again up to the peak to get the live set at that instant
    live, cur = {}, 0
    for idx, ev in enumerate(events):
        if idx > best_idx:
            break
        act = ev.get("action")
        if act == "alloc":
            live[ev["addr"]] = ev
        elif act == "free_requested":
            live.pop(ev["addr"], None)

    def short(frame) -> str:
        return f"{os.path.basename(frame.get('filename', '?'))}:{frame.get('line', '?')}:{frame.get('name', '?')}"

    groups = collections.defaultdict(lambda: [0, 0])
    blocks = []
    for ev in live.values():
        frames = ev.get("frames") or []
        sig = " < ".join(short(fr) for fr in frames[:3]) or "(no python frame)"
        groups[sig][0] += 1
        groups[sig][1] += ev["size"]
        blocks.append((ev["size"], [short(fr) for fr in frames[:8]]))
    blocks.sort(key=lambda b: -b[0])
    rows = sorted(({"stack_top3": k, "blocks": v[0], "mib": round(v[1] / MIB, 1)} for k, v in groups.items()),
                  key=lambda r: -r["mib"])
    return {"events_in_trace": len(events), "peak_new_mib_in_trace": round(best / MIB, 1),
            "peak_event_index": best_idx, "base_alloc_before_step_mib": round(base_alloc_mib, 1),
            "peak_total_mib_estimate": round(base_alloc_mib + best / MIB, 1),
            "live_blocks_at_peak": len(live),
            "by_stack_top3": rows[:40],
            "largest_blocks": [{"mib": round(sz / MIB, 1), "frames": fr} for sz, fr in blocks[:25]]}


class Controller:
    def __init__(self, trainer, out_dir: str, warmup: int, framework: str):
        import torch
        self.torch = torch
        self.cuda = torch.cuda.is_available()
        self.trainer = trainer
        self.framework = framework
        self.out = os.path.join(out_dir, "prof")
        os.makedirs(self.out, exist_ok=True)
        self.warm, self.n = warmup, N_PROF
        self.profiling = False
        self.snapshotting = False
        self.in_backward = False
        self.prof = None
        self.step_idx = 0
        self.cur = {}
        self.steps = {}
        self._step_ctx = None
        self._phase_ctx = None
        self.range_counts = collections.Counter()
        self.meta = {"framework": framework, "warmup": warmup, "n_profiled": self.n,
                     "torch": torch.__version__, "cuda": self.cuda, "errors": []}
        if self.cuda:
            self.meta["device"] = torch.cuda.get_device_name(0)
            self.meta["capability"] = list(torch.cuda.get_device_capability(0))
            self.meta["tf32_matmul"] = bool(torch.backends.cuda.matmul.allow_tf32)
        # memory right after the trainer was built = after model load + wrap (before any step)
        if self.cuda:
            self.meta["after_load"] = {"alloc_mib": round(torch.cuda.memory_allocated() / MIB, 1),
                                       "peak_mib_since_process_start": round(torch.cuda.max_memory_allocated() / MIB, 1),
                                       "reserved_mib": round(torch.cuda.memory_reserved() / MIB, 1)}
        try:
            self.meta["params"] = param_table(trainer.model)
        except Exception:   # noqa: BLE001
            self.meta["errors"].append("param_table: " + traceback.format_exc()[-600:])
        self.body = find_body(trainer.model)
        self.meta["body_found"] = self.body is not None
        self.meta["n_layers"] = len(self.body.layers) if self.body is not None else None
        self._install_hooks()
        self._flush()

    # ------------------------------------------------------------------ helpers
    def _flush(self):
        self.meta["steps"] = self.steps
        self.meta["range_counts"] = dict(self.range_counts)
        _write(os.path.join(self.out, "prof_meta.json"), self.meta)

    def _rf(self, name: str):
        if not self.profiling:
            return None
        self.range_counts[name.split("_")[0] if name.startswith("TCO::fwd_layer") or name.startswith("TCO::recompute_layer") else name] += 1
        ctx = self.torch.profiler.record_function(name)
        ctx.__enter__()
        return ctx

    @staticmethod
    def _end(ctx):
        if ctx is not None:
            ctx.__exit__(None, None, None)

    def phase(self, name: str):
        if not self.profiling:
            return
        self._end(self._phase_ctx)
        self._phase_ctx = self._rf("TCO::" + name)

    def mark(self, key: str):
        if not self.cuda:
            return
        cu = self.torch.cuda
        self.cur.setdefault("mem", {})[key] = {"peak_mib": round(cu.max_memory_allocated() / MIB, 1),
                                               "cur_mib": round(cu.memory_allocated() / MIB, 1)}
        cu.reset_peak_memory_stats()

    # ------------------------------------------------------------------ hooks
    def _install_hooks(self):
        trainer = self.trainer
        ctl = self

        orig_cl = trainer.compute_loss

        def compute_loss(*a, **k):
            ctl.mark("pre_fwd")          # peak since step start: inputs, trl pre-processing
            try:
                return orig_cl(*a, **k)
            finally:
                ctl.mark("head_ce")      # peak between decoder-body end and compute_loss end (lm_head, CE, metrics)
                ctl.in_backward = True
                ctl.phase("backward")
        trainer.compute_loss = compute_loss

        orig_ts = trainer.training_step

        def training_step(*a, **k):
            try:
                return orig_ts(*a, **k)
            finally:
                ctl.mark("backward")     # peak between compute_loss end and backward end (incl. recompute)
                ctl.in_backward = False
                ctl.phase("clip")
        trainer.training_step = training_step

        if self.body is not None:
            def body_pre(module, args):
                ctl.mark("fwd_pre_body")
                ctl.phase("fwd_body")

            def body_post(module, args, output):
                ctl.mark("fwd_body")     # peak inside embeddings + all layers + final norm (forward only)
                ctl.phase("head_ce")

            self.body.register_forward_pre_hook(body_pre)
            self.body.register_forward_hook(body_post)
            for idx, layer in enumerate(self.body.layers):
                # Wrap forward (try/finally) instead of using a post-hook: non-reentrant torch checkpoint raises
                # _StopRecomputationError out of the recomputed layer ("early stop"), which skips forward hooks.
                layer.forward = self._layer_wrap(idx, layer.forward)

        if self.framework == "soup":
            try:
                import soup_cli.trainer.rewind_hf as rewind_mod
                orig_rows = rewind_mod.row_losses

                def row_losses(*a, **k):
                    ctx = ctl._rf("TCO::soup_rewind_row_losses")
                    try:
                        return orig_rows(*a, **k)
                    finally:
                        ctl._end(ctx)
                rewind_mod.row_losses = row_losses
                self.meta["soup_rewind_patched"] = True
            except Exception as exc:   # noqa: BLE001
                self.meta["soup_rewind_patched"] = f"no: {type(exc).__name__}: {exc}"

    def _layer_wrap(self, idx, orig):
        ctl = self

        def forward(*a, **k):
            if not ctl.profiling:
                return orig(*a, **k)
            ctx = ctl._rf(("TCO::recompute_layer_" if ctl.in_backward else "TCO::fwd_layer_") + str(idx))
            try:
                return orig(*a, **k)
            finally:
                ctl._end(ctx)
        return forward

    # ------------------------------------------------------------------ callback events
    def on_step_begin(self, state, control):
        self.step_idx = int(state.global_step) + 1
        self.cur = {"step": self.step_idx}
        if self.cuda:
            self.torch.cuda.synchronize()
        self.mark("pre_step")
        if self.step_idx == self.warm + 1:
            self._start_profiler()
        if self.step_idx == self.warm + self.n + 1 and DO_SNAPSHOT and self.cuda:
            self._start_snapshot()
        if self.profiling:
            self._step_ctx = self._rf("TCO::step")
            self._phase_ctx = self._rf("TCO::prep")
        self.cur["t0"] = time.perf_counter()

    def on_pre_opt(self):
        self.phase("optimizer")
        self.mark("clip")                # peak of gradient clipping (between training_step end and here)

    def on_opt(self):
        self.mark("optimizer")           # peak inside optimizer.step
        self.phase("post")

    def on_step_end(self, state, control):
        if self.cuda:
            self.torch.cuda.synchronize()
        t1 = time.perf_counter()
        self.cur["wall_s"] = round(t1 - self.cur.get("t0", t1), 5)
        self.mark("post")                # cur_mib here = weights + optimizer states (grads already freed)
        self.cur["profiled"] = bool(self.profiling)
        self.cur["snapshot_step"] = bool(self.snapshotting)
        if self.profiling:
            self._end(self._phase_ctx)
            self._phase_ctx = None
            self._end(self._step_ctx)
            self._step_ctx = None
        self.cur.pop("t0", None)
        self.steps[str(self.step_idx)] = self.cur
        done = int(state.global_step)
        if self.profiling and done == self.warm + self.n:
            self._stop_profiler()
            if not (DO_SNAPSHOT and self.cuda):
                control.should_training_stop = True
        if self.snapshotting and done == self.warm + self.n + 1:
            self._stop_snapshot()
            control.should_training_stop = True
        if not DO_SNAPSHOT and done >= self.warm + self.n:
            control.should_training_stop = True
        self._flush()

    # ------------------------------------------------------------------ profiler
    def _start_profiler(self):
        torch = self.torch
        acts = [torch.profiler.ProfilerActivity.CPU]
        if self.cuda:
            acts.append(torch.profiler.ProfilerActivity.CUDA)
        self.prof = torch.profiler.profile(activities=acts, record_shapes=True, profile_memory=False,
                                           with_stack=False, with_flops=False)
        self.prof.start()
        self.profiling = True
        self.meta["profile_started_at_step"] = self.step_idx

    def _stop_profiler(self):
        prof = self.prof
        self.profiling = False
        try:
            if self.cuda:
                self.torch.cuda.synchronize()
            prof.stop()
        except Exception:   # noqa: BLE001
            self.meta["errors"].append("prof.stop: " + traceback.format_exc()[-800:])
            return
        raw = os.path.join(self.out, "trace.json")
        try:
            prof.export_chrome_trace(raw)
            with open(raw, "rb") as fin, gzip.open(raw + ".gz", "wb", compresslevel=6) as fout:
                shutil.copyfileobj(fin, fout)
            self.meta["trace_raw_mb"] = round(os.path.getsize(raw) / 1e6, 1)
            self.meta["trace_gz_mb"] = round(os.path.getsize(raw + ".gz") / 1e6, 1)
            os.remove(raw)
        except Exception:   # noqa: BLE001
            self.meta["errors"].append("export: " + traceback.format_exc()[-800:])
        for sort_key in ("self_cuda_time_total", "self_device_time_total", "cuda_time_total", "device_time_total"):
            try:
                table = prof.key_averages().table(sort_by=sort_key, row_limit=120, max_name_column_width=110)
                with open(os.path.join(self.out, "key_averages_by_self_gpu_time.txt"), "w", encoding="utf-8") as fh:
                    fh.write(f"# sorted by {sort_key}; {self.n} profiled steps (totals are over all of them)\n")
                    fh.write(table)
                break
            except Exception:   # noqa: BLE001
                continue
        for sort_key in ("self_cuda_time_total", "self_device_time_total"):
            try:
                table = prof.key_averages(group_by_input_shape=True).table(
                    sort_by=sort_key, row_limit=90, max_name_column_width=60, max_shapes_column_width=90)
                with open(os.path.join(self.out, "key_averages_by_shape.txt"), "w", encoding="utf-8") as fh:
                    fh.write(f"# group_by_input_shape, sorted by {sort_key}; {self.n} profiled steps\n")
                    fh.write(table)
                break
            except Exception:   # noqa: BLE001
                continue
        try:
            table = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=40, max_name_column_width=90)
            with open(os.path.join(self.out, "key_averages_by_self_cpu_time.txt"), "w", encoding="utf-8") as fh:
                fh.write(table)
        except Exception:   # noqa: BLE001
            pass
        self.prof = None
        self._flush()

    # ------------------------------------------------------------------ allocator snapshot (one extra step)
    def _start_snapshot(self):
        cu = self.torch.cuda
        try:
            self._snap_base = cu.memory_allocated() / MIB
            cu.memory._record_memory_history(enabled="all", context="all", stacks="python", max_entries=2_000_000)
            self.snapshotting = True
        except Exception:   # noqa: BLE001
            self.meta["errors"].append("record_memory_history: " + traceback.format_exc()[-600:])

    def _stop_snapshot(self):
        cu = self.torch.cuda
        self.snapshotting = False
        try:
            snap = cu.memory._snapshot()
            cu.memory._record_memory_history(enabled=None)
            self.meta["snapshot_summary"] = snapshot_summary(snap, self._snap_base)
        except Exception:   # noqa: BLE001
            self.meta["errors"].append("snapshot: " + traceback.format_exc()[-1200:])

    def finish(self):
        self._flush()


def attach(trainer, out_dir: str, warmup: int, framework: str):
    """Returns a transformers.TrainerCallback to be added by the caller (so the add happens in one place)."""
    from transformers import TrainerCallback
    ctl = Controller(trainer, out_dir, warmup, framework)

    class ProfCB(TrainerCallback):
        def on_step_begin(self, args, state, control, **kw):
            ctl.on_step_begin(state, control)

        def on_pre_optimizer_step(self, args, state, control, **kw):
            ctl.on_pre_opt()

        def on_optimizer_step(self, args, state, control, **kw):
            ctl.on_opt()

        def on_step_end(self, args, state, control, **kw):
            ctl.on_step_end(state, control)

        def on_train_end(self, args, state, control, **kw):
            ctl.finish()

    return ProfCB()
