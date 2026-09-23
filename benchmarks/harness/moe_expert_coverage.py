"""Which experts does one training step actually touch? (MoE step 0, `.claude/plan.md`)

Expert-granularity streaming is only worth building if a step reads FEWER experts
than the whole layer. This measures that directly, per layer, on a real model and
a real corpus:

* **union coverage** — the fraction of a layer's experts that at least one token
  in the step routes to. This is the number that decides the feature: streaming
  an expert at a time saves READS only to the extent that coverage is below 1.0.
  Below it, the saving is memory alone.
* **the uniform-random baseline for the same (experts, top_k, tokens)**, printed
  beside every coverage figure, because coverage on its own is not interpretable.
  With E experts, k per token and T tokens routed independently and uniformly,
  the expected coverage is ``1 - (1 - k/E)**T`` — for OLMoE's 64/8 at 512 tokens
  that is 1 - e^-64, i.e. indistinguishable from 1.0. **A high coverage is
  therefore the DEFAULT, not a finding; only a coverage well below the baseline
  says routing is concentrated enough to stream.**
* **routing skew** — what share of a layer's token-to-expert assignments the
  busiest 10 / 25 / 50% of its experts take, plus a Gini coefficient. Skew is
  what a hybrid tier (hot experts resident, cold ones streamed) would live on,
  and it can be strong while coverage is ~1.0: "every expert is touched" and
  "the traffic is concentrated in a few" are different statements.
* **hot-set stability** — the Jaccard overlap of a layer's busiest quartile
  between consecutive steps, and against the corpus-wide busiest quartile. A
  persisted hot set is only useful if the set is stable; this is the number that
  says whether it is.
* **one-layer-ahead predictability** — the share of layer K+1's TRAFFIC that
  would have been caught by prefetching (a) the set layer K itself just used, or
  (b) layer K+1's corpus-wide busiest quartile. This is the precondition for any
  throughput claim, and it costs nothing once the indices are captured: today's
  prefetcher is a PERFECT predictor because the layer walk is deterministic,
  while which experts a layer needs is unknown until its router has run. Expert
  granularity trades a perfect prefetch for a conditional one, and these two
  numbers say what the conditional one would be worth. **Read them only where
  coverage is well below 1.0** — when nearly every expert is touched, "layer K's
  used set" is nearly every expert and a hit rate near 1.0 says only that
  prefetching everything works, which is what the layer tier already does.

No Soup import: this is a model-and-corpus measurement that must run against a
stock transformers install, and keeping it standalone means a reader can check it
without the project. Timing is deliberately NOT reported — the forward runs with
hooks on every router and, on CPU, under no throughput claim at all.

Router discovery is by SHAPE, not by an architecture table. In transformers 5.17
every MoE decoder (olmoe, qwen3_moe, mixtral, granitemoe, deepseek*) exposes a
``*TopKRouter`` module returning ``(router_logits, router_scores, router_indices)``
with integer indices of shape ``(tokens, top_k)``; this hooks anything shaped like
that and falls back to taking ``topk`` of a float ``(tokens, num_experts)`` output
itself. A model where neither holds is REFUSED by name rather than measured
wrongly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Refuse absurd inputs rather than allocating for them.
_MAX_SHAPES = 8
_MAX_BATCHES = 512
_MAX_TOKENS_PER_STEP = 1 << 20


# =====================================================================
# Corpus
# =====================================================================
def _read_jsonl_text(path: str, fields: Sequence[str], limit: Optional[int]) -> List[str]:
    """Every string a row carries under one of ``fields``, chat rows included."""
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            for name in fields:
                value = row.get(name)
                if isinstance(value, str) and value.strip():
                    out.append(value)
                elif isinstance(value, list):  # chat: [{"role","content"}, ...]
                    for message in value:
                        if isinstance(message, dict):
                            content = message.get("content")
                            if isinstance(content, str) and content.strip():
                                out.append(content)
            if limit is not None and len(out) >= limit:
                break
    return out


def _read_hf_text(spec: str, limit: Optional[int]) -> List[str]:
    """``hf:<dataset>[:<config>][:<split>][:<field>]`` through `datasets`, streamed."""
    from datasets import load_dataset

    parts = spec[3:].split(":")
    name = parts[0]
    config = parts[1] if len(parts) > 1 and parts[1] else None
    split = parts[2] if len(parts) > 2 and parts[2] else "train"
    field_name = parts[3] if len(parts) > 3 and parts[3] else None
    stream = load_dataset(name, config, split=split, streaming=True)
    out: List[str] = []
    for row in stream:
        if field_name is not None:
            value = row.get(field_name)
        else:  # first string column
            value = next((v for v in row.values() if isinstance(v, str)), None)
        if isinstance(value, str) and value.strip():
            out.append(value)
        if limit is not None and len(out) >= limit:
            break
    return out


def load_corpus(spec: str, fields: Sequence[str], limit: Optional[int]) -> List[str]:
    if spec.startswith("hf:"):
        return _read_hf_text(spec, limit)
    if os.path.isdir(spec):
        out: List[str] = []
        for root, dirs, names in os.walk(spec):
            # Sorted IN PLACE, which is what makes os.walk deterministic: it
            # yields directories in filesystem order otherwise, and on the dev
            # box that order is not sorted (data/providers before
            # data/_fixtures), so the same directory would concatenate
            # differently on another machine and move every number that reads
            # it. Note this CHANGES the order relative to results published
            # before it, which is why those carry their own corpus digest.
            dirs.sort()
            for name in sorted(names):
                if name.endswith((".py", ".md", ".txt")):
                    full = os.path.join(root, name)
                    with open(full, "r", encoding="utf-8", errors="replace") as handle:
                        out.append(handle.read())
                if limit is not None and len(out) >= limit:
                    return out
        return out
    if spec.endswith(".jsonl"):
        return _read_jsonl_text(spec, fields, limit)
    with open(spec, "r", encoding="utf-8", errors="replace") as handle:
        return [handle.read()]


def corpus_revision(spec: str) -> Optional[str]:
    """The git commit of a DIRECTORY corpus, when it is inside a repository.

    A directory corpus is not a fixture: it is whatever the tree holds at the
    moment it is read. Measured the hard way -- two runs of one configuration 43
    minutes apart disagreed on the code corpus because `origin/main` had been
    merged in between, changing 13 files under it, and the per-expert counts
    moved by 6.7e-03 while prose and math reproduced to 0.000e+00. The digest
    already caught it; this says WHICH tree, so the difference is explicable
    rather than merely visible.
    """
    if spec.startswith("hf:") or not os.path.isdir(spec):
        return None
    import subprocess

    try:
        out = subprocess.run(
            ["git", "-C", spec, "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return None
    revision = out.stdout.strip()
    if out.returncode != 0 or not revision:
        return None
    dirty = subprocess.run(
        ["git", "-C", spec, "status", "--porcelain", "--", "."],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    return revision + ("-dirty" if dirty.stdout.strip() else "")


def pack_token_stream(texts: Sequence[str], tokenizer, seq: int, seed: int) -> "Any":
    """One long id stream, sliced into exact ``seq``-length chunks.

    Packing rather than padding, on purpose: a padded batch routes its PAD
    positions too, and those assignments would land in the counts as if they
    were traffic. Every token measured here is a real one.
    """
    import torch

    ids: List[int] = []
    for text in texts:
        ids.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
    usable = (len(ids) // seq) * seq
    if usable == 0:
        raise ValueError(
            f"corpus holds {len(ids)} tokens, fewer than one {seq}-token chunk; "
            f"pass more data or a shorter --seq"
        )
    chunks = torch.tensor(ids[:usable], dtype=torch.long).view(-1, seq)
    generator = torch.Generator().manual_seed(seed)
    return chunks[torch.randperm(chunks.shape[0], generator=generator)]


# =====================================================================
# Router discovery
# =====================================================================
@dataclass
class RouterHook:
    name: str
    layer: int
    module: Any
    top_k: int
    n_experts: int
    seen: List[Any] = field(default_factory=list)

    def take(self) -> "Any":
        """The indices captured since the last call, as one (tokens, top_k) tensor."""
        import torch

        if not self.seen:
            raise RuntimeError(f"router {self.name!r} produced nothing this forward")
        out = torch.cat([part.reshape(-1, part.shape[-1]) for part in self.seen], dim=0)
        self.seen.clear()
        return out


def _indices_from_output(output: Any, top_k: int, n_experts: int) -> Optional["Any"]:
    """The top-k expert indices in a router's output, whatever shape it came in."""
    import torch

    candidates = output if isinstance(output, (tuple, list)) else [output]
    for item in candidates:
        if isinstance(item, torch.Tensor) and not item.is_floating_point():
            if item.ndim >= 2 and item.shape[-1] == top_k:
                return item
    for item in candidates:
        if isinstance(item, torch.Tensor) and item.is_floating_point():
            if item.ndim >= 2 and item.shape[-1] == n_experts:
                return item.topk(top_k, dim=-1).indices
    return None


def attach_routers(model, n_experts: int, top_k: int) -> List[RouterHook]:
    """Hook every module that looks like a top-k router, in layer order."""
    hooks: List[RouterHook] = []
    for name, module in model.named_modules():
        cls = type(module).__name__
        looks_like_router = cls.endswith("TopKRouter") or (
            name.endswith(".gate") and "moe" in type(module).__name__.lower()
        )
        if not looks_like_router:
            continue
        layer = -1
        for part in name.split("."):
            if part.isdigit():
                layer = int(part)
                break
        hooks.append(
            RouterHook(
                name=name, layer=layer, module=module, top_k=top_k, n_experts=n_experts
            )
        )
    if not hooks:
        raise ValueError(
            "no top-k router module found on this model. This probe measures MoE "
            "routing; a dense model has nothing to measure, and an MoE whose router "
            "is shaped differently must be taught to this harness rather than "
            "guessed at."
        )
    hooks.sort(key=lambda hook: (hook.layer, hook.name))

    def make(hook: RouterHook):
        def capture(_module, _args, output):
            indices = _indices_from_output(output, hook.top_k, hook.n_experts)
            if indices is None:
                raise RuntimeError(
                    f"router {hook.name!r} returned something this harness cannot read: "
                    f"expected integer indices with last dim {hook.top_k} or float logits "
                    f"with last dim {hook.n_experts}"
                )
            hook.seen.append(indices.detach().to("cpu"))

        return capture

    for hook in hooks:
        hook.module.register_forward_hook(make(hook))
    return hooks


# =====================================================================
# Statistics
# =====================================================================
def host_state() -> Dict[str, Any]:
    """What else was on the box, recorded INTO the results rather than beside them.

    A routing count cannot be moved by memory pressure -- the top-k of a router
    is deterministic given the weights and the tokens -- so this is not a
    correction to any number here. It is provenance: a record whose box state
    comes from a wrapper that was never committed is a record a reader cannot
    check, which is what the review of PR #992 found. Recorded before and after
    every model.

    ``python_processes`` counts processes whose name starts with "python",
    which on Windows includes the venv launcher's redirector as well as the
    interpreter it starts -- one logical run can therefore show as two.
    """
    state: Dict[str, Any] = {"at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    try:
        import psutil
    except ImportError:
        state["note"] = "psutil not installed; host state not recorded"
        return state
    memory = psutil.virtual_memory()
    state["available_gb"] = round(memory.available / 1e9, 2)
    state["memory_used_percent"] = memory.percent
    try:
        state["swap_used_gb"] = round(psutil.swap_memory().used / 1e9, 2)
    except (OSError, RuntimeError):  # pragma: no cover - platform dependent
        state["swap_used_gb"] = None
    count = 0
    for process in psutil.process_iter(["name"]):
        name = (process.info.get("name") or "").lower()
        if name.startswith("python"):
            count += 1
    state["python_processes"] = count
    return state


def harness_fingerprint() -> str:
    """SHA-256 of this file, so a result names the code that produced it.

    Today's lesson twice over: a run that does not record which source tree it
    imported is a run whose arm cannot be proved afterwards.
    """
    return hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()[:16]


def uniform_coverage(n_experts: int, top_k: int, tokens: int) -> float:
    """Expected coverage if every token picked ``top_k`` experts uniformly.

    The baseline every measured coverage has to be read against: at OLMoE's
    64 experts / top-8 and 512 tokens this is 1 - (1 - 1/8)^512 = 1.0 to 29
    decimal places, so "coverage is ~100%" is what chance alone predicts.
    """
    return 1.0 - (1.0 - top_k / n_experts) ** tokens


def gini(counts: Sequence[int]) -> float:
    """0.0 = every expert takes an equal share, 1.0 = one expert takes it all."""
    total = sum(counts)
    if total <= 0 or len(counts) < 2:
        return 0.0
    ordered = sorted(counts)
    n = len(ordered)
    weighted = sum((index + 1) * value for index, value in enumerate(ordered))
    return (2.0 * weighted) / (n * total) - (n + 1.0) / n


def top_share(counts: Sequence[int], fraction: float) -> float:
    """Share of all assignments taken by the busiest ``fraction`` of experts."""
    total = sum(counts)
    if total <= 0:
        return 0.0
    keep = max(1, int(round(len(counts) * fraction)))
    return sum(sorted(counts, reverse=True)[:keep]) / total


def hot_set(counts: Sequence[int], fraction: float = 0.25) -> set:
    keep = max(1, int(round(len(counts) * fraction)))
    order = sorted(range(len(counts)), key=lambda index: (-counts[index], index))
    return set(order[:keep])


def jaccard(left: set, right: set) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def traffic_hit_rate(counts: Sequence[int], prefetched: set) -> float:
    """Share of a layer's ASSIGNMENTS that land on an already-prefetched expert.

    Weighted by traffic rather than by expert, because a prefetch that catches
    the busiest experts and misses three idle ones has done its job.
    """
    total = sum(counts)
    if total <= 0:
        return 1.0
    return sum(value for expert, value in enumerate(counts) if expert in prefetched) / total


def used_set(counts: Sequence[int]) -> set:
    return {expert for expert, value in enumerate(counts) if value > 0}


# =====================================================================
# The measurement
# =====================================================================
@dataclass
class LayerStats:
    layer: int
    name: str
    counts: List[int]
    coverage: List[float] = field(default_factory=list)
    per_step_hot: List[set] = field(default_factory=list)
    per_step_counts: List[List[int]] = field(default_factory=list)


def measure_shape(
    model,
    hooks: Sequence[RouterHook],
    chunks,
    *,
    batch: int,
    seq: int,
    batches: int,
    device: str,
    offset: int,
) -> Tuple[List[LayerStats], int, int]:
    """Run ``batches`` steps of ``batch`` x ``seq`` and reduce each step at once."""
    import torch

    n_experts = hooks[0].n_experts
    stats = [LayerStats(layer=hook.layer, name=hook.name, counts=[0] * n_experts) for hook in hooks]
    available = chunks.shape[0]
    used_steps = 0
    tokens_per_step = batch * seq
    for step in range(batches):
        start = (offset + step * batch) % available
        rows = [chunks[(start + index) % available] for index in range(batch)]
        ids = torch.stack(rows).to(device)
        with torch.no_grad():
            model(input_ids=ids, use_cache=False)
        for hook, entry in zip(hooks, stats):
            indices = hook.take()
            flat = indices.reshape(-1)
            counted = torch.bincount(flat, minlength=n_experts).tolist()
            touched = int((torch.tensor(counted) > 0).sum().item())
            entry.coverage.append(touched / n_experts)
            entry.per_step_hot.append(hot_set(counted))
            entry.per_step_counts.append(counted)
            for expert, value in enumerate(counted):
                entry.counts[expert] += value
        used_steps += 1
    return stats, used_steps, tokens_per_step


def summarise(
    stats: Sequence[LayerStats], n_experts: int, top_k: int, tokens: int
) -> Dict[str, Any]:
    baseline = uniform_coverage(n_experts, top_k, tokens)
    # Could the experts of layer K+1 be prefetched while layer K computes?
    # Two candidate predictors, both free from what is already captured:
    # the set layer K itself just used, and layer K+1's corpus-wide busiest
    # quartile (what a persisted heat file would hold). Both are reported as
    # the share of layer K+1's TRAFFIC they would have caught.
    #
    # READ THESE ONLY WHERE COVERAGE IS WELL BELOW 1.0. When almost every
    # expert is touched, "layer K's used set" is almost every expert, so a hit
    # rate near 1.0 says nothing except that prefetching everything works --
    # which is what the layer tier already does.
    ahead: Dict[int, Dict[str, float]] = {}
    for index in range(1, len(stats)):
        previous, current = stats[index - 1], stats[index]
        global_hot = hot_set(current.counts)
        from_previous, from_heat, set_overlap = [], [], []
        for step in range(min(len(previous.per_step_counts), len(current.per_step_counts))):
            previous_used = used_set(previous.per_step_counts[step])
            from_previous.append(traffic_hit_rate(current.per_step_counts[step], previous_used))
            from_heat.append(traffic_hit_rate(current.per_step_counts[step], global_hot))
            set_overlap.append(jaccard(previous_used, used_set(current.per_step_counts[step])))
        if from_previous:
            ahead[current.layer] = {
                "traffic_caught_by_previous_layers_set": sum(from_previous) / len(from_previous),
                "traffic_caught_by_corpus_hot25": sum(from_heat) / len(from_heat),
                "used_set_jaccard_with_previous_layer": sum(set_overlap) / len(set_overlap),
            }
    layers = []
    for entry in stats:
        global_hot = hot_set(entry.counts)
        step_to_step = [
            jaccard(entry.per_step_hot[index], entry.per_step_hot[index + 1])
            for index in range(len(entry.per_step_hot) - 1)
        ]
        to_global = [jaccard(one, global_hot) for one in entry.per_step_hot]
        layers.append(
            {
                "layer": entry.layer,
                "router": entry.name,
                "coverage_mean": sum(entry.coverage) / len(entry.coverage),
                "coverage_min": min(entry.coverage),
                "coverage_max": max(entry.coverage),
                "experts_touched_mean": (sum(entry.coverage) / len(entry.coverage)) * n_experts,
                "gini": gini(entry.counts),
                "top10_share": top_share(entry.counts, 0.10),
                "top25_share": top_share(entry.counts, 0.25),
                "top50_share": top_share(entry.counts, 0.50),
                "hot25_jaccard_step_to_step": (
                    sum(step_to_step) / len(step_to_step) if step_to_step else None
                ),
                "hot25_jaccard_to_global": sum(to_global) / len(to_global),
                "one_layer_ahead": ahead.get(entry.layer),
                "counts": entry.counts,
            }
        )
    coverages = [one["coverage_mean"] for one in layers]
    ginis = [one["gini"] for one in layers]
    caught = [
        one["one_layer_ahead"]["traffic_caught_by_previous_layers_set"]
        for one in layers
        if one["one_layer_ahead"]
    ]
    heat = [
        one["one_layer_ahead"]["traffic_caught_by_corpus_hot25"]
        for one in layers
        if one["one_layer_ahead"]
    ]
    return {
        "traffic_caught_by_previous_layers_set_mean": (
            sum(caught) / len(caught) if caught else None
        ),
        "traffic_caught_by_corpus_hot25_mean": sum(heat) / len(heat) if heat else None,
        "tokens_per_step": tokens,
        "uniform_baseline_coverage": baseline,
        "coverage_mean_over_layers": sum(coverages) / len(coverages),
        "coverage_min_layer": min(coverages),
        "coverage_max_layer": max(coverages),
        "gini_mean_over_layers": sum(ginis) / len(ginis),
        "layers": layers,
    }


# =====================================================================
# CLI
# =====================================================================
def parse_shape(text: str) -> Tuple[int, int]:
    try:
        left, right = text.lower().split("x")
        batch, seq = int(left), int(right)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--shape wants BxS, e.g. 4x512; got {text!r}") from exc
    if batch < 1 or seq < 1:
        raise argparse.ArgumentTypeError(f"--shape wants positive numbers; got {text!r}")
    if batch * seq > _MAX_TOKENS_PER_STEP:
        raise argparse.ArgumentTypeError(
            f"--shape {text} is more than {_MAX_TOKENS_PER_STEP} tokens"
        )
    return batch, seq


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=True, help="model id or local path")
    parser.add_argument(
        "--data",
        action="append",
        required=True,
        help=(
            "corpus: a .jsonl, a .txt, a directory of .py/.md/.txt, or "
            "hf:<dataset>[:<config>][:<split>][:<field>]. Repeatable; each is "
            "measured separately, because routing skew is a property of the text"
        ),
    )
    parser.add_argument("--data-label", action="append", default=None, help="name per --data")
    parser.add_argument("--fields", default="text,content,output,instruction,messages")
    parser.add_argument("--shape", action="append", type=parse_shape, default=None, help="BxS")
    parser.add_argument("--batches", type=int, default=16, help="steps measured per shape")
    parser.add_argument("--limit-rows", type=int, default=2000)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--load-4bit",
        action="store_true",
        help=(
            "load the experts and attention as NF4 through bitsandbytes, so a model "
            "too large for the card in bf16 still fits. The ROUTER is untouched by "
            "this: in transformers 5.17 a top-k router holds its weight as a bare "
            "nn.Parameter and F.linear, not an nn.Linear, so replace_with_bnb_linear "
            "never sees it. What quantisation can still move is the hidden state the "
            "router reads, which is why the record measures a bf16 control rather "
            "than assuming the effect is nil"
        ),
    )
    parser.add_argument("--out", required=True, help="JSON results file")
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    shapes = args.shape or [(1, 512), (4, 512), (1, 2048)]
    if len(shapes) > _MAX_SHAPES:
        parser.error(f"at most {_MAX_SHAPES} shapes")
    if not 1 <= args.batches <= _MAX_BATCHES:
        parser.error(f"--batches must be between 1 and {_MAX_BATCHES}")

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    n_experts = getattr(config, "num_experts", None) or getattr(config, "num_local_experts", None)
    top_k = getattr(config, "num_experts_per_tok", None)
    if not n_experts or not top_k:
        print(
            f"{args.model} declares no MoE routing (num_experts={n_experts}, "
            f"num_experts_per_tok={top_k}); there is nothing here to measure.",
            file=sys.stderr,
        )
        return 2

    print(f"loading {args.model} ({n_experts} experts, top-{top_k}) on {device}/{args.dtype} ...")
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    load_kwargs: Dict[str, Any] = {
        "dtype": dtype,
        "trust_remote_code": args.trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if args.load_4bit:
        from transformers import BitsAndBytesConfig

        if device != "cuda":
            parser.error("--load-4bit needs --device cuda; bitsandbytes has no CPU kernel")
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    model.eval()
    if not args.load_4bit:
        model.to(device)
    hooks = attach_routers(model, n_experts=n_experts, top_k=top_k)
    print(f"  {len(hooks)} routers, loaded in {time.perf_counter() - started:.1f} s")

    labels = args.data_label or []
    results: Dict[str, Any] = {
        "meta": {
            "model": args.model,
            "label": args.label,
            "n_experts": n_experts,
            "top_k": top_k,
            "n_routers": len(hooks),
            "n_shared_experts": getattr(config, "n_shared_experts", None)
            or getattr(config, "shared_expert_intermediate_size", None),
            "model_type": getattr(config, "model_type", None),
            "hidden_layers": getattr(config, "num_hidden_layers", None),
            "device": device,
            "dtype": args.dtype,
            "load_4bit": bool(args.load_4bit),
            "torch": torch.__version__,
            "seed": args.seed,
            "batches_per_shape": args.batches,
            "shapes": [f"{b}x{s}" for b, s in shapes],
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "harness_sha256_16": harness_fingerprint(),
            "host_before": host_state(),
        },
        "corpora": [],
    }

    for index, spec in enumerate(args.data):
        label = labels[index] if index < len(labels) else spec
        texts = load_corpus(spec, args.fields.split(","), args.limit_rows)
        digest = hashlib.sha256(chr(0).join(texts).encode("utf-8")).hexdigest()[:16]
        print(f"corpus {label!r}: {len(texts)} documents, sha256[:16]={digest}")
        entry: Dict[str, Any] = {
            "label": label,
            "spec": spec,
            "documents": len(texts),
            "sha256_16": digest,
            "git_revision": corpus_revision(spec),
            "shapes": {},
        }
        for batch, seq in shapes:
            chunks = pack_token_stream(texts, tokenizer, seq, args.seed)
            stats, steps, tokens = measure_shape(
                model,
                hooks,
                chunks,
                batch=batch,
                seq=seq,
                batches=min(args.batches, max(1, chunks.shape[0] // batch)),
                device=device,
                offset=0,
            )
            summary = summarise(stats, n_experts, top_k, tokens)
            summary["steps"] = steps
            summary["chunks_available"] = int(chunks.shape[0])
            entry["shapes"][f"{batch}x{seq}"] = summary
            print(
                f"  {batch}x{seq} ({tokens} tok, {steps} steps): "
                f"coverage {summary['coverage_mean_over_layers']:.3f} "
                f"(uniform baseline {summary['uniform_baseline_coverage']:.3f}), "
                f"gini {summary['gini_mean_over_layers']:.3f}"
            )
        results["corpora"].append(entry)

    results["meta"]["host_after"] = host_state()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=1)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
