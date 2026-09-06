"""Shared layer-streaming setup for every trainer wrapper that supports it.

v0.72.4 — extracted verbatim from ``trainer/sft.py`` so that SFT and the four
preference losses (DPO / ORPO / SimPO / KTO) cannot drift. There is exactly one
copy of the NF4 pre-flight, the RAM/disk tier decision, the VRAM fit refusal and
the runtime release; a per-wrapper copy would be five places to fix the next
time any of them is wrong.

The move is behaviour-preserving for SFT by design, so v0.72.0-.3's
bit-exactness gates remain valid without being re-run. The ONE addition is
``_STREAM_ROWS_PER_EXAMPLE``: DPO, ORPO and SimPO build their forward through
TRL's ``concatenated_inputs`` + ``torch.cat``, so 2 x ``batch_size`` rows reach
the model in a single tensor. v0.72.3's VRAM estimator was validated on the
property that it NEVER under-predicts, and budgeting those three at 1x rows
would break exactly that — on Windows the consequence is not an exception but a
silent WDDM spill to host memory that makes the run an order of magnitude
slower with no error at all.

NO top-level torch: this module is imported by five trainer modules.
"""

import contextlib
import math
import os
import shutil
from dataclasses import dataclass

from rich.console import Console
from rich.panel import Panel

console = Console()


#: How far over the predicted budget `training.stream_vram_probe` may still
#: defer to a measurement. The formula's worst measured error is 0.787x (21%
#: under, seq 6144), so anything beyond a small multiple is not the formula
#: being wrong — it is the config being too big, and that is refusable by
#: arithmetic without touching the GPU.
_PROBE_DEFERRAL_CEILING = 4.0


def _validate_qwen4_streaming_mode(*, arch: str, task: str, quant: str) -> None:
    """Keep unvalidated Qwen4 training modes outside the streamed path."""
    if arch != "qwen4_exp":
        return
    if task != "sft":
        raise ValueError(
            "Qwen4-Exp layer streaming is initially validated for task='sft' "
            f"only; got task={task!r}. Preference-loss parity is pending."
        )
    if quant != "none":
        raise ValueError(
            "Qwen4-Exp layer streaming currently requires quantization='none'. "
            "Its exact PLE path is validated, but streamed NF4 parity is pending."
        )


def _validate_qwen4_ngram_disk(*, disk_kind: str, weights_dir: str) -> None:
    """Refuse sparse PLE mmap on media outside the measured SSD classes."""
    if disk_kind in ("nvme", "ssd"):
        return
    raise ValueError(
        "training.stream_ngram_source='disk' needs an SSD or NVMe "
        f"checkpoint volume; detected {disk_kind!r} at {weights_dir}. "
        "Move the checkpoint or set an accurate training.stream_disk_kind override."
    )


def _resolve_qwen4_ngram_source(
    *,
    oq_ngram: bool,
    requested: str,
    store_total: int,
    ngram_bytes: int,
    free_ram: int,
    resident_ram: int = 0,
    total_ram: int | None = None,
    stream_source: str,
) -> str:
    """Resolve Qwen4 PLE storage and refuse unsupported oQ materialisation."""
    from soup_cli.utils.layer_stream import (
        PHYSICAL_RAM_TIER_HEADROOM,
        RAM_TIER_HEADROOM,
    )

    if oq_ngram:
        if requested == "ram":
            raise ValueError(
                "oQ PLE embeddings require "
                "training.stream_ngram_source='disk' (or 'auto'): the packed "
                "source stays read-only and only requested rows are dequantized."
            )
        return "disk"
    if requested != "auto":
        return requested
    ram_budget = free_ram * RAM_TIER_HEADROOM
    base_in_ram = (
        store_total
        if stream_source != "disk" and store_total + resident_ram < ram_budget
        else 0
    )
    ram_bytes = base_in_ram + ngram_bytes
    fits_available_ram = ram_bytes + resident_ram < ram_budget
    physical_limit = (
        None
        if total_ram is None
        else total_ram * PHYSICAL_RAM_TIER_HEADROOM
    )
    fits_physical_ram = (
        physical_limit is None
        or ram_bytes + resident_ram < physical_limit
    )
    return "ram" if fits_available_ram and fits_physical_ram else "disk"


def _validate_qwen4_ngram_ram_fit(
    *,
    stream_source: str,
    ngram_source: str,
    required_ram: int,
    free_ram: int,
    resident_ram: int = 0,
    total_ram: int | None = None,
) -> None:
    """Refuse a RAM base or PLE before either source allocates its store."""
    from soup_cli.utils.layer_stream import (
        PHYSICAL_RAM_TIER_HEADROOM,
        PHYSICAL_RAM_TIER_HEADROOM_PERCENT,
        RAM_TIER_HEADROOM,
    )

    ram_required = stream_source == "ram" or ngram_source == "ram"
    total_required_ram = required_ram + int(resident_ram)
    if (
        ram_required
        and total_ram is not None
        and total_required_ram >= total_ram * PHYSICAL_RAM_TIER_HEADROOM
    ):
        policy = (
            "training.stream_ngram_source='ram'"
            if ngram_source == "ram"
            else "training.stream_source='ram'"
        )
        fallback = (
            "stream_ngram_source='auto' to use read-only SSD streaming"
            if ngram_source == "ram"
            else "stream_source='auto' to allow the disk tier"
        )
        raise ValueError(
            f"{policy} but the base plus resident extras and selected PLE "
            f"storage needs {total_required_ram / 1e9:.1f} GB, "
            "which exceeds "
            f"{PHYSICAL_RAM_TIER_HEADROOM_PERCENT}% of physical RAM "
            f"({total_ram / 1e9:.1f} GB). Set {fallback}, free RAM, "
            "or pick a smaller base."
        )
    if ram_required and total_required_ram >= free_ram * RAM_TIER_HEADROOM:
        policy = (
            "training.stream_ngram_source='ram'"
            if ngram_source == "ram"
            else "training.stream_source='ram'"
        )
        fallback = (
            "stream_ngram_source='auto' to use read-only SSD streaming"
            if ngram_source == "ram"
            else "stream_source='auto' to allow the disk tier"
        )
        raise ValueError(
            f"{policy} but the base plus resident extras and selected PLE "
            f"storage needs {total_required_ram / 1e9:.1f} GB and only "
            f"{free_ram / 1e9:.1f} GB of RAM is free. Set {fallback}, free RAM, "
            "or pick a smaller base."
        )


def _warn_if_ngram_source_unused(
    *, arch: str, requested: str, ngram_bytes: int, notify
) -> None:
    """Make a user-supplied PLE policy visible when the checkpoint has no PLE."""
    if arch == "qwen4_exp" and requested != "auto" and not ngram_bytes:
        notify(
            "[yellow]training.stream_ngram_source="
            f"{requested!r} has no effect: this Qwen4 checkpoint has no PLE "
            "N-gram table.[/]"
        )


@dataclass(frozen=True)
class _ProbePlan:
    """What the post-build measured probe (#349) needs from the pre-flight.

    Carried forward rather than recomputed so the shape the probe measures and
    the shape the formula budgeted are the same by construction — two
    independent derivations of ``rows`` would be free to drift, and the whole
    point is to compare the two numbers against each other.
    """

    rows: int
    seq_len: int
    vocab_size: int
    predicted_bytes: int
    available_bytes: int


def _existing_disk_anchor(path: str) -> str:
    """Nearest existing ancestor, for paths whose final cache dir is not made yet."""
    anchor = os.path.realpath(os.path.expanduser(path))
    while not os.path.exists(anchor):
        parent = os.path.dirname(anchor)
        if parent == anchor:
            raise OSError(f"cannot locate an existing filesystem ancestor for {path!r}")
        anchor = parent
    return anchor


def _disk_volume(path: str) -> tuple[int, int]:
    """Filesystem identity and currently free bytes for a prospective write."""
    anchor = _existing_disk_anchor(path)
    return int(os.stat(anchor).st_dev), int(shutil.disk_usage(anchor).free)


def _render_stream_disk_preflight(
    *,
    source_bytes: int,
    materialized_copy_bytes: int,
    materialize_bytes: int,
    materialized_path: str,
    shard_bytes: int,
    shard_write_bytes: int,
    shard_path: str,
) -> None:
    """Print and enforce the complete on-disk cost before either cache writes."""
    writes = (
        ("materialized weight copy", materialized_path, materialize_bytes),
        ("layer-shard cache", shard_path, shard_write_bytes),
    )
    required_by_device: dict[int, int] = {}
    free_by_device: dict[int, int] = {}
    labels_by_device: dict[int, list[str]] = {}
    for label, path, required in writes:
        if required <= 0:
            continue
        device, free = _disk_volume(path)
        required_by_device[device] = required_by_device.get(device, 0) + required
        free_by_device[device] = min(free_by_device.get(device, free), free)
        labels_by_device.setdefault(device, []).append(label)

    projected_total = source_bytes + materialized_copy_bytes + shard_bytes
    additional = materialize_bytes + shard_write_bytes
    lines = [
        f"HF/local source: {source_bytes / 1e9:.2f} GB",
        (
            f"Soup materialized copy: {materialized_copy_bytes / 1e9:.2f} GB "
            f"({'write required' if materialize_bytes else 'no write required'})"
        ),
        (
            f"Layer-shard cache: {shard_bytes / 1e9:.2f} GB "
            f"({'write required' if shard_write_bytes else 'reusable'})"
        ),
        f"Projected total on disk: {projected_total / 1e9:.2f} GB",
        f"Additional writes before training: {additional / 1e9:.2f} GB",
    ]
    for device in sorted(required_by_device):
        required = required_by_device[device]
        free = free_by_device[device]
        labels = " + ".join(labels_by_device[device])
        lines.append(
            f"Free on target volume ({labels}): {free / 1e9:.2f} GB"
        )
        if required > free:
            console.print(
                Panel("\n".join(lines), title="Layer streaming disk pre-flight")
            )
            raise ValueError(
                f"layer streaming needs {required / 1e9:.2f} GB of additional "
                f"disk space for {labels}, but only {free / 1e9:.2f} GB is free. "
                f"Refusing before copying or sharding. Free disk space, point "
                f"SOUP_SPECTRUM_CACHE_DIR / SOUP_LAYER_STREAM_CACHE_DIR at a "
                f"larger contained volume, or choose a smaller base."
            )
    console.print(Panel("\n".join(lines), title="Layer streaming disk pre-flight"))


def _distributed_launch() -> bool:
    """True when the process was launched by torchrun / accelerate / deepspeed.

    Those set ``WORLD_SIZE`` and HF then reports ``n_gpu == 1`` per process, so
    ``nn.DataParallel`` is never applied and the guard below must not fire.
    A malformed value is treated as non-distributed: refusing with a clear
    message beats proceeding into a raw torch error.
    """
    try:
        return int(os.environ.get("WORLD_SIZE", "1") or "1") > 1
    except (TypeError, ValueError):
        return False


def refuse_if_data_parallel(device) -> None:
    """Refuse layer streaming when HF Trainer would wrap the model in DataParallel.

    ``TrainingArguments`` sets ``_n_gpu = torch.cuda.device_count()`` for a
    non-distributed run, and ``Trainer._wrap_model`` then does
    ``model = nn.DataParallel(model)`` whenever ``n_gpu > 1``. DataParallel
    replicates by requiring every parameter to live on ``device_ids[0]``, and
    layer streaming keeps the decoder on ``meta`` by design — the two are
    incompatible by construction, not by accident.

    Without this the user gets torch's bare ``module must have its parameters
    and buffers on device cuda:0 ... but found one of them on device: meta``,
    which names nothing they set and points at nothing they can change.

    Refusing rather than silently dropping to one GPU is deliberate and matches
    the rest of this path (the VRAM fit decision refuses too): a run that
    quietly used 1 of 8 visible cards would look like it was using all of them.
    """
    if not str(device).startswith("cuda"):
        return
    import torch

    if not torch.cuda.is_available():
        return
    visible = torch.cuda.device_count()
    if visible <= 1 or _distributed_launch():
        return
    raise ValueError(
        f"training.stream_layers=true, but {visible} CUDA devices are visible. "
        f"transformers wraps the model in nn.DataParallel whenever more than one "
        f"GPU is visible and the run is not distributed, and DataParallel requires "
        f"every parameter on cuda:0 — layer streaming keeps the decoder on 'meta' "
        f"by design, so the two cannot be combined. Layer streaming is a "
        f"single-GPU technique: re-run with one card visible, e.g. "
        f"CUDA_VISIBLE_DEVICES=0, or set stream_layers=false to train resident "
        f"across all {visible}."
    )


class StreamingSetupMixin:
    """Builds a layer-streamed model in place of the resident load.

    Requires the host wrapper to provide ``self.device``,
    ``self._trust_remote_code``, and to accept ``self.model`` / ``self.tokenizer``
    / ``self._stream_runtime`` being set.
    """

    #: Rows that reach the model per dataset example. 1 for a plain causal LM
    #: step; 2 for a loss whose forward concatenates chosen and rejected.
    _STREAM_ROWS_PER_EXAMPLE = 1

    #: Set by :meth:`_setup_streaming_transformers`; absent on a resident run.
    _stream_runtime = None

    @staticmethod
    def _stream_shape_config(model_config):
        """Text sub-config for multimodal wrappers; plain config otherwise."""
        text_config = getattr(model_config, "text_config", None)
        return text_config if text_config is not None else model_config

    @staticmethod
    def _stream_intermediate_size(model_config) -> int:
        """Activation width estimate, including Qwen3.5 MoE text configs."""
        direct = int(getattr(model_config, "intermediate_size", 0) or 0)
        if direct:
            return direct
        moe = int(getattr(model_config, "moe_intermediate_size", 0) or 0)
        per_tok = int(getattr(model_config, "num_experts_per_tok", 0) or 0)
        shared = int(getattr(model_config, "shared_expert_intermediate_size", 0) or 0)
        return moe * max(per_tok, 1) + shared

    @staticmethod
    def _stream_total_experts(model_config) -> int:
        """Expert instances per layer when the config describes an MoE model."""
        shape_cfg = StreamingSetupMixin._stream_shape_config(model_config)
        for cfg in (shape_cfg, model_config):
            for key in (
                "num_local_experts",
                "num_experts",
                "n_routed_experts",
                "moe_num_experts",
            ):
                value = getattr(cfg, key, None)
                if isinstance(value, (int, float)) and value > 1:
                    return int(value)
        return 0

    @staticmethod
    def _stream_layer_budget_bytes(layer_specs) -> int:
        """Per-buffer bytes from the same union spec the runtime pool uses."""
        from soup_cli.utils.layer_stream import dtype_bytes
        from soup_cli.utils.layer_stream_runtime import RamSource

        merged = RamSource.merge_layer_specs(layer_specs)
        return sum(
            math.prod(shape) * dtype_bytes(stored) for shape, stored in merged.values()
        )

    @contextlib.contextmanager
    def _training_context(self, *contexts):
        """The `with` block every trainer runs `trainer.train()` inside.

        Its whole job is ordering: ``_close_stream_runtime`` is registered
        FIRST so it runs LAST, after every other context has unwound, and it
        runs even when training raises. That matters because an OOM mid-run is
        a realistic outcome on exactly the small cards this feature targets,
        and on the disk tier the runtime holds one open shard handle per decoder
        layer — which is the case that leaks across back-to-back runs in one
        process (`soup sweep`, the web UI).

        Yields the stack so a caller can enter further contexts conditionally.
        """
        with contextlib.ExitStack() as stack:
            stack.callback(self._close_stream_runtime)
            for context in contexts:
                stack.enter_context(context)
            yield stack

    def _setup_streaming_transformers(self, cfg, tcfg):
        """v0.72.0 BETA — layer streaming. The resident base load NEVER happens.

        Builds the skeleton on ``meta`` (``accelerate.init_empty_weights``),
        materialises only embeddings / final norm / LoRA, and streams each
        decoder layer from CPU RAM into a small pool of pre-allocated VRAM
        buffers. Peak VRAM becomes the size of ONE layer instead of the model.
        """
        from dataclasses import replace

        from peft import LoraConfig, TaskType
        from transformers import AutoConfig, AutoTokenizer

        # BEFORE the tokenizer load, the weight resolve and the shard write:
        # this configuration cannot work, and finding out minutes into disk I/O
        # is worse than finding out now.
        refuse_if_data_parallel(self.device)

        from soup_cli.utils.layer_shard import (
            QUANT_NF4,
            QUANT_NONE,
            checkpoint_source_components,
            estimate_oq_stream_cache_bytes,
            fingerprint_source_files,
            inspect_shard_cache,
            resolve_shard_dir,
            shard_checkpoint,
            source_weight_bytes,
        )
        from soup_cli.utils.layer_stream import (
            RAM_TIER_HEADROOM,
            TIER_DISK,
            TIER_RAM,
            build_stream_plan,
            dtype_bytes,
            estimate_stream_store_bytes,
            free_ram_bytes,
            render_stream_panel,
            resolve_disk_kind,
            resolve_stream_dtype,
            stream_arch_of,
            total_ram_bytes,
        )
        from soup_cli.utils.layer_stream_runtime import (
            RamSource,
            build_meta_skeleton,
            build_streamed_model,
            expandable_segments_status,
            extras_resident_bytes,
            large_layer_buffer_bytes,
            large_layer_store_bytes,
            quantised_layer_suffixes,
        )
        from soup_cli.utils.moe import detect_moe_model, get_moe_target_modules
        from soup_cli.utils.qwen4_ple import external_tensor_bytes
        from soup_cli.utils.spectrum_scan import resolve_model_weights

        console.print(f"[dim]Loading tokenizer: {cfg.base}[/]")
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.base, trust_remote_code=self._trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_config = AutoConfig.from_pretrained(
            cfg.base, trust_remote_code=self._trust_remote_code
        )
        # Allowlist, not a heuristic — a half-supported architecture streams
        # weights into the wrong module and mis-trains silently.
        arch = stream_arch_of(model_config)

        on_cuda = str(self.device).startswith("cuda")
        # #385 — ASK THE CARD. bf16 needs Ampere, and a T4 (Colab free), a P100
        # (Kaggle), a V100 or a GTX 16xx does not have it. Hardcoding bf16 here
        # made the entire free tier unsupported without saying so, and could not
        # fail on the Ampere card every published measurement came from.
        dtype = resolve_stream_dtype(str(self.device))

        # v0.72.2 — NF4. The decoder linears ship as packed nibbles + per-block
        # absmax, so the RAM store is ~0.26x its bf16 size; embeddings, norms and
        # an untied head stay at `dtype`, exactly as replace_with_bnb_linear
        # leaves them.
        quant = QUANT_NF4 if tcfg.quantization == "4bit" else QUANT_NONE
        _validate_qwen4_streaming_mode(
            arch=arch, task=getattr(cfg, "task", "sft"), quant=quant
        )
        # #321 — the streamed skeleton and the shards must quantise with the
        # SAME double-quant setting or the streamed-vs-resident bit-exactness
        # claim breaks. Read the flag once here (resolving the tri-state unset to
        # the shipped default) and thread it into both the sharder (its cache
        # already keys on double_quant) and the skeleton.
        double_quant = tcfg.double_quant_on

        shard_dir = resolve_shard_dir(cfg.base)
        quant_device_kind = str(self.device).split(":", 1)[0] if quant == QUANT_NF4 else ""

        def _disk_preflight(weights_plan) -> None:
            shard_estimate = estimate_stream_store_bytes(
                weights_plan.source_bytes,
                dtype=dtype,
                quant=quant,
                double_quant=double_quant,
            )
            oq_shard_estimate = estimate_oq_stream_cache_bytes(
                weights_plan.weights_dir,
                dtype=dtype,
                arch=arch,
            )
            if oq_shard_estimate is not None:
                shard_estimate = oq_shard_estimate
            cached = None
            if not weights_plan.needs_materialization:
                source_components = checkpoint_source_components(
                    weights_plan.weights_dir,
                    weights_plan.source_files,
                    include_config=arch == "qwen4_exp",
                )
                cached, _reason = inspect_shard_cache(
                    shard_dir,
                    dtype,
                    fingerprint_source_files(source_components),
                    source_components,
                    quant,
                    double_quant,
                    quant_device_kind,
                    "qwen4_ple" if arch == "qwen4_exp" else "",
                )
            _render_stream_disk_preflight(
                source_bytes=weights_plan.source_bytes,
                materialized_copy_bytes=weights_plan.materialized_copy_bytes,
                materialize_bytes=weights_plan.materialize_bytes,
                materialized_path=weights_plan.weights_dir,
                shard_bytes=shard_estimate,
                shard_write_bytes=0 if cached is not None else shard_estimate,
                shard_path=shard_dir,
            )

        weights_dir = resolve_model_weights(
            cfg.base,
            before_materialize=_disk_preflight,
        )

        # Cheap size probe BEFORE sharding: re-writing a checkpoint we are
        # about to refuse for not fitting in RAM costs minutes of disk I/O.
        # Charged at the STREAMED rate, not the on-disk one — an 8B bf16
        # checkpoint is 16 GB on disk but only ~4.2 GB of NF4 store, and
        # comparing the raw file size would refuse exactly the runs NF4 enables.
        early_free_ram = free_ram_bytes()
        if early_free_ram is not None:
            source_bytes = source_weight_bytes(weights_dir)
            store_estimate = estimate_stream_store_bytes(
                source_bytes, dtype=dtype, quant=quant, double_quant=double_quant
            )
            # Qwen4's source size includes the PLE table, which the sharder
            # leaves external. Its exact RAM/disk decision is made from the
            # safetensors header below; counting it here would reject the very
            # `stream_ngram_source: disk` run this path enables.
            if (
                arch != "qwen4_exp"
                and store_estimate >= early_free_ram * RAM_TIER_HEADROOM
                and tcfg.stream_source == "ram"
            ):
                as_streamed = (
                    ""
                    if quant == QUANT_NONE
                    else f" ({store_estimate / 1e9:.1f} GB once quantised to NF4)"
                )
                raise ValueError(
                    f"training.stream_source='ram' but {cfg.base} is "
                    f"{source_bytes / 1e9:.1f} GB on disk{as_streamed} and only "
                    f"{early_free_ram / 1e9:.1f} GB of RAM is free. Set "
                    f"stream_source='auto' to fall back to the NVMe disk tier, "
                    f"free RAM, or pick a smaller base."
                )

        # The authoritative list of weights to quantise is whatever
        # replace_with_bnb_linear actually converts, read off a meta skeleton —
        # not a hard-coded name list that would drift per architecture.
        #
        # This builds a second, throwaway skeleton (build_streamed_model makes
        # its own). Deliberate: a meta skeleton allocates NO weight storage, so
        # the cost is module-tree construction only, and threading a pre-built
        # model into build_streamed_model would couple suffix discovery to model
        # construction for no memory saving.
        quant_suffixes = ()
        moe_targets = None
        is_moe = False
        if quant == QUANT_NF4:
            probe = build_meta_skeleton(
                cfg.base,
                dtype=dtype,
                quant=quant,
                trust_remote_code=self._trust_remote_code,
            )
            is_moe = detect_moe_model(probe)
            if tcfg.moe_lora and is_moe:
                moe_targets = get_moe_target_modules(probe)
            quant_suffixes = quantised_layer_suffixes(probe)
            del probe
        elif tcfg.moe_lora:
            probe = build_meta_skeleton(
                cfg.base,
                dtype=dtype,
                quant=quant,
                trust_remote_code=self._trust_remote_code,
            )
            is_moe = detect_moe_model(probe)
            if is_moe:
                moe_targets = get_moe_target_modules(probe)
            del probe

        console.print(f"[dim]Preparing layer shards -> {shard_dir}[/]")
        index = shard_checkpoint(
            weights_dir,
            shard_dir,
            dtype=dtype,
            arch=arch,
            quant=quant,
            quant_suffixes=quant_suffixes,
            double_quant=double_quant,
            # Quantise on the device that will run the model: CPU and CUDA agree
            # on the packed nibbles but not on every float32 nested statistic.
            quant_device=str(self.device),
            notify=console.print,
        )

        layer_specs = RamSource.layer_specs_from_shards(shard_dir, index.n_layers)
        # Measured from the shard headers, not derived from `total_params`:
        # under NF4 a layer holds packed uint8 alongside float32 statistics, so
        # element counts no longer convert to bytes at a single rate.
        layer_byte_sizes = [
            sum(math.prod(shape) * dtype_bytes(stored) for shape, stored in per_layer.values())
            for per_layer in layer_specs
        ]
        layer_bytes = self._stream_layer_budget_bytes(layer_specs)
        layer_store_bytes = sum(layer_byte_sizes)
        embed_bytes = extras_resident_bytes(shard_dir)
        large_store_bytes = large_layer_store_bytes(shard_dir, index)
        large_buffer_bytes = large_layer_buffer_bytes(shard_dir, index)
        ngram_bytes = external_tensor_bytes(
            getattr(index, "external_tensors", None) or {}
        )

        free_ram = free_ram_bytes()
        total_ram = total_ram_bytes()
        if free_ram is None:
            console.print(
                "[yellow]psutil unavailable — cannot size the RAM tier; "
                "proceeding and letting the allocation fail loudly if it must[/]"
            )
            free_ram = (
                layer_bytes * index.n_layers + large_store_bytes + embed_bytes
            ) * 10

        store_total = layer_store_bytes + large_store_bytes
        ngram_source = "disk"
        if ngram_bytes:
            requested_ngram = tcfg.stream_ngram_source
            oq_ngram = any(
                hasattr(spec, "bits") for spec in index.external_tensors.values()
            )
            ngram_source = _resolve_qwen4_ngram_source(
                oq_ngram=oq_ngram,
                requested=requested_ngram,
                store_total=store_total,
                ngram_bytes=ngram_bytes,
                free_ram=free_ram,
                resident_ram=embed_bytes,
                total_ram=total_ram,
                stream_source=tcfg.stream_source,
            )
            storage = "CPU RAM"
            if ngram_source == "disk":
                ngram_disk = resolve_disk_kind(
                    weights_dir, tcfg.stream_disk_kind, notify=console.print
                )
                ngram_disk_kind = ngram_disk.kind
                _validate_qwen4_ngram_disk(
                    disk_kind=ngram_disk_kind, weights_dir=weights_dir
                )
                storage = f"read-only {ngram_disk_kind.upper()} mmap"
            console.print(
                f"[cyan]Qwen4 PLE:[/] {ngram_bytes / 1e9:.2f} GB via {storage} "
                f"(stream_ngram_source={tcfg.stream_ngram_source!r})"
            )
        _warn_if_ngram_source_unused(
            arch=arch,
            requested=getattr(tcfg, "stream_ngram_source", "auto"),
            ngram_bytes=ngram_bytes,
            notify=console.print,
        )
        # Checked BEFORE build_stream_plan so a `ram`-only run is refused with
        # the message about stream_source rather than choose_tier's generic
        # "needs NVMe or more RAM" — and without paying the ~9 s disk probe for
        # an answer that cannot change the outcome.
        required_ram = store_total + (ngram_bytes if ngram_source == "ram" else 0)
        _validate_qwen4_ngram_ram_fit(
            stream_source=tcfg.stream_source,
            ngram_source=ngram_source,
            required_ram=required_ram,
            free_ram=free_ram,
            resident_ram=embed_bytes,
            total_ram=total_ram,
        )
        plan_free_ram = free_ram
        if ngram_source == "ram":
            plan_free_ram = max(
                0,
                free_ram - math.ceil(ngram_bytes / RAM_TIER_HEADROOM),
            )
        plan = build_stream_plan(
            arch=arch,
            n_layers=index.n_layers,
            layer_bytes=layer_bytes,
            embed_bytes=embed_bytes,
            store_bytes=layer_store_bytes,
            large_store_bytes=large_store_bytes,
            large_buffer_bytes=large_buffer_bytes,
            available_ram_bytes=plan_free_ram,
            total_ram_bytes=total_ram,
            # The page-locked ceiling is a property of the box, not of free RAM;
            # rather than probe it destructively we attempt the pinned store and
            # fall back loudly (see layer_stream_runtime._build_source).
            pinned_limit_bytes=None,
            buffers=tcfg.stream_buffers,
            # v0.72.3: the REAL media type, not a constant. Passed as a callable
            # because probing costs ~9 s on Windows and the answer only matters
            # when the base does not fit in RAM. #365: honour a
            # stream_disk_kind override (with a loud detected-vs-override notice)
            # for a disk the auto-probe still misreads.
            disk_kind=lambda: resolve_disk_kind(
                shard_dir, tcfg.stream_disk_kind, notify=console.print
            ),
            # #366: training.stream_pin (None/False/True) overrides the automatic
            # pinning choice so the pageable escape hatch is reachable from config.
            stream_pin=tcfg.stream_pin,
        )
        # v0.72.3 — the disk overflow tier is live, so a base that does not fit
        # in RAM is no longer fatal. `stream_source` decides: 'ram' insists,
        # 'disk' forces, 'auto' (the default) takes RAM when it fits and falls
        # back to disk when it does not. build_stream_plan already refused a
        # non-NVMe disk, so reaching here with tier='disk' means NVMe.
        tier = TIER_DISK if tcfg.stream_source == "disk" else plan.tier
        if tier != plan.tier:
            # The panel is rendered from `plan`, so a forced tier has to be
            # reflected there or the pre-flight reports "tier ram" immediately
            # before the runtime announces it is streaming from disk. Every
            # field that describes the RAM store is corrected with it, so no
            # consumer can read a stale value.
            plan = replace(
                plan,
                tier=tier,
                store_bytes=0,
                large_store_bytes=0,
                pinned=False,
                notes=plan.notes
                + (
                    "streaming from disk because stream_source='disk' was set, "
                    "not because RAM was short. Nothing is held resident, and "
                    "the slowdown versus the RAM tier is unmeasured on this "
                    "hardware.",
                ),
            )
        # v0.72.3 — VRAM pre-flight. Streaming bounds the WEIGHTS; activations
        # and the logits tensor are untouched by it and both scale with batch x
        # seq. On a large-vocab model the logits term alone dwarfs the buffer
        # pool (measured: 146x at batch 8), so a plan that reports only tier and
        # buffer sizes will happily green-light a config that cannot run.
        forecast_lines, probe_plan = self._stream_budget_lines(
            cfg,
            tcfg,
            model_config=model_config,
            layer_bytes=layer_bytes,
            embed_bytes=embed_bytes,
            large_layer_bytes=large_buffer_bytes,
            index=index,
            on_cuda=on_cuda,
        )
        console.print(render_stream_panel(plan, forecast_lines))
        console.print(
            "[yellow]Layer streaming is BETA:[/] slower than resident training, "
            "but this model may not run resident on this card at all."
        )
        if on_cuda:
            enabled, why_not = expandable_segments_status()
            if not enabled:
                console.print(
                    f"[dim]expandable_segments allocator hint not enabled: {why_not}[/]"
                )

        from soup_cli.utils.peft_wiring import resolve_lora_target_modules

        target_modules = resolve_lora_target_modules(model_config, tcfg.lora.target_modules)
        if tcfg.moe_lora and is_moe and moe_targets:
            target_modules = moe_targets
            console.print(
                f"[green]ScatterMoE LoRA:[/] targeting {len(moe_targets)} module patterns"
            )
        lora_config = LoraConfig(
            r=tcfg.lora.r,
            lora_alpha=tcfg.lora.alpha,
            lora_dropout=tcfg.lora.dropout,
            target_modules=target_modules,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
            use_dora=tcfg.lora.use_dora,
            use_rslora=tcfg.lora.use_rslora,
        )

        # #366 / #434 — CUDA host pinning is inapplicable on every non-CUDA
        # target. An explicit stream_pin=true is honoured by saying so, not by
        # dropping it silently. On MPS the pageable CPU source is also what keeps
        # the frozen base out of the accelerator allocator.
        if tcfg.stream_pin is True and not on_cuda:
            console.print(
                "[yellow]training.stream_pin=true, but no CUDA device is present: "
                "CUDA host pinning does not apply to this target. Proceeding with "
                "a pageable CPU source.[/]"
            )

        model, runtime = build_streamed_model(
            model_id=cfg.base,
            shard_dir=shard_dir,
            index=index,
            lora_config=lora_config,
            device=self.device,
            dtype=dtype,
            buffers=tcfg.stream_buffers,
            pin=plan.pinned and on_cuda,
            # #366: on the RAM tier stream_pin=true refuses rather than silently
            # falling back to a pageable store; on the disk tier the runtime
            # announces that pinning is inapplicable (no RAM store to lock); on
            # non-CUDA targets the notice above covers it. require_pin only carries
            # the CUDA RAM-tier refusal, so it is gated on a real CUDA device.
            require_pin=(tcfg.stream_pin is True) and on_cuda,
            seed=tcfg.seed if getattr(tcfg, "seed", None) is not None else 0,
            trust_remote_code=self._trust_remote_code,
            console=console,
            quant=quant,
            double_quant=double_quant,
            tier=tier,
            weights_dir=weights_dir,
            ngram_source=ngram_source,
        )
        self.model = model
        self._stream_runtime = runtime
        if probe_plan is not None:
            self._run_stream_vram_probe(model, probe_plan)
        stats = runtime.stats()
        if stats["tier"] == TIER_RAM:
            source_line = (
                f"{stats['store_bytes'] / 1e9:.2f} GB "
                f"{'pinned' if stats['pinned'] else 'pageable'} RAM store"
            )
        else:
            source_line = (
                f"streamed from DISK ({stats['disk_bytes'] / 1e9:.2f} GB on an "
                f"NVMe volume, nothing held resident)"
            )
        large_runtime_buffer = stats.get("large_buffer_bytes", 0)
        decoder_buffers = stats["buffer_bytes"] - large_runtime_buffer
        buffer_line = (
            f"{stats['buffers']} x "
            f"{decoder_buffers / stats['buffers'] / 1e6:.0f} MB decoder buffers + "
            f"1 x {large_runtime_buffer / 1e6:.0f} MB large-layer slot"
        )
        console.print(
            f"[green]Layer streaming ready:[/] {stats['n_layers']} layers, "
            f"{source_line}, {buffer_line}"
        )

    def _close_stream_runtime(self) -> None:
        """Release the streaming weight source, if this run had one."""
        runtime = getattr(self, "_stream_runtime", None)
        if runtime is not None:
            runtime.close()

    def _estimate_adapter_params(self, tcfg, model_config) -> int:
        """Trainable adapter parameters, before the model exists.

        Deliberately coarse and biased HIGH: it assumes every targeted module is
        hidden x hidden. Gate/up/down projections are larger, but the whole
        adapter term is ~0.5% of a streaming step's peak, so precision here buys
        nothing while under-counting would eat into the safety margin.
        """
        shape_cfg = StreamingSetupMixin._stream_shape_config(model_config)
        hidden = int(getattr(shape_cfg, "hidden_size", 0) or 0)
        layers = int(getattr(shape_cfg, "num_hidden_layers", 0) or 0)
        targets = tcfg.lora.target_modules
        experts = StreamingSetupMixin._stream_total_experts(model_config)
        if isinstance(targets, (list, tuple)):
            target_names = [str(name) for name in targets]
            n_targets = len(target_names)
            if getattr(tcfg, "moe_lora", False) and experts > 1:
                expert_suffixes = {"gate_proj", "up_proj", "down_proj", "w1", "w2", "w3"}
                expert_patterns = {name for name in target_names if name in expert_suffixes}
                n_targets += (experts - 1) * len(expert_patterns)
        elif getattr(tcfg, "moe_lora", False) and experts > 1:
            n_targets = 4 + 3 * experts
        else:
            n_targets = 4
        return layers * n_targets * 2 * tcfg.lora.r * hidden

    def _stream_budget_lines(
        self,
        cfg,
        tcfg,
        *,
        model_config,
        layer_bytes,
        embed_bytes,
        index,
        on_cuda,
        large_layer_bytes=0,
    ):
        """Predict peak VRAM + bracket throughput, and REFUSE a run that cannot fit.

        Returns ``(panel_lines, probe_plan)``, where ``probe_plan`` is ``None``
        unless ``training.stream_vram_probe`` asked for the measured gate (#349).

        Raises when the step is predicted not to fit: on Linux that would be a
        hard OOM, and on Windows something worse — WDDM spills to host memory
        without raising, so the run silently becomes an order of magnitude
        slower and looks like the feature is merely slow. Under the probe the
        prediction is demoted to advice instead, because refusing here would
        prevent the measurement that exists to overrule it.
        """
        from soup_cli.utils.layer_stream import (
            LOGITS_BYTES_PER_ELEMENT,
            accumulation_advice,
            calibrated_logits_bytes_per_element,
            decide_stream_fit,
            estimate_logits_bytes,
            estimate_stream_peak_vram,
            forecast_stream_throughput,
            resolve_available_vram_bytes,
        )
        from soup_cli.utils.layer_stream_runtime import measure_gemm_tflops

        shape_cfg = self._stream_shape_config(model_config)
        vocab = int(getattr(shape_cfg, "vocab_size", 0) or 0)
        hidden = int(getattr(shape_cfg, "hidden_size", 0) or 0)
        inter = self._stream_intermediate_size(shape_cfg)
        seq_len = int(cfg.data.max_length)
        batch = tcfg.batch_size if isinstance(tcfg.batch_size, int) else 1
        # v0.72.4 — a paired loss concatenates chosen and rejected into ONE
        # tensor, so twice the rows reach the model per configured batch. The
        # estimator's contract is that it never under-predicts; budgeting a
        # paired loss at 1x rows would halve the logits term, which is the
        # dominant one (measured 146x the buffer pool at batch 8).
        rows = batch * self._STREAM_ROWS_PER_EXAMPLE
        if not (vocab and hidden and inter):
            # Never silently: skipping the budget also skips the refusal that
            # stops a run from OOMing (or, on Windows, spilling to host memory
            # and running an order of magnitude slower with no error at all).
            console.print(
                "[yellow]Layer streaming could not read vocab_size / hidden_size "
                "/ intermediate_size from the model config, so peak VRAM cannot "
                "be predicted — the pre-flight fit check is SKIPPED for this "
                "run.[/]"
            )
            # No shape to probe either: the probe needs vocab_size to build the
            # synthetic batch, which is one of the fields that could not be read.
            return (), None

        # calibrated_logits_bytes_per_element() is floored at LOGITS_BYTES_PER_ELEMENT,
        # so forwarding it here can only raise the budget, never lower it (issue #348).
        calibrated = calibrated_logits_bytes_per_element()
        predicted = estimate_stream_peak_vram(
            layer_bytes=layer_bytes,
            buffers=tcfg.stream_buffers,
            extras_bytes=embed_bytes,
            adapter_params=self._estimate_adapter_params(tcfg, model_config),
            vocab_size=vocab,
            hidden_size=hidden,
            intermediate_size=inter,
            n_layers=index.n_layers,
            seq_len=seq_len,
            batch_size=rows,
            logits_bytes_per_element=calibrated,
            large_layer_bytes=large_layer_bytes,
        )
        logits = estimate_logits_bytes(
            vocab_size=vocab, seq_len=seq_len, batch_size=rows, bytes_per_element=calibrated
        )
        paired = (
            "" if rows == batch else f" ({rows} rows — chosen+rejected are one concatenated tensor)"
        )
        lines = [
            f"  peak VRAM    ~{predicted / 1e9:.2f} GB at batch {batch} x seq "
            f"{seq_len}{paired} (logits {logits / 1e9:.2f} GB)"
        ]
        if calibrated > LOGITS_BYTES_PER_ELEMENT:
            lines.append(
                f"  logits       calibrated {calibrated:.3f} B/element on this stack, "
                f"above the shipped {LOGITS_BYTES_PER_ELEMENT:.0f}: budget raised to match"
            )

        if not on_cuda:
            # Nothing to measure against: the probe reads CUDA peak counters.
            # Say so rather than no-opping, mirroring the unreadable-config skip
            # above — a silently inactive gate reads exactly like an active one.
            if tcfg.stream_vram_probe:
                console.print(
                    "[yellow]training.stream_vram_probe is set but this run is "
                    "not on CUDA, so there is no peak to measure — the pre-flight "
                    "falls back to the predicted budget.[/]"
                )
            return tuple(lines), None

        import torch

        measured_available = int(torch.cuda.mem_get_info()[0])
        available = resolve_available_vram_bytes(
            measured_bytes=measured_available, override_bytes=tcfg.stream_vram_override
        )
        fit = decide_stream_fit(predicted_bytes=predicted, available_bytes=available)
        if not fit.fits:
            if not tcfg.stream_vram_probe:
                raise ValueError(fit.reason)
            if predicted > available * _PROBE_DEFERRAL_CEILING:
                # The probe exists to settle a DISAGREEMENT, and the largest
                # disagreement ever measured is 21% (formula 0.787x the real peak
                # at seq 6144). A config predicted several times over budget is
                # not a disagreement, and deferring it would trade a free
                # arithmetic refusal for minutes of sharding plus a real
                # allocation attempt at that shape — driven by a soup.yaml whose
                # author need not be whoever runs it.
                raise ValueError(
                    f"{fit.reason} training.stream_vram_probe cannot overrule a "
                    f"prediction this far over budget "
                    f"({predicted / available:.1f}x): the probe corrects a "
                    f"margin, not an order of magnitude."
                )
            # #349 — with the probe on, the formula is advisory: refusing here
            # would stop the measurement that exists to overrule it from ever
            # being taken. Say so rather than passing silently, because the
            # build about to happen is minutes of work that may still be refused.
            console.print(
                f"[yellow]Predicted over budget "
                f"({predicted / 1e9:.2f} GB vs {available / 1e9:.2f} GB free), but "
                f"training.stream_vram_probe is on: measuring the real peak "
                f"before deciding.[/]"
            )
        if tcfg.stream_vram_override is None:
            lines.append(f"  free VRAM    {available / 1e9:.2f} GB")
        else:
            lines.append(
                f"  free VRAM    {available / 1e9:.2f} GB (training.stream_vram_override; "
                f"driver reports {measured_available / 1e9:.2f} GB)"
            )

        # A per-card TFLOPS constant baked into the source would be a
        # fabrication; measuring the user's own card in this session is the only
        # honest input, and the result is reported as a bracket because real
        # streamed runs landed at 68%-100% of their measured ceiling.
        ceiling = measure_gemm_tflops(device=str(self.device))
        if ceiling is not None and index.total_params:
            shaped = forecast_stream_throughput(
                params=index.total_params,
                effective_tflops=ceiling.tflops,
                tokens_per_epoch=0,
                sm_clock_mhz=ceiling.sm_clock_mhz,
            )
            clock = f" @ {ceiling.sm_clock_mhz} MHz" if ceiling.sm_clock_mhz else ""
            lines.append(
                f"  forecast     {shaped.tokens_per_sec_low:.0f}-"
                f"{shaped.tokens_per_sec_ceiling:.0f} tok/s — a compute-bound "
                f"bound, not a promise"
            )
            lines.append(
                f"               (from {ceiling.tflops:.2f} TFLOPS measured on "
                f"this card now using {ceiling.dtype}{clock})"
            )
        advice = accumulation_advice(batch_size=batch, accum=tcfg.gradient_accumulation_steps)
        if advice is not None:
            lines.append(f"  [yellow]![/] {advice}")
        plan = None
        if tcfg.stream_vram_probe:
            plan = _ProbePlan(
                rows=rows,
                seq_len=seq_len,
                vocab_size=vocab,
                predicted_bytes=predicted,
                available_bytes=available,
            )
        return tuple(lines), plan

    def _run_stream_vram_probe(self, model, plan: _ProbePlan) -> None:
        """Measure one real step and let THAT decide, not the formula (#349).

        Raises when the measured peak does not fit. The streaming runtime is
        released first: it holds the pinned RAM store, and `setup()` raising is
        outside the ExitStack that `_training_context` installs around training.
        """
        from soup_cli.utils.layer_stream import decide_measured_fit
        from soup_cli.utils.layer_stream_runtime import measure_step_peak_bytes

        try:
            peak = measure_step_peak_bytes(
                model,
                rows=plan.rows,
                seq_len=plan.seq_len,
                vocab_size=plan.vocab_size,
                device=str(self.device),
            )
        except Exception:
            # measure_step_peak_bytes validates its own arguments and raises
            # before its internal handler exists. Unreachable today (the plan is
            # only built from a validated shape), but the docstring promises the
            # runtime is released before anything propagates, and a promise that
            # holds only for the paths written so far is the kind that breaks
            # when a second caller appears.
            self._close_stream_runtime()
            raise
        if peak is None:
            # Instrument failure, not a verdict. Fall back to the formula so the
            # run is never left with no gate at all — but if the formula had
            # already refused, honour that refusal rather than proceeding on
            # the strength of a probe that did not happen.
            console.print(
                "[yellow]The measured VRAM probe could not run; falling back to "
                "the predicted budget.[/]"
            )
            if plan.predicted_bytes > plan.available_bytes:
                self._close_stream_runtime()
                raise ValueError(
                    f"a streaming step is predicted to need "
                    f"{plan.predicted_bytes / 1e9:.2f} GB of VRAM but only "
                    f"{plan.available_bytes / 1e9:.2f} GB is free, and the "
                    f"measured probe that could have overruled that prediction "
                    f"failed to run. Lower training.batch_size or "
                    f"data.max_length."
                )
            return
        if peak.failed:
            # The probe ran a real CUDA op and it raised. The fit is unknown and
            # the context may be unusable, so this refuses rather than falling
            # back to the prediction: "the arithmetic was happy" is not a reason
            # to keep driving a device that just failed.
            self._close_stream_runtime()
            raise ValueError(
                f"the measured VRAM probe raised {peak.error} while running one "
                f"step at batch {plan.rows} x seq {plan.seq_len}. The fit could "
                f"not be established and the CUDA context may no longer be "
                f"usable, so this run is refused rather than continued on the "
                f"predicted budget ({plan.predicted_bytes / 1e9:.2f} GB). Re-run "
                f"without training.stream_vram_probe to use the prediction."
            )
        if peak.oom:
            self._close_stream_runtime()
            raise ValueError(
                f"a streaming step at batch {plan.rows} x seq {plan.seq_len} ran "
                f"out of VRAM while being measured (predicted "
                f"{plan.predicted_bytes / 1e9:.2f} GB, "
                f"{plan.available_bytes / 1e9:.2f} GB free). Lower "
                f"training.batch_size or data.max_length."
            )
        fit = decide_measured_fit(
            measured_bytes=peak.peak_bytes,
            predicted_bytes=plan.predicted_bytes,
            available_bytes=plan.available_bytes,
        )
        console.print(
            f"[dim]measured peak {peak.peak_bytes / 1e9:.2f} GB "
            f"({peak.reserved_bytes / 1e9:.2f} GB reserved) in "
            f"{peak.seconds:.2f} s at batch {plan.rows} x seq {plan.seq_len}; "
            f"predicted {plan.predicted_bytes / 1e9:.2f} GB[/]"
        )
        if not fit.fits:
            self._close_stream_runtime()
            raise ValueError(fit.reason)
