"""Shared PEFT wiring helpers — LoRA config, multi-trainer ReLoRA, and patches.

Centralises PEFT LoRA construction plus the v0.39.0 Part B (ReLoRA callback)
and Part D (surgical PEFT patches) wiring previously inlined only in the SFT
trainer. Every Transformers-backend trainer calls these helpers from its setup
and training paths.

Helpers swallow per-patch exceptions at DEBUG level — best-effort by design;
training never crashes because a Gemma4 swap or 3-D dropout strip failed.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)


# PEFT 0.20 has no automatic LoRA mapping for the Qwen3.5 text architectures.
# Qwen3.5 mixes fused linear-attention blocks with ordinary attention blocks,
# so the policy covers the input/output projections of both block kinds.
QWEN35_TEXT_LORA_TARGETS = (
    "q_proj",
    "v_proj",
    "in_proj_qkv",
    "out_proj",
)

# Qwen4-Exp combines ordinary QSA projections, fused Gated DeltaNet
# projections, shared experts, PLE projections, and gated-residual mixers.
# PEFT has no qwen4_exp_text default mapping yet. A short suffix list would
# silently omit one of those new paths, so ``all-linear`` is the deliberate
# text-decoder policy. The routed experts themselves are 3-D nn.Parameters,
# not nn.Linear modules; adapting those needs PEFT ``target_parameters`` and is
# tracked separately from this safe linear-module baseline.
QWEN4_EXP_TEXT_LORA_TARGETS = "all-linear"

# Qwen4-Exp's routed experts keep their projections as two raw 3-D
# ``nn.Parameter`` tensors per decoder layer. ``all-linear`` cannot see them;
# PEFT 0.20's ``target_parameters`` path can adapt both, with the expert axis at
# dimension zero.
QWEN4_EXP_TEXT_LORA_TARGET_PARAMETERS = (
    "mlp.experts.gate_up_proj",
    "mlp.experts.down_proj",
)


#: PEFT has no LoRA mapping for any MoE text architecture Soup ships a recipe
#: for: every ``model_type`` below returns ``None`` from peft 0.20's
#: ``TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING``, so
#: ``target_modules: auto`` reached peft as ``None`` and the attach raised
#: ``No target_modules passed but also no target_parameters found`` (#1070).
#:
#: Measured while building this table, and worth knowing before extending it:
#: peft's default mapping is unreachable for a MoE model whether or not it has
#: an entry. ``LoraModel._prepare_adapter_config`` applies the default only
#: ``if peft_config.target_modules is None``, and peft's MoE config conversion
#: has already replaced ``None`` with an empty ``set()`` by then. Traced on
#: peft 0.20 / transformers 5.16.1 with tiny CPU models:
#:
#:     [llama]     before=None    after={'q_proj', 'v_proj'}  -> attach OK
#:     [mixtral]   before=set()   after=set()                 -> ValueError
#:     [qwen3_moe] before=set()   after=set()                 -> ValueError
#:
#: ``mixtral`` is the one MoE ``model_type`` peft DOES map, and it fails anyway.
#: It is deliberately absent from this table -- Soup ships no ``mixtral`` recipe,
#: and delegating keeps today's behaviour rather than widening #1070's fix into
#: an architecture nobody reported.
#:
#: Each entry is the attention projections that architecture actually defines,
#: enumerated by building the base's config from the Hub, shrinking it, and
#: instantiating it on the meta device (see the PR for the probe). Routed
#: experts are deliberately absent: on ``qwen3_moe``, ``deepseek_v3`` and
#: ``deepseek_v4`` they are 3-D ``nn.Parameter`` tensors that ``target_modules``
#: cannot address at all, and adapting them is ``target_parameters`` work
#: (#798). This is the safe linear-module baseline, the same policy as
#: :data:`QWEN35_TEXT_LORA_TARGETS`.
_DEEPSEEK_V3_ATTENTION = (
    "q_a_proj",
    "q_b_proj",
    "kv_a_proj_with_mqa",
    "kv_b_proj",
    "o_proj",
)

MOE_TEXT_LORA_TARGETS: dict[str, Any] = {
    "qwen3_moe": ("q_proj", "k_proj", "v_proj", "o_proj"),
    # GLM-4.6 (``zai-org/GLM-4.6``, 92 layers). Ordinary attention, the same
    # four projections as ``qwen3_moe``: enumerated on the meta device, every
    # layer carries ``self_attn.{q,k,v,o}_proj`` and nothing else that is a
    # ``nn.Linear``. The ``q_norm`` / ``k_norm`` siblings are RMSNorms, not
    # Linears, so a suffix list cannot reach them by accident. Not ``glm_moe_dsa``
    # (GLM-5/5.1): that one is V3-shaped attention plus the DSA indexer, and its
    # names are different. Previously invisible to the ratchet because the
    # recipe named ``THUDM/glm-4.6``, which 404s (#1132 repoints it).
    "glm4_moe": ("q_proj", "k_proj", "v_proj", "o_proj"),
    # Granite 4.0 (``ibm-granite/granite-4.0-tiny-base-preview``) is a HYBRID:
    # ``config.layer_types`` is 36 ``linear_attention`` blocks and only 4
    # ``full_attention`` ones, and only those 4 of the 40 decoder layers define a
    # ``self_attn`` at all (measured: layers 5, 15, 25, 35). So this entry adapts
    # a TENTH of the decoder, which is why ``granitemoehybrid`` also has a row in
    # :data:`PARTIAL_COVERAGE_NOTES` -- a comment here is not where a user reads
    # why their adapter is small.
    #
    # The other 36 layers are Mamba-2 blocks (``mamba.in_proj`` 36,
    # ``mamba.out_proj`` 36, plus a ``conv1d``) and every layer carries
    # ``shared_mlp.input_linear`` / ``shared_mlp.output_linear`` (40 each). They
    # are deliberately NOT adapted: this table is the attention-projection
    # baseline on every other row, and whether LoRA on a state-space projection
    # trains well is a separate decision from #1070's "auto resolved to nothing".
    # A user who wants them writes an explicit ``target_modules`` list.
    # Routed experts are fused 3-D ``block_sparse_moe.experts`` parameters, so
    # ``target_modules`` cannot reach them here either (#798).
    "granitemoehybrid": ("q_proj", "k_proj", "v_proj", "o_proj"),
    # Two MoE bases whose recipes set no MoE flag at all, so the flag-based
    # sizing of this table missed them; found by attaching every shipped config
    # on the meta device. Both keep their experts as fused 3-D parameters, like
    # qwen3_moe, and expose the same four attention projections.
    "gpt_oss": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "minimax_m2": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "deepseek_v3": _DEEPSEEK_V3_ATTENTION,
    # V4 splits the output projection and drops the MQA compression on the
    # key/value side, so its names are not V3's.
    "deepseek_v4": ("q_a_proj", "q_b_proj", "kv_proj", "o_a_proj", "o_b_proj"),
    # GLM-5.1 is V3-shaped attention plus a DSA indexer (``wq_b``, ``wk``,
    # ``weights_proj``). The indexer is left alone: it selects which tokens
    # attend, and adapting it is a different decision from adapting attention.
    "glm_moe_dsa": _DEEPSEEK_V3_ATTENTION,
    # Kimi K2.5/K2.6 declare ``architectures: ["DeepseekV3ForCausalLM"]`` and an
    # ``auto_map`` onto ``modeling_deepseek.DeepseekV3ForCausalLM`` in their
    # ``text_config``, so the text tower is V3's module layout. Taken from the
    # config rather than enumerated, because the repo needs trust_remote_code
    # and this table is not worth executing remote code for.
    "kimi_k25": _DEEPSEEK_V3_ATTENTION,
    # ``kimi_k2`` is both the text tower of K2.5/K2.6 and the OUTER type of
    # Kimi-K2-Thinking, a shipped base with no MoE flag in its recipe -- so it is
    # sweep-derived, not only defensive. (The first version of this table had it
    # for the text tower and covered Kimi-K2-Thinking by luck; #1102 review, F3.)
    "kimi_k2": _DEEPSEEK_V3_ATTENTION,
    # A vision-language wrapper: ``vision_tower`` has its own ``q_proj`` /
    # ``k_proj`` / ``v_proj``, so a suffix list would silently adapt the image
    # encoder for a text fine-tune. peft treats a string as a regex, which is
    # how the language tower is named without a name-match fallback.
    # ``.*`` in front, because peft fullmatches a STRING target against the whole
    # module key, and the key under the class vision SFT loads
    # (``AutoModelForImageTextToText``) is ``model.language_model...``. My first
    # version anchored at ``language_model`` and matched nothing (#1102 review,
    # F2); its tests passed because they used hand-written keys without the
    # ``model.`` prefix, not keys read off the model.
    "minimax_m3_vl": r".*language_model\..*\.self_attn\.(q_proj|k_proj|v_proj|o_proj)",
    # Defensive, not sweep-derived: no shipped base reports this type. It is the
    # text tower MiniMax-M3's wrapper exposes, and a config loaded without the
    # wrapper reaches the resolver as this type instead of ``minimax_m3_vl``.
    "minimax_m3_vl_text": ("q_proj", "k_proj", "v_proj", "o_proj"),
}

#: Entries of :data:`MOE_TEXT_LORA_TARGETS` that cover only PART of the decoder,
#: each with what is left unadapted. A declared table rather than a branch in the
#: resolver: the next hybrid architecture is a row here, and an entry that grows
#: partial coverage is a reviewed edit instead of an ``if`` someone forgets.
#:
#: The note is PRINTED, not just commented, because "your LoRA touched a tenth of
#: the model" is not something a user can be expected to infer from a silent
#: ``target_modules: auto``.
PARTIAL_COVERAGE_NOTES: dict[str, str] = {
    "granitemoehybrid": (
        "the remaining layers are Mamba-2 blocks (mamba.in_proj / mamba.out_proj), "
        "and every layer's shared-expert projections (shared_mlp.input_linear / "
        "shared_mlp.output_linear) and fused routed experts are unadapted too. "
        "Set training.lora.target_modules explicitly to include them."
    ),
}

#: A decoder-layer index in a module key: ``model.layers.7.self_attn`` -> ``7``.
_DECODER_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def _self_attention_coverage(model: Any) -> tuple[int, int] | None:
    """``(layers defining a self_attn, decoder layers)``, or ``None`` if unknown.

    COUNTED off the model when the caller passed one, so the notice describes the
    checkpoint in hand rather than the one this table was measured on -- the
    granite-4.0 family ships several depths, and a hard-coded "4 of 40" would be
    wrong on all but one of them. ``stream_setup`` resolves from a bare config, so
    ``config.layer_types`` is the fallback; a model with neither gets the notice
    without a fraction rather than no notice.

    Only consulted for a ``model_type`` in :data:`PARTIAL_COVERAGE_NOTES`, i.e.
    for a text-only decoder, so ``layers.N`` is unambiguous here. A hybrid VL
    wrapper would need the tower prefix taken into account first.
    """
    try:
        named_modules = getattr(model, "named_modules", None)
        if callable(named_modules):
            total: set[str] = set()
            attentive: set[str] = set()
            for name, _module in named_modules():
                match = _DECODER_LAYER_RE.search(str(name))
                if match is None:
                    continue
                total.add(match.group(1))
                if str(name).endswith(".self_attn"):
                    attentive.add(match.group(1))
            if total:
                return len(attentive), len(total)
    except Exception:  # noqa: BLE001 -- a notice never breaks a training run
        logger.debug("self-attention coverage not countable", exc_info=True)

    config = getattr(model, "config", model)
    config = getattr(config, "text_config", None) or config
    layer_types = getattr(config, "layer_types", None)
    if isinstance(layer_types, (list, tuple)) and layer_types:
        return sum(1 for kind in layer_types if kind == "full_attention"), len(layer_types)
    return None


def _note_partial_coverage(model_type: Any, model: Any, console: Any) -> None:
    """Print the partial-coverage advisory for ``model_type``, if it has one."""
    note = PARTIAL_COVERAGE_NOTES.get(model_type)
    if note is None or console is None:
        return
    coverage = _self_attention_coverage(model)
    scope = (
        f"{coverage[0]} of {coverage[1]} decoder layers"
        if coverage is not None
        else "only the layers that define one"
    )
    try:
        console.print(
            f"[yellow]Partial LoRA coverage:[/yellow] {model_type} -- "
            f"target_modules: auto adapts the attention projections of {scope}; "
            f"{note}"
        )
    except Exception:  # noqa: BLE001 -- never crash on console issues.
        logger.debug("partial-coverage notice not printed", exc_info=True)


def _model_types(model: Any) -> set[Any]:
    """Return outer/text model types without importing Transformers."""
    config = getattr(model, "config", model)
    text_config = getattr(config, "text_config", None)
    return {
        getattr(config, "model_type", None),
        getattr(text_config, "model_type", None),
    }


def _peft_has_a_default_for(model_types: set[Any]) -> bool:
    """Does PEFT map any of these ``model_type`` values to LoRA targets itself?

    Asked rather than assumed, so an architecture PEFT knows is never refused
    here: that over-refusal is the mistake #1074's review caught one axis over.
    A PEFT too old or too new to expose the mapping is treated as "yes", which
    keeps the old delegate-and-let-PEFT-decide behaviour.

    Deliberately conservative rather than accurate. A mapped MoE architecture
    (``mixtral``) still fails to attach, because peft's MoE conversion empties
    ``target_modules`` before the default is consulted -- see the note on
    :data:`MOE_TEXT_LORA_TARGETS`. Answering "yes" there means Soup delegates and
    the user sees peft's error, exactly as before #1070, instead of Soup
    refusing an architecture it was never asked about.
    """
    try:
        from peft.utils.constants import (
            TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING as PEFT_DEFAULTS,
        )
    except Exception:  # noqa: BLE001 — PEFT layout is not a promise
        return True
    return any(PEFT_DEFAULTS.get(value) for value in model_types if value is not None)


class UnmappedTargets:
    """``target_modules: auto`` mapped to nothing Soup or peft knows (#1070).

    Returned by :func:`resolve_lora_target_modules` instead of raising, so the
    refusal is decided by :func:`build_lora_config` -- the last step every
    trainer takes before ``get_peft_model``. Deciding it in the resolver refused
    before a later step could supply targets: ``moe_lora`` replaces
    ``target_modules`` *after* the resolver at every MoE-wired call site, and
    ``target_parameters`` alone is enough for peft. Measured on a real
    ``qwen2_moe`` with ``moe_lora: true``: main attached 7 modules, and the
    raise-in-the-resolver version refused it (#1102 review, F1).

    Falsy, like the ``None`` it stands in for, so code that tests
    ``if target_modules`` treats it as "no modules" rather than as a list.
    """

    __slots__ = ("model_types",)

    def __init__(self, model_types: list[str]) -> None:
        self.model_types = model_types

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return f"UnmappedTargets({self.model_types!r})"


def resolve_lora_target_modules(model: Any, configured: Any, console: Any = None) -> Any:
    """Resolve ``target_modules: auto`` for models PEFT does not know yet.

    Existing architectures remain delegated to PEFT by returning ``None``.
    Explicit user targets are returned unchanged. Qwen3.5 uses a wrapper
    config (``qwen3_5``) around ``qwen3_5_text`` and its MoE counterpart, so
    inspect both configs without importing Transformers or PEFT at module load.
    Qwen4-Exp's causal-LM loader exposes ``qwen4_exp_text`` directly.

    An architecture neither Soup nor peft maps returns :class:`UnmappedTargets`
    rather than raising; :func:`build_lora_config` decides, once a ``moe_lora``
    override and ``target_parameters`` have had their chance (#1070).

    ``console`` is optional so the pure resolution stays callable without one.
    When it is supplied and the resolved entry covers only part of the decoder
    (:data:`PARTIAL_COVERAGE_NOTES`), the advisory is printed here -- this is the
    one place that knows WHICH entry was chosen.
    """
    if configured != "auto" and configured != ["auto"]:
        return configured

    model_types = _model_types(model)
    if model_types & {
        "qwen3_5",
        "qwen3_5_text",
        "qwen3_5_moe",
        "qwen3_5_moe_text",
    }:
        return list(QWEN35_TEXT_LORA_TARGETS)
    if "qwen4_exp_text" in model_types:
        return QWEN4_EXP_TEXT_LORA_TARGETS
    # The MoE table. The wrapper ``model_type`` is preferred over the text one
    # where both are present, because a vision-language wrapper names its
    # language tower in the module path and the text-only entry does not.
    for value in (getattr(getattr(model, "config", model), "model_type", None),):
        if value in MOE_TEXT_LORA_TARGETS:
            _note_partial_coverage(value, model, console)
            return _as_targets(MOE_TEXT_LORA_TARGETS[value])
    for value in sorted(str(v) for v in model_types if v is not None):
        if value in MOE_TEXT_LORA_TARGETS:
            _note_partial_coverage(value, model, console)
            return _as_targets(MOE_TEXT_LORA_TARGETS[value])

    named = sorted(value for value in model_types if isinstance(value, str))
    if not named or _peft_has_a_default_for(model_types):
        # No ``model_type`` string to name is not the same as an architecture we
        # know to be unmappable -- a config that does not declare one, or a test
        # double standing in for a model, is delegated exactly as before #1070.
        return None
    return UnmappedTargets(named)


def _as_targets(entry: Any) -> Any:
    """A tuple becomes a fresh list; a string is a PEFT regex and stays one."""
    return entry if isinstance(entry, str) else list(entry)


def resolve_lora_target_parameters(model: Any, configured: Any) -> Any:
    """Resolve opt-in raw-parameter LoRA targets for supported architectures.

    ``None`` and an empty list disable raw-parameter targeting. Explicit lists
    always win unchanged. ``auto`` fails closed on unknown architectures so a
    user cannot request expert adaptation and silently train only modules.
    """
    if configured is None or configured == []:
        return configured
    if configured != "auto":
        return configured
    if "qwen4_exp_text" in _model_types(model):
        return list(QWEN4_EXP_TEXT_LORA_TARGET_PARAMETERS)
    raise ValueError(
        "training.lora.target_parameters='auto' has no mapping for model_type="
        f"{sorted(str(value) for value in _model_types(model) if value is not None)!r}; "
        "provide an explicit parameter-name list or omit target_parameters"
    )


def build_lora_config_kwargs(
    lora_cfg: Any,
    *,
    target_modules: Any,
    target_parameters: Any,
    task_type: Any,
) -> dict[str, Any]:
    """Build the shared PEFT LoRA kwargs used by every trainer path."""
    kwargs = {
        "r": lora_cfg.r,
        "lora_alpha": lora_cfg.alpha,
        "lora_dropout": lora_cfg.dropout,
        "target_modules": target_modules,
        "target_parameters": target_parameters,
        "task_type": task_type,
        "bias": "none",
        "use_dora": lora_cfg.use_dora,
        "use_rslora": lora_cfg.use_rslora,
    }
    rank_pattern = lora_cfg.rank_pattern
    alpha_pattern = lora_cfg.alpha_pattern
    if rank_pattern:
        kwargs["rank_pattern"] = dict(rank_pattern)
    if alpha_pattern:
        kwargs["alpha_pattern"] = dict(alpha_pattern)
    # ``use_olora`` is the legacy spelling. Schema validation aligns it to
    # init_strategy='olora'; retain this defensive read so callers that build a
    # schema-like config double cannot silently lose the requested method.
    init_strategy = (
        "olora"
        if getattr(lora_cfg, "use_olora", False)
        else getattr(lora_cfg, "init_strategy", "random")
    )
    if init_strategy != "random":
        kwargs["init_lora_weights"] = init_strategy
    if init_strategy == "loftq":
        from soup_cli.utils.loftq_init import build_loftq_config

        kwargs["loftq_config"] = build_loftq_config(
            loftq_iter=lora_cfg.loftq_iter,
            loftq_bits=lora_cfg.loftq_bits,
        )
    return kwargs


def build_peft_config_spec(
    lora_cfg: Any,
    *,
    target_modules: Any,
    task_type: Any,
    target_parameters: Any = None,
) -> dict[str, Any]:
    """Return the PEFT class name and kwargs for the configured adapter.

    VeRA is a distinct PEFT tuner, not a LoRA option. Keeping this branch next
    to the shared LoRA kwargs is what makes every trainer consume the same
    method choice instead of silently constructing ordinary LoRA.
    """
    if getattr(lora_cfg, "use_vera", False):
        return {
            "peft_cls": "VeraConfig",
            "init_kwargs": {
                "r": lora_cfg.r,
                "target_modules": target_modules,
                "task_type": task_type,
                "vera_dropout": lora_cfg.dropout,
                "bias": "none",
            },
        }
    return {
        "peft_cls": "LoraConfig",
        "init_kwargs": build_lora_config_kwargs(
            lora_cfg,
            target_modules=target_modules,
            target_parameters=target_parameters,
            task_type=task_type,
        ),
    }


def _settle_unmapped(target_modules: Any, target_parameters: Any) -> Any:
    """Refuse an unmappable ``auto`` here, where the final targets are known.

    Reached only if no ``moe_lora`` override replaced the value. With
    ``target_parameters`` peft has something to attach to, so the modules half
    is simply empty. Without them, refuse by name -- a better message than
    peft's ``No target_modules passed``, and one that no longer points a dense
    model at a MoE-only table (#1102 review, F5).
    """
    if not isinstance(target_modules, UnmappedTargets):
        return target_modules
    if target_parameters:
        return None
    raise ValueError(
        "training.lora.target_modules='auto' has no mapping for model_type="
        f"{target_modules.model_types!r}: neither Soup's table nor peft's own "
        "defaults cover it, so there is nothing to attach a LoRA adapter to. "
        "Give an explicit training.lora.target_modules list -- the module "
        "names are in model.named_modules() -- or, for a Mixture-of-Experts "
        "model, set training.moe_lora: true. The architectures Soup maps are "
        "in utils/peft_wiring.py (#1070)."
    )


def build_lora_config(
    lora_cfg: Any,
    *,
    target_modules: Any,
    task_type: Any,
    target_parameters: Any = None,
) -> Any:
    """Build the configured PEFT adapter through the single shared path.

    Keeping the PEFT import inside this function preserves Soup's lazy-import
    boundary while ensuring every trainer consumes new shared LoRA fields such
    as ``rank_pattern`` and ``alpha_pattern`` automatically.
    """
    import peft

    target_modules = _settle_unmapped(target_modules, target_parameters)
    spec = build_peft_config_spec(
        lora_cfg,
        target_modules=target_modules,
        target_parameters=target_parameters,
        task_type=task_type,
    )
    config_cls = getattr(peft, spec["peft_cls"])
    return config_cls(**spec["init_kwargs"])


def apply_pre_lora_patches(model: Any, base: str) -> None:
    """Run pre-LoRA surgical patches (v0.39.0 Part D, multi-trainer in v0.40.6).

    Qwen4-Exp's compatibility patch is fail-closed because an unpatched legacy
    Torch forward cannot train. Gemma4's best-effort ``ClippableLinear`` ->
    ``nn.Linear`` swap remains gated by ``is_gemma4_model(base)``.
    """
    from soup_cli.utils.qwen4_compat import apply_qwen4_exp_scatter_compat

    apply_qwen4_exp_scatter_compat(model)

    from soup_cli.utils.peft_patches import apply_gemma4_clippable_patch, is_gemma4_model

    if not is_gemma4_model(base):
        return
    try:
        apply_gemma4_clippable_patch(model)
    except Exception as exc:  # noqa: BLE001 — best-effort patch, log + continue
        logger.debug("apply_gemma4_clippable_patch skipped: %s", exc)


def apply_post_lora_patches(model: Any) -> None:
    """Run post-LoRA surgical patches (v0.39.0 Part D, multi-trainer in v0.40.6).

    Currently: 3-D fused-MoE expert dropout strip. Architecture-detected via
    ``weight.ndim == 3`` inside the helper; safe to call unconditionally.
    """
    from soup_cli.utils.peft_patches import strip_lora_dropout_for_3d_experts

    try:
        strip_lora_dropout_for_3d_experts(model)
    except Exception as exc:  # noqa: BLE001 — best-effort patch, log + continue
        logger.debug("strip_lora_dropout_for_3d_experts skipped: %s", exc)


def attach_relora_callback(trainer: Any, tcfg: Any) -> bool:
    """Attach :class:`ReLoRACallback` when ``training.relora_steps`` is set.

    Returns ``True`` when a callback was attached, ``False`` otherwise.
    The schema-level cross-validator (``_validate_relora_supported_tasks``)
    already enforces the transformer-backend requirement, so this helper
    trusts the caller's task/backend.
    """
    relora_steps = getattr(tcfg, "relora_steps", None)
    # Use `is None` (not `not relora_steps`) so a schema-bypassing caller that
    # passes `relora_steps=0` surfaces as a loud `ReLoRAPolicy` ValueError
    # rather than a silent skip. Matches project policy (v0.34.0 / v0.39.0).
    if relora_steps is None:
        return False
    # Pydantic schema guarantees these fields exist on `TrainingConfig`. Read
    # them directly so a misnamed attr fails loudly with `AttributeError`.
    from soup_cli.utils.relora import ReLoRACallback, ReLoRAPolicy

    policy = ReLoRAPolicy(
        steps=int(relora_steps),
        warmup_ratio=float(tcfg.relora_warmup_ratio),
        reset_optimizer=bool(tcfg.relora_reset_optimizer),
        prune_ratio=float(tcfg.relora_prune_ratio),
    )
    trainer.add_callback(ReLoRACallback(policy=policy))
    return True


def build_loraplus_optimizer(model: Any, args: Any, tcfg: Any) -> Any:
    """Build a PEFT LoRA+ optimizer from a model + ``TrainingArguments``, or ``None``.

    Returns ``None`` when ``training.loraplus_lr_ratio`` is unset, so a caller can
    treat "no LoRA+" and "a LoRA+ optimizer" uniformly. This is the shared core of
    the two wiring shapes LoRA+ needs:

    - :func:`attach_loraplus_optimizer` assigns the result onto an already-built
      ``trainer.optimizer`` — the SFT-family and preference/RL trainers, whose
      ``Trainer.create_optimizer`` runs lazily at ``train()`` and so respects a
      pre-assigned optimizer, building the scheduler around it.
    - The PPO wrapper hands it to the trainer *constructor* via
      ``optimizers=(optimizer, None)``, because ``trl.experimental``'s
      ``PPOTrainer`` builds its optimizer and scheduler eagerly inside
      ``__init__``; a post-construction assignment there trains the B group at
      ``lr * ratio`` but leaves ``self.lr_scheduler`` bound to the discarded
      default optimizer (see ``trainer/ppo.py``).

    LoRA+ is not a ``TrainingArguments`` field — it belongs to PEFT's optimizer
    construction (``create_loraplus_optimizer``), which gives the LoRA B matrices
    a learning rate of ``lr * loraplus_lr_ratio`` while A stays at ``lr``.
    Forwarding it into ``TrainingArguments`` raised ``TypeError`` before the first
    step, so the advertised option always crashed (#724).

    The optimizer class and its betas/eps come from the run's configured optimizer
    via ``Trainer.get_optimizer_cls_and_kwargs``, so LoRA+ uses the same optimizer
    the user asked for; weight decay is applied through PEFT's own
    ``loraplus_weight_decay`` (the plain ``weight_decay`` kwarg is ignored by
    ``create_loraplus_optimizer``).
    """
    ratio = getattr(tcfg, "loraplus_lr_ratio", None)
    if ratio is None:
        return None

    # GaLore projects full-parameter gradients; LoRA+ tunes LoRA A/B matrices.
    # They cannot both own the optimizer — fail loudly rather than let this
    # silently override the GaLore optimizer set on TrainingArguments.
    if getattr(tcfg, "use_galore", False):
        raise ValueError(
            "training.loraplus_lr_ratio is mutually exclusive with "
            "training.use_galore: LoRA+ tunes LoRA A/B matrices while GaLore "
            "projects full-parameter gradients. Enable one, not both."
        )
    if getattr(tcfg, "use_lorafa", False):
        raise ValueError(
            "training.loraplus_lr_ratio is mutually exclusive with "
            "training.use_lorafa: LoRA+ tunes LoRA A/B matrices while LoRA-FA "
            "freezes LoRA A matrices. Enable one, not both."
        )

    from peft import PeftModel
    from peft.optimizers import create_loraplus_optimizer
    from transformers import Trainer

    if not isinstance(model, PeftModel):
        raise ValueError(
            "training.loraplus_lr_ratio requires a LoRA (PEFT) model, but the "
            "active run has no adapter. Add a lora config or remove "
            "loraplus_lr_ratio."
        )

    optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(args)
    # create_loraplus_optimizer takes lr explicitly and re-inserts it into the
    # per-group kwargs itself; drop the duplicate so it is not passed twice.
    optimizer_kwargs.pop("lr", None)
    return create_loraplus_optimizer(
        model=model,
        optimizer_cls=optimizer_cls,
        lr=args.learning_rate,
        loraplus_lr_ratio=float(ratio),
        loraplus_weight_decay=args.weight_decay,
        **optimizer_kwargs,
    )


def add_value_model_param_groups(optimizer: Any, value_model: Any, args: Any) -> int:
    """Add a PPO value model's trainable parameters to a LoRA+ optimizer.

    trl's ``PPOTrainer`` builds its default optimizer over
    ``PolicyAndValueWrapper(policy, value_model)``, so the critic trains with
    the policy. A LoRA+ optimizer injected through ``optimizers=`` is built over
    the PEFT policy alone, so without this the critic's parameters belong to no
    optimizer and its value head never leaves its random initialisation (#745).

    The critic gets what ``Trainer.create_optimizer`` would have given it: a
    decay and a no-decay group at ``args.learning_rate``, split by the installed
    transformers' own ``get_decay_parameter_names`` (which does not use
    ``self``). Parameters the optimizer already holds are skipped, so a critic
    that shares tensors with the policy is not added twice. Empty groups are
    not added.

    Returns how many tensors were added.
    """
    from transformers import Trainer

    held = {id(p) for group in optimizer.param_groups for p in group["params"]}
    decay_names = set(Trainer.get_decay_parameter_names(None, value_model))
    decay, no_decay = [], []
    for name, param in value_model.named_parameters():
        if not param.requires_grad or id(param) in held:
            continue
        (decay if name in decay_names else no_decay).append(param)
    for params, weight_decay in ((decay, args.weight_decay), (no_decay, 0.0)):
        if params:
            optimizer.add_param_group(
                {
                    "params": params,
                    "lr": args.learning_rate,
                    "weight_decay": weight_decay,
                }
            )
    return len(decay) + len(no_decay)


def attach_loraplus_optimizer(trainer: Any, tcfg: Any) -> bool:
    """Attach a PEFT LoRA+ optimizer when ``training.loraplus_lr_ratio`` is set.

    Assigning ``trainer.optimizer`` here is respected because
    ``Trainer.create_optimizer`` builds one only when ``self.optimizer is None``,
    and the scheduler is still built from it with the configured warmup/schedule.
    That ordering holds for every trainer whose optimizer is created lazily at
    ``train()`` time; PPO is the one exception (its ``trl.experimental`` trainer
    builds the optimizer and scheduler eagerly in ``__init__``), and it uses
    :func:`build_loraplus_optimizer` with constructor injection instead — see
    ``trainer/ppo.py``.

    Returns ``True`` when an optimizer was attached, ``False`` otherwise.
    """
    optimizer = build_loraplus_optimizer(trainer.model, trainer.args, tcfg)
    if optimizer is None:
        return False
    trainer.optimizer = optimizer
    return True


def attach_lorafa_optimizer(trainer: Any, tcfg: Any) -> bool:
    """Attach a PEFT LoRA-FA optimizer when ``training.use_lorafa`` is set.

    LoRA-FA (Frozen-A LoRA, arXiv:2308.03303) freezes the LoRA A matrices and
    only updates the B matrices (#725). Freezing A eliminates the need to retain
    input activations for backpropagating through A, cutting adapter-rank
    activation memory retention substantially.

    Like LoRA+, LoRA-FA is not a ``TrainingArguments`` field — it belongs to
    PEFT's optimizer construction (``create_lorafa_optimizer``). Assigning
    ``trainer.optimizer`` post-construction is respected because
    ``Trainer.create_optimizer`` builds one only when ``self.optimizer is None``,
    and the scheduler is still derived from it with the configured warmup/schedule.

    Weight decay is passed directly through PEFT's ``weight_decay`` argument.
    The optimizer uses the learning rate from ``trainer.args.learning_rate``, and
    betas/eps from ``Trainer.get_optimizer_cls_and_kwargs`` are preserved.
    Conflicting configurations (GaLore, LoRA+, or a non-LoRA run) are rejected
    with explicit error messages.

    Returns ``True`` when an optimizer was attached, ``False`` otherwise.
    """
    if not getattr(tcfg, "use_lorafa", False):
        return False

    # GaLore projects full-parameter gradients; LoRA-FA tunes LoRA B matrices.
    # They cannot both own the optimizer — fail loudly rather than let this
    # silently override the GaLore optimizer set on TrainingArguments.
    if getattr(tcfg, "use_galore", False):
        raise ValueError(
            "training.use_lorafa is mutually exclusive with "
            "training.use_galore: LoRA-FA tunes LoRA B matrices while GaLore "
            "projects full-parameter gradients. Enable one, not both."
        )

    # LoRA+ provides separate learning rates for A and B; LoRA-FA freezes A.
    # They cannot be combined on the same run.
    if getattr(tcfg, "loraplus_lr_ratio", None) is not None:
        raise ValueError(
            "training.use_lorafa is mutually exclusive with "
            "training.loraplus_lr_ratio: LoRA-FA freezes LoRA A matrices while "
            "LoRA+ tunes them with separate learning rates. Enable one, not both."
        )

    # VeRA trains scaling vectors, not LoRA B matrices; create_lorafa_optimizer
    # finds no trainable lora_* parameters and silently degrades to plain AdamW.
    if getattr(getattr(tcfg, "lora", None), "use_vera", False):
        raise ValueError(
            "training.use_lorafa is mutually exclusive with training.lora.use_vera: "
            "VeRA freezes random projection matrices and trains scaling vectors, "
            "so peft's create_lorafa_optimizer finds no trainable lora_* matrices "
            "and silently degrades to plain AdamW."
        )

    # LoRA-FA optimizes gradients using an AdamW projection in LoraFAOptimizer.
    # An explicitly configured non-AdamW optimizer would be silently overridden.
    opt_name = getattr(tcfg, "optimizer", None)
    if opt_name is not None and opt_name not in (
        "adamw_torch",
        "adamw",
        "adamw_hf",
        "adamw_torch_fused",
    ):
        raise ValueError(
            f"training.use_lorafa uses an AdamW-based gradient projection and is "
            f"incompatible with training.optimizer={opt_name!r}. Leave optimizer unset "
            f"(defaulting to adamw) or use 'adamw_torch'."
        )

    from peft import PeftModel
    from peft.optimizers import create_lorafa_optimizer
    from transformers import Trainer

    model = trainer.model
    if not isinstance(model, PeftModel):
        raise ValueError(
            "training.use_lorafa requires a LoRA (PEFT) model, but the "
            "active run has no adapter. Add a lora config or disable "
            "use_lorafa."
        )

    r = None
    lora_alpha = None
    if hasattr(model, "peft_config") and model.peft_config:
        active = getattr(model, "active_adapter", None)
        if isinstance(active, str) and active in model.peft_config:
            adapter_cfg = model.peft_config[active]
        else:
            adapter_cfg = next(iter(model.peft_config.values()))
        r = getattr(adapter_cfg, "r", None)
        lora_alpha = getattr(adapter_cfg, "lora_alpha", None)
    if r is None and hasattr(tcfg, "lora") and tcfg.lora is not None:
        r = getattr(tcfg.lora, "r", None)
        lora_alpha = getattr(tcfg.lora, "alpha", None)
    if r is None or lora_alpha is None:
        raise ValueError(
            "training.use_lorafa requires explicit lora rank and alpha. "
            "Configure training.lora.r and training.lora.alpha or ensure the "
            "PEFT model provides them."
        )

    _, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(trainer.args)
    optimizer = create_lorafa_optimizer(
        model=model,
        r=int(r),
        lora_alpha=int(lora_alpha),
        lr=trainer.args.learning_rate,
        weight_decay=trainer.args.weight_decay,
    )
    if "betas" in optimizer_kwargs:
        for group in optimizer.param_groups:
            group["betas"] = optimizer_kwargs["betas"]
    if "eps" in optimizer_kwargs:
        for group in optimizer.param_groups:
            group["eps"] = optimizer_kwargs["eps"]

    _fixup_lorafa_state_dict_devices(optimizer)
    trainer.optimizer = optimizer
    return True


def _fixup_lorafa_state_dict_devices(optimizer: Any) -> Any:
    """Ensure tensors in optimizer.state are cast to their parameter's device on load_state_dict.

    `LoraFAOptimizer` keys adapter states by string names rather than parameter
    references or integer IDs (e.g. 'base_model.model...lora'). When PyTorch's
    `torch.optim.Optimizer.load_state_dict` executes during checkpoint resume, its
    per-parameter device-casting loop checks `id_map` which only contains integer
    parameter IDs. As a result, states indexed by string name (such as `exp_avg_B`
    and `exp_avg_sq_B`) remain on the deserialized storage device (typically CPU),
    causing a device mismatch runtime error on GPU when `opt.step()` runs.

    This hook casts all string-keyed tensors in `optimizer.state` to the target
    parameter device whenever `load_state_dict` is called.
    """
    import torch

    def _cast_state_tensors(opt: Any) -> None:
        for group in opt.param_groups:
            params = group.get("params", [])
            names = group.get("names", [])
            param_list = []
            for p, n in zip(params, names):
                if "lora" in n:
                    param_list.append(p)
                    if len(param_list) == 2:
                        name = n[: n.find("lora")] + "lora"
                        target_device = param_list[1].device  # LoRA B parameter
                        if name in opt.state:
                            for k, v in list(opt.state[name].items()):
                                if isinstance(v, torch.Tensor) and v.device != target_device:
                                    opt.state[name][k] = v.to(target_device)
                        param_list = []
                else:
                    if n in opt.state:
                        for k, v in list(opt.state[n].items()):
                            if isinstance(v, torch.Tensor) and v.device != p.device:
                                opt.state[n][k] = v.to(p.device)

    optimizer.register_load_state_dict_post_hook(_cast_state_tensors)
    return optimizer


def apply_lisa_setup(model: Any, tcfg: Any, console: Any = None) -> bool:
    """Prepare ``model`` for LISA full fine-tuning (v0.71.34 #267, #307).

    Returns ``True`` when LISA is enabled (the caller must then SKIP its LoRA
    path entirely), ``False`` otherwise.

    The model is deliberately left FULLY trainable here: HF builds the
    optimizer before the first callback fires, so every decoder parameter has
    to be in a param group for :class:`~soup_cli.utils.lisa.LisaCallback` to be
    able to re-activate it later. The callback then flips ``requires_grad``
    each interval — frozen parameters produce no gradient and AdamW skips
    them. ``enable_input_require_grads`` keeps gradient checkpointing working
    without a LoRA adapter, exactly as the Spectrum branch does.

    Centralised (rather than inlined per trainer) for the same reason
    ``block_expansion.apply_block_expansion_if_configured`` is: the SFT and
    pretrain trainers must not drift on what "LISA is on" means.
    """
    if not getattr(tcfg, "lisa_enabled", False):
        return False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if console is not None:
        console.print(
            f"[green]LISA:[/] layerwise importance sampling "
            f"({tcfg.lisa_num_layers} layer(s) every "
            f"{tcfg.lisa_interval_steps} steps, LoRA off)"
        )
    return True


def attach_lisa_callback(trainer: Any, tcfg: Any) -> bool:
    """Attach :class:`LisaCallback` when ``training.lisa_enabled`` is set.

    Returns ``True`` when a callback was attached, ``False`` otherwise. The
    schema-level cross-validator (``_validate_lisa_compat``) already enforces
    the ``_LISA_SUPPORTED_TASKS`` / transformers / text / quantization=none
    gate and mutual exclusion, so this helper trusts the caller's task/backend
    (v0.71.34 #267; ``pretrain`` added in #307).
    """
    if not getattr(tcfg, "lisa_enabled", False):
        return False
    from soup_cli.utils.lisa import LisaCallback, LisaPolicy

    # Read schema fields directly (they are guaranteed to exist on
    # TrainingConfig) so a misnamed attr fails loudly — mirrors
    # attach_relora_callback. seed is a fixed 0 (LISA reproducibility does not
    # need a user knob today; add a schema field if that changes).
    policy = LisaPolicy(
        num_layers=int(tcfg.lisa_num_layers),
        interval_steps=int(tcfg.lisa_interval_steps),
        reset_optimizer=bool(tcfg.lisa_reset_optimizer),
        seed=0,
        train_embeddings=bool(tcfg.lisa_train_embeddings),
    )
    trainer.add_callback(LisaCallback(policy=policy))
    return True


def attach_curriculum_callback(
    trainer: Any,
    tcfg: Any,
    output_dir: str,
    console: Any = None,
) -> bool:
    """Attach :class:`DynamicCurriculumCallback` when ``curriculum_dynamic=true``.

    Returns ``True`` when attached, ``False`` otherwise. The schema-level
    cross-validator (``_validate_curriculum_dynamic_supported``) gates by
    backend / task, so this helper trusts the caller's config.

    Args:
        trainer: HF Trainer (or duck-typed equivalent with ``add_callback``).
        tcfg: ``SoupConfig.training`` model.
        output_dir: Directory under cwd to write
            ``curriculum_history.jsonl`` (the BETA history record).
        console: Optional Rich Console for the BETA advisory.
    """
    if not getattr(tcfg, "curriculum_dynamic", False):
        return False
    # Lazy import — the callback module touches transformers + torch.
    from soup_cli.monitoring.curriculum_callback import (
        DynamicCurriculumCallback,
    )
    from soup_cli.utils.curriculum_dynamic import DynamicCurriculumPolicy

    policy = DynamicCurriculumPolicy(
        num_buckets=int(tcfg.curriculum_buckets),
        recompute_every_n_steps=int(
            getattr(tcfg, "curriculum_dynamic_recompute_steps", 50) or 50
        ),
        floor=float(getattr(tcfg, "curriculum_dynamic_floor", 0.05) or 0.05),
        temperature=float(
            getattr(tcfg, "curriculum_dynamic_temperature", 1.0) or 1.0
        ),
    )
    # v0.71.5 #149 — thread curriculum_metric so the callback can bucket by
    # loss / perplexity percentile (round-robin fallback for `length`). Any
    # value that is not one of the three valid metrics (e.g. a missing field
    # or a test MagicMock) falls back to `length` so the callback always
    # constructs.
    curriculum_metric = getattr(tcfg, "curriculum_metric", "length")
    if curriculum_metric not in ("length", "perplexity", "loss"):
        curriculum_metric = "length"
    try:
        callback = DynamicCurriculumCallback(
            policy=policy,
            output_dir=output_dir,
            curriculum_metric=curriculum_metric,
        )
    except (TypeError, ValueError) as exc:
        logger.debug("attach_curriculum_callback rejected: %s", exc)
        return False
    trainer.add_callback(callback)
    if console is not None:
        try:
            console.print(
                "[yellow]BETA:[/yellow] dynamic curriculum callback attached "
                f"(buckets={policy.num_buckets}, recompute_every="
                f"{policy.recompute_every_n_steps})"
            )
        except Exception:  # noqa: BLE001 — never crash on console issues.
            pass
    return True


def attach_grpo_stability_callback(trainer: Any, tcfg: Any) -> bool:
    """Attach :class:`GRPOStabilityCallback` when any v0.50.0 Part D knob is set.

    Returns ``True`` when a callback was attached, ``False`` otherwise.
    Mirrors the v0.40.6 / v0.53.5 / v0.53.6 callback-attach pattern.

    The schema-level cross-validator already gates these fields to
    ``task='grpo'`` on non-mlx backends, so this helper trusts the caller.
    """
    stability_fields = (
        "ref_model_ema_alpha",
        "replay_buffer_size",
        "async_grpo_prefetch",
        "tis_threshold",
        "mask_truncated_completions",
        "defer_rerolling",
        "skip_zero_advantage",
        "off_policy_mask_threshold",
    )
    # `is None` policy (matches v0.40.6 review-fix policy on `attach_relora_callback`)
    has_any = False
    for field_name in stability_fields:
        val = getattr(tcfg, field_name, None)
        # bools count as set when True; numeric fields count when not None.
        if isinstance(val, bool):
            if val:
                has_any = True
                break
        elif val is not None:
            has_any = True
            break
    if not has_any:
        return False
    from soup_cli.monitoring.grpo_stability_callback import GRPOStabilityCallback

    try:
        callback = GRPOStabilityCallback(
            ref_model_ema_alpha=tcfg.ref_model_ema_alpha,
            replay_buffer_size=tcfg.replay_buffer_size,
            async_grpo_prefetch=bool(tcfg.async_grpo_prefetch),
            tis_threshold=tcfg.tis_threshold,
            mask_truncated_completions=bool(tcfg.mask_truncated_completions),
            defer_rerolling=bool(tcfg.defer_rerolling),
            skip_zero_advantage=bool(tcfg.skip_zero_advantage),
            off_policy_mask_threshold=tcfg.off_policy_mask_threshold,
        )
    except (TypeError, ValueError) as exc:
        logger.debug("attach_grpo_stability_callback rejected: %s", exc)
        return False
    trainer.add_callback(callback)
    return True


def rl_callbacks_need_buffer(tcfg: Any) -> bool:
    """True when a reward-fn capture buffer is needed (v0.71.11 #235/#240).

    The reward-hack + echo-trap callbacks observe the GRPO step's rewards
    + completions through the shared
    :class:`~soup_cli.utils.rl_signal_buffer.RLSignalBuffer`. The
    RL-checkpoint callback does not.
    """
    return (
        getattr(tcfg, "reward_hack_detector", None) is not None
        or bool(getattr(tcfg, "echo_trap_enabled", False))
        or getattr(tcfg, "reward_hack_mitigation", "off") != "off"
    )


def _attach_reward_hack(
    trainer: Any,
    tcfg: Any,
    *,
    buffer: Any,
    tokenizer: Any,
    output_dir: str,
    task: str,
    rl_checkpoint_cb: Any = None,
) -> int:
    """Attach the reward-hack callback: mitigation controller (v0.71.26) when a
    ``reward_hack_mitigation`` mode is set, else the plain v0.70.0 detector.

    Returns 1 when a callback was attached, 0 otherwise. The mitigation
    controller SUBSUMES the plain detector (they share the same signal), so
    exactly one of the two is ever attached. ``rl_checkpoint_cb`` is the
    (already-built) RL-checkpoint callback the pid_lagrangian rollback ladder
    restores from.
    """
    import os

    detector = getattr(tcfg, "reward_hack_detector", None)
    mitigation = getattr(tcfg, "reward_hack_mitigation", "off")
    if mitigation != "off" and detector is not None:
        from soup_cli.utils.reward_hack_control import (
            BangBangPolicy,
            MitigationLogWriter,
            PIDLagrangianPolicy,
            RewardHackMitigationCallback,
        )

        try:
            writer = MitigationLogWriter(
                os.path.join(output_dir, "mitigation_log.jsonl")
            )
            signals = tuple(
                getattr(tcfg, "reward_hack_signals", None) or ("info_rm",)
            )
            bang_bang = None
            pid = None
            if mitigation == "kl_control":
                bang_bang = BangBangPolicy(
                    beta_floor=tcfg.reward_hack_beta_floor,
                    beta_ceil=tcfg.reward_hack_beta_ceil,
                    trip_band=tcfg.reward_hack_trip_band,
                    release_band=tcfg.reward_hack_release_band,
                    dwell_steps=tcfg.reward_hack_dwell_steps,
                    release_patience=tcfg.reward_hack_release_patience,
                    kl_gain=tcfg.reward_hack_kl_gain,
                )
            elif mitigation == "pid_lagrangian":
                pid = PIDLagrangianPolicy(
                    kp=tcfg.reward_hack_pid_kp,
                    ki=tcfg.reward_hack_pid_ki,
                    kd=tcfg.reward_hack_pid_kd,
                    signal_target=tcfg.reward_hack_signal_target,
                    beta_floor=tcfg.reward_hack_beta_floor,
                    beta_ceil=tcfg.reward_hack_beta_ceil,
                    integral_clamp=tcfg.reward_hack_integral_clamp,
                )
            callback = RewardHackMitigationCallback(
                mode=mitigation,
                detector=detector,
                log_writer=writer,
                signals=signals,
                buffer=buffer,
                tokenizer=tokenizer,
                task=task,
                bang_bang=bang_bang,
                pid=pid,
                rollback=bool(getattr(tcfg, "reward_hack_rollback", False)),
                rollback_patience=int(
                    getattr(tcfg, "reward_hack_rollback_patience", 3)
                ),
                max_recovery_attempts=int(
                    getattr(tcfg, "reward_hack_max_recovery_attempts", 2)
                ),
                rl_checkpoint_cb=rl_checkpoint_cb,
                smoothing=getattr(tcfg, "reward_hack_signal_smoothing", "none"),
                smoothing_window=int(
                    getattr(tcfg, "reward_hack_smoothing_window", 8)
                ),
                conservative_on_disagreement=bool(
                    getattr(tcfg, "reward_hack_conservative_on_disagreement", False)
                ),
            )
            trainer.add_callback(callback)
            callback.attach(trainer)
            return 1
        except (TypeError, ValueError, OSError) as exc:
            # A user explicitly enabled mitigation — a silent drop would leave
            # them believing a safety controller is active when it is not.
            # Warn LOUDLY (e.g. output dir outside cwd fails the log writer).
            logger.warning(
                "reward-hack mitigation callback NOT attached (%s): %s. "
                "Training will proceed WITHOUT mitigation.",
                type(exc).__name__,
                exc,
            )
            return 0
    if detector is not None:
        from soup_cli.utils.reward_hacking import build_reward_hack_callback

        try:
            trainer.add_callback(
                build_reward_hack_callback(
                    detector=detector,
                    halt_on_hack=bool(getattr(tcfg, "reward_hack_halt", False)),
                    buffer=buffer,
                )
            )
            return 1
        except (TypeError, ValueError) as exc:
            logger.debug("attach reward-hack callback rejected: %s", exc)
            return 0
    return 0


def attach_rl_callbacks(
    trainer: Any,
    tcfg: Any,
    *,
    buffer: Any = None,
    tokenizer: Any = None,
    output_dir: str = ".",
    task: str = "grpo",
) -> int:
    """Attach the v0.71.11 live RL callbacks; return how many were attached.

    Wires (when their schema fields are set):
    - reward-hacking detector (#235) — reads ``buffer``.
    - echo-trap detector (#240) — reads ``buffer`` + ``tokenizer``.
    - mid-epoch RL checkpoint (#238) — saves under ``output_dir``.

    The schema cross-validators already gate these fields to RL tasks on
    non-mlx backends, so this helper trusts the caller's config.
    """
    attached = 0
    # Build the RL-checkpoint callback FIRST so the pid_lagrangian rollback
    # ladder can be handed a reference to restore from.
    ckpt_cb = _build_rl_checkpoint_cb(tcfg, output_dir=output_dir, task=task)
    if ckpt_cb is not None:
        trainer.add_callback(ckpt_cb)
        attached += 1

    attached += _attach_reward_hack(
        trainer,
        tcfg,
        buffer=buffer,
        tokenizer=tokenizer,
        output_dir=output_dir,
        task=task,
        rl_checkpoint_cb=ckpt_cb,
    )

    if bool(getattr(tcfg, "echo_trap_enabled", False)):
        from soup_cli.utils.echo_trap import build_echo_trap_callback

        try:
            trainer.add_callback(
                build_echo_trap_callback(
                    threshold=float(getattr(tcfg, "echo_trap_threshold", 0.6)),
                    halt_on_trap=bool(getattr(tcfg, "echo_trap_halt", False)),
                    tokenizer_aware=bool(
                        getattr(tcfg, "echo_trap_tokenizer_aware", False)
                    ),
                    buffer=buffer,
                    tokenizer=tokenizer,
                )
            )
            attached += 1
        except (TypeError, ValueError) as exc:
            logger.debug("attach echo-trap callback rejected: %s", exc)

    return attached


def _build_rl_checkpoint_cb(tcfg: Any, *, output_dir: str, task: str) -> Any:
    """Build the mid-epoch RL-checkpoint callback (or None if not configured)."""
    save_every = getattr(tcfg, "rl_checkpoint_save_every_steps", None)
    if save_every is None:
        return None
    from soup_cli.utils.rl_checkpoint import (
        RLCheckpointConfig,
        build_rl_checkpoint_callback,
    )

    try:
        ckpt_cfg = RLCheckpointConfig(
            save_every_steps=int(save_every),
            include_optimizer_state=bool(
                getattr(tcfg, "rl_checkpoint_include_optimizer", True)
            ),
            include_ref_model=bool(
                getattr(tcfg, "rl_checkpoint_include_ref_model", False)
            ),
            include_rollout_buffer=bool(
                getattr(tcfg, "rl_checkpoint_include_rollout_buffer", False)
            ),
            keep_last=int(getattr(tcfg, "rl_checkpoint_keep_last", 3)),
        )
        return build_rl_checkpoint_callback(
            ckpt_cfg, output_dir=output_dir, task=task
        )
    except (TypeError, ValueError) as exc:
        logger.debug("build RL-checkpoint callback rejected: %s", exc)
        return None


def attach_plugin_callback(trainer: Any, console: Any = None) -> bool:
    """Attach :class:`SoupPluginCallback` when any enabled plugin implements a hook.

    Returns ``True`` when a callback was attached, ``False`` otherwise
    (no plugins enabled OR none implement any hook — the build helper
    short-circuits to ``None`` in that case so the trainer pays zero
    overhead).

    Failures inside individual plugin hooks are swallowed at WARNING
    inside the callback itself; this helper only handles the
    construction failure path (transformers not importable / plugin
    registry corrupted).
    """
    try:
        from soup_cli.monitoring.plugin_callback import build_plugin_callback

        callback = build_plugin_callback()
    except Exception as exc:  # noqa: BLE001 — plugin infra must not crash training
        logger.debug("attach_plugin_callback skipped: %s", exc)
        return False
    if callback is None:
        return False
    try:
        trainer.add_callback(callback)
    except Exception as exc:  # noqa: BLE001
        logger.debug("attach_plugin_callback add_callback failed: %s", exc)
        return False
    if console is not None:
        try:
            # Number of plugins is the count of distinct (plugin_name, hooks)
            # pairs the callback snapshot will fan out to.
            from soup_cli.plugins import list_plugins

            n_enabled = sum(1 for s in list_plugins().values() if s.enabled)
            console.print(
                f"[dim]Plugin callback attached ({n_enabled} enabled plugin(s)).[/]"
            )
        except Exception:  # noqa: BLE001
            pass
    return True


#: torch.compile wraps the module and every state-dict key gains this segment.
_COMPILE_PREFIX = "_orig_mod."


def strip_compile_prefix(output_dir: str) -> int:
    """Rewrite a saved adapter's keys to their canonical, loadable form.

    #335 — under ``training.use_fsdp2_compile`` the HF Trainer saves through the
    ``torch.compile`` wrapper, so every key comes out as
    ``_orig_mod.base_model.model...`` instead of ``base_model.model...``. The
    tensors are genuinely trained, but ``PeftModel.from_pretrained`` matches none
    of them: it emits ``UserWarning: Found missing adapter keys`` and leaves
    ``lora_B`` at its zero initialisation, so the adapter is a no-op. Measured on
    4xH100: **0 of 96** non-zero against 96/96 for the paired non-compile run,
    reproduced 3/3, with the run exiting 0 throughout.

    Returns the number of keys rewritten — 0 when there was nothing to do, which
    is the ordinary case and must stay a no-op rather than a rewrite of every
    run's adapter file.
    """
    import os

    path = os.path.join(output_dir, "adapter_model.safetensors")
    if not os.path.isfile(path):
        # full fine-tuning writes no adapter; a completed run must not end in a
        # crash just because there is nothing here to normalise
        return 0

    import tempfile

    from safetensors.torch import load_file, save_file

    tensors = load_file(path)
    if not any(key.startswith(_COMPILE_PREFIX) for key in tensors):
        return 0

    rewritten = {}
    changed = 0
    for key, value in tensors.items():
        # clone: load_file MEMORY-MAPS the file, and writing over a live mapping
        # fails on Windows with `os error 1224` (the same trap adapter_fuse.py
        # documents for an in-place save_pretrained). Cloning detaches the
        # tensors from the mapping so it can be released before the write.
        target = key[len(_COMPILE_PREFIX):] if key.startswith(_COMPILE_PREFIX) else key
        if target != key:
            changed += 1
        rewritten[target] = value.clone()
    if len(rewritten) != len(tensors):
        raise ValueError(
            "stripping the torch.compile prefix would collide two adapter keys; "
            "the checkpoint carries both spellings of the same weight"
        )
    del tensors

    # atomic: a half-written adapter is worse than the prefixed one, which at
    # least still holds the trained numbers
    handle, tmp_path = tempfile.mkstemp(
        dir=os.path.dirname(path) or ".", suffix=".safetensors"
    )
    os.close(handle)
    try:
        save_file(rewritten, tmp_path)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    return changed


def _build_compile_prefix_callback_class() -> type:
    """Construct ``CompilePrefixCallback`` with transformers as its parent.

    Built inside a function rather than at module scope so importing this module
    stays free of transformers: every transformer-backend trainer imports it for
    the ``attach_*`` helpers above, and today that import pulls in neither
    transformers nor torch. Mirrors
    ``monitoring/plugin_callback._build_callback_class``.
    """
    from rich.markup import escape
    from transformers import TrainerCallback

    class CompilePrefixCallback(TrainerCallback):
        """Normalise each ``checkpoint-*`` adapter as the Trainer writes it.

        #351: :func:`strip_compile_prefix` ran once, on ``self._output_dir``
        after the final ``save_model``. The HF Trainer writes its periodic
        checkpoints through that SAME ``save_model``, with
        ``output_dir=<run>/checkpoint-N``, so they come out carrying the prefix
        exactly as the final save did and nothing ever repaired them. Measured
        at 70B on 8xH100: 320 canonical keys in the output root, 320 prefixed
        ones in ``checkpoint-100``.

        Resuming is the case that decides how bad that is.
        ``PeftModel.from_pretrained`` at least warns.
        ``Trainer._load_from_checkpoint`` calls ``model.load_adapter(...)`` and
        drops the return value, and ``load_adapter`` deliberately does not warn
        (it returns the missing keys in the load result instead), while the
        unexpected ``_orig_mod.`` keys are dropped by
        ``load_state_dict(strict=False)``. So a resumed run silently continues
        from a re-zeroed ``lora_B``, which is #335's failure with its one
        warning removed.

        Subclasses ``TrainerCallback`` so it inherits the no-op default for
        every other event: HF's ``CallbackHandler.call_event`` dispatches via
        ``getattr(cb, event)`` with no ``hasattr`` guard, so a duck-typed
        callback survives wiring and then dies on ``on_epoch_begin`` (#308).
        """

        def __init__(self, output_dir: str = "", console: Any = None) -> None:
            super().__init__()
            # Fallback only: under real training the run directory comes from
            # ``args.output_dir``. Mirrors HFPushCallback.
            self.output_dir = output_dir
            self.console = console

        def on_save(self, args, state, control, **kwargs) -> None:
            """Normalise the checkpoint ``_save_checkpoint`` has just written."""
            # ``save_model`` writes the adapter only where ``args.should_save``
            # is true, but this event is dispatched on every rank. Without the
            # guard all 8 ranks of the run this was measured on would rewrite
            # one file at once. Default True so a single-process run (and a
            # test driving the callback directly) still does the work.
            #
            # ``args.should_save`` rather than ``state.is_world_process_zero``
            # because it is the same condition that decided whether this rank
            # wrote the file at all: ``TrainingArguments.should_save`` is
            # ``local_process_index == 0`` under ``save_on_each_node`` and
            # ``process_index == 0`` otherwise. Under ``save_on_each_node`` the
            # two disagree, and every node but the first would then keep a
            # checkpoint it had written and never repaired.
            if not getattr(args, "should_save", True):
                return
            step = int(getattr(state, "global_step", 0) or 0)
            if step <= 0:
                return
            output_dir = getattr(args, "output_dir", None) or self.output_dir
            if not output_dir:
                return

            checkpoint = os.path.join(output_dir, f"checkpoint-{step}")
            try:
                renamed = strip_compile_prefix(checkpoint)
            except Exception as exc:  # noqa: BLE001
                # Never take a multi-hour run down over one checkpoint: the
                # rewrite is atomic, so the prefixed file is still there and
                # still holds the trained numbers. But say so loudly: the
                # checkpoint that was left alone is a dead adapter, and a dead
                # adapter nobody hears about is the entire defect.
                message = (
                    f"could not normalise {checkpoint}: {exc}. It keeps "
                    "torch.compile's key prefix, so it will load as an adapter "
                    "of zeros; the run directory's final save is unaffected."
                )
                logger.warning("CompilePrefixCallback: %s", message)
                # escape: the path and the exception text are both interpolated,
                # and an unescaped `[` in either would be eaten as Rich markup.
                self._print(f"[yellow]{escape(message)}[/]")
                return
            if renamed:
                self._print(
                    f"[dim]Normalised {renamed} adapter keys in "
                    f"checkpoint-{step} saved through torch.compile's wrapper[/]"
                )

        def _print(self, message: str) -> None:
            if self.console is None:
                return
            try:
                self.console.print(message)
            except Exception:  # noqa: BLE001 (never crash on console issues)
                pass

    return CompilePrefixCallback


def build_compile_prefix_callback(output_dir: str = "", console: Any = None) -> Any:
    """Return a ``CompilePrefixCallback`` for ``output_dir``."""
    callback_cls = _build_compile_prefix_callback_class()
    return callback_cls(output_dir=output_dir, console=console)


def attach_compile_prefix_callback(
    trainer: Any,
    tcfg: Any,
    output_dir: str,
    console: Any = None,
) -> bool:
    """Attach :class:`CompilePrefixCallback` when ``use_fsdp2_compile`` is set.

    Returns ``True`` when attached, ``False`` otherwise. Gated on
    ``use_fsdp2_compile`` alone, matching the final-save call site in
    ``sft.py``: ``torch_compile`` is only really switched on when an FSDP config
    is present too (see :func:`utils.fsdp.apply_fsdp_training_kwargs`), but
    :func:`strip_compile_prefix` is a no-op on an adapter that has no prefix, so
    the narrower condition would buy nothing and let the two gates drift.

    Args:
        trainer: HF Trainer (or duck-typed equivalent with ``add_callback``).
        tcfg: ``SoupConfig.training`` model.
        output_dir: The run directory HF writes ``checkpoint-N`` under. Used
            only if ``TrainingArguments.output_dir`` is missing at save time.
        console: Optional Rich Console, for the same per-save note the final
            save prints.
    """
    if not getattr(tcfg, "use_fsdp2_compile", False):
        return False
    trainer.add_callback(
        build_compile_prefix_callback(output_dir=output_dir, console=console)
    )
    return True
