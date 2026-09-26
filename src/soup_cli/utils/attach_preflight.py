"""Can a shipped config attach a LoRA adapter at all? (#1116)

A config can parse through the schema, name a repo that resolves on the Hub, and
still be untrainable: ``target_modules: auto`` resolves to nothing for an
architecture neither Soup nor peft maps, and ``get_peft_model`` raises. Nothing
checked that until now --- ``validate-recipes`` (#330) parses YAML,
``check_recipe_repo_ids`` (#677) resolves repo ids, ``soup doctor --config``
(#903) reads a declared table, and ``soup adapters audit`` (#763) runs after
training. #1070 (every shipped MoE recipe), #798's dropout refusal and #1074's
``templates/moe.yaml`` blocker were each found by a person reading code.

The check builds the architecture on the **meta device** from ``config.json``
alone --- no weights, no download of a checkpoint, no GPU --- and then runs
Soup's own target resolution and the real ``get_peft_model``. The verdict is
therefore the trainer's verdict rather than a re-implementation of it.

Two disciplines carried over from #677, because without them a guard becomes
noise and gets muted:

**A failure to load the architecture is not a failure to attach.** A gated repo,
a repo needing ``trust_remote_code``, and a config this ``transformers`` cannot
read are all :data:`Verdict.UNVERIFIED` --- "could not check" --- never
:data:`Verdict.CANNOT_ATTACH`. Whether such a repo is missing is #677's
question, and answering it here in a second voice is how two guards disagree.

**Remote code is never executed.** ``trust_remote_code`` stays off, and a repo
that needs it reports as unverifiable. A preflight that runs arbitrary code from
the Hub to decide whether a recipe is healthy is a worse problem than the one it
solves.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

#: Layers are cut to this many before the model is built. Every module peft
#: matches on is per-layer and identically named in layer 0 and layer 40, so the
#: attach is unchanged --- and a 671B config becomes a skeleton that builds in
#: milliseconds.
PREFLIGHT_LAYERS = 2


class Verdict(enum.Enum):
    """Four outcomes, not two. Collapsing UNVERIFIED into CANNOT_ATTACH reports
    every gated repo as a broken recipe, which is how a guard gets muted."""

    ATTACHES = "attaches"
    NO_ADAPTER = "no_adapter"
    CANNOT_ATTACH = "cannot_attach"
    UNVERIFIED = "unverified"


@dataclass(frozen=True)
class AttachCheck:
    """One config's result. ``detail`` always says *why* for a non-ATTACHES
    verdict, because a bare verdict sends the reader back to the code."""

    name: str
    base: str
    task: str
    verdict: Verdict
    detail: str = ""
    model_type: Optional[str] = None
    adapted: int = 0
    expert_modules: int = 0
    vision_modules: int = 0
    targets: Any = None
    stage: str = ""


#: Substrings that mean "this config could not be loaded", mapped to the reason.
#: Matched against the exception text only (``transformers`` raises plain
#: ``ValueError``/``OSError`` for all of them), first match wins, so the ORDER is
#: the #677 lesson: "gated repo" goes first, ahead of "is not a local folder",
#: so a gated repo is never filed as an unresolvable id if one message ever
#: carries both.
_UNVERIFIABLE_MARKERS: tuple[tuple[str, str], ...] = (
    ("gated repo", "gated repo; set HF_TOKEN to check it"),
    ("trust_remote_code", "needs trust_remote_code, which this check never enables"),
    ("custom code", "needs trust_remote_code, which this check never enables"),
    ("Unrecognized model", "this transformers cannot read the config"),
    ("does not appear to have a file named config.json", "no config.json in the repo"),
    ("is not a local folder", "repo id does not resolve anonymously (see #677)"),
)


def classify_load_failure(exc: BaseException) -> str:
    """Why a base's config could not be loaded, in words, or "" if unrecognised.

    Kept separate from the check so the mapping is testable without a model, and
    so an unrecognised failure is visible as an empty string rather than being
    silently filed under the last branch.
    """
    for marker, reason in _UNVERIFIABLE_MARKERS:
        if marker in str(exc):
            return reason
    return ""


def shrink_for_preflight(hf_config: Any, layers: int = PREFLIGHT_LAYERS) -> Any:
    """Cut every sub-config's layer count, in place, and return the config.

    Sub-configs matter: a vision-language wrapper keeps the text tower's depth in
    ``text_config``, and leaving that at 61 builds 61 layers of a model whose
    module names repeat after the first.
    """
    targets = [hf_config]
    for attribute in ("text_config", "vision_config", "audio_config"):
        sub = getattr(hf_config, attribute, None)
        if sub is not None:
            targets.append(sub)
    for target in targets:
        current = getattr(target, "num_hidden_layers", None)
        if isinstance(current, int) and current > layers:
            target.num_hidden_layers = max(
                layers,
                _first_sparse_depth(target, current),
                _first_of_each_kind_depth(target, current),
            )
    return hf_config


def _first_of_each_kind_depth(config: Any, current: int) -> int:
    """Layers needed to include one layer of every kind in ``layer_types``.

    A hybrid decoder (Granite 4.0-H, Qwen3.5) puts attention at only some depths:
    two layers kept only Mamba or linear-attention layers, so ``q_proj`` was
    reported missing on a config the trainer attaches (#1117 review).
    """
    kinds = getattr(config, "layer_types", None)
    if not isinstance(kinds, (list, tuple)) or not kinds:
        return 0
    first: dict = {}
    for index, kind in enumerate(kinds[:current]):
        first.setdefault(kind, index)
    return max(first.values()) + 1


def _first_sparse_depth(config: Any, current: int) -> int:
    """Layers needed to include the first MoE layer, or 0 for a dense config.

    A shrink to two layers kept only dense layers on DeepSeek-V3 and GLM-5.1
    (``first_k_dense_replace: 3``), so their ``moe_lora`` recipes were reported as
    attaching with no expert ever built (found by a local CodeRabbit review).
    Each rule is the one the architecture's own modeling code applies.
    """
    first_dense = getattr(config, "first_k_dense_replace", None)
    moe_freq = getattr(config, "moe_layer_freq", None)
    sparse_step = getattr(config, "decoder_sparse_step", None)
    interleave = getattr(config, "interleave_moe_layer_step", None)
    mlp_only = getattr(config, "mlp_only_layers", None) or []
    fields = (first_dense, moe_freq, sparse_step, interleave)
    if not any(isinstance(v, int) and v > 0 for v in fields):
        return 0

    def is_sparse(i: int) -> bool:
        if i in mlp_only:
            return False
        if isinstance(first_dense, int) and i < first_dense:
            return False
        if isinstance(moe_freq, int) and moe_freq > 0 and i % moe_freq != 0:
            return False
        if isinstance(sparse_step, int) and sparse_step > 0 and (i + 1) % sparse_step != 0:
            return False
        if isinstance(interleave, int) and interleave > 0 and (i + 1) % interleave != 0:
            return False
        return True

    for i in range(current):
        if is_sparse(i):
            return i + 1
    return 0


def count_adapted(model: Any) -> tuple[int, int, int]:
    """``(adapted, expert modules, vision modules)`` for an attached model.

    The two breakdowns are what make the report say something a bare count
    cannot: whether a MoE config reached its experts, and whether a
    vision-language config quietly adapted its image encoder during a text
    fine-tune.
    """
    # ``.lora_A`` is the ModuleDict peft inserts once per adapted base module.
    # Counting ``lora_A.default`` as well double-counts every one of them, and
    # counting ONLY ``lora_A.default`` silently misses a non-default adapter
    # name -- peft names the inner Linear after the adapter.
    names = [name for name, _ in model.named_modules() if name.endswith(".lora_A")]
    experts = sum(1 for name in names if "expert" in name)
    vision = sum(1 for name in names if "vision" in name or "visual" in name)
    return len(names), experts, vision


_NO_ADAPTER_TASKS = {
    "prm": "the prm trainer full-fine-tunes and builds no adapter",
    "moe_lora_routing": "moe_lora_routing loads existing adapters and builds none",
}
_CLASSIFIER_TASKS = frozenset({"classifier", "reranker", "cross_encoder"})


def plan_adapter(cfg: Any) -> tuple[bool, str]:
    """``(has an adapter to attach, why not)``, without touching the network.

    ``lora.r: 0`` is full fine-tuning (#700) and MLX attaches through a different
    path entirely, so neither is a failure --- and reporting them as one would
    put 3 permanent red rows in a report meant to be all green.
    """
    if getattr(cfg, "backend", None) == "mlx":
        return False, "mlx backend attaches through a different path"
    tcfg = getattr(cfg, "training", None)
    lora = getattr(tcfg, "lora", None)
    if lora is None or not getattr(lora, "r", 0):
        return False, "lora.r: 0 --- full fine-tuning, no adapter"
    task = getattr(cfg, "task", "")
    # Trainers that never build an adapter, whatever lora.r says: answering for one
    # would report on an adapter the run does not have.
    if task in _NO_ADAPTER_TASKS:
        return False, _NO_ADAPTER_TASKS[task]
    if task in _CLASSIFIER_TASKS and not getattr(tcfg, "classifier_lora", False):
        return False, "classifier_lora is off --- that trainer full-fine-tunes"
    if task == "asr" and not getattr(tcfg, "asr_lora", False):
        # trainer/asr.py attaches only with asr_lora on (and r > 0, checked above).
        return False, "asr_lora is off --- that trainer full-fine-tunes"
    return True, ""


def check_attach(
    name: str,
    cfg: Any,
    *,
    load_hf_config: Callable[[str], Any],
    build_model: Callable[[Any, str], Any],
    attach: Optional[Callable[[Any, Any], Any]] = None,
    console: Any = None,
) -> AttachCheck:
    """Run one config through resolution and attach; never raise.

    ``load_hf_config`` and ``build_model`` are injected so the decision logic is
    testable without the Hub. The target resolution is deliberately NOT
    injectable --- it is :func:`trainer_lora_config`, the trainer's own sequence
    --- and ``attach`` defaults to the real ``get_peft_model``: stubbing either
    would make this check assert its own beliefs about peft and the trainer
    rather than their behaviour, which is the #826 mistake.
    """
    base = getattr(cfg, "base", "")
    task = getattr(cfg, "task", "")
    row = dict(name=name, base=base, task=task)

    wanted, why_not = plan_adapter(cfg)
    if not wanted:
        return AttachCheck(**row, verdict=Verdict.NO_ADAPTER, detail=why_not)

    try:
        hf_config = load_hf_config(base)
    except Exception as exc:  # noqa: BLE001 --- every loader failure is a verdict
        reason = classify_load_failure(exc) or f"{type(exc).__name__}: {exc}"
        return AttachCheck(
            **row, verdict=Verdict.UNVERIFIED, detail=reason, stage="config"
        )

    model_type = getattr(hf_config, "model_type", None)
    row["model_type"] = model_type
    try:
        model = build_model(shrink_for_preflight(hf_config), loader_for(cfg))
    except Exception as exc:  # noqa: BLE001
        reason = classify_load_failure(exc)
        return AttachCheck(
            **row,
            verdict=Verdict.UNVERIFIED if reason else Verdict.CANNOT_ATTACH,
            detail=reason or f"{type(exc).__name__}: {exc}",
            stage="build",
        )

    try:
        peft_config, targets = trainer_lora_config(model, cfg, console)
    except Exception as exc:  # noqa: BLE001 --- Soup's own refusals land here
        return AttachCheck(
            **row,
            verdict=Verdict.CANNOT_ATTACH,
            detail=f"{type(exc).__name__}: {exc}",
            stage="resolve",
        )

    if attach is None:
        attach = _attach_with_peft
    try:
        attached = attach(model, peft_config)
    except Exception as exc:  # noqa: BLE001
        return AttachCheck(
            **row,
            verdict=Verdict.CANNOT_ATTACH,
            detail=f"{type(exc).__name__}: {exc}",
            targets=targets,
            stage="attach",
        )

    adapted, experts, vision = count_adapted(attached)
    return AttachCheck(
        **row,
        verdict=Verdict.ATTACHES if adapted else Verdict.CANNOT_ATTACH,
        detail="" if adapted else "peft attached, but no module carries an adapter",
        adapted=adapted,
        expert_modules=experts,
        vision_modules=vision,
        targets=targets,
        stage="attach",
    )


def trainer_lora_config(model: Any, cfg: Any, console: Any = None) -> tuple[Any, Any]:
    """The adapter config a LoRA trainer builds, in the order it builds it.

    ``resolve_lora_target_modules`` -> ``resolve_lora_target_parameters`` ->
    the ``moe_lora`` override -> ``build_lora_config``: the same four calls, in the
    same order, as ``trainer/sft.py`` and the other MoE-wired trainers. The first
    version of this check stopped after the first call and built its own
    ``LoraConfig``, so a recipe with ``moe_lora: true`` -- which the trainer
    rescues by replacing ``target_modules`` -- was reported unable to attach while
    the real trainer attached it. Measured on ``qwen3-30b-a3b-sft``: this check
    said FAILS, the trainer's path attached 12 modules.

    The config's OWN r / alpha / dropout go in unchanged: normalising them to a
    tidy ``dropout: 0.0`` would have hidden #798, whose dropout was the schema
    default the recipes inherited.

    ``task_type`` is left unset: a task head wants a loaded model
    (``prepare_inputs_for_generation`` and friends), and the adapter injection is
    what is under test, not the head.

    ``moe_lora`` is applied for every task that reaches this sequence: #1148 and
    #1179 wired every trainer that builds an adapter, and config load refuses it
    where none is built. ``asr`` takes its own path above. The one split by
    modality is SFT's: its vision and audio branches (``_setup_vision_transformers``,
    ``_setup_audio_transformers``) call only ``resolve_lora_target_modules`` and
    ``build_lora_config``, so for those the MoE and ``target_parameters`` steps are
    skipped here too -- the same split :func:`loader_for` makes for the class.
    ``classifier`` (with ``classifier_lora``), ``distill`` and ``unlearn`` run the
    resolver, the MoE step and ``build_lora_config`` but never
    ``resolve_lora_target_parameters``, which this sequence runs for them too. No
    shipped config sets ``lora.target_parameters``, so no verdict depends on the
    difference today.
    """
    from soup_cli.utils.moe import resolve_moe_lora_targets
    from soup_cli.utils.peft_wiring import (
        build_lora_config,
        resolve_lora_target_modules,
        resolve_lora_target_parameters,
    )

    tcfg = cfg.training
    lora = tcfg.lora
    if getattr(cfg, "task", "") == "asr":
        # trainer/asr.py never calls the resolver: auto means Whisper's q/v.
        targets = lora.target_modules
        if targets == "auto":
            targets = ["q_proj", "v_proj"]
        return build_lora_config(lora, target_modules=targets, task_type=None), targets
    targets = resolve_lora_target_modules(model, lora.target_modules, console)
    if getattr(cfg, "task", "") == "sft" and getattr(cfg, "modality", "text") in (
        "vision", "audio",
    ):
        # SFT's vision and audio branches make only these two calls: no MoE
        # rescue and no target_parameters. Report what that path attaches.
        peft_config = build_lora_config(lora, target_modules=targets, task_type=None)
        return peft_config, targets
    target_parameters = resolve_lora_target_parameters(
        model, getattr(lora, "target_parameters", None)
    )
    # No console: its "targeting N module patterns" line names no config, so in
    # catalogue mode it is noise on stderr (#1117 review). The verdict is the same.
    targets = resolve_moe_lora_targets(model, tcfg, targets, None)
    peft_config = build_lora_config(
        lora,
        target_modules=targets,
        task_type=None,
        target_parameters=target_parameters,
    )
    return peft_config, targets


def _attach_with_peft(model: Any, peft_config: Any) -> Any:
    """The real ``get_peft_model``, on the config the trainer would build."""
    from peft import get_peft_model

    return get_peft_model(model, peft_config)


@dataclass
class PreflightReport:
    """Every row, plus the one question CI asks: did anything fail to attach?"""

    checks: list[AttachCheck] = field(default_factory=list)

    def by_verdict(self, verdict: Verdict) -> list[AttachCheck]:
        return [check for check in self.checks if check.verdict is verdict]

    @property
    def failures(self) -> list[AttachCheck]:
        return self.by_verdict(Verdict.CANNOT_ATTACH)

    @property
    def exit_code(self) -> int:
        """``EXIT_GATE_FAILED`` (2) only for CANNOT_ATTACH, the gate/verdict
        contract. An unverifiable config is not a failure --- it is a config this
        machine could not answer for."""
        from soup_cli.utils.exit_codes import EXIT_GATE_FAILED, EXIT_OK

        return EXIT_GATE_FAILED if self.failures else EXIT_OK


#: The auto-class each trainer loads, read off the trainers rather than guessed.
#: SFT chooses by MODALITY, not task -- ``trainer/sft.py`` loads a causal LM for
#: text, ``AutoModelForImageTextToText`` for vision and ``AutoModel`` for audio --
#: so keying this on the task alone built every vision recipe as a causal LM.
_MODALITY_AUTO_CLASS = {
    "vision": ("AutoModelForImageTextToText",),
    "audio": ("AutoModel",),
}
_TASK_AUTO_CLASS = {
    "embedding": ("AutoModel",),
    "reward_model": ("AutoModelForSequenceClassification",),
    # trainer/classifier.py serves all three.
    "classifier": ("AutoModelForSequenceClassification",),
    "reranker": ("AutoModelForSequenceClassification",),
    "cross_encoder": ("AutoModelForSequenceClassification",),
    # trainer/asr.py loads the Whisper seq2seq class by name, not an auto-class.
    "asr": ("WhisperForConditionalGeneration",),
}
#: Only SFT reads ``modality`` (#1117 review): every other trainer loads its one
#: class whatever the config says, so a vision DPO config is a causal LM there.
_MODALITY_TASKS = frozenset({"sft"})
#: ONE class per path, with no fallback, because no trainer has one. My first
#: version fell back from ``AutoModelForCausalLM`` to ``AutoModel``, so a config
#: the causal-LM class refuses -- MiniMax-M3 with no ``modality``, which cannot load
#: in SFT at all (#1145) -- was built one level down and reported as attaching.
#: That is the failure the #1116 ruling named: a preflight instantiating at a
#: different level from the trainer reports green on a config that cannot attach.
_DEFAULT_AUTO_CLASS = ("AutoModelForCausalLM",)


def loader_for(cfg: Any) -> tuple[str, ...]:
    """The auto-classes to try, in the order the config's trainer would.

    A task with its own class wins. After that only SFT lets ``modality`` choose
    (``trainer/sft.py`` is the one trainer that reads it); every other task loads
    a causal LM, so answering by modality there reported a vision DPO config as
    attaching while DPO's own ``AutoModelForCausalLM`` refused it (#1117 review).
    """
    task = getattr(cfg, "task", "")
    if task in _TASK_AUTO_CLASS:
        return _TASK_AUTO_CLASS[task]
    if task not in _MODALITY_TASKS:
        return _DEFAULT_AUTO_CLASS
    return _MODALITY_AUTO_CLASS.get(getattr(cfg, "modality", "text"), _DEFAULT_AUTO_CLASS)


def load_hf_config(base: str) -> Any:
    """``config.json`` alone, anonymously, with remote code refused."""
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(base, trust_remote_code=False)


def build_on_meta(hf_config: Any, classes: tuple[str, ...]) -> Any:
    """Instantiate the architecture with no storage behind any parameter.

    ``classes`` comes from :func:`loader_for`, tried in order; the last error is
    raised if none works, so the report says which class refused rather than
    "could not build".
    """
    import torch
    import transformers

    last: Optional[BaseException] = None
    for class_name in classes:
        factory = getattr(transformers, class_name, None)
        if factory is None:
            continue
        try:
            with torch.device("meta"):
                # trust_remote_code=False explicitly. Omitted, transformers does
                # not refuse a custom-code architecture -- it PROMPTS on stdin
                # ("Do you wish to run the custom code? [y/N]"), which hangs a
                # terminal, corrupts --json on stdout, and runs remote code for
                # anyone who answers y. load_hf_config passing False is not
                # enough: a config can load while its modeling code is remote
                # (Kimi-K2.5 does exactly this).
                if hasattr(factory, "from_config"):
                    return factory.from_config(hf_config, trust_remote_code=False)
                # A concrete class (asr's WhisperForConditionalGeneration) has no
                # from_config and no remote code: the trainer constructs it by name.
                return factory(hf_config)
        except Exception as exc:  # noqa: BLE001 --- try the next class
            last = exc
    raise last if last is not None else RuntimeError("no model class available")
