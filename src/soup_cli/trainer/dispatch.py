"""One task to trainer-wrapper dispatch, shared by ``soup train`` and ``soup sweep``.

The two commands used to carry their own ``if/elif`` chains over ``cfg.task``.
``soup sweep``'s copy was last extended in v0.40.0 and never grew the MLX route,
so for ten task values and for every ``backend: mlx`` config it built the SFT
wrapper while the run registry recorded the configured task (#1213). One
function, called from both, is what stops that from drifting again: the next
task the schema grows has to reach both commands or neither.

Imports stay inside the function. Every trainer wrapper and the MLX registry are
lazy-imported elsewhere in this package, and an MLX-only install must not pull
the transformers/TRL surface (``soup_cli.trainer.sft``) in at import time.
"""

from __future__ import annotations

from typing import Any


def build_trainer(cfg: Any, **trainer_kwargs: Any):
    """Build the trainer wrapper for ``cfg.task`` on ``cfg.backend``.

    ``trainer_kwargs`` are forwarded to the wrapper unchanged: ``soup train``
    passes ``device``, ``report_to``, ``deepspeed_config``, ``fsdp_config`` and
    ``trust_remote_code``; ``soup sweep`` passes ``device`` only.

    This module names ``deepspeed_config`` because it forwards the keyword, and
    it builds no trainer of its own, so it is exempt from the #359 guard scan
    (see ``_GUARD_EXEMPT`` in ``tests/test_issue359_deepspeed_guard_coverage.py``).

    A task with no wrapper raises ``ValueError`` naming the task. There is no
    SFT fallback on purpose: a task value the schema accepts but this function
    does not know has to fail in both commands, loudly, rather than train the
    wrong objective in one of them.
    """
    from soup_cli.trainer.mlx_routing import resolve_trainer

    mlx_cls, trainer_kwargs = resolve_trainer(cfg, trainer_kwargs)
    if mlx_cls is not None:
        return mlx_cls(cfg, **trainer_kwargs)

    task = getattr(cfg, "task", None)
    if task == "sft":
        # The transformers/TRL SFT surface stays behind this import so an
        # MLX-only install does not load it.
        from soup_cli.trainer.sft import SFTTrainerWrapper

        return SFTTrainerWrapper(cfg, **trainer_kwargs)
    if task == "dpo":
        from soup_cli.trainer.dpo import DPOTrainerWrapper

        return DPOTrainerWrapper(cfg, **trainer_kwargs)
    if task == "online_dpo":
        # v0.71.31 — on-policy generation judged by a pairwise judge or a
        # reward_model in the loop.
        from soup_cli.trainer.online_dpo import OnlineDPOTrainerWrapper

        return OnlineDPOTrainerWrapper(cfg, **trainer_kwargs)
    if task == "grpo":
        from soup_cli.trainer.grpo import GRPOTrainerWrapper

        return GRPOTrainerWrapper(cfg, **trainer_kwargs)
    if task == "ppo":
        from soup_cli.trainer.ppo import PPOTrainerWrapper

        return PPOTrainerWrapper(cfg, **trainer_kwargs)
    if task == "kto":
        from soup_cli.trainer.kto import KTOTrainerWrapper

        return KTOTrainerWrapper(cfg, **trainer_kwargs)
    if task == "orpo":
        from soup_cli.trainer.orpo import ORPOTrainerWrapper

        return ORPOTrainerWrapper(cfg, **trainer_kwargs)
    if task == "simpo":
        from soup_cli.trainer.simpo import SimPOTrainerWrapper

        return SimPOTrainerWrapper(cfg, **trainer_kwargs)
    if task == "ipo":
        from soup_cli.trainer.ipo import IPOTrainerWrapper

        return IPOTrainerWrapper(cfg, **trainer_kwargs)
    if task == "bco":
        from soup_cli.trainer.bco import BCOTrainerWrapper

        return BCOTrainerWrapper(cfg, **trainer_kwargs)
    if task == "preference":
        from soup_cli.trainer.preference import PreferenceTrainerWrapper

        return PreferenceTrainerWrapper(cfg, **trainer_kwargs)
    if task == "reward_model":
        from soup_cli.trainer.reward_model import RewardModelTrainerWrapper

        return RewardModelTrainerWrapper(cfg, **trainer_kwargs)
    if task == "pretrain":
        from soup_cli.trainer.pretrain import PretrainTrainerWrapper

        return PretrainTrainerWrapper(cfg, **trainer_kwargs)
    if task == "embedding":
        from soup_cli.trainer.embedding import EmbeddingTrainerWrapper

        return EmbeddingTrainerWrapper(cfg, **trainer_kwargs)
    if task == "distill":
        # v0.53.2 #133 — knowledge distillation (student + frozen teacher).
        from soup_cli.trainer.distill import DistillTrainerWrapper

        return DistillTrainerWrapper(cfg, **trainer_kwargs)
    if task == "prm":
        # v0.53.11 #126 — Process Reward Model trainer.
        from soup_cli.trainer.prm import PRMTrainerWrapper

        return PRMTrainerWrapper(cfg, **trainer_kwargs)
    if task in ("classifier", "reranker", "cross_encoder"):
        # v0.53.2 #132 — sequence-classification head.
        from soup_cli.trainer.classifier import ClassifierTrainerWrapper

        return ClassifierTrainerWrapper(cfg, **trainer_kwargs)
    if task == "unlearn":
        # v0.71.9 #193 — NPO / SimNPO / RMU unlearning.
        from soup_cli.trainer.unlearn import UnlearnTrainerWrapper

        return UnlearnTrainerWrapper(cfg, **trainer_kwargs)
    if task == "moe_lora_routing":
        # v0.71.12 #222 — MoLE per-token routing over N frozen task LoRAs.
        from soup_cli.trainer.mole_routing import MoleRoutingTrainerWrapper

        return MoleRoutingTrainerWrapper(cfg, **trainer_kwargs)
    if task == "tts":
        # v0.71.20 #131 — TTS fine-tuning (SFT-style next-token CE over text +
        # audio-codec-token sequences; per-family templating).
        from soup_cli.trainer.tts import TTSTrainerWrapper

        return TTSTrainerWrapper(cfg, **trainer_kwargs)
    if task == "asr":
        # v0.71.32 — ASR (Whisper) fine-tuning via Seq2SeqTrainer.
        from soup_cli.trainer.asr import AsrTrainerWrapper

        return AsrTrainerWrapper(cfg, **trainer_kwargs)

    raise ValueError(
        f"no trainer wrapper for task={task!r} on backend="
        f"{getattr(cfg, 'backend', 'transformers')!r}. Add the task to "
        "soup_cli/trainer/dispatch.py::build_trainer."
    )
