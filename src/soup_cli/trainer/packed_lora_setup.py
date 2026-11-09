"""``training.packed_lora`` in the SFT trainer: the checks before the load, the hook after
``get_peft_model``, and the teardown when training ends.

The mixin is kept out of ``trainer/sft.py`` so that file only gains a base class and four call
sites. Nothing here imports torch, Triton or bitsandbytes at module level; the kernels are reached
through ``soup_cli.kernels.packed_lora.loader`` only for a run that switched them on.

The order inside ``SFTTrainerWrapper._setup_transformers``:

1. ``_begin_packed_lora`` (before the tokenizer and the model are downloaded): the machine and the
   model's config are checked, and a run the kernels cannot take stops here with the reasons, or
   trains with the stock step under ``on_unsupported: stock``.
2. ``_packed_lora_kbit_context`` around peft's k-bit preparation call: the embedding and the
   lm_head are left in the compute dtype instead of being copied to fp32 (a 4.4 GiB transient for
   a 7B model, which does not fit an 8 GB card next to the 4-bit layers).
3. ``_install_packed_lora`` (after the LoRA exists): the model is read, the start-up checks run, the
   memory plan is made, and the hook goes on the causal LM. A reason found only here (a model whose
   layers the kernels do not handle) is treated as in 1.
4. ``_close_packed_lora`` when training ends, or when set-up fails: the hook comes off, the pinned
   host memory is given back and every table is on the GPU again.

Gradient checkpointing is left as the user configured it: a packed step does not run the decoder
layers that checkpoint, so it never fires there, and a step that falls back to the stock forward
is checkpointed as the user asked.
"""

from __future__ import annotations

import contextlib
from typing import Any

from rich.console import Console

from soup_cli.utils.terminal import for_terminal

console = Console()


def packed_lora_enabled(tcfg: Any) -> bool:
    """Whether ``training.packed_lora`` is on."""
    block = getattr(tcfg, "packed_lora", None)
    return block is not None and bool(block.enabled)


def compute_dtype_of(causal_lm: Any) -> Any:
    """The compute dtype of the 4-bit layers of a Qwen2-shaped causal LM (the dtype its LoRA
    matmuls and the kernels run in), or ``None`` when its first projection is not a 4-bit layer."""
    projection = causal_lm.model.layers[0].self_attn.q_proj
    base = getattr(projection, "base_layer", projection)
    state = getattr(getattr(base, "weight", None), "quant_state", None)
    return getattr(state, "dtype", None)


class PackedLoraSetupMixin:
    """The ``training.packed_lora`` steps of ``SFTTrainerWrapper``."""

    #: Set by ``_begin_packed_lora`` once the run passed the checks before the load.
    _packed_lora_wanted = False
    #: The ``PackedRun`` and the hook of a run that is using the kernels.
    _packed_lora_run = None
    _packed_lora_hook = None

    def _packed_lora_say(self, message: str) -> None:
        console.print(f"[dim]\\[packed_lora] {for_terminal(message)}[/]")

    def _packed_lora_cannot(self, cfg: Any, reasons: list[str]) -> None:
        """A run the kernels cannot take: stop with the reasons, or say so and train stock."""
        text = " ".join(reasons)
        if cfg.training.packed_lora.on_unsupported == "stock":
            console.print(
                f"[yellow]Packed LoRA kernels not used:[/] {for_terminal(text)} "
                "Training with the stock step (training.packed_lora.on_unsupported: stock)."
            )
            return
        raise ValueError(
            f"{text} Set training.packed_lora.on_unsupported: stock to train with the stock step "
            "instead, or remove training.packed_lora."
        )

    # -- 1. before the load ---------------------------------------------------------------------
    def _begin_packed_lora(self, cfg: Any, tcfg: Any) -> None:
        """Check the machine and the model's config before anything is downloaded."""
        self._packed_lora_wanted = False
        if not packed_lora_enabled(tcfg):
            return
        from soup_cli.kernels.packed_lora import preflight as checks

        sharding = None
        if getattr(self, "fsdp_config", None):
            sharding = "FSDP"
        elif getattr(self, "deepspeed_config", None):
            sharding = "DeepSpeed"
        result = checks.preflight(
            cfg,
            checks.read_model_config(cfg.base, self._trust_remote_code),
            checks.read_machine(),
            sharding=sharding,
        )
        if result.problems:
            self._packed_lora_cannot(cfg, result.problems)
            return
        # Memory that is not free right now: `soup train` refuses it unless --allow-oom-attempt
        # (commands/train.py), so a run that got here was told; anyone else is warned.
        for problem in result.fit_problems:
            console.print(f"[yellow]Packed LoRA kernels:[/] {for_terminal(problem)}")
        for note in result.notes:
            self._packed_lora_say(note)
        if tcfg.gradient_checkpointing:
            self._packed_lora_say(
                "gradient checkpointing applies to stock steps only: a packed step keeps its "
                "saved activations in the stash"
            )
        self._packed_lora_wanted = True

    # -- 2. around peft's k-bit preparation -------------------------------------------------------
    def _packed_lora_kbit_context(self) -> Any:
        """A context for peft's k-bit preparation call: the tables stay in the compute
        dtype. A no-op for a run that does not use the kernels."""
        if not self._packed_lora_wanted:
            return contextlib.nullcontext()
        from soup_cli.kernels.packed_lora import loader

        try:
            runtime = loader.load()
        except loader.KernelsUnavailableError:
            return contextlib.nullcontext()  # _install_packed_lora reports it
        return runtime.packed_model.keep_tables_in_compute_dtype(self.model)

    # -- 3. after get_peft_model ------------------------------------------------------------------
    def _install_packed_lora(self, cfg: Any, tcfg: Any) -> None:
        """Prepare the packed run on the LoRA model and put the hook on its causal LM."""
        if not self._packed_lora_wanted:
            return
        from soup_cli.kernels.packed_lora import budget, loader
        from soup_cli.kernels.packed_lora.hook import FP32_EMBEDS_ATTR, PackedLoraHook
        from soup_cli.kernels.packed_lora.policy import Policy
        from soup_cli.utils.stream_loss_chunk import causal_lm_of

        causal = causal_lm_of(self.model)
        try:
            runtime = loader.load()
        except loader.KernelsUnavailableError as exc:
            self._packed_lora_cannot(cfg, [f"training.packed_lora cannot run here: {exc}."])
            self._packed_lora_hook = self._embedding_hook(causal, FP32_EMBEDS_ATTR)
            return
        run = runtime.PackedRun(Policy.from_config(cfg), log=self._packed_lora_say)
        dtype = compute_dtype_of(causal)
        if dtype is None:
            from soup_cli.utils.gpu import get_compute_dtype

            dtype = get_compute_dtype()
        tokens = tcfg.batch_size * cfg.data.max_length
        try:
            # The tables the plan keeps off the GPU are parked at the first step, after the trainer
            # and accelerate have seen the model on the GPU.
            run.prepare(causal, dtype, tokens, park=False)
        except (runtime.packed_model.UnsupportedError, budget.BudgetTooSmallError) as exc:
            run.close()
            self._packed_lora_cannot(cfg, [f"training.packed_lora cannot run this model: {exc}"])
            self._packed_lora_hook = self._embedding_hook(causal, FP32_EMBEDS_ATTR)
            return
        except BaseException:
            run.close()
            raise
        self._packed_lora_run = run
        self._packed_lora_hook = PackedLoraHook(causal, run)

    def _embedding_hook(self, causal: Any, attr: str) -> Any:
        """After a refusal under ``on_unsupported: stock``: the stock forward still needs the
        embedding as an fp32 table would give it if ``_begin_packed_lora`` kept the table in the
        compute dtype. ``None`` when nothing changed."""
        if not getattr(causal, attr, False):
            return None
        from soup_cli.kernels.packed_lora.hook import PackedLoraHook

        return PackedLoraHook(causal, None)

    # -- 4. the end -------------------------------------------------------------------------------
    def _close_packed_lora(self) -> None:
        """Take the hook off the model and give the run's memory back. Safe to call twice."""
        hook, run = self._packed_lora_hook, self._packed_lora_run
        self._packed_lora_hook = self._packed_lora_run = None
        if hook is None and run is None:
            return
        if hook is not None:
            hook.remove()
            from soup_cli.utils.stream_loss_chunk import HOOK_ATTR

            trainer = getattr(self, "trainer", None)
            if trainer is not None and getattr(trainer, HOOK_ATTR, None) is hook:
                setattr(trainer, HOOK_ATTR, None)
        if run is not None:
            reasons = "; ".join(f"{n}x {why}" for why, n in run.stock_reasons.items())
            self._packed_lora_say(
                f"steps: {run.steps_packed} packed, {run.steps_stock} stock"
                + (f" ({reasons})" if reasons else "")
            )
            run.close()
