"""Online DPO trainer — wraps ``trl.OnlineDPOTrainer`` (v0.71.31).

Unlike offline DPO (which reads static ``prompt/chosen/rejected`` rows), Online
DPO generates two completions per prompt ON-POLICY at each step and asks a
*judge* (an LLM judge) OR a *reward model* which is better — the winner becomes
``chosen``, the loser ``rejected``. The judge closes the loop.

Data is prompt-only (like GRPO): Soup's ``{"messages": [...]}`` rows are
normalized to the OnlineDPO ``prompt`` column (chat, minus the assistant turn).

**Cross-version adapter.** TRL changed the OnlineDPO API between 0.19.x and 1.x:

- **trl 0.19.x** — ``from trl import OnlineDPOTrainer``; the judge is a
  ``BasePairwiseJudge`` (swap-debiased *pairwise* comparison, via
  :func:`soup_cli.eval.judge.make_soup_pairwise_judge`); a reward model is
  passed as ``reward_model=`` / ``reward_processing_class=``.
- **trl 1.x** — pairwise judges were removed; ``OnlineDPOTrainer`` moved to
  ``trl.experimental.online_dpo`` and ranks completions by ``reward_funcs=``. The
  same Soup ``JudgeEvaluator`` is adapted to a *pointwise* reward function
  (:func:`soup_cli.eval.judge.make_judge_reward_func`) — the same pointwise judge
  ``soup data best-of-n`` uses. Reward models pass as ``reward_funcs=[rm]`` /
  ``reward_processing_classes=[tok]``.

``_ONLINE_DPO_JUDGE_OVERRIDE`` is a test seam for injecting a synthetic Soup
evaluator (has ``.compare_pair`` + ``.evaluate``); it is adapted to whichever
API the installed trl exposes.
"""

import time
from pathlib import Path
from typing import Optional

from rich.console import Console

from soup_cli.config.schema import SoupConfig
from soup_cli.utils.gpu import estimate_batch_size, model_size_from_name

console = Console()

# Test seam: when set, replaces the URL-built judge (used by the offline
# synthetic-judge smoke). A Soup evaluator (``.compare_pair`` + ``.evaluate``).
_ONLINE_DPO_JUDGE_OVERRIDE = None


def _import_online_dpo():
    """Import ``OnlineDPOConfig``/``OnlineDPOTrainer`` across trl versions.

    trl 0.19.x exposes them at the top level; trl 1.x moved them to
    ``trl.experimental.online_dpo``.
    """
    try:
        from trl import OnlineDPOConfig, OnlineDPOTrainer

        return OnlineDPOConfig, OnlineDPOTrainer
    except ImportError:
        pass
    try:
        from trl.experimental.online_dpo import OnlineDPOConfig, OnlineDPOTrainer

        return OnlineDPOConfig, OnlineDPOTrainer
    except ImportError as exc:  # pragma: no cover — trl ships in [train]
        raise ImportError(
            "task='online_dpo' requires trl with OnlineDPO support "
            "(pip install 'soup-cli[train]')"
        ) from exc


def _trl_has_judges() -> bool:
    """True on trl 0.19.x (pairwise ``BasePairwiseJudge`` API), False on trl 1.x."""
    try:
        from trl import BasePairwiseJudge  # noqa: F401

        return True
    except ImportError:
        return False

# Fallback chat template for base models that ship none. Unlike the shared
# ``constants.DEFAULT_CHAT_TEMPLATE``, this one emits an assistant generation
# cue on ``add_generation_prompt`` so on-policy generation continues the
# model's turn (TRL always renders prompts with ``add_generation_prompt=True``).
_FALLBACK_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "{{ message['content'] + '\n' }}"
    "{% elif message['role'] == 'user' %}"
    "{{ 'User: ' + message['content'] + '\n' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ 'Assistant: ' + message['content'] + '\n' }}"
    "{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}{{ 'Assistant: ' }}{% endif %}"
)


class OnlineDPOTrainerWrapper:
    """High-level wrapper for Online DPO training from SoupConfig."""

    def __init__(
        self,
        config: SoupConfig,
        device: str = "cuda",
        report_to: str = "none",
        deepspeed_config: Optional[str] = None,
        fsdp_config: Optional[dict] = None,
        trust_remote_code: bool = False,
    ):
        self.config = config
        self.device = device
        self.report_to = report_to
        self.deepspeed_config = deepspeed_config
        self.fsdp_config = fsdp_config
        self.trust_remote_code = trust_remote_code
        self.model = None
        self.tokenizer = None
        self.peft_config = None
        self.trainer = None
        from soup_cli.utils.trust_remote import (
            model_requires_trust_remote_code,
            resolve_trust_remote_code,
        )

        requires = model_requires_trust_remote_code(config.base) or False
        self._trust_remote_code = resolve_trust_remote_code(
            config.base,
            requested=trust_remote_code,
            console=console,
            requires_remote_code=requires,
        )

    @staticmethod
    def _to_prompt_rows(rows: list[dict]) -> list[dict]:
        """Normalize Soup rows -> OnlineDPO prompt-only rows (chat ``prompt``).

        - ``{"prompt": "text"}`` -> a single user turn.
        - ``{"messages": [...]}`` -> the conversation up to and INCLUDING the
          last user turn (interleaved assistant turns are KEPT so multi-turn
          context and user/assistant alternation are preserved); the trailing
          assistant turn is dropped because the model generates it on-policy.
        """
        out = []
        for row in rows:
            if isinstance(row.get("prompt"), str):
                out.append({"prompt": [{"role": "user", "content": row["prompt"]}]})
            elif isinstance(row.get("messages"), list):
                msgs = row["messages"]
                last_user = -1
                for idx, msg in enumerate(msgs):
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        last_user = idx
                if last_user >= 0:
                    prompt_msgs = [
                        m
                        for m in msgs[: last_user + 1]
                        if isinstance(m, dict)
                        and m.get("role") in ("system", "user", "assistant")
                    ]
                    out.append({"prompt": prompt_msgs})
        return out

    def setup(self, dataset: dict):
        """Load model, tokenizer, build the OnlineDPO trainer (judge in loop)."""
        from datasets import Dataset

        OnlineDPOConfig, OnlineDPOTrainer = _import_online_dpo()  # noqa: N806 (classes)

        from soup_cli.trainer.sft import _enable_hf_transfer_progress

        _enable_hf_transfer_progress()

        cfg = self.config
        tcfg = cfg.training

        self._setup_transformers(cfg, tcfg)

        # --- Batch size (online DPO generates -> ~2x memory per sample) ---
        batch_size = tcfg.batch_size
        if batch_size == "auto":
            from soup_cli.utils.gpu import get_gpu_info

            gpu_info = get_gpu_info()
            model_size = model_size_from_name(cfg.base)
            batch_size = estimate_batch_size(
                model_params_b=model_size,
                seq_length=cfg.data.max_length,
                gpu_memory_bytes=gpu_info["memory_total_bytes"],
                quantization=tcfg.quantization,
                lora_r=tcfg.lora.r,
            )
            batch_size = max(1, batch_size // 2)
            console.print(f"[green]Auto batch size (online DPO):[/] {batch_size}")

        # --- Dataset (prompt-only) ---
        prompt_rows = self._to_prompt_rows(dataset["train"])
        if not prompt_rows:
            raise ValueError(
                "online_dpo: no usable prompts (rows need a 'prompt' string or "
                "'messages' with a user/system turn)"
            )
        train_ds = Dataset.from_list(prompt_rows)

        # --- Output dir ---
        output_dir = Path(cfg.output)
        if cfg.experiment_name:
            output_dir = output_dir / cfg.experiment_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # --- Warmup from ratio ---
        import math

        total_steps = (
            math.ceil(len(train_ds) / batch_size / tcfg.gradient_accumulation_steps)
            * tcfg.epochs
        )
        warmup_steps = int(total_steps * tcfg.warmup_ratio)

        odpo_config = OnlineDPOConfig(
            output_dir=str(output_dir),
            num_train_epochs=tcfg.epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=tcfg.gradient_accumulation_steps,
            learning_rate=tcfg.lr,
            warmup_steps=warmup_steps,
            weight_decay=tcfg.weight_decay,
            max_grad_norm=tcfg.max_grad_norm,
            optim=tcfg.optimizer,
            lr_scheduler_type=tcfg.scheduler,
            logging_steps=tcfg.logging_steps,
            save_steps=tcfg.save_steps,
            save_total_limit=3,
            bf16=self.device == "cuda",
            report_to=self.report_to,
            beta=tcfg.dpo_beta,
            loss_type=tcfg.online_dpo_loss_type,
            max_new_tokens=tcfg.online_dpo_max_new_tokens,
            max_length=cfg.data.max_length,
            use_cpu=self.device == "cpu",
        )

        judge_or_reward = self._build_judge_or_reward(tcfg)

        self.trainer = OnlineDPOTrainer(
            model=self.model,
            args=odpo_config,
            train_dataset=train_ds,
            processing_class=self.tokenizer,
            peft_config=self.peft_config,
            **judge_or_reward,
        )

        # Curriculum + plugin callbacks (relora is a no-op unless relora_steps).
        from soup_cli.utils.peft_wiring import (
            attach_curriculum_callback,
            attach_plugin_callback,
        )

        attach_curriculum_callback(self.trainer, tcfg, str(output_dir), console)
        attach_plugin_callback(self.trainer, console)

        self._output_dir = str(output_dir)

    def _setup_transformers(self, cfg: SoupConfig, tcfg) -> None:
        """Load model + tokenizer; build (but do NOT apply) the LoRA config.

        Online DPO passes ``peft_config`` to TRL, which wraps the model + builds
        the reference on demand (adapter-disable). So — unlike offline DPO — we
        do not ``get_peft_model`` here.
        """
        from peft import LoraConfig, TaskType
        from transformers import AutoModelForCausalLM, AutoTokenizer

        console.print(f"[dim]Loading tokenizer: {cfg.base}[/]")
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.base, trust_remote_code=self._trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Online DPO renders conversational prompts -> a chat template is
        # required. Fall back to a template WITH a generation cue when the model
        # ships none (so add_generation_prompt actually opens the assistant turn).
        if self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = _FALLBACK_CHAT_TEMPLATE

        from soup_cli.utils.quant_menu import build_quantization_config_for_loader

        quant_config_obj = build_quantization_config_for_loader(
            tcfg=tcfg, base=cfg.base, console=console,
        )

        console.print(f"[dim]Loading model: {cfg.base}[/]")
        dev_map = "cpu" if self.device == "cpu" else "auto"
        model_kwargs = {
            "trust_remote_code": self._trust_remote_code, "device_map": dev_map,
        }
        if quant_config_obj is not None:
            model_kwargs["quantization_config"] = quant_config_obj

        self.model = AutoModelForCausalLM.from_pretrained(cfg.base, **model_kwargs)

        from soup_cli.utils.data_pipeline import apply_vocab_expansion

        apply_vocab_expansion(self.tokenizer, self.model, cfg.data)

        # QLoRA stabilization: k-bit-loaded models need fp32 layer-norm casting +
        # input-grad enabling BEFORE LoRA. OnlineDPOTrainer's peft_config path
        # does NOT do this internally (unlike DPOTrainer), so do it here.
        if tcfg.quantization in ("4bit", "8bit", "mxfp4"):
            from peft import prepare_model_for_kbit_training

            self.model = prepare_model_for_kbit_training(self.model)

        target_modules = tcfg.lora.target_modules
        if target_modules == "auto":
            target_modules = None

        self.peft_config = LoraConfig(
            r=tcfg.lora.r,
            lora_alpha=tcfg.lora.alpha,
            lora_dropout=tcfg.lora.dropout,
            target_modules=target_modules,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
            use_dora=tcfg.lora.use_dora,
            use_rslora=tcfg.lora.use_rslora,
        )

        # Surgical PEFT patches operate on the base model (Gemma4 ClippableLinear).
        from soup_cli.utils.peft_wiring import apply_pre_lora_patches

        apply_pre_lora_patches(self.model, cfg.base)

    @staticmethod
    def _judge_kwargs(evaluator, has_judges: bool) -> dict:
        """Adapt a Soup evaluator to the installed trl's judge/reward API."""
        if has_judges:  # trl 0.19.x — swap-debiased pairwise judge
            from soup_cli.eval.judge import make_soup_pairwise_judge

            return {"judge": make_soup_pairwise_judge(evaluator)}
        # trl 1.x — pointwise reward function (judges were removed)
        from soup_cli.eval.judge import make_judge_reward_func

        return {"reward_funcs": [make_judge_reward_func(evaluator)]}

    def _build_judge_or_reward(self, tcfg) -> dict:
        """Resolve the OnlineDPO reward signal: judge OR reward_model.

        Precedence: the test seam, then the judge URL, then a reward model. The
        schema cross-validator guarantees exactly one of judge/reward is set for
        a real config. The returned kwargs adapt to the installed trl version
        (``judge=`` on 0.19.x, ``reward_funcs=`` on 1.x).
        """
        has_judges = _trl_has_judges()
        if _ONLINE_DPO_JUDGE_OVERRIDE is not None:
            return self._judge_kwargs(_ONLINE_DPO_JUDGE_OVERRIDE, has_judges)
        if tcfg.online_dpo_judge:
            from soup_cli.eval.gate import _parse_judge_url
            from soup_cli.eval.judge import JudgeEvaluator, validate_judge_api_base

            provider, model, api_base = _parse_judge_url(tcfg.online_dpo_judge)
            validate_judge_api_base(api_base)
            evaluator = JudgeEvaluator(provider=provider, model=model, api_base=api_base)
            return self._judge_kwargs(evaluator, has_judges)
        if tcfg.reward_model:
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )

            reward = AutoModelForSequenceClassification.from_pretrained(
                tcfg.reward_model,
                num_labels=1,
                trust_remote_code=self._trust_remote_code,
            )
            reward_tok = AutoTokenizer.from_pretrained(
                tcfg.reward_model, trust_remote_code=self._trust_remote_code
            )
            if has_judges:  # trl 0.19.x
                return {"reward_model": reward, "reward_processing_class": reward_tok}
            # trl 1.x — a reward model is one of reward_funcs
            return {
                "reward_funcs": [reward],
                "reward_processing_classes": [reward_tok],
            }
        raise ValueError(
            "online_dpo needs training.online_dpo_judge or training.reward_model"
        )

    def train(
        self,
        display: Optional[object] = None,
        tracker: Optional[object] = None,
        run_id: str = "",
        resume_from_checkpoint: Optional[str] = None,
    ) -> dict:
        """Run Online DPO training and return a results summary."""
        start = time.time()

        if display:
            from soup_cli.monitoring.callback import SoupTrainerCallback

            self.trainer.add_callback(
                SoupTrainerCallback(
                    display, tracker=tracker, run_id=run_id,
                    loss_watchdog=self.config.training.loss_watchdog,
                    loss_watchdog_threshold=self.config.training.loss_watchdog_threshold,
                    loss_watchdog_patience=self.config.training.loss_watchdog_patience,
                    eval_gate_config=self.config.training.eval_gate,
                )
            )

        from soup_cli.utils.v028_features import activation_offloading_context

        with activation_offloading_context(self.config.training, self._output_dir):
            self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        duration = time.time() - start

        self.trainer.save_model(self._output_dir)
        self.tokenizer.save_pretrained(self._output_dir)

        logs = self.trainer.state.log_history
        train_losses = [entry["loss"] for entry in logs if "loss" in entry]

        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        duration_str = f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"

        return {
            "initial_loss": train_losses[0] if train_losses else 0,
            "final_loss": train_losses[-1] if train_losses else 0,
            "duration": duration_str,
            "duration_secs": duration,
            "output_dir": self._output_dir,
            "total_steps": self.trainer.state.global_step,
        }
