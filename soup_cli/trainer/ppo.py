"""PPO (Proximal Policy Optimization) trainer — wraps trl.PPOTrainer.

PPO is the classic RLHF alignment method: generate completions, score them
with a reward model (or reward function), then optimize the policy using
clipped surrogate objectives with a KL penalty against a frozen reference model.

Full RLHF pipeline:  SFT → Reward Model → PPO
"""

import time
from pathlib import Path
from typing import Optional

from rich.console import Console

from soup_cli.config.schema import SoupConfig
from soup_cli.utils.gpu import estimate_batch_size, model_size_from_name

console = Console()


class PPOTrainerWrapper:
    """High-level wrapper for PPO training from SoupConfig.

    PPO generates completions for each prompt, scores them with a reward model
    or reward function, and optimizes using proximal policy optimization with
    a KL penalty against a frozen reference model.

    Supports two reward sources:
      - reward_model: path/HF ID of a trained reward model (AutoModelForSequenceClassification)
      - reward_fn: callable reward function (same as GRPO — 'accuracy', 'format', or custom .py)

    At least one of reward_model or reward_fn must be specified.
    """

    def __init__(
        self,
        config: SoupConfig,
        device: str = "cuda",
        report_to: str = "none",
        deepspeed_config: Optional[str] = None,
        fsdp_config: Optional[dict] = None,
    ):
        self.config = config
        self.device = device
        self.report_to = report_to
        self.deepspeed_config = deepspeed_config
        self.fsdp_config = fsdp_config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.reward_model_instance = None
        self.reward_fn = None

    def setup(self, dataset: dict):
        """Load model, tokenizer, reward model/fn, apply LoRA, create PPO trainer."""
        from datasets import Dataset

        # Import PPOTrainer/PPOConfig — trl >=0.28 moved to trl.experimental
        ppo_trainer_cls, ppo_config_cls, is_experimental = _import_ppo_classes()

        # Enable Rich progress bar for HuggingFace downloads
        from soup_cli.trainer.sft import _enable_hf_transfer_progress

        _enable_hf_transfer_progress()

        cfg = self.config
        tcfg = cfg.training
        use_unsloth = cfg.backend == "unsloth"

        # --- Load reward source ---
        self._setup_reward(cfg, tcfg)

        if use_unsloth:
            self._setup_unsloth(cfg, tcfg)
        else:
            self._setup_transformers(cfg, tcfg)

        trainable, total = self.model.get_nb_trainable_parameters()
        pct = 100 * trainable / total
        console.print(
            f"[green]LoRA applied:[/] {trainable:,} trainable"
            f" / {total:,} total ({pct:.2f}%)"
        )

        # --- Batch size ---
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
            # PPO needs memory for policy + ref model + reward model → conservative
            batch_size = max(1, batch_size // 4)
            console.print(f"[green]Auto batch size (PPO):[/] {batch_size}")

        # --- Dataset ---
        train_data = _prepare_ppo_dataset(dataset["train"])
        train_ds = Dataset.from_list(train_data)

        # Tokenize dataset: trl experimental PPOTrainer expects input_ids,
        # not raw text. Add input_ids/attention_mask from prompt_text.
        max_len = cfg.data.max_length

        def _tokenize_ppo(examples):
            return self.tokenizer(
                examples["prompt_text"],
                truncation=True,
                max_length=max_len,
                padding=False,
            )

        train_ds = train_ds.map(_tokenize_ppo, batched=True)

        # --- Output dir ---
        output_dir = Path(cfg.output)
        if cfg.experiment_name:
            output_dir = output_dir / cfg.experiment_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # --- PPO config ---
        # Build kwargs, handling trl version differences
        import inspect

        ppo_kwargs = {
            "output_dir": str(output_dir),
            "per_device_train_batch_size": batch_size,
            "gradient_accumulation_steps": tcfg.gradient_accumulation_steps,
            "learning_rate": tcfg.lr,
        }

        # FSDP2 — alternative to DeepSpeed
        if self.fsdp_config:
            ppo_kwargs.update(self.fsdp_config)

        ppo_params = inspect.signature(ppo_config_cls).parameters

        # trl renamed ppo_epochs -> num_ppo_epochs in newer versions
        if "num_ppo_epochs" in ppo_params:
            ppo_kwargs["num_ppo_epochs"] = tcfg.ppo_epochs
        elif "ppo_epochs" in ppo_params:
            ppo_kwargs["ppo_epochs"] = tcfg.ppo_epochs

        if "cliprange" in ppo_params:
            ppo_kwargs["cliprange"] = tcfg.ppo_clip_ratio
        if "init_kl_coef" in ppo_params:
            ppo_kwargs["init_kl_coef"] = tcfg.ppo_kl_penalty

        # Optional params that may not exist in all trl versions
        if "log_with" in ppo_params:
            ppo_kwargs["log_with"] = (
                self.report_to if self.report_to != "none" else None
            )
        elif "report_to" in ppo_params:
            ppo_kwargs["report_to"] = self.report_to

        if "optimize_cuda_cache" in ppo_params:
            ppo_kwargs["optimize_cuda_cache"] = self.device == "cuda"

        # CPU support: trl PPOConfig requires use_cpu=True when no CUDA
        if self.device == "cpu" and "use_cpu" in ppo_params:
            ppo_kwargs["use_cpu"] = True

        ppo_config = ppo_config_cls(**ppo_kwargs)

        # --- Build reward functions list for PPOTrainer ---
        reward_funcs = []
        if self.reward_model_instance is not None:
            reward_funcs.append(self.reward_model_instance)
        if self.reward_fn is not None:
            reward_funcs.append(self.reward_fn)

        # --- Trainer ---
        ppo_trainer_params = inspect.signature(ppo_trainer_cls.__init__).parameters

        # Store for use in train() — experimental API has different .train() signature
        self._is_experimental = is_experimental

        if is_experimental:
            # trl >=0.28 experimental API: PPOTrainer(args, processing_class,
            #   model, ref_model, reward_model, train_dataset, value_model, ...)
            # ref_model=None is fine (auto-creates from policy model).
            # reward_model and value_model are required nn.Modules.
            reward_model_obj = self._get_or_create_reward_model(cfg, tcfg)
            value_model_obj = self._create_value_model(cfg, tcfg)

            trainer_kwargs = {
                "args": ppo_config,
                "processing_class": self.tokenizer,
                "model": self.model,
                "ref_model": None,
                "reward_model": reward_model_obj,
                "train_dataset": train_ds,
                "value_model": value_model_obj,
            }
            self._dataset_in_constructor = True
            self.trainer = ppo_trainer_cls(**trainer_kwargs)

        elif "args" in ppo_trainer_params:
            # trl >=0.28 non-experimental (transitional API)
            trainer_kwargs = {
                "model": self.model,
                "args": ppo_config,
                "processing_class": self.tokenizer,
            }
            if "train_dataset" in ppo_trainer_params:
                trainer_kwargs["train_dataset"] = train_ds
            elif "dataset" in ppo_trainer_params:
                trainer_kwargs["dataset"] = train_ds
            if reward_funcs and "reward_funcs" in ppo_trainer_params:
                trainer_kwargs["reward_funcs"] = reward_funcs
            # Pass ref/reward/value models if required positionally
            if "ref_model" in ppo_trainer_params:
                trainer_kwargs["ref_model"] = None
            if "reward_model" in ppo_trainer_params:
                trainer_kwargs["reward_model"] = self._get_or_create_reward_model(
                    cfg, tcfg
                )
            if "value_model" in ppo_trainer_params:
                trainer_kwargs["value_model"] = self._create_value_model(cfg, tcfg)
            self._dataset_in_constructor = (
                "train_dataset" in trainer_kwargs or "dataset" in trainer_kwargs
            )
            self.trainer = ppo_trainer_cls(**trainer_kwargs)

        else:
            # trl <0.28: PPOTrainer(config=, model=, tokenizer=, dataset=)
            trainer_kwargs = {
                "model": self.model,
                "config": ppo_config,
                "tokenizer": self.tokenizer,
                "dataset": train_ds,
            }
            self._dataset_in_constructor = True
            self.trainer = ppo_trainer_cls(**trainer_kwargs)

        self._output_dir = str(output_dir)
        self._train_ds = train_ds
        self._batch_size = batch_size
        self._num_epochs = tcfg.epochs
        self._max_length = cfg.data.max_length

    def _get_or_create_reward_model(self, cfg, tcfg):
        """Get existing reward model or create one for trl experimental PPO API.

        The experimental PPOTrainer requires an nn.Module reward model (not a callable).
        If we have a loaded reward model instance, use it. Otherwise, load one from
        the configured reward_model path, or create a fresh AutoModelForSequenceClassification
        from the base model.
        """
        if self.reward_model_instance is not None:
            return self.reward_model_instance

        # If a reward_model path is configured, load it
        if tcfg.reward_model:
            return _load_reward_model(tcfg.reward_model, self.device)

        # Fallback: create a sequence classification model from the base model
        from transformers import AutoModelForSequenceClassification

        console.print(
            "[yellow]No reward_model path specified. Creating reward model "
            f"from base model: {cfg.base}[/]"
        )
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            cfg.base,
            trust_remote_code=True,
            num_labels=1,
            device_map="auto" if self.device != "cpu" else None,
        )
        reward_model.eval()
        return reward_model

    def _create_value_model(self, cfg, tcfg):
        """Create a value model for trl experimental PPO API.

        The value model estimates state values for GAE advantage estimation.
        We create an AutoModelForSequenceClassification from the base model.
        """
        from transformers import AutoModelForSequenceClassification

        console.print(f"[dim]Creating value model from: {cfg.base}[/]")
        value_model = AutoModelForSequenceClassification.from_pretrained(
            cfg.base,
            trust_remote_code=True,
            num_labels=1,
            device_map="auto" if self.device != "cpu" else None,
        )
        return value_model

    def _setup_reward(self, cfg, tcfg):
        """Load reward model and/or reward function."""
        # Reward model (pre-trained classifier)
        if tcfg.reward_model:
            self.reward_model_instance = _load_reward_model(
                tcfg.reward_model, self.device,
            )
            console.print(f"[green]Reward model loaded:[/] {tcfg.reward_model}")

        # Reward function (callable — reuse GRPO reward functions)
        if tcfg.reward_fn:
            from soup_cli.trainer.rewards import load_reward_fn

            self.reward_fn = load_reward_fn(tcfg.reward_fn)

        if self.reward_model_instance is None and self.reward_fn is None:
            console.print(
                "[yellow]Warning: No reward_model or reward_fn specified. "
                "Using default reward_fn='accuracy'.[/]"
            )
            from soup_cli.trainer.rewards import load_reward_fn

            self.reward_fn = load_reward_fn("accuracy")

    def _setup_transformers(self, cfg, tcfg):
        """Load model via standard transformers + peft pipeline."""
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        console.print(f"[dim]Loading tokenizer: {cfg.base}[/]")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.base, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        bnb_config = None
        if tcfg.quantization == "4bit":
            from soup_cli.utils.gpu import get_compute_dtype

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=get_compute_dtype(),
                bnb_4bit_use_double_quant=True,
            )
        elif tcfg.quantization == "8bit":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        console.print(f"[dim]Loading model: {cfg.base}[/]")
        # On CPU, use device_map="cpu" to avoid meta tensors from "auto"
        dev_map = "cpu" if self.device == "cpu" else "auto"
        model_kwargs = {"trust_remote_code": True, "device_map": dev_map}
        if bnb_config:
            model_kwargs["quantization_config"] = bnb_config

        self.model = AutoModelForCausalLM.from_pretrained(cfg.base, **model_kwargs)

        if tcfg.quantization in ("4bit", "8bit"):
            self.model = prepare_model_for_kbit_training(self.model)

        target_modules = tcfg.lora.target_modules
        if target_modules == "auto":
            target_modules = None

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
        self.model = get_peft_model(self.model, lora_config)

        # QAT — insert fake quantization ops after LoRA
        if tcfg.quantization_aware:
            from soup_cli.utils.qat import prepare_model_for_qat

            self.model = prepare_model_for_qat(self.model)

    def _setup_unsloth(self, cfg, tcfg):
        """Load model via unsloth FastLanguageModel (2-5x faster)."""
        from soup_cli.utils.unsloth import load_model_and_tokenizer

        console.print(f"[dim]Loading model via [bold]unsloth[/]: {cfg.base}[/]")
        self.model, self.tokenizer = load_model_and_tokenizer(
            model_name=cfg.base,
            max_seq_length=cfg.data.max_length,
            quantization=tcfg.quantization,
            lora_r=tcfg.lora.r,
            lora_alpha=tcfg.lora.alpha,
            lora_dropout=tcfg.lora.dropout,
            target_modules=tcfg.lora.target_modules,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def train(
        self,
        display: Optional[object] = None,
        tracker: Optional[object] = None,
        run_id: str = "",
        resume_from_checkpoint: Optional[str] = None,
    ) -> dict:
        """Run PPO training loop and return results summary.

        Supports two trl APIs:
          - trl >=0.28: uses built-in trainer.train() (OnlineDPOTrainer style)
          - trl <0.28: manual loop with generate() + step()
        """
        # Detect API: trl >=0.28 PPOTrainer has a .train() from OnlineDPOTrainer
        has_builtin_train = hasattr(self.trainer, "train") and not hasattr(
            self.trainer, "step"
        )

        if has_builtin_train:
            return self._train_builtin(display, tracker, run_id, resume_from_checkpoint)
        return self._train_manual(display, tracker, run_id)

    def _train_builtin(self, display, tracker, run_id, resume_from_checkpoint):
        """Train using trl >=0.28 built-in trainer.train() method."""
        start = time.time()

        # If dataset wasn't accepted by constructor, set it on the trainer
        if not self._dataset_in_constructor:
            if hasattr(self.trainer, "train_dataset"):
                self.trainer.train_dataset = self._train_ds
            elif hasattr(self.trainer, "dataset"):
                self.trainer.dataset = self._train_ds

        if display:
            from soup_cli.monitoring.callback import SoupTrainerCallback

            self.trainer.add_callback(
                SoupTrainerCallback(display, tracker=tracker, run_id=run_id)
            )

        # trl experimental PPOTrainer.train() does not accept resume_from_checkpoint
        import inspect

        train_params = inspect.signature(self.trainer.train).parameters
        if resume_from_checkpoint and "resume_from_checkpoint" in train_params:
            self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        else:
            if resume_from_checkpoint:
                console.print(
                    "[yellow]Warning: This PPOTrainer does not support "
                    "resume_from_checkpoint -- starting from scratch.[/]"
                )
            self.trainer.train()
        duration = time.time() - start

        # Save final model (LoRA adapter)
        self.trainer.save_model(self._output_dir)
        self.tokenizer.save_pretrained(self._output_dir)

        # Extract metrics
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

    def _train_manual(self, display, tracker, run_id):
        """Train using trl <0.28 manual loop: generate() + step()."""
        import torch

        start = time.time()
        step = 0
        all_rewards = []
        log_history = []

        for epoch in range(self._num_epochs):
            for batch_idx in range(0, len(self._train_ds), self._batch_size):
                batch_end = min(batch_idx + self._batch_size, len(self._train_ds))
                batch = self._train_ds[batch_idx:batch_end]

                # Tokenize prompts
                prompt_texts = batch["prompt_text"]
                query_tensors = [
                    self.tokenizer.encode(p, return_tensors="pt").squeeze()
                    for p in prompt_texts
                ]

                # Generate completions
                response_tensors = []
                for query in query_tensors:
                    gen_kwargs = {
                        "max_new_tokens": min(256, self._max_length // 2),
                        "do_sample": True,
                        "top_p": 0.9,
                        "temperature": 0.7,
                    }
                    response = self.trainer.generate(query.unsqueeze(0), **gen_kwargs)
                    response_tensors.append(response.squeeze()[len(query):])

                # Compute rewards
                rewards = self._compute_rewards(
                    query_tensors, response_tensors, batch,
                )

                # PPO step
                stats = self.trainer.step(query_tensors, response_tensors, rewards)
                step += 1

                mean_reward = torch.stack(rewards).mean().item()
                all_rewards.append(mean_reward)

                log_entry = {
                    "step": step,
                    "epoch": epoch + 1,
                    "loss": stats.get("ppo/loss/total", 0),
                    "reward": mean_reward,
                    "kl": stats.get("ppo/mean_kl", 0),
                    "lr": stats.get("ppo/learning_rate", self.config.training.lr),
                }
                log_history.append(log_entry)

                # Update display
                if display and hasattr(display, "update"):
                    display.update(
                        step=step,
                        epoch=epoch + 1,
                        loss=log_entry["loss"],
                        lr=log_entry["lr"],
                    )

                # Update tracker
                if tracker and run_id:
                    tracker.log_metrics(
                        run_id=run_id,
                        step=step,
                        epoch=epoch + 1,
                        loss=log_entry["loss"],
                        lr=log_entry["lr"],
                    )

        duration = time.time() - start

        # Save final model (LoRA adapter)
        self.model.save_pretrained(self._output_dir)
        self.tokenizer.save_pretrained(self._output_dir)

        # Extract metrics
        losses = [entry["loss"] for entry in log_history if "loss" in entry]

        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        duration_str = f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"

        return {
            "initial_loss": losses[0] if losses else 0,
            "final_loss": losses[-1] if losses else 0,
            "duration": duration_str,
            "duration_secs": duration,
            "output_dir": self._output_dir,
            "total_steps": step,
        }

    def _compute_rewards(self, query_tensors, response_tensors, batch):
        """Compute rewards using reward model and/or reward function."""
        import torch

        rewards = []
        num_samples = len(query_tensors)

        if self.reward_model_instance is not None:
            # Score with reward model
            for query, response in zip(query_tensors, response_tensors):
                full_ids = torch.cat([query, response]).unsqueeze(0)
                full_ids = full_ids.to(self.reward_model_instance.device)
                with torch.no_grad():
                    output = self.reward_model_instance(full_ids)
                    score = output.logits[:, -1].squeeze().float()
                rewards.append(score.cpu())

        elif self.reward_fn is not None:
            # Score with reward function
            completions = []
            for response in response_tensors:
                text = self.tokenizer.decode(response, skip_special_tokens=True)
                completions.append([{"role": "assistant", "content": text}])

            kwargs = {}
            if "answer" in batch:
                kwargs["answer"] = batch["answer"]

            scores = self.reward_fn(completions, **kwargs)
            rewards = [torch.tensor(score, dtype=torch.float32) for score in scores]

        else:
            # Fallback: zero reward
            rewards = [torch.tensor(0.0) for _ in range(num_samples)]

        return rewards


def _import_ppo_classes():
    """Import PPOTrainer and PPOConfig, handling trl version differences.

    trl >=0.28 moved PPOTrainer to trl.experimental with a new API that requires
    ref_model, reward_model, train_dataset, and value_model as positional args.
    The old trl import still works in 0.28 (deprecated) but is removed in 0.29.

    Returns:
        (PPOTrainer, PPOConfig, is_experimental): tuple with the classes and a flag
        indicating whether the experimental API (with required positional args) is used.
    """
    try:
        from trl.experimental.ppo import PPOConfig, PPOTrainer
        return PPOTrainer, PPOConfig, True
    except (ImportError, ModuleNotFoundError):
        pass

    from trl import PPOConfig, PPOTrainer
    return PPOTrainer, PPOConfig, False


def _load_reward_model(model_path: str, device: str = "cuda"):
    """Load a pre-trained reward model for PPO scoring.

    Reward models are typically AutoModelForSequenceClassification that output
    a scalar reward score for a given input sequence.
    """
    from transformers import AutoModelForSequenceClassification

    console.print(f"[dim]Loading reward model: {model_path}[/]")
    dev_map = "cpu" if device == "cpu" else "auto"
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map=dev_map,
        num_labels=1,
    )
    reward_model.eval()
    return reward_model


def _prepare_ppo_dataset(data: list[dict]) -> list[dict]:
    """Convert dataset rows to PPO format.

    PPO expects each row to have a 'prompt_text' field (string for tokenization)
    and optionally an 'answer' field (for reward function scoring).

    Input can be:
      - messages format: [{"role": "user", "content": "..."}, ...]
      - DPO format: {"prompt": "...", "chosen": "...", "rejected": "..."}
      - prompt field: {"prompt": "..."}
      - alpaca format: {"instruction": "...", "output": "..."}

    Returns list of dicts with 'prompt_text' (string) and optional 'answer'.
    """
    prepared = []
    for row in data:
        if "prompt" in row and isinstance(row["prompt"], str):
            entry = {"prompt_text": row["prompt"]}
            if "answer" in row:
                entry["answer"] = row["answer"]
            prepared.append(entry)
        elif "messages" in row:
            messages = row["messages"]
            # Use user messages as prompt text
            prompt_parts = [
                msg["content"] for msg in messages if msg["role"] in ("system", "user")
            ]
            entry = {"prompt_text": " ".join(prompt_parts)}
            prepared.append(entry)
        elif "prompt" in row and isinstance(row["prompt"], list):
            # Message list → join content
            prompt_parts = [msg.get("content", "") for msg in row["prompt"]]
            entry = {"prompt_text": " ".join(prompt_parts)}
            if "answer" in row:
                entry["answer"] = row["answer"]
            prepared.append(entry)
        else:
            # Alpaca fallback
            instruction = row.get("instruction", row.get("input", ""))
            entry = {"prompt_text": str(instruction)}
            if "output" in row:
                entry["answer"] = row["output"]
            prepared.append(entry)
    return prepared
