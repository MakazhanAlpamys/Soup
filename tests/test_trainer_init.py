"""Tests for trainer wrapper constructors and basic attributes (no GPU needed)."""

from soup_cli.config.schema import SoupConfig


def _make_config(**overrides):
    """Create a minimal SoupConfig for testing."""
    base = {
        "base": "test-model",
        "data": {"train": "./data.jsonl", "format": "alpaca"},
    }
    base.update(overrides)
    return SoupConfig(**base)


class TestSFTTrainerInit:
    """Test SFTTrainerWrapper constructor."""

    def test_default_attributes(self):
        from soup_cli.trainer.sft import SFTTrainerWrapper

        cfg = _make_config()
        wrapper = SFTTrainerWrapper(cfg, device="cpu")
        assert wrapper.config == cfg
        assert wrapper.device == "cpu"
        assert wrapper.report_to == "none"
        assert wrapper.deepspeed_config is None
        assert wrapper.model is None
        assert wrapper.tokenizer is None
        assert wrapper.trainer is None

    def test_custom_report_to(self):
        from soup_cli.trainer.sft import SFTTrainerWrapper

        cfg = _make_config()
        wrapper = SFTTrainerWrapper(cfg, device="cuda", report_to="wandb")
        assert wrapper.report_to == "wandb"

    def test_deepspeed_config(self):
        from soup_cli.trainer.sft import SFTTrainerWrapper

        cfg = _make_config()
        wrapper = SFTTrainerWrapper(
            cfg, device="cuda", deepspeed_config="/path/to/ds.json"
        )
        assert wrapper.deepspeed_config == "/path/to/ds.json"

    def test_transformers_vocab_expansion_adds_tokens_and_resizes(
        self, monkeypatch
    ):
        """data.add_new_tokens/new_special_tokens must affect text trainers."""
        import sys
        import types
        from types import SimpleNamespace

        from soup_cli.trainer.sft import SFTTrainerWrapper

        cfg = _make_config(
            data={
                "train": "./data.jsonl",
                "format": "alpaca",
                "add_new_tokens": ["<new_a>", "<old>"],
                "new_special_tokens": ["<special_a>"],
                "resize_vocab": True,
            },
            training={"quantization": "none"},
        )
        calls = {"add_tokens": None, "add_special_tokens": None, "resize": None}

        class _Tokenizer:
            pad_token = None
            eos_token = "<eos>"

            def __init__(self):
                self.vocab = {"<old>": 0}

            def add_tokens(self, tokens):
                calls["add_tokens"] = list(tokens)
                added = 0
                for token in tokens:
                    if token not in self.vocab:
                        self.vocab[token] = len(self.vocab)
                        added += 1
                return added

            def add_special_tokens(self, data):
                tokens = list(data["additional_special_tokens"])
                calls["add_special_tokens"] = tokens
                added = 0
                for token in tokens:
                    if token not in self.vocab:
                        self.vocab[token] = len(self.vocab)
                        added += 1
                return added

            def __len__(self):
                return len(self.vocab)

        class _Model:
            config = SimpleNamespace()

            def resize_token_embeddings(self, size):
                calls["resize"] = size

            def parameters(self):
                return []

        tokenizer = _Tokenizer()
        model = _Model()

        fake_transformers = types.SimpleNamespace(
            AutoTokenizer=types.SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: tokenizer
            ),
            AutoModelForCausalLM=types.SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: model
            ),
        )
        fake_peft = types.SimpleNamespace(
            LoraConfig=lambda **kwargs: SimpleNamespace(**kwargs),
            TaskType=SimpleNamespace(CAUSAL_LM="CAUSAL_LM"),
            get_peft_model=lambda model_obj, _cfg: model_obj,
            prepare_model_for_kbit_training=lambda model_obj: model_obj,
        )
        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
        monkeypatch.setitem(sys.modules, "peft", fake_peft)
        monkeypatch.setattr(
            "soup_cli.utils.quant_menu.build_quantization_config_for_loader",
            lambda **kwargs: None,
        )
        monkeypatch.setattr("soup_cli.utils.moe.detect_moe_model", lambda _m: False)
        monkeypatch.setattr("soup_cli.utils.moe.get_moe_target_modules", lambda _m: [])
        monkeypatch.setattr(
            "soup_cli.utils.block_expansion.apply_block_expansion_if_configured",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            "soup_cli.utils.moe_quant.apply_moe_expert_quant_if_configured",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            "soup_cli.utils.peft_wiring.apply_pre_lora_patches",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            "soup_cli.utils.peft_wiring.apply_post_lora_patches",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            "soup_cli.utils.v028_features.apply_v028_speed_memory",
            lambda *args, **kwargs: None,
        )

        wrapper = object.__new__(SFTTrainerWrapper)
        wrapper.config = cfg
        wrapper.device = "cpu"
        wrapper._trust_remote_code = False
        wrapper.model = None
        wrapper.tokenizer = None

        wrapper._setup_transformers(cfg, cfg.training)

        assert calls["add_tokens"] == ["<new_a>", "<old>"]
        assert calls["add_special_tokens"] == ["<special_a>"]
        assert calls["resize"] == len(tokenizer)
    def test_vision_vocab_expansion_adds_tokens_and_resizes(self, monkeypatch):
        """Vision SFT should apply configured vocabulary expansion."""

        import sys
        import types
        from types import SimpleNamespace

        from soup_cli.trainer.sft import SFTTrainerWrapper

        cfg = _make_config(
            modality="vision",
            data={
                "train": "./data.jsonl",
                "format": "llava",
                "add_new_tokens": ["<vision_new>", "<old>"],
                "new_special_tokens": ["<vision_special>"],
                "resize_vocab": True,
            },
            training={"quantization": "none"},
        )

        calls = {
            "add_tokens": None,
            "add_special_tokens": None,
            "resize": None,
        }

        class _Tokenizer:
            def __init__(self):
                self.vocab = {"<old>": 0}

            def get_vocab(self):
                return self.vocab

            def add_tokens(self, tokens):
                calls["add_tokens"] = list(tokens)
                for t in tokens:
                    self.vocab.setdefault(t, len(self.vocab))
                return len(tokens)

            def add_special_tokens(self, data):
                tokens = list(data["additional_special_tokens"])
                calls["add_special_tokens"] = tokens
                for t in tokens:
                    self.vocab.setdefault(t, len(self.vocab))
                return len(tokens)

            def __len__(self):
                return len(self.vocab)

        class _Processor:
            def __init__(self):
                self.tokenizer = _Tokenizer()

        class _Model:
            config = SimpleNamespace()

            def resize_token_embeddings(self, size):
                calls["resize"] = size

            def parameters(self):
                return []

        processor = _Processor()
        model = _Model()

        fake_transformers = types.SimpleNamespace(
            AutoProcessor=types.SimpleNamespace(from_pretrained=lambda *a, **k: processor),
            AutoModelForVision2Seq=types.SimpleNamespace(from_pretrained=lambda *a, **k: model),
        )

        fake_peft = types.SimpleNamespace(
            LoraConfig=lambda **kwargs: SimpleNamespace(**kwargs),
            get_peft_model=lambda m, cfg: m,
            prepare_model_for_kbit_training=lambda m: m,
        )

        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
        monkeypatch.setitem(sys.modules, "peft", fake_peft)

        monkeypatch.setattr(
            "soup_cli.utils.quant_menu.build_quantization_config_for_loader",
            lambda **kwargs: None,
        )

        wrapper = object.__new__(SFTTrainerWrapper)
        wrapper.config = cfg
        wrapper.device = "cpu"
        wrapper._trust_remote_code = False

        wrapper._setup_vision_transformers(cfg, cfg.training)

        assert calls["add_tokens"] == ["<vision_new>"]
        assert calls["add_special_tokens"] == ["<vision_special>"]
        assert calls["resize"] == len(processor.tokenizer)

    def test_audio_vocab_expansion_adds_tokens_and_resizes(self, monkeypatch):
        """Audio SFT should apply configured vocabulary expansion."""

        import sys
        import types
        from types import SimpleNamespace

        from soup_cli.trainer.sft import SFTTrainerWrapper

        cfg = _make_config(
            modality="audio",
            data={
                "train": "./data.jsonl",
                "format": "audio",
                "add_new_tokens": ["<audio_new>", "<old>"],
                "new_special_tokens": ["<audio_special>"],
                "resize_vocab": True,
            },
            training={"quantization": "none"},
        )

        calls = {
            "add_tokens": None,
            "add_special_tokens": None,
            "resize": None,
        }

        class _Tokenizer:
            def __init__(self):
                self.vocab = {"<old>": 0}

            def get_vocab(self):
                return self.vocab

            def add_tokens(self, tokens):
                calls["add_tokens"] = list(tokens)
                for t in tokens:
                    self.vocab.setdefault(t, len(self.vocab))
                return len(tokens)

            def add_special_tokens(self, data):
                tokens = list(data["additional_special_tokens"])
                calls["add_special_tokens"] = tokens
                for t in tokens:
                    self.vocab.setdefault(t, len(self.vocab))
                return len(tokens)

            def __len__(self):
                return len(self.vocab)

        class _Processor:
            def __init__(self):
                self.tokenizer = _Tokenizer()

        class _Model:
            config = SimpleNamespace()

            def resize_token_embeddings(self, size):
                calls["resize"] = size

            def parameters(self):
                return []

        processor = _Processor()
        model = _Model()

        fake_transformers = types.SimpleNamespace(
            AutoProcessor=types.SimpleNamespace(from_pretrained=lambda *a, **k: processor),
            AutoModel=types.SimpleNamespace(from_pretrained=lambda *a, **k: model),
        )

        fake_peft = types.SimpleNamespace(
            LoraConfig=lambda **kwargs: SimpleNamespace(**kwargs),
            get_peft_model=lambda m, cfg: m,
            prepare_model_for_kbit_training=lambda m: m,
        )

        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
        monkeypatch.setitem(sys.modules, "peft", fake_peft)

        monkeypatch.setattr(
            "soup_cli.utils.quant_menu.build_quantization_config_for_loader",
            lambda **kwargs: None,
        )

        wrapper = object.__new__(SFTTrainerWrapper)
        wrapper.config = cfg
        wrapper.device = "cpu"
        wrapper._trust_remote_code = False

        wrapper._setup_audio_transformers(cfg, cfg.training)

        assert calls["add_tokens"] == ["<audio_new>"]
        assert calls["add_special_tokens"] == ["<audio_special>"]
        assert calls["resize"] == len(processor.tokenizer)

class TestDPOTrainerInit:
    """Test DPOTrainerWrapper constructor."""

    def test_default_attributes(self):
        from soup_cli.trainer.dpo import DPOTrainerWrapper

        cfg = _make_config(task="dpo")
        wrapper = DPOTrainerWrapper(cfg, device="cpu")
        assert wrapper.config == cfg
        assert wrapper.device == "cpu"
        assert wrapper.model is None
        assert wrapper.ref_model is None
        assert wrapper.tokenizer is None
        assert wrapper.trainer is None

    def test_report_to_wandb(self):
        from soup_cli.trainer.dpo import DPOTrainerWrapper

        cfg = _make_config(task="dpo")
        wrapper = DPOTrainerWrapper(cfg, device="cpu", report_to="wandb")
        assert wrapper.report_to == "wandb"


class TestGRPOTrainerInit:
    """Test GRPOTrainerWrapper constructor."""

    def test_default_attributes(self):
        from soup_cli.trainer.grpo import GRPOTrainerWrapper

        cfg = _make_config(task="grpo")
        wrapper = GRPOTrainerWrapper(cfg, device="cpu")
        assert wrapper.config == cfg
        assert wrapper.device == "cpu"
        assert wrapper.model is None
        assert wrapper.tokenizer is None


class TestRewardModelTrainerInit:
    """Test RewardModelTrainerWrapper constructor."""

    def test_default_attributes(self):
        from soup_cli.trainer.reward_model import RewardModelTrainerWrapper

        cfg = _make_config(task="reward_model")
        wrapper = RewardModelTrainerWrapper(cfg, device="cpu")
        assert wrapper.config == cfg
        assert wrapper.device == "cpu"
        assert wrapper.model is None
        assert wrapper.tokenizer is None
        assert wrapper.trainer is None


class TestPPOTrainerInit:
    """Test PPOTrainerWrapper constructor."""

    def test_default_attributes(self):
        from soup_cli.trainer.ppo import PPOTrainerWrapper

        cfg = _make_config(task="ppo")
        wrapper = PPOTrainerWrapper(cfg, device="cpu")
        assert wrapper.config == cfg
        assert wrapper.device == "cpu"


class TestTrainTaskRouting:
    """Test that train command routes to correct trainer based on task."""

    def test_sft_is_default_task(self):
        cfg = _make_config()
        assert cfg.task == "sft"

    def test_dpo_task(self):
        cfg = _make_config(task="dpo")
        assert cfg.task == "dpo"

    def test_grpo_task(self):
        cfg = _make_config(task="grpo")
        assert cfg.task == "grpo"

    def test_ppo_task(self):
        cfg = _make_config(task="ppo")
        assert cfg.task == "ppo"

    def test_reward_model_task(self):
        cfg = _make_config(task="reward_model")
        assert cfg.task == "reward_model"

    def test_backend_default_is_transformers(self):
        cfg = _make_config()
        assert cfg.backend == "transformers"

    def test_backend_unsloth(self):
        cfg = _make_config(backend="unsloth")
        assert cfg.backend == "unsloth"

    def test_modality_default_is_text(self):
        cfg = _make_config()
        assert cfg.modality == "text"

    def test_modality_vision(self):
        cfg = _make_config(modality="vision")
        assert cfg.modality == "vision"


class TestEnableHfTransferProgress:
    """Test _enable_hf_transfer_progress utility."""

    def test_enables_progress_bars(self):
        from soup_cli.trainer.sft import _enable_hf_transfer_progress

        # Should not raise
        _enable_hf_transfer_progress()
