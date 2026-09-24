"""#1099: ``training.moe_lora`` loaded on five more LoRA tasks and none read it.

#1074 wired the flag into sft / pretrain / dpo / kto / orpo / simpo / grpo. It
still **loaded** on ``ipo``, ``bco``, ``reward_model``, ``ppo`` and ``embedding``,
where no trainer read it — the #748 / v0.75.0 silently-ignored-setting class.

Wired rather than refused, per the ruling on the issue: all five build their
adapter through ``peft_wiring.build_lora_config``, which is the condition the
maintainer named for preferring the wiring. Measured on ``main`` before writing
any of this — all five loaded ``moe_lora: true`` and no shipped recipe, template
or example sets it on one of these tasks, so no warn-then-refuse staging applies.

Each test drives the trainer's own ``_setup_transformers`` (or ``setup``) far
enough to attach LoRA, with only the loaders stubbed: ``get_peft_model`` and the
target-module resolution are the real ones, and each failure names the trainer
it came from.

#1151 adds the last three trainers that build through ``build_lora_config`` and
still accepted the flag unread: ``classifier`` (also ``reranker`` and
``cross_encoder``), ``distill`` and ``unlearn``. The two tasks with no expert
to adapt, ``asr`` (Whisper) and ``moe_lora_routing`` (no adapter built), are
refused at config load instead, and so is the classifier family without
``classifier_lora: true`` and ``lora.r > 0``, where that trainer full-fine-tunes.
``prm``, which fine-tunes every base parameter, is refused too (ruled on #1179).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from soup_cli.config.loader import load_config_from_string

transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")

#: task -> (wrapper module, wrapper class, auto-class stubbed, data format)
_TASKS = {
    "ipo": ("soup_cli.trainer.ipo", "IPOTrainerWrapper", "AutoModelForCausalLM", "dpo"),
    "bco": ("soup_cli.trainer.bco", "BCOTrainerWrapper", "AutoModelForCausalLM", "kto"),
    "reward_model": (
        "soup_cli.trainer.reward_model",
        "RewardModelTrainerWrapper",
        "AutoModelForSequenceClassification",
        "dpo",
    ),
    "ppo": ("soup_cli.trainer.ppo", "PPOTrainerWrapper", "AutoModelForCausalLM", "dpo"),
    # Found beyond the issue's five: it too builds through build_lora_config and
    # accepted the flag unread. TRL attaches its config later, so _attach does.
    "online_dpo": (
        "soup_cli.trainer.online_dpo",
        "OnlineDPOTrainerWrapper",
        "AutoModelForCausalLM",
        "auto",
    ),
    "embedding": (
        "soup_cli.trainer.embedding",
        "EmbeddingTrainerWrapper",
        "AutoModel",
        "embedding",
    ),
    # #1151: the three that build through build_lora_config and were still unread.
    # classifier also serves reranker and cross_encoder (one trainer, one call).
    "classifier": (
        "soup_cli.trainer.classifier",
        "ClassifierTrainerWrapper",
        "AutoModelForSequenceClassification",
        "auto",
    ),
    "distill": (
        "soup_cli.trainer.distill",
        "DistillTrainerWrapper",
        "AutoModelForCausalLM",
        "auto",
    ),
    "unlearn": (
        "soup_cli.trainer.unlearn",
        "UnlearnTrainerWrapper",
        "AutoModelForCausalLM",
        "auto",
    ),
}

#: #1151: tasks where the flag is refused at config load (no expert to adapt).
_REFUSED = ("asr", "moe_lora_routing", "prm")

#: task -> the extra fields its config needs to load (schema-required ones,
#: plus classifier's opt-in LoRA gate).
_EXTRA = {
    "online_dpo": {"training": {"online_dpo_judge": "ollama://llama3.1"}},
    "classifier": {"training": {"num_labels": 2, "classifier_lora": True}},
    "distill": {"training": {"teacher_model": "org/tiny-teacher"}},
    "unlearn": {
        "training": {"unlearn_method": "npo"},
        "data": {"forget_set": "./forget.jsonl"},
    },
    "moe_lora_routing": {"training": {"mole_task_adapters": ["./a", "./b"]}},
    # prm refuses an adapter-bearing lora block first (#795), so the refusal under
    # test is only reached at r: 0.
    "prm": {"training": {"lora": {"r": 0}}},
}

_SMALL = dict(
    vocab_size=64,
    hidden_size=16,
    intermediate_size=32,
    moe_intermediate_size=16,
    num_hidden_layers=1,
    num_attention_heads=2,
    num_key_value_heads=1,
    num_experts=4,
    num_experts_per_tok=2,
    decoder_sparse_step=1,
    max_position_embeddings=64,
)


def _moe_model(auto_class: str):
    """A real Qwen3-MoE of the head shape each trainer loads, ~10k parameters.

    The experts are fused 3-D weights (``mlp.experts.gate_up_proj``), which is the
    shape that makes ``target_modules: auto`` resolve to nothing.
    """
    from transformers import (
        Qwen3MoeConfig,
        Qwen3MoeForCausalLM,
        Qwen3MoeForSequenceClassification,
        Qwen3MoeModel,
    )

    config = Qwen3MoeConfig(**_SMALL, num_labels=1)
    return {
        "AutoModelForCausalLM": Qwen3MoeForCausalLM,
        "AutoModelForSequenceClassification": Qwen3MoeForSequenceClassification,
        "AutoModel": Qwen3MoeModel,
    }[auto_class](config)


def _dense_model(auto_class: str):
    """The control: same sizes, no experts."""
    from transformers import (
        Qwen3Config,
        Qwen3ForCausalLM,
        Qwen3ForSequenceClassification,
        Qwen3Model,
    )

    small = {
        key: value
        for key, value in _SMALL.items()
        if key not in ("moe_intermediate_size", "num_experts", "num_experts_per_tok",
                       "decoder_sparse_step")
    }
    config = Qwen3Config(**small, num_labels=1)
    return {
        "AutoModelForCausalLM": Qwen3ForCausalLM,
        "AutoModelForSequenceClassification": Qwen3ForSequenceClassification,
        "AutoModel": Qwen3Model,
    }[auto_class](config)


def _config(task: str, *, moe_lora: bool, dropout: float = 0.0):
    import yaml

    raw = {
        "base": "org/tiny-moe",
        "task": task,
        "data": {"train": "./x.jsonl", "format": _TASKS[task][3] if task in _TASKS else "auto"},
        "training": {"moe_lora": moe_lora, "lora": {"r": 4, "dropout": dropout}},
    }
    for section, fields in _EXTRA.get(task, {}).items():
        raw[section].update(fields)
    return load_config_from_string(yaml.safe_dump(raw))


def _attach(task: str, monkeypatch, *, moe_lora: bool, dense: bool = False):
    """Run the trainer's own setup far enough to attach LoRA; return the model."""
    import importlib

    module_path, class_name, auto_class, _ = _TASKS[task]
    module = importlib.import_module(module_path)
    wrapper_cls = getattr(module, class_name)

    tokenizer = SimpleNamespace(
        pad_token=None, eos_token="</s>", pad_token_id=0, chat_template="{{ x }}"
    )
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *_a, **_k: tokenizer
    )
    build = _dense_model if dense else _moe_model
    monkeypatch.setattr(
        getattr(transformers, auto_class),
        "from_pretrained",
        lambda *_a, **_k: build(auto_class),
    )

    cfg = _config(task, moe_lora=moe_lora)
    wrapper = object.__new__(wrapper_cls)
    wrapper.config = cfg
    wrapper.device = "cpu"
    wrapper.trust_remote_code = False
    wrapper._trust_remote_code = False
    wrapper._raw_trust_remote_code = False  # distill resolves the teacher's own
    wrapper.method = getattr(cfg.training, "unlearn_method", None)
    wrapper.model = None
    wrapper.tokenizer = None
    try:
        if hasattr(wrapper, "_setup_transformers"):
            wrapper._setup_transformers(cfg, cfg.training)
        else:
            # classifier / distill / unlearn: setup(dataset) attaches LoRA first
            # and only then reads the rows, which we do not supply.
            wrapper.setup({})
    except Exception as exc:  # noqa: BLE001 — setup continues past the LoRA attach
        # Only setup(dataset) reads rows after the attach (classifier / distill /
        # unlearn, fed no rows here). A ValueError out of _setup_transformers is
        # still the trainer's own refusal and must surface (#1179 review: the
        # tolerance was first applied to #1148's six trainers too).
        reads_rows_after_attach = not hasattr(wrapper, "_setup_transformers")
        if isinstance(exc, ValueError) and not reads_rows_after_attach:
            raise
        if _adapted(wrapper.model or object()):
            pass
        elif isinstance(exc, ValueError):
            raise
        else:
            raise AssertionError(
                f"{task}: setup failed before the LoRA attach: {exc!r}"
            ) from exc
    if getattr(wrapper, "peft_config", None) is not None and not _adapted(wrapper.model):
        # online_dpo hands its config to TRL, which attaches it; do the same.
        from peft import get_peft_model

        return get_peft_model(wrapper.model, wrapper.peft_config)
    return wrapper.model


def _adapted(model) -> list[str]:
    return [
        name
        for name, _ in getattr(model, "named_modules", lambda: [])()
        if name.endswith("lora_A.default") or name.endswith("lora_A")
    ]


@pytest.mark.parametrize("task", sorted(_TASKS))
class TestMoeLoraReachesTheAdapter:
    def test_moe_lora_true_adapts_the_experts(self, task, monkeypatch):
        """Each trainer separately, so removing one helper call fails a test that
        names that trainer rather than a shared one."""
        adapted = _adapted(_attach(task, monkeypatch, moe_lora=True))

        experts = [name for name in adapted if "experts" in name]
        assert experts, f"{task}: no expert module carries an adapter: {adapted}"

    def test_without_the_flag_no_expert_is_adapted(self, task, monkeypatch):
        """The control: the experts above are the flag's doing. Without it, on
        ``main`` ``target_modules: auto`` resolves to nothing for ``qwen3_moe``
        and the attach is refused; once #1102 maps ``qwen3_moe`` it adapts the
        attention projections only. Either way no expert carries an adapter,
        so this holds whichever of #1099 and #1102 merges first -- asserting
        the refusal alone passed on ``main`` and failed on top of #1102."""
        try:
            adapted = _adapted(_attach(task, monkeypatch, moe_lora=False))
        except ValueError as exc:
            assert "target_modules" in str(exc), exc
            adapted = []

        assert not [name for name in adapted if "experts" in name], adapted

    def test_a_dense_base_is_untouched_by_the_flag(self, task, monkeypatch):
        """The other control: the wiring must not change a dense model's targets,
        whatever the flag says."""
        off = _adapted(_attach(task, monkeypatch, moe_lora=False, dense=True))
        on = _adapted(_attach(task, monkeypatch, moe_lora=True, dense=True))

        assert off == on, (off, on)
        assert on, f"{task}: the dense control adapted nothing"


class TestTheFlagIsAcceptedOnEveryWiredTask:
    """The defect as reported: it loads and nothing reads it. These pin that the
    load still works, so the fix is a wiring change and not a new refusal."""

    @pytest.mark.parametrize("task", sorted(_TASKS))
    def test_the_config_still_loads(self, task):
        cfg = _config(task, moe_lora=True)

        assert cfg.training.moe_lora is True
        assert cfg.task == task


class TestTheFlagIsRefusedWhereNothingCouldReadIt:
    """#1151: ``asr`` trains Whisper (no experts) and ``moe_lora_routing`` builds
    no adapter, so the flag is refused at load with a message naming the task."""

    @pytest.mark.parametrize("task", _REFUSED)
    def test_moe_lora_true_is_refused(self, task):
        with pytest.raises(ValueError, match=f"moe_lora.*task={task!r}"):
            _config(task, moe_lora=True)

    @pytest.mark.parametrize("task", _REFUSED)
    def test_without_the_flag_the_task_still_loads(self, task):
        """The control: the refusal is the flag's doing, not the task's."""
        cfg = _config(task, moe_lora=False)

        assert cfg.task == task
        assert cfg.training.moe_lora is False


class TestTheClassifierFamilyNeedsItsAdapter:
    """#1151: ``trainer/classifier.py`` builds an adapter only with
    ``classifier_lora: true`` and ``lora.r > 0``; otherwise it full-fine-tunes,
    and ``moe_lora`` would load unread. Refused unless that adapter exists."""

    @staticmethod
    def _cfg(task, *, classifier_lora, r=4):
        import yaml

        training = {"moe_lora": True, "lora": {"r": r, "dropout": 0.0}}
        training["num_labels"] = 2
        if classifier_lora:
            training["classifier_lora"] = True
        return load_config_from_string(yaml.safe_dump({
            "base": "org/tiny-moe",
            "task": task,
            "data": {"train": "./x.jsonl", "format": "auto"},
            "training": training,
        }))

    @pytest.mark.parametrize("task", ["classifier", "reranker", "cross_encoder"])
    def test_without_classifier_lora_it_is_refused(self, task):
        with pytest.raises(ValueError, match=f"moe_lora.*task={task!r}.*classifier_lora"):
            self._cfg(task, classifier_lora=False)

    @pytest.mark.parametrize("task", ["classifier", "reranker", "cross_encoder"])
    def test_with_classifier_lora_it_loads(self, task):
        """The control: the wired path still loads."""
        assert self._cfg(task, classifier_lora=True).training.moe_lora is True

    def test_classifier_lora_at_rank_zero_is_refused_too(self):
        """``r: 0`` builds no adapter either, as the trainer's own gate decides."""
        with pytest.raises(ValueError, match=r"moe_lora.*lora\.r > 0"):
            self._cfg("classifier", classifier_lora=True, r=0)


class TestNoShippedConfigNeededStaging:
    """The precondition the issue names: if a shipped config had set the flag on
    one of these tasks, the standing rule would be warn-and-ignore this release
    rather than a straight wiring change (or refusal). None does -- and the
    tasks do appear in the catalogue, so this is not passing because the tasks
    are absent."""

    def test_no_recipe_template_or_example_sets_moe_lora_on_these_tasks(self):
        import pathlib

        import yaml

        from soup_cli.recipes.catalog import RECIPES

        root = pathlib.Path(__file__).resolve().parents[1]
        swept = set(_TASKS) | set(_REFUSED) | {"reranker", "cross_encoder"}
        offenders, tasks_seen = [], set()
        for name, recipe in RECIPES.items():
            loaded = yaml.safe_load(recipe.yaml_str)
            task = loaded.get("task")
            if task in swept:
                tasks_seen.add(task)
                if (loaded.get("training") or {}).get("moe_lora"):
                    offenders.append(f"recipe {name} ({task})")
        for base in (root / "src" / "soup_cli" / "templates", root / "examples"):
            for path in sorted(base.rglob("*.yaml")):
                try:
                    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
                except yaml.YAMLError:
                    continue
                if not isinstance(loaded, dict):
                    continue
                training = loaded.get("training") or {}
                if not isinstance(training, dict):
                    continue
                if loaded.get("task") in swept and training.get("moe_lora"):
                    offenders.append(f"{path.relative_to(root).as_posix()}")

        assert offenders == [], (
            "a shipped config sets moe_lora on a task this PR wires or refuses; the "
            "standing rule is warn-and-ignore for a release first: " + "; ".join(offenders)
        )
        # A floor, not just "non-empty" (#1148 review nit): these six are the swept
        # tasks the recipe catalogue uses today. If the sweep stops matching them,
        # it is no longer checking what it claims to.
        expected = {"asr", "embedding", "ipo", "online_dpo", "ppo", "reward_model"}
        assert tasks_seen >= expected, f"sweep missed {sorted(expected - tasks_seen)}"


class TestEveryAdapterBuildingTrainerReadsTheFlag:
    """#1179 review: this class has taken three PRs (#798, #1148, #1179). A new
    trainer module that builds a LoRA adapter must also pick the MoE targets, or be
    declared here as one whose task refuses ``moe_lora`` at config load."""

    #: Trainer modules whose task refuses the flag (schema ``_validate_moe_lora_task``).
    REFUSED_MODULES = frozenset({"asr.py"})

    @staticmethod
    def _calls(path):
        import ast

        names = set()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call):
                func = node.func
                names.add(getattr(func, "id", None) or getattr(func, "attr", None))
        return names

    def test_no_trainer_builds_an_adapter_without_the_moe_helper(self):
        import pathlib

        import soup_cli.trainer as trainer_pkg

        root = pathlib.Path(trainer_pkg.__file__).parent
        builders = {
            p.name: self._calls(p)
            for p in sorted(root.glob("*.py"))
            if "build_lora_config" in self._calls(p)
        }
        missing = sorted(
            name for name, calls in builders.items()
            if "resolve_moe_lora_targets" not in calls and name not in self.REFUSED_MODULES
        )

        assert len(builders) >= 17, f"the scan found only {sorted(builders)}"
        assert missing == [], f"these build an adapter but never read moe_lora: {missing}"

    def test_the_refused_modules_really_are_refused(self):
        """The exemption list must not drift from the schema's refusal."""
        with pytest.raises(ValueError, match="moe_lora.*task='asr'"):
            _config("asr", moe_lora=True)


class TestThePathsThatNeverReadIt:
    """A local CodeRabbit review of #1179 found the flag still loading unread by
    PATH, not task: every ``_setup_unsloth`` attaches ``utils/unsloth.py``'s fixed
    attention list, and SFT's vision / audio setups skip the MoE step."""

    @pytest.mark.parametrize("task", ["sft", "dpo", "grpo", "kto", "pretrain"])
    def test_unsloth_refuses_it_on_every_task(self, task):
        import yaml

        fmt = {"dpo": "dpo", "kto": "kto", "grpo": "alpaca", "pretrain": "plaintext"}.get(
            task, "alpaca"
        )
        raw = {
            "base": "org/m", "task": task, "backend": "unsloth",
            "data": {"train": "./x.jsonl", "format": fmt},
            "training": {"moe_lora": True, "lora": {"r": 4, "dropout": 0.0}},
        }
        with pytest.raises(ValueError, match="moe_lora is not applied on backend='unsloth'"):
            load_config_from_string(yaml.safe_dump(raw))

    @pytest.mark.parametrize(("modality", "fmt"), [("vision", "llava"), ("audio", "audio")])
    def test_sft_vision_and_audio_refuse_it(self, modality, fmt):
        import yaml

        raw = {
            "base": "org/m", "task": "sft", "modality": modality,
            "data": {"train": "./x.jsonl", "format": fmt},
            "training": {"moe_lora": True, "lora": {"r": 4, "dropout": 0.0}},
        }
        with pytest.raises(ValueError, match=f"modality={modality!r}"):
            load_config_from_string(yaml.safe_dump(raw))

    def test_sft_text_on_transformers_still_loads(self):
        """The control: the wired path is untouched."""
        cfg = _config("sft", moe_lora=True)

        assert cfg.training.moe_lora is True and cfg.backend == "transformers"
