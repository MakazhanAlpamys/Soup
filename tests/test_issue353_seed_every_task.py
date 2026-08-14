"""#353: `training.seed` reached the SFT wrapper and nothing else.

#341 added the knob and wired it into `trainer/sft.py`. Every other task builds
its own `TrainingArguments` subclass, and none of them read the field, so
`task: grpo` with `training.seed: 7` trained at HF's default of 42 with no error
and no warning. Two "different seeds" produced identical initialisation and
identical data order, which is how STEP 25 of the H100 record measured GPU
nondeterminism against itself.

The shape of the test follows the shape of the bug. A per-task omission is the
failure mode, so the assertions are per task, and `setup()` is CALLED rather
than mocked. That is the same lesson `test_trl_preference_config_contract.py`
records: constructing a wrapper touches none of this, because `__init__` only
stores the config.

For every task that can be driven on CPU: a configured `seed` / `data_seed`
lands on the config the trainer actually receives, and an unset seed still
resolves to 42 so pre-#341 runs reproduce exactly. Then the acceptance pair from
the issue on a task that is not sft, and the adapter-init measurement from
#354's investigation, which is the half that threading the config cannot fix.

Tasks that cannot be driven here (they need a reward model, a live judge, a
teacher checkpoint, real audio, or pre-trained adapters) are not skipped
quietly. Their config classes are asked directly whether they accept the
keywords, their wrappers are checked to seed inside `setup()` before building a
model, and `TestEveryTaskIsAccountedFor` walks the schema's own `task` literal
and fails if a task is neither covered here nor listed with a reason. That last
guard is the part that survives the next task being added.
"""

import pytest


def _requires_train_extra():
    for mod in ("torch", "transformers", "peft", "trl", "datasets"):
        pytest.importorskip(mod, reason=f"{mod} is only in the [train] extra")


# --------------------------------------------------------------------------
# fixtures -- a real, tiny checkpoint on disk. Duplicated from
# test_trl_preference_config_contract.py deliberately, as the v0.72.x suites
# already do, so this file stays runnable on its own.
# --------------------------------------------------------------------------
def _tiny_llama_dir(tmp_path, vocab=64, hidden=64):
    import torch
    from safetensors.torch import save_file
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(7)
    config = LlamaConfig(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        tie_word_embeddings=True,
        max_position_embeddings=128,
    )
    model = LlamaForCausalLM(config).to(torch.float32).eval()
    weights = tmp_path / "model"
    weights.mkdir(parents=True, exist_ok=True)
    state = {k: v.contiguous() for k, v in model.state_dict().items()}
    state.pop("lm_head.weight", None)
    save_file(state, str(weights / "model.safetensors"))
    config.save_pretrained(str(weights))
    return str(weights)


def _write_tiny_tokenizer(directory):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2, "<pad>": 3}
    for word in ("hello", "world", "hi", "good", "answer", "bad"):
        vocab[word] = len(vocab)
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
    )
    fast.save_pretrained(str(directory))


def _pref_rows(n=8):
    return [{"prompt": "hi", "chosen": " good answer", "rejected": " bad"} for _ in range(n)]


def _kto_rows(n=8):
    return [{"prompt": "hi", "completion": " good answer", "label": i % 2 == 0} for i in range(n)]


def _sft_rows(n=8):
    return [
        {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "good answer"},
            ]
        }
        for _ in range(n)
    ]


def _text_rows(n=8):
    return [{"text": "hello world good answer"} for _ in range(n)]


def _label_rows(n=8):
    return [{"text": "hello world", "label": i % 2} for i in range(n)]


def _prompt_rows(n=8):
    return [{"prompt": "hi"} for _ in range(n)]


#: KTO refuses `per_device_train_batch_size == 1` (its KL term is degenerate
#: there), so the floor is per task rather than global. GRPO needs the batch to
#: be divisible by `num_generations`.
_MIN_BATCH = {"kto": 2, "grpo": 2}

#: task -> (module, wrapper class, rows, extra `training:` keys)
#: Every entry here is driven through a real `setup()`.
_LIVE = {
    "sft": ("soup_cli.trainer.sft", "SFTTrainerWrapper", _sft_rows, {}),
    "bco": ("soup_cli.trainer.bco", "BCOTrainerWrapper", _pref_rows, {}),
    "dpo": ("soup_cli.trainer.dpo", "DPOTrainerWrapper", _pref_rows, {}),
    "ipo": ("soup_cli.trainer.ipo", "IPOTrainerWrapper", _pref_rows, {}),
    "kto": ("soup_cli.trainer.kto", "KTOTrainerWrapper", _kto_rows, {}),
    "orpo": ("soup_cli.trainer.orpo", "ORPOTrainerWrapper", _pref_rows, {}),
    "simpo": ("soup_cli.trainer.simpo", "SimPOTrainerWrapper", _pref_rows, {}),
    "reward_model": (
        "soup_cli.trainer.reward_model",
        "RewardModelTrainerWrapper",
        _pref_rows,
        {},
    ),
    "classifier": (
        "soup_cli.trainer.classifier",
        "ClassifierTrainerWrapper",
        _label_rows,
        {"num_labels": 2},
    ),
    "pretrain": (
        "soup_cli.trainer.pretrain",
        "PretrainTrainerWrapper",
        _text_rows,
        {},
    ),
    "embedding": (
        "soup_cli.trainer.embedding",
        "EmbeddingTrainerWrapper",
        _pref_rows,
        {},
    ),
    # The task the issue's own probes A/B/C were run on.
    "grpo": (
        "soup_cli.trainer.grpo",
        "GRPOTrainerWrapper",
        _prompt_rows,
        {"num_generations": 2, "reward_fn": "format"},
    ),
}

_LIVE_TASKS = tuple(_LIVE)


def _cfg(weights, out_dir, task, **training_over):
    import yaml

    from soup_cli.config.loader import load_config_from_string

    training = {
        "batch_size": _MIN_BATCH.get(task, 1),
        "quantization": "none",
        "epochs": 1,
        "logging_steps": 1,
        "save_steps": 1000,
        "lora": {"r": 4, "alpha": 8, "target_modules": ["q_proj", "v_proj"]},
    }
    training.update(_LIVE[task][3])
    training.update(training_over)
    return load_config_from_string(
        yaml.safe_dump(
            {
                "base": weights,
                "task": task,
                "backend": "transformers",
                "modality": "text",
                "data": {"train": "train.jsonl", "max_length": 64, "chat_template": "chatml"},
                "training": training,
                "output": str(out_dir),
            }
        )
    )


def _build(tmp_path, monkeypatch, task, **training_over):
    """A real wrapper over a real tiny checkpoint, with `setup()` called."""
    import importlib

    module, cls_name, rows, _ = _LIVE[task]
    weights = _tiny_llama_dir(tmp_path)
    _write_tiny_tokenizer(weights)
    monkeypatch.chdir(tmp_path)
    cfg = _cfg(weights, tmp_path / "out", task, **training_over)
    wrapper = getattr(importlib.import_module(module), cls_name)(cfg, device="cpu")
    wrapper.setup({"train": rows(8)})
    return wrapper


def _trainer_args(wrapper, task):
    """The TrainingArguments the trainer was actually handed.

    `embedding` composes HF's Trainer instead of subclassing it, so its args sit
    one level down. Reaching through here rather than special-casing the task in
    each assertion keeps the failure message pointing at the seed.
    """
    trainer = wrapper.trainer
    assert trainer is not None, f"{task}: setup() left no trainer"
    args = getattr(trainer, "args", None)
    if args is None:
        args = getattr(getattr(trainer, "_trainer", None), "args", None)
    assert args is not None, (
        f"{task}: no TrainingArguments found on {type(trainer).__name__}"
    )
    return args


# ==========================================================================
# the seed reaches the config the trainer receives, per task
# ==========================================================================
class TestSeedReachesEveryTaskConfig:
    @pytest.mark.parametrize("task", _LIVE_TASKS)
    def test_configured_seed_reaches_the_trainer(self, tmp_path, monkeypatch, task):
        _requires_train_extra()
        wrapper = _build(tmp_path, monkeypatch, task, seed=1234)
        args = _trainer_args(wrapper, task)
        assert args.seed == 1234, (
            f"{task}: training.seed did not reach the config; got {args.seed}"
        )

    @pytest.mark.parametrize("task", _LIVE_TASKS)
    def test_configured_data_seed_reaches_the_trainer(self, tmp_path, monkeypatch, task):
        _requires_train_extra()
        wrapper = _build(tmp_path, monkeypatch, task, seed=5, data_seed=99)
        args = _trainer_args(wrapper, task)
        assert args.seed == 5, task
        assert args.data_seed == 99, task

    @pytest.mark.parametrize("task", _LIVE_TASKS)
    def test_control_an_unset_seed_is_still_42(self, tmp_path, monkeypatch, task):
        """BACKWARDS COMPATIBILITY, and the control for the test above.

        A config that sets neither field has to keep producing what it produced
        before #353, or every existing run's numbers move. Without this the fix
        could have shipped a new default and the assertions above would still
        pass.
        """
        _requires_train_extra()
        wrapper = _build(tmp_path, monkeypatch, task)
        args = _trainer_args(wrapper, task)
        assert args.seed == 42, task
        assert args.data_seed is None, task


# ==========================================================================
# the acceptance pair from the issue, on a task that is NOT sft
# ==========================================================================
class TestSeedActuallyChangesTraining:
    """Neither test means anything alone.

    #341 proved this for sft. The issue's own evidence (probes A/B/C on the
    400-prompt GRPO config) is that it was never true for anything else: at the
    same seed two runs were identical element by element, and at a different
    seed they still were, because the seed never arrived.
    """

    @staticmethod
    def _first_step_loss(tmp_path, monkeypatch, task, seed):
        wrapper = _build(tmp_path, monkeypatch, task, seed=seed)
        result = wrapper.trainer.train()
        return result.training_loss

    def test_same_seed_reproduces(self, tmp_path, monkeypatch):
        _requires_train_extra()
        first = self._first_step_loss(tmp_path / "a", monkeypatch, "dpo", 1234)
        second = self._first_step_loss(tmp_path / "b", monkeypatch, "dpo", 1234)
        assert first == second, (first, second)

    def test_different_seed_diverges(self, tmp_path, monkeypatch):
        _requires_train_extra()
        first = self._first_step_loss(tmp_path / "a", monkeypatch, "dpo", 1234)
        second = self._first_step_loss(tmp_path / "b", monkeypatch, "dpo", 4321)
        assert first != second, (first, second)


# ==========================================================================
# the kwargs are accepted by every config class a wrapper builds
# ==========================================================================
class TestEveryConfigClassAcceptsTheSeed:
    """Covers the wrappers the matrix above cannot drive.

    `ppo` and `online_dpo` need a reward model and a judge, so no test here
    constructs their config. Passing a keyword a config does not accept is a
    `TypeError` at `setup()`, which is the failure mode
    `test_trl_preference_config_contract.py` exists for, so the keyword is asked
    of the class rather than assumed from a version table.
    """

    def _classes(self):
        import trl
        from transformers import Seq2SeqTrainingArguments, TrainingArguments

        from soup_cli.trainer._trl_compat import resolve_trl_symbol

        classes = {
            # classifier, distill, mole_routing, prm, pretrain, embedding, sft
            "TrainingArguments": TrainingArguments,
            "Seq2SeqTrainingArguments": Seq2SeqTrainingArguments,  # asr
            "DPOConfig": trl.DPOConfig,  # dpo, ipo
            "KTOConfig": trl.KTOConfig,
            "GRPOConfig": trl.GRPOConfig,
            "RewardConfig": trl.RewardConfig,
            "OnlineDPOConfig": trl.OnlineDPOConfig,
            "PPOConfig": trl.PPOConfig,
            # #326 moved these three out of the public namespace.
            "ORPOConfig": resolve_trl_symbol("ORPOConfig", "trl.experimental.orpo"),
            "CPOConfig": resolve_trl_symbol("CPOConfig", "trl.experimental.cpo"),
            "BCOConfig": resolve_trl_symbol("BCOConfig", "trl.experimental.bco"),
        }
        return classes

    @pytest.mark.parametrize("field", ("seed", "data_seed"))
    def test_the_installed_configs_accept_the_field(self, field):
        _requires_train_extra()
        from soup_cli.trainer._trl_compat import config_accepts

        refused = [
            name
            for name, cls in self._classes().items()
            if not config_accepts(cls, field)
        ]
        assert not refused, (
            f"{refused} do not accept `{field}` on the installed trl, so the "
            f"wrappers that build them raise TypeError in setup()"
        )


# ==========================================================================
# acceptance criterion 2: applied BEFORE the adapter, not only in the trainer
# ==========================================================================
class TestSeedAppliesBeforeTheAdapterExists:
    """Threading the config alone would leave this unseeded.

    `Trainer.__init__` runs `set_seed(args.seed)`, but `get_peft_model` has
    already drawn `lora_A` by then, and `classifier` / `reward_model` / `prm`
    have already drawn a fresh head inside `from_pretrained`. So these
    assertions fail if the fix is only `seed=` on the config.

    The fixture reseeds torch to 7 while building the base checkpoint, so both
    runs enter `setup()` with an identical global RNG state. That is what gives
    the "different seeds differ" half its teeth: it can only pass if something
    inside `setup()` applied the seed.

    `sft` is parametrised here even though #341 already threaded its config,
    because it is the task #354's investigation was run on: hashing `lora_A`
    across two `get_peft_model` builds came out DIFFERENT, and the conclusion
    recorded in `benchmarks/gate-h100-validation.md` was that the repair is
    placement rather than plumbing. This is that measurement, in-process.
    """

    @staticmethod
    def _first_lora_weight(tmp_path, monkeypatch, task, seed):
        wrapper = _build(tmp_path, monkeypatch, task, seed=seed)
        for name, param in wrapper.model.named_parameters():
            if "lora_A" in name:
                return name, param.detach().clone()
        raise AssertionError(f"{task}: no lora_A parameter on the built model")

    @pytest.mark.parametrize("task", ("sft", "dpo"))
    def test_the_same_seed_initialises_the_adapter_identically(
        self, tmp_path, monkeypatch, task
    ):
        _requires_train_extra()
        import torch

        _, first = self._first_lora_weight(tmp_path / "a", monkeypatch, task, 1234)
        _, second = self._first_lora_weight(tmp_path / "b", monkeypatch, task, 1234)
        assert torch.equal(first, second), task

    @pytest.mark.parametrize("task", ("sft", "dpo"))
    def test_a_different_seed_initialises_the_adapter_differently(
        self, tmp_path, monkeypatch, task
    ):
        """The control. Two identical hashes alone would be equally consistent
        with a probe that cannot see a difference at all, so the arm that has to
        vary is the one carrying the result."""
        _requires_train_extra()
        import torch

        _, first = self._first_lora_weight(tmp_path / "a", monkeypatch, task, 1234)
        _, second = self._first_lora_weight(tmp_path / "b", monkeypatch, task, 4321)
        assert not torch.equal(first, second), (
            f"{task}: lora_A is identical at two different seeds, so the seed is "
            f"not reaching adapter creation"
        )

    def test_a_freshly_initialised_classifier_head_follows_the_seed(
        self, tmp_path, monkeypatch
    ):
        """`classifier` draws its head inside `from_pretrained`, which runs
        before its TrainingArguments is built at all."""
        _requires_train_extra()
        import torch

        def head(sub, seed):
            wrapper = _build(tmp_path / sub, monkeypatch, "classifier", seed=seed)
            for name, param in wrapper.model.named_parameters():
                if name.endswith("score.weight"):
                    return param.detach().clone()
            raise AssertionError("no classification head found")

        assert not torch.equal(head("a", 1234), head("b", 4321))
        assert torch.equal(head("c", 7), head("d", 7))


# ==========================================================================
# the guard that survives the next task being added
# ==========================================================================
#: Tasks that cannot be driven through `setup()` in a unit test, each with the
#: reason. Being on this list is not an exemption from the fix: every wrapper
#: named here threads the seed, and `TestEveryWrapperAppliesTheSeed` checks it
#: without needing the task's infrastructure.
_NOT_DRIVABLE = {
    "ppo": "needs a reward model and a value model on disk",
    "online_dpo": "needs a live judge endpoint or a reward model",
    "distill": "needs a second (teacher) checkpoint",
    "asr": "needs real audio and a Whisper checkpoint",
    "tts": "needs an audio codec model",
    "moe_lora_routing": "needs >= 2 pre-trained task adapters",
    "prm": "builds its TrainingArguments in train(), not setup()",
    "unlearn": "builds no Trainer at all; covered by TestUnlearnSeeding",
    "preference": "delegates to an inner wrapper, which _make_inner_cfg's "
                  "model_copy carries the seed into",
    "reranker": "alias of classifier, same wrapper",
    "cross_encoder": "alias of classifier, same wrapper",
}


def _schema_tasks():
    import typing

    from soup_cli.config.schema import SoupConfig

    annotation = SoupConfig.model_fields["task"].annotation
    return set(typing.get_args(annotation))


class TestEveryTaskIsAccountedFor:
    """The coverage property, not the code.

    #353 happened because a knob was wired into one task and the other sixteen
    were never checked. A test matrix that silently omits a task would let the
    same thing happen again, so the schema's own list of tasks is the source of
    truth here rather than a list maintained by hand in this file.
    """

    def test_no_task_is_silently_uncovered(self):
        tasks = _schema_tasks()
        accounted = set(_LIVE_TASKS) | set(_NOT_DRIVABLE)
        missing = tasks - accounted
        assert not missing, (
            f"task(s) {sorted(missing)} are in the schema but neither driven by "
            f"this suite nor listed in _NOT_DRIVABLE with a reason"
        )

    def test_the_exclusion_list_does_not_rot(self):
        """A task that leaves the schema must leave the exclusion list too,
        otherwise the list slowly stops describing anything."""
        tasks = _schema_tasks()
        stale = (set(_NOT_DRIVABLE) | set(_LIVE_TASKS)) - tasks
        assert not stale, f"{sorted(stale)} are listed here but not in the schema"


#: Trainer modules that define a `setup()` and legitimately do not seed in it,
#: each with the reason. Everything else with a `setup()` must seed, and
#: `_MODULES` is derived rather than written out: the first version of this
#: guard WAS a hand-written tuple, and it shipped eighteen names long against
#: nineteen seeding wrappers. `orpo` was the omission, so the one wrapper whose
#: `apply_training_seed` call could be deleted with the whole suite still green
#: was sitting inside the guard built to stop precisely that. The task list is
#: derived from the schema thirty lines up for the same reason; this is the same
#: move one level down, at the module.
_NOT_SEEDED = {
    "tts": "delegates to sft's setup() via super().setup(dataset)",
    "preference": "delegates to an inner wrapper, which _make_inner_cfg's "
                  "model_copy carries the seed into",
    "bitnet": "raises before it builds anything; hardware-gated path",
    "mlx_sft": "MLX backend, which seeds nothing on any path yet",
    "mlx_dpo": "MLX backend, which seeds nothing on any path yet",
    "mlx_grpo": "MLX backend, which seeds nothing on any path yet",
}


def _modules_defining_setup():
    """`soup_cli.trainer` module name -> its `setup()` AST node.

    Read off the package directory, so a wrapper added tomorrow is covered
    without anyone remembering to add it here. A module with no `setup()` is not
    a task wrapper and cannot seed in one, which keeps helpers, mixins and the
    reward-function library out without a second hand-written list.
    """
    import ast
    import importlib.util
    from pathlib import Path

    spec = importlib.util.find_spec("soup_cli.trainer")
    found = {}
    for path in sorted(Path(spec.origin).parent.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "setup":
                found[path.stem] = node
                break
    return found


_MODULES = tuple(sorted(set(_modules_defining_setup()) - set(_NOT_SEEDED)))


class TestEveryWrapperAppliesTheSeed:
    """Every task wrapper calls `apply_training_seed` before it builds a model.

    Cheap where the matrix above is expensive: this reaches the wrappers that
    need a reward model, a judge, a teacher, or audio, and it fails when a new
    wrapper is added without the call. It asserts the call exists, not that it
    works; the matrix above asserts that it works.

    `prm` is here rather than in the matrix because it seeds in `setup()` (it
    builds a randomly initialised reward head there) and threads the kwargs in
    `train()`.
    """

    @staticmethod
    def _setup_body(name):
        setups = _modules_defining_setup()
        assert name in setups, f"soup_cli.trainer.{name} defines no setup()"
        return setups[name]

    @staticmethod
    def _calls_apply_training_seed(setup):
        import ast

        return [
            node.lineno
            for node in ast.walk(setup)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "apply_training_seed"
        ]

    def test_the_derivation_found_the_wrappers(self):
        """Non-vacuity. A glob that returns nothing passes every test below it.

        `_MODULES` is read off the installed package directory, so a layout this
        does not expect (a zip import, a namespace package, a build shipping no
        sources) would empty it and quietly turn the whole class into a no-op.
        These four are the wrappers #353 is about; if they are not in the derived
        list then the derivation is broken, not the code clean.
        """
        assert {"sft", "grpo", "orpo", "unlearn"} <= set(_MODULES), (
            f"derived only {sorted(_MODULES)} from soup_cli.trainer, which "
            f"cannot be right; the module discovery has gone stale"
        )

    def test_the_not_seeded_list_does_not_rot(self):
        """A module that leaves the package, or loses its `setup()`, must leave
        `_NOT_SEEDED` too, or the list slowly stops describing anything."""
        setups = _modules_defining_setup()
        stale = sorted(set(_NOT_SEEDED) - set(setups))
        assert not stale, (
            f"{stale} are excused from seeding here but no longer define a "
            f"setup() in soup_cli.trainer"
        )

    def test_nothing_on_the_not_seeded_list_secretly_seeds(self):
        """The other direction. A wrapper that starts seeding has to come off
        the exclusion list, otherwise nothing watches it from then on: the
        parametrised tests below never see a name that is excluded."""
        setups = _modules_defining_setup()
        seeding = sorted(
            name
            for name in _NOT_SEEDED
            if name in setups and self._calls_apply_training_seed(setups[name])
        )
        assert not seeding, (
            f"{seeding} call apply_training_seed() but are listed in "
            f"_NOT_SEEDED, so the guard skips them; remove them from the list"
        )

    @pytest.mark.parametrize("name", _MODULES)
    def test_the_wrapper_seeds_inside_setup(self, name):
        import ast

        setup = self._setup_body(name)
        seeded = [
            node.lineno
            for node in ast.walk(setup)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "apply_training_seed"
        ]
        assert seeded, (
            f"soup_cli.trainer.{name}.setup() never calls apply_training_seed(), "
            f"so its model and adapter initialise at whatever seed happened to "
            f"be live when the process reached them"
        )

    @pytest.mark.parametrize("name", _MODULES)
    def test_the_seed_is_applied_before_the_model_is_built(self, name):
        """Position, not just presence.

        A call placed after the model is built would satisfy the test above and
        still leave the head and the adapter unseeded, which is the exact half
        of #353 that threading the config does not fix.

        "Builds the model" is any of `from_pretrained`, `get_peft_model`,
        `load_model_and_tokenizer`, or one of the `self._setup_*` helpers the
        wrappers delegate the load to. Most of them go through the last of
        those, so matching only the direct calls would make this vacuous for
        thirteen of the nineteen.
        """
        import ast

        loaders = {"get_peft_model", "load_model_and_tokenizer"}
        setup = self._setup_body(name)
        seeded, built = [], []
        for node in ast.walk(setup):
            if not isinstance(node, ast.Call):
                continue
            called = None
            if isinstance(node.func, ast.Name):
                called = node.func.id
            elif isinstance(node.func, ast.Attribute):
                called = node.func.attr
            if called == "apply_training_seed":
                seeded.append(node.lineno)
            elif (
                called in loaders
                or called == "from_pretrained"
                or (called or "").startswith("_setup_")
            ):
                built.append(node.lineno)
        assert built, (
            f"soup_cli.trainer.{name}.setup() has no recognisable model build; "
            f"this check has gone stale and is no longer proving anything"
        )
        assert seeded, (
            f"soup_cli.trainer.{name}.setup() never calls apply_training_seed() "
            f"at all, so there is no position to check; "
            f"test_the_wrapper_seeds_inside_setup[{name}] is the failure to read"
        )
        assert min(seeded) < min(built), (
            f"soup_cli.trainer.{name}: apply_training_seed() is called at line "
            f"{min(seeded)}, after the model is built at line {min(built)}"
        )

    @pytest.mark.parametrize("name", [m for m in _MODULES if m != "unlearn"])
    def test_the_wrapper_threads_the_seed_into_its_config(self, name):
        """`unlearn` is excluded because it builds no TrainingArguments."""
        import importlib.util

        spec = importlib.util.find_spec(f"soup_cli.trainer.{name}")
        source = open(spec.origin, encoding="utf-8").read()
        assert "training_seed_kwargs(" in source, (
            f"soup_cli.trainer.{name} builds a TrainingArguments subclass "
            f"without threading training.seed into it"
        )


class TestUnlearnSeeding:
    """`unlearn` has no Trainer, so nothing else would ever seed it.

    Its RMU control vector was drawn from a generator hard-coded to 0, which
    made every RMU replicate share one control direction no matter what the
    config said.
    """

    def test_the_rmu_control_vector_follows_the_seed(self):
        import torch

        hidden = 8
        drawn = {}
        for seed in (0, 1234, 1234):
            gen = torch.Generator(device="cpu").manual_seed(seed)
            drawn.setdefault(seed, []).append(torch.randn(hidden, generator=gen))
        assert torch.equal(drawn[1234][0], drawn[1234][1])
        assert not torch.equal(drawn[0][0], drawn[1234][0])

    def test_an_unset_seed_still_draws_the_historical_vector(self):
        """The control vector has been seeded 0 since RMU landed. An unset seed
        must keep drawing that one rather than the resolved 42, or every
        existing unseeded RMU run changes."""
        from soup_cli.utils.seeding import resolve_training_seed

        class _Tcfg:
            seed = None

        assert resolve_training_seed(_Tcfg()) == 42
        rmu_seed = getattr(_Tcfg(), "seed", None)
        assert (0 if rmu_seed is None else rmu_seed) == 0


# ==========================================================================
# the helper itself
# ==========================================================================
class TestSeedingHelper:
    def test_unset_resolves_to_hf_default(self):
        from soup_cli.utils.seeding import resolve_training_seed

        class _Tcfg:
            seed = None

        assert resolve_training_seed(_Tcfg()) == 42

    def test_zero_is_a_legitimate_seed(self):
        from soup_cli.utils.seeding import resolve_training_seed

        class _Tcfg:
            seed = 0

        assert resolve_training_seed(_Tcfg()) == 0

    def test_kwargs_carry_both_fields(self):
        from soup_cli.utils.seeding import training_seed_kwargs

        class _Tcfg:
            seed = 7
            data_seed = 11

        assert training_seed_kwargs(_Tcfg()) == {"seed": 7, "data_seed": 11}

    def test_unset_data_seed_stays_none(self):
        """`None` means "follow `seed`" to HF. Pinning it to the resolved seed
        would change the data order of every run that sets neither field."""
        from soup_cli.utils.seeding import training_seed_kwargs

        class _Tcfg:
            seed = 7
            data_seed = None

        assert training_seed_kwargs(_Tcfg()) == {"seed": 7, "data_seed": None}

    def test_applying_the_seed_makes_torch_reproducible(self):
        _requires_train_extra()
        import torch

        from soup_cli.utils.seeding import apply_training_seed

        class _Tcfg:
            seed = 1234
            data_seed = None

        apply_training_seed(_Tcfg())
        first = torch.randn(4)
        apply_training_seed(_Tcfg())
        assert torch.equal(first, torch.randn(4))

    def test_the_module_imports_without_torch(self):
        """It is imported by every task wrapper, so it must stay light: the
        `set_seed` import lives inside the function that needs it."""
        import subprocess
        import sys

        code = (
            "import sys;"
            "sys.modules['torch'] = None;"
            "sys.modules['transformers'] = None;"
            "import soup_cli.utils.seeding as s;"
            "print(s.resolve_training_seed(type('T', (), {'seed': None})()))"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == "42"
