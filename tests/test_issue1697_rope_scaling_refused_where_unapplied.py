"""#1697 — ``training.rope_scaling_type`` is refused at load where nothing applies it.

The field loaded on all 23 tasks and every path, and two methods read it:
``SFTTrainerWrapper._setup_transformers`` (``tts`` reaches it through ``super()``)
and ``PretrainTrainerWrapper._setup_transformers``. Everywhere else the run
trained on the checkpoint's own RoPE and said nothing. Those configs now stop at
load, naming the field, the task or the setting that leaves the reading setup,
and what to change.

Nothing is trained here: the gate is a config validator, and the two ratchet
classes tie it to the trainers by reading their source.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import textwrap
import typing
from pathlib import Path

import pytest
import yaml

from soup_cli.config.loader import load_config_from_string
from soup_cli.config.schema import ROPE_SCALING_APPLYING_TASKS, TEMPLATES, SoupConfig
from soup_cli.trainer.mlx_sft import MLXSFTTrainerWrapper
from soup_cli.trainer.pretrain import PretrainTrainerWrapper
from soup_cli.trainer.sft import SFTTrainerWrapper

_TASKS = tuple(typing.get_args(SoupConfig.model_fields["task"].annotation))
# Written out, not read from ROPE_SCALING_APPLYING_TASKS: a task dropped from or
# added to the constant must not move its own tests with it.
_APPLYING = ("sft", "tts", "pretrain")
_REFUSED = tuple(task for task in _TASKS if task not in _APPLYING)
_TYPES = ("linear", "dynamic", "yarn", "llama3")
_YARN_FIELDS = ("yarn_factor", "yarn_attn_factor", "yarn_beta_fast", "yarn_beta_slow")

# The minimum each task needs to load at all (as in the #1497 tests).
_TOP = {"tts": {"modality": "audio_out"}}
_DATA = {
    "dpo": {"format": "dpo"}, "kto": {"format": "kto"}, "pretrain": {"format": "plaintext"},
    "unlearn": {"forget_set": "./f.jsonl"},
}
_TRAINING = {
    "preference": {"preference_loss": "dpo"},
    "tts": {"tts_family": "orpheus"},
    "classifier": {"num_labels": 2}, "reranker": {"num_labels": 2},
    "cross_encoder": {"num_labels": 2},
    "unlearn": {"unlearn_method": "npo"},
    "moe_lora_routing": {"mole_task_adapters": ["./a", "./b"]},
    "distill": {"teacher_model": "org/t"},
    "online_dpo": {"online_dpo_judge": "pairrm"},
}
# preference builds one of these per preference_loss (trainer/preference.py).
_PREFERENCE_DELEGATES = ("dpo", "simpo", "orpo", "ipo", "bco")


def _yaml(task="sft", top=None, **training):
    raw = {
        "base": "org/m", "task": task, **_TOP.get(task, {}), **(top or {}),
        "data": {"train": "./x.jsonl", **_DATA.get(task, {})},
        "training": {**_TRAINING.get(task, {}), **training},
    }
    return yaml.safe_dump(raw)


def _refusal(text):
    with pytest.raises(ValueError) as info:
        load_config_from_string(text)
    return " ".join(str(info.value).split())


def test_the_task_list_is_the_one_this_file_was_written_for():
    assert len(_TASKS) == 23
    assert set(_APPLYING) <= set(_TASKS)
    assert ROPE_SCALING_APPLYING_TASKS == frozenset({"sft", "tts", "pretrain"})


class TestRefusedOutsideTheThreeTasks:
    @pytest.mark.parametrize("rope_type", _TYPES)
    @pytest.mark.parametrize("task", _REFUSED)
    def test_the_message_names_the_field_the_task_and_the_fix(self, task, rope_type):
        message = _refusal(_yaml(task, rope_scaling_type=rope_type))

        assert (
            f"training.rope_scaling_type is not applied by task={task!r}: only "
            "task='sft', 'tts' and 'pretrain' rescale RoPE"
        ) in message, message
        assert "would train on the checkpoint's own RoPE" in message
        assert message.endswith("Use one of those tasks, or remove the key.")

    @pytest.mark.parametrize("task", _REFUSED)
    def test_without_the_field_the_task_still_loads(self, task):
        assert load_config_from_string(_yaml(task)).training.rope_scaling_type is None

    @pytest.mark.parametrize("backend", ["transformers", "unsloth"])
    def test_the_task_refusal_does_not_depend_on_the_backend(self, backend):
        """``dpo`` on unsloth is refused for the task, not for the path."""
        message = _refusal(_yaml("dpo", {"backend": backend}, rope_scaling_type="yarn"))

        assert "is not applied by task='dpo': only task='sft'" in message, message

    def test_a_dumped_config_still_reloads(self):
        """A dumped config writes the field out as null; that is not a request."""
        dumped = yaml.safe_dump(
            load_config_from_string(_yaml("dpo")).model_dump(mode="json")
        )

        assert "rope_scaling_type: null" in dumped
        assert load_config_from_string(dumped).task == "dpo"


# The paths of the issue's table that load today: (task, top-level keys, training
# keys, the setting the message names, the change it asks for).
_OFF_PATH = {
    "sft-mlx": (
        "sft", {"backend": "mlx"}, {}, "backend='mlx'", "backend: transformers",
    ),
    "sft-vision": (
        "sft", {"modality": "vision"}, {}, "modality='vision'", "modality: text",
    ),
    "sft-audio": (
        "sft", {"modality": "audio"}, {}, "modality='audio'", "modality: text",
    ),
    "sft-stream": (
        "sft", {}, {"stream_layers": True, "batch_size": 1},
        "training.stream_layers=true", "stream_layers: false",
    ),
    "sft-unsloth": (
        "sft", {"backend": "unsloth"}, {}, "backend='unsloth'", "backend: transformers",
    ),
    "tts-unsloth": (
        "tts", {"backend": "unsloth"}, {}, "backend='unsloth'", "backend: transformers",
    ),
    "pretrain-unsloth": (
        "pretrain", {"backend": "unsloth"}, {}, "backend='unsloth'",
        "backend: transformers",
    ),
}
# Only sft has a modality or a streaming setup to leave.
_WHERE = {
    "sft": "on the text path with backend: transformers and no layer streaming",
    "tts": "with backend: transformers",
    "pretrain": "with backend: transformers",
}


class TestRefusedOffTheReadingSetup:
    @pytest.mark.parametrize("rope_type", _TYPES)
    @pytest.mark.parametrize("case", list(_OFF_PATH))
    def test_the_message_names_the_field_the_path_and_the_fix(self, case, rope_type):
        task, top, training, setting, change = _OFF_PATH[case]

        message = _refusal(_yaml(task, top, **training, rope_scaling_type=rope_type))

        assert (
            f"training.rope_scaling_type is not applied by task={task!r} with {setting}: "
            f"RoPE is rescaled only {_WHERE[task]}, so"
        ) in message, message
        assert "would train on the checkpoint's own RoPE" in message
        assert message.endswith(f"Use {change}, or remove the key.")

    @pytest.mark.parametrize("case", list(_OFF_PATH))
    def test_without_the_field_the_path_still_loads(self, case):
        """The refusal is about the field, not about the path."""
        task, top, training, _, _ = _OFF_PATH[case]

        cfg = load_config_from_string(_yaml(task, top, **training))

        assert cfg.training.rope_scaling_type is None

    @pytest.mark.parametrize(
        ("top", "settings", "change"),
        [
            (
                {"backend": "unsloth", "modality": "vision"},
                "modality='vision' and backend='unsloth'",
                "modality: text and backend: transformers",
            ),
            (
                {"backend": "unsloth", "modality": "audio"},
                "modality='audio' and backend='unsloth'",
                "modality: text and backend: transformers",
            ),
            (
                {"backend": "mlx", "modality": "vision"},
                "backend='mlx' and modality='vision'",
                "backend: transformers and modality: text",
            ),
        ],
    )
    def test_every_setting_off_the_path_is_named(self, top, settings, change):
        """Changing one of two settings would still leave the run off the text
        path, so both are named, in the order the trainer checks them."""
        message = _refusal(_yaml("sft", top, rope_scaling_type="dynamic"))

        assert f"with {settings}:" in message, message
        assert message.endswith(f"Use {change}, or remove the key.")

    def test_the_template_with_its_unsloth_hint_taken_is_refused(self):
        """The issue's example: ``soup init --template longcontext``, then the
        ``# backend: unsloth`` line every other template suggests."""
        raw = yaml.safe_load(TEMPLATES["longcontext"])
        assert raw["training"]["rope_scaling_type"] == "dynamic"
        raw["backend"] = "unsloth"

        message = _refusal(yaml.safe_dump(raw))

        assert "is not applied by task='sft' with backend='unsloth'" in message, message


class TestTheYarnKeysAreNamedWithIt:
    """The ``yarn_*`` keys are legal only with ``rope_scaling_type: yarn``, so a
    message that said to remove one key would send the config into the next
    refusal."""

    def test_on_a_task_that_does_not_apply_them(self):
        message = _refusal(_yaml("dpo", rope_scaling_type="yarn", yarn_factor=4.0))

        assert (
            "training.rope_scaling_type and training.yarn_factor are not applied by "
            "task='dpo'"
        ) in message, message
        assert message.endswith("Use one of those tasks, or remove both keys.")

    def test_on_a_path_that_does_not_apply_them(self):
        message = _refusal(
            _yaml(
                "sft", {"backend": "unsloth"}, rope_scaling_type="yarn",
                yarn_factor=4.0, yarn_attn_factor=1.0, yarn_beta_fast=32, yarn_beta_slow=1,
            )
        )

        assert (
            "training.rope_scaling_type, training.yarn_factor, training.yarn_attn_factor, "
            "training.yarn_beta_fast and training.yarn_beta_slow are not applied by "
            "task='sft' with backend='unsloth'"
        ) in message, message
        assert message.endswith("Use backend: transformers, or remove the 5 keys.")

    def test_following_the_message_loads(self):
        raw = yaml.safe_load(_yaml("dpo", rope_scaling_type="yarn", yarn_factor=4.0))
        for key in ("rope_scaling_type", "yarn_factor"):
            del raw["training"][key]

        assert load_config_from_string(yaml.safe_dump(raw)).task == "dpo"


class TestTheReadingSetupsStillLoad:
    # Written out rather than derived from the tables above, so a bad table
    # cannot also remove its control.
    @pytest.mark.parametrize("rope_type", _TYPES)
    @pytest.mark.parametrize("task", ["sft", "tts", "pretrain"])
    def test_on_transformers(self, task, rope_type):
        cfg = load_config_from_string(
            _yaml(task, {"backend": "transformers"}, rope_scaling_type=rope_type)
        )

        assert cfg.training.rope_scaling_type == rope_type

    def test_yarn_with_its_tunables(self):
        cfg = load_config_from_string(_yaml("sft", rope_scaling_type="yarn", yarn_factor=4.0))

        assert cfg.training.yarn_factor == 4.0

    def test_stream_layers_false_is_not_streaming(self):
        cfg = load_config_from_string(
            _yaml("sft", stream_layers=False, rope_scaling_type="dynamic")
        )

        assert cfg.training.rope_scaling_type == "dynamic"

    @pytest.mark.parametrize("modality", ["vision", "audio"])
    def test_pretrain_has_one_transformers_setup_whatever_the_modality(self, modality):
        """``PretrainTrainerWrapper.setup`` branches on the backend only, so
        ``modality`` does not take a pretrain run away from the read."""
        cfg = load_config_from_string(
            _yaml("pretrain", {"modality": modality}, rope_scaling_type="dynamic")
        )

        assert cfg.training.rope_scaling_type == "dynamic"

    def test_the_longcontext_template(self):
        cfg = load_config_from_string(TEMPLATES["longcontext"])

        assert (cfg.task, cfg.backend) == ("sft", "transformers")
        assert cfg.training.rope_scaling_type == "dynamic"

    def test_the_longcontext_recipe(self):
        from soup_cli.recipes.catalog import get_recipe

        cfg = load_config_from_string(get_recipe("llama3.1-8b-longctx").yaml_str)

        assert (cfg.task, cfg.backend) == ("sft", "transformers")
        assert cfg.training.rope_scaling_type == "yarn"

    def test_the_longcontext_template_no_longer_suggests_unsloth(self):
        """Every other template carries the hint; this one sets a field that
        unsloth's setup does not read."""
        from importlib import resources

        packaged = (
            resources.files("soup_cli.templates").joinpath("longcontext.yaml").read_text("utf-8")
        )

        for text in (TEMPLATES["longcontext"], packaged):
            assert "rope_scaling_type: dynamic" in text
            assert "unsloth" not in text


class TestAlreadyRefusedForAnotherReason:
    """Two rows of the issue's table never reached the new check, before or
    after: they are listed so nobody counts them as covered by it."""

    def test_pretrain_on_mlx(self):
        message = _refusal(_yaml("pretrain", {"backend": "mlx"}, rope_scaling_type="yarn"))

        assert "MLX backend only ships SFT" in message, message

    def test_pretrain_with_stream_layers(self):
        message = _refusal(
            _yaml("pretrain", stream_layers=True, batch_size=1, rope_scaling_type="yarn")
        )

        assert "training.stream_layers supports task in" in message, message


class TestSoupDoctorReportsIt:
    @pytest.mark.parametrize(
        ("task", "top", "expected"),
        [
            ("dpo", {}, "training.rope_scaling_type is not applied by task='dpo'"),
            (
                "sft", {"backend": "mlx"},
                "training.rope_scaling_type is not applied by task='sft' with backend='mlx'",
            ),
        ],
    )
    def test_doctor_exits_2_naming_it(self, tmp_path, task, top, expected):
        from typer.testing import CliRunner

        from soup_cli.cli import app
        from tests.conftest import strip_ansi

        path = tmp_path / "soup.yaml"
        path.write_text(_yaml(task, top, rope_scaling_type="dynamic"), encoding="utf-8")

        result = CliRunner().invoke(app, ["doctor", "--config", str(path)])

        assert result.exit_code == 2, result.output
        assert expected in " ".join(strip_ansi(result.output).split())


class TestValidatorOrder:
    def test_it_sits_before_the_backend_refusal(self):
        """#1357's backend refusal stays last; this one refuses on every
        backend, so it belongs in front of that."""
        validators = SoupConfig.__pydantic_decorators__.model_validators
        after = [name for name, dec in validators.items() if dec.info.mode == "after"]

        assert (
            after.index("_validate_freeze_is_applied")
            < after.index("_validate_rope_scaling_is_applied")
            < after.index("_validate_unsloth_has_a_setup")
        ), after[-4:]
        assert after[-1] == "_validate_unsloth_has_a_setup"


def _reads_rope(source):
    """True when ``source`` reads the field as an attribute. Comments and
    docstrings do not count. Like the #1497 scan, a read through a helper
    outside the scanned code, or through ``getattr``, is not seen."""
    return any(
        isinstance(node, ast.Attribute) and node.attr == "rope_scaling_type"
        for node in ast.walk(ast.parse(textwrap.dedent(source)))
    )


def _wrapper_reads_rope(cls):
    """The same question for a wrapper and the soup classes it inherits from."""
    return any(
        _reads_rope(inspect.getsource(klass))
        for klass in cls.__mro__
        if klass.__module__.startswith("soup_cli.")
    )


def _dispatched_wrappers():
    """task -> wrapper class, read from the if-chain in ``trainer/dispatch.py``
    (the #1497 reader: a task added to the dispatcher is scanned without anyone
    remembering this file)."""
    from soup_cli.trainer import dispatch

    tree = ast.parse(Path(inspect.getsourcefile(dispatch)).read_text(encoding="utf-8"))
    found = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
            continue
        test = node.test
        if not (isinstance(test.left, ast.Name) and test.left.id == "task"):
            continue
        right = test.comparators[0]
        if isinstance(test.ops[0], ast.Eq) and isinstance(right, ast.Constant):
            tasks = [right.value]
        elif isinstance(test.ops[0], ast.In) and isinstance(right, ast.Tuple):
            tasks = [elt.value for elt in right.elts]
        else:
            continue
        imports = [n for n in node.body if isinstance(n, ast.ImportFrom)]
        assert len(imports) == 1, f"expected one import under task={tasks}"
        module = importlib.import_module(imports[0].module)
        cls = getattr(module, imports[0].names[0].name)
        for task in tasks:
            found[task] = cls
    return found


class TestTheTaskSplitFollowsTheTrainers:
    """A wrapper that reads the field must not have its task refused, and a
    wrapper that does not must."""

    def test_every_task_has_a_wrapper(self):
        assert set(_dispatched_wrappers()) == set(_TASKS)

    @pytest.mark.parametrize("task", _TASKS)
    def test_the_schema_list_matches_the_wrapper(self, task):
        wrappers = _dispatched_wrappers()
        if task == "preference":
            applies = all(_wrapper_reads_rope(wrappers[t]) for t in _PREFERENCE_DELEGATES)
        else:
            applies = _wrapper_reads_rope(wrappers[task])

        assert (task in ROPE_SCALING_APPLYING_TASKS) == applies, (
            f"task={task!r}: its trainer {'reads' if applies else 'does not read'} "
            "rope_scaling_type; move it in or out of ROPE_SCALING_APPLYING_TASKS"
        )


def _setup_call(body):
    """The ``self._setup_*`` method a one-statement branch calls, or None."""
    if len(body) != 1 or not isinstance(body[0], ast.Expr):
        return None
    call = body[0].value
    if (
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr.startswith("_setup_")
    ):
        return call.func.attr
    return None


def _setup_dispatch_order(wrapper):
    """The ``self._setup_*`` methods of the if/else chain in ``wrapper.setup``
    that picks the load path, in order."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(wrapper.setup)))
    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or _setup_call(node.body) is None:
            continue
        order = []
        branch = node
        while True:
            order.append(_setup_call(branch.body))
            if len(branch.orelse) == 1 and isinstance(branch.orelse[0], ast.If):
                branch = branch.orelse[0]
                continue
            order.append(_setup_call(branch.orelse))
            break
        return order
    return []


class TestThePathSplitFollowsTheSetups:
    """Each refused setting routes to a setup that does not read the field; the
    allowed ones do. A setup that starts rescaling RoPE fails here until its
    setting is dropped from the refusal."""

    @pytest.mark.parametrize("wrapper", [SFTTrainerWrapper, PretrainTrainerWrapper])
    def test_the_transformers_setups_read_the_field(self, wrapper):
        """Guard on the scan itself: if it stopped seeing these reads, every
        path would look refusable and the tests below would still pass."""
        assert _reads_rope(inspect.getsource(wrapper._setup_transformers))

    @pytest.mark.parametrize(
        ("wrapper", "method"),
        [
            (SFTTrainerWrapper, "_setup_vision_transformers"),
            (SFTTrainerWrapper, "_setup_audio_transformers"),
            (SFTTrainerWrapper, "_setup_streaming_transformers"),
            (SFTTrainerWrapper, "_setup_unsloth"),
            (PretrainTrainerWrapper, "_setup_unsloth"),
        ],
    )
    def test_the_refused_setups_do_not_read_it(self, wrapper, method):
        assert not _reads_rope(inspect.getsource(getattr(wrapper, method))), (
            f"{wrapper.__name__}.{method} now reads rope_scaling_type: drop its "
            "setting from the #1697 refusal in _validate_rope_scaling_is_applied"
        )

    def test_the_mlx_trainer_does_not_read_it(self):
        assert not _reads_rope(inspect.getsource(MLXSFTTrainerWrapper)), (
            "MLXSFTTrainerWrapper now reads rope_scaling_type: drop backend='mlx' "
            "from the #1697 refusal in _validate_rope_scaling_is_applied"
        )

    def test_the_sft_message_follows_the_dispatch_order(self):
        """The refusal names settings in the order ``setup`` checks them (MLX is
        resolved before the wrapper is built, so it comes first)."""
        assert _setup_dispatch_order(SFTTrainerWrapper) == [
            "_setup_vision_transformers",
            "_setup_audio_transformers",
            "_setup_streaming_transformers",
            "_setup_unsloth",
            "_setup_transformers",
        ]

    def test_pretrain_has_only_the_two_setups(self):
        assert _setup_dispatch_order(PretrainTrainerWrapper) == [
            "_setup_unsloth",
            "_setup_transformers",
        ]
