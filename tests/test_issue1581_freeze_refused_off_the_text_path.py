"""#1581 — ``freeze_layers`` / ``freeze_ratio`` are refused on the ``sft`` paths that ignore them.

#1532 refused the two fields on every task but ``sft`` and ``tts``. Inside those
two tasks only ``SFTTrainerWrapper._setup_transformers`` reads them, so a run on
``backend: mlx``, ``modality: vision``, ``modality: audio``,
``training.stream_layers: true`` or ``backend: unsloth`` (the last one also on
``tts``) loaded, trained every layer and said nothing. Those configs now stop at
load, naming the field, the settings that leave the text path and what to
change. :class:`TestTheRefusalFollowsTheSetups` ties each refused setting to the
setup it routes to, so wiring a real freeze into one of them has to drop it here.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest
import yaml

from soup_cli.config.loader import load_config_from_string
from soup_cli.trainer.mlx_sft import MLXSFTTrainerWrapper
from soup_cli.trainer.sft import SFTTrainerWrapper

_ONE_FIELD = [("freeze_layers", 2), ("freeze_ratio", 0.5)]


def _yaml(task="sft", top=None, **training):
    raw = {"base": "org/m", "task": task, **(top or {})}
    raw["data"] = {"train": "./x.jsonl"}
    if task == "tts":
        raw.setdefault("modality", "audio_out")
        training = {"tts_family": "orpheus", **training}
    raw["training"] = training
    return yaml.safe_dump(raw)


def _refusal(text):
    with pytest.raises(ValueError) as info:
        load_config_from_string(text)
    return " ".join(str(info.value).split())


# The six cases from the issue: (task, top-level keys, training keys, the
# settings the message names, the change it asks for).
_CASES = {
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
}


class TestRefusedOffTheTextPath:
    @pytest.mark.parametrize("case", list(_CASES))
    @pytest.mark.parametrize(("field", "value"), _ONE_FIELD)
    def test_the_message_names_the_field_the_path_and_the_fix(self, case, field, value):
        task, top, training, settings, change = _CASES[case]

        message = _refusal(_yaml(task, top, **training, **{field: value}))

        assert (
            f"training.{field} is not applied by task={task!r} with {settings}:"
        ) in message, message
        assert "frozen only on the text path with backend: transformers" in message
        assert "would train every layer" in message
        assert f"Use {change}, or remove the key." in message

    @pytest.mark.parametrize("case", list(_CASES))
    def test_without_the_fields_the_config_still_loads(self, case):
        """The refusal is about the two fields, not about the path."""
        task, top, training, _, _ = _CASES[case]

        cfg = load_config_from_string(_yaml(task, top, **training))

        assert cfg.training.freeze_layers is None
        assert cfg.training.freeze_ratio is None

    def test_both_fields_are_named_in_one_refusal(self):
        message = _refusal(
            _yaml("sft", {"backend": "unsloth"}, freeze_layers=2, freeze_ratio=0.5)
        )

        assert (
            "training.freeze_layers and training.freeze_ratio are not applied by "
            "task='sft' with backend='unsloth'"
        ) in message, message
        assert "or remove both keys." in message

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
        message = _refusal(_yaml("sft", top, freeze_layers=2))

        assert f"with {settings}:" in message, message
        assert f"Use {change}, or remove the key." in message

    def test_the_issue_reproduction_is_refused(self):
        """The issue's vision config."""
        message = _refusal(
            "base: Qwen/Qwen2-VL-2B-Instruct\ntask: sft\nmodality: vision\n"
            "data: {train: data.jsonl, format: chatml}\ntraining: {freeze_layers: 2}\n"
        )

        assert "with modality='vision'" in message, message

    def test_a_task_outside_sft_keeps_the_1497_message(self):
        """``dpo`` on unsloth is refused for the task, not for the path."""
        message = _refusal(_yaml("dpo", {"backend": "unsloth"}, freeze_layers=2))

        assert "is not applied by task='dpo': only task='sft'" in message, message
        assert "Use task: sft, or remove the key." in message


class TestTheTextPathStillLoads:
    # Written out rather than derived from _CASES, so a bad case table cannot
    # also remove its control.
    @pytest.mark.parametrize(("field", "value"), _ONE_FIELD)
    @pytest.mark.parametrize("task", ["sft", "tts"])
    def test_on_transformers(self, task, field, value):
        cfg = load_config_from_string(
            _yaml(task, {"backend": "transformers"}, **{field: value})
        )

        assert getattr(cfg.training, field) == value

    @pytest.mark.parametrize("r", [0, 8])
    def test_with_and_without_lora(self, r):
        """#341: ``freeze_layers`` stays legal with full fine-tuning (``r: 0``)."""
        cfg = load_config_from_string(
            _yaml("sft", freeze_layers=1, quantization="none", lora={"r": r})
        )

        assert cfg.training.freeze_layers == 1

    def test_stream_layers_false_is_not_streaming(self):
        cfg = load_config_from_string(_yaml("sft", stream_layers=False, freeze_layers=2))

        assert cfg.training.freeze_layers == 2

    def test_a_dumped_config_still_reloads(self):
        """A dumped vision config writes both fields out as null; that is not a
        request."""
        cfg = load_config_from_string(_yaml("sft", {"modality": "vision"}))
        dumped = yaml.safe_dump(cfg.model_dump(mode="json"))

        assert "freeze_layers: null" in dumped
        assert load_config_from_string(dumped).modality == "vision"


class TestSoupDoctorReportsIt:
    def test_doctor_exits_2_naming_the_path(self, tmp_path):
        from typer.testing import CliRunner

        from soup_cli.cli import app
        from tests.conftest import strip_ansi

        path = tmp_path / "soup.yaml"
        path.write_text(_yaml("sft", {"backend": "mlx"}, freeze_layers=2), encoding="utf-8")

        result = CliRunner().invoke(app, ["doctor", "--config", str(path)])

        assert result.exit_code == 2, result.output
        out = " ".join(strip_ansi(result.output).split())
        assert "training.freeze_layers is not applied by task='sft' with backend='mlx'" in out


def _reads_the_freeze_fields(source):
    """True when ``source`` reads either field as an attribute. Comments and
    docstrings do not count. Like the #1497 scan, a read through a helper
    outside the scanned code, or through ``getattr``, is not seen."""
    for node in ast.walk(ast.parse(textwrap.dedent(source))):
        if isinstance(node, ast.Attribute) and node.attr in ("freeze_layers", "freeze_ratio"):
            return True
    return False


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


def _setup_dispatch_order():
    """The ``self._setup_*`` methods of the if/elif chain in
    ``SFTTrainerWrapper.setup`` that picks the load path, in order."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(SFTTrainerWrapper.setup)))
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
        if len(order) > 2:
            return order
    return []


class TestTheRefusalFollowsTheSetups:
    """Each refused setting routes to a setup that does not read the fields; the
    one allowed path does. A setup that starts applying a real freeze fails here
    until its setting is dropped from the refusal."""

    def test_the_text_path_reads_the_fields(self):
        """Guard on the scan itself: if it stopped seeing this read, every path
        would look refusable and the tests below would still pass."""
        assert _reads_the_freeze_fields(inspect.getsource(SFTTrainerWrapper._setup_transformers))

    @pytest.mark.parametrize(
        "method",
        [
            "_setup_vision_transformers",
            "_setup_audio_transformers",
            "_setup_streaming_transformers",
            "_setup_unsloth",
        ],
    )
    def test_the_refused_setups_do_not_read_them(self, method):
        source = inspect.getsource(getattr(SFTTrainerWrapper, method))

        assert not _reads_the_freeze_fields(source), (
            f"{method} now reads freeze_layers / freeze_ratio: drop its setting "
            "from the #1581 refusal in _validate_freeze_is_applied"
        )

    def test_the_mlx_trainer_does_not_read_them(self):
        assert not _reads_the_freeze_fields(inspect.getsource(MLXSFTTrainerWrapper)), (
            "MLXSFTTrainerWrapper now reads freeze_layers / freeze_ratio: drop "
            "backend='mlx' from the #1581 refusal in _validate_freeze_is_applied"
        )

    def test_the_message_follows_the_dispatch_order(self):
        """The refusal names settings in the order ``setup`` checks them (MLX is
        resolved before the wrapper is built, so it comes first)."""
        assert _setup_dispatch_order() == [
            "_setup_vision_transformers",
            "_setup_audio_transformers",
            "_setup_streaming_transformers",
            "_setup_unsloth",
            "_setup_transformers",
        ]
