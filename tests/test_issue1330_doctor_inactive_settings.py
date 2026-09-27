"""#1330 — the doctor reports a written field only when its value is *on*.

``check_config`` keyed its rows on ``model_fields_set`` alone, so a setting
written in its off position (``loss_watchdog: false``) reported exactly like
one switched on. Because the schema refuses the ``true`` value of the three
``REJECTED`` rows at load, every "rejected" row the doctor could print was a
false positive, and a config written out in full by ``autopilot``'s
``write_yaml`` reloaded with 15 rows where the minimal original had 0.

No test here needs a GPU, MLX, the network, or a model download.
"""

from __future__ import annotations

import textwrap

import pytest

from soup_cli.config.backend_support import REJECTED, check_config, unsupported_for
from soup_cli.config.loader import load_config
from tests.test_issue755_backend_support_registry import _yaml_block


@pytest.fixture
def config_at(tmp_path):
    """A local soup.yaml builder (same shape as the #755 fixture).

    pytest 9 refuses calling a fixture directly, so the builder is repeated
    here rather than imported through the #755 fixture.
    """

    def _make(task: str, backend: str, training: str = "") -> str:
        train_file = tmp_path / "train.jsonl"
        train_file.write_text('{"instruction": "a", "output": "b"}\n', encoding="utf-8")
        body = textwrap.dedent(
            f"""\
            base: some-model
            task: {task}
            backend: {backend}
            data:
              train: {train_file}
              format: alpaca
            """
        )
        body += "training:\n" + _yaml_block(training or "epochs: 1")
        body += f"output: {tmp_path / 'out'}\n"
        path = tmp_path / "soup.yaml"
        path.write_text(body, encoding="utf-8")
        return str(path)

    return _make


#: Acceptance criterion 1: writing these in their off position is not a gap.
_OFF_POSITION_FIELDS = (
    "loss_watchdog",
    "loss_spike_recovery",
    "grad_accum_auto_tune",
    "quantization_aware",
    "use_liger",
)


class TestOffPositionWritesAreNotGaps:
    def test_five_off_position_writes_get_the_all_clear(self, config_at):
        cfg = load_config(
            config_at(
                "sft",
                "mlx",
                "\n".join(f"  {name}: false" for name in _OFF_POSITION_FIELDS),
            )
        )
        assert check_config(cfg) == []


class TestWriteYamlRoundTripReportsTheSame:
    """Acceptance criterion 2: a full dump must report like the minimal one."""

    @pytest.mark.parametrize(
        "minimal",
        ["", "  seed: 42", "  use_liger: true", "  neftune_alpha: 5"],
    )
    def test_round_tripped_config_reports_the_same_rows(
        self, tmp_path, config_at, minimal
    ):
        from soup_cli.autopilot.generate_config import write_yaml

        cfg = load_config(config_at("sft", "mlx", minimal))
        expected = sorted(e.field for e in check_config(cfg))

        dumped = tmp_path / "roundtrip.yaml"
        write_yaml(cfg, dumped)
        reloaded = load_config(str(dumped))
        # The full dump writes every default-off field the registry tracks...
        written = reloaded.training.model_fields_set
        assert written >= {name for name in _OFF_POSITION_FIELDS if name in written}
        # ...yet reports exactly what the minimal config did.
        assert sorted(e.field for e in check_config(reloaded)) == expected


class TestZeroValuesStayActive:
    def test_seed_zero_is_still_reported(self, config_at):
        """Acceptance criterion 3: 0 is a real seed, not "off"."""
        cfg = load_config(config_at("sft", "mlx", "  seed: 0"))
        reported = {e.field for e in check_config(cfg)}
        assert "training.seed" in reported


class TestRejectedRowsCannotFireOnALoadableConfig:
    """Acceptance criterion 4: each REJECTED row either fires on a loadable
    config or its active value is refused at load (here: the latter, so the
    rows stay as documentation of why the value cannot be used on MLX)."""

    @pytest.mark.parametrize(
        "entry",
        [e for e in unsupported_for("sft", "mlx") if e.status == REJECTED],
        ids=lambda entry: entry.field,
    )
    def test_active_value_is_refused_at_load(self, entry):
        from soup_cli.config.schema import SoupConfig

        name = entry.field.split(".", 1)[1]
        with pytest.raises(ValueError, match=rf"\b{name}\b"):
            SoupConfig(
                base="sshleifer/tiny-gpt2",
                task="sft",
                backend="mlx",
                data={"train": "train.jsonl"},
                training={name: True},
            )
