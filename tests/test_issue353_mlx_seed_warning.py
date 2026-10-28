"""#353's fourth acceptance criterion on MLX, and what replaced it.

#353 asked for "parse-time rejection (or a loud warning) when ``training.seed``
is set on a task that ignores it". MLX SFT read neither ``training.seed`` nor
``training.data_seed``, so it warned. #1850 seeds it for real (``mx.random``
before the adapter is built, numpy's global RNG before mlx-lm's ``train()``
shuffles), and that test said to remove the warning in the same commit. The
behaviour is pinned in tests/test_mlx_sft_seed.py (real mlx) and
tests/test_mlx_sft_seed_wiring.py (everywhere). What stays here needs no mlx:
a set seed / data_seed is no longer announced as ignored, and the
controls: the unsupported-feature list still fires, and says nothing about
seeds.
"""

from io import StringIO

import pytest


def _mlx_cfg(**training):
    from soup_cli.config.loader import load_config_from_string

    body = {"epochs": 1, "lr": 2e-4, "batch_size": 1}
    body.update(training)
    fields = ", ".join(f"{k}: {v}" for k, v in body.items())
    return load_config_from_string(
        "base: mlx-community/tiny\n"
        "task: sft\n"
        "backend: mlx\n"
        "data: {train: d.jsonl, format: chatml}\n"
        f"training: {{{fields}}}\n"
        "output: ./out\n"
    )


def _warnings_for(monkeypatch, **training):
    """Run the wrapper's own unsupported-feature check and return what it printed."""
    from rich.console import Console

    from soup_cli.trainer import mlx_sft

    buffer = StringIO()
    monkeypatch.setattr(mlx_sft, "console", Console(file=buffer, width=200))
    mlx_sft.MLXSFTTrainerWrapper(_mlx_cfg(**training))._check_unsupported()
    return buffer.getvalue()


class TestTheSeedIsNotAnnouncedAsIgnored:
    @pytest.mark.parametrize("field", ["seed", "data_seed"])
    def test_no_ignored_warning(self, monkeypatch, field):
        """MLX SFT honours both fields since #1850, so a run that sets one must
        not be told it is ignored."""
        out = _warnings_for(monkeypatch, **{field: 11})
        assert f"training.{field}" not in out

    def test_control_an_unseeded_run_says_nothing_about_seeds(self, monkeypatch):
        """CONTROL. A warning that fires on every MLX run is noise, and noise is
        how the next real one gets ignored."""
        out = _warnings_for(monkeypatch)
        assert "seed" not in out.lower()

    def test_control_the_existing_unsupported_warnings_still_fire(self, monkeypatch):
        """CONTROL. Removing the seed lines (#1850) must not have taken the rest
        of the list with them: GaLore and friends are still reported."""
        out = _warnings_for(monkeypatch, use_galore=True)
        assert "GaLore" in out
        assert "seed" not in out.lower()


@pytest.mark.parametrize("field", ["seed", "data_seed"])
def test_the_schema_still_accepts_the_field_on_mlx(field):
    """Accepted, and since #1850 honoured: a config that is valid on
    transformers stays loadable when the backend switches to MLX."""
    cfg = _mlx_cfg(**{field: 11})
    assert getattr(cfg.training, field) == 11
