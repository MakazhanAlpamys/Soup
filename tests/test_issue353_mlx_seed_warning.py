"""#353's fourth acceptance criterion, for the one backend that was unreachable
when the issue was written.

#353 asks for "parse-time rejection (or a loud warning) when ``training.seed``
is set on a task that ignores it". #381 threaded the seed through all eighteen
transformers task wrappers, and named the MLX backend as the one path that
reads neither ``training.seed`` nor ``training.data_seed``: ``mlx_sft`` /
``mlx_dpo`` / ``mlx_grpo`` reference no seed and no ``mx.random`` on any path.

That gap only became reachable *between* the issue being filed and #381
landing: ``backend: mlx`` was never dispatched at all until #362 (#363), so a
user could not previously get an MLX run to ignore anything. Now they can, and
silence is the failure mode #353 exists to remove — a seeded MLX run that is
not seeded looks exactly like a seeded one until two replicates disagree.

Seeding MLX itself (``mx.random.seed``) is separate work with its own RNG and
is deliberately NOT what this pins. What is pinned is that the run says so.
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


class TestTheSeedIsAnnouncedAsIgnored:
    def test_a_set_seed_is_named(self, monkeypatch):
        out = _warnings_for(monkeypatch, seed=7)
        assert "seed" in out
        # The user set `training.seed`; the message has to be greppable as that,
        # not as some paraphrase they cannot search for.
        assert "training.seed" in out

    def test_a_set_data_seed_is_named(self, monkeypatch):
        out = _warnings_for(monkeypatch, data_seed=3)
        assert "training.data_seed" in out

    def test_seed_zero_still_warns(self, monkeypatch):
        """0 is a real seed, not "unset". A truthiness check here would drop the
        one value a user picks precisely because it is conventional."""
        out = _warnings_for(monkeypatch, seed=0)
        assert "training.seed" in out

    def test_both_are_named_together(self, monkeypatch):
        out = _warnings_for(monkeypatch, seed=7, data_seed=3)
        assert "training.seed" in out
        assert "training.data_seed" in out


class TestItStaysQuietOtherwise:
    def test_control_an_unseeded_run_says_nothing_about_seeds(self, monkeypatch):
        """CONTROL. A warning that fires on every MLX run is noise, and noise is
        how the next real one gets ignored."""
        out = _warnings_for(monkeypatch)
        assert "seed" not in out.lower()

    def test_control_the_existing_unsupported_warnings_still_fire(self, monkeypatch):
        """CONTROL. The seed lines are appended to the same list that already
        reports GaLore and friends, so this pins that the list was extended
        rather than replaced."""
        out = _warnings_for(monkeypatch, use_galore=True)
        assert "GaLore" in out
        assert "seed" not in out.lower()


def test_the_mlx_wrappers_still_do_not_read_the_seed(monkeypatch):
    """The warning is only honest while it is true. If a future change seeds MLX
    for real, this fails and whoever did it removes the warning in the same
    commit instead of leaving a run lying about itself."""
    import inspect

    from soup_cli.trainer import mlx_dpo, mlx_grpo, mlx_sft

    for module in (mlx_sft, mlx_dpo, mlx_grpo):
        source = inspect.getsource(module)
        # `_check_unsupported` legitimately mentions the field it is warning
        # about, so the sources are checked for an actual RNG call instead.
        assert "mx.random.seed" not in source, (
            f"{module.__name__} now seeds MLX; drop the 'ignored' warning in "
            f"mlx_sft._check_unsupported and delete this test"
        )


@pytest.mark.parametrize("field", ["seed", "data_seed"])
def test_the_schema_still_accepts_the_field_on_mlx(field):
    """A warning, not a rejection: #353 allows either, and a config that is
    valid on transformers should not become unloadable by switching backend."""
    cfg = _mlx_cfg(**{field: 11})
    assert getattr(cfg.training, field) == 11
