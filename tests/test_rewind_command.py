"""`soup rewind` — spike detection, row ranking, previews, and the command itself."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.monitoring.rewind_log import RewindLog, read_rewind_log

# ---------------------------------------------------------------- fixtures --

# The poisoned row: 8.0 loss over 3000 supervised tokens against three
# ordinary rows of 1.0 over 100. Chosen so that a share computed WITHOUT
# tokens (8/11 = 0.73) fails the > 0.9 assertion below, and one computed
# with them (24000/24300 = 0.988) passes.
SPIKE_ROWS = [10, 11, 12, 13]
SPIKE_LOSS = [1.0, 1.0, 1.0, 8.0]
SPIKE_TOKENS = [100, 100, 100, 3000]


def _build(path: Path, entries, *, fingerprint: str = "fp0", task: str = "sft"):
    """Write `entries` (step, micro, rows, losses, tokens) and read them back."""
    log = RewindLog(
        path,
        backend="transformers",
        task=task,
        n_rows=64,
        batch_size=4,
        grad_accum=1,
        dataset_fingerprint=fingerprint,
    )
    for step, micro, rows, losses, tokens in entries:
        log.record_batch(
            step=step, micro=micro, rows=rows, row_loss=losses, row_tokens=tokens
        )
    log.close()
    assert log.dropped == 0, "fixture batches must all be recorded"
    return read_rewind_log(path)


def _run_entries(*, n=60, spike_step=41, nan_step=None, unmeasured=()):
    """A flat 1.0-loss run with one loaded spike step, optional nan / blank steps."""
    entries = []
    for step in range(1, n + 1):
        if step in unmeasured:
            # Recorded, but zero supervised tokens -> step_loss() is None.
            entries.append((step, 0, [step], [1.0], [0]))
        elif nan_step is not None and step == nan_step:
            entries.append((step, 0, [step], [math.nan], [10]))
        elif step == spike_step:
            entries.append((step, 0, SPIKE_ROWS, SPIKE_LOSS, SPIKE_TOKENS))
        else:
            entries.append((step, 0, [step], [1.0], [10]))
    return entries


@pytest.fixture
def plain_console(monkeypatch):
    """Wide, un-forced console so Rich neither wraps nor injects colour.

    The environment variables alone are not enough: ``commands/rewind.py``
    builds its module-level ``Console`` at import, so whichever test imports it
    first fixes its width. In a full-suite run that was a default-width
    console, and a long tmp path wrapped mid-assertion. The console itself is
    replaced for the duration of the test.
    """
    from rich.console import Console

    import soup_cli.commands.rewind as rewind_cmd

    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.delenv("CLICOLOR_FORCE", raising=False)
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setattr(
        rewind_cmd,
        "console",
        Console(force_terminal=False, no_color=True, width=10_000, soft_wrap=True),
    )


def _patch_tracker(monkeypatch, target):
    import soup_cli.experiment.tracker as trk

    monkeypatch.setattr(trk.ExperimentTracker, "__init__", lambda self, *a, **k: None)
    monkeypatch.setattr(
        trk.ExperimentTracker, "list_runs", lambda self, limit=50: [target]
    )
    monkeypatch.setattr(trk.ExperimentTracker, "get_run", lambda self, rid: target)


def _invoke(args):
    from soup_cli.cli import app

    return CliRunner().invoke(app, args)


# ------------------------------------------------------------ find_spikes --


def test_single_loaded_step_is_the_only_spike(tmp_path):
    from soup_cli.utils.rewind import find_spikes

    run = _build(tmp_path / RewindLog.FILENAME, _run_entries())
    spikes = find_spikes(run)
    assert [s.step for s in spikes] == [41]
    assert spikes[0].severity == "warning"
    assert spikes[0].baseline == pytest.approx(1.0)
    # 24300 weighted loss over 3300 tokens = 7.3636..., against a 1.0 baseline.
    assert spikes[0].ratio == pytest.approx(24300 / 3300)


def test_ratio_is_exactly_loss_over_baseline(tmp_path):
    """A 5.0-against-1.0 step reports ratio 5.0, not some rescaling of it."""
    from soup_cli.utils.rewind import find_spikes

    entries = _run_entries()
    entries[40] = (41, 0, [41], [5.0], [10])
    run = _build(tmp_path / RewindLog.FILENAME, entries)
    spikes = find_spikes(run)
    assert [(s.step, s.ratio, s.severity) for s in spikes] == [(41, 5.0, "warning")]


def test_an_unmeasured_step_is_never_itself_a_spike(tmp_path):
    from soup_cli.utils.rewind import find_spikes

    run = _build(tmp_path / RewindLog.FILENAME, _run_entries(unmeasured={20, 21}))
    assert [s.step for s in find_spikes(run)] == [41]


def test_unmeasured_steps_never_enter_the_baseline(tmp_path):
    """Three real steps, then twenty blank ones, then a 1.5.

    Skipping the unmeasured steps leaves a baseline of [1, 1, 1] -> median 1.0,
    and 1.5 is not > 2 x 1.0, so nothing fires. Let the blanks into the window
    (as 0.0, or as None) and the median collapses to 0.0 (or the comparison
    explodes) and step 24 is reported as a spike it is not.
    """
    from soup_cli.utils.rewind import find_spikes

    entries = [(s, 0, [s], [1.0], [10]) for s in (1, 2, 3)]
    entries += [(s, 0, [s], [1.0], [0]) for s in range(4, 24)]
    entries.append((24, 0, [24], [1.5], [10]))
    run = _build(tmp_path / RewindLog.FILENAME, entries)
    assert find_spikes(run) == []


def test_non_finite_loss_is_a_critical_spike_listed_first(tmp_path):
    from soup_cli.utils.rewind import find_spikes

    run = _build(tmp_path / RewindLog.FILENAME, _run_entries(nan_step=50))
    spikes = find_spikes(run)
    assert [(s.step, s.severity) for s in spikes] == [(50, "critical"), (41, "warning")]
    assert math.isnan(spikes[0].loss)
    assert spikes[0].ratio == math.inf


def test_fewer_than_three_baseline_steps_cannot_fire(tmp_path):
    from soup_cli.utils.rewind import find_spikes

    entries = [
        (1, 0, [1], [1.0], [10]),
        (2, 0, [2], [1.0], [10]),
        (3, 0, [3], [10.0], [10]),
    ]
    run = _build(tmp_path / RewindLog.FILENAME, entries)
    assert find_spikes(run) == []


def test_loss_exactly_at_the_factor_is_not_a_spike(tmp_path):
    """The threshold is strict: loss == factor x baseline stays quiet.

    Doubling is what a normal batch of harder rows does; the report is for
    steps that blow past it.
    """
    from soup_cli.utils.rewind import find_spikes

    entries = [(s, 0, [s], [1.0], [10]) for s in range(1, 11)]
    entries.append((11, 0, [11], [2.0], [10]))
    run = _build(tmp_path / RewindLog.FILENAME, entries)
    assert find_spikes(run) == []


# -------------------------------------------------------------- rank_rows --


def test_rank_rows_weights_loss_by_tokens(tmp_path):
    from soup_cli.utils.rewind import rank_rows

    run = _build(
        tmp_path / RewindLog.FILENAME, [(1, 0, SPIKE_ROWS, SPIKE_LOSS, SPIKE_TOKENS)]
    )
    ranked = rank_rows(run, 1)
    assert [r.row for r in ranked] == [13, 10, 11, 12]
    assert sum(r.share for r in ranked) == pytest.approx(1.0, abs=1e-9)
    assert ranked[0].row == 13
    assert ranked[0].share > 0.9
    assert ranked[0].tokens == 3000


def test_rank_rows_spans_every_micro_batch(tmp_path):
    from soup_cli.utils.rewind import rank_rows

    run = _build(
        tmp_path / RewindLog.FILENAME,
        [
            (1, 0, [10, 11], [1.0, 1.0], [100, 100]),
            (1, 1, [12, 13], [1.0, 8.0], [100, 3000]),
        ],
    )
    ranked = rank_rows(run, 1)
    assert [r.row for r in ranked] == [13, 10, 11, 12]
    assert sum(r.share for r in ranked) == pytest.approx(1.0, abs=1e-9)


def test_rank_rows_zero_token_step_has_zero_shares(tmp_path):
    from soup_cli.utils.rewind import rank_rows

    run = _build(tmp_path / RewindLog.FILENAME, [(1, 0, [0, 1], [1.0, 1.0], [0, 0])])
    ranked = rank_rows(run, 1)
    assert [r.share for r in ranked] == [0.0, 0.0]


# ------------------------------------------------------------ the command --


def test_missing_log_names_the_path_and_the_config_field(
    tmp_path, plain_console, monkeypatch
):
    _patch_tracker(
        monkeypatch, {"run_id": "r1", "output_dir": str(tmp_path), "status": "completed"}
    )
    result = _invoke(["rewind"])
    assert result.exit_code == 1, result.output
    assert str(tmp_path / RewindLog.FILENAME) in result.output
    assert "training.rewind_log" in result.output


def test_step_and_json_name_the_poisoned_row(tmp_path, plain_console, monkeypatch):
    _build(tmp_path / RewindLog.FILENAME, _run_entries())
    _patch_tracker(
        monkeypatch, {"run_id": "r1", "output_dir": str(tmp_path), "status": "completed"}
    )
    out = tmp_path / "out.json"
    result = _invoke(["rewind", "--step", "41", "--json", str(out)])
    assert result.exit_code == 0, result.output
    payload = json.loads(out.read_text())
    assert set(payload) == {"run_id", "step", "spikes", "rows"}
    assert payload["run_id"] == "r1"
    assert payload["step"] == 41
    assert payload["rows"][0]["row"] == 13
    assert payload["rows"][0]["share"] > 0.9
    assert payload["spikes"][0]["step"] == 41


def test_default_details_the_worst_spike(tmp_path, plain_console, monkeypatch):
    _build(tmp_path / RewindLog.FILENAME, _run_entries())
    _patch_tracker(
        monkeypatch, {"run_id": "r1", "output_dir": str(tmp_path), "status": "completed"}
    )
    out = tmp_path / "out.json"
    result = _invoke(["rewind", "--json", str(out)])
    assert result.exit_code == 0, result.output
    assert json.loads(out.read_text())["step"] == 41


def test_worst_spike_is_the_first_not_the_last(tmp_path, plain_console, monkeypatch):
    """Two spikes: the critical nan at 50 outranks the warning at 41."""
    _build(tmp_path / RewindLog.FILENAME, _run_entries(nan_step=50))
    _patch_tracker(
        monkeypatch, {"run_id": "r1", "output_dir": str(tmp_path), "status": "completed"}
    )
    out = tmp_path / "out.json"
    result = _invoke(["rewind", "--json", str(out)])
    assert result.exit_code == 0, result.output
    payload = json.loads(out.read_text())
    assert [s["step"] for s in payload["spikes"]] == [50, 41]
    assert payload["step"] == 50


def test_no_spikes_says_so(tmp_path, plain_console, monkeypatch):
    _build(
        tmp_path / RewindLog.FILENAME,
        [(s, 0, [s], [1.0], [10]) for s in range(1, 30)],
    )
    _patch_tracker(
        monkeypatch, {"run_id": "r1", "output_dir": str(tmp_path), "status": "completed"}
    )
    result = _invoke(["rewind"])
    assert result.exit_code == 0, result.output
    assert "No loss spikes recorded" in result.output


def test_unknown_step_exits_one(tmp_path, plain_console, monkeypatch):
    _build(tmp_path / RewindLog.FILENAME, _run_entries())
    _patch_tracker(
        monkeypatch, {"run_id": "r1", "output_dir": str(tmp_path), "status": "completed"}
    )
    result = _invoke(["rewind", "--step", "9999"])
    assert result.exit_code == 1, result.output
    assert "9999" in result.output


def test_no_output_dir_exits_one(tmp_path, plain_console, monkeypatch):
    _patch_tracker(monkeypatch, {"run_id": "r1", "output_dir": "", "status": "completed"})
    result = _invoke(["rewind"])
    assert result.exit_code == 1, result.output
    assert "output_dir" in result.output


def test_no_preview_never_loads_the_dataset(tmp_path, plain_console, monkeypatch):
    import soup_cli.utils.rewind as rewind_utils

    _build(tmp_path / RewindLog.FILENAME, _run_entries())
    _patch_tracker(
        monkeypatch,
        {
            "run_id": "r1",
            "output_dir": str(tmp_path),
            "status": "completed",
            "config_json": json.dumps(_chat_config(tmp_path)),
        },
    )

    def _boom(*a, **k):
        raise AssertionError("load_row_previews must not run under --no-preview")

    monkeypatch.setattr(rewind_utils, "load_row_previews", _boom)
    result = _invoke(["rewind", "--step", "41", "--no-preview"])
    assert result.exit_code == 0, result.output
    assert "Preview" not in result.output


# ------------------------------------------------------- previews / config --


CHAT_ROWS = [
    {
        "messages": [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "alpha reply"},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "two"},
            {"role": "assistant", "content": "beta reply"},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "three"},
            {"role": "assistant", "content": "gamma reply"},
        ]
    },
]


def _chat_config(tmp_path: Path) -> dict:
    """A validated SoupConfig dict whose data.train is a real 3-row JSONL file."""
    from soup_cli.config.schema import SoupConfig

    data_path = tmp_path / "train.jsonl"
    data_path.write_text("\n".join(json.dumps(r) for r in CHAT_ROWS) + "\n")
    # val_split 0: the default 0.1 carves int(3 * 0.9) = 2 train rows out of 3,
    # and the log's row ids index the post-split train list, not the file.
    cfg = SoupConfig.model_validate(
        {
            "base": "HuggingFaceTB/SmolLM2-135M",
            "task": "sft",
            "data": {"train": str(data_path), "val_split": 0.0},
        }
    )
    return cfg.model_dump(mode="json")


def _real_fingerprint(config: dict) -> str:
    from soup_cli.config.schema import SoupConfig
    from soup_cli.data.loader import load_dataset
    from soup_cli.monitoring.rewind_log import dataset_fingerprint

    return dataset_fingerprint(load_dataset(SoupConfig.model_validate(config).data)["train"])


def test_load_row_previews_happy_path(tmp_path):
    from soup_cli.utils.rewind import load_row_previews

    config = _chat_config(tmp_path)
    previews = load_row_previews(
        config, [0, 1, 2, 99], expected_fingerprint=_real_fingerprint(config)
    )
    assert previews is not None
    assert previews[0] == "alpha reply"
    assert previews[1] == "beta reply"
    assert previews[2] == "gamma reply"
    assert previews[99] == "<row not in dataset>"


def test_load_row_previews_truncates(tmp_path):
    from soup_cli.utils.rewind import load_row_previews

    config = _chat_config(tmp_path)
    previews = load_row_previews(
        config, [0], expected_fingerprint=_real_fingerprint(config), width=3
    )
    assert previews == {0: "alp…"}


def test_load_row_previews_refuses_a_different_dataset(tmp_path):
    from soup_cli.utils.rewind import load_row_previews

    config = _chat_config(tmp_path)
    assert load_row_previews(config, [0], expected_fingerprint="not-the-fingerprint") is None


def test_fingerprint_mismatch_withholds_previews_but_lists_rows(
    tmp_path, plain_console, monkeypatch
):
    config = _chat_config(tmp_path)
    _build(tmp_path / RewindLog.FILENAME, _run_entries(), fingerprint="stale-fingerprint")
    _patch_tracker(
        monkeypatch,
        {
            "run_id": "r1",
            "output_dir": str(tmp_path),
            "status": "completed",
            "config_json": json.dumps(config),
        },
    )
    out = tmp_path / "out.json"
    result = _invoke(["rewind", "--step", "41", "--json", str(out)])
    assert result.exit_code == 0, result.output
    assert "Preview withheld" in result.output
    assert json.loads(out.read_text())["rows"][0]["row"] == 13


def test_previews_reach_the_table(tmp_path, plain_console, monkeypatch):
    config = _chat_config(tmp_path)
    _build(
        tmp_path / RewindLog.FILENAME,
        [(s, 0, [s % 3], [1.0], [10]) for s in range(1, 41)]
        + [(41, 0, [0, 1, 2], [1.0, 1.0, 8.0], [100, 100, 3000])],
        fingerprint=_real_fingerprint(config),
    )
    _patch_tracker(
        monkeypatch,
        {
            "run_id": "r1",
            "output_dir": str(tmp_path),
            "status": "completed",
            "config_json": json.dumps(config),
        },
    )
    result = _invoke(["rewind", "--step", "41"])
    assert result.exit_code == 0, result.output
    assert "gamma reply" in result.output
    assert "Preview withheld" not in result.output


def test_help_exits_zero():
    result = _invoke(["rewind", "--help"])
    assert result.exit_code == 0, result.output


def test_a_crashed_run_without_output_dir_is_found_through_its_config(
    tmp_path, plain_console, monkeypatch
):
    """The tracker writes output_dir only in finish_run; a failed run has none."""
    monkeypatch.chdir(tmp_path)
    _build(tmp_path / "out_run" / RewindLog.FILENAME, _run_entries())
    _patch_tracker(
        monkeypatch,
        {
            "run_id": "r1",
            "output_dir": None,
            "status": "failed",
            "config_json": json.dumps({"output": "./out_run"}),
        },
    )
    out = tmp_path / "out.json"
    result = _invoke(["rewind", "--no-preview", "--json", str(out)])
    assert result.exit_code == 0, result.output
    assert json.loads(out.read_text())["step"] == 41


def test_experiment_name_layout_is_tried_before_the_bare_output(
    tmp_path, plain_console, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    _build(tmp_path / "out" / "exp1" / RewindLog.FILENAME, _run_entries(spike_step=41))
    _build(tmp_path / "out" / RewindLog.FILENAME, _run_entries(spike_step=30))
    _patch_tracker(
        monkeypatch,
        {
            "run_id": "r1",
            "output_dir": "",
            "status": "failed",
            "config_json": json.dumps({"output": "out", "experiment_name": "exp1"}),
        },
    )
    out = tmp_path / "out.json"
    result = _invoke(["rewind", "--no-preview", "--json", str(out)])
    assert result.exit_code == 0, result.output
    assert json.loads(out.read_text())["step"] == 41
