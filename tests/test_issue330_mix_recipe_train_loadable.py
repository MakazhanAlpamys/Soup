"""`soup data mix --optimize` writes a recipe `soup train` cannot load (issue #330).

`render_mix_recipe_yaml` emitted `data.train` as a YAML list, but `DataConfig.train`
was typed `str` — so the recipe it writes for a human to splice into `soup.yaml`
failed `load_config_from_string` with `data -> train: Input should be a valid
string`. #330's fix collapsed `data.train` to the single highest-weighted dataset
from the search, since there was no training-time reader for a real mixture yet.

#443 wired `data.interleave` into `load_dataset()` and widened `DataConfig.train`
to accept a list, so this module's renderer went back to emitting the full
dataset list (the shape #330 originally had to move away from) — it is now a
config `soup train` can actually load. The tests below were updated in place
(renamed where the assertion itself flipped) rather than deleted, since the
schema-loadability guarantee they pin still matters, just against the new shape.
"""

from __future__ import annotations

from soup_cli.config.loader import load_config_from_string
from soup_cli.utils.data_mix import MixOptimizationReport, render_mix_recipe_yaml


def _make_files(tmp_path, names):
    for n in names:
        (tmp_path / n).write_text("{}\n")


def _report(datasets, weights, tmp_path) -> MixOptimizationReport:
    return MixOptimizationReport(
        datasets=tuple(str(tmp_path / d) for d in datasets),
        candidates=(),
        best_weights=tuple(weights),
        best_eval_loss=0.123,
        partial=False,
        elapsed_seconds=10.0,
    )


def _splice_into_config(data_block_text: str) -> str:
    return "base: test-base\ntask: sft\n" + data_block_text


def test_rendered_recipe_loads_through_config_schema(tmp_path):
    report = _report(["a.jsonl", "b.jsonl"], [0.7, 0.3], tmp_path)
    text = render_mix_recipe_yaml(report)
    # Strip the leading comment lines — only the `data:` block onward is
    # spliced into a real soup.yaml, matching how a human would use it.
    data_block = text[text.index("data:"):]
    cfg = load_config_from_string(_splice_into_config(data_block))
    assert cfg.data.train == [str(tmp_path / "a.jsonl"), str(tmp_path / "b.jsonl")]


def test_train_contains_all_datasets_in_report_order(tmp_path):
    # #443 superseded #330/#442's single-best-dataset collapse now that the
    # trainer can consume a real mixture — this test's old name/assertion
    # ("highest weighted dataset only") is intentionally reversed, not
    # deleted: the full dataset list must survive, in report.datasets order,
    # index-aligned with data.interleave.probs.
    report = _report(
        ["a.jsonl", "b.jsonl", "c.jsonl"], [0.2, 0.55, 0.25], tmp_path
    )
    text = render_mix_recipe_yaml(report)
    data_block = text[text.index("data:"):]
    cfg = load_config_from_string(_splice_into_config(data_block))
    assert cfg.data.train == [
        str(tmp_path / "a.jsonl"),
        str(tmp_path / "b.jsonl"),
        str(tmp_path / "c.jsonl"),
    ]
    assert cfg.data.interleave == {"strategy": "probs", "probs": [0.2, 0.55, 0.25]}


def test_full_ranked_breakdown_still_in_comments(tmp_path):
    # The human-review value of the original recipe (every dataset + its
    # weight) must survive collapsing `data.train` to one path.
    report = _report(["a.jsonl", "b.jsonl"], [0.7, 0.3], tmp_path)
    text = render_mix_recipe_yaml(report)
    assert "0.700000" in text
    assert "0.300000" in text
    assert str(tmp_path / "a.jsonl") in text
    assert str(tmp_path / "b.jsonl") in text


def test_apply_cli_prints_dataset_list_shape_recipe(tmp_path, monkeypatch):
    # #443 flipped the canonical `--optimize` output shape: `train:` is now
    # followed by a YAML list ("-" markers), not a single quoted string —
    # the reverse of what this test asserted pre-#443.
    _make_files(tmp_path, ["a.jsonl", "b.jsonl"])
    monkeypatch.chdir(tmp_path)
    from typer.testing import CliRunner

    from soup_cli.cli import app

    # Force a wide terminal — the datasets resolve to absolute paths under
    # tmp_path, long enough that Rich's default wrapping would otherwise
    # break a path across the 200-char window this test inspects.
    runner = CliRunner(env={"COLUMNS": "300"})
    result = runner.invoke(
        app,
        [
            "data", "mix",
            "--optimize",
            "--datasets", "a.jsonl,b.jsonl",
            "--budget", "60s",
            "--num-probes", "2",
            "--output", "rec.yaml",
        ],
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))
    result = runner.invoke(app, ["data", "mix", "--apply", "rec.yaml"])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    # Rich wraps long lines to the console width, so compare with whitespace
    # collapsed rather than by exact line — "train:" must be followed by a
    # "-" list marker (the new shape), not directly by a path.
    compact = "".join(result.output.split())
    assert "train:" in compact, result.output
    after = compact[compact.index("train:") + len("train:"):]
    assert after.startswith("-"), result.output
    assert "a.jsonl" in after[:400], result.output
    assert "b.jsonl" in after[:400], result.output


def test_render_recipe_rejects_empty_weights(tmp_path):
    report = _report([], [], tmp_path)
    try:
        render_mix_recipe_yaml(report)
    except ValueError as exc:
        assert "best_weights" in str(exc)
    else:
        raise AssertionError("expected ValueError for empty best_weights")


def test_render_recipe_rejects_mismatched_weights_length(tmp_path):
    report = _report(["a.jsonl", "b.jsonl"], [0.5, 0.3, 0.2], tmp_path)
    try:
        render_mix_recipe_yaml(report)
    except ValueError as exc:
        assert "best_weights" in str(exc)
    else:
        raise AssertionError("expected ValueError for mismatched lengths")


def test_apply_cli_quotes_path_needing_quoting(tmp_path, monkeypatch):
    # Maintainer's repro: an unquoted `train: odd: name.jsonl` line is not
    # valid YAML when pasted back — the --apply echo must quote it the same
    # way the renderer does. Exercises the legacy single-string on-disk
    # shape (pre-#443, or a search over exactly one dataset) — #443's
    # renderer no longer emits single-string form for >= 2 datasets, but
    # --apply must still round-trip old recipe files correctly.
    import yaml

    monkeypatch.chdir(tmp_path)
    (tmp_path / "odd.yaml").write_text(
        'data:\n'
        '  interleave:\n'
        '    strategy: probs\n'
        '    probs:\n'
        '      - 1.000000\n'
        '  train: "odd: name.jsonl"\n'
    )
    from typer.testing import CliRunner

    from soup_cli.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["data", "mix", "--apply", "odd.yaml"])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    data_block = result.output[result.output.index("data:"):]
    loaded = yaml.safe_load(data_block)
    assert loaded["data"]["train"] == "odd: name.jsonl"


def test_apply_handles_pre_fix_list_shaped_recipe(tmp_path, monkeypatch):
    # A recipe written by the pre-#330-fix code (data.train as a YAML list,
    # by accident — there was no training-time reader yet) must still print
    # via --apply rather than crash. This happens to be byte-identical to
    # #443's new intentional list shape from the CLI's point of view (both
    # go through the "else" branch in commands/data_mix.py), but the two
    # are conceptually distinct on-disk provenances, hence keeping this
    # test separate from test_apply_handles_new_multi_dataset_recipe below.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "old.yaml").write_text(
        "data:\n"
        "  interleave:\n"
        "    strategy: probs\n"
        "    probs:\n"
        "      - 0.6\n"
        "      - 0.4\n"
        "  train:\n"
        "    - a.jsonl\n"
        "    - b.jsonl\n"
    )
    from typer.testing import CliRunner

    from soup_cli.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["data", "mix", "--apply", "old.yaml"])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "a.jsonl" in result.output
    assert "b.jsonl" in result.output


def test_apply_handles_new_multi_dataset_recipe(tmp_path, monkeypatch):
    # #443's canonical current shape: write a real recipe via --optimize
    # (so it goes through render_mix_recipe_yaml, not a hand-written
    # fixture), then confirm --apply round-trips data.train as a list and
    # data.interleave.probs stays index-aligned with it.
    _make_files(tmp_path, ["a.jsonl", "b.jsonl", "c.jsonl"])
    monkeypatch.chdir(tmp_path)
    from typer.testing import CliRunner

    from soup_cli.cli import app

    # Force a wide terminal — tmp_path is long enough on this OS that Rich's
    # default-width line wrapping would otherwise break the path across
    # lines mid-token, which yaml.safe_load below can't parse.
    runner = CliRunner(env={"COLUMNS": "300"})
    result = runner.invoke(
        app,
        [
            "data", "mix",
            "--optimize",
            "--datasets", "a.jsonl,b.jsonl,c.jsonl",
            "--budget", "60s",
            "--num-probes", "2",
            "--output", "rec3.yaml",
        ],
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))
    result = runner.invoke(app, ["data", "mix", "--apply", "rec3.yaml"])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    data_block = result.output[result.output.index("data:"):]
    from pathlib import Path

    import yaml

    loaded = yaml.safe_load(data_block)
    assert isinstance(loaded["data"]["train"], list)
    # --optimize resolves --datasets to absolute paths under tmp_path.
    basenames = {Path(p).name for p in loaded["data"]["train"]}
    assert basenames == {"a.jsonl", "b.jsonl", "c.jsonl"}
    assert len(loaded["data"]["interleave"]["probs"]) == 3
