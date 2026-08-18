"""`soup data mix --optimize` writes a recipe `soup train` cannot load (issue #330).

`render_mix_recipe_yaml` emitted `data.train` as a YAML list, but `DataConfig.train`
is typed `str` — so the recipe it writes for a human to splice into `soup.yaml`
failed `load_config_from_string` with `data -> train: Input should be a valid
string`. `data.train` now renders as the single highest-weighted dataset from the
search; `data.interleave` is unchanged (it already round-trips fine — the bug was
only ever `train`).
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
    assert cfg.data.train == str(tmp_path / "a.jsonl")


def test_train_is_the_highest_weighted_dataset(tmp_path):
    report = _report(
        ["a.jsonl", "b.jsonl", "c.jsonl"], [0.2, 0.55, 0.25], tmp_path
    )
    text = render_mix_recipe_yaml(report)
    data_block = text[text.index("data:"):]
    cfg = load_config_from_string(_splice_into_config(data_block))
    assert cfg.data.train == str(tmp_path / "b.jsonl")


def test_full_ranked_breakdown_still_in_comments(tmp_path):
    # The human-review value of the original recipe (every dataset + its
    # weight) must survive collapsing `data.train` to one path.
    report = _report(["a.jsonl", "b.jsonl"], [0.7, 0.3], tmp_path)
    text = render_mix_recipe_yaml(report)
    assert "0.700000" in text
    assert "0.300000" in text
    assert str(tmp_path / "a.jsonl") in text
    assert str(tmp_path / "b.jsonl") in text


def test_apply_cli_prints_new_string_shape_recipe(tmp_path, monkeypatch):
    _make_files(tmp_path, ["a.jsonl", "b.jsonl"])
    monkeypatch.chdir(tmp_path)
    from typer.testing import CliRunner

    from soup_cli.cli import app

    runner = CliRunner()
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
    # collapsed rather than by exact line — "train:" must be followed
    # directly by the path, not by a "-" list marker (the old shape).
    compact = "".join(result.output.split())
    assert "train:" in compact, result.output
    after = compact[compact.index("train:") + len("train:"):]
    assert not after.startswith("-"), result.output
    assert "a.jsonl" in after[:200], result.output


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
    # way the renderer does.
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
    # A recipe written by the pre-fix code (data.train as a YAML list) must
    # still print via --apply rather than crash — old files on disk survive
    # the fix.
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
