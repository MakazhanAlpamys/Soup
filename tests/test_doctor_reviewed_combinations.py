"""Review coverage for the shipped recipes and the doctor config report."""

from __future__ import annotations

import contextlib
import io
import warnings

import pytest
from rich.console import Console

from soup_cli.config.backend_support import (
    IGNORED,
    REGISTRY,
    check_config,
    is_reviewed,
    unsupported_for,
)
from soup_cli.config.loader import load_config, load_config_from_string
from soup_cli.config.schema import resolve_mlx_task
from soup_cli.recipes.catalog import RECIPES
from tests.conftest import strip_ansi

#: Combinations a shipped recipe uses and nobody has reviewed yet, in the order the
#: issue proposes (recipe count). Each review removes its rows; nothing is added.
NOT_YET_REVIEWED = frozenset({
    ("grpo", "transformers", "text"),
    ("dpo", "transformers", "text"),
    ("embedding", "transformers", "text"),
    ("asr", "transformers", "text"),
    ("tts", "transformers", "audio_out"),
    ("pretrain", "transformers", "text"),
    ("kto", "transformers", "text"),
    ("orpo", "transformers", "text"),
    ("simpo", "transformers", "text"),
    ("reward_model", "transformers", "text"),
    ("ppo", "transformers", "text"),
    ("ipo", "transformers", "text"),
    ("online_dpo", "transformers", "text"),
    ("grpo", "transformers", "vision"),
    ("dpo", "transformers", "vision"),
})

_SFT_TEXT_KNOBS = {
    "training.dpo_beta": 0.05,
    "training.kto_beta": 0.2,
    "training.orpo_beta": 0.2,
    "training.simpo_gamma": 0.25,
    "training.cpo_alpha": 0.5,
    "training.ipo_tau": 0.2,
    "training.bco_beta": 0.2,
    "training.grpo_beta": 0.05,
    "training.num_generations": 6,
    "training.reward_fn": "format",
    "training.verifiable_domain": "math",
    "training.ppo_epochs": 2,
    "training.ppo_clip_ratio": 0.1,
    "training.ppo_kl_penalty": 0.025,
    "training.reward_model": "example/reward-model",
    "training.embedding_loss": "triplet",
    "training.embedding_margin": 0.25,
    "training.embedding_pooling": "cls",
    "training.embedding_temperature": 0.025,
}


@pytest.fixture(scope="module")
def recipe_combinations() -> dict[tuple[str, str, str], str]:
    """Every (task, backend, modality) a shipped recipe uses -> one recipe that uses it."""
    found: dict[tuple[str, str, str], str] = {}
    for name, meta in sorted(RECIPES.items()):
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cfg = load_config_from_string(meta.yaml_str)
        task = cfg.task
        if cfg.backend == "mlx":
            task = resolve_mlx_task(task, cfg.training.preference_loss)
        found.setdefault((task, cfg.backend, cfg.modality), name)
    return found


# ---------------------------------------------------------------------------
# The guard over the shipped recipes
# ---------------------------------------------------------------------------


def _assert_recipe_coverage(combos):
    assert len(combos) >= 19, combos  # a loop over nothing passes
    uncovered = sorted(
        f"{combo} (e.g. recipe {recipe!r})"
        for combo, recipe in combos.items()
        if not is_reviewed(*combo) and combo not in NOT_YET_REVIEWED
    )
    assert not uncovered, (
        "Shipped recipes use combinations with no backend_support.REGISTRY key: review "
        "the trainer's setup path and add a key (an empty tuple if nothing is ignored):\n  "
        + "\n  ".join(uncovered)
    )


def test_every_recipe_combination_is_reviewed_or_listed_as_waiting(recipe_combinations):
    _assert_recipe_coverage(recipe_combinations)


def test_the_waiting_list_only_shrinks(recipe_combinations):
    reviewed_now = sorted(combo for combo in NOT_YET_REVIEWED if is_reviewed(*combo))
    assert not reviewed_now, (
        f"These now have a REGISTRY key; remove them from NOT_YET_REVIEWED: {reviewed_now}"
    )
    unused = sorted(NOT_YET_REVIEWED - set(recipe_combinations))
    assert not unused, f"No shipped recipe uses these any more; drop them: {unused}"


def test_the_guard_can_fail(recipe_combinations, monkeypatch):
    combo = ("sft", "transformers", "text")
    monkeypatch.delitem(REGISTRY, combo)

    with pytest.raises(AssertionError) as failure:
        _assert_recipe_coverage(recipe_combinations)

    assert str(combo) in str(failure.value)
    assert recipe_combinations[combo] in str(failure.value)


# ---------------------------------------------------------------------------
# Reviewed versus not reviewed
# ---------------------------------------------------------------------------


def test_a_present_key_is_reviewed_even_when_empty():
    assert REGISTRY[("sft", "unsloth", "text")] == ()
    assert is_reviewed("sft", "unsloth", "text")
    assert unsupported_for("sft", "unsloth", "text") == ()


def test_an_absent_key_is_not_reviewed():
    assert not is_reviewed("grpo", "transformers", "text")
    assert unsupported_for("grpo", "transformers", "text") == ()


def test_the_mlx_and_unsloth_modality_fallback_counts_as_reviewed(monkeypatch):
    monkeypatch.delitem(REGISTRY, ("sft", "mlx", "vision"))
    assert is_reviewed("sft", "mlx", "vision")
    assert unsupported_for("sft", "mlx", "vision") is REGISTRY[("sft", "mlx", "text")]
    assert is_reviewed("sft", "unsloth", "audio")
    assert not is_reviewed("sft", "transformers", "audio_out")


# ---------------------------------------------------------------------------
# sft/transformers/text
# ---------------------------------------------------------------------------


@pytest.fixture()
def write_config(tmp_path):
    def _write(task: str, training: str = "  epochs: 1\n") -> str:
        (tmp_path / "train.jsonl").write_text('{"instruction": "a", "output": "b"}\n',
                                              encoding="utf-8")
        path = tmp_path / "soup.yaml"
        path.write_text(
            f"base: some-model\ntask: {task}\ndata:\n  train: {tmp_path / 'train.jsonl'}\n"
            f"  format: alpaca\ntraining:\n{training}",
            encoding="utf-8",
        )
        return str(path)

    return _write


def test_the_sft_text_table_holds_only_ignored_other_task_knobs():
    fields = [entry.field for entry in REGISTRY[("sft", "transformers", "text")]]
    assert sorted(fields) == sorted(_SFT_TEXT_KNOBS)
    assert {entry.status for entry in REGISTRY[("sft", "transformers", "text")]} == {IGNORED}


@pytest.mark.parametrize(("field", "value"), _SFT_TEXT_KNOBS.items())
def test_an_other_task_knob_set_on_sft_is_reported(field, value):
    name = field.split(".", 1)[1]
    with contextlib.redirect_stdout(io.StringIO()):
        cfg = load_config_from_string(
            f"base: m\ntask: sft\ndata:\n  train: d.jsonl\ntraining:\n  {name}: {value}\n"
        )
    assert [entry.field for entry in check_config(cfg)] == [field]


def test_a_plain_sft_config_reports_nothing():
    with contextlib.redirect_stdout(io.StringIO()):
        cfg = load_config_from_string("base: m\ntask: sft\ndata:\n  train: d.jsonl\n")
    assert check_config(cfg) == []


# ---------------------------------------------------------------------------
# The doctor leg
# ---------------------------------------------------------------------------


def _doctor(path: str, capsys, monkeypatch) -> str:
    import soup_cli.commands.doctor as doctor_module

    monkeypatch.setattr(doctor_module, "console", Console(width=200))
    doctor_module._report_config_support(load_config(path))
    return " ".join(strip_ansi(capsys.readouterr().out).split())


def test_doctor_says_an_unreviewed_combination_is_unreviewed(write_config, capsys, monkeypatch):
    out = _doctor(write_config("grpo"), capsys, monkeypatch)
    assert "task=grpo backend=transformers has not been reviewed" in out
    assert "None of the" not in out  # no all-clear for a table nobody wrote


def test_doctor_names_the_modality_of_an_unreviewed_combination(tmp_path, capsys, monkeypatch):
    path = tmp_path / "soup.yaml"
    path.write_text(RECIPES["minimax-m3-dpo"].yaml_str, encoding="utf-8")

    output = _doctor(str(path), capsys, monkeypatch)

    assert "task=dpo backend=transformers modality=vision has not been reviewed" in output
    assert "None of the" not in output


def test_doctor_recognises_an_effective_mlx_preference_task(tmp_path, capsys, monkeypatch):
    path = tmp_path / "soup.yaml"
    path.write_text(
        "base: local-test-model\ntask: preference\nbackend: mlx\n"
        "data:\n  train: train.jsonl\n  format: dpo\n"
        "training:\n  preference_loss: dpo\n",
        encoding="utf-8",
    )

    output = _doctor(str(path), capsys, monkeypatch)
    count = len(unsupported_for("dpo", "mlx"))
    assert count > 0
    assert f"None of the {count} setting(s) known to be unread on task=preference" in output
    assert "has not been reviewed" not in output


def test_doctor_names_an_other_task_knob_on_sft(write_config, capsys, monkeypatch):
    out = _doctor(write_config("sft", "  kto_beta: 0.3\n"), capsys, monkeypatch)
    assert "training.kto_beta" in out
    assert "trainer/kto.py" in out
    assert "has not been reviewed" not in out


def test_doctor_gives_the_reviewed_all_clear_on_a_plain_sft_config(
    write_config, capsys, monkeypatch
):
    out = _doctor(write_config("sft"), capsys, monkeypatch)
    count = len(unsupported_for("sft", "transformers"))
    assert count == 19
    assert "None of the 19 setting(s) known to be unread on task=sft backend=transformers" in out


def test_load_config_accepts_the_written_file(write_config):
    # Sanity for the fixture the doctor tests share.
    assert load_config(write_config("grpo")).task == "grpo"
