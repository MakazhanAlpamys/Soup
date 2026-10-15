"""#1815 — ``soup doctor --config`` tells a reviewed combination from an unreviewed one.

``config/backend_support.py`` declared tables for 3 of the 19 (task, backend,
modality) combinations the shipped recipes use, and for the other 16 (147
recipes) ``soup doctor --config`` printed "None of the 0 setting(s) known to be
unread ... is switched on", an all-clear about a table nobody had written.

Now an absent REGISTRY key reads as "not reviewed" and doctor says so; a
present key with an empty tuple is "reviewed, nothing ignored". A guard computes
the combinations from every shipped recipe and fails on one that is neither
reviewed nor listed below as waiting, so a recipe on a new combination cannot
ship uncovered. The list only shrinks: a task family's review removes its rows.

This pull request reviews ``sft/transformers/text`` (81 recipes): thirteen knobs
of other tasks load on ``task: sft`` and no module on the SFT path reads them.
"""

from __future__ import annotations

import contextlib
import io
import warnings

import pytest
from rich.console import Console

from soup_cli.config.backend_support import (
    REGISTRY,
    check_config,
    is_reviewed,
    unsupported_for,
)
from soup_cli.config.loader import load_config, load_config_from_string
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

_SFT_TEXT_KNOBS = [
    "training.kto_beta", "training.orpo_beta", "training.simpo_gamma",
    "training.cpo_alpha", "training.ipo_tau", "training.bco_beta",
    "training.num_generations", "training.ppo_epochs", "training.ppo_clip_ratio",
    "training.ppo_kl_penalty", "training.embedding_margin",
    "training.embedding_pooling", "training.embedding_temperature",
]


def _recipe_combinations() -> dict[tuple[str, str, str], str]:
    """Every (task, backend, modality) a shipped recipe uses -> one recipe that uses it."""
    found: dict[tuple[str, str, str], str] = {}
    for name, meta in sorted(RECIPES.items()):
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cfg = load_config_from_string(meta.yaml_str)
        found.setdefault((cfg.task, cfg.backend, cfg.modality), name)
    return found


# ---------------------------------------------------------------------------
# The guard over the shipped recipes
# ---------------------------------------------------------------------------


def test_every_recipe_combination_is_reviewed_or_listed_as_waiting():
    combos = _recipe_combinations()
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


def test_the_waiting_list_only_shrinks():
    reviewed_now = sorted(combo for combo in NOT_YET_REVIEWED if is_reviewed(*combo))
    assert not reviewed_now, (
        f"These now have a REGISTRY key; remove them from NOT_YET_REVIEWED: {reviewed_now}"
    )
    unused = sorted(NOT_YET_REVIEWED - set(_recipe_combinations()))
    assert not unused, f"No shipped recipe uses these any more; drop them: {unused}"


def test_the_guard_can_fail(monkeypatch):
    # Deleting a reviewed key makes its recipes uncovered.
    from soup_cli.config import backend_support as bs

    monkeypatch.delitem(bs.REGISTRY, ("sft", "transformers", "text"))
    combos = _recipe_combinations()
    assert ("sft", "transformers", "text") in combos
    assert not is_reviewed("sft", "transformers", "text")


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


def test_the_mlx_and_unsloth_modality_fallback_counts_as_reviewed():
    assert is_reviewed("sft", "mlx", "vision")
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


def test_the_sft_text_table_holds_the_thirteen_other_task_knobs():
    fields = [entry.field for entry in REGISTRY[("sft", "transformers", "text")]]
    assert sorted(fields) == sorted(_SFT_TEXT_KNOBS)


@pytest.mark.parametrize("field", _SFT_TEXT_KNOBS)
def test_an_other_task_knob_set_on_sft_is_reported(field):
    from soup_cli.config.schema import TrainingConfig

    name = field.split(".", 1)[1]
    default = TrainingConfig.model_fields[name].get_default(call_default_factory=True)
    if name == "embedding_pooling":
        value = "cls"
    elif isinstance(default, float):
        value = default / 2 if default else 0.5  # inside every bound the schema sets
    else:
        value = (default or 0) + 1
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
    from soup_cli.commands.doctor import doctor

    monkeypatch.setattr(doctor_module, "console", Console(width=200))
    doctor(nccl=False, disk=False, config=path)
    return " ".join(strip_ansi(capsys.readouterr().out).split())


def test_doctor_says_an_unreviewed_combination_is_unreviewed(write_config, capsys, monkeypatch):
    out = _doctor(write_config("grpo"), capsys, monkeypatch)
    assert "task=grpo backend=transformers has not been reviewed" in out
    assert "None of the" not in out  # no all-clear for a table nobody wrote


def test_doctor_names_an_other_task_knob_on_sft(write_config, capsys, monkeypatch):
    out = _doctor(write_config("sft", "  kto_beta: 0.3\n"), capsys, monkeypatch)
    assert "training.kto_beta" in out
    assert "trainer/kto.py" in out
    assert "has not been reviewed" not in out


def test_doctor_gives_the_reviewed_all_clear_on_a_plain_sft_config(
    write_config, capsys, monkeypatch
):
    out = _doctor(write_config("sft"), capsys, monkeypatch)
    n = len(unsupported_for("sft", "transformers"))
    assert n == 13
    assert f"None of the {n} setting(s) known to be unread on task=sft backend=transformers" in out


def test_load_config_accepts_the_written_file(write_config):
    # Sanity for the fixture the doctor tests share.
    assert load_config(write_config("grpo")).task == "grpo"
