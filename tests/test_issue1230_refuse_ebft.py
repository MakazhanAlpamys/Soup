"""#1230 -- ``training.ebft_variant`` is refused at config load.

``ebft_variant`` wrapped the SFT trainer's ``compute_loss`` and added
``apply_ebft_loss(outputs.logits, inputs["labels"])`` to TRL's cross-entropy.
The kernel scores ``log_softmax(logits[:, t])`` against ``labels[:, t]`` -- the
position's own input token, with no causal shift -- so on the issue's two
hand-built predictors the term is 0 for one that copies its input and 30.6 for
a perfect next-token predictor: it rewarded copying. On the real trainer the
added term equalled the UNSHIFTED cross-entropy to 0.00e+00. Shifting it does
not rescue it: shifted, the term at temperature 1 IS the model's own
cross-entropy, so the loss would count cross-entropy twice.

The ruling is "refuse until the intended EBFT objective is implemented from its
reference". This file pins:

* the refusal, for every non-null value -- the two that used to load
  (``structured``, ``strided``) and every other one, which used to get a
  ``Literal`` error pointing at those two -- through the loader and through
  direct construction, ahead of the older task and backend gates;
* the message: it names the setting, says the variant is not yet a distinct
  objective, why (unshifted it rewards copying, shifted it duplicates the
  cross-entropy) and points at the issue;
* ``ebft_temperature`` on its own no longer tells the user to set the refused
  field;
* ``soup train`` and ``soup doctor --config`` stop at config load;
* the controls that must NOT change: null and unset still load, the EBFT hook
  stays a no-op for every config that loads, and GDPO -- which shares
  ``utils/ebft_gdpo.py`` -- loads, gates and computes exactly as before;
* every shipped recipe, template and example still loads;
* no doc still shows ``ebft_variant`` as a setting to copy.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer
from pydantic import ValidationError
from typer.testing import CliRunner

from soup_cli.config.loader import load_config_from_string
from soup_cli.config.schema import SoupConfig, TrainingConfig
from tests.conftest import strip_ansi

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The YAML spellings that loaded before this change: the field is a strict
#: ``Literal["structured", "strided"]``, so only those two strings, bare or
#: quoted, ever reached the trainer.
YAML_VALUES_THAT_LOADED = ["structured", "strided", "'structured'", '"strided"']

#: Every other non-null value. These were already refused, but by the
#: ``Literal`` check, whose message pointed at ``structured`` / ``strided`` --
#: a user following it walked straight into the refusal above.
YAML_OTHER_NON_NULL_VALUES = ["Structured", "STRIDED", "bogus", "''", "true", "1"]

#: The same property for a caller that builds the model directly (``soup
#: sweep``, ``soup rewind`` and autopilot construct it).
PYTHON_NON_NULL_VALUES = ["structured", "strided", "Structured", "bogus", "", True, 1]

#: Spellings of "no variant". They must keep loading, as ``None``.
YAML_NULL_SPELLINGS = ["null", "Null", "NULL", "~", ""]

GDPO_VARIANTS = ["standard", "length_normalized", "margin"]


def _plain(text: "str | None") -> str:
    """Rich colours per character and wraps, so compare stripped + collapsed."""
    return " ".join(strip_ansi(text).split())


def _yaml(
    value: str,
    *,
    task: str = "sft",
    backend: str = "transformers",
    training_extra: str = "",
) -> str:
    return (
        f"base: some-model\ntask: {task}\nbackend: {backend}\n"
        "data: {train: ./data.jsonl}\n"
        f"training:\n  ebft_variant: {value}\n{training_extra}"
    )


def _assert_is_the_1230_refusal(message: str) -> None:
    flat = " ".join(message.split())
    assert "training.ebft_variant" in flat, flat
    assert "#1230" in flat, flat
    # Not the Literal error, which sends the user to a value that is refused.
    assert "Input should be" not in flat, flat


# --------------------------------------------------------------------------
# the refusal
# --------------------------------------------------------------------------


def test_the_message_names_the_setting_the_reason_and_the_issue():
    with pytest.raises(ValueError) as excinfo:
        load_config_from_string(_yaml("structured"))
    message = " ".join(str(excinfo.value).split())

    assert "training.ebft_variant" in message
    assert "#1230" in message
    assert "not yet a distinct objective" in message
    # Why, both halves: as shipped, and what the obvious repair would give.
    assert "no causal shift" in message
    assert "rewards copying" in message
    assert "cross-entropy twice" in message
    # How to make the config load again.
    assert "Remove ebft_variant and ebft_temperature" in message


@pytest.mark.parametrize("value", YAML_VALUES_THAT_LOADED)
def test_the_loader_refuses_every_value_that_used_to_load(value):
    with pytest.raises(ValueError) as excinfo:
        load_config_from_string(_yaml(value))
    _assert_is_the_1230_refusal(str(excinfo.value))


@pytest.mark.parametrize("value", YAML_OTHER_NON_NULL_VALUES)
def test_the_loader_refuses_every_other_non_null_value_with_the_same_message(value):
    with pytest.raises(ValueError) as excinfo:
        load_config_from_string(_yaml(value))
    _assert_is_the_1230_refusal(str(excinfo.value))


@pytest.mark.parametrize("value", PYTHON_NON_NULL_VALUES, ids=repr)
def test_direct_construction_refuses_every_non_null_value(value):
    with pytest.raises(ValidationError) as excinfo:
        TrainingConfig(ebft_variant=value)
    _assert_is_the_1230_refusal(str(excinfo.value))


def test_the_error_is_located_at_the_field_so_the_cli_names_it():
    with pytest.raises(ValidationError) as excinfo:
        SoupConfig(
            base="some-model",
            data={"train": "./data.jsonl"},
            training={"ebft_variant": "strided"},
        )
    locations = [error["loc"] for error in excinfo.value.errors()]
    assert locations == [("training", "ebft_variant")], locations


#: Configs the older gates refused with their OWN message (task, backend), plus
#: the SFT configs that loaded. The #1230 refusal must come first.
_OLDER_GATES = {
    "sft-with-temperature": {"training_extra": "  ebft_temperature: 0.5\n"},
    "task-dpo": {"task": "dpo"},
    "task-pretrain": {"task": "pretrain"},
    "backend-mlx": {"backend": "mlx"},
    "backend-unsloth": {"backend": "unsloth"},
}


@pytest.mark.parametrize("gate", sorted(_OLDER_GATES))
def test_the_refusal_comes_before_every_older_gate(gate):
    with pytest.raises(ValueError) as excinfo:
        load_config_from_string(_yaml("strided", **_OLDER_GATES[gate]))
    _assert_is_the_1230_refusal(str(excinfo.value))


def test_ebft_temperature_alone_no_longer_asks_for_the_refused_field():
    """It used to say "requires training.ebft_variant to be set" -- advice
    that now ends in the refusal above. It still refuses, pointing at #1230."""
    with pytest.raises(ValueError) as excinfo:
        load_config_from_string(
            "base: some-model\ntask: sft\ndata: {train: ./data.jsonl}\n"
            "training:\n  ebft_temperature: 0.5\n"
        )
    message = " ".join(str(excinfo.value).split())
    assert "training.ebft_temperature" in message, message
    assert "ebft_variant, which is refused" in message, message
    assert "#1230" in message, message
    assert "to be set" not in message, message


# --------------------------------------------------------------------------
# the entry points a user runs
# --------------------------------------------------------------------------


def _write_config(tmp_path: Path, value: str, *, backend: str = "transformers") -> Path:
    train = tmp_path / "train.jsonl"
    train.write_text(
        '{"messages": [{"role": "user", "content": "q"}, '
        '{"role": "assistant", "content": "a"}]}\n',
        encoding="utf-8",
    )
    path = tmp_path / "soup.yaml"
    path.write_text(
        "base: some-model\ntask: sft\n"
        f"backend: {backend}\n"
        f"data:\n  train: {train.as_posix()}\n  format: chatml\n"
        "training:\n  quantization: none\n"
        f"  ebft_variant: {value}\n"
        "  ebft_temperature: 1.0\n"
        "  lora: {r: 8, alpha: 16}\n"
        f"output: {(tmp_path / 'out').as_posix()}\n",
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("extra_args", [[], ["--dry-run"]], ids=["train", "dry-run"])
def test_soup_train_stops_at_config_load(tmp_path, monkeypatch, extra_args):
    """Nothing past the loader may run: no setup panel for a run whose loss
    would reward copying."""
    from soup_cli.cli import app

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.chdir(tmp_path)
    config = _write_config(tmp_path, "structured")

    result = CliRunner().invoke(app, ["train", "--config", str(config), *extra_args])

    output = _plain(result.output)
    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert "training -> ebft_variant" in output, output
    assert "#1230" in output, output
    assert "not yet a distinct objective" in output, output
    for later_line in ("Training Setup", "LoRA applied", "Loading model"):
        assert later_line not in output, (later_line, output)


def test_soup_doctor_config_reports_the_config_as_unloadable(tmp_path, capsys):
    """Before the fix this config loaded; now it cannot, and an unloadable
    config is exit 2."""
    from soup_cli.commands.doctor import doctor

    config = _write_config(tmp_path, "strided")

    with pytest.raises(typer.Exit) as excinfo:
        doctor(nccl=False, disk=False, config=str(config))

    assert excinfo.value.exit_code == 2
    output = _plain(capsys.readouterr().out)
    assert "training -> ebft_variant" in output, output
    assert "#1230" in output, output


# --------------------------------------------------------------------------
# controls -- these pass with and without the fix, by design
# --------------------------------------------------------------------------


class TestNoVariantStillLoads:
    @pytest.mark.parametrize("spelling", YAML_NULL_SPELLINGS)
    def test_every_null_spelling_still_loads_as_none(self, spelling):
        cfg = load_config_from_string(_yaml(spelling))
        assert cfg.training.ebft_variant is None
        assert cfg.training.ebft_temperature is None

    def test_the_default_is_still_none(self):
        cfg = SoupConfig(base="some-model", data={"train": "./data.jsonl"})
        assert cfg.training.ebft_variant is None
        assert cfg.training.ebft_temperature is None

    def test_direct_construction_with_none_still_works(self):
        assert TrainingConfig(ebft_variant=None).ebft_variant is None


def _loadable_configs() -> list[str]:
    return [
        "base: some-model\ntask: sft\ndata: {train: ./data.jsonl}\n",
        _yaml("null"),
        "base: some-model\ntask: dpo\ndata: {train: ./data.jsonl}\n"
        "training: {gdpo_variant: margin}\n",
    ]


@pytest.mark.parametrize("text", _loadable_configs(), ids=["sft", "sft-null", "dpo-gdpo"])
def test_the_ebft_hook_is_a_no_op_for_every_config_that_loads(text):
    """``SFTTrainerWrapper.train()`` calls this hook unconditionally; with the
    field refused it can never install the term."""
    from soup_cli.utils.ebft_gdpo import attach_ebft_compute_loss

    def original(*args, **kwargs):
        raise AssertionError("not called")

    cfg = load_config_from_string(text)
    trainer = SimpleNamespace(compute_loss=original)
    assert attach_ebft_compute_loss(trainer, cfg.training) is False
    assert trainer.compute_loss is original
    assert not getattr(trainer, "_soup_ebft_wrapped", False)


class TestGdpoIsUnchanged:
    """GDPO shares ``utils/ebft_gdpo.py``; this change leaves its config
    surface, gates and kernel as they were. These controls do not show that
    GDPO reaches the real DPO trainer: on trl 0.29 it does not (#1309), and
    the fix for #1309 owns these tests if it changes what loads."""

    @pytest.mark.parametrize("variant", GDPO_VARIANTS)
    def test_every_gdpo_variant_still_loads_on_dpo(self, variant):
        cfg = load_config_from_string(
            "base: some-model\ntask: dpo\ndata: {train: ./data.jsonl}\n"
            f"training: {{gdpo_variant: {variant}, dpo_beta: 0.2}}\n"
        )
        assert cfg.training.gdpo_variant == variant
        assert cfg.training.dpo_beta == 0.2
        assert cfg.training.ebft_variant is None

    @pytest.mark.parametrize("variant", GDPO_VARIANTS)
    def test_every_gdpo_variant_still_loads_on_preference(self, variant):
        cfg = load_config_from_string(
            "base: some-model\ntask: preference\ndata: {train: ./data.jsonl}\n"
            f"training: {{preference_loss: dpo, gdpo_variant: {variant}}}\n"
        )
        assert cfg.training.gdpo_variant == variant

    def test_the_gdpo_task_gate_is_unchanged(self):
        with pytest.raises(
            ValueError, match=r"gdpo_variant requires task in \('dpo', 'preference'\)"
        ):
            load_config_from_string(
                "base: some-model\ntask: sft\ndata: {train: ./data.jsonl}\n"
                "training: {gdpo_variant: standard}\n"
            )

    def test_the_gdpo_backend_gate_is_unchanged(self):
        with pytest.raises(ValueError, match="gdpo_variant is not supported on backend=mlx"):
            load_config_from_string(
                "base: some-model\ntask: dpo\nbackend: mlx\ndata: {train: ./data.jsonl}\n"
                "training: {gdpo_variant: standard}\n"
            )

    @pytest.mark.parametrize("variant", GDPO_VARIANTS)
    def test_the_gdpo_kernel_still_computes_its_closed_form(self, variant):
        import torch
        from torch.nn.functional import logsigmoid

        from soup_cli.utils.ebft_gdpo import apply_gdpo_loss

        pc = torch.tensor([-1.0, -2.5, -0.3])
        pr = torch.tensor([-1.7, -2.0, -3.1])
        rc = torch.tensor([-1.2, -2.2, -0.9])
        rr = torch.tensor([-1.1, -2.4, -2.8])
        lens_c = torch.tensor([4, 9, 2])
        lens_r = torch.tensor([6, 3, 0])  # 0 is clamped to 1 by the kernel
        beta, margin = 0.3, 0.25
        if variant == "length_normalized":
            expected = -logsigmoid(
                beta * (pc / lens_c.float() - pr / lens_r.float().clamp(min=1.0))
            ).mean()
        else:
            logits = beta * ((pc - pr) - (rc - rr))
            if variant == "margin":
                logits = logits - margin
            expected = -logsigmoid(logits).mean()

        loss = apply_gdpo_loss(
            policy_chosen_logps=pc,
            policy_rejected_logps=pr,
            ref_chosen_logps=rc,
            ref_rejected_logps=rr,
            chosen_lens=lens_c,
            rejected_lens=lens_r,
            variant=variant,
            beta=beta,
            margin=margin,
        )
        assert torch.allclose(loss, expected, atol=1e-7), (loss, expected)

    @pytest.mark.parametrize("variant", GDPO_VARIANTS)
    def test_the_gdpo_hook_still_installs_from_a_loaded_config(self, variant):
        """The hook's own behaviour, on a stub trainer that exposes
        ``dpo_loss``. trl 0.29's ``DPOTrainer`` has no ``dpo_loss``, so on the
        real trainer the hook returns False today (#1309); this change leaves
        GDPO exactly as it was."""
        import torch

        from soup_cli.utils.ebft_gdpo import apply_gdpo_loss, attach_gdpo_compute_loss

        def trl_dpo_loss(*args, **kwargs):
            raise AssertionError("the GDPO wrapper replaces TRL's dpo_loss")

        cfg = load_config_from_string(
            "base: some-model\ntask: dpo\ndata: {train: ./data.jsonl}\n"
            f"training: {{gdpo_variant: {variant}, dpo_beta: 0.2}}\n"
        )
        trainer = SimpleNamespace(dpo_loss=trl_dpo_loss)
        assert attach_gdpo_compute_loss(trainer, cfg.training) is True
        assert trainer._soup_gdpo_wrapped is True

        pc = torch.tensor([-1.0, -2.5])
        pr = torch.tensor([-1.7, -2.0])
        rc = torch.tensor([-1.2, -2.2])
        rr = torch.tensor([-1.1, -2.4])
        lens = torch.tensor([3, 5])
        losses, chosen_rewards, rejected_rewards = trainer.dpo_loss(
            pc, pr, rc, rr, chosen_lens=lens, rejected_lens=lens
        )
        expected = apply_gdpo_loss(
            policy_chosen_logps=pc,
            policy_rejected_logps=pr,
            ref_chosen_logps=rc,
            ref_rejected_logps=rr,
            chosen_lens=lens,
            rejected_lens=lens,
            variant=variant,
            beta=0.2,
            margin=0.0,
        )
        assert torch.allclose(losses, expected.expand_as(pc))
        assert torch.allclose(chosen_rewards, 0.2 * (pc - rc))
        assert torch.allclose(rejected_rewards, 0.2 * (pr - rr))


# --------------------------------------------------------------------------
# shipped configs
# --------------------------------------------------------------------------


def _shipped_config_texts() -> list[tuple[str, str]]:
    from soup_cli.recipes.catalog import RECIPES

    texts = [(f"recipe:{name}", meta.yaml_str) for name, meta in RECIPES.items()]
    for directory in (REPO_ROOT / "src" / "soup_cli" / "templates", REPO_ROOT / "examples"):
        assert directory.is_dir(), directory
        for path in sorted(directory.rglob("*.yaml")):
            texts.append(
                (path.relative_to(REPO_ROOT).as_posix(), path.read_text(encoding="utf-8"))
            )
    return texts


def test_every_shipped_recipe_template_and_example_still_loads():
    texts = _shipped_config_texts()
    assert len(texts) > 200, len(texts)
    refused = []
    for name, text in texts:
        try:
            cfg = load_config_from_string(text)
        except ValueError as exc:
            refused.append((name, str(exc)[:200]))
            continue
        assert cfg.training.ebft_variant is None, name
    assert refused == []


# --------------------------------------------------------------------------
# docs
# --------------------------------------------------------------------------

_VARIANT_SETTING = re.compile(r"^\s*ebft_variant:\s*(structured|strided)\b", re.MULTILINE)


def test_no_doc_still_shows_ebft_variant_as_a_setting_to_copy():
    targets = [*sorted(REPO_ROOT.glob("README*.md")), *sorted((REPO_ROOT / "docs").rglob("*.md"))]
    offenders = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in targets
        if _VARIANT_SETTING.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == [], f"these docs still show ebft_variant as a setting: {offenders}"


@pytest.mark.parametrize(
    "heading", ["## EBFT / GDPO Loss Variants", "## EBFT + GDPO (BETA, v0.52.0)"]
)
def test_the_ebft_docs_say_it_is_refused_and_gdpo_is_not(heading):
    text = (REPO_ROOT / "docs" / "training.md").read_text(encoding="utf-8")
    section = text.split(heading, 1)[1].split("\n## ", 1)[0]
    flat = " ".join(section.split())
    assert "#1230" in flat or "issues/1230" in flat, flat
    assert "refused" in flat, flat
    assert "gdpo_variant" in flat, flat
