"""#1430 — what `soup train --cloud` actually ships.

The stub carries one base64 blob: the config. Two things used to be lost
between the local run and the remote one, and both failed *after* the paid boot
with nothing said locally:

1. the local files the config points at (`data.train: ./data/train.jsonl` in
   all 21 built-in templates), and
2. the CLI overrides applied to the loaded config before the cloud branch,
   which were printed locally as enabled and absent from the remote run.

Plan-only throughout: no account, no network, no paid call.
"""

import base64
import json
import re
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.config.loader import load_config
from tests.conftest import strip_ansi

runner = CliRunner()

CLOUDS = ("lambda", "modal")


def _shipped_configs(stub: Path) -> list:
    """Every base64 blob in the rendered stub that decodes to a soup config."""
    text = stub.read_text(encoding="utf-8")
    blobs = [b for b in re.findall(r"[A-Za-z0-9+/=]{40,}", text) if len(b) % 4 == 0]
    decoded = [base64.b64decode(b).decode("utf-8", "replace") for b in blobs]
    return [d for d in decoded if "base:" in d]


def _rewrite(cfg, path: Path, **data) -> None:
    """Write `cfg` back with `data` overridden — the only handle tests need."""
    for key, value in data.items():
        setattr(cfg.data, key, value)
    path.write_text(
        yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )


@pytest.fixture
def chat_config(tmp_path, monkeypatch):
    """A `soup init --template chat` config with a local data.train (#1430)."""
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", "--template", "chat", "--output", "soup.yaml"])
    assert result.exit_code == 0, result.output
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "data" / "train.jsonl").write_text(
        json.dumps({"instruction": "q DATA-MARKER", "input": "", "output": "a"}) + "\n",
        encoding="utf-8",
    )
    return tmp_path / "soup.yaml"


def _plan(cloud, *extra):
    return runner.invoke(
        app, ["train", "--config", "soup.yaml", "--cloud", cloud, "--gpu", "a100", *extra]
    )


@pytest.mark.parametrize("cloud", CLOUDS)
def test_a_local_data_file_stops_the_plan_instead_of_the_paid_run(chat_config, cloud):
    """The default `soup init` then `--cloud` flow always hit this."""
    result = _plan(cloud)

    assert result.exit_code != 0, result.output
    assert "data.train" in strip_ansi(result.output), strip_ansi(result.output)
    assert not Path(f"soup_{cloud}_app.py").exists(), "a stub was rendered anyway"


@pytest.mark.parametrize("cloud", CLOUDS)
def test_a_hub_dataset_id_is_left_alone(chat_config, cloud):
    """A Hub id resolves on the remote machine, so it must still plan."""
    _rewrite(load_config(chat_config), chat_config, train="teknium/OpenHermes-2.5")

    result = _plan(cloud)

    assert result.exit_code == 0, result.output
    stub = Path(f"soup_{cloud}_app.py")
    assert stub.exists()
    assert _shipped_configs(stub), "no config was embedded"


@pytest.mark.parametrize("cloud", CLOUDS)
def test_an_override_reaches_the_remote_config(chat_config, cloud):
    """The override is printed locally, so it has to be in the stub too."""
    cfg = load_config(chat_config)
    cfg.data.train = "teknium/OpenHermes-2.5"
    cfg.task = "grpo"
    cfg.training.reward_fn = "accuracy"
    chat_config.write_text(
        yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )

    result = _plan(cloud, "--reward-hack-detector", "info_rm", "--reward-hack-halt")

    assert result.exit_code == 0, result.output
    shipped = "\n".join(_shipped_configs(Path(f"soup_{cloud}_app.py")))
    assert "reward_hack_detector: info_rm" in shipped, shipped
    assert "reward_hack_halt: true" in shipped, shipped
    # And it is the effective config, not the file's text.
    assert shipped != chat_config.read_text(encoding="utf-8"), (
        "the stub embedded the file text again, so the override was dropped"
    )


@pytest.mark.parametrize("cloud", CLOUDS)
def test_echo_trap_tokenizer_aware_reaches_the_remote_config(chat_config, cloud):
    cfg = load_config(chat_config)
    cfg.data.train = "teknium/OpenHermes-2.5"
    cfg.task = "grpo"
    cfg.training.reward_fn = "accuracy"
    cfg.training.echo_trap_enabled = True
    chat_config.write_text(
        yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )

    result = _plan(cloud, "--echo-trap-tokenizer-aware")

    assert result.exit_code == 0, result.output
    shipped = "\n".join(_shipped_configs(Path(f"soup_{cloud}_app.py")))
    assert "echo_trap_tokenizer_aware: true" in shipped, shipped


@pytest.mark.parametrize("cloud", CLOUDS)
def test_minillm_on_policy_reaches_the_remote_config(chat_config, cloud):
    cfg = load_config(chat_config)
    cfg.data.train = "teknium/OpenHermes-2.5"
    cfg.task = "distill"
    cfg.training.teacher_model = "HuggingFaceTB/SmolLM2-360M"
    cfg.training.minillm_enabled = True
    cfg.training.minillm_teacher_mix_ratio = 0.5
    chat_config.write_text(
        yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )

    result = _plan(cloud, "--minillm-on-policy")

    assert result.exit_code == 0, result.output
    shipped = "\n".join(_shipped_configs(Path(f"soup_{cloud}_app.py")))
    assert "minillm_on_policy: true" in shipped, shipped
    assert "minillm_teacher_mix_ratio: 0.5" in shipped, shipped


@pytest.mark.parametrize("cloud", CLOUDS)
def test_reward_hack_mitigation_reaches_the_remote_config(chat_config, cloud):
    cfg = load_config(chat_config)
    cfg.data.train = "teknium/OpenHermes-2.5"
    cfg.task = "grpo"
    cfg.training.reward_fn = "accuracy"
    chat_config.write_text(
        yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )

    result = _plan(
        cloud,
        "--reward-hack-detector",
        "info_rm",
        "--reward-hack-mitigation",
        "kl_control",
    )

    assert result.exit_code == 0, " ".join(strip_ansi(result.output).split())
    shipped = "\n".join(_shipped_configs(Path(f"soup_{cloud}_app.py")))
    assert "reward_hack_mitigation: kl_control" in shipped, shipped


@pytest.mark.parametrize("cloud", CLOUDS)
def test_an_unsupported_local_field_is_named_too(chat_config, cloud):
    """Not just data.train: anything the loader opens locally is named."""
    _rewrite(load_config(chat_config), chat_config,
             train="teknium/OpenHermes-2.5", image_dir="./images")

    result = _plan(cloud)

    assert result.exit_code != 0, result.output
    assert "data.image_dir" in strip_ansi(result.output), strip_ansi(result.output)


@pytest.mark.parametrize("argv", [["--push-as", "me/soup"], ["--gate", "suite.yaml"]])
def test_flags_handled_after_the_cloud_branch_are_refused(chat_config, argv, tmp_path):
    """`--push-as` and `--gate` run locally; a cloud run ignored them silently."""
    _rewrite(load_config(chat_config), chat_config, train="teknium/OpenHermes-2.5")
    (tmp_path / "suite.yaml").write_text("tasks: []\n", encoding="utf-8")

    result = _plan("lambda", *argv)

    assert result.exit_code != 0, result.output
    assert argv[0] in strip_ansi(result.output), strip_ansi(result.output)
    assert not Path("soup_lambda_app.py").exists(), "a stub was rendered anyway"


class TestUnshippedInputs:
    """The classifier behind the refusal."""

    def _cfg(self, train="teknium/OpenHermes-2.5"):
        from soup_cli.config.loader import load_config_from_string

        return load_config_from_string(
            "base: HuggingFaceTB/SmolLM2-135M\ntask: grpo\n"
            f"data:\n  train: {train}\noutput: ./out\n"
        )

    def test_a_hub_id_is_not_unshipped(self):
        from soup_cli.cloud._shipped import unshipped_inputs

        assert unshipped_inputs(self._cfg()) == []

    def test_a_local_path_is_unshipped(self):
        from soup_cli.cloud._shipped import unshipped_inputs

        assert unshipped_inputs(self._cfg("./data/train.jsonl")) == ["data.train"]

    def test_a_list_is_judged_entry_by_entry(self):
        """A Hub id beside a local file is still a local file."""
        from soup_cli.cloud._shipped import unshipped_inputs

        cfg = self._cfg()
        cfg.data.train = ["teknium/OpenHermes-2.5", "./data/train.jsonl"]
        assert unshipped_inputs(cfg) == ["data.train"]

        cfg.data.train = ["teknium/OpenHermes-2.5", "mlfoundations/dclm-baseline-1.0"]
        assert unshipped_inputs(cfg) == []

    def test_a_remote_uri_is_not_unshipped(self):
        """The loader fetches an allowlisted URI over the network."""
        from soup_cli.cloud._shipped import unshipped_inputs

        assert unshipped_inputs(self._cfg("s3://bucket/train.jsonl")) == []

    def test_a_bare_existing_path_is_unshipped(self, tmp_path, monkeypatch):
        from soup_cli.cloud._shipped import unshipped_inputs

        monkeypatch.chdir(tmp_path)
        (tmp_path / "merged").mkdir()
        cfg = self._cfg()
        cfg.base = "merged"
        assert unshipped_inputs(cfg) == ["base"]
        cfg.base = "HuggingFaceTB/SmolLM2-135M"  # same shape, nothing on disk: a Hub id
        assert unshipped_inputs(cfg) == []

    def test_a_drive_letter_path_is_unshipped_even_when_absent(self):
        from soup_cli.cloud._shipped import unshipped_inputs

        cfg = self._cfg()
        cfg.base = "C:\\models\\merged"
        assert unshipped_inputs(cfg) == ["base"]

    def test_a_local_reward_file_is_unshipped(self, value):
        from soup_cli.cloud._shipped import unshipped_inputs

        cfg = self._cfg()
        cfg.training.reward_fn = value
        assert unshipped_inputs(cfg) == ["training.reward_fn"]

    def test_builtin_reward_names_are_not_unshipped(self):
        from soup_cli.cloud._shipped import unshipped_inputs

        cfg = self._cfg()
        cfg.training.reward_fn = "accuracy, format"
        assert unshipped_inputs(cfg) == []

    def test_a_field_soup_train_never_reads_is_not_refused(self):
        from soup_cli.cloud._shipped import unshipped_inputs

        cfg = self._cfg()
        cfg.training.checkpoint_eval_tasks = "./tasks.jsonl"
        assert unshipped_inputs(cfg) == []

    def test_every_local_spelling_is_unshipped_even_when_absent(self, value):
        """None of these exists on disk, so only the spelling can say 'local'."""
        from soup_cli.cloud._shipped import unshipped_inputs

        cfg = self._cfg()
        cfg.base = value
        assert unshipped_inputs(cfg) == ["base"]
        from soup_cli.cloud._shipped import unshipped_inputs

        cfg = self._cfg()
        cfg.training.ra_dit_retriever_model = "sentence-transformers/all-mpnet-base-v2"
        assert unshipped_inputs(cfg) == []
        cfg.training.prm_reward = "someuser/soup-prm-135m"
        assert unshipped_inputs(cfg) == []

    def test_a_local_model_path_is_unshipped(self, tmp_path, monkeypatch):
        from soup_cli.cloud._shipped import unshipped_inputs

        monkeypatch.chdir(tmp_path)
        for path in (
            "./models/merged",
            "./models/teacher",
            "./models/rm",
            "./adapters/a",
            "./adapters/b",
        ):
            (tmp_path / path).mkdir(parents=True, exist_ok=True)
        cfg = self._cfg()
        cfg.base = "./models/merged"
        cfg.training.teacher_model = "./models/teacher"
        cfg.training.reward_model = "./models/rm"
        cfg.training.mole_task_adapters = ["./adapters/a", "./adapters/b"]
        assert unshipped_inputs(cfg) == [
            "base",
            "training.mole_task_adapters",
            "training.reward_model",
            "training.teacher_model",
        ]

    def test_the_message_names_the_escape_hatch(self):
        from soup_cli.cloud._shipped import refuse_unshipped_inputs

        with pytest.raises(ValueError, match="Hub dataset id"):
            refuse_unshipped_inputs(self._cfg("./data/train.jsonl"))

