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
import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.cloud._local_inputs import (
    _MAX_LOCAL_TRAIN_INPUT_BYTES,
    prepare_local_train_inputs,
)
from soup_cli.config.loader import load_config
from tests.conftest import strip_ansi

runner = CliRunner()

CLOUDS = ("lambda", "modal", "runpod")


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
def test_a_local_data_file_is_shipped_in_the_plan(chat_config, cloud):
    """Local training data is shipped by supported backends only."""
    result = _plan(cloud)

    if cloud == "runpod":
        assert result.exit_code == 2
        assert "--cloud runpod does not ship local data.train files" in result.output
        assert not Path("soup_runpod_app.py").exists()
        return

    assert result.exit_code == 0, result.output
    assert "Inputs:" in result.output, result.output
    assert "inputs/train-0.jsonl" in result.output, result.output
    assert "bytes" in result.output, result.output

    stub = Path(f"soup_{cloud}_app.py")
    assert stub.exists()

    shipped = "\n".join(_shipped_configs(stub))
    assert "inputs/train-0.jsonl" in shipped, shipped

    stub_text = stub.read_text(encoding="utf-8")
    assert "train-0.jsonl" in stub_text
    assert "local training input" in stub_text.lower()


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
    cfg.training.quantization = "none"  # distill never quantises the base
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


def _grpo_hub_config(chat_config) -> None:
    """Turn the chat config into a grpo run on a Hub dataset, so it can plan."""
    cfg = load_config(chat_config)
    cfg.data.train = "teknium/OpenHermes-2.5"
    cfg.task = "grpo"
    cfg.training.reward_fn = "accuracy"
    chat_config.write_text(
        yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )


@pytest.mark.parametrize("cloud", CLOUDS)
def test_the_shipped_config_loads_the_way_the_remote_run_loads_it(chat_config, cloud):
    """The remote `soup train` loads the embedded text through every validator."""
    from soup_cli.config.loader import load_config_from_string

    _grpo_hub_config(chat_config)

    result = _plan(cloud, "--reward-hack-detector", "info_rm", "--reward-hack-halt")

    assert result.exit_code == 0, " ".join(strip_ansi(result.output).split())
    (shipped,) = _shipped_configs(Path(f"soup_{cloud}_app.py"))
    reloaded = load_config_from_string(shipped)
    assert reloaded.training.reward_hack_detector == "info_rm"
    assert reloaded.training.reward_hack_halt is True
    assert reloaded.data.train == "teknium/OpenHermes-2.5"


@pytest.mark.parametrize("cloud", CLOUDS)
def test_an_override_the_schema_refuses_stops_the_plan_before_a_stub(chat_config, cloud):
    """`--reward-hack-detector rm_ensemble` with a mitigation, and the default
    `reward_hack_signals: [info_rm]`, is a config `soup train` itself refuses.

    The overrides are assigned after the load, so no validator ran on the result;
    the remote run would reload it and die after the paid boot."""
    _grpo_hub_config(chat_config)

    result = _plan(
        cloud,
        "--reward-hack-detector",
        "rm_ensemble",
        "--reward-hack-mitigation",
        "log_only",
    )

    output = " ".join(strip_ansi(result.output).split())
    assert result.exit_code == 2, output
    assert "Cloud plan failed" in output, output
    assert "would be refused on the remote machine" in output, output
    assert (
        "reward_hack_signals contains 'info_rm', but the active detector is 'rm_ensemble'"
        in output
    ), output
    assert not Path(f"soup_{cloud}_app.py").exists(), "a stub was rendered anyway"


@pytest.mark.parametrize("cloud", CLOUDS)
def test_a_detector_on_a_task_the_schema_refuses_stops_the_plan(chat_config, cloud):
    """The detector flag on an sft config: only grpo and ppo accept it."""
    _rewrite(load_config(chat_config), chat_config, train="teknium/OpenHermes-2.5")

    result = _plan(cloud, "--reward-hack-detector", "info_rm")

    output = " ".join(strip_ansi(result.output).split())
    assert result.exit_code == 2, output
    assert "Cloud plan failed" in output, output
    assert "only valid on task in {'grpo', 'ppo'}; got task='sft'" in output, output
    assert not Path(f"soup_{cloud}_app.py").exists(), "a stub was rendered anyway"


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


@pytest.mark.parametrize(
    "argv, phrase, runs",
    [
        (["--push-as", "me/soup"], "--push-as is not supported with --cloud", "it runs locally"),
        (["--gate", "suite.yaml"], "--gate is not supported with --cloud", "it runs locally"),
        (
            ["--push-as", "me/soup", "--gate", "suite.yaml"],
            "--push-as, --gate are not supported with --cloud",
            "they run locally",
        ),
    ],
    ids=["push-as", "gate", "both"],
)
def test_the_flag_refusal_names_every_flag_with_the_right_verb(
    chat_config, argv, phrase, runs, tmp_path
):
    """One flag reads `is` and `it runs`, two read `are` and `they run`."""
    _rewrite(load_config(chat_config), chat_config, train="teknium/OpenHermes-2.5")
    (tmp_path / "suite.yaml").write_text("tasks: []\n", encoding="utf-8")

    result = _plan("lambda", *argv)

    output = " ".join(strip_ansi(result.output).split())
    assert result.exit_code == 2, output
    assert phrase in output, output
    assert runs in output, output
    assert "both run locally" not in output, output


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

    def test_a_local_path_is_not_unshipped(self):
        from soup_cli.cloud._shipped import unshipped_inputs

        assert unshipped_inputs(self._cfg("./data/train.jsonl")) == []

    def test_a_list_is_judged_entry_by_entry(self):
        """Local files are handled separately from unsupported inputs."""
        from soup_cli.cloud._shipped import unshipped_inputs

        cfg = self._cfg()
        cfg.data.train = ["teknium/OpenHermes-2.5", "./data/train.jsonl"]
        assert unshipped_inputs(cfg) == []

        cfg.data.train = [
            "teknium/OpenHermes-2.5",
            "mlfoundations/dclm-baseline-1.0",
        ]
        assert unshipped_inputs(cfg) == []

    @pytest.mark.parametrize(
        "value",
        ["s3://bucket/train.jsonl", "https://example.com/train.jsonl", "file:///t/train.jsonl"],
    )
    def test_a_uri_is_not_unshipped(self, value):
        """Anything spelled `scheme://` is not judged as a local file here. The scheme
        allowlist is the loader's job (`validate_remote_uri`), so the plan does not
        enforce one: `https://` and `file://` plan like `s3://`."""
        from soup_cli.cloud._shipped import unshipped_inputs

        assert unshipped_inputs(self._cfg(value)) == []

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

    @pytest.mark.parametrize(
        "value",
        ["./rewards.py", "accuracy, ./rewards.py", "rewards.py", ".\\rewards", "~\\rewards"],
    )
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

    def test_a_directory_named_like_a_builtin_reward_does_not_refuse_the_builtin(
        self, tmp_path, monkeypatch
    ):
        """The resolver checks the built-in names before it looks for a `.py` file, so
        a `format/` or `accuracy/` directory beside the config changes nothing."""
        from soup_cli.cloud._shipped import unshipped_inputs

        monkeypatch.chdir(tmp_path)
        (tmp_path / "format").mkdir()
        (tmp_path / "accuracy").mkdir()
        cfg = self._cfg()
        cfg.training.reward_fn = "accuracy, format"
        assert unshipped_inputs(cfg) == []
        # The same directory is still a local path for a field that reads one.
        cfg.base = "format"
        assert unshipped_inputs(cfg) == ["base"]

    def test_a_field_soup_train_never_reads_is_not_refused(self):
        from soup_cli.cloud._shipped import unshipped_inputs

        cfg = self._cfg()
        cfg.training.checkpoint_eval_tasks = "./tasks.jsonl"
        assert unshipped_inputs(cfg) == []

    @pytest.mark.parametrize(
        "value",
        [
            "./nowhere/model",
            "../nowhere/model",
            "/nowhere/model",
            "~/nowhere/model",
            "\\\\nowhere\\share\\model",
            "\\nowhere\\model",
            ".\\nowhere\\model",
            "..\\nowhere\\model",
            "~\\nowhere\\model",
        ],
    )
    def test_every_local_spelling_is_unshipped_even_when_absent(self, value):
        """None of these exists on disk, so only the spelling can say 'local'."""
        from soup_cli.cloud._shipped import unshipped_inputs

        cfg = self._cfg()
        cfg.base = value
        assert unshipped_inputs(cfg) == ["base"]

    def test_a_hub_model_id_is_not_unshipped(self):
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

    def test_image_dir_is_refused(self):
        from soup_cli.cloud._shipped import refuse_unshipped_inputs

        cfg = self._cfg()
        cfg.data.image_dir = "./images"

        with pytest.raises(ValueError, match="data.image_dir"):
            refuse_unshipped_inputs(cfg)

    def test_a_local_training_file_is_not_refused(self):
        from soup_cli.cloud._shipped import refuse_unshipped_inputs

        refuse_unshipped_inputs(self._cfg("./data/train.jsonl"))

class TestDataTrainDirectory:
    """A data.train with no data suffix is a Hub id OR a local dataset directory."""

    def _cfg(self, train):
        from soup_cli.config.loader import load_config_from_string

        return load_config_from_string(
            "base: HuggingFaceTB/SmolLM2-135M\ntask: sft\n"
            f"data:\n  train: {train}\noutput: ./out\n"
        )

    @pytest.mark.parametrize(
        "value",
        ["./hfdir", "../hfdir", "/nowhere/hfdir", "C:\\x\\hfdir", "~/hfdir"],
    )
    def test_a_directory_spelled_as_a_path_is_unshipped(self, value):
        from soup_cli.cloud._shipped import unshipped_inputs

        assert unshipped_inputs(self._cfg(value)) == ["data.train"]

    def test_a_bare_existing_directory_is_unshipped(self, tmp_path, monkeypatch):
        from soup_cli.cloud._shipped import unshipped_inputs

        monkeypatch.chdir(tmp_path)
        (tmp_path / "hfdir").mkdir()
        assert unshipped_inputs(self._cfg("hfdir")) == ["data.train"]
        assert unshipped_inputs(self._cfg("teknium/OpenHermes-2.5")) == []

    @pytest.mark.parametrize("cloud", CLOUDS)
    def test_a_local_dataset_directory_stops_the_plan(self, chat_config, cloud):
        """The CLI path: `train: ./hfdir` exited 0 and rendered a stub before."""
        (chat_config.parent / "hfdir").mkdir()
        _rewrite(load_config(chat_config), chat_config, train="./hfdir")

        result = _plan(cloud)

        output = " ".join(strip_ansi(result.output).split())
        assert result.exit_code == 2, output
        assert "data.train" in output, output
        assert not Path(f"soup_{cloud}_app.py").exists(), "a stub was rendered anyway"


class TestRefusalHint:
    """The refusal names the Hub id that fits the refused field (#1584 review)."""

    def _cfg(self, train="teknium/OpenHermes-2.5"):
        from soup_cli.config.loader import load_config_from_string

        return load_config_from_string(
            "base: HuggingFaceTB/SmolLM2-135M\ntask: grpo\n"
            f"data:\n  train: {train}\noutput: ./out\n"
        )

    def _message(self, cfg):
        from soup_cli.cloud._shipped import refuse_unshipped_inputs

        with pytest.raises(ValueError) as excinfo:
            refuse_unshipped_inputs(cfg)
        return str(excinfo.value)

    def test_a_local_model_path_points_at_a_model_id(self):
        cfg = self._cfg()
        cfg.base = "./models/merged"

        message = self._message(cfg)

        assert "base" in message
        assert "Hub model id" in message, message
        assert "dataset id" not in message, message

    def test_a_local_dataset_points_at_a_dataset_id(self):
        message = self._message(self._cfg("./data/train-dir"))

        assert "Hub dataset id" in message, message
        assert "model id" not in message, message

    def test_a_dataset_and_a_model_path_name_both(self):
        cfg = self._cfg("./data/train-dir")
        cfg.base = "./models/merged"

        message = self._message(cfg)

        assert "Hub dataset id or model id" in message, message

    @pytest.mark.parametrize("field", ["forget_set", "retain_set"])
    def test_an_unlearn_dataset_points_at_a_dataset_id(self, field):
        """Both accept a Hub dataset name (their field descriptions say so)."""
        cfg = self._cfg()
        setattr(cfg.data, field, "./forget.jsonl")

        message = self._message(cfg)

        assert f"data.{field}" in message, message
        assert "Hub dataset id" in message, message
        assert "Run locally." not in message, message

    def test_a_field_with_no_hub_form_says_to_run_locally(self):
        """A reward file has no Hub id, so neither hint fits."""
        cfg = self._cfg()
        cfg.training.reward_fn = "./rewards.py"

        message = self._message(cfg)

        assert "training.reward_fn" in message
        assert "Hub" not in message, message
        assert "Run locally." in message, message


@pytest.mark.parametrize(
    "module, function",
    [
        ("lambda_labs", "plan_lambda_run"),
        ("modal", "plan_modal_run"),
        ("runpod", "plan_runpod_run"),
    ],
)
def test_a_plan_embeds_the_config_text_it_is_given_not_the_file(
    tmp_path, monkeypatch, module, function
):
    """`soup train` passes the effective config; every backend has to take it."""
    import importlib

    monkeypatch.chdir(tmp_path)
    (tmp_path / "soup.yaml").write_text(
        "base: from-the-file\ndata: {train: public/dataset}\noutput: out\n",
        encoding="utf-8",
    )
    effective = "base: public/model\ndata: {train: public/dataset}\noutput: out\n"
    backend = importlib.import_module(f"soup_cli.cloud.{module}")

    plan = getattr(backend, function)(
        "soup.yaml",
        gpu="a100",
        output_dir="out",
        soup_version="0.77.2",
        config_yaml=effective,
    )
    stub = Path(backend.write_stub(plan))

    assert _shipped_configs(stub) == [effective]


def test_runpod_still_reads_the_file_when_no_config_text_is_given(tmp_path, monkeypatch):
    """The fallback for a direct caller, as for the Lambda and Modal plans."""
    from soup_cli.cloud import runpod

    monkeypatch.chdir(tmp_path)
    on_disk = "base: public/model\ndata: {train: public/dataset}\noutput: out\n"
    (tmp_path / "soup.yaml").write_text(on_disk, encoding="utf-8")

    plan = runpod.plan_runpod_run(
        "soup.yaml", gpu="a100", output_dir="out", soup_version="0.77.2"
    )

    assert _shipped_configs(Path(runpod.write_stub(plan))) == [on_disk]


def test_local_train_input_rejects_missing_file(chat_config):
    cfg = load_config(chat_config)
    cfg.data.train = "data/missing.jsonl"

    with pytest.raises(ValueError, match=r"data\.train\[0\].*does not exist"):
        prepare_local_train_inputs(cfg)


def test_local_train_input_rejects_file_outside_working_directory(
    chat_config, tmp_path
):
    cfg = load_config(chat_config)
    outside = tmp_path.parent / "outside.jsonl"
    outside.write_text('{"text": "outside"}\n', encoding="utf-8")
    cfg.data.train = str(outside)

    with pytest.raises(ValueError, match="inside the working directory"):
        prepare_local_train_inputs(cfg, cwd=str(tmp_path))


def test_local_train_input_enforces_fixed_256_mib_limit():
    assert _MAX_LOCAL_TRAIN_INPUT_BYTES == 256 * 1024 * 1024


def test_local_train_input_accepts_file_at_exact_size_limit(
    chat_config, monkeypatch, tmp_path
):
    from soup_cli.cloud import _local_inputs

    limit = 10
    monkeypatch.setattr(_local_inputs, "_MAX_LOCAL_TRAIN_INPUT_BYTES", limit)

    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    train_file = data_dir / "train.jsonl"
    train_file.write_bytes(b"1234567890")

    cfg = load_config(chat_config)
    cfg.data.train = "data/train.jsonl"

    manifest = prepare_local_train_inputs(cfg, cwd=str(tmp_path))

    assert len(manifest) == 1
    assert manifest[0].size_bytes == limit


def test_local_train_input_rejects_individual_file_over_limit(
    chat_config, monkeypatch, tmp_path
):
    from soup_cli.cloud import _local_inputs

    limit = 10
    monkeypatch.setattr(_local_inputs, "_MAX_LOCAL_TRAIN_INPUT_BYTES", limit)

    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    train_file = data_dir / "train.jsonl"
    train_file.write_bytes(b"12345678901")

    cfg = load_config(chat_config)
    cfg.data.train = "data/train.jsonl"

    with pytest.raises(
        ValueError,
        match=r"data\.train\[0\].*11 bytes.*maximum allowed is .*10 bytes",
    ):
        prepare_local_train_inputs(cfg, cwd=str(tmp_path))


def test_local_train_input_rejects_aggregate_size_over_limit(
    chat_config, monkeypatch, tmp_path
):
    from soup_cli.cloud import _local_inputs

    monkeypatch.setattr(_local_inputs, "_MAX_LOCAL_TRAIN_INPUT_BYTES", 10)

    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)

    first = data_dir / "first.jsonl"
    second = data_dir / "second.jsonl"
    first.write_bytes(b"123456")
    second.write_bytes(b"12345")

    cfg = load_config(chat_config)
    cfg.data.train = ["data/first.jsonl", "data/second.jsonl"]

    with pytest.raises(
        ValueError,
        match=r"total .*11 bytes.*after adding data\.train\[1\].*maximum allowed is .*10 bytes",
    ):
        prepare_local_train_inputs(cfg, cwd=str(tmp_path))


def test_local_train_inputs_get_deterministic_remote_paths(chat_config):
    cfg = load_config(chat_config)
    cfg.data.train = ["data/train.jsonl", "teknium/OpenHermes-2.5"]

    manifest = prepare_local_train_inputs(cfg)

    assert len(manifest) == 1
    assert manifest[0].remote_path == "inputs/train-0.jsonl"
    assert cfg.data.train == [
        "inputs/train-0.jsonl",
        "teknium/OpenHermes-2.5",
    ]
    assert manifest[0].size_bytes == Path("data/train.jsonl").stat().st_size


@pytest.mark.parametrize(
    "entries",
    [
        ["data/first.jsonl", "data/second.jsonl"],
        ["data/second.jsonl", "data/first.jsonl"],
    ],
)
def test_local_train_inputs_assign_distinct_remote_paths(
    chat_config, entries
):
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    (data_dir / "first.jsonl").write_text('{"text": "first"}\n')
    (data_dir / "second.jsonl").write_text('{"text": "second"}\n')

    cfg = load_config(chat_config)
    cfg.data.train = entries

    manifest = prepare_local_train_inputs(cfg)

    assert len(manifest) == 2
    assert len({item.remote_path for item in manifest}) == 2
    assert [item.remote_path for item in manifest] == [
        "inputs/train-0.jsonl",
        "inputs/train-1.jsonl",
    ]


def test_local_train_input_rejects_directory_with_data_file_suffix(
    chat_config, tmp_path
):
    (tmp_path / "dataset.jsonl").mkdir()

    cfg = load_config(chat_config)
    cfg.data.train = "dataset.jsonl"

    with pytest.raises(
        ValueError,
        match="does not exist or is not a regular file",
    ):
        prepare_local_train_inputs(cfg, cwd=str(tmp_path))


def test_local_train_input_rejects_dotfile(chat_config, tmp_path):
    hidden_file = tmp_path / ".hidden.jsonl"
    hidden_file.write_text('{"text": "test"}\n', encoding="utf-8")

    cfg = load_config(chat_config)
    cfg.data.train = ".hidden.jsonl"

    with pytest.raises(
        ValueError,
        match="dotfiles and dot-directories are not allowed",
    ):
        prepare_local_train_inputs(cfg, cwd=str(tmp_path))


def test_local_train_input_rejects_dot_directory(chat_config, tmp_path):
    hidden_dir = tmp_path / "data" / ".hidden"
    hidden_dir.mkdir(parents=True)
    train_file = hidden_dir / "train.jsonl"
    train_file.write_text('{"text": "test"}\n', encoding="utf-8")

    cfg = load_config(chat_config)
    cfg.data.train = "data/.hidden/train.jsonl"

    with pytest.raises(
        ValueError,
        match="dotfiles and dot-directories are not allowed",
    ):
        prepare_local_train_inputs(cfg, cwd=str(tmp_path))


@pytest.mark.requires_symlink
def test_local_train_input_rejects_file_symlink(chat_config, tmp_path):
    target = tmp_path / "data" / "train.jsonl"
    link = tmp_path / "data" / "linked.jsonl"

    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target)

    cfg = load_config(chat_config)
    cfg.data.train = "data/linked.jsonl"

    with pytest.raises(ValueError, match="symlinks are not allowed"):
        prepare_local_train_inputs(cfg, cwd=str(tmp_path))


@pytest.mark.requires_symlink
def test_local_train_input_rejects_symlinked_parent(chat_config, tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir(exist_ok=True)
    (outside / "train.jsonl").write_text('{"text": "test"}\n', encoding="utf-8")
    link = tmp_path / "linked_data"

    link.symlink_to(outside, target_is_directory=True)

    cfg = load_config(chat_config)
    cfg.data.train = "linked_data/train.jsonl"

    with pytest.raises(ValueError, match="symlinks are not allowed"):
        prepare_local_train_inputs(cfg, cwd=str(tmp_path))


@pytest.mark.skipif(os.name != "nt", reason="Windows junction test")
def test_local_train_input_rejects_junction_parent(chat_config, tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-junction-target"
    outside.mkdir(exist_ok=True)
    (outside / "train.jsonl").write_text(
        '{"text": "test"}\n',
        encoding="utf-8",
    )

    junction = tmp_path / "junction_data"

    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(outside)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        pytest.skip(f"Junction creation unavailable: {result.stderr}")

    cfg = load_config(chat_config)
    cfg.data.train = "junction_data/train.jsonl"

    with pytest.raises(ValueError, match="symlinks are not allowed"):
        prepare_local_train_inputs(cfg, cwd=str(tmp_path))
