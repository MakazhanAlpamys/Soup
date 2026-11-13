"""`soup train --push-as` creates its Hub repository private; `--push-public` opts out.

The rule under test:

* a repository that `--push-as <repo>` CREATES is private unless `--push-public` is given;
* `--push-public` without `--push-as` is refused, naming both flags;
* a repository that already exists is used as it is: nothing that could change its
  visibility is sent to the Hub, and the run says so;
* one line states what happened (created private / created public / existing, unchanged);
* the training callback uploads to the same repository and never creates it again.

The Hub is a recording fake: no test here opens a connection.
"""

from __future__ import annotations

import json
import re
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    """ANSI-stripped, whitespace-collapsed CLI output, safe to substring-match."""
    return " ".join(_ANSI_RE.sub("", text or "").split())


REPO = "user/my-model"

CREATED_PRIVATE = "Created private Hub repository user/my-model."
CREATED_PUBLIC = "Created public Hub repository user/my-model."
EXISTING_UNCHANGED = "Using existing Hub repository user/my-model; its visibility is unchanged."
REFUSAL = (
    "--push-public requires --push-as <repo>: it sets the visibility of the "
    "repository --push-as creates, so it does nothing without it."
)


class _FakeHub:
    """Records every call the push path makes; models only what it touches."""

    def __init__(self, *, existing: bool = False, probe_error: Exception | None = None,
                 branches: tuple[str, ...] = ()):
        self.existing = existing
        self.probe_error = probe_error
        self.branches = branches
        self.calls: list[tuple[str, dict]] = []

    def _record(self, name: str, kwargs: dict) -> None:
        self.calls.append((name, dict(kwargs)))

    def of(self, name: str) -> list[dict]:
        return [kwargs for call, kwargs in self.calls if call == name]

    def repo_exists(self, repo_id, **kwargs):
        self._record("repo_exists", {"repo_id": repo_id, **kwargs})
        if self.probe_error is not None:
            raise self.probe_error
        return self.existing

    def create_repo(self, **kwargs):
        self._record("create_repo", kwargs)

    def create_branch(self, **kwargs):
        self._record("create_branch", kwargs)

    def upload_folder(self, **kwargs):
        self._record("upload_folder", kwargs)

    def list_repo_refs(self, **kwargs):
        self._record("list_repo_refs", kwargs)
        return types.SimpleNamespace(
            branches=[types.SimpleNamespace(name=name) for name in self.branches]
        )


def _use_hub(monkeypatch, hub: _FakeHub) -> _FakeHub:
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    monkeypatch.setattr("soup_cli.monitoring.hf_push.get_hf_api", lambda **_: hub)
    return hub


def _write_project(tmp_path: Path) -> Path:
    (tmp_path / "train.jsonl").write_text(
        json.dumps({"instruction": "hello", "output": "world"}) + "\n", encoding="utf-8"
    )
    config = tmp_path / "soup.yaml"
    config.write_text(
        "base: test-org/test-model\n"
        "task: sft\n"
        "data: {train: train.jsonl, val_split: 0.0}\n"
        "output: out\n",
        encoding="utf-8",
    )
    return config


def _run_train(tmp_path, monkeypatch, extra_args):
    """Run the real `soup train` command with the trainer stubbed out.

    Returns (CliRunner result, the stub's HF trainer). The callback `--push-as`
    attaches is `hf_trainer.add_callback.call_args.args[0]`.
    """
    import soup_cli.commands.train as train_cmd
    import soup_cli.trainer.dispatch as dispatch_mod

    config = _write_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "test.db"))
    stub = MagicMock()
    stub.train.return_value = {
        "initial_loss": 2.0,
        "final_loss": 1.0,
        "total_steps": 1,
        "duration_secs": 0.5,
        "duration": "0.5s",
        "output_dir": str(tmp_path / "out"),
    }
    monkeypatch.setattr(dispatch_mod, "build_trainer", lambda cfg, **kw: stub)
    monkeypatch.setattr(train_cmd, "_should_run_diagnose_gate_on_rank", lambda: True)
    result = runner.invoke(app, ["train", "--config", str(config), "--yes", *extra_args])
    return result, stub.trainer


def _attached_callback(hf_trainer):
    assert hf_trainer.add_callback.call_count == 1, hf_trainer.add_callback.call_args_list
    return hf_trainer.add_callback.call_args.args[0]


# --------------------------------------------------------------------------- #
# ensure_hub_repo: what is sent to the Hub                                    #
# --------------------------------------------------------------------------- #


class TestEnsureHubRepo:
    def test_new_repository_is_created_private(self):
        from soup_cli.monitoring.hf_push import REPO_CREATED, ensure_hub_repo

        hub = _FakeHub(existing=False)
        outcome = ensure_hub_repo(hub, REPO, private=True)
        assert outcome == REPO_CREATED
        assert hub.of("create_repo") == [{"repo_id": REPO, "private": True, "exist_ok": True}]

    def test_new_repository_is_created_public_when_asked(self):
        from soup_cli.monitoring.hf_push import REPO_CREATED, ensure_hub_repo

        hub = _FakeHub(existing=False)
        outcome = ensure_hub_repo(hub, REPO, private=False)
        assert outcome == REPO_CREATED
        assert hub.of("create_repo") == [{"repo_id": REPO, "private": False, "exist_ok": True}]

    @pytest.mark.parametrize("private", [True, False])
    def test_existing_repository_receives_no_create_call(self, private):
        from soup_cli.monitoring.hf_push import REPO_EXISTING, ensure_hub_repo

        hub = _FakeHub(existing=True)
        outcome = ensure_hub_repo(hub, REPO, private=private)
        assert outcome == REPO_EXISTING
        assert hub.of("create_repo") == []
        sent = [kwargs for _name, kwargs in hub.calls]
        assert not any("private" in kwargs or "visibility" in kwargs for kwargs in sent), sent

    def test_failed_probe_never_creates_blind(self):
        from soup_cli.monitoring.hf_push import ensure_hub_repo

        hub = _FakeHub(probe_error=RuntimeError("hub unreachable"))
        with pytest.raises(RuntimeError, match="hub unreachable"):
            ensure_hub_repo(hub, REPO, private=True)
        assert hub.of("create_repo") == []


class TestDescribePushRepo:
    def test_created_private_line(self):
        from soup_cli.monitoring.hf_push import REPO_CREATED, describe_push_repo

        assert describe_push_repo(REPO, REPO_CREATED, private=True) == CREATED_PRIVATE

    def test_created_public_line(self):
        from soup_cli.monitoring.hf_push import REPO_CREATED, describe_push_repo

        assert describe_push_repo(REPO, REPO_CREATED, private=False) == CREATED_PUBLIC

    @pytest.mark.parametrize("private", [True, False])
    def test_existing_line_does_not_claim_a_visibility(self, private):
        from soup_cli.monitoring.hf_push import REPO_EXISTING, describe_push_repo

        line = describe_push_repo(REPO, REPO_EXISTING, private=private)
        assert line == EXISTING_UNCHANGED
        assert "private" not in line.lower().replace("visibility", "")
        assert "public" not in line.lower()


# --------------------------------------------------------------------------- #
# The defaults of the callback and its factory                                #
# --------------------------------------------------------------------------- #


class TestDefaultsArePrivate:
    def test_callback_default_is_private(self, monkeypatch):
        from soup_cli.monitoring.hf_push import HFPushCallback

        hub = _use_hub(monkeypatch, _FakeHub(existing=False))
        callback = HFPushCallback(repo_id=REPO, token="t1")
        assert callback.prepare_repo() == "created"
        assert hub.of("create_repo") == [{"repo_id": REPO, "private": True, "exist_ok": True}]

    def test_factory_default_is_private(self, monkeypatch, tmp_path):
        from soup_cli.monitoring.hf_push import build_push_callback

        hub = _use_hub(monkeypatch, _FakeHub(existing=False))
        callback = build_push_callback(repo_id=REPO, output_dir=str(tmp_path))
        assert callback is not None
        assert callback.prepare_repo() == "created"
        assert hub.of("create_repo") == [{"repo_id": REPO, "private": True, "exist_ok": True}]


# --------------------------------------------------------------------------- #
# soup train, end to end up to the trainer (trainer stubbed, Hub faked)       #
# --------------------------------------------------------------------------- #


class TestSoupTrainPushAs:
    def test_default_creates_a_private_repository(self, tmp_path, monkeypatch):
        hub = _use_hub(monkeypatch, _FakeHub(existing=False))
        result, hf_trainer = _run_train(tmp_path, monkeypatch, ["--push-as", REPO])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert hub.of("create_repo") == [{"repo_id": REPO, "private": True, "exist_ok": True}]
        output = _plain(result.output)
        assert CREATED_PRIVATE in output
        assert CREATED_PUBLIC not in output
        _attached_callback(hf_trainer)

    def test_push_public_creates_a_public_repository(self, tmp_path, monkeypatch):
        hub = _use_hub(monkeypatch, _FakeHub(existing=False))
        result, hf_trainer = _run_train(
            tmp_path, monkeypatch, ["--push-as", REPO, "--push-public"]
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert hub.of("create_repo") == [{"repo_id": REPO, "private": False, "exist_ok": True}]
        output = _plain(result.output)
        assert CREATED_PUBLIC in output
        assert CREATED_PRIVATE not in output
        _attached_callback(hf_trainer)

    @pytest.mark.parametrize("extra", [[], ["--push-public"]], ids=["default", "push-public"])
    def test_existing_repository_keeps_its_visibility(self, tmp_path, monkeypatch, extra):
        hub = _use_hub(monkeypatch, _FakeHub(existing=True))
        result, hf_trainer = _run_train(tmp_path, monkeypatch, ["--push-as", REPO, *extra])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert hub.of("create_repo") == []
        sent = [kwargs for _name, kwargs in hub.calls]
        assert not any("private" in kwargs or "visibility" in kwargs for kwargs in sent), sent
        output = _plain(result.output)
        assert EXISTING_UNCHANGED in output
        assert "Created private" not in output
        assert "Created public" not in output
        _attached_callback(hf_trainer)

    def test_the_callback_uploads_without_creating_the_repository_again(
        self, tmp_path, monkeypatch
    ):
        hub = _use_hub(monkeypatch, _FakeHub(existing=False))
        result, hf_trainer = _run_train(tmp_path, monkeypatch, ["--push-as", REPO])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        callback = _attached_callback(hf_trainer)
        assert len(hub.of("create_repo")) == 1
        assert len(hub.of("repo_exists")) == 1

        checkpoint = tmp_path / "out" / "checkpoint-20"
        checkpoint.mkdir(parents=True)
        (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
        args = types.SimpleNamespace(output_dir=str(tmp_path / "out"))
        control = types.SimpleNamespace()
        callback.on_train_begin(args, types.SimpleNamespace(), control)
        callback.on_save(args, types.SimpleNamespace(global_step=20, epoch=1), control)

        assert len(hub.of("create_repo")) == 1, hub.calls
        assert len(hub.of("repo_exists")) == 1, hub.calls
        uploads = hub.of("upload_folder")
        assert len(uploads) == 1
        assert uploads[0]["repo_id"] == REPO
        assert uploads[0]["revision"] == "checkpoint-20"

    def test_unreachable_hub_disables_auto_push_and_creates_nothing(
        self, tmp_path, monkeypatch
    ):
        hub = _use_hub(monkeypatch, _FakeHub(probe_error=RuntimeError("hub unreachable")))
        result, hf_trainer = _run_train(tmp_path, monkeypatch, ["--push-as", REPO])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert hub.of("create_repo") == []
        hf_trainer.add_callback.assert_not_called()
        output = _plain(result.output)
        assert (
            "--push-as: could not prepare user/my-model on the Hub; "
            "auto-push is disabled for this run" in output
        )
        assert "Created private" not in output

    def test_trainer_without_add_callback_does_not_touch_the_hub(
        self, tmp_path, monkeypatch
    ):
        import soup_cli.commands.train as train_cmd
        import soup_cli.trainer.dispatch as dispatch_mod

        hub = _use_hub(monkeypatch, _FakeHub(existing=False))
        config = _write_project(tmp_path)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "test.db"))
        stub = MagicMock()
        stub.trainer = object()  # no add_callback
        stub.train.return_value = {
            "initial_loss": 2.0, "final_loss": 1.0, "total_steps": 1,
            "duration_secs": 0.5, "duration": "0.5s", "output_dir": str(tmp_path / "out"),
        }
        monkeypatch.setattr(dispatch_mod, "build_trainer", lambda cfg, **kw: stub)
        monkeypatch.setattr(train_cmd, "_should_run_diagnose_gate_on_rank", lambda: True)
        result = runner.invoke(
            app, ["train", "--config", str(config), "--yes", "--push-as", REPO]
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert hub.calls == []
        assert "trainer does not expose add_callback" in _plain(result.output)


# --------------------------------------------------------------------------- #
# --push-public without --push-as                                             #
# --------------------------------------------------------------------------- #


class TestPushPublicNeedsPushAs:
    @pytest.mark.parametrize(
        "extra",
        [[], ["--dry-run"], ["--find-lr"], ["--cloud", "modal"]],
        ids=["plain", "dry-run", "find-lr", "cloud"],
    )
    def test_refused_naming_both_flags(self, tmp_path, monkeypatch, extra):
        hub = _use_hub(monkeypatch, _FakeHub())
        config = _write_project(tmp_path)
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["train", "--config", str(config), "--push-public", *extra])
        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert REFUSAL in _plain(result.output)
        assert hub.calls == []

    def test_refusal_does_not_need_the_config_to_exist(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(
            app, ["train", "--config", str(tmp_path / "absent.yaml"), "--push-public"]
        )
        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert REFUSAL in _plain(result.output)

    def test_push_public_with_push_as_is_accepted(self, tmp_path, monkeypatch):
        _use_hub(monkeypatch, _FakeHub(existing=False))
        result, _hf_trainer = _run_train(
            tmp_path, monkeypatch, ["--push-as", REPO, "--push-public"]
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert REFUSAL not in _plain(result.output)


# --------------------------------------------------------------------------- #
# --hf-resume and multi-GPU re-exec                                           #
# --------------------------------------------------------------------------- #


class TestHfResumeNeverCreatesARepository:
    def test_prepare_hf_resume_only_reads(self, tmp_path, monkeypatch):
        from soup_cli.monitoring import hf_push

        hub = _use_hub(monkeypatch, _FakeHub(existing=True, branches=("main", "checkpoint-5")))
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            hf_push, "_download_checkpoint", lambda **kwargs: str(tmp_path / "out" / "ckpt")
        )
        resumed = hf_push.prepare_hf_resume(REPO, str(tmp_path / "out"), token="t1")
        assert resumed is not None
        assert {name for name, _kwargs in hub.calls} == {"list_repo_refs"}

    def test_hf_resume_on_a_new_repository_trains_fresh_and_creates_it_private(
        self, tmp_path, monkeypatch
    ):
        hub = _use_hub(monkeypatch, _FakeHub(existing=False, branches=()))
        result, _hf_trainer = _run_train(
            tmp_path, monkeypatch, ["--push-as", REPO, "--hf-resume"]
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        output = _plain(result.output)
        assert "no checkpoint branch found; starting fresh" in output
        assert hub.of("create_repo") == [{"repo_id": REPO, "private": True, "exist_ok": True}]
        assert CREATED_PRIVATE in output

    def test_hf_resume_with_push_public_creates_it_public(self, tmp_path, monkeypatch):
        hub = _use_hub(monkeypatch, _FakeHub(existing=False, branches=()))
        result, _hf_trainer = _run_train(
            tmp_path, monkeypatch, ["--push-as", REPO, "--hf-resume", "--push-public"]
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert hub.of("create_repo") == [{"repo_id": REPO, "private": False, "exist_ok": True}]
        assert CREATED_PUBLIC in _plain(result.output)


class TestMultiGpuReexecKeepsPushPublic:
    def test_flag_follows_push_as_into_the_launched_command(self):
        from soup_cli.utils.launcher import collect_reexec_passthrough

        extra = collect_reexec_passthrough(push_as=REPO, push_public=True)
        assert extra[extra.index("--push-as") + 1] == REPO
        assert "--push-public" in extra

    def test_flag_is_absent_by_default(self):
        from soup_cli.utils.launcher import collect_reexec_passthrough

        assert "--push-public" not in collect_reexec_passthrough(push_as=REPO)

    @staticmethod
    def _printed_launch_command(tmp_path, monkeypatch, extra_args) -> str:
        """Run `soup train --gpus 4 --no-reexec` and return the printed launch command."""
        from soup_cli.utils import launcher as launcher_mod
        from soup_cli.utils import topology as topo_mod

        for var in (
            "RANK", "WORLD_SIZE", "LOCAL_RANK", "ACCELERATE_MIXED_PRECISION",
            "ACCELERATE_USE_DEEPSPEED", "ACCELERATE_USE_FSDP",
        ):
            monkeypatch.delenv(var, raising=False)
        config = _write_project(tmp_path)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            topo_mod, "detect_topology", lambda: {"gpu_count": 4, "interconnect": "PCIe"}
        )
        monkeypatch.setattr(topo_mod, "resolve_num_gpus", lambda spec: 4)
        monkeypatch.setattr(launcher_mod, "is_in_distributed", lambda: False)
        result = runner.invoke(
            app,
            ["train", "--config", str(config), "--gpus", "4", "--no-reexec", "--yes",
             *extra_args],
        )
        assert result.exit_code == 1, (result.output, repr(result.exception))
        printed = _plain(result.output)
        assert "Multi-GPU launch required" in printed, printed
        # Rich may wrap the command anywhere, so compare without whitespace.
        return printed.replace(" ", "")

    def test_the_train_command_hands_the_flag_to_the_launcher(self, tmp_path, monkeypatch):
        printed = self._printed_launch_command(
            tmp_path, monkeypatch, ["--push-as", REPO, "--push-public"]
        )
        assert f"--push-as{REPO}--push-public" in printed, printed

    def test_the_launch_command_has_no_flag_the_user_did_not_give(self, tmp_path, monkeypatch):
        printed = self._printed_launch_command(tmp_path, monkeypatch, ["--push-as", REPO])
        assert f"--push-as{REPO}" in printed, printed
        assert "--push-public" not in printed, printed


# --------------------------------------------------------------------------- #
# Help text                                                                   #
# --------------------------------------------------------------------------- #


class TestHelpText:
    @staticmethod
    def _help(name: str) -> str:
        from typer.main import get_command

        command = get_command(app).commands["train"]
        (param,) = [p for p in command.params if p.name == name]
        return " ".join((param.help or "").split())

    def test_push_as_help_states_the_visibility_rule(self):
        text = self._help("push_as")
        assert "created by this flag is private unless --push-public is given" in text
        assert "an existing repository is used as it is and its visibility is not changed" in text

    def test_push_public_help(self):
        text = self._help("push_public")
        assert "Requires --push-as" in text
        assert "public instead of private" in text
        assert "does not change" in text

    def test_push_public_is_a_switch_that_defaults_off(self):
        from typer.main import get_command

        command = get_command(app).commands["train"]
        (param,) = [p for p in command.params if p.name == "push_public"]
        assert param.opts == ["--push-public"]
        assert param.is_flag is True
        assert param.default is False
