"""`soup train --push-as` creates its Hub repository private; `--push-public` opts out.

The rule under test:

* a repository that `--push-as <repo>` CREATES is private unless `--push-public` is given;
* `--push-public` without `--push-as` is refused, naming both flags;
* a repository that already exists is used as it is: nothing that could change its
  visibility is sent to the Hub, and the run says so, naming its visibility when it can be read;
* "Created ..." is printed only when the create call itself succeeded: a conflict answer
  from the Hub means the repository exists, so the line says that instead;
* when the Hub refuses to create a PRIVATE repository, the run says so and does not fall
  back to a public one by itself;
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
EXISTING_PRIVATE = (
    "Using existing Hub repository user/my-model (private); its visibility is unchanged."
)
EXISTING_PUBLIC = (
    "Using existing Hub repository user/my-model (public); its visibility is unchanged."
)
PRIVATE_REFUSED = (
    "--push-as: the Hub refused to create the private repository user/my-model "
    "(HTTP 402 or 403: a plan limit, or a token without permission to create it). "
    "Auto-push is disabled for this run. Pass --push-public to create a public repository "
    "instead, or create user/my-model yourself first and run again."
)
COULD_NOT_PREPARE = (
    "--push-as: could not prepare user/my-model on the Hub; auto-push is disabled for this run"
)
REFUSAL = (
    "--push-public requires --push-as <repo>: it sets the visibility of the "
    "repository --push-as creates, so it does nothing without it."
)


class _FakeResponse:
    """The slice of an HTTP response an `HfHubHTTPError` reads."""

    def __init__(self, status_code: int):
        self.status_code = status_code
        self.headers: dict = {}
        self.request = None
        self.text = ""
        self.url = "https://hub.invalid/api/repos/create"

    def json(self):
        raise ValueError("no body")


def _http_error(status: int, message: str = "hub answered"):
    """A real `huggingface_hub` HTTP error with the given status code."""
    from huggingface_hub.utils import HfHubHTTPError

    return HfHubHTTPError(message, response=_FakeResponse(status))


class _FakeHub:
    """Records every call the push path makes; models only what it touches.

    ``existing`` is what ``repo_exists`` answers. ``create_error`` is raised by
    ``create_repo`` after the call is recorded. ``info_private`` is what
    ``repo_info(...).private`` reads; ``info_error`` makes ``repo_info`` raise
    (a repository the token cannot read).
    """

    def __init__(self, *, existing: bool = False, probe_error: Exception | None = None,
                 branches: tuple[str, ...] = (), create_error: Exception | None = None,
                 info_private: object = None, info_error: Exception | None = None):
        self.existing = existing
        self.probe_error = probe_error
        self.branches = branches
        self.create_error = create_error
        self.info_private = info_private
        self.info_error = info_error
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

    def repo_info(self, **kwargs):
        self._record("repo_info", kwargs)
        if self.info_error is not None:
            raise self.info_error
        return types.SimpleNamespace(private=self.info_private)

    def create_repo(self, **kwargs):
        self._record("create_repo", kwargs)
        if self.create_error is not None:
            raise self.create_error

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


CREATE_PRIVATE = {"repo_id": REPO, "private": True, "exist_ok": False}
CREATE_PUBLIC = {"repo_id": REPO, "private": False, "exist_ok": False}


class TestEnsureHubRepo:
    def test_new_repository_is_created_private(self):
        from soup_cli.monitoring.hf_push import REPO_CREATED, ensure_hub_repo

        hub = _FakeHub(existing=False)
        state = ensure_hub_repo(hub, REPO, private=True)
        assert (state.outcome, state.private) == (REPO_CREATED, True)
        assert hub.of("create_repo") == [CREATE_PRIVATE]

    def test_new_repository_is_created_public_when_asked(self):
        from soup_cli.monitoring.hf_push import REPO_CREATED, ensure_hub_repo

        hub = _FakeHub(existing=False)
        state = ensure_hub_repo(hub, REPO, private=False)
        assert (state.outcome, state.private) == (REPO_CREATED, False)
        assert hub.of("create_repo") == [CREATE_PUBLIC]

    def test_a_just_created_repository_costs_no_extra_hub_call(self):
        from soup_cli.monitoring.hf_push import ensure_hub_repo

        hub = _FakeHub(existing=False)
        ensure_hub_repo(hub, REPO, private=True)
        assert [name for name, _kwargs in hub.calls] == ["repo_exists", "create_repo"]

    @pytest.mark.parametrize("private", [True, False])
    def test_existing_repository_receives_no_create_call(self, private):
        from soup_cli.monitoring.hf_push import REPO_EXISTING, ensure_hub_repo

        hub = _FakeHub(existing=True, info_private=False)
        state = ensure_hub_repo(hub, REPO, private=private)
        assert state.outcome == REPO_EXISTING
        assert hub.of("create_repo") == []
        sent = [kwargs for _name, kwargs in hub.calls]
        assert not any("private" in kwargs or "visibility" in kwargs for kwargs in sent), sent

    def test_failed_probe_never_creates_blind(self):
        from soup_cli.monitoring.hf_push import ensure_hub_repo

        hub = _FakeHub(probe_error=RuntimeError("hub unreachable"))
        with pytest.raises(RuntimeError, match="hub unreachable"):
            ensure_hub_repo(hub, REPO, private=True)
        assert hub.of("create_repo") == []

    # -- the visibility of an existing repository, when it can be read --

    @pytest.mark.parametrize(("readable", "expected"), [(True, True), (False, False)])
    def test_existing_repository_reports_the_visibility_it_has(self, readable, expected):
        from soup_cli.monitoring.hf_push import ensure_hub_repo

        hub = _FakeHub(existing=True, info_private=readable)
        state = ensure_hub_repo(hub, REPO, private=True)
        assert state.private is expected
        assert hub.of("repo_info") == [{"repo_id": REPO}]

    def test_unreadable_visibility_is_reported_as_unknown(self):
        from soup_cli.monitoring.hf_push import ensure_hub_repo

        hub = _FakeHub(existing=True, info_error=_http_error(403))
        state = ensure_hub_repo(hub, REPO, private=True)
        assert state.private is None

    @pytest.mark.parametrize("garbage", [None, "private", 1, object()])
    def test_a_visibility_that_is_not_a_bool_is_unknown(self, garbage):
        from soup_cli.monitoring.hf_push import ensure_hub_repo

        hub = _FakeHub(existing=True, info_private=garbage)
        assert ensure_hub_repo(hub, REPO, private=True).private is None

    # -- "Created" only when the create call itself succeeded --

    def test_conflict_from_the_hub_means_the_repository_exists(self):
        from soup_cli.monitoring.hf_push import REPO_EXISTING, ensure_hub_repo

        # The probe could not see the repository (a token without read access),
        # the Hub then answered 409 to the create.
        hub = _FakeHub(existing=False, create_error=_http_error(409), info_error=_http_error(404))
        state = ensure_hub_repo(hub, REPO, private=True)
        assert (state.outcome, state.private) == (REPO_EXISTING, None)
        assert hub.of("create_repo") == [CREATE_PRIVATE]

    @pytest.mark.parametrize("readable", [True, False])
    def test_create_race_reports_the_visibility_the_other_run_gave_it(self, readable):
        from soup_cli.monitoring.hf_push import REPO_EXISTING, ensure_hub_repo

        # Probe: missing. Another run creates it. Our create: 409. Its visibility is
        # whatever that run chose, not what we asked for.
        hub = _FakeHub(existing=False, create_error=_http_error(409), info_private=readable)
        state = ensure_hub_repo(hub, REPO, private=not readable)
        assert (state.outcome, state.private) == (REPO_EXISTING, readable)

    @pytest.mark.parametrize("status", [400, 401, 404, 429, 500])
    def test_other_http_errors_are_not_read_as_an_existing_repository(self, status):
        from soup_cli.monitoring.hf_push import ensure_hub_repo

        hub = _FakeHub(existing=False, create_error=_http_error(status))
        with pytest.raises(Exception) as excinfo:  # noqa: PT011 - the type is the hub's own
            ensure_hub_repo(hub, REPO, private=True)
        assert excinfo.value is hub.create_error

    def test_a_non_http_error_is_not_read_as_an_existing_repository(self):
        from soup_cli.monitoring.hf_push import ensure_hub_repo

        hub = _FakeHub(existing=False, create_error=RuntimeError("connection reset"))
        with pytest.raises(RuntimeError, match="connection reset"):
            ensure_hub_repo(hub, REPO, private=True)

    # -- a refused PRIVATE create never turns into a public one --

    @pytest.mark.parametrize("status", [402, 403])
    def test_refused_private_create_raises_and_is_not_retried_public(self, status):
        from soup_cli.monitoring.hf_push import PrivateRepositoryRefusedError, ensure_hub_repo

        hub = _FakeHub(existing=False, create_error=_http_error(status))
        with pytest.raises(PrivateRepositoryRefusedError) as excinfo:
            ensure_hub_repo(hub, REPO, private=True)
        assert excinfo.value.repo_id == REPO
        assert hub.of("create_repo") == [CREATE_PRIVATE]

    @pytest.mark.parametrize("status", [402, 403])
    def test_refused_public_create_is_not_called_a_private_refusal(self, status):
        from soup_cli.monitoring.hf_push import PrivateRepositoryRefusedError, ensure_hub_repo

        hub = _FakeHub(existing=False, create_error=_http_error(status))
        with pytest.raises(Exception) as excinfo:  # noqa: PT011
            ensure_hub_repo(hub, REPO, private=False)
        assert not isinstance(excinfo.value, PrivateRepositoryRefusedError)
        assert hub.of("create_repo") == [CREATE_PUBLIC]


class TestDescribePushRepo:
    @staticmethod
    def _state(outcome, private):
        from soup_cli.monitoring.hf_push import RepoState

        return RepoState(outcome, private)

    def test_created_private_line(self):
        from soup_cli.monitoring.hf_push import REPO_CREATED, describe_push_repo

        assert describe_push_repo(REPO, self._state(REPO_CREATED, True)) == CREATED_PRIVATE

    def test_created_public_line(self):
        from soup_cli.monitoring.hf_push import REPO_CREATED, describe_push_repo

        assert describe_push_repo(REPO, self._state(REPO_CREATED, False)) == CREATED_PUBLIC

    @pytest.mark.parametrize(
        ("private", "expected"),
        [(True, EXISTING_PRIVATE), (False, EXISTING_PUBLIC), (None, EXISTING_UNCHANGED)],
        ids=["private", "public", "unreadable"],
    )
    def test_existing_line_names_the_visibility_only_when_it_is_known(self, private, expected):
        from soup_cli.monitoring.hf_push import REPO_EXISTING, describe_push_repo

        assert describe_push_repo(REPO, self._state(REPO_EXISTING, private)) == expected

    def test_an_unknown_outcome_is_refused_not_reported_as_created(self):
        from soup_cli.monitoring.hf_push import describe_push_repo

        with pytest.raises(ValueError, match="unknown repository outcome"):
            describe_push_repo(REPO, self._state("renamed", True))


class TestDescribePrepareFailure:
    def test_private_refusal_names_the_repository_and_both_ways_out(self):
        from soup_cli.monitoring.hf_push import (
            PrivateRepositoryRefusedError,
            describe_prepare_failure,
        )

        line = describe_prepare_failure(REPO, PrivateRepositoryRefusedError(REPO))
        assert line == PRIVATE_REFUSED

    @pytest.mark.parametrize(
        "error", [None, RuntimeError("boom"), _http_error(500)], ids=["none", "runtime", "http"]
    )
    def test_anything_else_gets_the_generic_line_without_the_error_text(self, error):
        from soup_cli.monitoring.hf_push import describe_prepare_failure

        assert describe_prepare_failure(REPO, error) == COULD_NOT_PREPARE


# --------------------------------------------------------------------------- #
# The defaults of the callback and its factory                                #
# --------------------------------------------------------------------------- #


class TestDefaultsArePrivate:
    def test_callback_default_is_private(self, monkeypatch):
        from soup_cli.monitoring.hf_push import HFPushCallback

        hub = _use_hub(monkeypatch, _FakeHub(existing=False))
        callback = HFPushCallback(repo_id=REPO, token="t1")
        assert callback.prepare_repo().outcome == "created"
        assert hub.of("create_repo") == [CREATE_PRIVATE]

    def test_factory_default_is_private(self, monkeypatch, tmp_path):
        from soup_cli.monitoring.hf_push import build_push_callback

        hub = _use_hub(monkeypatch, _FakeHub(existing=False))
        callback = build_push_callback(repo_id=REPO, output_dir=str(tmp_path))
        assert callback is not None
        assert callback.prepare_repo().outcome == "created"
        assert hub.of("create_repo") == [CREATE_PRIVATE]


# --------------------------------------------------------------------------- #
# soup train, end to end up to the trainer (trainer stubbed, Hub faked)       #
# --------------------------------------------------------------------------- #


class TestSoupTrainPushAs:
    def test_default_creates_a_private_repository(self, tmp_path, monkeypatch):
        hub = _use_hub(monkeypatch, _FakeHub(existing=False))
        result, hf_trainer = _run_train(tmp_path, monkeypatch, ["--push-as", REPO])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert hub.of("create_repo") == [CREATE_PRIVATE]
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
        assert hub.of("create_repo") == [CREATE_PUBLIC]
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


class TestSoupTrainPushAsReportsWhatHappened:
    """The printed line follows what the Hub did, not what the run asked for."""

    @pytest.mark.parametrize("extra", [[], ["--push-public"]], ids=["default", "push-public"])
    def test_a_created_repository_costs_no_visibility_lookup(self, tmp_path, monkeypatch, extra):
        hub = _use_hub(monkeypatch, _FakeHub(existing=False))
        result, _hf_trainer = _run_train(tmp_path, monkeypatch, ["--push-as", REPO, *extra])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert hub.of("repo_info") == []

    @pytest.mark.parametrize("extra", [[], ["--push-public"]], ids=["default", "push-public"])
    @pytest.mark.parametrize(
        ("is_private", "line"),
        [(True, EXISTING_PRIVATE), (False, EXISTING_PUBLIC)],
        ids=["private", "public"],
    )
    def test_an_existing_repository_is_reported_with_the_visibility_it_has(
        self, tmp_path, monkeypatch, extra, is_private, line
    ):
        hub = _use_hub(monkeypatch, _FakeHub(existing=True, info_private=is_private))
        result, hf_trainer = _run_train(tmp_path, monkeypatch, ["--push-as", REPO, *extra])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        output = _plain(result.output)
        assert line in output
        assert "Created" not in output
        assert hub.of("create_repo") == []
        _attached_callback(hf_trainer)

    def test_an_unreadable_existing_repository_keeps_the_plain_line(self, tmp_path, monkeypatch):
        hub = _use_hub(monkeypatch, _FakeHub(existing=True, info_error=_http_error(403)))
        result, hf_trainer = _run_train(tmp_path, monkeypatch, ["--push-as", REPO])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        output = _plain(result.output)
        assert EXISTING_UNCHANGED in output
        assert "(private)" not in output
        assert "(public)" not in output
        assert hub.of("create_repo") == []
        _attached_callback(hf_trainer)

    def test_a_repository_the_probe_cannot_see_is_not_reported_as_created(
        self, tmp_path, monkeypatch
    ):
        hub = _use_hub(
            monkeypatch,
            _FakeHub(existing=False, create_error=_http_error(409), info_error=_http_error(404)),
        )
        result, hf_trainer = _run_train(tmp_path, monkeypatch, ["--push-as", REPO])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        output = _plain(result.output)
        assert EXISTING_UNCHANGED in output
        assert "Created" not in output
        assert hub.of("create_repo") == [CREATE_PRIVATE]
        _attached_callback(hf_trainer)

    def test_a_create_race_is_reported_with_the_visibility_the_other_run_chose(
        self, tmp_path, monkeypatch
    ):
        # We asked for private; the run that got there first made it public.
        hub = _use_hub(
            monkeypatch,
            _FakeHub(existing=False, create_error=_http_error(409), info_private=False),
        )
        result, hf_trainer = _run_train(tmp_path, monkeypatch, ["--push-as", REPO])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        output = _plain(result.output)
        assert EXISTING_PUBLIC in output
        assert "Created" not in output
        assert CREATED_PRIVATE not in output
        assert hub.of("create_repo") == [CREATE_PRIVATE]
        _attached_callback(hf_trainer)

    @pytest.mark.parametrize("status", [402, 403])
    def test_a_refused_private_create_is_named_and_never_retried_public(
        self, tmp_path, monkeypatch, status
    ):
        hub = _use_hub(monkeypatch, _FakeHub(existing=False, create_error=_http_error(status)))
        result, hf_trainer = _run_train(tmp_path, monkeypatch, ["--push-as", REPO])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        output = _plain(result.output)
        assert PRIVATE_REFUSED in output
        assert "Created" not in output
        assert hub.of("create_repo") == [CREATE_PRIVATE]
        assert not [kwargs for kwargs in hub.of("create_repo") if kwargs["private"] is False]
        hf_trainer.add_callback.assert_not_called()

    @pytest.mark.parametrize("status", [402, 403, 500])
    def test_a_refused_public_create_does_not_claim_a_private_refusal(
        self, tmp_path, monkeypatch, status
    ):
        hub = _use_hub(monkeypatch, _FakeHub(existing=False, create_error=_http_error(status)))
        result, hf_trainer = _run_train(
            tmp_path, monkeypatch, ["--push-as", REPO, "--push-public"]
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        output = _plain(result.output)
        assert COULD_NOT_PREPARE in output
        assert "refused to create the private repository" not in output
        assert hub.of("create_repo") == [CREATE_PUBLIC]
        hf_trainer.add_callback.assert_not_called()

    def test_the_refusal_line_does_not_echo_the_hub_error_text(self, tmp_path, monkeypatch):
        secret = "token hf_secret_value rejected"
        _use_hub(
            monkeypatch,
            _FakeHub(existing=False, create_error=_http_error(403, message=secret)),
        )
        result, _hf_trainer = _run_train(tmp_path, monkeypatch, ["--push-as", REPO])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "hf_secret_value" not in _plain(result.output)


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
        assert hub.of("create_repo") == [CREATE_PRIVATE]
        assert CREATED_PRIVATE in output

    def test_hf_resume_with_push_public_creates_it_public(self, tmp_path, monkeypatch):
        hub = _use_hub(monkeypatch, _FakeHub(existing=False, branches=()))
        result, _hf_trainer = _run_train(
            tmp_path, monkeypatch, ["--push-as", REPO, "--hf-resume", "--push-public"]
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert hub.of("create_repo") == [CREATE_PUBLIC]
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
        # Rich may wrap the command anywhere inside the panel, and the wrap puts the
        # panel's border characters between two halves of an argument, so compare
        # without whitespace and without box-drawing characters (U+2500..U+257F).
        return "".join(
            char for char in printed
            if not char.isspace() and not 0x2500 <= ord(char) <= 0x257F
        )

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
