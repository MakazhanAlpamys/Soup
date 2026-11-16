"""The optimizer state of an RL checkpoint is loaded with ``weights_only=True``.

``RLCheckpointCallback.restore_checkpoint`` (used by the ``pid_lagrangian`` rollback
ladder) reads ``<output>/rl-checkpoints/step-<N>/optimizer.pt``. That file is a torch
pickle. It used to be read with ``weights_only=False``, which runs whatever a pickle
asks for; the symlink check next to it only covered one way of planting a file.

What these tests pin:

* a round trip written by the real ``save_checkpoint`` is restored (the saved object is
  plain tensors and containers, so ``weights_only=True`` reads it);
* a file written the same way before this change still loads;
* a pickle that needs code to load is refused by name, the code does not run, and the
  run is not crashed (``restore_checkpoint`` still never raises);
* a truncated or non-optimizer file is refused by name too, never loaded some other way;
* the file is only read when the config says the checkpoints carry optimizer state,
  and only from inside the run directory;
* a save refuses to write a state the safe loader could not read back.

The "payload" below only creates a marker file under ``tmp_path``.
"""

from __future__ import annotations

import ast
import logging
import os
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src" / "soup_cli" / "utils" / "rl_checkpoint.py"


def _run_payload(marker: str) -> str:
    """What a pickle that needs code to load would call. Harmless here: it writes one small file."""
    Path(marker).write_text("payload ran", encoding="utf-8")
    return "ran"


class _Payload:
    """A pickle whose load calls ``_run_payload``; ``weights_only=True`` must refuse it."""

    def __init__(self, marker: Path) -> None:
        self.marker = str(marker)

    def __reduce__(self):
        return (_run_payload, (self.marker,))


@pytest.fixture
def cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # the run dir of the callback must stay under cwd
    return tmp_path


def _adamw(steps: int = 1):
    import torch

    model = torch.nn.Linear(4, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(steps):
        opt.zero_grad()
        model(torch.randn(3, 4)).sum().backward()
        opt.step()
    return opt


def _callback(*, include_optimizer: bool = True, output_dir: str = "run"):
    from soup_cli.utils.rl_checkpoint import RLCheckpointConfig, build_rl_checkpoint_callback

    return build_rl_checkpoint_callback(
        RLCheckpointConfig(save_every_steps=1, include_optimizer_state=include_optimizer),
        output_dir=output_dir,
        task="grpo",
    )


def _step_dir(output_dir: str = "run", step: int = 1) -> str:
    path = os.path.join(output_dir, "rl-checkpoints", f"step-{step}")
    os.makedirs(path, exist_ok=True)
    return path


def _optimizer_file(output_dir: str = "run", step: int = 1) -> str:
    return os.path.join(_step_dir(output_dir, step), "optimizer.pt")


def _snapshot(opt) -> dict:
    """A detached copy of the numeric optimizer state, comparable with ``_same``."""
    out = {}
    for index, state in opt.state_dict()["state"].items():
        out[index] = {key: value.clone() for key, value in state.items()}
    return out


def _same(first: dict, second: dict) -> bool:
    import torch

    if first.keys() != second.keys():
        return False
    return all(
        first[idx].keys() == second[idx].keys()
        and all(torch.equal(first[idx][key], second[idx][key]) for key in first[idx])
        for idx in first
    )


def _message(caplog) -> str:
    return " ".join(rec.getMessage() for rec in caplog.records)


# --------------------------------------------------------------------------- round trips


def test_round_trip_written_by_the_real_save_is_restored(cwd):
    import torch

    from soup_cli.utils.rl_checkpoint import load_optimizer_state

    cb = _callback()
    opt = _adamw()
    saved = _snapshot(opt)
    ckpt_dir = cb.save_checkpoint(step=1, model=None, optimizer=opt)
    opt_path = os.path.join(ckpt_dir, "optimizer.pt")

    # What was saved is plain: the strict loader itself reads it.
    assert set(torch.load(opt_path, map_location="cpu", weights_only=True)) == {
        "state",
        "param_groups",
    }
    assert set(load_optimizer_state(opt_path)) == {"state", "param_groups"}

    # Drift the optimizer, then roll back.
    model_param = opt.param_groups[0]["params"][0]
    model_param.grad = torch.ones_like(model_param)
    opt.step()
    assert not _same(_snapshot(opt), saved)

    assert cb.restore_checkpoint(step=1, model=None, optimizer=opt) is True
    assert _same(_snapshot(opt), saved)


def _stock_optimizer(name: str, params):
    import torch

    if name == "sgd-momentum":
        return torch.optim.SGD(params, lr=0.1, momentum=0.9)
    if name == "adam-amsgrad":
        return torch.optim.Adam(params, lr=1e-3, amsgrad=True)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=1e-3, momentum=0.5)
    return torch.optim.Adagrad(params, lr=1e-2)


@pytest.mark.parametrize("name", ["sgd-momentum", "adam-amsgrad", "rmsprop", "adagrad"])
def test_other_stock_optimizers_round_trip_through_the_strict_loader(cwd, name):
    import torch

    param = torch.nn.Parameter(torch.zeros(3))
    opt = _stock_optimizer(name, [param])
    param.grad = torch.ones(3)
    opt.step()
    saved = _snapshot(opt)
    cb = _callback()
    ckpt_dir = cb.save_checkpoint(step=1, model=None, optimizer=opt)
    assert os.path.isfile(os.path.join(ckpt_dir, "optimizer.pt"))

    param.grad = torch.ones(3) * 5
    opt.step()
    assert cb.restore_checkpoint(step=1, model=None, optimizer=opt) is True
    assert _same(_snapshot(opt), saved)


def test_a_plain_file_in_the_old_layout_still_loads(cwd):
    """The old save was ``torch.save(optimizer.state_dict(), optimizer.pt)``: unchanged."""
    import torch

    opt = _adamw()
    saved = _snapshot(opt)
    torch.save(opt.state_dict(), _optimizer_file())
    param = opt.param_groups[0]["params"][0]
    param.grad = torch.ones_like(param)
    opt.step()

    cb = _callback()
    assert cb.restore_checkpoint(step=1, model=None, optimizer=opt) is True
    assert _same(_snapshot(opt), saved)


# --------------------------------------------------------------------------- pickle that needs code


def _write_payload_file(path: str, marker: Path, *, shape: str) -> None:
    import pickle

    import torch

    if shape == "bare":
        torch.save(_Payload(marker), path)
    elif shape == "inside-a-state-dict":
        state = _adamw().state_dict()
        state["extra"] = _Payload(marker)
        torch.save(state, path)
    elif shape == "legacy-format":
        # The pre-zip torch layout older torch versions wrote.
        torch.save(_Payload(marker), path, _use_new_zipfile_serialization=False)
    elif shape == "raw-pickle":
        with open(path, "wb") as handle:
            pickle.dump(_Payload(marker), handle)
    else:  # pragma: no cover
        raise AssertionError(shape)


SHAPES = ["bare", "inside-a-state-dict", "legacy-format"]


@pytest.mark.parametrize("shape", SHAPES)
def test_the_payload_fixture_is_live_when_loaded_the_unsafe_way(cwd, shape):
    """Control: without this, 'the marker is absent' below could pass vacuously."""
    import torch

    marker = cwd / "marker-control"
    opt_path = _optimizer_file()
    _write_payload_file(opt_path, marker, shape=shape)
    assert not marker.exists()

    torch.load(opt_path, map_location="cpu", weights_only=False)

    assert marker.read_text(encoding="utf-8") == "payload ran"


@pytest.mark.parametrize("shape", SHAPES)
def test_restore_refuses_a_pickle_that_needs_code_and_does_not_run_it(cwd, shape, caplog):
    marker = cwd / "marker-restore"
    opt = _adamw()
    before = _snapshot(opt)
    _write_payload_file(_optimizer_file(), marker, shape=shape)
    cb = _callback()

    with caplog.at_level(logging.WARNING):
        restored = cb.restore_checkpoint(step=1, model=None, optimizer=opt)

    assert restored is False
    assert not marker.exists(), "the payload ran: the file was loaded with weights_only=False"
    assert _same(_snapshot(opt), before), "a refused file must not touch the optimizer"
    text = _message(caplog)
    assert "optimizer.pt" in text and "was refused" in text, text


@pytest.mark.parametrize("shape", SHAPES)
def test_load_optimizer_state_raises_the_named_error_and_does_not_run_the_payload(cwd, shape):
    from soup_cli.utils.rl_checkpoint import RLCheckpointLoadError, load_optimizer_state

    marker = cwd / "marker-loader"
    opt_path = _optimizer_file()
    _write_payload_file(opt_path, marker, shape=shape)

    with pytest.raises(RLCheckpointLoadError) as excinfo:
        load_optimizer_state(opt_path)

    assert not marker.exists()
    text = str(excinfo.value)
    # Names the file, says it was refused and why, and says what to do.
    assert "optimizer.pt" in text, text
    assert "was refused" in text, text
    assert "tensors and plain containers" in text, text
    assert "Delete this file" in text, text
    assert "written by soup" in text, text
    assert "Traceback" not in text, text
    # A named refusal is a ValueError a caller can catch; it is not a bare pickle error.
    assert isinstance(excinfo.value, ValueError)


def test_a_raw_pickle_that_is_not_a_torch_file_is_refused_too(cwd):
    from soup_cli.utils.rl_checkpoint import RLCheckpointLoadError, load_optimizer_state

    marker = cwd / "marker-raw"
    opt_path = _optimizer_file()
    _write_payload_file(opt_path, marker, shape="raw-pickle")

    with pytest.raises(RLCheckpointLoadError, match="was refused"):
        load_optimizer_state(opt_path)
    assert not marker.exists()


def test_the_loader_never_falls_back_to_the_unsafe_load():
    """Cheap first line: no ``weights_only=False`` in the module. The tests above are the proof."""
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    offenders = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.keyword)
        and node.arg == "weights_only"
        and not (isinstance(node.value, ast.Constant) and node.value.value is True)
    ]
    assert offenders == [], f"weights_only is not literally True at lines {offenders}"


# --------------------------------------------------------------------------- bad files


def test_a_truncated_file_is_refused_by_name_not_a_traceback(cwd, caplog):
    from soup_cli.utils.rl_checkpoint import RLCheckpointLoadError, load_optimizer_state

    cb = _callback()
    opt = _adamw()
    ckpt_dir = cb.save_checkpoint(step=1, model=None, optimizer=opt)
    opt_path = os.path.join(ckpt_dir, "optimizer.pt")
    data = Path(opt_path).read_bytes()
    assert len(data) > 200
    Path(opt_path).write_bytes(data[: len(data) // 2])
    before = _snapshot(opt)

    with pytest.raises(RLCheckpointLoadError) as excinfo:
        load_optimizer_state(opt_path)
    text = str(excinfo.value)
    assert "optimizer.pt" in text and "truncated or corrupt" in text, text
    assert "Delete this file" in text and "written by soup" in text, text

    with caplog.at_level(logging.WARNING):
        restored = cb.restore_checkpoint(step=1, model=None, optimizer=opt)
    assert restored is False
    assert "truncated or corrupt" in _message(caplog)
    assert _same(_snapshot(opt), before)


def test_an_empty_file_is_refused_by_name(cwd):
    from soup_cli.utils.rl_checkpoint import RLCheckpointLoadError, load_optimizer_state

    opt_path = _optimizer_file()
    Path(opt_path).write_bytes(b"")

    with pytest.raises(RLCheckpointLoadError, match="truncated or corrupt"):
        load_optimizer_state(opt_path)


@pytest.mark.parametrize("shape", ["tensor", "dict-without-optimizer-keys"])
def test_a_plain_file_that_is_not_an_optimizer_state_is_refused(cwd, shape):
    import torch

    from soup_cli.utils.rl_checkpoint import RLCheckpointLoadError, load_optimizer_state

    opt_path = _optimizer_file()
    torch.save(
        torch.ones(3) if shape == "tensor" else {"weights": torch.ones(3)},
        opt_path,
    )

    with pytest.raises(RLCheckpointLoadError, match="not an optimizer state dict") as excinfo:
        load_optimizer_state(opt_path)
    assert "optimizer.pt" in str(excinfo.value)


def test_a_missing_file_is_refused_by_name(cwd):
    from soup_cli.utils.rl_checkpoint import RLCheckpointLoadError, load_optimizer_state

    with pytest.raises(RLCheckpointLoadError, match="cannot be read"):
        load_optimizer_state(os.path.join("run", "rl-checkpoints", "step-1", "optimizer.pt"))


def test_a_state_the_optimizer_cannot_apply_is_logged_and_not_a_crash(cwd, caplog):
    """Plain and well-formed, but for different parameters: restore must not raise."""
    import torch

    saved_opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(2))], lr=0.1, momentum=0.9)
    saved_opt.param_groups[0]["params"][0].grad = torch.ones(2)
    saved_opt.step()
    torch.save(saved_opt.state_dict(), _optimizer_file())
    two_groups = torch.optim.SGD(
        [
            {"params": [torch.nn.Parameter(torch.zeros(2))]},
            {"params": [torch.nn.Parameter(torch.zeros(2))]},
        ],
        lr=0.1,
    )
    cb = _callback()

    with caplog.at_level(logging.WARNING):
        restored = cb.restore_checkpoint(step=1, model=None, optimizer=two_groups)

    assert restored is False
    assert "could not be applied" in _message(caplog), _message(caplog)


# --------------------------------------------------------------------------- when it is read


def test_restore_does_not_read_optimizer_pt_when_the_config_excludes_it(cwd):
    """include_optimizer_state=False: a file already sitting there was not written by this run."""
    import torch

    opt = _adamw()
    planted = _adamw(steps=3)
    torch.save(planted.state_dict(), _optimizer_file())
    before = _snapshot(opt)
    cb = _callback(include_optimizer=False)

    assert cb.restore_checkpoint(step=1, model=None, optimizer=opt) is False
    assert _same(_snapshot(opt), before), "optimizer.pt was read although the config excludes it"


def test_a_planted_pickle_next_to_a_run_that_saves_no_optimizer_state_never_runs(cwd):
    marker = cwd / "marker-excluded"
    opt = _adamw()
    cb = _callback(include_optimizer=False)
    # The run saves step 1 without optimizer state; a file was already in that directory.
    _write_payload_file(_optimizer_file(), marker, shape="bare")
    cb.save_checkpoint(step=1, model=None, optimizer=opt)

    assert cb.restore_checkpoint(step=1, model=None, optimizer=opt) is False
    assert not marker.exists()


@pytest.mark.requires_symlink
@pytest.mark.parametrize("target", ["outside-the-run-dir", "inside-the-run-dir"])
def test_a_symlinked_optimizer_file_is_still_refused(cwd, target, caplog):
    """A link is refused even when its target is a valid file inside the run directory."""
    import torch

    opt = _adamw()
    link = _optimizer_file()
    if target == "outside-the-run-dir":
        elsewhere = cwd / "elsewhere.pt"
    else:
        elsewhere = cwd / "run" / "other-state.pt"
    torch.save(_adamw(steps=3).state_dict(), elsewhere)
    os.symlink(str(elsewhere), link)
    before = _snapshot(opt)
    cb = _callback()

    with caplog.at_level(logging.WARNING):
        restored = cb.restore_checkpoint(step=1, model=None, optimizer=opt)

    assert restored is False
    assert _same(_snapshot(opt), before)
    assert "it is a symlink" in _message(caplog), _message(caplog)


# --------------------------------------------------------------------------- containment


def test_an_output_dir_that_climbs_out_of_cwd_is_refused_at_build(cwd):
    from soup_cli.utils.rl_checkpoint import RLCheckpointConfig, build_rl_checkpoint_callback

    with pytest.raises(ValueError, match="output_dir must stay under the current directory"):
        build_rl_checkpoint_callback(
            RLCheckpointConfig(save_every_steps=1),
            output_dir=os.path.join("run", "..", "..", "escape"),
            task="grpo",
        )


@pytest.mark.requires_symlink
def test_a_checkpoint_directory_linked_to_outside_the_run_dir_is_not_read(cwd, caplog):
    """``rl-checkpoints`` -> a directory elsewhere: a valid plain file there is still refused."""
    import torch

    opt = _adamw()
    outside = cwd / "outside"
    (outside / "step-1").mkdir(parents=True)
    torch.save(_adamw(steps=3).state_dict(), outside / "step-1" / "optimizer.pt")
    (cwd / "run").mkdir()
    os.symlink(str(outside), os.path.join("run", "rl-checkpoints"), target_is_directory=True)
    before = _snapshot(opt)
    cb = _callback()

    with caplog.at_level(logging.WARNING):
        restored = cb.restore_checkpoint(step=1, model=None, optimizer=opt)

    assert restored is False
    assert _same(_snapshot(opt), before)
    text = _message(caplog)
    assert "optimizer.pt" in text and "outside the run directory" in text, text


@pytest.mark.requires_symlink
def test_a_step_directory_linked_to_outside_the_run_dir_is_not_read(cwd, caplog):
    import torch

    opt = _adamw()
    outside = cwd / "outside-step"
    outside.mkdir()
    torch.save(_adamw(steps=3).state_dict(), outside / "optimizer.pt")
    os.makedirs(os.path.join("run", "rl-checkpoints"))
    os.symlink(
        str(outside), os.path.join("run", "rl-checkpoints", "step-1"), target_is_directory=True
    )
    before = _snapshot(opt)
    cb = _callback()

    with caplog.at_level(logging.WARNING):
        restored = cb.restore_checkpoint(step=1, model=None, optimizer=opt)

    assert restored is False
    assert _same(_snapshot(opt), before)
    assert "outside the run directory" in _message(caplog), _message(caplog)


# --------------------------------------------------------------------------- what is saved


def test_is_plain_state_accepts_what_the_strict_loader_reads():
    import torch

    from soup_cli.utils.rl_checkpoint import _is_plain_state

    state = {
        "state": {0: {"step": torch.tensor(1.0), "exp_avg": torch.zeros(2)}},
        "param_groups": [
            {
                "lr": 1e-3,
                "betas": (0.9, 0.999),
                "amsgrad": False,
                "foreach": None,
                "name": "g0",
                "params": [0],
            },
        ],
    }
    assert _is_plain_state(state) is True


@pytest.mark.parametrize(
    "value",
    [
        object(),
        {1, 2},
        lambda: None,
        {"a": {"b": object()}},
        [1, [2, [3, object()]]],
        {object(): 1},
        type("Sub", (dict,), {})(),
    ],
    ids=["object", "set", "lambda", "nested-dict", "nested-list", "object-key", "dict-subclass"],
)
def test_is_plain_state_rejects_anything_else(value):
    from soup_cli.utils.rl_checkpoint import _is_plain_state

    assert _is_plain_state(value) is False


def test_is_plain_state_stops_on_a_self_referencing_container():
    from soup_cli.utils.rl_checkpoint import _is_plain_state

    loop: list = []
    loop.append(loop)
    assert _is_plain_state(loop) is False


def test_a_save_refuses_to_write_a_state_the_strict_loader_could_not_read(cwd, caplog):
    import json

    class _Weird:
        pass

    opt = _adamw()
    opt.param_groups[0]["note"] = _Weird()
    cb = _callback()

    with caplog.at_level(logging.WARNING):
        ckpt_dir = cb.save_checkpoint(step=1, model=None, optimizer=opt)

    assert sorted(os.listdir(ckpt_dir)) == ["manifest.json"], os.listdir(ckpt_dir)
    manifest = json.loads(Path(ckpt_dir, "manifest.json").read_text(encoding="utf-8"))
    assert manifest["has_optimizer"] is False
    text = _message(caplog)
    assert "not saved" in text and "tensors and plain containers" in text, text


def test_a_save_that_cannot_write_leaves_no_temp_file(cwd):
    class _Weird:
        pass

    opt = _adamw()
    opt.param_groups[0]["note"] = _Weird()
    ckpt_dir = _callback().save_checkpoint(step=1, model=None, optimizer=opt)

    assert not any(name.endswith(".tmp") for name in os.listdir(ckpt_dir))
