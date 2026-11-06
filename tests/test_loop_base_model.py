"""The pre-wired loop trains the real base model, never a stand-in.

A ``registry://<id>`` served model used to make the train stage render
``base: hf-internal-testing/tiny-random-gpt2``: every iteration trained a
random 5 MB test model and reported success. The base now comes from
``soup loop init --base-model``, the registry entry's ``base_model``, or the
served model itself; when none is known, ``init --pre-wired`` and
``watch --pre-wired`` refuse and the train stage skips with a named reason.
"""

from __future__ import annotations

import json
import os
import re

import pytest
import yaml
from typer.testing import CliRunner

import soup_cli.utils.loop_stages as ls
from soup_cli.utils.loop_state import LoopState, read_state

runner = CliRunner()
_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def _clean(text: str) -> str:
    return " ".join(_ANSI.sub("", text).split())


@pytest.fixture
def cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "registry.db"))
    for name in ("SOUP_LOOP_TRACE_DIR", "SOUP_LOOP_THUMBS_DB", "SOUP_LOOP_SERVE_ENDPOINT"):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / "tasks.jsonl").write_text('{"prompt": "2+2", "expected": "4"}\n', encoding="utf-8")
    return tmp_path


def _push(base_model: str = "Qwen/Qwen2.5-0.5B-Instruct", name: str = "support-bot") -> str:
    from soup_cli.registry.store import RegistryStore

    with RegistryStore() as store:
        return store.push(
            name=name, tag="v1", base_model=base_model, task="sft", run_id=None,
            config={"base": base_model},
        )


def _state(**kwargs) -> LoopState:
    kwargs.setdefault("served_model", "registry://missing")
    return LoopState(eval_suite="e", baseline="b", **kwargs)


def _invoke(*args: str):
    from soup_cli.commands.loop import app

    result = runner.invoke(app, list(args))
    return result, _clean(result.output)


# ---------------------------------------------------------------------------
# resolve_base_model
# ---------------------------------------------------------------------------


def test_explicit_base_model_wins(cwd):
    assert ls.resolve_base_model("registry://missing", "org/explicit") == "org/explicit"


def test_plain_served_model_trains_from_itself(cwd):
    assert ls.resolve_base_model("./models/base") == "./models/base"
    assert ls.resolve_base_model("Qwen/Qwen2.5-0.5B-Instruct") == "Qwen/Qwen2.5-0.5B-Instruct"


def test_registry_served_model_resolves_to_entry_base(cwd):
    entry_id = _push("Qwen/Qwen2.5-0.5B-Instruct")
    assert ls.resolve_base_model(f"registry://{entry_id}") == "Qwen/Qwen2.5-0.5B-Instruct"
    assert ls.resolve_base_model("registry://support-bot:v1") == "Qwen/Qwen2.5-0.5B-Instruct"


def test_missing_registry_database_is_refused_and_not_created(cwd):
    with pytest.raises(ValueError, match="no registry database") as info:
        ls.resolve_base_model("registry://abc12")
    assert "--base-model" in str(info.value)
    assert not (cwd / "registry.db").exists()


def test_unknown_registry_entry_is_refused_by_ref(cwd):
    _push()
    with pytest.raises(ValueError, match="no registry entry matches 'nope'") as info:
        ls.resolve_base_model("registry://nope")
    assert "--base-model" in str(info.value)


def test_entry_without_a_base_model_is_refused(cwd):
    entry_id = _push(base_model="unknown")
    with pytest.raises(ValueError, match=f"registry entry {entry_id} records no base model"):
        ls.resolve_base_model(f"registry://{entry_id}")


# ---------------------------------------------------------------------------
# The train stage
# ---------------------------------------------------------------------------


def test_rendered_config_uses_the_resolved_base(cwd):
    entry_id = _push("Qwen/Qwen2.5-0.5B-Instruct")
    raw = yaml.safe_load(
        ls._render_dpo_yaml(_state(served_model=f"registry://{entry_id}"), "p.jsonl", "out")
    )
    assert raw["base"] == "Qwen/Qwen2.5-0.5B-Instruct"


def test_rendered_config_uses_loop_yaml_base_model(cwd):
    raw = yaml.safe_load(ls._render_dpo_yaml(_state(base_model="org/m"), "p.jsonl", "out"))
    assert raw["base"] == "org/m"


def test_unresolvable_base_never_renders_a_stand_in_model(cwd):
    with pytest.raises(ValueError, match="no registry database"):
        ls._render_dpo_yaml(_state(), "p.jsonl", "out")


def test_train_stage_skips_with_a_named_reason_when_base_is_unknown(cwd, monkeypatch):
    calls = []
    monkeypatch.setattr(ls, "_TRAIN_RUNNER", lambda argv, **kw: calls.append(argv))
    (cwd / "pairs.jsonl").write_text(
        json.dumps({"prompt": "q", "chosen": "a", "rejected": "b"}) + "\n", encoding="utf-8"
    )
    out = ls.train_dpo_from_pairs(_state(), {"pairs_path": "pairs.jsonl", "pairs_harvested": 1})
    assert out["skipped"] is True and out["run_id"] is None
    assert out["notes"].startswith("base_model_unknown: registry://missing: no registry database")
    assert calls == []
    assert not os.path.isdir(os.path.join(".soup-loops", "adapters"))


def test_train_stage_passes_the_resolved_base_to_soup_train(cwd, monkeypatch):
    seen = {}

    class _Done:
        returncode = 0

    def fake_run(argv, **kwargs):
        seen["yaml"] = yaml.safe_load(open(argv[argv.index("--config") + 1], encoding="utf-8"))
        return _Done()

    monkeypatch.setattr(ls, "_TRAIN_RUNNER", fake_run)
    (cwd / "pairs.jsonl").write_text(
        json.dumps({"prompt": "q", "chosen": "a", "rejected": "b"}) + "\n", encoding="utf-8"
    )
    entry_id = _push("HuggingFaceTB/SmolLM2-135M-Instruct")
    out = ls.train_dpo_from_pairs(
        _state(served_model=f"registry://{entry_id}"),
        {"pairs_path": "pairs.jsonl", "pairs_harvested": 1},
    )
    assert out["skipped"] is False
    assert seen["yaml"]["base"] == "HuggingFaceTB/SmolLM2-135M-Instruct"


# ---------------------------------------------------------------------------
# soup loop init / status / watch
# ---------------------------------------------------------------------------


def test_init_records_the_registry_entry_base(cwd):
    entry_id = _push("Qwen/Qwen2.5-0.5B-Instruct")
    result, text = _invoke(
        "init", f"registry://{entry_id}", "--eval", "tasks.jsonl", "--baseline", "b", "--pre-wired"
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert read_state().base_model == "Qwen/Qwen2.5-0.5B-Instruct"
    assert "base_model: Qwen/Qwen2.5-0.5B-Instruct" in text

    status, status_text = _invoke("status")
    assert status.exit_code == 0, (status.output, repr(status.exception))
    assert re.search(r"base_model\W+Qwen/Qwen2\.5-0\.5B-Instruct", status_text), status_text


def test_init_pre_wired_refuses_an_unknown_base(cwd):
    result, text = _invoke(
        "init", "registry://abc12", "--eval", "tasks.jsonl", "--baseline", "b", "--pre-wired"
    )
    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert "cannot determine the base model" in text
    assert "--base-model" in text
    assert not (cwd / ".soup" / "loop.yaml").exists()


def test_init_without_pre_wired_warns_and_keeps_going(cwd):
    result, text = _invoke("init", "registry://abc12", "--eval", "tasks.jsonl", "--baseline", "b")
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "base model unknown" in text
    assert "soup loop watch --pre-wired refuses" in text
    assert read_state().base_model is None


def test_init_base_model_flag_overrides_the_registry(cwd):
    result, _ = _invoke(
        "init", "registry://abc12", "--eval", "tasks.jsonl", "--baseline", "b",
        "--pre-wired", "--base-model", "org/explicit",
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert read_state().base_model == "org/explicit"


def test_init_refuses_an_empty_base_model(cwd):
    result, text = _invoke(
        "init", "./base", "--eval", "tasks.jsonl", "--baseline", "b", "--base-model", "  "
    )
    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert "--base-model must not be empty" in text


def test_init_plain_served_model_records_it_as_base(cwd):
    result, _ = _invoke("init", "./base", "--eval", "tasks.jsonl", "--baseline", "b", "--pre-wired")
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert read_state().base_model == "./base"


@pytest.mark.parametrize("detach", [False, True])
def test_watch_pre_wired_refuses_an_unknown_base_before_starting(cwd, monkeypatch, detach):
    _invoke("init", "registry://abc12", "--eval", "tasks.jsonl", "--baseline", "b")
    import subprocess

    def no_spawn(*args, **kwargs):
        raise AssertionError("watch must refuse before spawning a child")

    monkeypatch.setattr(subprocess, "Popen", no_spawn)
    args = ["watch", "--pre-wired", "--max-iterations", "1", "--poll-interval", "1"]
    if detach:
        args.append("--detach")
    result, text = _invoke(*args)
    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert "cannot determine the base model" in text
    assert read_state().iteration_count == 0


def test_state_file_without_base_model_still_loads(cwd):
    os.makedirs(".soup")
    legacy = {"served_model": "./base", "eval_suite": "e", "baseline": "b", "status": "stopped"}
    with open(os.path.join(".soup", "loop.yaml"), "w", encoding="utf-8") as fh:
        json.dump(legacy, fh)
    assert read_state().base_model is None


def test_base_model_field_is_validated():
    with pytest.raises(ValueError, match="base_model must not be empty"):
        _state(base_model="")
    with pytest.raises(ValueError, match="base_model must not contain NUL"):
        _state(base_model="a\x00b")


# ---------------------------------------------------------------------------
# A served LoRA adapter directory trains from its own base
# ---------------------------------------------------------------------------


def _adapter_dir(root, config) -> str:
    path = root / "adapters" / "chat"
    path.mkdir(parents=True)
    (path / "adapter_config.json").write_text(
        config if isinstance(config, str) else json.dumps(config), encoding="utf-8"
    )
    return str(path)


def test_adapter_dir_trains_from_its_base_model(cwd):
    served = _adapter_dir(cwd, {"base_model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct"})
    assert ls.resolve_base_model(served) == "Qwen/Qwen2.5-0.5B-Instruct"
    assert ls.resolve_base_model_source(served)[1] == f"adapter_config.json in {served}"


@pytest.mark.parametrize(
    "config",
    [{"base_model_name_or_path": ""}, {"r": 8}, ["not", "a", "mapping"], "{not json"],
    ids=["empty", "missing", "not-a-mapping", "invalid-json"],
)
def test_adapter_dir_without_a_base_is_refused(cwd, config):
    served = _adapter_dir(cwd, config)
    with pytest.raises(ValueError, match="adapter_config.json") as info:
        ls.resolve_base_model(served)
    assert "--base-model" in str(info.value)


def test_explicit_base_model_beats_the_adapter_config(cwd):
    served = _adapter_dir(cwd, {"base_model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct"})
    assert ls.resolve_base_model(served, "org/explicit") == "org/explicit"


def test_init_names_where_the_base_came_from(cwd):
    served = _adapter_dir(cwd, {"base_model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct"})
    result, text = _invoke(
        "init", served, "--eval", "tasks.jsonl", "--baseline", "b", "--pre-wired"
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert read_state().base_model == "Qwen/Qwen2.5-0.5B-Instruct"
    assert "Qwen/Qwen2.5-0.5B-Instruct (from adapter_config.json in" in text


def test_init_pre_wired_refuses_an_adapter_dir_without_a_base(cwd):
    served = _adapter_dir(cwd, {"r": 8})
    result, text = _invoke(
        "init", served, "--eval", "tasks.jsonl", "--baseline", "b", "--pre-wired"
    )
    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert "names no base_model_name_or_path" in text
