"""Process-isolated MLX teacher capture tests for AutoDistill Milestone B1."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import os
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from soup_cli.autodistill.contract import (
    AutoDistillPlan,
    CaptureToken,
    ShardManifest,
    build_plan_estimate,
    canonical_json_bytes,
    canonicalize_jsonl_bytes,
)
from soup_cli.autodistill.mlx_worker import (
    MlxTeacherCaptureResult,
    run_mlx_teacher_capture_process,
)

FIXTURES = Path(__file__).parent / "fixtures" / "autodistill" / "v1"
_MLX_VERSION = "0.31.0-test"


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_fake_mlx_runtime(
    root: Path,
    *,
    mlx_lm_version: str = _MLX_VERSION,
    parameter_dtypes: tuple[str, ...] = ("float32",),
) -> Path:
    fake_root = root / "fake-runtime"
    mlx_root = fake_root / "mlx"
    mlx_lm_root = fake_root / "mlx_lm"
    mlx_root.mkdir(parents=True)
    mlx_lm_root.mkdir()
    (mlx_root / "__init__.py").write_text('__version__ = "0.30.0-test"\n', encoding="utf-8")
    (mlx_root / "core.py").write_text(
        """import os

float32 = "float32"

def _trace(event):
    with open(os.environ["SOUP_TEST_MLX_TRACE"], "a", encoding="utf-8") as handle:
        handle.write(event + "\\n")

def array(value):
    return value

def astype(value, dtype):
    assert dtype == float32
    _trace("astype:float32")
    return value

def eval(value):
    _trace("eval")

def clear_cache():
    _trace("clear_cache")
""",
        encoding="utf-8",
    )
    parameter_entries = ", ".join(
        f'"weight_{index}": self._Parameter("mlx.core.{dtype}")'
        for index, dtype in enumerate(parameter_dtypes)
    )
    (mlx_lm_root / "__init__.py").write_text(
        f'''import os

__version__ = "{mlx_lm_version}"

def _trace(event):
    with open(os.environ["SOUP_TEST_MLX_TRACE"], "a", encoding="utf-8") as handle:
        handle.write(event + "\\n")

class _Tokenizer:
    chat_template = "{{{{ messages }}}}"

class _Row:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        _trace("tolist")
        return list(self._values)

class _Logits:
    def __init__(self, sequence_length, vocab_size=8):
        self.shape = (1, sequence_length, vocab_size)
        self._vocab_size = vocab_size

    def __getitem__(self, key):
        _, position, _ = key
        if position < 0:
            position += self.shape[1]
        values = [float(token_id) / 10.0 for token_id in range(self._vocab_size)]
        values[(position + 1) % self._vocab_size] += 3.0
        return _Row(values)

class _Model:
    class _Parameter:
        def __init__(self, dtype):
            self.dtype = dtype

    def parameters(self):
        return {{{parameter_entries}}}

    def __call__(self, inputs):
        _trace("forward:" + str(len(inputs[0])))
        return _Logits(len(inputs[0]))

def load(model_path):
    _trace("load:" + model_path)
    return _Model(), _Tokenizer()
''',
        encoding="utf-8",
    )
    (mlx_lm_root / "utils.py").write_text(
        """from . import _Tokenizer, _trace

def load_tokenizer(model_path):
    _trace("load-tokenizer:" + model_path)
    return _Tokenizer()
""",
        encoding="utf-8",
    )
    return fake_root


def _local_plan(
    tmp_path: Path,
    *,
    backend_version: str = _MLX_VERSION,
    truncation: str = "none",
    max_sequence_length: int = 16,
) -> tuple[AutoDistillPlan, Path, Path, Path]:
    teacher_root = tmp_path / "teacher"
    tokenizer_root = tmp_path / "tokenizer"
    dataset_root = tmp_path / "dataset"
    teacher_root.mkdir()
    tokenizer_root = teacher_root
    dataset_root.mkdir()
    config = b'{"model_type":"fixture"}\n'
    weights = b"teacher-weights"
    tokenizer = b'{"version":"1.0"}\n'
    template = "{{ messages }}"
    dataset = (
        b'{"example_id":"example-1","prompt_token_ids":[1,2],'
        b'"schema":"soup.autodistill.tokenized-teacher-example.v1",'
        b'"target_token_ids":[3,4,5]}\n'
    )
    (teacher_root / "config.json").write_bytes(config)
    (teacher_root / "model.safetensors").write_bytes(weights)
    (teacher_root / "tokenizer.json").write_bytes(tokenizer)
    (dataset_root / "prompts.jsonl").write_bytes(dataset)

    payload = json.loads((FIXTURES / "plan.json").read_text(encoding="utf-8"))
    payload["teacher"]["config_sha256"] = _sha(config)
    payload["teacher"]["weights"] = [
        {"path": "model.safetensors", "bytes": len(weights), "sha256": _sha(weights)}
    ]
    payload["tokenizer"].update(
        {
            "vocab_size": 8,
            "chat_template_sha256": _sha(template.encode()),
            "renderer": f"mlx-lm@{backend_version}",
            "files": [
                {"path": "tokenizer.json", "bytes": len(tokenizer), "sha256": _sha(tokenizer)}
            ],
        }
    )
    normalized = canonicalize_jsonl_bytes(dataset)
    payload["dataset"].update(
        {
            "normalized_sha256": _sha(normalized),
            "rows": 1,
            "source_files": [
                {"path": "prompts.jsonl", "bytes": len(dataset), "sha256": _sha(dataset)}
            ],
        }
    )
    payload["capture"].update(
        {
            "planned_token_count": 3,
            "vocab_size": 8,
            "backend": "mlx",
            "backend_version": backend_version,
            "dtype": "float32",
            "max_sequence_length": max_sequence_length,
            "truncation": truncation,
        }
    )
    payload["probability_policy"].update(
        {
            "top_k": 3,
            "log_probability_bytes": 4,
            "tail_mass_bytes": 8,
            "entropy_bytes": 8,
        }
    )
    estimate = build_plan_estimate(
        token_count=3,
        vocab_size=8,
        top_k=3,
        max_forced_tokens_per_position=2,
        token_id_bytes=4,
        log_probability_bytes=4,
        tail_mass_bytes=8,
        entropy_bytes=8,
    )
    payload["estimate"] = estimate.model_dump(mode="json")
    return AutoDistillPlan.model_validate(payload), teacher_root, tokenizer_root, dataset_root


def _runtime_environment(monkeypatch, tmp_path: Path, fake_root: Path) -> Path:
    trace = tmp_path / "mlx-trace.txt"
    source_root = Path(__file__).parents[1] / "src"
    existing = os.environ.get("PYTHONPATH", "")
    entries = [os.fspath(fake_root), os.fspath(source_root)]
    if existing:
        entries.append(existing)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(entries))
    monkeypatch.setenv("SOUP_TEST_MLX_TRACE", os.fspath(trace))
    return trace


def test_worker_module_is_import_light_and_has_no_student_input():
    import soup_cli.autodistill.mlx_worker as module

    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    top_level_imports = {
        alias.name.split(".", 1)[0]
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    top_level_imports.update(
        node.module.split(".", 1)[0]
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module
    )

    assert not {"mlx", "mlx_lm", "torch", "transformers"} & top_level_imports
    assert "student" not in inspect.signature(run_mlx_teacher_capture_process).parameters


def test_real_child_process_captures_full_trajectory_and_exits(monkeypatch, tmp_path):
    plan, teacher_root, tokenizer_root, dataset_root = _local_plan(tmp_path)
    fake_root = _write_fake_mlx_runtime(tmp_path)
    trace = _runtime_environment(monkeypatch, tmp_path, fake_root)
    publication_root = tmp_path / "publication"

    result = run_mlx_teacher_capture_process(
        plan=plan,
        teacher_root=teacher_root,
        tokenizer_root=tokenizer_root,
        dataset_root=dataset_root,
        publication_root=publication_root,
        shard_id="shard-0001",
        transaction_id="transaction-0001",
        python_executable=sys.executable,
        timeout_seconds=30,
    )

    assert result.worker_exit_confirmed is True
    assert result.student_loaded is False
    assert result.worker_pid != os.getpid()
    assert result.row_count == result.token_count == 3
    receipt = json.loads(
        (publication_root / ".workers/transaction-0001/worker-receipt.json").read_bytes()
    )
    assert receipt["inference_dtype"] == "float32"
    assert receipt["floating_parameter_dtypes"] == ["float32"]
    assert receipt["quantization"] == "none"
    persisted = MlxTeacherCaptureResult.model_validate_json(
        (publication_root / ".workers/transaction-0001/result.json").read_bytes()
    )
    assert persisted == result
    manifest = ShardManifest.model_validate_json(
        (publication_root / "shards/shard-0001/manifest.available.json").read_bytes()
    )
    assert manifest.state == "available"
    rows = tuple(
        CaptureToken.model_validate_json(line)
        for line in (publication_root / "shards/shard-0001/capture.jsonl").read_bytes().splitlines()
    )
    assert [row.position for row in rows] == [0, 1, 2]
    assert [row.context_token_ids for row in rows] == [(1, 2), (1, 2, 3), (1, 2, 3, 4)]
    events = trace.read_text(encoding="utf-8").splitlines()
    assert events.count(f"load:{teacher_root}") == 1
    assert events.count(f"load-tokenizer:{tokenizer_root}") == 1
    assert [event for event in events if event.startswith("forward:")] == ["forward:4"]
    assert events.count("astype:float32") == 3
    assert events.count("eval") == 3
    assert events[-1] == "clear_cache"


def test_controller_rejects_available_manifest_not_bound_to_worker_receipt(
    monkeypatch,
    tmp_path,
):
    import soup_cli.autodistill.mlx_worker as worker_module

    plan, teacher_root, tokenizer_root, dataset_root = _local_plan(tmp_path)
    fake_root = _write_fake_mlx_runtime(tmp_path)
    _runtime_environment(monkeypatch, tmp_path, fake_root)
    publication_root = tmp_path / "publication"
    original_read_control = worker_module._read_control

    def tamper_after_receipt(path, model_type):
        receipt = original_read_control(path, model_type)
        manifest_path = publication_root / "shards/shard-0001/manifest.available.json"
        manifest = ShardManifest.model_validate_json(manifest_path.read_bytes())
        tampered = ShardManifest.model_validate(
            manifest.model_copy(update={"plan_sha256": "f" * 64}).model_dump(
                by_alias=True
            )
        )
        manifest_path.write_bytes(canonical_json_bytes(tampered) + b"\n")
        return receipt

    monkeypatch.setattr(worker_module, "_read_control", tamper_after_receipt)

    with pytest.raises(ValueError, match="receipt does not match the available shard"):
        run_mlx_teacher_capture_process(
            plan=plan,
            teacher_root=teacher_root,
            tokenizer_root=tokenizer_root,
            dataset_root=dataset_root,
            publication_root=publication_root,
            shard_id="shard-0001",
            transaction_id="transaction-0001",
            python_executable=sys.executable,
            timeout_seconds=30,
        )


def test_runtime_version_mismatch_fails_before_publication(monkeypatch, tmp_path):
    plan, teacher_root, tokenizer_root, dataset_root = _local_plan(tmp_path)
    fake_root = _write_fake_mlx_runtime(tmp_path, mlx_lm_version="different-version")
    _runtime_environment(monkeypatch, tmp_path, fake_root)
    publication_root = tmp_path / "publication"

    with pytest.raises(RuntimeError, match="does not match capture.backend_version"):
        run_mlx_teacher_capture_process(
            plan=plan,
            teacher_root=teacher_root,
            tokenizer_root=tokenizer_root,
            dataset_root=dataset_root,
            publication_root=publication_root,
            shard_id="shard-0001",
            transaction_id="transaction-0001",
            python_executable=sys.executable,
            timeout_seconds=30,
        )

    assert not (publication_root / "shards/shard-0001").exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("dtype", "bfloat16", "floating parameter dtypes"),
        ("quantization", "q4", "quantization does not match"),
    ],
)
def test_declared_runtime_identity_must_match_loaded_teacher(
    monkeypatch,
    tmp_path,
    field,
    value,
    message,
):
    plan, teacher_root, tokenizer_root, dataset_root = _local_plan(tmp_path)
    payload = plan.model_dump(mode="json", by_alias=True)
    payload["capture"][field] = value
    plan = AutoDistillPlan.model_validate(payload)
    fake_root = _write_fake_mlx_runtime(tmp_path)
    _runtime_environment(monkeypatch, tmp_path, fake_root)
    publication_root = tmp_path / "publication"

    with pytest.raises(RuntimeError, match=message):
        run_mlx_teacher_capture_process(
            plan=plan,
            teacher_root=teacher_root,
            tokenizer_root=tokenizer_root,
            dataset_root=dataset_root,
            publication_root=publication_root,
            shard_id="shard-0001",
            transaction_id="transaction-0001",
            python_executable=sys.executable,
            timeout_seconds=30,
        )

    assert not (publication_root / "shards/shard-0001").exists()


def test_quantized_teacher_records_declared_dtype_and_float32_auxiliaries(
    monkeypatch,
    tmp_path,
):
    plan, teacher_root, tokenizer_root, dataset_root = _local_plan(tmp_path)
    config = {"model_type": "fixture", "quantization": {"bits": 4, "group_size": 64}}
    config_bytes = canonical_json_bytes(config) + b"\n"
    (teacher_root / "config.json").write_bytes(config_bytes)
    active = {"quantization": config["quantization"]}
    quantization = f"config-sha256:{_sha(canonical_json_bytes(active))}"
    payload = plan.model_dump(mode="json", by_alias=True)
    payload["teacher"]["config_sha256"] = _sha(config_bytes)
    payload["capture"].update({"dtype": "bfloat16", "quantization": quantization})
    plan = AutoDistillPlan.model_validate(payload)
    fake_root = _write_fake_mlx_runtime(
        tmp_path,
        parameter_dtypes=("bfloat16", "float32"),
    )
    _runtime_environment(monkeypatch, tmp_path, fake_root)
    publication_root = tmp_path / "publication"

    run_mlx_teacher_capture_process(
        plan=plan,
        teacher_root=teacher_root,
        tokenizer_root=tokenizer_root,
        dataset_root=dataset_root,
        publication_root=publication_root,
        shard_id="shard-0001",
        transaction_id="transaction-0001",
        python_executable=sys.executable,
        timeout_seconds=30,
    )

    receipt = json.loads(
        (publication_root / ".workers/transaction-0001/worker-receipt.json").read_bytes()
    )
    assert receipt["inference_dtype"] == "bfloat16"
    assert receipt["floating_parameter_dtypes"] == ["bfloat16", "float32"]
    assert receipt["quantization"] == quantization


def test_declared_left_truncation_uses_one_forward_per_exact_context(monkeypatch, tmp_path):
    plan, teacher_root, tokenizer_root, dataset_root = _local_plan(
        tmp_path,
        truncation="left",
        max_sequence_length=2,
    )
    fake_root = _write_fake_mlx_runtime(tmp_path)
    trace = _runtime_environment(monkeypatch, tmp_path, fake_root)
    publication_root = tmp_path / "publication"

    run_mlx_teacher_capture_process(
        plan=plan,
        teacher_root=teacher_root,
        tokenizer_root=tokenizer_root,
        dataset_root=dataset_root,
        publication_root=publication_root,
        shard_id="shard-0001",
        transaction_id="transaction-0001",
        python_executable=sys.executable,
        timeout_seconds=30,
    )

    rows = tuple(
        CaptureToken.model_validate_json(line)
        for line in (publication_root / "shards/shard-0001/capture.jsonl").read_bytes().splitlines()
    )
    assert [row.context_token_ids for row in rows] == [(1, 2), (2, 3), (3, 4)]
    events = trace.read_text(encoding="utf-8").splitlines()
    assert [event for event in events if event.startswith("forward:")] == [
        "forward:2",
        "forward:2",
        "forward:2",
    ]


def test_worker_publishes_only_requested_example_shard_from_fully_bound_dataset(
    monkeypatch,
    tmp_path,
):
    plan, teacher_root, tokenizer_root, dataset_root = _local_plan(tmp_path)
    first = (dataset_root / "prompts.jsonl").read_bytes()
    second = (
        b'{"example_id":"example-2","prompt_token_ids":[6],'
        b'"schema":"soup.autodistill.tokenized-teacher-example.v1",'
        b'"target_token_ids":[2,3]}\n'
    )
    dataset = first + second
    (dataset_root / "prompts.jsonl").write_bytes(dataset)
    payload = plan.model_dump(mode="json", by_alias=True)
    payload["dataset"].update(
        {
            "normalized_sha256": _sha(canonicalize_jsonl_bytes(dataset)),
            "rows": 2,
            "source_files": [
                {"path": "prompts.jsonl", "bytes": len(dataset), "sha256": _sha(dataset)}
            ],
        }
    )
    payload["capture"]["planned_token_count"] = 5
    payload["estimate"] = build_plan_estimate(
        token_count=5,
        vocab_size=8,
        top_k=3,
        max_forced_tokens_per_position=2,
        token_id_bytes=4,
        log_probability_bytes=4,
        tail_mass_bytes=8,
        entropy_bytes=8,
    ).model_dump(mode="json")
    plan = AutoDistillPlan.model_validate(payload)
    fake_root = _write_fake_mlx_runtime(tmp_path)
    _runtime_environment(monkeypatch, tmp_path, fake_root)
    publication_root = tmp_path / "publication"

    result = run_mlx_teacher_capture_process(
        plan=plan,
        teacher_root=teacher_root,
        tokenizer_root=tokenizer_root,
        dataset_root=dataset_root,
        publication_root=publication_root,
        shard_id="shard-0002",
        transaction_id="transaction-0002",
        example_start=1,
        example_end=2,
        python_executable=sys.executable,
        timeout_seconds=30,
    )

    assert result.row_count == result.token_count == 2
    rows = tuple(
        CaptureToken.model_validate_json(line)
        for line in (publication_root / "shards/shard-0002/capture.jsonl").read_bytes().splitlines()
    )
    assert [row.example_id for row in rows] == ["example-2", "example-2"]
    assert [row.context_token_ids for row in rows] == [(6,), (6, 2)]


def test_changed_bound_dataset_fails_before_model_load(monkeypatch, tmp_path):
    plan, teacher_root, tokenizer_root, dataset_root = _local_plan(tmp_path)
    fake_root = _write_fake_mlx_runtime(tmp_path)
    trace = _runtime_environment(monkeypatch, tmp_path, fake_root)
    (dataset_root / "prompts.jsonl").write_bytes(b'{"changed":true}\n')

    with pytest.raises(RuntimeError, match="byte count mismatch|sha256 mismatch"):
        run_mlx_teacher_capture_process(
            plan=plan,
            teacher_root=teacher_root,
            tokenizer_root=tokenizer_root,
            dataset_root=dataset_root,
            publication_root=tmp_path / "publication",
            shard_id="shard-0001",
            transaction_id="transaction-0001",
            python_executable=sys.executable,
            timeout_seconds=30,
        )

    assert not trace.exists()


def test_changed_bound_tokenizer_fails_before_model_load(monkeypatch, tmp_path):
    plan, teacher_root, tokenizer_root, dataset_root = _local_plan(tmp_path)
    fake_root = _write_fake_mlx_runtime(tmp_path)
    trace = _runtime_environment(monkeypatch, tmp_path, fake_root)
    (tokenizer_root / "tokenizer.json").write_bytes(b'{"changed":true}\n')

    with pytest.raises(RuntimeError, match="byte count mismatch|sha256 mismatch"):
        run_mlx_teacher_capture_process(
            plan=plan,
            teacher_root=teacher_root,
            tokenizer_root=tokenizer_root,
            dataset_root=dataset_root,
            publication_root=tmp_path / "publication",
            shard_id="shard-0001",
            transaction_id="transaction-0001",
            python_executable=sys.executable,
            timeout_seconds=30,
        )

    assert not trace.exists()


def test_non_mlx_plan_is_refused_before_worker_directory_is_created(tmp_path):
    plan, teacher_root, tokenizer_root, dataset_root = _local_plan(tmp_path)
    payload = plan.model_dump(mode="json", by_alias=True)
    payload["capture"]["backend"] = "transformers"
    wrong_plan = AutoDistillPlan.model_validate(payload)
    publication_root = tmp_path / "publication"

    with pytest.raises(ValidationError, match="capture.backend=mlx"):
        run_mlx_teacher_capture_process(
            plan=wrong_plan,
            teacher_root=teacher_root,
            tokenizer_root=tokenizer_root,
            dataset_root=dataset_root,
            publication_root=publication_root,
            shard_id="shard-0001",
            transaction_id="transaction-0001",
        )

    assert not publication_root.exists()


def test_worker_ids_fail_closed_before_spawn(tmp_path):
    plan, teacher_root, tokenizer_root, dataset_root = _local_plan(tmp_path)
    publication_root = tmp_path / "publication"
    with pytest.raises(ValidationError, match="transaction_id"):
        run_mlx_teacher_capture_process(
            plan=plan,
            teacher_root=teacher_root,
            tokenizer_root=tokenizer_root,
            dataset_root=dataset_root,
            publication_root=publication_root,
            shard_id="shard-0001",
            transaction_id="../escape",
        )
    assert not publication_root.exists()


def test_canonical_tokenizer_can_be_loaded_from_a_separate_root(monkeypatch, tmp_path):
    plan, teacher_root, tokenizer_root, dataset_root = _local_plan(tmp_path)
    canonical_root = tmp_path / "canonical-tokenizer"
    canonical_root.mkdir()
    (canonical_root / "tokenizer.json").write_bytes(
        (tokenizer_root / "tokenizer.json").read_bytes()
    )
    fake_root = _write_fake_mlx_runtime(tmp_path)
    trace = _runtime_environment(monkeypatch, tmp_path, fake_root)

    result = run_mlx_teacher_capture_process(
        plan=plan,
        teacher_root=teacher_root,
        tokenizer_root=canonical_root,
        dataset_root=dataset_root,
        publication_root=tmp_path / "publication",
        shard_id="shard-0001",
        transaction_id="transaction-0001",
        python_executable=sys.executable,
        timeout_seconds=30,
    )

    assert result.student_loaded is False
    events = trace.read_text(encoding="utf-8").splitlines()
    assert events.count(f"load:{teacher_root}") == 1
    assert events.count(f"load-tokenizer:{canonical_root}") == 1
