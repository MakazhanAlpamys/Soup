"""User-supplied UTF-8 files accept a leading BOM without losing row content (#2136)."""

from __future__ import annotations

import codecs
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from tests.cli_runner import CliRunner
from tests.conftest import strip_ansi

_TEXTS = ["first\ufeffvalue", "\ufeffsecond value"]


@pytest.fixture(params=[False, True], ids=["utf8", "utf8-bom"])
def bom(request: pytest.FixtureRequest) -> bool:
    return request.param


@pytest.fixture(params=["\n", "\r\n"], ids=["lf", "crlf"])
def newline(request: pytest.FixtureRequest) -> str:
    return request.param


def _write_input(path: Path, text: str, bom: bool, newline: str) -> None:
    payload = text.replace("\n", newline).encode("utf-8")
    path.write_bytes((codecs.BOM_UTF8 if bom else b"") + payload)


def _text_rows(suffix: str) -> str:
    rows = [json.dumps({"text": text}, ensure_ascii=False) for text in _TEXTS]
    return "\n".join(rows if suffix == ".jsonl" else _TEXTS) + "\n"


@pytest.mark.parametrize("suffix", [".jsonl", ".txt"])
def test_tokenizer_keeps_first_record_and_embedded_feff(
    tmp_path: Path, suffix: str, bom: bool, newline: str
) -> None:
    from soup_cli.commands.tokenizer import _extract_texts

    path = tmp_path / ("corpus" + suffix)
    _write_input(path, _text_rows(suffix), bom, newline)
    assert _extract_texts(path) == _TEXTS


def test_draft_keeps_first_jsonl_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bom: bool, newline: str
) -> None:
    from soup_cli.commands.draft import _read_jsonl

    monkeypatch.chdir(tmp_path)
    path = tmp_path / "prompts.jsonl"
    _write_input(path, _text_rows(".jsonl"), bom, newline)
    assert _read_jsonl(str(path), "--prompts", max_rows=10) == [{"text": text} for text in _TEXTS]


@pytest.mark.parametrize("suffix", [".jsonl", ".txt"])
def test_edit_covariance_corpus_extracts_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suffix: str, bom: bool, newline: str
) -> None:
    from soup_cli.commands.edit import _load_cov_corpus

    monkeypatch.chdir(tmp_path)
    path = tmp_path / ("corpus" + suffix)
    _write_input(path, _text_rows(suffix), bom, newline)
    assert _load_cov_corpus(str(path)) == _TEXTS


@pytest.mark.parametrize("suffix", [".jsonl", ".txt"])
def test_shrink_calibration_extracts_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suffix: str, bom: bool, newline: str
) -> None:
    from soup_cli.commands.shrink import _load_calib

    monkeypatch.chdir(tmp_path)
    path = tmp_path / ("calib" + suffix)
    _write_input(path, _text_rows(suffix), bom, newline)
    assert _load_calib(str(path)) == _TEXTS


def test_diagnose_evidence_keeps_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bom: bool, newline: str
) -> None:
    from soup_cli.commands.diagnose import _load_evidence

    monkeypatch.chdir(tmp_path)
    payload = {"scores": {}, "note": _TEXTS[0]}
    path = tmp_path / "evidence.json"
    _write_input(path, json.dumps(payload, ensure_ascii=False, indent=2), bom, newline)
    assert _load_evidence(str(path)) == payload


def test_can_attestation_keeps_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bom: bool, newline: str
) -> None:
    from soup_cli.commands.can import _load_attestation_files

    monkeypatch.chdir(tmp_path)
    payload = {
        "_type": "https://in-toto.io/Statement/v1",
        "predicateType": "https://slsa.dev/provenance/v1",
        "predicate": {"note": _TEXTS[0]},
    }
    path = tmp_path / "statement.json"
    _write_input(path, json.dumps(payload, ensure_ascii=False, indent=2), bom, newline)
    assert _load_attestation_files([str(path)]) == [payload]


@pytest.mark.parametrize("suffix", [".jsonl", ".txt"])
def test_bench_passes_complete_prompts_to_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suffix: str, bom: bool, newline: str
) -> None:
    import torch

    import soup_cli.commands.infer as infer
    from soup_cli.commands.bench import app

    monkeypatch.chdir(tmp_path)
    (tmp_path / "model").mkdir()
    path = tmp_path / ("prompts" + suffix)
    rows = [json.dumps({"prompt": text}, ensure_ascii=False) for text in _TEXTS]
    content = "\n".join(rows if suffix == ".jsonl" else _TEXTS) + "\n"
    _write_input(path, content, bom, newline)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(infer, "_load_model", Mock(return_value=("model", "tokenizer")))
    generate = Mock(return_value=("answer", 10))
    monkeypatch.setattr(infer, "_generate", generate)

    result = CliRunner().invoke(app, ["infer", "model", "--prompts-file", path.name])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "Running 2 test inferences" in strip_ansi(result.output)
    prompts = [call.args[2][0]["content"] for call in generate.call_args_list]
    assert prompts[-2:] == _TEXTS


@pytest.mark.parametrize("sandbox", [False, True], ids=["heuristic", "sandbox"])
def test_agent_scores_every_prediction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sandbox: bool, bom: bool, newline: str
) -> None:
    import soup_cli.trainer.rewards as rewards
    import soup_cli.utils.agent_sandbox as agent_sandbox
    from soup_cli.commands.agent import app

    monkeypatch.chdir(tmp_path)
    spec = {
        "openapi": "3.0.0",
        "info": {"title": "Synthetic", "version": "1"},
        "paths": {
            "/ping": {
                "get": {
                    "operationId": "ping",
                    "parameters": [{"name": "name", "in": "query", "schema": {"type": "string"}}],
                    "responses": {"200": {"description": "OK"}},
                }
            },
        },
    }
    (tmp_path / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
    rows = [{"tool": "ping", "arguments": {"name": text}} for text in _TEXTS]
    content = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
    _write_input(tmp_path / "pred.jsonl", content, bom, newline)
    score = Mock(return_value="ok")
    monkeypatch.setattr(agent_sandbox, "score_sandbox", score)
    monkeypatch.setattr(rewards, "_get_isolation_strategy", lambda: "synthetic")
    args = ["eval", "--spec", "spec.json", "--predictions", "pred.jsonl"]
    result = CliRunner().invoke(app, args + (["--sandbox"] if sandbox else []))
    assert result.exit_code == 0, (result.output, repr(result.exception))
    plain = " ".join(strip_ansi(result.output).split())
    assert "Predictions: 2" in plain
    if sandbox:
        assert [call.kwargs["arguments"]["name"] for call in score.call_args_list] == _TEXTS
    else:
        assert "Tool match: 2 (100.0%)" in plain
        assert "Args valid: 2 (100.0%)" in plain
        score.assert_not_called()


def test_generate_context_and_output_encoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bom: bool, newline: str
) -> None:
    import soup_cli.commands.generate as generate
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    context = "\n".join(_TEXTS) + "\n"
    _write_input(tmp_path / "context.txt", context, bom, newline)
    example = {"instruction": _TEXTS[0], "input": "", "output": _TEXTS[1]}
    batch = Mock(return_value=[example])
    monkeypatch.setattr(generate, "_generate_batch", batch)
    result = CliRunner().invoke(
        app,
        [
            "data",
            "generate",
            "--prompt",
            "synthetic",
            "--provider",
            "local",
            "--context",
            "context.txt",
            "--template",
            "qa",
            "--count",
            "1",
            "--output",
            "output.jsonl",
        ],
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert batch.call_args.kwargs["context_text"] == context
    output = (tmp_path / "output.jsonl").read_bytes()
    assert not output.startswith(codecs.BOM_UTF8)
    assert json.loads(output.decode("utf-8")) == example


@pytest.mark.parametrize("quant_method, expected", [("gptq", "gptq"), ("gp\ufefftq", None)])
def test_autopilot_reads_local_quantization_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    quant_method: str,
    expected: str | None,
    bom: bool,
    newline: str,
) -> None:
    from soup_cli.autopilot.decisions import detect_prequantized_format_from_path

    monkeypatch.chdir(tmp_path)
    model = tmp_path / "neutral-model"
    model.mkdir()
    config = {"quantization_config": {"quant_method": quant_method}}
    _write_input(
        model / "config.json", json.dumps(config, ensure_ascii=False, indent=2), bom, newline
    )
    assert detect_prequantized_format_from_path(str(model)) == expected


@pytest.mark.parametrize(
    "statement_bom, sidecar_bom",
    [(False, False), (True, False), (False, True), (True, True)],
    ids=["plain", "statement-bom", "sidecar-bom", "both-bom"],
)
@pytest.mark.parametrize("integrity", ["valid", "tampered", "wrong-key"])
def test_attest_bom_preserves_signature_and_trust_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    statement_bom: bool,
    sidecar_bom: bool,
    integrity: str,
    newline: str,
) -> None:
    from soup_cli.commands.attest import app
    from soup_cli.utils.signing import (
        generate_ed25519_private_pem,
        private_key_from_pem,
        public_key_pem,
    )

    monkeypatch.chdir(tmp_path)
    private_pem = generate_ed25519_private_pem()
    (tmp_path / "signing.pem").write_text(private_pem, encoding="utf-8")
    emitted = CliRunner().invoke(
        app,
        [
            "emit",
            "--stage",
            "train",
            "--subject",
            "synthetic",
            "--sha",
            "a" * 64,
            "--sign",
            "ed25519",
            "--key",
            "signing.pem",
            "--invocation",
            _TEXTS[0],
            "--output",
            "statement.json",
        ],
    )
    assert emitted.exit_code == 0, (emitted.output, repr(emitted.exception))
    statement = tmp_path / "statement.json"
    sidecar = tmp_path / "statement.json.sig"
    assert not statement.read_bytes().startswith(codecs.BOM_UTF8)
    assert not sidecar.read_bytes().startswith(codecs.BOM_UTF8)
    document = json.loads(statement.read_text(encoding="utf-8"))
    signature = json.loads(sidecar.read_text(encoding="utf-8"))
    if integrity == "tampered":
        document["predicate"]["tampered"] = True
    if integrity == "wrong-key":
        private_pem = generate_ed25519_private_pem()
    public_pem = public_key_pem(private_key_from_pem(private_pem))
    (tmp_path / "trusted.pub").write_text(public_pem, encoding="utf-8")
    # Change formatting and line endings as well as the file's BOM. Verification
    # must still use emit's canonical JSON, retaining the embedded U+FEFF.
    _write_input(
        statement, json.dumps(document, ensure_ascii=False, indent=1), statement_bom, newline
    )
    _write_input(sidecar, json.dumps(signature, indent=1), sidecar_bom, newline)

    result = CliRunner().invoke(
        app,
        [
            "verify",
            statement.name,
            "--signature",
            sidecar.name,
            "--public-key",
            "trusted.pub",
        ],
    )
    plain = " ".join(strip_ansi(result.output).split())
    if integrity == "valid":
        assert result.exit_code == 0, (plain, repr(result.exception))
        assert "Attestation signature valid" in plain
    else:
        assert result.exit_code == 3, (plain, repr(result.exception))
        expected = "Attestation signature INVALID" if integrity == "tampered" else "untrusted key"
        assert expected in plain


@pytest.mark.parametrize("reader", ["draft", "edit", "shrink", "diagnose", "can"])
def test_bom_readers_still_enforce_raw_file_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], reader: str
) -> None:
    import importlib

    import typer

    monkeypatch.chdir(tmp_path)
    module = importlib.import_module(f"soup_cli.commands.{reader}")
    text = '{"_type":"type","predicateType":"predicate","text":"ok","scores":{}}'
    path = tmp_path / "input.jsonl"
    _write_input(path, text, True, "\n")
    limit = len(path.read_bytes()) - 1  # Decoded text fits; raw bytes including the BOM do not.
    caps = {
        "draft": "_MAX_INPUT_BYTES",
        "edit": "_MAX_COV_CORPUS_BYTES",
        "shrink": "_MAX_INPUT_BYTES",
        "diagnose": "_MAX_EVIDENCE_BYTES",
    }
    if reader == "can":
        from soup_cli.cans import schema

        monkeypatch.setattr(schema, "_MAX_ATTESTATION_BYTES", limit)
    else:
        monkeypatch.setattr(module, caps[reader], limit)
    with pytest.raises((ValueError, typer.BadParameter, typer.Exit)) as caught:
        if reader == "draft":
            module._read_jsonl(str(path), "--prompts", max_rows=10)
        elif reader == "edit":
            module._load_cov_corpus(str(path))
        elif reader == "shrink":
            module._load_calib(str(path))
        elif reader == "diagnose":
            module._load_evidence(str(path))
        else:
            module._load_attestation_files([str(path)])
    message = " ".join((str(caught.value) + strip_ansi(capsys.readouterr().out)).split())
    assert "too large" in message or "exceeds" in message
