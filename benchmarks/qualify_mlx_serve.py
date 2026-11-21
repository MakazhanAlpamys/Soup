"""Qualify the real CLI/HTTP native server; no throughput or model-quality claim.

Run from the checkout in a standalone .[mlx,serve] + httpx environment:
  python -m benchmarks.qualify_mlx_serve --fixture-dir /path/inside/workdir/fixture \
      --workdir /path/inside/workdir --report evidence.json
Or pass --model to an already downloaded, revision-pinned dense checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(part)
    return value.hexdigest()


def reference(model: Any, tokenizer: Any, messages: list[dict], max_tokens: int) -> dict:
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    from soup_cli.utils.vllm import _render_chat_prompt

    rendered, templated = _render_chat_prompt(messages, tokenizer, fallback_on_error=False)
    prompt = tokenizer.encode(rendered, add_special_tokens=not templated)
    responses = list(
        stream_generate(
            model, tokenizer, prompt=prompt, max_tokens=max_tokens, sampler=make_sampler(temp=0)
        )
    )
    last = responses[-1]
    return dict(
        text="".join(response.text for response in responses),
        finish_reason=last.finish_reason,
        usage=dict(
            prompt_tokens=len(prompt),
            completion_tokens=last.generation_tokens,
            total_tokens=len(prompt) + last.generation_tokens,
        ),
    )


def qualify(args: argparse.Namespace) -> None:
    import httpx
    import mlx.core as mx

    from soup_cli.utils.mlx import get_chip_info, get_unified_memory_bytes
    from soup_cli.utils.mlx_generation import load_native_model

    present = [
        name
        for name in ("torch", "trl", "peft", "datasets", "accelerate", "bitsandbytes")
        if importlib.util.find_spec(name) is not None
    ]
    if present:
        raise RuntimeError(f"Use a standalone native environment; found {present}")
    source = Path(__file__).resolve().parents[1]
    workdir = args.workdir.resolve()
    fixture = None
    if args.fixture_dir is not None:
        from benchmarks.harness.mlx_serve_fixture import save_fixture

        fixture = save_fixture(args.fixture_dir.resolve())
        path, base_control = fixture["adapter"], fixture["base"]
        prompts = ["red green", "t6 red"]
    else:
        path, base_control = args.model.resolve(), None
        prompts = ["What is 2 + 2?", "Name one primary color."]
    messages = [[dict(role="user", content=prompt)] for prompt in prompts]
    model, tokenizer = load_native_model(str(path), require_dense=True)
    expected = [reference(model, tokenizer, prompt, args.max_tokens) for prompt in messages]
    controls = None
    if base_control is not None:
        base, base_tokenizer = load_native_model(str(base_control), require_dense=True)
        controls = [reference(base, base_tokenizer, prompt, args.max_tokens) for prompt in messages]
        if not any(a["text"] != b["text"] for a, b in zip(expected, controls, strict=True)):
            raise AssertionError("Qualification must detect an ignored adapter")
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    environment = os.environ.copy()
    environment.update(
        PYTHONPATH=os.pathsep.join([str(source / "src"), str(source)]),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TERM="xterm-256color",
    )
    environment.pop("FORCE_COLOR", None)
    environment.pop("NO_COLOR", None)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.report.with_suffix(".server.log")
    command = [
        sys.executable,
        "-m",
        "soup_cli",
        "serve",
        "--backend",
        "mlx",
        "--model",
        str(path),
        "--port",
        str(port),
    ]
    started = time.perf_counter()
    rows = []
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command, cwd=workdir, env=environment, stdout=log, stderr=subprocess.STDOUT
        )
        try:
            with httpx.Client(
                base_url=f"http://127.0.0.1:{port}", timeout=30, trust_env=False
            ) as client:
                deadline = time.monotonic() + 60
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f"CLI exited before readiness; inspect {log_path}")
                    try:
                        health = client.get("/health", timeout=1)
                        if health.status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Public CLI never bound a ready HTTP server")
                    time.sleep(0.1)
                ready_seconds = time.perf_counter() - started
                assert health.json()["status"] == "ok" and health.json()["device"] == "mlx"
                identity = client.get("/v1/models").json()["data"][0]["id"]
                assert identity == path.name
                # Revisit the first prompt after an unrelated request to test isolation.
                for index in [0, 1, 0]:
                    body = dict(
                        model=identity,
                        messages=messages[index],
                        temperature=0,
                        max_tokens=args.max_tokens,
                    )
                    response = client.post("/v1/chat/completions", json=body)
                    response.raise_for_status()
                    value = response.json()
                    choice = value["choices"][0]
                    assert choice["message"]["content"] == expected[index]["text"]
                    assert choice["finish_reason"] == expected[index]["finish_reason"]
                    assert value["usage"] == expected[index]["usage"]
                    streaming_started = time.perf_counter()
                    first_content = None
                    frames = []
                    done = False
                    with client.stream(
                        "POST", "/v1/chat/completions", json={**body, "stream": True}
                    ) as streamed:
                        streamed.raise_for_status()
                        for line in streamed.iter_lines():
                            if line == "data: [DONE]":
                                done = True
                            elif line.startswith("data: "):
                                frame = json.loads(line[6:])
                                frames.append(frame)
                                if frame["choices"][0]["delta"].get("content"):
                                    if first_content is None:
                                        first_content = time.perf_counter() - streaming_started
                    text = "".join(
                        frame["choices"][0]["delta"].get("content", "") for frame in frames
                    )
                    assert done and text == expected[index]["text"]
                    assert frames[0]["choices"][0]["delta"] == {"role": "assistant"}
                    assert (
                        frames[-1]["choices"][0]["finish_reason"]
                        == expected[index]["finish_reason"]
                    )
                    assert frames[-1]["usage"] == expected[index]["usage"]
                    rows.append(
                        dict(
                            case=index,
                            direct=expected[index],
                            http=value,
                            streaming_content=text,
                            streaming_usage=frames[-1]["usage"],
                            first_content_seconds=first_content,
                            stream_seconds=time.perf_counter() - streaming_started,
                        )
                    )
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
                raise RuntimeError("Public CLI did not shut down after SIGINT") from None
    manifest = {
        str(file.relative_to(source)): digest(file)
        for file in sorted((source / "src").rglob("*.py"))
    }
    model_manifest = {
        str(file.relative_to(path)): digest(file)
        for file in sorted(path.iterdir())
        if file.is_file()
    }
    report = dict(
        soup_base_sha=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source, text=True
        ).strip(),
        source_files_sha256={
            name: value
            for name, value in manifest.items()
            if name
            in {
                "src/soup_cli/commands/serve.py",
                "src/soup_cli/utils/mlx_generation.py",
                "src/soup_cli/utils/mlx_serve.py",
            }
        },
        working_source_tree_sha256=hashlib.sha256(
            json.dumps(manifest, sort_keys=True).encode()
        ).hexdigest(),
        benchmark_sha256=digest(Path(__file__)),
        fixture_generator_sha256=digest(source / "benchmarks/harness/mlx_serve_fixture.py"),
        python=platform.python_version(),
        versions={
            name: importlib.metadata.version(name)
            for name in (
                "mlx",
                "mlx-lm",
                "transformers",
                "fastapi",
                "starlette",
                "uvicorn",
                "httpx",
                "pydantic",
            )
        },
        hardware=get_chip_info(),
        unified_memory_bytes=get_unified_memory_bytes(),
        device=str(mx.default_device()),
        metal_available=mx.metal.is_available(),
        training_packages_present=present,
        model_files_sha256=model_manifest,
        base_files_sha256={
            str(file.relative_to(base_control)): digest(file)
            for file in sorted(base_control.iterdir())
            if file.is_file()
        }
        if base_control is not None
        else None,
        model_identity=args.model_identity,
        model_revision=args.model_revision,
        cases=messages,
        public_cli=command,
        readiness_seconds=ready_seconds,
        child_returncode=process.returncode,
        results=rows,
        base_only_controls=controls,
        fixture={
            key: str(value) if isinstance(value, Path) else value for key, value in fixture.items()
        }
        if fixture
        else None,
        limits="CLI/HTTP lifecycle and greedy equality only; fixture is not pretrained quality; "
        "observed arrival times are not throughput benchmarks; concurrent/cancellation "
        "sensitivity is separately covered by deterministic offline ASGI tests",
    )
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    from rich.console import Console

    Console().print(f"Saved native serving qualification: {args.report}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--model", type=Path)
    target.add_argument("--fixture-dir", type=Path)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--model-identity", default="offline-trained-tiny-llama")
    parser.add_argument("--model-revision", default="fixture-files-sha256")
    qualify(parser.parse_args())
