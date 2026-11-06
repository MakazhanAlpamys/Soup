"""The continual-learning loop on a real model (SmolLM2-135M-Instruct), CUDA only.

``tests/test_loop_end_to_end.py`` checks the mechanics with stubs in every CI run.
This smoke test runs the same cycle with nothing stubbed but the HTTP transport:
a real model served by the soup serve app, real thumbs, the real ``soup train``
(DPO) and ``soup ship`` subprocesses, a real PEFT load through
``POST /v1/adapters/load``, real canary generations, promotion and rollback.

Run with: ``pytest -m smoke tests/test_loop_end_to_end_smoke.py``. It downloads
the model on first use and takes minutes. ``soup ship`` decides on its own; a
one-pair DPO run is not expected to win, so the test checks that a real verdict
was reached (never SKIPPED) and then canaries the trained adapter directly to
exercise the rollout on the real model.
"""

from __future__ import annotations

import dataclasses
import json
import os
from urllib.parse import urlparse

import pytest

pytestmark = pytest.mark.smoke

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
TOKEN = "smoke-" + "s" * 26
ENDPOINT = "http://127.0.0.1:8000"


def _cuda() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.device_count() > 0


def test_the_loop_on_a_real_model(tmp_path, monkeypatch):
    # Checked here, not in a skipif: collecting this file must not import torch.
    if not _cuda():
        pytest.skip("the loop trains in 4-bit, which needs a CUDA GPU")
    import httpx
    import torch
    from fastapi.testclient import TestClient
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from typer.testing import CliRunner

    import soup_cli.commands.serve as serve
    import soup_cli.utils.loop_stages as ls
    from soup_cli.commands.loop import app as loop_app
    from soup_cli.monitoring.trace_logger import TraceLogWriter
    from soup_cli.utils import local_rl
    from soup_cli.utils.loop_daemon import watch
    from soup_cli.utils.loop_iteration import list_iterations, read_iteration
    from soup_cli.utils.loop_state import read_state

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "registry.db"))
    monkeypatch.setenv("SOUP_LOOP_SERVE_ENDPOINT", ENDPOINT)
    monkeypatch.setenv(ls.TOOL_AUTH_TOKEN_ENV, TOKEN)
    for name in ("_DEPLOY_POSTER", "_LOAD_POSTER", "_TRAIN_RUNNER", "_SHIP_RUNNER"):
        monkeypatch.setattr(ls, name, None)
    (tmp_path / "tasks.jsonl").write_text(
        "".join(
            json.dumps({"prompt": f"What is {a} + {b}? Answer with the number.",
                        "expected": str(a + b)}) + "\n"
            for a, b in [(2, 2), (3, 4), (5, 1)]
        ),
        encoding="utf-8",
    )
    os.makedirs(os.path.join(".soup-loops", "adapters"))
    local_rl.init_local_rl_db("thumbs.db")

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to("cuda")
    model.eval()
    app = serve._create_app(
        model_obj=model, tokenizer=tokenizer, device="cuda", model_name=MODEL,
        max_tokens_default=16, adapter_map={}, peft_adapter_names=set(),
        trace_log_writer=TraceLogWriter(os.path.join(".soup-loops", "traces", "serve.jsonl")),
        record_thumbs_db="thumbs.db", auth_token=TOKEN,
        adapter_dir=os.path.realpath(os.path.join(".soup-loops", "adapters")),
    )
    client = TestClient(app, base_url=ENDPOINT)
    headers = {"Authorization": f"Bearer {TOKEN}"}

    def post(url, headers=None, json=None, timeout=None, trust_env=None, **_):
        resp = client.post(urlparse(url).path, headers=headers or {}, json=json)
        return httpx.Response(resp.status_code, content=resp.content,
                              headers={"content-type": resp.headers.get("content-type", "")})

    monkeypatch.setattr(httpx, "post", post)
    # Loading and deactivating read the reply through post_capped.
    monkeypatch.setattr("soup_cli.utils.http_response.post_capped", post)

    def chat(content, conversation_id=None):
        body = {"messages": [{"role": "user", "content": content}], "max_tokens": 16}
        if conversation_id:
            body["conversation_id"] = conversation_id
        resp = client.post("/v1/chat/completions", json=body)
        assert resp.status_code == 200, resp.text
        return resp.json()["choices"][0]["message"]["content"]

    # Users: two answers to one prompt, one liked, one disliked.
    prompt = "Give one tip to keep a houseplant alive."
    for thumb in ("up", "down"):
        resp = client.post("/v1/thumbs", headers=headers,
                           json={"prompt": prompt, "response": chat(prompt), "thumb": thumb})
        assert resp.status_code == 200, resp.text

    result = CliRunner().invoke(loop_app, [
        "init", "./served", "--eval", "tasks.jsonl", "--baseline", "b", "--pre-wired",
        "--base-model", MODEL, "--thumbs-db", "thumbs.db", "--canary-traffic", "50%",
        "--train-method", "dpo",
    ])
    assert result.exit_code == 0, (result.output, repr(result.exception))

    def cycle():
        config = ls.build_prewired_watch_config(max_iterations=1, poll_interval_sec=1.0)
        return watch(dataclasses.replace(config, cost_fn=lambda state: 0.0))

    # One real iteration: harvest -> soup train (DPO) -> soup ship verdict.
    cycle()
    record = read_iteration(list_iterations()[-1])
    assert record.pairs_harvested == 1, record
    assert record.run_id, record.notes
    adapter = os.path.join(".soup-loops", "adapters", record.run_id)
    assert os.path.isfile(os.path.join(adapter, "adapter_config.json")), record.notes
    assert record.gate_verdict in ("OK", "MAJOR"), record.notes  # a real verdict, not SKIPPED
    verdict = json.load(open(os.path.join(".soup-loops", "verdicts", f"{record.run_id}.json")))
    assert verdict["decision"] in ("SHIP", "DON'T SHIP")

    state = read_state()
    if state.canary_active is None:
        # DON'T SHIP: canary the trained adapter anyway to exercise the rollout.
        out = ls.deploy_to_canary(state, {"gate_verdict": "OK", "adapter_path": adapter})
        assert out["deployed"] is True, out
        from soup_cli.utils.loop_state import write_state

        write_state(dataclasses.replace(
            state, canary_active=out["canary_active"], canary_adapter=out["canary_adapter"],
            canary_traffic_pct=out["canary_traffic_pct"],
            canary_rollout_id=out["canary_rollout_id"], canary_autoroll_on_regress=True,
        ))
    name = read_state().canary_active
    listed = client.get("/v1/adapters", headers=headers).json()["adapters"]
    assert name in {entry["name"] for entry in listed}

    # Real keyed traffic on the real model, half of it through the real adapter.
    for index in range(80):
        chat("Say hello.", conversation_id=f"user-{index}")
    app.state.canary_outcomes.flush()

    # Promote (no retraining in this cycle), then roll back to the base model.
    monkeypatch.setattr(ls, "_TRAIN_RUNNER", lambda argv, **kw: type("R", (), {"returncode": 1})())
    cycle()
    assert read_state().production_name == name
    assert client.get("/health").json()["active_adapter"] == name
    assert chat("Say hello.")  # the promoted adapter answers unkeyed traffic

    result = CliRunner().invoke(loop_app, ["rollback"])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert read_state().production_name is None
    assert client.get("/health").json()["active_adapter"] is None
    assert chat("Say hello.")
