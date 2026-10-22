"""The Web UI's training log panel opens, and follows the run's log.

``index.html`` ships a "Training Logs" card (``#train-log-panel``) hidden with
``display:none``, and ``app.js`` has ``connectTrainingSSE()`` to show it and
stream ``/api/train/logs`` into it. Nothing called that function, so starting a
run from the New Training page showed "Training started! PID: ..." and never a
line of output.

The page now follows the log when a run is started from it, and when it loads
while a run is already in progress. The static scan reads the source; the node
runs execute ``app.js`` against a stubbed browser and check what the page does.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[1] / "src" / "soup_cli" / "ui" / "static"
APP_JS = STATIC / "app.js"
SAFE_JS = STATIC / "safe_html.js"
INDEX_HTML = STATIC / "index.html"

NODE = shutil.which("node")


def _code_without_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"(?m)^\s*//.*$", "", text)


class TestStaticScan:
    def test_the_log_panel_ships_hidden(self):
        # What the page must undo: if the markup ever ships the card visible,
        # the node tests below would pass without the page doing anything.
        html = INDEX_HTML.read_text(encoding="utf-8")
        assert re.search(r'<div id="train-log-panel"[^>]*style="display:none"', html)
        assert 'id="log-output"' in html

    def test_connect_training_sse_has_callers(self):
        code = _code_without_comments(APP_JS.read_text(encoding="utf-8"))
        calls = re.findall(r"(?<!function )\bconnectTrainingSSE\(\)", code)
        assert len(calls) >= 2, "the start handler and the page load must both call it"


def test_node_is_present_on_ci():
    """On CI a missing node is a failure, not a skip (CI pins it with setup-node)."""
    if not os.environ.get("CI"):
        pytest.skip("local run: node is optional")
    assert NODE is not None, "CI must provide node so the log-panel tests run"


# Runs safe_html.js + app.js in a vm context with a stubbed browser. fetch answers
# by path; EventSource is a spy the scenario can push events through.
_HARNESS = r"""
const vm = require('vm');
const fs = require('fs');
const [SAFE, APP, OPTS, BODY] = JSON.parse(process.argv[1]);
const rec = { fetch: [], sources: [], confirms: 0 };
const elements = {};
const HIDDEN = ['train-log-panel', 'train-progress', 'live-badge'];
const element = (id) => elements[id] || (elements[id] = {
  id, value: id === 'config-editor' ? OPTS.yaml : '', checked: true, textContent: '',
  innerHTML: '', className: '', scrollTop: 0, scrollHeight: 0,
  style: { display: HIDDEN.includes(id) ? 'none' : '' },
  replaceChildren() {}, appendChild(child) { return child; },
});
let tickets = 0;
const answers = {
  '/api/train/start': () => (OPTS.startStatus === 200
    ? [200, { started: true, pid: 4242 }]
    : [OPTS.startStatus, { detail: 'Training already in progress' }]),
  '/api/train/status': () => [200, { running: OPTS.running, pid: 4242 }],
  '/api/templates': () => [200, { templates: { sft: 'base: tiny' } }],
  '/api/recipes': () => [200, { recipes: [] }],
  '/api/auth/ticket': () => [200, { ticket: 'T' + (tickets += 1) }],
};
function FakeEventSource(url) {
  this.url = url;
  this.readyState = 1;
  this.closed = false;
  this.handlers = {};
  rec.sources.push(this);
}
FakeEventSource.CLOSED = 2;
FakeEventSource.prototype.addEventListener = function (type, fn) { this.handlers[type] = fn; };
FakeEventSource.prototype.close = function () { this.closed = true; this.readyState = 2; };
FakeEventSource.prototype.toJSON = function () { return { url: this.url, closed: this.closed }; };
const ctx = {
  URLSearchParams, URL, AbortController, TextDecoder, console, JSON, Promise, Object, Array,
  String, Error, Math, Date, parseInt, encodeURIComponent, setTimeout, clearTimeout,
  setInterval: () => 0, clearInterval: () => {},
  location: {
    origin: 'http://ui.test', pathname: '/', search: '', hash: '',
    get href() { return this.origin + this.pathname + this.search + this.hash; },
  },
  history: { state: null, replaceState() {}, pushState() {} },
  document: {
    addEventListener() {},
    getElementById: (id) => (OPTS.missing.includes(id) ? null : element(id)),
    createElement: (tag) => element(tag + '#new'),
    querySelector: () => null,
    querySelectorAll: () => [],
  },
  fetch: async (url, opts = {}) => {
    const path = String(url).split('?')[0];
    rec.fetch.push({ path, method: opts.method || 'GET' });
    if (path === '/api/auth/ticket' && OPTS.ticketDelay) {
      await new Promise((resolve) => setTimeout(resolve, OPTS.ticketDelay));
    }
    const [status, body] = (answers[path] || (() => [404, { detail: 'no route' }]))();
    return { status, ok: status < 400, statusText: '', json: async () => body };
  },
  prompt: () => '',
  alert: () => {},
  confirm: () => { rec.confirms += 1; return OPTS.confirm; },
  EventSource: FakeEventSource,
  Element: function () {},
};
ctx.window = ctx;
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(SAFE, 'utf8'), ctx);
vm.runInContext(fs.readFileSync(APP, 'utf8'), ctx);
ctx.rec = rec;
ctx.el = element;
// Lets every pending promise and zero-delay timer of the page run.
ctx.settle = async (ms = 30) => { await new Promise((resolve) => setTimeout(resolve, ms)); };
vm.runInContext('(async () => { ' + BODY + '\n})()', ctx)
  .then((out) => {
    const shown = {};
    for (const id of ['train-log-panel', 'train-progress', 'live-badge']) {
      shown[id] = element(id).style.display;
    }
    process.stdout.write(JSON.stringify({
      out: out === undefined ? null : out, rec, shown,
      log: element('log-output').textContent, status: element('config-status').innerHTML,
    }));
    process.exit(0);
  })
  .catch((e) => { process.stderr.write(String(e && e.stack || e)); process.exit(1); });
"""


def _run(body: str, **opts) -> dict:
    options = {
        "yaml": "base: tiny",
        "running": False,
        "confirm": True,
        "startStatus": 200,
        "ticketDelay": 0,
        "missing": [],
    }
    options.update(opts)
    args = json.dumps([str(SAFE_JS), str(APP_JS), options, body])
    res = subprocess.run(
        [NODE, "-e", _HARNESS, args], capture_output=True, text=True, timeout=30,
        encoding="utf-8", check=False,
    )
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout)


def _log_streams(data: dict) -> list[dict]:
    return [src for src in data["rec"]["sources"] if src["url"].startswith("/api/train/logs")]


@pytest.mark.skipif(NODE is None, reason="node not installed")
class TestStartingARunOpensTheLogPanel:
    def test_the_panel_is_shown_and_the_log_stream_is_opened(self):
        data = _run("await startTraining(); await settle();", running=True)

        assert "Training started! PID: 4242" in data["status"]
        assert data["shown"]["train-log-panel"] == "block"
        assert data["shown"]["live-badge"] == "inline-flex"
        streams = [src for src in _log_streams(data) if not src["closed"]]
        assert len(streams) == 1, data["rec"]["sources"]
        # The single-use ticket is how an EventSource, which cannot set a
        # header, gets past the token check.
        assert streams[0]["url"] == "/api/train/logs?ticket=T1"

    def test_log_lines_reach_the_panel_in_order(self):
        data = _run(
            "await startTraining(); await settle();"
            "const es = rec.sources[rec.sources.length - 1];"
            "es.onmessage({ data: JSON.stringify({ line: 'step 1 loss 2.31', id: 0 }) });"
            "es.onmessage({ data: JSON.stringify({ line: 'step 2 loss 2.07', id: 1 }) });",
            running=True,
        )

        assert data["log"] == "step 1 loss 2.31\nstep 2 loss 2.07\n"

    def test_the_end_of_the_run_closes_the_stream_and_keeps_the_log_readable(self):
        data = _run(
            "await startTraining(); await settle();"
            "const es = rec.sources[rec.sources.length - 1];"
            "es.onmessage({ data: JSON.stringify({ line: 'saving adapter', id: 0 }) });"
            "es.handlers.done();",
            running=True,
        )

        assert data["log"] == "saving adapter\n[Training finished]\n"
        assert all(src["closed"] for src in _log_streams(data))
        assert data["shown"]["live-badge"] == "none"
        assert data["shown"]["train-log-panel"] == "block"

    def test_a_run_that_exits_at_once_still_gets_its_log(self):
        # The status refresh that follows the start already says "not running";
        # the output of such a run is the error message the user needs.
        data = _run("await startTraining(); await settle();", running=False)

        assert data["shown"]["train-log-panel"] == "block"
        assert len(_log_streams(data)) == 1

    def test_start_and_the_status_refresh_open_one_stream_not_two(self):
        # The start handler connects, then refreshes the page, which sees a run
        # in progress. The ticket request is slow enough for the two to overlap.
        data = _run(
            "await startTraining(); await settle(120);", running=True, ticketDelay=40
        )

        streams = _log_streams(data)
        assert len([src for src in streams if not src["closed"]]) == 1, streams
        assert len(streams) == 1, streams

    def test_a_connect_overtaken_while_it_waits_for_its_ticket_opens_nothing(self):
        # A page load is still fetching its ticket when a new run is started:
        # only the newer connect may open a stream, or every line shows twice.
        data = _run(
            "const first = connectTrainingSSE(); const second = connectTrainingSSE();"
            "await first; await second; await settle(120);"
            "for (const es of rec.sources) {"
            "  if (!es.closed) es.onmessage({ data: JSON.stringify({ line: 'step 1', id: 0 }) });"
            "}",
            ticketDelay=40,
        )

        assert len(_log_streams(data)) == 1, data["rec"]["sources"]
        assert data["log"] == "step 1\n"

    def test_a_declined_confirmation_starts_and_opens_nothing(self):
        data = _run("await startTraining(); await settle();", confirm=False)

        assert data["rec"]["confirms"] == 1
        assert "/api/train/start" not in [req["path"] for req in data["rec"]["fetch"]]
        assert data["rec"]["sources"] == []
        assert data["shown"]["train-log-panel"] == "none"

    def test_a_refused_start_opens_nothing(self):
        data = _run("await startTraining(); await settle();", startStatus=409)

        assert "Error: Training already in progress" in data["status"]
        assert data["rec"]["sources"] == []
        assert data["shown"]["train-log-panel"] == "none"


@pytest.mark.skipif(NODE is None, reason="node not installed")
class TestLoadingThePageDuringARun:
    def test_a_run_in_progress_is_followed(self):
        # A reload, or a second tab: the server replays the log from line 0.
        data = _run("await loadTrainingPage(); await settle();", running=True)

        assert data["shown"]["train-log-panel"] == "block"
        assert data["shown"]["live-badge"] == "inline-flex"
        assert [src["url"] for src in _log_streams(data)] == ["/api/train/logs?ticket=T1"]

    def test_no_run_means_no_panel_and_no_stream(self):
        data = _run("await loadTrainingPage(); await settle();", running=False)

        assert data["rec"]["sources"] == []
        assert data["shown"]["train-log-panel"] == "none"
        assert data["shown"]["live-badge"] == "none"

    def test_revisiting_the_page_does_not_restart_a_stream_it_already_follows(self):
        data = _run(
            "await loadTrainingPage(); await settle();"
            "const es = rec.sources[0];"
            "es.onmessage({ data: JSON.stringify({ line: 'step 1', id: 0 }) });"
            "await loadTrainingPage(); await settle();",
            running=True,
        )

        assert len(_log_streams(data)) == 1
        assert data["log"] == "step 1\n"

    def test_a_reconnect_after_the_stream_ended_starts_from_an_empty_panel(self):
        # The new connection replays every line, so the old text must go first.
        data = _run(
            "await loadTrainingPage(); await settle();"
            "rec.sources[0].onmessage({ data: JSON.stringify({ line: 'step 1', id: 0 }) });"
            "rec.sources[0].handlers.done();"
            "await loadTrainingPage(); await settle();"
            "rec.sources[1].onmessage({ data: JSON.stringify({ line: 'step 1', id: 0 }) });",
            running=True,
        )

        assert len(_log_streams(data)) == 2
        assert data["log"] == "step 1\n"

    def test_a_page_without_the_panel_does_not_fail(self):
        data = _run(
            "await loadTrainingPage(); await settle();",
            running=True,
            missing=["train-log-panel"],
        )

        assert data["rec"]["sources"] == []


@pytest.mark.skipif(NODE is None, reason="node not installed")
class TestTheProgressCardStaysHidden:
    def test_nothing_shows_a_bar_that_no_route_can_move(self):
        # No route reports a total step count for the run this server started,
        # so the card would sit at "Step 0/0" for the whole run.
        data = _run("await startTraining(); await settle();", running=True)

        assert data["shown"]["train-progress"] == "none"
