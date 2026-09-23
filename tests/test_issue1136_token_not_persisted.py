"""#1136 — the Web UI auth token lives only in a closure in app.js.

It used to sit in ``sessionStorage`` and on ``window._authToken``, where any script
running in the UI origin could read it. The static scan catches a storage API coming
back; the node run executes app.js and catches the token leaking under any other name.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[1] / "src" / "soup_cli" / "ui" / "static"
APP_JS = STATIC / "app.js"
SAFE_JS = STATIC / "safe_html.js"

SECRET = "S3cretTokenValue1136xyz"


def _code_without_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"(?m)^\s*//.*$", "", text)


class TestStaticScan:
    @pytest.mark.parametrize(
        "api", ["sessionStorage", "localStorage", "document.cookie", "indexedDB", "_authToken"]
    )
    def test_app_js_uses_no_persistent_carrier(self, api):
        assert api not in _code_without_comments(APP_JS.read_text(encoding="utf-8"))

    def test_bearer_header_is_built_in_one_place(self):
        # Four call sites used to build their own header; each was a way around the closure.
        code = _code_without_comments(APP_JS.read_text(encoding="utf-8"))
        assert code.count("'Bearer '") == 1


NODE = shutil.which("node")

# Runs safe_html.js + app.js in a vm context with a stubbed browser. Web storage, the
# cookie setter, fetch and prompt are spies; the scenario body runs after load.
_HARNESS = r"""
const vm = require('vm');
const fs = require('fs');
const [SAFE, APP, SECRET, SEARCH, PROMPT, STATUSES, BODY] = JSON.parse(process.argv[1]);
const rec = { storage: [], cookie: [], fetch: [], prompt: 0, replaced: null };
const statuses = STATUSES.slice();
const store = (name) => ({
  setItem: (k, v) => rec.storage.push([name, 'set', k, String(v)]),
  getItem: (k) => { rec.storage.push([name, 'get', k]); return null; },
  removeItem: (k) => rec.storage.push([name, 'remove', k]),
  clear: () => rec.storage.push([name, 'clear']),
});
const ctx = {
  URLSearchParams, console, JSON, Promise, Object, Array, String, Error,
  setTimeout, clearTimeout, setInterval: () => 0, clearInterval: () => {},
  location: { search: SEARCH, pathname: '/', hash: '' },
  history: {
    replaceState: (_s, _t, url) => {
      // As a browser does: the new URL replaces location, the old one is gone.
      rec.replaced = url;
      ctx.location.search = url.includes('?') ? url.slice(url.indexOf('?')) : '';
    },
  },
  sessionStorage: store('session'),
  localStorage: store('local'),
  document: {
    addEventListener: () => {},
    getElementById: () => null,
    querySelector: () => null,
    querySelectorAll: () => [],
    get cookie() { return ''; },
    set cookie(v) { rec.cookie.push(String(v)); },
  },
  fetch: async (url, opts = {}) => {
    rec.fetch.push({ url, method: opts.method || 'GET', headers: { ...(opts.headers || {}) } });
    // An entry is a status, or [status, delayMs] to make a response arrive late.
    const [status, delay] = [].concat(statuses.length ? statuses.shift() : 200, 0);
    if (delay) await new Promise((resolve) => setTimeout(resolve, delay));
    return {
      status, ok: status < 400, statusText: '',
      json: async () => (status < 400 ? { ticket: 'T1' } : { detail: 'Unauthorized' }),
    };
  },
  prompt: () => { rec.prompt += 1; return PROMPT; },
  EventSource: function () {},
  Element: function () {},
};
ctx.window = ctx;
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(SAFE, 'utf8'), ctx);
const appSource = fs.readFileSync(APP, 'utf8');
vm.runInContext(appSource, ctx);

// Every name app.js declares at top level, plus every property of the global object.
function leaks() {
  const names = new Set(Object.getOwnPropertyNames(ctx));
  const decl = /^(?:async\s+)?(?:function\*?|const|let|var|class)\s+([\w$]+)/gm;
  for (const m of appSource.matchAll(decl)) names.add(m[1]);
  const found = [];
  const seen = new Set();
  const walk = (value, path, depth) => {
    if (typeof value === 'string') { if (value.includes(SECRET)) found.push(path); return; }
    if (value === null || (typeof value !== 'object' && typeof value !== 'function')) return;
    if (seen.has(value) || depth > 4) return;
    seen.add(value);
    for (const key of Object.getOwnPropertyNames(value)) {
      const desc = Object.getOwnPropertyDescriptor(value, key);
      if (desc && 'value' in desc) walk(desc.value, path + '.' + key, depth + 1);
    }
  };
  for (const name of names) {
    let value;
    try { value = vm.runInContext(name, ctx); } catch (e) { continue; }
    walk(value, name, 0);
  }
  return found;
}

ctx.leaks = leaks;
vm.runInContext('(async () => { ' + BODY + '\n})()', ctx)
  .then((out) => {
    process.stdout.write(JSON.stringify({ out: out === undefined ? null : out, rec }));
  })
  .catch((e) => { process.stderr.write(String(e && e.stack || e)); process.exit(1); });
"""


@pytest.mark.skipif(NODE is None, reason="node not installed")
class TestTokenUnderNode:
    def _run(self, body: str, search: str = f"?token={SECRET}&tab=1",
             prompt: "str | None" = None, statuses: "list[int] | None" = None):
        args = json.dumps(
            [str(SAFE_JS), str(APP_JS), SECRET, search, prompt, statuses or [], body]
        )
        res = subprocess.run(
            [NODE, "-e", _HARNESS, args], capture_output=True, text=True, timeout=30
        )
        assert res.returncode == 0, res.stderr
        data = json.loads(res.stdout)
        return data["out"], data["rec"]

    def test_authenticated_request_carries_the_token(self):
        # Control: without it the leak tests below pass vacuously on a token never loaded.
        _, rec = self._run("await authFetch('/api/runs');")
        assert rec["fetch"][0]["headers"]["Authorization"] == f"Bearer {SECRET}"

    def test_token_never_reaches_web_storage_or_cookies(self):
        _, rec = self._run("await getAuthTicket();")
        assert not [call for call in rec["storage"] if SECRET in json.dumps(call)]
        assert not [value for value in rec["cookie"] if SECRET in value]

    def test_token_unreachable_from_any_global_name(self):
        out, _ = self._run("await getAuthTicket(); return leaks();")
        assert out == []

    def test_leak_scan_finds_a_global_holding_the_token(self):
        # Control for the scan itself: a token on a global must be found.
        out, _ = self._run(f"window.someName = {{ nested: '{SECRET}' }}; return leaks();")
        assert [path for path in out if path.endswith("someName.nested")], out

    def test_token_is_stripped_from_the_url(self):
        _, rec = self._run("")
        assert rec["replaced"] == "/?tab=1"

    def test_sse_ticket_post_carries_the_token(self):
        out, rec = self._run("return await getAuthTicket();")
        assert out == "T1"
        (call,) = rec["fetch"]
        assert call["url"] == "/api/auth/ticket"
        assert call["method"] == "POST"
        assert call["headers"]["Authorization"] == f"Bearer {SECRET}"

    def test_reload_prompts_once_and_retries_concurrent_401s(self):
        out, rec = self._run(
            "const rs = await Promise.all([authFetch('/a'), authFetch('/b')]);"
            "return rs.map(r => r.status);",
            search="", prompt="NEW", statuses=[401, 401],
        )
        assert out == [200, 200]
        assert rec["prompt"] == 1
        assert [call["headers"].get("Authorization") for call in rec["fetch"][2:]] == [
            "Bearer NEW", "Bearer NEW",
        ]

    def test_late_401_after_the_prompt_retries_without_asking_again(self):
        out, rec = self._run(
            "const rs = await Promise.all([authFetch('/a'), authFetch('/b')]);"
            "return rs.map(r => r.status);",
            search="", prompt="NEW", statuses=[401, [401, 50]],
        )
        assert out == [200, 200]
        assert rec["prompt"] == 1

    def test_replacing_window_fetch_later_does_not_see_the_header(self):
        out, rec = self._run(
            "const seen = []; window.fetch = (u, o) => { seen.push(o); return null; };"
            "await authFetch('/api/runs'); return seen.length;"
        )
        assert out == 0
        assert rec["fetch"][0]["headers"]["Authorization"] == f"Bearer {SECRET}"

    @pytest.mark.parametrize("answer", [None, ""], ids=["cancel", "empty"])
    def test_declined_prompt_is_not_repeated(self, answer):
        out, rec = self._run(
            "const a = await authFetch('/a'); const b = await authFetch('/b');"
            "return [a.status, b.status];",
            search="", prompt=answer, statuses=[401, 401],
        )
        assert out == [401, 401]
        assert rec["prompt"] == 1
        assert len(rec["fetch"]) == 2
