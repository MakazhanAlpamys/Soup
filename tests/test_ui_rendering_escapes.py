"""Web UI rendering: values are escaped, markup goes through one sink."""

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
INDEX = STATIC / "index.html"


def _code_without_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"(?m)^\s*//.*$", "", text)


class TestSingleMarkupSink:
    def test_app_js_never_assigns_inner_html(self):
        code = _code_without_comments(APP_JS.read_text(encoding="utf-8"))
        assert "innerHTML" not in code, "render through setHtml(el, html`...`)"

    def test_safe_html_is_the_only_inner_html_site(self):
        code = _code_without_comments(SAFE_JS.read_text(encoding="utf-8"))
        assert code.count("innerHTML") == 1

    @pytest.mark.parametrize(
        "sink", ["outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function"]
    )
    def test_no_other_html_or_code_sinks(self, sink):
        for path in (APP_JS, SAFE_JS):
            assert sink not in _code_without_comments(path.read_text(encoding="utf-8")), (
                path.name, sink,
            )

    def test_every_set_html_call_uses_html_template_or_constant(self):
        code = _code_without_comments(APP_JS.read_text(encoding="utf-8"))
        for match in re.finditer(r"setHtml\(([^,]+),\s*", code):
            rest = code[match.end():match.end() + 40].lstrip()
            assert rest.startswith(("html`", "'", '"')) or re.match(r"[A-Za-z_]\w*\s*\)", rest), (
                "setHtml second argument must be html`...`, a plain string literal, "
                f"or a variable holding SafeHtml: {rest!r}"
            )

    def test_no_untagged_template_joined_into_markup(self):
        # `.map(x => html`...`).join('')` turns SafeHtml into a string that html``
        # would then escape; arrays are rendered directly instead.
        code = _code_without_comments(APP_JS.read_text(encoding="utf-8"))
        assert ".join('')" not in code and '.join("")' not in code

    def test_index_loads_safe_html_before_app(self):
        text = INDEX.read_text(encoding="utf-8")
        assert text.index("/static/safe_html.js") < text.index("/static/app.js")

    def test_trusted_html_is_used_only_by_render_markdown(self):
        # trustedHtml skips escaping; its one caller builds markup from an already
        # escaped string. A second caller must be reviewed, so it fails here.
        code = _code_without_comments(APP_JS.read_text(encoding="utf-8"))
        assert code.count("trustedHtml(") == 1, code.count("trustedHtml(")
        start = code.index("function renderMarkdown(")
        end = code.index("\n}", start)
        assert start < code.index("trustedHtml(") < end, "trustedHtml outside renderMarkdown"


NODE = shutil.which("node")


def test_ci_runs_the_javascript_tests():
    """On CI a missing node is a failure, not a skip.

    The class below holds the only tests that execute the real escaping
    logic; everything above it reads the source text. If node ever leaves
    the runner image those tests would vanish quietly, so CI pins node
    (`actions/setup-node`) and this test is what notices when it does not.
    """
    if os.environ.get("CI", "").lower() != "true":
        pytest.skip("local run: node is optional")
    assert NODE is not None, "CI must provide node so the escaping tests run"


@pytest.mark.skipif(NODE is None, reason="node not installed")
class TestSafeHtmlUnderNode:
    def _run(self, body: str):
        script = (
            f"const m = require({json.dumps(str(SAFE_JS))});\n"
            "const out = (() => {" + body + "})();\n"
            "process.stdout.write(JSON.stringify(out));"
        )
        res = subprocess.run([NODE, "-e", script], capture_output=True, text=True, timeout=30)
        assert res.returncode == 0, res.stderr
        return json.loads(res.stdout)

    def test_escape_html_all_five(self):
        out = self._run("return m.escapeHtml(`<a href=\"x\" title='y'>&</a>`);")
        assert out == "&lt;a href=&quot;x&quot; title=&#39;y&#39;&gt;&amp;&lt;/a&gt;"

    def test_escape_html_nullish(self):
        out = self._run(
            "return [m.escapeHtml(null), m.escapeHtml(undefined), m.escapeHtml(0)];"
        )
        assert out == ["", "", "0"]

    def test_html_escapes_interpolations(self):
        out = self._run("return m.html`<span>${'<img src=x onerror=1>'}</span>`.value;")
        assert out == "<span>&lt;img src=x onerror=1&gt;</span>"

    def test_html_escapes_attribute_quotes(self):
        out = self._run("return m.html`<option value=\"${'a\" onmouseover=\"x'}\">`.value;")
        assert "onmouseover=\"x" not in out and "&quot;" in out

    def test_html_keeps_nested_safe_html_and_arrays(self):
        out = self._run(
            "const rows = ['<b>', 'ok'].map(v => m.html`<td>${v}</td>`);"
            "return m.html`<tr>${rows}</tr>`.value;"
        )
        assert out == "<tr><td>&lt;b&gt;</td><td>ok</td></tr>"

    def test_html_drops_nullish_and_false(self):
        assert self._run("return m.html`a${null}b${undefined}c${false}d`.value;") == "abcd"

    def test_html_keeps_numbers(self):
        assert self._run("return m.html`${0}|${1.5}`.value;") == "0|1.5"

    def test_trusted_html_passes_through(self):
        assert self._run("return m.html`${m.trustedHtml('<br>')}`.value;") == "<br>"

    def test_plain_string_is_not_trusted(self):
        assert self._run("return m.html`${'<br>'}`.value;") == "&lt;br&gt;"


INLINE_HANDLER = re.compile(r"""\son[a-z]+\s*=\s*["']""", re.I)


class TestNoInlineHandlers:
    @pytest.mark.parametrize("path", [INDEX, APP_JS], ids=["index", "app"])
    def test_no_on_attributes(self, path):
        text = path.read_text(encoding="utf-8")
        found = INLINE_HANDLER.findall(text)
        assert not found, (path.name, found)

    def test_handler_pattern_ignores_content_attribute(self):
        meta = '<meta name="viewport" content="width=device-width, initial-scale=1.0">'
        assert not INLINE_HANDLER.findall(meta)
        assert INLINE_HANDLER.findall('<button onclick="go()">')

    def test_no_javascript_urls(self):
        for path in (INDEX, APP_JS):
            assert "javascript:" not in path.read_text(encoding="utf-8").lower()

    def test_every_data_action_is_registered(self):
        code = APP_JS.read_text(encoding="utf-8")
        used = set(re.findall(r'data-(?:action|change|keydown)="([A-Za-z_]\w*)"',
                              code + INDEX.read_text(encoding="utf-8")))
        block = re.search(r"const ACTIONS = Object\.freeze\(\{(.*?)\}\);", code, re.S)
        assert block, "ACTIONS map missing"
        registered = set(re.findall(r"^\s*([A-Za-z_]\w*)\s*[:,(]", block.group(1), re.M))
        assert used, "no data-action attributes found"
        assert used <= registered, used - registered

    def test_every_input_mirror_target_exists(self):
        text = INDEX.read_text(encoding="utf-8")
        targets = re.findall(r'data-input-mirror="([\w-]+)"', text)
        assert len(targets) == 3, targets
        for target in targets:
            assert f'id="{target}"' in text, target

    def test_modal_backdrop_closes_only_on_its_own_target(self):
        text = INDEX.read_text(encoding="utf-8")
        overlay = re.search(r'<div class="modal-overlay"[^>]*>', text).group(0)
        assert "data-backdrop-close" in overlay, overlay
        code = _code_without_comments(APP_JS.read_text(encoding="utf-8"))
        assert "event.target === backdrop" in code

    def test_no_inline_script_blocks(self):
        text = INDEX.read_text(encoding="utf-8")
        for tag in re.findall(r"<script\b[^>]*>", text, re.I):
            assert "src=" in tag, tag
