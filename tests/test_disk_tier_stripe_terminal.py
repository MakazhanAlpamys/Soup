"""R4 fix wave, finding F — text from stripe.json, the index or the variable reaches the
terminal only through ``for_terminal``, and a malformed marker is a clean miss.

``stripe.json`` sits on a drive outside the contained primary cache. Its content used to be
echoed into a Rich-markup message: ESC bytes reached the terminal raw, ``[/]`` raised
MarkupError and ``"[" * 200000`` raised RecursionError, both aborting setup. Paths with a
``[tag]`` in them were consumed as markup. Every test here prints through a REAL Rich console,
because a ``notify=list.append`` spy cannot see any of that.
"""

import io
import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from rich.console import Console  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from soup_cli.utils.layer_shard import (  # noqa: E402
    STRIPE_MARKER_NAME,
    inspect_shard_cache,
    layer_shard_path,
    read_shard_index,
    shard_checkpoint,
    stripe_dirs,
)


def _weights(tmp_path, n_layers=3):
    torch.manual_seed(0)
    state = {}
    for idx in range(n_layers):
        pre = f"model.layers.{idx}."
        state[pre + "self_attn.q_proj.weight"] = torch.randn(8, 8)
        state[pre + "input_layernorm.weight"] = torch.randn(8)
    state["model.embed_tokens.weight"] = torch.randn(32, 8)
    state["model.norm.weight"] = torch.randn(8)
    src = tmp_path / "weights"
    src.mkdir()
    save_file(state, str(src / "model.safetensors"))
    return str(src)


def _console():
    buffer = io.StringIO()
    console = Console(
        file=buffer, force_terminal=True, color_system=None, width=10_000, soft_wrap=True
    )
    return console, buffer


@pytest.fixture
def striped(tmp_path):
    src = _weights(tmp_path)
    stripe = tmp_path / "stripe"
    stripe.mkdir()
    out = str(tmp_path / "cache" / "model")
    index = shard_checkpoint(src, out, dtype="float32", stripe_roots=(str(stripe),))
    marker = os.path.join(stripe_dirs(out, index.stripe_roots)[1], STRIPE_MARKER_NAME)
    return src, out, str(stripe), index, marker


def _write(path, data: bytes):
    with open(path, "wb") as handle:
        handle.write(data)


class TestAMalformedMarkerIsACleanMiss:
    @pytest.mark.parametrize(
        "content",
        [
            b'"\\u001bc\\u001b[2J[/][bold red]x"',  # a bare JSON string: ESC, [/], markup
            b'{"source_fingerprint": "\\u001bc[/]", "position": 1}',  # a dict with the same
            b"[1, 2]",  # valid JSON, not an object
            b"[" * 3000,  # deep nesting under the size cap: json raises RecursionError
            b"[" * 200_000,  # the reviewer's probe: far past any marker Soup writes
            b"\xff\xfe not utf-8",
        ],
        ids=["bare-string", "dict", "list", "nested-3k", "nested-200k", "not-utf8"],
    )
    def test_it_re_shards_with_nothing_echoed_and_nothing_raised(self, striped, content):
        src, out, stripe, _index, marker = striped
        _write(marker, content)
        console, buffer = _console()
        index = shard_checkpoint(
            src, out, dtype="float32", stripe_roots=(stripe,), notify=console.print
        )
        said = buffer.getvalue()
        assert "Re-sharding layer cache" in said, said
        assert "\x1b" not in said, repr(said)
        assert "[/]" not in said.replace(marker, ""), said
        assert index.layer_roots == (0, 1, 0)

    def test_the_reason_says_the_marker_is_foreign_without_quoting_it(self, striped):
        _src, out, stripe, index, marker = striped
        _write(marker, b'{"source_fingerprint": "SECRET-LOOKING", "position": 1, "n_roots": 2}')
        found, reason = inspect_shard_cache(
            out,
            "float32",
            index.source_fingerprint,
            (),
            "none",
            False,
            "",
            "",
            stripe_roots=(stripe,),
        )
        assert found is None and "another cache" in reason and marker in reason, reason
        assert "SECRET-LOOKING" not in reason, reason

    def test_an_oversized_marker_is_not_parsed(self, striped, monkeypatch):
        import soup_cli.utils.layer_shard as layer_shard_mod

        _src, out, stripe, index, marker = striped
        _write(marker, b'{"pad": "' + b"x" * 10_000 + b'"}')
        parsed = []
        real_loads = layer_shard_mod.json.loads

        def spy(text, *args, **kwargs):
            # json.load (the index read) goes through loads too; count the marker only.
            if '"pad"' in (text if isinstance(text, str) else text.decode("utf-8", "replace")):
                parsed.append(1)
            return real_loads(text, *args, **kwargs)

        monkeypatch.setattr(layer_shard_mod.json, "loads", spy)
        found, reason = inspect_shard_cache(
            out,
            "float32",
            index.source_fingerprint,
            (),
            "none",
            False,
            "",
            "",
            stripe_roots=(stripe,),
        )
        assert found is None and "not a stripe marker" in reason, reason
        assert parsed == []


class TestPathsArePrintedLiterally:
    def test_a_stripe_root_with_markup_in_its_name_is_printed_as_written(self, tmp_path):
        src = _weights(tmp_path)
        stripe = tmp_path / "stripe[bold]x"
        stripe.mkdir()
        out = str(tmp_path / "cache" / "model")
        shard_checkpoint(src, out, dtype="float32")
        console, buffer = _console()
        shard_checkpoint(
            src, out, dtype="float32", stripe_roots=(str(stripe),), notify=console.print
        )
        said = buffer.getvalue()
        # Joined plainly, not as a list repr (which doubles every Windows backslash).
        assert os.path.realpath(stripe) in said, said

    def test_the_abandoned_folder_is_named_as_written(self, tmp_path):
        src = _weights(tmp_path)
        stripe = tmp_path / "stripe[bold]x"
        stripe.mkdir()
        out = str(tmp_path / "cache" / "model")
        index = shard_checkpoint(src, out, dtype="float32", stripe_roots=(str(stripe),))
        folder = stripe_dirs(out, index.stripe_roots)[1]
        console, buffer = _console()
        shard_checkpoint(src, out, dtype="float32", notify=console.print)
        assert folder in buffer.getvalue(), buffer.getvalue()

    def test_a_stale_copy_warning_prints_its_path_and_still_commits(self, tmp_path, monkeypatch):
        """Security LOW-4: this sink is in the post-commit branch that must not raise."""
        import soup_cli.utils.layer_shard as layer_shard_mod

        src = _weights(tmp_path)
        stripe = tmp_path / "stripe"
        stripe.mkdir()
        out = str(tmp_path / "cache[bold]x" / "model")
        shard_checkpoint(src, out, dtype="float32")
        stale = layer_shard_path(layer_shard_mod._validate_out_dir(out), 1)
        real_remove = layer_shard_mod.os.remove

        def refuse_stale(path, *args, **kwargs):
            if os.path.normcase(str(path)) == os.path.normcase(stale):
                raise PermissionError("[/] in use \x1b[2J")
            return real_remove(path, *args, **kwargs)

        monkeypatch.setattr(layer_shard_mod.os, "remove", refuse_stale)
        console, buffer = _console()
        index = shard_checkpoint(
            src, out, dtype="float32", stripe_roots=(str(stripe),), notify=console.print
        )
        said = buffer.getvalue()
        assert index.layer_roots == (0, 1, 0)
        assert read_shard_index(out).layer_roots == (0, 1, 0)
        assert stale in said and "Could not delete" in said, said
        assert "\x1b" not in said, repr(said)
