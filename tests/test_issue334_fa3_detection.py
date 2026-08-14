"""#334 — the FlashAttention 3 branch could never fire, and would have lied if it did.

`check_flash_attn_available` selected FA3 when `import flash_attn` reported a major
version >= 3. **No such distribution exists.** Dao-AILab ships FA3 as package
`flash_attn_3` / module `flash_attn_interface` (and FA4 as `flash_attn_4`), while
`flash_attn` itself tops out in the 2.x line. transformers gates FA3 on
`_is_package_available("flash_attn_3")`.

Verified on an H100: with `flash_attn` spoofed to 3.0.0, Soup requested
`attn_implementation="flash_attention_3"` while
`transformers.is_flash_attn_3_available()` was False — so transformers would have
rejected it. The branch cannot fire for any real install, and on Hopper hardware
users silently got FA2 or SDPA while the docs advertised FA3.
"""

import sys
import types

import pytest


def _fake_module(name, **attrs):
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


def _with_cuda(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)


class TestFa3IsNotClaimedFromTheFlashAttnVersion:
    def test_a_3x_flash_attn_alone_does_not_yield_fa3(self, monkeypatch):
        """The load-bearing one: a `flash_attn` claiming 3.0.0 while transformers
        says FA3 is unavailable must NOT select flash_attention_3 — transformers
        would reject the value and the user silently loses the kernel."""
        from soup_cli.utils import flash_attn as mod

        _with_cuda(monkeypatch)
        monkeypatch.setitem(
            sys.modules, "flash_attn", _fake_module("flash_attn", __version__="3.0.0")
        )
        monkeypatch.setattr(mod, "_transformers_says_fa3", lambda: False)
        monkeypatch.setattr(mod, "_transformers_says_fa2", lambda: True)
        assert mod.check_flash_attn_available() == "flash_attention_2"

    def test_fa3_is_selected_when_transformers_says_it_is_available(self, monkeypatch):
        """CONTROL. Delegating must not turn into "never FA3"."""
        from soup_cli.utils import flash_attn as mod

        _with_cuda(monkeypatch)
        monkeypatch.setattr(mod, "_transformers_says_fa3", lambda: True)
        assert mod.check_flash_attn_available() == "flash_attention_3"

    def test_is_flash_attn_v3_available_agrees_with_transformers(self, monkeypatch):
        from soup_cli.utils import flash_attn as mod

        monkeypatch.setitem(
            sys.modules, "flash_attn", _fake_module("flash_attn", __version__="3.0.0")
        )
        monkeypatch.setattr(mod, "_transformers_says_fa3", lambda: False)
        assert mod.is_flash_attn_v3_available() is False

    def test_nothing_installed_still_reports_none(self, monkeypatch):
        """CONTROL for the other end: no FlashAttention at all must stay None, not
        become FA2 by accident of the refactor."""
        from soup_cli.utils import flash_attn as mod

        _with_cuda(monkeypatch)
        monkeypatch.setattr(mod, "_transformers_says_fa3", lambda: False)
        monkeypatch.setattr(mod, "_transformers_says_fa2", lambda: False)
        assert mod.check_flash_attn_available() is None
