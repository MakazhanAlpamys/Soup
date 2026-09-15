"""``stream_pin: false`` must reach the store ALLOCATION, not just the panel (#623).

The open half of #623: on a build that HAS ``training.stream_pin``, does
``false`` actually produce a pageable store, or is the key accepted and
ignored while the store lands page-locked? Measured on ``main`` at
``2285b65`` over the real chain with a real CUDA device (RTX 3050 4 GB,
torch 2.6.0+cu124, transformers 5.16.1):

    stream_pin: false  -> src.pinned False, 0/27 store tensors is_pinned()
    default (unset)    -> src.pinned True,  27/27 store tensors is_pinned()

The pair is the point: the second row is the control that makes the first
non-vacuous — page-locking demonstrably happens through this exact chain,
and ``false`` demonstrably prevents it. These tests commit that measurement
so the wiring is an enforced fact instead of a comment-thread claim.

The issue suggested asserting the store is "anonymous (not shmem)". That
discriminator is wrong on the reporter's own stack: the #622 analysis
established that driver-pinned host pages are accounted under ``shmem-rss``
by the kernel (their dumps show 11–14 GB shmem-rss against <300 MB anon-rss
WHILE the store was pinned as requested — their 0.73.3 wheel predated the
field, so the default pinned path ran). Anon-vs-shmem cannot distinguish
"ignored" from "honoured-but-pinned"; ``is_pinned()`` — whether the
allocation went through ``cudaHostAlloc`` or the pageable heap — can, on
every OS.

Two layers, because a full real chain on a simulated CUDA device cannot
allocate GPU buffers:

* the wiring layer runs on EVERY CI cell — the real pre-flight with a
  simulated-CUDA device (``device`` is a string check; ``mem_get_info``
  stubbed — the ``test_v07203.py`` pattern) and ``build_streamed_model``
  spied, asserting the ``pin=`` argument the runtime boundary receives;
* the allocation layer runs on any box with a real CUDA device (skipped
  with a stated reason elsewhere, never silently) — no stubs, real
  ``RamSource``, ``is_pinned()`` on every store tensor, both directions.

Most hops BELOW the runtime boundary are not re-tested here:
``TestRequirePinSurvivesEveryHop`` (test_v07203.py) already drives the real
CPU chain through ``build_streamed_model -> install_streaming ->
_build_source -> RamSource`` for the pin-requested direction, and the
pre-flight note for ``stream_pin: false`` is pinned by
``TestAutoTierFallback``. CPU-only throughout; the tiny checkpoint is the
established harness fixture shape.

One hop below the boundary IS covered here, added once #647 closed the
allocation gap above and left the unpinned RAM-tier construction as the one
remaining piece unprotected on a CUDA-less CI cell:
``TestBuildSourceCpuOnlyPinFalse`` calls ``_build_source(pin=False,
tier="ram")`` directly, no simulation needed, since its unpinned branch never
requests page-locked memory.
"""

from __future__ import annotations

import io
import re
from unittest.mock import MagicMock

import pytest

#: Free RAM/VRAM to report so the tiny base fits every check with room to
#: spare — the point is the pin wiring, not the budgets.
_FREE_BYTES = 10_000_000_000
_PANEL_WIDTH = 200

#: Same helper as test_v07302.py / test_auto_tuning.py. Rich WRAPS as well as
#: colours, so a figure like `read_ahead=2` can arrive split across a line
#: break at any width — collapsing whitespace is what makes a substring match
#: mean what it looks like it means.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    """ANSI-stripped, whitespace-collapsed console output, safe to match."""
    return " ".join(_ANSI_RE.sub("", text).split())


#: The runtime's own summary, as distinct from the pre-flight panel printed
#: above it. They are different statements: the panel describes the PLAN,
#: before any source exists, so it can only name the SHAPE of the staging;
#: the ready line describes the source that actually got built and carries
#: the depth and the byte count. Both are asserted, separately, below.
_READY_MARKER = "Layer streaming ready:"


def _ready_line(text: str) -> str:
    """Just the `Layer streaming ready:` summary, ANSI-stripped and unwrapped."""
    plain = _plain(text)
    assert _READY_MARKER in plain, f"no ready line was printed at all: {plain}"
    return plain.split(_READY_MARKER, 1)[1]


def _cuda() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


requires_cuda = pytest.mark.skipif(not _cuda(), reason="needs a CUDA device")


def _write_tiny_tokenizer(weights_dir) -> None:
    import json as _json
    import os

    vocab = {f"<{i}>": i for i in range(64)}
    payload = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": None,
        "decoder": None,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "<0>"},
    }
    with open(os.path.join(weights_dir, "tokenizer.json"), "w", encoding="utf-8") as fh:
        _json.dump(payload, fh)
    with open(
        os.path.join(weights_dir, "tokenizer_config.json"), "w", encoding="utf-8"
    ) as fh:
        _json.dump(
            {"tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "<0>",
             "eos_token": "<0>", "pad_token": "<0>"}, fh
        )


def _tiny_checkpoint(tmp_path) -> str:
    """A real tiny on-disk Llama checkpoint (the test_v07203 harness shape).

    hidden_size 64 is load-bearing for NF4 absmax block math in the harness
    this is copied from; kept so the fixture stays interchangeable.
    """
    import torch
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, LlamaConfig

    weights = tmp_path / "model"
    torch.manual_seed(7)
    config = LlamaConfig(
        vocab_size=64, hidden_size=64, intermediate_size=64,
        num_hidden_layers=3, num_attention_heads=4,
        num_key_value_heads=2, tie_word_embeddings=True,
        max_position_embeddings=128,
    )
    model = AutoModelForCausalLM.from_config(config).to(torch.float32).eval()
    weights.mkdir(parents=True, exist_ok=True)
    state = {k: v.contiguous() for k, v in model.state_dict().items()}
    state.pop("lm_head.weight", None)
    save_file(state, str(weights / "model.safetensors"))
    config.save_pretrained(str(weights))
    _write_tiny_tokenizer(str(weights))
    return str(weights)


def _cfg_yaml(
    weights: str,
    stream_pin_yaml: str,
    *,
    stream_source: str = "ram",
    extra_training_yaml: str = "",
) -> str:
    return (
        f"base: {weights}\ntask: sft\nbackend: transformers\nmodality: text\n"
        "data:\n  train: data.jsonl\n  format: alpaca\n"
        "training:\n  batch_size: 1\n  gradient_accumulation_steps: 1\n"
        "  quantization: none\n  stream_layers: true\n"
        f"  stream_source: {stream_source}\n"
        f"{stream_pin_yaml}"
        f"{extra_training_yaml}"
        "  lora:\n    r: 4\n    target_modules: [q_proj, v_proj]\n"
    )


def _drive_wiring(
    tmp_path,
    monkeypatch,
    *,
    stream_pin_yaml: str,
    stream_source: str = "ram",
    extra_training_yaml: str = "",
):
    """Real pre-flight, simulated CUDA, spied runtime boundary.

    Returns ``(captured_kwargs, panel_text)``. ``build_streamed_model`` is a
    spy because a simulated CUDA device cannot allocate the real GPU buffer
    pool; everything upstream — config, plan, decide_pinning, the
    ``pin=plan.pinned and on_cuda`` expression — is the real thing.
    """
    from rich.console import Console

    import soup_cli.trainer.stream_setup as ss
    import soup_cli.utils.layer_stream as ls
    import soup_cli.utils.layer_stream_runtime as rt
    from soup_cli.config.loader import load_config_from_string
    from soup_cli.trainer.sft import SFTTrainerWrapper

    weights = _tiny_checkpoint(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOUP_LAYER_STREAM_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(
        "soup_cli.utils.spectrum_scan.resolve_model_weights", lambda *_a, **_k: weights
    )
    monkeypatch.setattr(ls, "free_ram_bytes", lambda: _FREE_BYTES)
    monkeypatch.setattr(
        ls, "classify_disk_kind", lambda *_a, **_k: ls.DiskClassification("nvme")
    )
    import torch

    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda *_a, **_k: (_FREE_BYTES, _FREE_BYTES)
    )
    monkeypatch.setattr(rt, "expandable_segments_status", lambda *_a, **_k: (True, ""))

    captured: dict = {}

    def spy_build(**kwargs):
        captured.update(kwargs)
        runtime = MagicMock()
        runtime.tier = kwargs.get("tier", "ram")
        runtime.stats.return_value = {
            "tier": runtime.tier,
            "store_bytes": 1_000_000,
            "disk_bytes": 1_000_000,
            "pinned": bool(kwargs.get("pin")),
            # Derived from the captured kwargs, exactly as `pinned` is: what
            # the ready line prints must trace back to what the trainer PASSED,
            # not to a constant this spy chose (#971).
            "read_ahead": kwargs.get("read_ahead"),
            "buffers": 2,
            "buffer_bytes": 4_000,
            "n_layers": 3,
        }
        return MagicMock(), runtime

    monkeypatch.setattr(rt, "build_streamed_model", spy_build)

    buffer = io.StringIO()
    monkeypatch.setattr(ss, "console", Console(file=buffer, width=_PANEL_WIDTH))

    cfg = load_config_from_string(
        _cfg_yaml(
            weights,
            stream_pin_yaml,
            stream_source=stream_source,
            extra_training_yaml=extra_training_yaml,
        )
    )
    wrapper = SFTTrainerWrapper(cfg)
    wrapper.device = "cuda"  # the on_cuda branch is a string check
    wrapper._setup_streaming_transformers(cfg, cfg.training)
    return captured, buffer.getvalue()


def _drive_allocation(tmp_path, monkeypatch, *, stream_pin_yaml: str, cache: str):
    """Real pre-flight, real CUDA device, NO stubs — returns the RamSource."""
    import soup_cli.utils.layer_stream as ls
    from soup_cli.config.loader import load_config_from_string
    from soup_cli.trainer.sft import SFTTrainerWrapper

    weights = _tiny_checkpoint(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOUP_LAYER_STREAM_CACHE_DIR", str(tmp_path / cache))
    monkeypatch.setattr(
        "soup_cli.utils.spectrum_scan.resolve_model_weights", lambda *_a, **_k: weights
    )
    monkeypatch.setattr(ls, "free_ram_bytes", lambda: _FREE_BYTES)
    monkeypatch.setattr(
        ls, "classify_disk_kind", lambda *_a, **_k: ls.DiskClassification("nvme")
    )

    cfg = load_config_from_string(_cfg_yaml(weights, stream_pin_yaml))
    wrapper = SFTTrainerWrapper(cfg)
    wrapper.device = "cuda"
    wrapper._setup_streaming_transformers(cfg, cfg.training)
    return wrapper


class TestTheWiringReachesTheRuntimeBoundary:
    """Runs on every CI cell: the pin= argument the runtime receives."""

    def test_stream_pin_false_reaches_the_boundary_as_pin_false(
        self, tmp_path, monkeypatch
    ) -> None:
        captured, panel = _drive_wiring(
            tmp_path, monkeypatch, stream_pin_yaml="  stream_pin: false\n"
        )
        assert captured["pin"] is False, (
            "stream_pin=false never became pin=False at the runtime boundary — "
            "a hop hardcoded it away"
        )
        assert captured["require_pin"] is False
        # The resolved state is what the operator sees (issue bullet 3).
        assert "pageable" in panel
        assert "(pinned)" not in panel

    def test_the_default_still_pins_on_a_cuda_target(self, tmp_path, monkeypatch) -> None:
        # The control: without it a mutant hardcoding pin=False at the
        # boundary would pass the test above.
        captured, panel = _drive_wiring(tmp_path, monkeypatch, stream_pin_yaml="")
        assert captured["pin"] is True, (
            "the default CUDA run must still request a pinned store — "
            "the false case above is only meaningful against this"
        )
        assert captured["require_pin"] is False
        assert "(pinned)" in panel


class TestBuildSourceCpuOnlyPinFalse:
    """The gap named in the issue thread once #647 landed: the two deepest
    hops below the runtime boundary (this file's own ``TestTheAllocationItself``,
    ``TestRequirePinSurvivesEveryHop`` in test_v07203.py) are CUDA-gated, so
    the RAM tier's unpinned path is unprotected on all nine CI cells. Drives
    the real, unmocked ``_build_source(pin=False, tier="ram")`` directly: its
    ``if not pin:`` branch never reaches ``pin_memory=True``, so nothing here
    needs a CUDA device.
    """

    def test_build_source_pin_false_ram_tier_yields_unpinned_ramsource(
        self, tmp_path
    ) -> None:
        import torch
        from safetensors.torch import save_file

        from soup_cli.utils.layer_shard import layer_shard_path
        from soup_cli.utils.layer_stream_runtime import RamSource, _build_source

        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        save_file(
            {"weight": torch.zeros(4, 4, dtype=torch.float32)},
            layer_shard_path(str(shard_dir), 0),
        )
        spec = {"weight": ((4, 4), "float32")}

        source, pinned = _build_source(
            str(shard_dir), 1, spec, pin=False, console=None, tier="ram"
        )

        assert isinstance(source, RamSource)
        assert pinned is False
        assert source.pinned is False
        assert source.get(0, "weight").is_pinned() is False


@requires_cuda
class TestTheAllocationItself:
    """The measurement from the issue thread, committed: no stubs, real
    ``RamSource``, ``is_pinned()`` on every store tensor, both directions in
    ONE test so "false is pageable" can never pass on a box where pinning
    silently never happens."""

    def test_false_allocates_pageable_and_default_pins(self, tmp_path, monkeypatch) -> None:
        import torch

        from soup_cli.utils.layer_stream_runtime import RamSource

        # The discriminator itself, before trusting it with the chain.
        assert torch.empty(1024, pin_memory=True).is_pinned()
        assert not torch.empty(1024).is_pinned()

        w_false = _drive_allocation(
            tmp_path, monkeypatch, stream_pin_yaml="  stream_pin: false\n",
            cache="cache_false",
        )
        try:
            src = w_false._stream_runtime.source
            assert isinstance(src, RamSource)
            assert src.pinned is False
            flags = [t.is_pinned() for held in src.store for t in held.values()]
            assert flags, "the store is empty — nothing was measured"
            assert not any(flags), (
                f"{sum(flags)}/{len(flags)} store tensors are page-locked "
                "despite stream_pin=false — the key was accepted and ignored"
            )
        finally:
            w_false._close_stream_runtime()
            del w_false

        w_default = _drive_allocation(
            tmp_path, monkeypatch, stream_pin_yaml="", cache="cache_default"
        )
        try:
            src = w_default._stream_runtime.source
            assert isinstance(src, RamSource)
            assert src.pinned is True
            flags = [t.is_pinned() for held in src.store for t in held.values()]
            assert flags and all(flags), (
                f"only {sum(flags)}/{len(flags)} store tensors pinned on the "
                "default run — the pageable result above would be vacuous on a "
                "box that cannot pin at all"
            )
        finally:
            w_default._close_stream_runtime()
            del w_default


class TestTheReadAheadDepthIsWiredAndVisible:
    """#971 / #748: `training.stream_read_ahead` must reach the runtime AND be
    seen where it takes effect.

    Same harness as the pin tests above — real config, real pre-flight, spied
    runtime boundary — because the value has two chances to be dropped: the
    `build_streamed_model(` call, and the ready line the operator reads. The
    spy derives `read_ahead` in its fake `stats()` from the kwargs it was
    given, so the printed figure can only come from what was actually passed.
    """

    def _drive_disk(self, tmp_path, monkeypatch, *, extra_training_yaml=""):
        return _drive_wiring(
            tmp_path,
            monkeypatch,
            stream_pin_yaml="",
            stream_source="disk",
            extra_training_yaml=extra_training_yaml,
        )

    def test_a_configured_depth_reaches_the_runtime_and_the_ready_line(
        self, tmp_path, monkeypatch
    ) -> None:
        captured, panel = self._drive_disk(
            tmp_path, monkeypatch, extra_training_yaml="  stream_read_ahead: 3\n"
        )
        assert captured["read_ahead"] == 3, (
            "stream_read_ahead never became read_ahead= at the runtime "
            "boundary — a hop dropped it, which is the #748 defect"
        )
        assert "read_ahead=3" in _ready_line(panel), (
            "the depth is applied but never shown, so an operator cannot tell "
            f"which depth the run is using: {_ready_line(panel)}"
        )

    def test_the_default_is_what_an_unset_config_gets(
        self, tmp_path, monkeypatch
    ) -> None:
        """The control. Without it a mutant hardcoding 3 at the boundary — or
        printing the literal the case above looks for — would pass."""
        from soup_cli.utils.async_disk_source import DEFAULT_STREAM_READ_AHEAD

        captured, panel = self._drive_disk(tmp_path, monkeypatch)
        assert captured["read_ahead"] == DEFAULT_STREAM_READ_AHEAD
        assert f"read_ahead={DEFAULT_STREAM_READ_AHEAD}" in _ready_line(panel), _ready_line(panel)
        assert "read_ahead=3" not in _ready_line(panel), _ready_line(panel)

    def test_the_ready_line_reports_its_staging_not_nothing(
        self, tmp_path, monkeypatch
    ) -> None:
        """The ready line used to say "nothing held resident". The async reader
        stages `read_ahead` layers in host memory, so that claim is now false
        and the honest figure has to be there instead.

        Scoped to the ready line on purpose: the pre-flight PANEL is rendered
        from `plan`, before any source exists, so it cannot carry the byte
        count. The test below pins what the panel says instead, so neither
        line can quietly go back to claiming nothing is held.
        """
        _captured, panel = self._drive_disk(tmp_path, monkeypatch)
        ready = _ready_line(panel)
        assert "nothing held resident" not in ready, ready
        assert "host staging" in ready, ready

    def test_the_preflight_panel_no_longer_claims_nothing_is_resident(
        self, tmp_path, monkeypatch
    ) -> None:
        """The panel was left saying "nothing held resident" when the ready
        line was fixed, because the planner has no staging figure to print.
        It still has none — but the CLAIM was wrong either way once a reader
        began holding layers, so the panel now states the shape and leaves the
        number to the line that has it.

        Asserted over the whole captured buffer rather than the ready line,
        because this is the one statement that is NOT on the ready line.
        """
        _captured, panel = self._drive_disk(tmp_path, monkeypatch)
        plain = _plain(panel)
        assert "nothing held resident" not in plain, plain
        assert "staged by an async reader" in plain, plain

    def test_the_forced_disk_note_says_why_and_does_not_over_promise_pinning(
        self, tmp_path, monkeypatch
    ) -> None:
        """The forced-tier note was the one rewritten string with NO test: a
        repo-wide grep for "not because RAM was short" found nothing, while its
        two siblings are pinned by `test_v07203.py::
        test_the_fallback_says_what_it_costs` and by the case above.

        It also asserted the staging is "in pinned host RAM" unconditionally,
        where `pin = plan.pinned and on_cuda` can be False and `_build_source`
        falls back to pageable — the ready line then reports the truth, one
        line under a note that has already claimed otherwise.
        """
        _captured, panel = self._drive_disk(tmp_path, monkeypatch)
        plain = _plain(panel)
        # WHY this tier: reached by the flag, not by RAM pressure. Without it
        # the panel reads identically to a genuine fallback.
        assert "not because RAM was short" in plain, plain
        assert "stream_source='disk'" in plain, plain
        # What it costs, with a source — the same contract as the auto note.
        assert "slower than the RAM tier" in plain, plain
        assert "gate-971-async-nvme-source.md" in plain, plain
        # And the honest pinning wording.
        assert "page-locked where the box allows" in plain, plain
        assert "in pinned host RAM" not in plain, plain

    def test_a_forced_disk_run_still_pins_its_staging(
        self, tmp_path, monkeypatch
    ) -> None:
        """`stream_source: disk` used to zero `plan.pinned`, which since #971
        also decides whether the STAGING is page-locked. Left alone, the same
        tier reached by `disk` staged pageable and by `auto` staged pinned —
        one tier, two behaviours, chosen by the spelling."""
        captured, _panel = self._drive_disk(tmp_path, monkeypatch)
        assert captured["tier"] == "disk"
        assert captured["pin"] is True, (
            "a forced-disk run on CUDA must still request pinned staging"
        )
