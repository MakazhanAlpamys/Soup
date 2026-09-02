"""`stream_source: auto` must see the physical ceiling, not just free RAM (#622).

The reproduction: Qwen2.5-32B NF4, a 16.10 GB store + 3.11 GB resident on a
30 GB host with 27 GB "available" — `choose_tier` sized the RAM tier against
`MemAvailable * 0.70` alone, the kernel OOM-killed the run (exit 137) before
the first step, twice, cold and warm shard cache. The store is
unevictable-class memory (page-locked via the CUDA host allocator; the
reporter's OOM dump accounts it under shmem-rss, and either accounting is
memory the kernel cannot reclaim), so "available" is the wrong denominator:
the ceiling is absolute, a share of TOTAL physical RAM.

The fix follows the dual-threshold shape proposed on the issue thread: RAM
tier requires BOTH `store < available * 0.70` (unchanged) AND
`store + resident < total * 0.55` (new, `RAM_TIER_PHYSICAL_BUDGET`). Auto
falls through to the NVMe disk tier with a note naming the ceiling; forced
`stream_source: ram` refuses — early, before sharding, and again at the
exact-bytes validator — with a message naming it.

What CANNOT be tested here, stated per the claim: the live exit-137
reproduction needs the reporter's Linux box (30 GB RAM, /dev/shm, a 32B
checkpoint). These tests drive the decision logic and the REAL pre-flight
(`_setup_streaming_transformers`, the test_v07203.py harness pattern) with
mocked ceilings — the wiring mutation that leaves unit tests green dies here.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock

import pytest

# --- the reproduction's numbers, from the issue body -----------------------
_STORE_BYTES = 16_100_000_000  # "base store 16.10 GB across 64 layers"
_RESIDENT_BYTES = 3_110_000_000  # "resident 3114 MB embeddings + adapters"
_BILLED = _STORE_BYTES + _RESIDENT_BYTES  # what the plan bills to the RAM tier
_HOST_TOTAL = 30_000_000_000  # 30 GB physical
_HOST_FREE_RUN2 = 29_000_000_000  # run 2 started with 27 GB free and warm cache;
# 29 GB keeps the DYNAMIC check passing (0.7 * 29 = 20.3 > 19.2) so the test
# isolates the absolute ceiling — the point of the issue is that free RAM
# alone green-lit the kill.
_BUDGET = 0.55  # RAM_TIER_PHYSICAL_BUDGET


class TestChooseTierGainsTheAbsoluteCeiling:
    def test_the_reproduction_lands_on_the_disk_tier(self) -> None:
        from soup_cli.utils.layer_stream import TIER_DISK, choose_tier

        assert (
            choose_tier(
                _BILLED, _HOST_FREE_RUN2, "nvme", total_ram_bytes=_HOST_TOTAL
            )
            == TIER_DISK
        ), "19.2 GB of unevictable store on a 30 GB host must not take the RAM tier"

    def test_the_dynamic_check_alone_is_what_used_to_pass(self) -> None:
        # The control that pins the bug's shape: without a total, the same
        # numbers take the RAM tier — MemAvailable * 0.70 cannot see the
        # ceiling. This is the pre-#622 behaviour, kept honest for callers
        # that genuinely cannot resolve total RAM (psutil absent).
        from soup_cli.utils.layer_stream import TIER_RAM, choose_tier

        assert choose_tier(_BILLED, _HOST_FREE_RUN2, "nvme") == TIER_RAM

    def test_a_large_host_is_not_forced_to_disk(self) -> None:
        # The false positive the issue thread warned about: a big workstation
        # keeps the RAM tier for the same store.
        from soup_cli.utils.layer_stream import TIER_RAM, choose_tier

        assert (
            choose_tier(_BILLED, 40_000_000_000, "nvme", total_ram_bytes=128_000_000_000)
            == TIER_RAM
        )

    def test_the_boundary_is_the_budget_exactly(self) -> None:
        from soup_cli.utils.layer_stream import TIER_DISK, TIER_RAM, choose_tier

        total = 40_000_000_000
        ceiling = int(total * _BUDGET)  # 22 GB
        assert choose_tier(ceiling, total, "nvme", total_ram_bytes=total) == TIER_DISK
        assert choose_tier(ceiling - 1, total, "nvme", total_ram_bytes=total) == TIER_RAM

    def test_the_non_nvme_refusal_names_the_binding_ceiling(self) -> None:
        # "More RAM" is not actionable advice when free RAM passed and the
        # absolute ceiling refused — the refusal must say which one bit.
        from soup_cli.utils.layer_stream import choose_tier

        with pytest.raises(ValueError) as excinfo:
            choose_tier(_BILLED, _HOST_FREE_RUN2, "hdd", total_ram_bytes=_HOST_TOTAL)
        message = str(excinfo.value)
        assert "physical ceiling" in message
        assert "55%" in message
        assert "30.0 GB total RAM" in message

    def test_the_refusal_without_a_total_keeps_its_old_shape(self) -> None:
        from soup_cli.utils.layer_stream import choose_tier

        with pytest.raises(ValueError) as excinfo:
            choose_tier(_BILLED, 10_000_000_000, "hdd")
        assert "physical ceiling" not in str(excinfo.value)


class TestThePredicate:
    def test_unknown_total_means_unenforced(self) -> None:
        from soup_cli.utils.layer_stream import ram_tier_over_physical_budget

        assert ram_tier_over_physical_budget(_BILLED, None) is False

    def test_over_and_under(self) -> None:
        from soup_cli.utils.layer_stream import ram_tier_over_physical_budget

        assert ram_tier_over_physical_budget(_BILLED, _HOST_TOTAL) is True
        assert ram_tier_over_physical_budget(10_000_000_000, _HOST_TOTAL) is False


class TestThePlanNoteSaysWhichCeilingRefused:
    def _plan(self, *, available: int, total: "int | None"):
        from soup_cli.utils.layer_stream import build_stream_plan

        return build_stream_plan(
            arch="qwen2",
            n_layers=64,
            layer_bytes=_STORE_BYTES // 64,
            embed_bytes=_RESIDENT_BYTES,
            available_ram_bytes=available,
            pinned_limit_bytes=None,
            disk_kind="nvme",
            total_ram_bytes=total,
        )

    def test_the_budget_fallback_note_names_the_ceiling_and_figures(self) -> None:
        from soup_cli.utils.layer_stream import TIER_DISK

        plan = self._plan(available=_HOST_FREE_RUN2, total=_HOST_TOTAL)
        assert plan.tier == TIER_DISK
        note = "\n".join(plan.notes)
        assert "physical ceiling" in note
        assert "55%" in note
        assert "#622" in note
        # The operator sees 27+ GB free and would read a bare "does not fit in
        # RAM" as a bug — the note must state why free RAM is not the measure.
        assert "free RAM alone does not make it safe" in note

    def test_a_dynamic_shortfall_keeps_the_original_note(self) -> None:
        # The control on note selection: plenty of physical RAM, little free —
        # the old note, no ceiling language.
        from soup_cli.utils.layer_stream import TIER_DISK

        plan = self._plan(available=10_000_000_000, total=500_000_000_000)
        assert plan.tier == TIER_DISK
        note = "\n".join(plan.notes)
        assert "base does not fit in RAM" in note
        assert "physical ceiling" not in note


class TestForcedRamRefusesAgainstTheCeiling:
    def test_the_exact_bytes_validator_names_the_ceiling(self) -> None:
        from soup_cli.trainer.stream_setup import _validate_qwen4_ngram_ram_fit

        with pytest.raises(ValueError) as excinfo:
            _validate_qwen4_ngram_ram_fit(
                stream_source="ram",
                ngram_source="disk",
                required_ram=_BILLED,
                free_ram=_HOST_FREE_RUN2,
                total_ram=_HOST_TOTAL,
            )
        message = str(excinfo.value)
        assert "stream_source='ram'" in message
        assert "physical ceiling" in message
        assert "OOM" in message

    def test_a_store_inside_both_ceilings_passes(self) -> None:
        from soup_cli.trainer.stream_setup import _validate_qwen4_ngram_ram_fit

        _validate_qwen4_ngram_ram_fit(
            stream_source="ram",
            ngram_source="disk",
            required_ram=10_000_000_000,
            free_ram=_HOST_FREE_RUN2,
            total_ram=_HOST_TOTAL,
        )

    def test_callers_that_do_not_know_total_ram_keep_the_old_contract(self) -> None:
        # The 602 pin exercises this shape; restated so the compatibility is
        # deliberate: no total -> only the dynamic check, unchanged message.
        from soup_cli.trainer.stream_setup import _validate_qwen4_ngram_ram_fit

        with pytest.raises(ValueError, match="GB of RAM is free"):
            _validate_qwen4_ngram_ram_fit(
                stream_source="ram",
                ngram_source="disk",
                required_ram=81,
                free_ram=100,
            )


# ---------------------------------------------------------------------------
# The wiring. Unit tests of choose_tier/build_stream_plan would stay green if
# stream_setup never passed total_ram_bytes — the #628 lesson ("nothing
# crossed the wiring"). These drive the REAL _setup_streaming_transformers
# over the test_v07203.py harness pattern: probes patched on the source
# module (stream_setup imports inside the method), build_streamed_model
# stubbed so nothing reaches a kernel, disk kind pinned so the tier cannot
# depend on the runner's hardware.
# ---------------------------------------------------------------------------


def _write_tiny_tokenizer(weights_dir) -> None:
    """A minimal tokenizer so the trainer's AutoTokenizer load succeeds."""
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
    """A real tiny on-disk Llama checkpoint (the test_v07203 harness shape)."""
    import torch
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, LlamaConfig

    weights = tmp_path / "model"
    torch.manual_seed(7)
    # hidden_size 64 is load-bearing for the NF4 absmax block math in the
    # harness this is copied from; kept so the fixture stays interchangeable.
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


def _drive_preflight(tmp_path, monkeypatch, *, free_ram, total_ram, stream_source):
    """Run the real pre-flight with mocked ceilings; return (captured, panel)."""
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
    monkeypatch.setattr(ls, "free_ram_bytes", lambda: free_ram)
    monkeypatch.setattr(ls, "total_ram_bytes", lambda: total_ram)
    # Pinned rather than probed: the real media type differs between this box
    # (NVMe) and a CI runner (often "unknown"), and an environment-dependent
    # tier would make these flaky rather than wrong.
    monkeypatch.setattr(
        ls, "classify_disk_kind", lambda *_a, **_k: ls.DiskClassification("nvme")
    )

    captured: dict = {}

    def fake_build(**kwargs):
        captured.update(kwargs)
        runtime = MagicMock()
        runtime.tier = kwargs.get("tier", "ram")
        runtime.stats.return_value = {
            "tier": runtime.tier,
            "store_bytes": 1_000_000,
            "disk_bytes": 1_000_000,
            "pinned": False,
            "buffers": 2,
            "buffer_bytes": 4_000,
            "n_layers": 3,
        }
        return MagicMock(), runtime

    monkeypatch.setattr(rt, "build_streamed_model", fake_build)

    buffer = io.StringIO()
    monkeypatch.setattr(ss, "console", Console(file=buffer, width=200))

    cfg = load_config_from_string(
        f"base: {weights}\ntask: sft\nbackend: transformers\nmodality: text\n"
        "data:\n  train: data.jsonl\n  format: alpaca\n"
        "training:\n  batch_size: 1\n  gradient_accumulation_steps: 1\n"
        "  quantization: none\n  stream_layers: true\n"
        f"  stream_source: {stream_source}\n"
        "  lora:\n    r: 4\n    target_modules: [q_proj, v_proj]\n"
    )
    wrapper = SFTTrainerWrapper(cfg)
    wrapper.device = "cpu"
    wrapper._setup_streaming_transformers(cfg, cfg.training)
    return captured, buffer.getvalue()


class TestTheRealPreflightEnforcesTheCeiling:
    # The tiny checkpoint's store is ~10^5 bytes; a 1000-byte "host" puts it
    # over the 550-byte budget exactly the way 19.2 GB sits over 16.5 GB in
    # the reproduction — same predicate, scaled.
    _TINY_HOST_TOTAL = 1_000
    _BIG_HOST_TOTAL = 100_000_000_000

    def test_auto_falls_to_disk_and_the_panel_names_the_ceiling(
        self, tmp_path, monkeypatch
    ) -> None:
        captured, panel = _drive_preflight(
            tmp_path, monkeypatch,
            free_ram=10_000_000_000,  # dynamic check passes with room to spare
            total_ram=self._TINY_HOST_TOTAL,
            stream_source="auto",
        )
        assert captured["tier"] == "disk", (
            "the tier decision never reached the runtime — the plan threaded "
            "the RAM tier despite the physical ceiling"
        )
        assert "tier disk" in panel
        assert "physical ceiling" in panel
        assert "55%" in panel

    def test_auto_keeps_ram_when_the_ceiling_is_not_binding(
        self, tmp_path, monkeypatch
    ) -> None:
        # The control: without it a mutant that ALWAYS falls back to disk
        # would pass the test above.
        captured, panel = _drive_preflight(
            tmp_path, monkeypatch,
            free_ram=10_000_000_000,
            total_ram=self._BIG_HOST_TOTAL,
            stream_source="auto",
        )
        assert captured["tier"] == "ram"
        assert "tier ram" in panel
        assert "physical ceiling" not in panel

    def test_forced_ram_is_refused_before_sharding_naming_the_ceiling(
        self, tmp_path, monkeypatch
    ) -> None:
        # The early estimate-based refusal: minutes of shard I/O must not be
        # spent on a run the absolute ceiling has already sentenced — and the
        # refusal must not read as a free-RAM problem.
        with pytest.raises(ValueError) as excinfo:
            _drive_preflight(
                tmp_path, monkeypatch,
                free_ram=10_000_000_000,
                total_ram=self._TINY_HOST_TOTAL,
                stream_source="ram",
            )
        message = str(excinfo.value)
        assert "stream_source='ram'" in message
        assert "physical ceiling" in message
        assert "free RAM is free" not in message
        # Refused BEFORE the shards were written.
        assert not (tmp_path / "cache").exists() or not any(
            (tmp_path / "cache").rglob("layer_*.safetensors")
        )

    def test_forced_ram_proceeds_when_the_ceiling_is_not_binding(
        self, tmp_path, monkeypatch
    ) -> None:
        captured, _ = _drive_preflight(
            tmp_path, monkeypatch,
            free_ram=10_000_000_000,
            total_ram=self._BIG_HOST_TOTAL,
            stream_source="ram",
        )
        assert captured["tier"] == "ram"


class TestTotalRamProbe:
    def test_total_ram_bytes_reads_psutil_total(self, monkeypatch) -> None:
        import soup_cli.utils.layer_stream as ls

        class _VM:
            total = 30_000_000_000
            available = 27_000_000_000

        import psutil

        monkeypatch.setattr(psutil, "virtual_memory", lambda: _VM())
        assert ls.total_ram_bytes() == 30_000_000_000
        # Distinct from the free probe: the whole point of #622 is that these
        # two numbers answer different questions.
        assert ls.free_ram_bytes() == 27_000_000_000
