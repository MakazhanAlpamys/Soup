"""#1212: the Autopilot Decisions panel's kernel notes.

The full `soup-cli[liger]` hint must reach the terminal (Rich reads `[liger]` as
markup), and no note may appear where a flag is off for another reason: a
non-SFT goal, or a pre-Ampere card.
"""

import json

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from tests.conftest import strip_ansi

runner = CliRunner()


def _plain(text):
    return " ".join(strip_ansi(text).split())


def _write_data(path, count=20):
    rows = [
        {"instruction": f"q{index} " * 8, "output": f"a{index} " * 4}
        for index in range(count)
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def _run_autopilot(tmp_path, monkeypatch, goal, compute_capability):
    """Autopilot on a patched card with neither kernel package installed."""
    from soup_cli.autopilot import decisions, generate_config
    from soup_cli.autopilot.analyzer import HardwareProfile
    from soup_cli.commands import autopilot as autopilot_cmd

    profile = HardwareProfile(
        device="cuda",
        gpu_name="rtx4090",
        vram_gb=24.0,
        compute_capability=compute_capability,
        system_ram_gb=64.0,
    )
    monkeypatch.setattr(generate_config, "analyze_hardware", lambda: profile)
    monkeypatch.setattr(autopilot_cmd, "analyze_hardware", lambda: profile)
    monkeypatch.setattr(decisions, "check_liger_available", lambda: False)
    monkeypatch.setattr(decisions, "check_flash_attn_available", lambda: None)
    # A fixed, wide terminal so the hint cannot wrap inside the panel border.
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.chdir(tmp_path)
    _write_data(tmp_path / "ap.jsonl")
    result = runner.invoke(
        app,
        [
            "autopilot",
            "--model", "HuggingFaceTB/SmolLM2-135M-Instruct",
            "--data", "ap.jsonl",
            "--goal", goal,
            "--output", "ap.yaml",
            "--yes",
        ],
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))
    return _plain(result.output)


def test_panel_prints_the_liger_extra_literally(tmp_path, monkeypatch):
    plain = _run_autopilot(tmp_path, monkeypatch, "chat", 8.6)
    assert (
        'Liger Kernel: off (liger-kernel not installed; '
        'pip install "soup-cli[liger]" to enable)'
    ) in plain
    assert (
        "Flash Attention: off (flash-attn not installed; "
        "pip install flash-attn --no-build-isolation to enable)"
    ) in plain


@pytest.mark.parametrize(
    ("goal", "compute_capability"),
    [("reasoning", 8.6), ("alignment", 8.6), ("chat", 7.5)],
)
def test_panel_shows_plain_false_when_no_install_note_applies(
    tmp_path, monkeypatch, goal, compute_capability
):
    # Non-SFT goals leave the flags off because of the task (#806), and a
    # pre-Ampere card leaves them off because of the card: neither is an
    # install problem, so the panel must not blame a missing package.
    plain = _run_autopilot(tmp_path, monkeypatch, goal, compute_capability)
    assert "Flash Attention: False" in plain
    assert "Liger Kernel: False" in plain
    assert "not installed" not in plain


# ---------------------------------------------------------------------------
# Install-aware kernel flags (#1212) — moved from tests/test_autopilot.py so
# the panel-pinning tests above and the config-level tests live together.
# ---------------------------------------------------------------------------


class TestInstallAwareKernelFlags:
    """#1212 — autopilot must not enable flags the environment cannot train.

    Neither ``liger-kernel`` (the separate ``[liger]`` extra) nor
    ``flash-attn`` is part of a standard install, so enabling them blind on
    Ampere+ wrote configs that ``soup train`` then refused.
    """

    def _build(self, tmp_path, monkeypatch, goal, compute_capability, *, liger, flash):
        from soup_cli.autopilot import decisions, generate_config
        from soup_cli.autopilot.analyzer import HardwareProfile

        monkeypatch.setattr(
            generate_config,
            "analyze_hardware",
            lambda: HardwareProfile(
                device="cuda",
                gpu_name="rtx4090",
                vram_gb=24.0,
                compute_capability=compute_capability,
                system_ram_gb=64.0,
            ),
        )
        monkeypatch.setattr(decisions, "check_liger_available", lambda: liger)
        monkeypatch.setattr(
            decisions,
            "check_flash_attn_available",
            lambda: "flash_attention_2" if flash else None,
        )
        return generate_config.build_soup_config(
            model="HuggingFaceTB/SmolLM2-135M-Instruct",
            data_path=str(_write_data(tmp_path / "data.jsonl")),
            goal=goal,
            vram_gb=24.0,
        )

    @pytest.mark.parametrize("goal", ["chat", "code", "classification", "tool-calling"])
    def test_missing_packages_disable_flags_for_every_sft_goal(
        self, tmp_path, monkeypatch, goal
    ):
        config = self._build(tmp_path, monkeypatch, goal, 8.6, liger=False, flash=False)
        assert config.task == "sft"
        assert config.training.use_flash_attn is False
        assert config.training.use_liger is False

    @pytest.mark.parametrize("compute_capability", [8.0, 8.6, 8.9, 9.0, 12.0])
    def test_present_packages_enable_flags_across_ampere_plus(
        self, tmp_path, monkeypatch, compute_capability
    ):
        config = self._build(
            tmp_path, monkeypatch, "chat", compute_capability, liger=True, flash=True
        )
        assert config.training.use_flash_attn is True
        assert config.training.use_liger is True

    def test_pre_ampere_stays_off_even_with_packages(self, tmp_path, monkeypatch):
        config = self._build(tmp_path, monkeypatch, "chat", 7.5, liger=True, flash=True)
        assert config.training.use_flash_attn is False
        assert config.training.use_liger is False

    def test_liger_only_enables_exactly_liger(self, tmp_path, monkeypatch):
        config = self._build(tmp_path, monkeypatch, "chat", 8.6, liger=True, flash=False)
        assert config.training.use_liger is True
        assert config.training.use_flash_attn is False

    def test_flash_only_enables_exactly_flash(self, tmp_path, monkeypatch):
        config = self._build(tmp_path, monkeypatch, "chat", 8.6, liger=False, flash=True)
        assert config.training.use_liger is False
        assert config.training.use_flash_attn is True

    def _patch_cli_hardware(self, monkeypatch, compute_capability=8.6):
        from soup_cli.autopilot import decisions, generate_config
        from soup_cli.autopilot.analyzer import HardwareProfile
        from soup_cli.commands import autopilot as autopilot_cmd

        profile = HardwareProfile(
            device="cuda",
            gpu_name="rtx4090",
            vram_gb=24.0,
            compute_capability=compute_capability,
            system_ram_gb=64.0,
        )
        monkeypatch.setattr(generate_config, "analyze_hardware", lambda: profile)
        monkeypatch.setattr(autopilot_cmd, "analyze_hardware", lambda: profile)
        monkeypatch.setattr(decisions, "check_liger_available", lambda: False)
        monkeypatch.setattr(decisions, "check_flash_attn_available", lambda: None)

    def test_panel_names_missing_packages(self, tmp_path, monkeypatch):
        self._patch_cli_hardware(monkeypatch)
        monkeypatch.chdir(tmp_path)
        _write_data(tmp_path / "ap.jsonl")

        result = runner.invoke(
            app,
            [
                "autopilot",
                "--model", "HuggingFaceTB/SmolLM2-135M-Instruct",
                "--data", "ap.jsonl",
                "--goal", "chat",
                "--output", "ap_chat.yaml",
                "--yes",
            ],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        plain = _plain(result.output)
        assert "Flash Attention: off (flash-attn not installed" in plain
        assert "Liger Kernel: off (liger-kernel not installed" in plain
        # The full hints wrap inside the Rich panel at unpredictable widths
        # (border glyphs land between the wrapped words), so the panel test
        # asserts the package names; the literal `soup-cli[liger]` hint is
        # asserted with a wide terminal in test_panel_prints_the_liger_extra_
        # literally above.

    def test_autopilot_config_survives_train_dry_run_without_packages(
        self, tmp_path, monkeypatch
    ):
        from soup_cli.commands import train as train_cmd

        self._patch_cli_hardware(monkeypatch)
        # `soup train` on a CPU CI box: the device is CUDA-shaped for the
        # kernel validators, and unknown VRAM (bytes=0) skips the fit gate.
        monkeypatch.setattr(
            train_cmd, "detect_device", lambda backend=None: ("cuda", "CUDA (test)")
        )
        monkeypatch.setattr(
            train_cmd,
            "get_gpu_info",
            lambda backend=None: {
                "memory_total": "24.0 GB",
                "memory_total_bytes": 0,
                "gpu_count": 1,
            },
        )
        monkeypatch.chdir(tmp_path)
        _write_data(tmp_path / "ap.jsonl")

        written = runner.invoke(
            app,
            [
                "autopilot",
                "--model", "HuggingFaceTB/SmolLM2-135M-Instruct",
                "--data", "ap.jsonl",
                "--goal", "chat",
                "--output", "ap_chat.yaml",
                "--yes",
            ],
        )
        assert written.exit_code == 0, (written.output, repr(written.exception))

        trained = runner.invoke(
            app, ["train", "--config", "ap_chat.yaml", "--dry-run", "--yes"]
        )
        assert trained.exit_code == 0, _plain(trained.output)
        assert "Data OK" in _plain(trained.output)

    def test_liger_refusal_keeps_extra_name_visible(self, tmp_path, monkeypatch):
        from soup_cli.commands import train as train_cmd
        from soup_cli.utils import liger as liger_utils

        monkeypatch.chdir(tmp_path)
        data_path = _write_data(tmp_path / "data.jsonl")
        config_path = tmp_path / "soup.yaml"
        config_path.write_text(
            "base: sshleifer/tiny-gpt2\n"
            "task: sft\n"
            f"output: {tmp_path / 'out'}\n"
            "data:\n"
            f"  train: {data_path}\n"
            "training:\n"
            "  use_liger: true\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(
            train_cmd, "detect_device", lambda backend=None: ("cuda", "CUDA (test)")
        )
        monkeypatch.setattr(
            train_cmd,
            "get_gpu_info",
            lambda backend=None: {
                "memory_total": "24.0 GB",
                "memory_total_bytes": 0,
                "gpu_count": 1,
            },
        )
        monkeypatch.setattr(liger_utils, "check_liger_available", lambda: False)

        result = runner.invoke(
            app, ["train", "--config", str(config_path), "--dry-run", "--yes"]
        )
        assert result.exit_code == 1
        plain = _plain(result.output)
        assert "liger-kernel is not installed" in plain
        # Before #1212 the hint's [liger] was swallowed as a Rich markup tag,
        # so the one instruction printed told users to reinstall what they
        # already had.
        assert 'pip install "soup-cli[liger]"' in plain
