"""``soup train``'s refusals for ``training.packed_lora`` (CPU).

The check runs before anything is loaded, so it needs no GPU: the machine and the model's config are
values the tests hand in. Output is read the way a user reads it: through a Rich console whose
colour is stripped and whose whitespace is collapsed (colour escapes land inside words on Linux and
macOS CI; Windows hides that).
"""

from __future__ import annotations

import io
import re

import pytest
import typer
from rich.console import Console

from soup_cli.commands import train as train_mod
from soup_cli.config.schema import SoupConfig
from soup_cli.kernels.packed_lora import preflight as checks
from soup_cli.kernels.packed_lora import support
from soup_cli.kernels.packed_lora.preflight import Machine

GIB = 1 << 30
MIB = 1 << 20
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def plain(text: str) -> str:
    return " ".join(_ANSI.sub("", text).split())


QWEN_1_5B = dict(
    vocab_size=151936, hidden_size=1536, intermediate_size=8960, num_hidden_layers=28,
    num_attention_heads=12, num_key_value_heads=2, tie_word_embeddings=True)


def machine(**changes):
    base = dict(
        system="Linux", has_triton=True, compute_capability=(12, 0), bf16_supported=True,
        world_size=1, free_ram_bytes=64 * GIB, free_vram_bytes=7000 * MIB,
        total_vram_bytes=8151 * MIB)
    return Machine(**{**base, **changes})


def soup(packed=None, **training):
    return SoupConfig(
        base="Qwen/Qwen2.5-1.5B",
        data={"train": "tests/fixtures/sample_train.jsonl", "max_length": 512},
        training={"quantization": "4bit", "batch_size": 8,
                  "lora": {"r": 16, "alpha": 32, "dropout": 0.0},
                  "packed_lora": {"enabled": True} if packed is None else packed, **training},
    )


@pytest.fixture
def screen(monkeypatch):
    """The module's console, writing to a buffer; read it with ``plain(screen.getvalue())``."""
    buffer = io.StringIO()
    monkeypatch.setattr(train_mod, "console", Console(file=buffer, width=200, force_terminal=False))
    return buffer


@pytest.fixture
def host(monkeypatch):
    """A Linux machine with the 8 GB card of the dev box, and a Qwen2.5-1.5B config."""
    from transformers import Qwen2Config

    state = {"machine": machine(), "config": Qwen2Config(**QWEN_1_5B)}
    monkeypatch.setattr(checks, "read_machine", lambda: state["machine"])
    monkeypatch.setattr(checks, "read_model_config", lambda base, trust=False: state["config"])
    monkeypatch.setattr(support, "attention_signature_problems", lambda: [])
    return state


def run(cfg, **kwargs):
    return train_mod._packed_lora_preflight(cfg, allow_oom_attempt=False, **kwargs)


class TestAcceptedRun:
    def test_without_the_block_nothing_is_checked_and_the_generic_estimate_applies(
            self, monkeypatch, screen):
        def explode(*args, **kwargs):
            raise AssertionError("looked at a run that did not ask for the kernels")

        monkeypatch.setattr(checks, "preflight", explode)
        assert run(soup({"enabled": False})) is False
        assert screen.getvalue() == ""

    def test_a_run_they_can_take_returns_true_and_prints_the_plan(self, host, screen):
        assert run(soup({"enabled": True, "gpu_budget_mib": 3000})) is True
        text = plain(screen.getvalue())
        assert "Packed LoRA kernels: plan: weights resident" in text
        assert "est peak" in text and "of budget 3000 MiB" in text
        assert "4096 tokens" in text

    def test_with_no_budget_it_says_what_the_floor_is(self, host, screen):
        assert run(soup()) is True
        text = plain(screen.getvalue())
        assert "no GPU budget, so what fits stays resident" in text
        assert re.search(r"floor is about \d+ MiB at 4096 tokens per step", text)

    def test_the_trust_remote_code_flag_reaches_the_config_read(self, host, monkeypatch, screen):
        seen = []
        monkeypatch.setattr(checks, "read_model_config",
                            lambda base, trust=False: seen.append((base, trust)) or host["config"])
        run(soup(), trust_remote_code=True)
        assert seen == [("Qwen/Qwen2.5-1.5B", True)]


class TestRefusedRun:
    def test_the_reason_the_way_out_and_exit_one(self, host, screen):
        host["machine"] = machine(system="Windows")
        with pytest.raises(typer.Exit) as excinfo:
            run(soup())
        assert excinfo.value.exit_code == 1
        text = plain(screen.getvalue())
        assert "Packed LoRA error: training.packed_lora runs on Linux (and WSL2) only; " \
               "this is Windows." in text
        assert "Set training.packed_lora.on_unsupported: stock to train with the stock step " \
               "instead, or remove training.packed_lora." in text

    @pytest.mark.parametrize("label", ["--fsdp full_shard", "--deepspeed"])
    def test_a_sharded_run_is_refused(self, host, screen, label):
        with pytest.raises(typer.Exit):
            run(soup(), sharding=label)
        assert f"supports one GPU only; {label} is set" in plain(screen.getvalue())

    def test_every_reason_is_printed(self, host, screen):
        host["machine"] = machine(system="Windows", has_triton=False, world_size=2)
        with pytest.raises(typer.Exit):
            run(soup())
        assert plain(screen.getvalue()).count("Packed LoRA error:") == 3

    def test_a_model_the_kernels_do_not_run(self, host, screen):
        from transformers import LlamaConfig

        host["config"] = LlamaConfig()
        with pytest.raises(typer.Exit):
            run(soup())
        assert "supports Qwen2ForCausalLM only" in plain(screen.getvalue())

    def test_a_budget_below_the_floor_names_the_minimum(self, host, screen):
        with pytest.raises(typer.Exit):
            run(soup({"enabled": True, "gpu_budget_mib": 600}))
        assert "gpu_budget_mib=600 is below the minimum of" in plain(screen.getvalue())

    def test_a_reason_cannot_restyle_the_terminal(self, host, monkeypatch, screen):
        hostile = checks.Preflight(
            problems=["training.packed_lora [bold red]x[/] \x1b[31m"])
        monkeypatch.setattr(checks, "preflight", lambda *args, **kwargs: hostile)
        with pytest.raises(typer.Exit):
            run(soup())
        raw = screen.getvalue()
        assert "\x1b" not in raw and "[bold red]x[/]" in raw

    def test_under_on_unsupported_stock_it_says_why_and_the_generic_estimate_applies(
            self, host, screen):
        host["machine"] = machine(system="Windows")
        assert run(soup({"enabled": True, "on_unsupported": "stock"})) is False
        text = plain(screen.getvalue())
        assert "Packed LoRA kernels not used: training.packed_lora runs on Linux" in text
        assert "training with the stock step" in text


class TestMemoryThatIsNotFree:
    def test_a_busy_card_is_refused_with_the_way_out(self, host, screen):
        host["machine"] = machine(free_vram_bytes=2500 * MIB)
        with pytest.raises(typer.Exit) as excinfo:
            run(soup({"enabled": True, "gpu_budget_mib": 3000}))
        assert excinfo.value.exit_code == 1
        text = plain(screen.getvalue())
        assert "of GPU memory is free right now" in text
        assert "pass --allow-oom-attempt to try anyway" in text

    def test_not_enough_pinned_ram_is_refused(self, host, screen):
        host["machine"] = machine(free_ram_bytes=1 * GIB)
        with pytest.raises(typer.Exit):
            run(soup({"enabled": True, "gpu_budget_mib": 3000}))
        assert "pinned host memory" in plain(screen.getvalue())

    def test_allow_oom_attempt_goes_on_and_says_so(self, host, screen):
        host["machine"] = machine(free_ram_bytes=1 * GIB)
        result = train_mod._packed_lora_preflight(
            soup({"enabled": True, "gpu_budget_mib": 3000}), allow_oom_attempt=True)
        assert result is True
        text = plain(screen.getvalue())
        assert "pinned host memory" in text
        assert "--allow-oom-attempt set: launching anyway." in text
        assert "Packed LoRA kernels: plan:" in text  # and the plan is still printed

    def test_allow_oom_attempt_does_not_override_an_unsupported_run(self, host, screen):
        host["machine"] = machine(system="Windows")
        with pytest.raises(typer.Exit):
            train_mod._packed_lora_preflight(soup(), allow_oom_attempt=True)


# --- through `soup train` --------------------------------------------------------------------


YAML = """\
base: Qwen/Qwen2.5-1.5B
task: sft
data: {{train: data.jsonl, format: alpaca, val_split: 0, max_length: 512}}
training:
  epochs: 1
  quantization: 4bit
  batch_size: 8
  lora: {{r: 16, alpha: 32, dropout: 0}}
  packed_lora: {{enabled: true{extra}}}
"""


@pytest.fixture
def project(tmp_path, monkeypatch, host):
    pytest.importorskip("torch")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "runs.db"))
    monkeypatch.setattr(
        "soup_cli.commands.train.detect_device", lambda backend=None: ("cpu", "CPU"))
    monkeypatch.setattr(
        "soup_cli.commands.train.get_gpu_info", lambda backend=None: {"memory_total": "N/A"})
    (tmp_path / "data.jsonl").write_text(
        '{"instruction":"hi", "output":"hello"}\n' * 4, encoding="utf-8")

    def write(extra=""):
        (tmp_path / "soup.yaml").write_text(YAML.format(extra=extra), encoding="utf-8")

    return write


def invoke(*args):
    from typer.testing import CliRunner

    from soup_cli.cli import app

    return CliRunner().invoke(app, ["train", "--config", "soup.yaml", *args])


class TestSoupTrain:
    def test_a_refused_run_exits_one_before_the_dataset_is_read(self, project, host,
                                                                 monkeypatch):
        host["machine"] = machine(system="Windows")
        project()

        def explode(*args, **kwargs):
            raise AssertionError("the dataset was read for a run that was refused")

        monkeypatch.setattr("soup_cli.commands.train.load_dataset", explode)
        result = invoke("--dry-run")
        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert "Packed LoRA error: training.packed_lora runs on Linux (and WSL2) only" in plain(
            result.output)

    def test_under_on_unsupported_stock_the_run_goes_on_with_the_generic_estimate(
            self, project, host, monkeypatch):
        host["machine"] = machine(system="Windows")
        project(", on_unsupported: stock")
        seen = []
        monkeypatch.setattr(train_mod, "_hardware_fit_preflight",
                            lambda *args, **kwargs: seen.append(1))
        result = invoke("--dry-run")
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "Packed LoRA kernels not used" in plain(result.output)
        assert seen == [1]

    def test_an_accepted_run_prints_its_plan_and_skips_the_generic_estimate(
            self, project, host, monkeypatch):
        project(", gpu_budget_mib: 3000")

        def explode(*args, **kwargs):
            raise AssertionError("the generic estimate priced a run the planner prices")

        monkeypatch.setattr(train_mod, "_hardware_fit_preflight", explode)
        result = invoke("--dry-run")
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "Packed LoRA kernels: plan: weights resident" in plain(result.output)

    def test_a_run_without_the_block_still_gets_the_generic_estimate(
            self, project, host, monkeypatch):
        import pathlib

        pathlib.Path("soup.yaml").write_text(
            YAML.replace("  packed_lora: {{enabled: true{extra}}}\n", "").format(extra=""),
            encoding="utf-8")
        seen = []
        monkeypatch.setattr(train_mod, "_hardware_fit_preflight",
                            lambda *args, **kwargs: seen.append(1))
        result = invoke("--dry-run")
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert seen == [1]
        assert "Packed LoRA" not in plain(result.output)

    def test_an_fsdp_run_is_refused_with_the_setting_named(self, project, host):
        project()
        result = invoke("--dry-run", "--fsdp", "full_shard")
        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert "supports one GPU only; --fsdp full_shard is set" in plain(result.output)

    def test_trust_remote_code_reaches_the_config_read(self, project, host, monkeypatch):
        project()
        seen = []
        monkeypatch.setattr(
            checks, "read_model_config",
            lambda base, trust=False: seen.append(trust) or host["config"])
        assert invoke("--dry-run").exit_code == 0
        result = invoke("--dry-run", "--trust-remote-code")
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert seen == [False, True]

    def test_allow_oom_attempt_is_passed_through(self, project, host, monkeypatch):
        host["machine"] = machine(free_ram_bytes=1 * GIB)
        project(", gpu_budget_mib: 3000")
        refused = invoke("--dry-run")
        assert refused.exit_code == 1
        allowed = invoke("--dry-run", "--allow-oom-attempt")
        assert allowed.exit_code == 0, (allowed.output, repr(allowed.exception))
        assert "--allow-oom-attempt set: launching anyway." in plain(allowed.output)
