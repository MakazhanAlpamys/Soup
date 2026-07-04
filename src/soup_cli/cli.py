"""Main CLI entry point — all commands registered here."""

import os
import sys

# UTF-8 stdio bootstrap (v0.40.1 Part A) — must run before any Rich console
# is constructed. On Windows, reconfigures sys.stdout/stderr to UTF-8 so β /
# ✓ / box-drawing chars don't crash with UnicodeEncodeError on cp1251/cp1252.
# POSIX: no-op.
from soup_cli.utils.encoding import force_utf8_stdio

force_utf8_stdio()
_utf8_bootstrap_done = True

import typer  # noqa: E402
from rich.console import Console  # noqa: E402

from soup_cli import __version__  # noqa: E402
from soup_cli.commands import (  # noqa: E402
    adapters,
    autopilot,
    bench,
    can,
    chat,
    cost,
    data,
    deploy,
    diff,
    eval,
    export,
    generate,
    history,
    infer,
    init,
    merge,
    migrate,
    profile,
    push,
    recipes,
    registry,
    runs,
    serve,
    sweep,
    train,
    ui,
)

# v0.44.0 — Live monitoring + standalone CLI wrappers.
from soup_cli.commands import (  # noqa: E402
    delinearize_llama4 as delinearize_llama4_cmd,
)
from soup_cli.commands import doctor as doctor_cmd  # noqa: E402
from soup_cli.commands import fetch as fetch_cmd  # noqa: E402
from soup_cli.commands import llama as llama_cmd  # noqa: E402
from soup_cli.commands import (  # noqa: E402
    merge_sharded_fsdp_weights as merge_sharded_fsdp_weights_cmd,
)
from soup_cli.commands import monitor as monitor_cmd  # noqa: E402
from soup_cli.commands import quantize as quantize_cmd  # noqa: E402
from soup_cli.commands import quickstart as quickstart_cmd  # noqa: E402
from soup_cli.commands import spectrum as spectrum_cmd  # noqa: E402
from soup_cli.commands import (  # noqa: E402
    tui as tui_cmd,
)
from soup_cli.commands import (  # noqa: E402
    why as why_cmd,
)
from soup_cli.utils.constants import GITHUB_URL  # noqa: E402

console = Console()

# Global verbose flag — set via callback, read by error handler
_verbose = False
# Global log level (resolved string), set by main() callback
_log_level = "normal"
# v0.71.3 #183 — audit-log opt-out, set by the --no-audit-log callback flag.
_audit_disabled = False

# Global options that consume a following value (so the audit command-splitter
# does not mistake the value for the subcommand name).
_GLOBAL_VALUE_OPTS = frozenset({"--log-level"})

app = typer.Typer(
    name="soup",
    help=(
        "Fine-tune and post-train LLMs in one command. No SSH, no config hell.\n\n"
        f"[dim]GitHub: {GITHUB_URL}[/]"
    ),
    no_args_is_help=True,
    rich_markup_mode="rich",
)

# Register sub-commands
app.command()(init.init)
app.command()(train.train)
app.command()(chat.chat)
app.command()(cost.cost)
app.command()(push.push)
app.command(name="export")(export.export)
app.command()(merge.merge)
app.add_typer(
    data.app, name="data",
    help="Dataset tools: inspect, convert, merge, dedup, validate, stats.",
)
app.add_typer(
    deploy.app, name="deploy",
    help="Deploy models: Ollama integration (deploy, list, remove).",
)
app.add_typer(runs.app, name="runs", help="Experiment tracking: list, show, compare runs.")
app.add_typer(
    eval.app, name="eval",
    help="Evaluate models: benchmarks, custom evals, LLM judge, leaderboard.",
)
app.command()(migrate.migrate)
app.add_typer(
    adapters.app, name="adapters",
    help="Adapter management: list, info, compare LoRA adapters.",
)
app.add_typer(
    recipes.app, name="recipes",
    help="Ready-made configs: list, show, use, search recipes for popular models.",
)
app.command()(serve.serve)
app.command()(sweep.sweep)
app.command(name="diff")(diff.diff)
app.command()(infer.infer)
app.command()(profile.profile)
app.command()(bench.bench)
app.command()(doctor_cmd.doctor)
app.command()(quickstart_cmd.quickstart)
app.command()(ui.ui)
app.command(name="autopilot")(autopilot.autopilot_cmd)
app.add_typer(
    registry.app, name="registry",
    help="Model Registry: push, list, show, diff, search, promote, delete.",
)
app.command(name="history")(history.history)
app.command(name="why")(why_cmd.why)
app.command(name="tui")(tui_cmd.tui)
app.add_typer(
    spectrum_cmd.app, name="spectrum",
    help="Spectrum SNR scan for targeted training (v0.71.23).",
)
app.add_typer(
    can.app, name="can",
    help="Soup Cans: pack/inspect/verify/fork shareable .can artifacts.",
)

# v0.44.0 — register Live Dashboard & UX commands.
app.command(name="monitor")(monitor_cmd.monitor)
app.command(name="fetch")(fetch_cmd.fetch)
app.command(name="quantize")(quantize_cmd.quantize)
app.command(name="merge-sharded-fsdp-weights")(
    merge_sharded_fsdp_weights_cmd.merge_sharded_fsdp_weights
)
app.command(name="delinearize-llama4")(delinearize_llama4_cmd.delinearize_llama4)
app.add_typer(
    llama_cmd.app,
    name="llama",
    help="Proxy to llama.cpp binaries (cli / mtmd-cli / gguf-split / server).",
)

# v0.45.0 Part A — Plugin system CLI.
from soup_cli.commands import plugins as plugins_cmd  # noqa: E402

app.add_typer(
    plugins_cmd.app,
    name="plugins",
    help="List, enable, disable Soup plugins (v0.45.0).",
)

# v0.46.0 Part B — Agent Forge.
from soup_cli.commands import agent as agent_cmd  # noqa: E402

app.add_typer(
    agent_cmd.app,
    name="agent",
    help="Agent Forge: spec -> tool-calling dataset / train / eval (v0.46.0).",
)

# Register data generate as a subcommand of data
data.app.command(name="generate")(generate.generate)

# v0.47.0 Part A — Synthetic Data Forge.
from soup_cli.commands import data_forge as _data_forge_cmd  # noqa: E402

data.app.command(name="forge")(_data_forge_cmd.forge)

# v0.47.0 Part B — Data Quality Moat.
from soup_cli.commands import data_score as _data_score_cmd  # noqa: E402

data.app.command(name="score")(_data_score_cmd.score)
data.app.command(name="decontaminate")(_data_score_cmd.decontaminate)
data.app.command(name="toxicity")(_data_score_cmd.toxicity)
data.app.command(name="langdetect")(_data_score_cmd.langdetect)
data.app.command(name="pii")(_data_score_cmd.pii)
data.app.command(name="educational")(_data_score_cmd.educational)

# v0.48.0 Part B — Data Mixing Optimizer (BETA).
from soup_cli.commands import data_mix as _data_mix_cmd  # noqa: E402

data.app.command(name="mix")(_data_mix_cmd.mix)

# v0.53.9 #15 — BPE tokenizer training.
from soup_cli.commands import tokenizer as _tokenizer_cmd  # noqa: E402

app.add_typer(
    _tokenizer_cmd.app,
    name="tokenizer",
    help="Tokenizer tools: train a BPE tokenizer from JSONL (v0.53.9).",
)

# v0.54.0 — `soup advise` pre-flight decision engine.
from soup_cli.commands import advise as _advise_cmd  # noqa: E402

app.add_typer(
    _advise_cmd.app,
    name="advise",
    help=(
        "Pre-flight decision: PROMPT_ENG / RAG / SFT / DPO / GRPO. Run "
        "BEFORE you spend 8 hours on a GPU (v0.54.0)."
    ),
)

# v0.56.0 — `soup diagnose` post-training failure-mode report card.
from soup_cli.commands import diagnose as _diagnose_cmd  # noqa: E402

app.command(
    name="diagnose",
    help=(
        "Post-training report card: forgetting / refusal / format / "
        "mode_collapse / memorization / contamination (v0.56.0)."
    ),
)(_diagnose_cmd.diagnose)

# v0.71.25 — `soup ship` SHIP / DON'T-SHIP verdict engine.
from soup_cli.commands import ship as _ship_cmd  # noqa: E402

app.add_typer(
    _ship_cmd.app,
    name="ship",
    help=(
        "SHIP / DON'T SHIP verdict after fine-tuning: task win AND no "
        "catastrophic forgetting, fused into one decision (v0.71.25)."
    ),
)

# v0.58.0 — `soup loop` CLI-first data flywheel capstone.
from soup_cli.commands import loop as _loop_cmd  # noqa: E402

app.add_typer(
    _loop_cmd.app,
    name="loop",
    help=(
        "Data flywheel: traces -> preference pairs -> DPO -> gate -> "
        "canary deploy -> rollback, all from the CLI (v0.58.0)."
    ),
)

# v0.59.0 — Governance & Provenance: BOM emit + attestation + audit log.
from soup_cli.commands import attest as _attest_cmd  # noqa: E402
from soup_cli.commands import audit_log as _audit_log_cmd  # noqa: E402
from soup_cli.commands import bom as _bom_cmd  # noqa: E402

app.add_typer(
    _bom_cmd.app,
    name="bom",
    help=(
        "CycloneDX ML-BOM + SPDX AI bill-of-materials emitter (v0.59.0)."
    ),
)
app.add_typer(
    _attest_cmd.app,
    name="attest",
    help=(
        "In-toto + SLSA-3 attestations per Soup Can stage (v0.59.0)."
    ),
)
app.add_typer(
    _audit_log_cmd.app,
    name="audit-log",
    help=(
        "HIPAA/SOC2-shaped JSONL audit log: tail + rotate (v0.59.0)."
    ),
)

# v0.60.0 — Supply Chain Security: airgap bundle assembler.
from soup_cli.commands import airgap as _airgap_cmd  # noqa: E402

app.command(name="airgap-bundle")(_airgap_cmd.airgap_bundle)

# v0.61.0 — Unlearning & Knowledge Edit: `soup edit set / diff`.
from soup_cli.commands import edit as _edit_cmd  # noqa: E402

app.add_typer(
    _edit_cmd.app,
    name="edit",
    help=(
        "Knowledge editing (ROME / MEMIT / AlphaEdit) - patch facts "
        "without re-training (v0.61.0)."
    ),
)

# v0.62.0 Part C — Activation steering: `soup steer train / apply / list`.
from soup_cli.commands import steer as _steer_cmd  # noqa: E402

app.add_typer(
    _steer_cmd.app,
    name="steer",
    help=(
        "Activation steering (CAA / ITI / RepE) - inference-time "
        "intervention without retraining (v0.62.0)."
    ),
)

# v0.63.0 Part A — Universal trace importer.
from soup_cli.commands import ingest as _ingest_cmd  # noqa: E402

app.command(
    name="ingest",
    help=(
        "Universal trace importer: Langfuse / LangSmith / Helicone / "
        "OpenPipe / OTel / OpenAI Stored Completions (v0.63.0)."
    ),
)(_ingest_cmd.ingest)

# v0.63.0 Part B — Strip shared system-prompt prefix.
from soup_cli.commands import prune_prompt as _prune_prompt_cmd  # noqa: E402

app.command(
    name="prune-prompt",
    help=(
        "Detect + strip a shared system-prompt prefix across training "
        "data so the FT model internalises it (v0.63.0)."
    ),
)(_prune_prompt_cmd.prune_prompt_cmd)

# v0.63.0 Part C — Active-learning sampler from prod traces.
from soup_cli.commands import active_sample as _active_sample_cmd  # noqa: E402

data.app.command(name="active-sample")(_active_sample_cmd.active_sample)

# v0.63.0 Part D — mSPRT A/B harness.
from soup_cli.commands import ab as _ab_cmd  # noqa: E402

app.command(
    name="ab",
    help=(
        "mSPRT sequential A/B harness on latency / judge_score / retry_rate "
        "with early-stop guarantees (v0.63.0)."
    ),
)(_ab_cmd.ab)

# v0.63.0 Part E — Online-eval drift alarm.
from soup_cli.commands import drift_alarm as _drift_alarm_cmd  # noqa: E402

app.command(
    name="drift-alarm",
    help=(
        "Rolling-KL drift alarm on output-token distribution with "
        "optional Slack/Discord webhook (v0.63.0)."
    ),
)(_drift_alarm_cmd.drift_alarm)

# v0.64.0 Part A — Tunability probe across candidate bases.
from soup_cli.commands import tunability as _tunability_cmd  # noqa: E402

app.command(
    name="tunability",
    help=(
        "Probe-train 6-10 small bases on a held-out slice + report "
        "Pareto frontier of (eval delta, train cost, license) (v0.64.0)."
    ),
)(_tunability_cmd.tunability_cmd)

# v0.64.0 Part B — Terraform-shape plan / apply.
from soup_cli.commands import plan as _plan_cmd  # noqa: E402

app.command(
    name="plan",
    help=(
        "Render a pre-flight training plan (cost / ETA / SHA / VRAM) "
        "and write soup.tfstate for `soup apply` to consult (v0.64.0)."
    ),
)(_plan_cmd.plan_cmd)

app.command(
    name="apply",
    help=(
        "Execute the planned training run, refusing on drift between "
        "soup.yaml and soup.tfstate (v0.64.0)."
    ),
)(_plan_cmd.apply_cmd)

# v0.64.0 Part C — Hermetic env lockfile.
from soup_cli.commands.env import env_app as _env_app  # noqa: E402

app.add_typer(_env_app, name="env")

# v0.64.0 Part E — Shell completions.
from soup_cli.commands import completions as _completions_cmd  # noqa: E402

app.command(
    name="completions",
    help=(
        "Emit a bash / zsh / fish completion script. Use with "
        "`eval \"$(soup completions bash)\"` (v0.64.0)."
    ),
)(_completions_cmd.completions_cmd)

# v0.64.0 Part F — License advisor.
from soup_cli.commands import license_advisor as _license_advisor_cmd  # noqa: E402

app.command(
    name="license-advisor",
    help=(
        "Recommend a license-clean base for a deploy target "
        "(b2c / defense / embedded) + flag downstream risk (v0.64.0)."
    ),
)(_license_advisor_cmd.license_advisor_cmd)

# v0.66.0 Part C/D/E — Post-train X-rays: sleeper / interference / sae-diff
# / probe pack.
from soup_cli.commands import probe as _probe_cmd  # noqa: E402

app.add_typer(
    _probe_cmd.app,
    name="probe",
    help=(
        "Activation probes: sleeper-agent defection / honesty / misuse / "
        "pairwise interference / SAE feature diff / probe pack (v0.66.0, "
        "truth+harm v0.71.8)."
    ),
)

# v0.67.0 Part E — soup.lock shared run lockfile.
from soup_cli.commands import lock as _lock_cmd  # noqa: E402

app.add_typer(
    _lock_cmd.app,
    name="lock",
    help="Shared run lockfile (write / show / check) - v0.67.0 Part E.",
)

# v0.68.0 — Anti-trend insurance (compile / distill-prompt / compile-tools /
# apple-adapter / local-rl).
from soup_cli.commands import apple_adapter as _apple_cmd  # noqa: E402
from soup_cli.commands import compile_cmd as _compile_cmd  # noqa: E402
from soup_cli.commands import compile_tools as _compile_tools_cmd  # noqa: E402
from soup_cli.commands import distill_prompt as _distill_prompt_cmd  # noqa: E402
from soup_cli.commands import local_rl as _local_rl_cmd  # noqa: E402

app.command(
    name="compile",
    help="Compile a DSPy / GEPA prompt program against an eval suite (v0.68.0).",
)(_compile_cmd.compile_cmd)

app.command(
    name="distill-prompt",
    help="Distill prompt-heavy traces into a small FT plan (v0.68.0).",
)(_distill_prompt_cmd.distill_prompt_cmd)

app.command(
    name="compile-tools",
    help="Compile / optimize tool schemas + descriptions (v0.68.0).",
)(_compile_tools_cmd.compile_tools_cmd)

app.command(
    name="apple-adapter",
    help="Convert / sign HF / MLX / Apple FoundationModels adapters (v0.68.0).",
)(_apple_cmd.apple_adapter_cmd)

app.add_typer(
    _local_rl_cmd.app,
    name="local-rl",
    help="Personal-LLM flywheel daemon (init / status / record / harvest / train) (v0.68.0).",
)

# v0.69.0 Part A — `soup build` (dbt-for-SFT DAG).
from soup_cli.commands import build as _build_cmd  # noqa: E402

app.command(
    name="build",
    help="dbt-for-SFT DAG: validate + plan dataset transforms (v0.69.0 Part A).",
)(_build_cmd.build_cmd)

# v0.69.0 Part B — `soup expect` (expectations suite).
from soup_cli.commands import expect as _expect_cmd  # noqa: E402

app.command(
    name="expect",
    help="Run an expectations suite against a JSONL dataset (v0.69.0 Part B).",
)(_expect_cmd.expect_cmd)

# v0.70.0 Part E — `soup iterative-dpo` (iterative DPO loop driver).
from soup_cli.commands import iterative_dpo as _iterative_dpo_cmd  # noqa: E402

app.add_typer(
    _iterative_dpo_cmd.app,
    name="iterative-dpo",
    help="Iterative DPO loop driver (v0.70.0 Part E).",
)

# v0.71.10 #200 — `soup ra-dit` (two-stage RA-DIT orchestrator).
from soup_cli.commands import ra_dit as _ra_dit_cmd  # noqa: E402

app.add_typer(
    _ra_dit_cmd.app,
    name="ra-dit",
    help="RA-DIT two-stage orchestrator: retriever -> generator (v0.71.10).",
)

# v0.71.27 — Fine-tune Doctor: chat-template doctor + loss-mask X-ray +
# preference linter.
from soup_cli.commands import data_doctor as _data_doctor_cmd  # noqa: E402

data.app.command(name="doctor")(_data_doctor_cmd.doctor)
data.app.command(name="lint")(_data_doctor_cmd.lint)

# v0.71.28 — MCP server: drive Soup from any MCP client (Claude Code / Cursor /
# Cline / Continue) over stdio.
from soup_cli.commands import mcp as _mcp_cmd  # noqa: E402

app.add_typer(
    _mcp_cmd.app,
    name="mcp",
    help="Model Context Protocol server - drive Soup from any MCP client (v0.71.28).",
)


def _rewrite_advise_argv(argv: list) -> list:
    """Inject `run` between `advise` and a non-subcommand first argument.

    Lets users type ``soup advise data.jsonl`` instead of the explicit
    ``soup advise run data.jsonl``. Click's group/positional collision
    makes the bare-positional design impossible at the parser level, so
    we rewrite argv before Typer ever sees it.

    Scope: ONLY fires when ``advise`` is the first non-script argument
    (``argv[1]``). Any other position is treated as unrelated data — a
    dataset path or option value that happens to contain the literal
    string ``"advise"`` MUST NOT trigger rewriting (code-review HIGH).
    """
    if len(argv) < 2 or argv[1] != "advise":
        return argv
    known_subs = {"run", "explain", "compare", "--help", "-h"}
    tail = argv[2:]
    if not tail:
        return argv
    first = tail[0]
    if first in known_subs or first.startswith("-"):
        return argv
    return argv[:2] + ["run"] + tail


@app.command()
def version(
    full: bool = typer.Option(False, "--full", "-f", help="Show system info and extras"),
    json_output: bool = typer.Option(False, "--json", help="Output in JSON format"),
):
    """Show Soup CLI version."""
    import json
    import platform

    if json_output:
        info = {
            "version": __version__,
            "python": platform.python_version(),
            "platform": platform.system().lower(),
        }

        if full:
            for lib in ["torch", "transformers", "peft", "trl", "datasets", "accelerate"]:
                try:
                    mod = __import__(lib)
                    if hasattr(mod, "__version__"):
                        info[lib] = mod.__version__
                except ImportError:
                    pass
            for name in ["fastapi", "vllm", "datasketch", "lm_eval", "deepspeed", "wandb"]:
                try:
                    mod = __import__(name)
                    if hasattr(mod, "__version__"):
                        info[name] = mod.__version__
                    elif hasattr(mod, "version"):
                        info[name] = mod.version
                    else:
                        info[name] = "installed"
                except ImportError:
                    pass

        console.print(json.dumps(info), highlight=False)
        return

    if not full:
        console.print(f"[bold green]soup[/] v{__version__}")
        console.print(f"[dim]{GITHUB_URL}[/]")
        return

    parts = [f"[bold green]soup[/] v{__version__}"]
    parts.append(f"Python {platform.python_version()}")

    # GPU info
    try:
        import torch
        if torch.cuda.is_available():
            parts.append(f"CUDA {torch.version.cuda}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            parts.append("MPS")
        else:
            parts.append("CPU only")
    except ImportError:
        parts.append("no torch")

    # Installed extras
    extras = []
    for name, label in [
        ("fastapi", "serve"),
        ("vllm", "serve-fast"),
        ("datasketch", "data"),
        ("lm_eval", "eval"),
        ("deepspeed", "deepspeed"),
        ("wandb", "wandb"),
    ]:
        try:
            __import__(name)
            extras.append(label)
        except ImportError:
            pass

    if extras:
        parts.append(f"extras: {', '.join(extras)}")

    console.print(" | ".join(parts))
    console.print(f"[dim]GitHub: [link={GITHUB_URL}]{GITHUB_URL}[/link][/]")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-V",
        help="Show full traceback on errors",
    ),
    log_level: str = typer.Option(
        "normal",
        "--log-level",
        help="Logging tier: quiet | normal | verbose | debug",
    ),
    no_audit_log: bool = typer.Option(
        False,
        "--no-audit-log",
        help=(
            "Disable the local HIPAA/SOC2 audit log for this invocation "
            "(also via SOUP_NO_AUDIT_LOG=1). Default: a one-line record per "
            "command under ~/.soup/audit.jsonl. v0.71.3."
        ),
    ),
):
    """Soup — fine-tune and post-train LLMs in one command."""
    global _verbose, _log_level, _audit_disabled
    _verbose = verbose
    _audit_disabled = no_audit_log
    from soup_cli.utils.log_level import (
        apply_logging_level,
        parse_log_level,
        setup_logging,
    )

    try:
        tier = parse_log_level(log_level)
    except ValueError as exc:
        console.print(f"[red]error:[/] {exc}")
        raise typer.Exit(code=2) from exc
    _log_level = tier.value
    setup_logging(tier)
    # v0.40.2 N1/G2: also push the tier into the root logger so third-party
    # libraries (transformers / peft / trl) respect QUIET / DEBUG. Without
    # this, all four levels were producing nearly-identical output.
    apply_logging_level(tier)


def _split_command_args(argv: list[str]) -> tuple[str, list[str]]:
    """Split ``argv`` into (command, remaining_args) for the audit record.

    Skips the program name (argv[0]) and global options. Global options that
    take a value (``--log-level``) consume their following token so the value
    is not mistaken for the subcommand. Returns ``("(root)", [...])`` when no
    subcommand is present.
    """
    tokens = list(argv[1:])
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in _GLOBAL_VALUE_OPTS:
            i += 2  # skip the option AND its value (even if the value is "-"-like)
            continue
        if isinstance(tok, str) and tok.startswith("-"):
            i += 1
            continue
        return tok, tokens[i + 1:]
    return "(root)", tokens


def _audit_env_opt_out() -> bool:
    val = (os.environ.get("SOUP_NO_AUDIT_LOG") or "").strip().lower()
    return val in {"1", "true", "yes", "on"}


def _emit_audit_event(argv: list[str], exit_code: int) -> None:
    """Append one HIPAA/SOC2 audit record for this command. Best-effort.

    v0.71.3 #183 — auto-instrumentation. Disabled by ``--no-audit-log`` /
    ``SOUP_NO_AUDIT_LOG``. Never raises: a broken audit log must never crash
    the CLI (mirrors v0.59.0 audit-log fail-soft policy).
    """
    if _audit_disabled or _audit_env_opt_out():
        return
    try:
        import getpass
        import platform
        from datetime import datetime, timezone

        from soup_cli.utils.audit_log import AuditEvent, append_audit_event

        command, args = _split_command_args(argv)
        command = (command or "(root)")[:64] or "(root)"
        try:
            operator = getpass.getuser() or "unknown"
        except (OSError, KeyError, ImportError):
            # getpass.getuser() can raise when no pwd / env user is resolvable.
            operator = "unknown"
        host = (platform.node() or "unknown")[:128] or "unknown"
        operator = (operator or "unknown")[:128] or "unknown"
        # Defensive re-cap, mirroring audit_log._MAX_ARGS (256) /
        # _MAX_ARG_LEN (1024) so AuditEvent.__post_init__ never rejects.
        capped_args = tuple(str(a)[:1024] for a in args[:256])
        code = exit_code if isinstance(exit_code, int) and not isinstance(
            exit_code, bool
        ) else 1
        ev = AuditEvent(
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            command=command,
            args=capped_args,
            exit_code=code,
            host_id=host,
            operator_id=operator,
        )
        append_audit_event(ev)
    except Exception:  # noqa: BLE001 — audit must never crash the CLI
        pass


def run():
    """Entry point with friendly error handling."""
    # v0.54.0 — rewrite `soup advise <data>` → `soup advise run <data>`.
    sys.argv = _rewrite_advise_argv(sys.argv)
    argv_snapshot = list(sys.argv)
    try:
        app()
    except SystemExit as exc:
        code = exc.code
        if code is None:
            resolved = 0
        elif isinstance(code, int) and not isinstance(code, bool):
            resolved = code
        else:
            resolved = 1
        _emit_audit_event(argv_snapshot, resolved)
        raise
    except typer.Exit as exc:
        # Defensive/unreachable under click standalone mode (typer.Exit is
        # converted to SystemExit inside app()); kept so a future click
        # behaviour change still audits exactly once.
        _emit_audit_event(argv_snapshot, getattr(exc, "exit_code", 1) or 0)
        raise
    except KeyboardInterrupt:
        _emit_audit_event(argv_snapshot, 130)
        console.print("\n[yellow]Interrupted.[/]")
        sys.exit(130)
    except Exception as exc:
        _emit_audit_event(argv_snapshot, 1)
        from soup_cli.utils.errors import format_friendly_error

        format_friendly_error(exc, verbose=_verbose)
        sys.exit(1)
    else:
        # Defensive/unreachable: app() raises SystemExit(0) on success under
        # click standalone mode, so this normal-return path rarely fires.
        _emit_audit_event(argv_snapshot, 0)


# When invoked via `soup` entry point, use run() for error handling.
# When invoked via `python -m soup_cli`, __main__.py calls run() directly.
if __name__ == "__main__":
    run()
