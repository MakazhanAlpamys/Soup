"""soup loop — CLI-first data flywheel (v0.58.0 capstone).

Subcommands:

    soup loop init <served-model> --eval <suite> --baseline <ref>
    soup loop status
    soup loop watch [--detach] [--max-iterations N]
    soup loop pause
    soup loop resume
    soup loop canary <new-adapter> --traffic 5% [--autoroll-on-regress]
    soup loop replay <iteration-id>
    soup loop rollback

State lives in ``.soup/loop.yaml``; per-iteration artifacts under
``.soup-loops/<iteration-id>/iteration.json``. Both paths are cwd-
contained + symlink-rejected (TOCTOU defence — matches v0.33.0 / v0.43.0
/ v0.55.0 / v0.56.0 / v0.57.0 policy).
"""

from __future__ import annotations

import atexit
import os
import subprocess
import sys
import time
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from soup_cli.utils.canary_router import CanaryPolicy
from soup_cli.utils.loop_budget import (
    parse_budget_string,
    reset_daily_counter_if_new_day,
    reset_monthly_spend_if_new_month,
)
from soup_cli.utils.loop_daemon import CANARY_MIN_SAMPLES, WatchAbortedError, WatchConfig, watch
from soup_cli.utils.loop_detach import (
    DETACHED_ENV,
    PidFileHeldError,
    claim_pid_file,
    describe_daemon,
    detached_popen_kwargs,
    display_path,
    log_file_path,
    open_log,
    pid_file_path,
    remove_pid_file_if_own,
    rotate_log_if_needed,
    rotate_open_log,
    rotated_log_path,
    write_pid_file,
)
from soup_cli.utils.loop_holdout import rollback_cursor_changes
from soup_cli.utils.loop_iteration import list_iterations, read_iteration
from soup_cli.utils.loop_log import (
    LoopLog,
    progress_line,
    rollback_changes,
    start_log,
    version_line,
    withdrawn_line,
)
from soup_cli.utils.loop_state import (
    ALL_LINEAR,
    CANARY_ON_TIMEOUT,
    SFT_RECIPE_KEYS,
    TRAIN_METHODS,
    VERIFY_METHODS,
    LoopState,
    format_wait_duration,
    init_state,
    parse_wait_duration,
    read_state,
    update_state,
    validate_sft_recipe_value,
    write_state,
)
from soup_cli.utils.terminal import for_terminal

app = typer.Typer(
    name="loop",
    help="Data flywheel: traces -> pairs -> train -> gate -> deploy.",
    no_args_is_help=True,
)
console = Console()


def _safe_read() -> LoopState:
    try:
        return read_state()
    except FileNotFoundError as exc:
        console.print(
            f"[red]No loop state found.[/] Run [bold]soup loop init[/] first."
            f"\n  detail: {for_terminal(str(exc))}"
        )
        raise typer.Exit(code=2)
    except (ValueError, TypeError) as exc:
        console.print(f"[red]loop state invalid:[/] {for_terminal(str(exc))}")
        raise typer.Exit(code=2)


def _update(mutate) -> LoopState:
    """``update_state`` for a command: its errors are one red line, never a traceback.

    A file locked for the whole wait (``TimeoutError`` names the lock file) exits 1; a value
    that makes the state invalid (the ``ValueError`` names the field) exits 2, as the
    commands' other refusals do.
    """
    try:
        return update_state(mutate)
    except TimeoutError as exc:
        console.print(f"[red]cannot update loop.yaml:[/] {for_terminal(exc)}")
        raise typer.Exit(code=1)
    except FileNotFoundError as exc:
        console.print(f"[red]cannot update loop.yaml:[/] {for_terminal(exc)}")
        raise typer.Exit(code=2)
    except (ValueError, TypeError) as exc:
        console.print(f"[red]invalid loop setting:[/] {for_terminal(exc)}; loop.yaml unchanged")
        raise typer.Exit(code=2)


#: What an sft loop trains on, said at init (the default method, so it is spelled out).
SFT_TRAINS_ON = (
    "sft trains on corrected answers and on answers rated good; an answer that was only "
    "rated bad gives no training row"
)


def _sft_value_label(key: str, value: object) -> str:
    """One sft recipe value as ``soup loop init`` and ``soup loop status`` print it."""
    if key == "target_modules":
        if value is None:
            return "auto (schema default)"
        return ", ".join(value) if isinstance(value, list) else str(value)
    if key == "lr":
        return f"{value:g}"
    if key == "quantization" and value == "auto":
        return "auto (none if the base fits in free VRAM, else 4bit)"
    return str(value)


def _sft_recipe_pairs(recipe: dict, keys: tuple) -> str:
    """``lora_r 16; lr 0.0002``: the named keys of ``recipe``, in the order of ``keys``."""
    return "; ".join(f"{key} {_sft_value_label(key, recipe[key])}" for key in keys)


def _sft_recipe_label(state: LoopState) -> str:
    """The effective sft recipe in one line, and which keys loop.yaml names."""
    own = tuple(key for key in SFT_RECIPE_KEYS if key in (state.sft_recipe or {}))
    text = _sft_recipe_pairs(state.effective_sft_recipe(), SFT_RECIPE_KEYS)
    if not own:
        return f"{text} (all built-in defaults)"
    if len(own) == len(SFT_RECIPE_KEYS):
        return f"{text} (all from loop.yaml)"
    return f"{text} (from loop.yaml: {', '.join(own)}; the rest built-in defaults)"


def _paraphraser_label(state: LoopState) -> str:
    """Who writes the paraphrases, in one line (only called with a paraphraser set)."""
    return f"{state.paraphraser_model} at {state.paraphraser_endpoint}"


def _paraphraser_line(state: LoopState) -> str:
    """The init panel line for a configured paraphraser (empty without), newline-ended."""
    if not state.paraphraser_endpoint:
        return ""
    return f"paraphraser:  [bold]{for_terminal(_paraphraser_label(state))}[/]\n"


def _paraphraser_from_flags(
    endpoint: Optional[str], model: Optional[str]
) -> "tuple[Optional[str], Optional[str]]":
    """The checked ``--paraphraser-endpoint`` / ``--paraphraser-model`` pair (exit 2 if bad).

    Both or neither: the loop's own serve endpoint ignores a model name, and another endpoint
    needs one. The endpoint goes through the same gate as SOUP_LOOP_SERVE_ENDPOINT.
    """
    if endpoint is None and model is None:
        return None, None
    if endpoint is None:
        console.print(
            "[red]--paraphraser-model needs --paraphraser-endpoint:[/] the loop's own serve "
            "endpoint answers with the model it serves and ignores a model name"
        )
        raise typer.Exit(code=2)
    if model is None:
        console.print(
            "[red]--paraphraser-endpoint needs --paraphraser-model:[/] the endpoint is asked "
            "for a model by name"
        )
        raise typer.Exit(code=2)
    model = model.strip()
    if not model or len(model) > 256 or any(ord(c) < 0x20 or ord(c) == 0x7F for c in model):
        console.print(
            "[red]invalid --paraphraser-model:[/] a model name of 1 to 256 characters, "
            "without control characters"
        )
        raise typer.Exit(code=2)
    from soup_cli.utils.loop_paraphrase import checked_endpoint

    try:
        checked = checked_endpoint(endpoint, "--paraphraser-endpoint")
    except ValueError as exc:
        console.print(f"[red]invalid --paraphraser-endpoint:[/] {for_terminal(exc)}")
        raise typer.Exit(code=2)
    return checked, model


def _train_method_note(state: LoopState) -> str:
    """The init panel lines under ``train_method`` (empty for dpo and kto), newline-ended."""
    if state.train_method != "sft":
        return ""
    note = (
        f"{SFT_TRAINS_ON}\n"
        f"with a feedback log, training starts at {state.min_new_edits} new edits "
        f"or {state.min_new_rated} new rated answers\n"
    )
    if state.sft_recipe:
        own = tuple(key for key in SFT_RECIPE_KEYS if key in state.sft_recipe)
        note += (
            "sft recipe from loop.yaml: "
            f"{for_terminal(_sft_recipe_pairs(state.effective_sft_recipe(), own))}\n"
        )
    return note


def _parse_target_modules(raw: str) -> object:
    """``--sft-target-modules``: ``all-linear``, ``auto`` (None: the schema default) or a list."""
    text = raw.strip()
    if text.lower() == ALL_LINEAR:
        return ALL_LINEAR
    if text.lower() == "auto":
        return None
    names = [name.strip() for name in text.split(",")]
    if not all(names):
        raise ValueError(
            "target_modules has an empty entry: use all-linear, auto, or a comma-separated "
            "list such as q_proj,v_proj"
        )
    return names


def _sft_recipe_from_flags(method: str, flags: tuple) -> Optional[dict]:
    """The ``sft_recipe`` the ``--sft-*`` flags name (None when none was given).

    ``flags`` is ``(flag, recipe key, value)`` triples; a value of None was not given.
    Exits 2, naming the flag, for a value out of range, and for any flag at all when the
    loop does not train sft (it would change nothing).
    """
    given = [(flag, key, value) for flag, key, value in flags if value is not None]
    if not given:
        return None
    if method != "sft":
        names = ", ".join(flag for flag, _, _ in given)
        console.print(
            f"[red]{names} only apply to --train-method sft[/] (this loop trains {method}, "
            "so they would change nothing)"
        )
        raise typer.Exit(code=2)
    recipe: dict = {}
    for flag, key, value in given:
        try:
            if key == "target_modules":
                value = _parse_target_modules(value)
            elif key == "quantization":
                value = value.strip().lower()
            validate_sft_recipe_value(key, value)
        except ValueError as exc:
            console.print(f"[red]invalid {flag}:[/] {for_terminal(exc)}")
            raise typer.Exit(code=2)
        recipe[key] = value
    return recipe


def _warn_log(log: LoopLog, *, hint: str = "") -> None:
    """Say which log lines could not be written (the command went on without them)."""
    for note in log.take_failures():
        console.print(f"[yellow]warning:[/] {for_terminal(note)}{hint}")


def _log_withdrawal(log: LoopLog, state: LoopState) -> None:
    """``soup loop rollback`` withdrew the running canary: close its iteration, say so."""
    if state.canary_active is None:
        return
    line = withdrawn_line(state, "withdrawn by soup loop rollback")
    if line is not None:
        log.iteration(line)
    iteration_id = state.canary_iteration["iteration_id"] if state.canary_iteration else None
    log.progress(progress_line(
        iteration_id, "rollback", "end", message=f"canary {state.canary_active} withdrawn"
    ))


@app.command("init")
def init_cmd(
    served_model: str = typer.Argument(..., help="Served model id (e.g. registry://abc12)."),
    eval_suite: str = typer.Option(..., "--eval", help="Eval suite path or registry ref."),
    baseline: str = typer.Option(..., "--baseline", help="Baseline registry id or file."),
    monthly_budget: Optional[str] = typer.Option(
        None, "--monthly-budget", help="Monthly USD cap (e.g. 50usd, 100)."
    ),
    max_runs_per_day: Optional[int] = typer.Option(
        None, "--max-runs-per-day", help="Cap on iteration starts per UTC day."
    ),
    pre_wired: bool = typer.Option(
        False, "--pre-wired",
        help=(
            "Use the pre-wired harvest/train/gate/deploy stages "
            "(utils/loop_stages.py) on watch instead of no-op stubs (v0.71.4)."
        ),
    ),
    thumbs_db: Optional[str] = typer.Option(
        None, "--thumbs-db",
        help=(
            "Local-RL SQLite written by `soup serve --record-thumbs`; the "
            "pre-wired harvest turns its thumbs into preference pairs. Must "
            "stay under cwd. SOUP_LOOP_THUMBS_DB overrides it."
        ),
    ),
    base_model: Optional[str] = typer.Option(
        None, "--base-model",
        help=(
            "Model the pre-wired train stage starts from. Default: the registry "
            "entry's base model for a registry://<id> served model, the "
            "base_model_name_or_path of a LoRA adapter directory, otherwise the "
            "served model itself."
        ),
    ),
    trace_dir: Optional[str] = typer.Option(
        None, "--trace-dir",
        help=(
            "Directory of conversation logs (soup serve --trace-log files, *.jsonl) "
            "the pre-wired harvest reads. Default: .soup-loops/traces. "
            "SOUP_LOOP_TRACE_DIR overrides it."
        ),
    ),
    verify: str = typer.Option(
        "ship", "--verify",
        help=(
            "The check before rollout: ship (a soup ship verdict against the "
            "production version; --eval is a task-eval JSONL) or eval-gate "
            "(--eval is an eval-gate suite YAML)."
        ),
    ),
    canary_traffic: str = typer.Option(
        "5%", "--canary-traffic",
        help=(
            "Share of keyed traffic a newly deployed adapter gets while its canary "
            'runs, e.g. "5%" (the default) or "10".'
        ),
    ),
    canary_max_wait: str = typer.Option(
        "24h", "--canary-max-wait",
        help=(
            "How long after a canary started its ratings may stay short of the 30 per side "
            "the quality verdict needs: <int>m, <int>h or <int>d, or plain seconds (at most "
            "30d, default 24h). Then --canary-on-timeout applies."
        ),
    ),
    canary_on_timeout: Optional[str] = typer.Option(
        None, "--canary-on-timeout",
        help=(
            "What happens when --canary-max-wait runs out: errors-only (the request-error "
            "verdict alone decides; the default), rollback (withdraw the canary) or wait "
            "(keep waiting, no time limit). A canary with no canary answers at all is "
            "withdrawn (timeout_no_answers) unless errors-only is given here: then it "
            "promotes with no evidence."
        ),
    ),
    min_new_pairs: int = typer.Option(
        1, "--min-new-pairs",
        help=(
            "Train only when at least this many preference pairs arrived since the last "
            "training run (the pre-wired harvest keeps a data cursor); the run still "
            "trains on all pairs. Default 1. An sft loop with a feedback log uses "
            "--min-new-edits and --min-new-rated instead."
        ),
    ),
    min_new_edits: int = typer.Option(
        8, "--min-new-edits",
        help=(
            "An sft loop with a feedback log trains once at least this many new edits "
            "(corrected answers newer than the last training run) have arrived, or "
            "--min-new-rated new rated answers, whichever comes first. Default 8."
        ),
    ),
    min_new_rated: int = typer.Option(
        30, "--min-new-rated",
        help=(
            "An sft loop with a feedback log trains once at least this many new rated "
            "answers (any thumb, rating or edit) have arrived, or --min-new-edits new "
            "edits, whichever comes first. Default 30."
        ),
    ),
    feedback_log: Optional[str] = typer.Option(
        None, "--feedback-log",
        help=(
            "Feedback log (JSONL: thumbs, ratings, edits and comments on logged "
            "answers) the pre-wired harvest reads. Default: .soup-loops/feedback.jsonl. "
            "SOUP_LOOP_FEEDBACK_LOG overrides it."
        ),
    ),
    train_method: str = typer.Option(
        "sft", "--train-method",
        help=(
            "What the pre-wired loop trains: sft (the default; corrected answers and "
            "answers rated good), dpo (preference pairs) or kto (answers labelled good "
            "or bad). A loop.yaml without this field keeps dpo."
        ),
    ),
    sft_target_modules: Optional[str] = typer.Option(
        None, "--sft-target-modules",
        help=(
            "sft only: the LoRA target modules of the training config: all-linear, auto "
            "(the config schema's default) or a comma-separated list such as "
            "q_proj,v_proj. Default: the loop's built-in sft recipe (soup loop status)."
        ),
    ),
    sft_lora_r: Optional[int] = typer.Option(
        None, "--sft-lora-r",
        help="sft only: LoRA rank of the training config, 1 to 1024. Default: the built-in recipe.",
    ),
    sft_lora_alpha: Optional[int] = typer.Option(
        None, "--sft-lora-alpha",
        help=(
            "sft only: LoRA alpha of the training config, 1 to 4096. Default: the built-in "
            "recipe."
        ),
    ),
    sft_lr: Optional[float] = typer.Option(
        None, "--sft-lr",
        help=(
            "sft only: learning rate of the training config, above 0 and at most 1 (for "
            "example 2e-4). Default: the built-in recipe."
        ),
    ),
    sft_epochs: Optional[int] = typer.Option(
        None, "--sft-epochs",
        help="sft only: epochs of the training config, 1 to 100. Default: the built-in recipe.",
    ),
    sft_batch_size: Optional[int] = typer.Option(
        None, "--sft-batch-size",
        help=(
            "sft only: per-device micro-batch of the training config, 1 to 1024. Default: "
            "the built-in recipe."
        ),
    ),
    sft_grad_accum: Optional[int] = typer.Option(
        None, "--sft-grad-accum",
        help=(
            "sft only: gradient accumulation steps of the training config, 1 to 1024 "
            "(micro-batch x accumulation = the batch per optimizer step). Default: the "
            "built-in recipe."
        ),
    ),
    sft_seed: Optional[int] = typer.Option(
        None, "--sft-seed",
        help=(
            "sft only: training seed of the training config, 0 to 4294967295. Default: "
            "the built-in recipe."
        ),
    ),
    sft_quantization: Optional[str] = typer.Option(
        None, "--sft-quantization",
        help=(
            "sft only: quantization of the training config: 4bit, none or auto (none if the "
            "base model fits in free VRAM when training starts, else 4bit). Default: the "
            "built-in recipe."
        ),
    ),
    paraphraser_endpoint: Optional[str] = typer.Option(
        None, "--paraphraser-endpoint",
        help=(
            "OpenAI-compatible base URL (this machine or LAN) of the model that writes "
            "paraphrases of the prompts of corrected answers, which the loop's verdict asks the "
            "new version instead of the original prompts. Needs --paraphraser-model. Default: "
            "the loop's own serve endpoint (SOUP_LOOP_SERVE_ENDPOINT) with the tool token."
        ),
    ),
    paraphraser_model: Optional[str] = typer.Option(
        None, "--paraphraser-model",
        help=(
            "Model name sent to --paraphraser-endpoint, recorded as the generator of the "
            "paraphrases. A small served model paraphrases poorly: name a better one here."
        ),
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite existing loop.yaml."),
) -> None:
    """Create the .soup/loop.yaml control file (one-time setup)."""
    budget_usd: Optional[float] = None
    if monthly_budget is not None:
        try:
            budget_usd = parse_budget_string(monthly_budget)
        except (TypeError, ValueError) as exc:
            console.print(f"[red]invalid --monthly-budget:[/] {for_terminal(str(exc))}")
            raise typer.Exit(code=2)
    if max_runs_per_day is not None and (
        isinstance(max_runs_per_day, bool) or max_runs_per_day < 1
    ):
        console.print("[red]--max-runs-per-day must be a positive int[/]")
        raise typer.Exit(code=2)
    for flag, value in (
        ("--min-new-pairs", min_new_pairs),
        ("--min-new-edits", min_new_edits),
        ("--min-new-rated", min_new_rated),
    ):
        if isinstance(value, bool) or value < 1:
            console.print(f"[red]{flag} must be a positive int[/]")
            raise typer.Exit(code=2)
    if thumbs_db is not None:
        from soup_cli.utils.local_rl import validate_db_path

        try:
            validate_db_path(thumbs_db)
        except (TypeError, ValueError) as exc:
            console.print(f"[red]invalid --thumbs-db:[/] {for_terminal(exc)}")
            raise typer.Exit(code=2)
    deploy_pct = _parse_traffic(canary_traffic, flag="--canary-traffic")
    if deploy_pct <= 0.0:
        console.print("[red]--canary-traffic must be above 0%[/]")
        raise typer.Exit(code=2)
    max_wait_sec = _parse_wait(canary_max_wait, flag="--canary-max-wait")
    timeout_policy = (
        "errors-only" if canary_on_timeout is None else canary_on_timeout
    ).strip().lower().replace("_", "-")
    if timeout_policy not in CANARY_ON_TIMEOUT:
        console.print(
            f"[red]--canary-on-timeout must be one of {', '.join(CANARY_ON_TIMEOUT)}, "
            f"got {for_terminal(repr(canary_on_timeout))}[/]"
        )
        raise typer.Exit(code=2)
    if base_model is not None and not base_model.strip():
        console.print("[red]--base-model must not be empty[/]")
        raise typer.Exit(code=2)
    if trace_dir is not None:
        if not trace_dir.strip():
            console.print("[red]--trace-dir must not be empty[/]")
            raise typer.Exit(code=2)
        if os.path.exists(trace_dir) and not os.path.isdir(trace_dir):
            console.print(f"[red]--trace-dir:[/] {for_terminal(trace_dir)} is not a directory")
            raise typer.Exit(code=2)
        if not os.path.isdir(trace_dir):
            console.print(
                f"[yellow]warning:[/] no conversation logs at {for_terminal(trace_dir)} yet; "
                "soup serve --trace-log <dir>/<file>.jsonl writes them"
            )
    method = train_method.strip().lower()
    if method not in TRAIN_METHODS:
        console.print(
            f"[red]--train-method must be one of {', '.join(TRAIN_METHODS)}, "
            f"got {for_terminal(repr(train_method))}[/]"
        )
        raise typer.Exit(code=2)
    sft_recipe = _sft_recipe_from_flags(
        method,
        (
            ("--sft-target-modules", "target_modules", sft_target_modules),
            ("--sft-lora-r", "lora_r", sft_lora_r),
            ("--sft-lora-alpha", "lora_alpha", sft_lora_alpha),
            ("--sft-lr", "lr", sft_lr),
            ("--sft-epochs", "epochs", sft_epochs),
            ("--sft-batch-size", "batch_size", sft_batch_size),
            ("--sft-grad-accum", "gradient_accumulation_steps", sft_grad_accum),
            ("--sft-seed", "seed", sft_seed),
            ("--sft-quantization", "quantization", sft_quantization),
        ),
    )
    if feedback_log is not None:
        if not feedback_log.strip():
            console.print("[red]--feedback-log must not be empty[/]")
            raise typer.Exit(code=2)
        if os.path.exists(feedback_log) and not os.path.isfile(feedback_log):
            what = "not a regular file"
            if os.path.isdir(feedback_log):
                what = "a directory, not a file"
            console.print(f"[red]--feedback-log:[/] {for_terminal(feedback_log)} is {what}")
            raise typer.Exit(code=2)
        if not os.path.exists(feedback_log):
            console.print(
                f"[yellow]warning:[/] no feedback log at {for_terminal(feedback_log)} yet; "
                "the loop reads it once feedback lines are written there"
            )
    paraphraser_endpoint, paraphraser_model = _paraphraser_from_flags(
        paraphraser_endpoint, paraphraser_model
    )
    from soup_cli.utils.loop_stages import check_verify_eval, resolve_base_model_source

    verify_method = verify.strip().lower().replace("-", "_")
    try:
        check_verify_eval(verify_method, eval_suite)
    except ValueError as exc:
        if pre_wired or verify_method not in VERIFY_METHODS:
            console.print(f"[red]invalid --eval / --verify:[/] {for_terminal(exc)}")
            raise typer.Exit(code=2)
        console.print(
            f"[yellow]warning:[/] {for_terminal(exc)}; "
            "soup loop watch --pre-wired refuses until it is fixed"
        )

    resolved_base: Optional[str] = None
    base_source = ""
    try:
        resolved_base, base_source = resolve_base_model_source(served_model, base_model)
    except ValueError as exc:
        if pre_wired:
            console.print(f"[red]cannot determine the base model:[/] {for_terminal(exc)}")
            raise typer.Exit(code=2)
        console.print(
            f"[yellow]warning:[/] base model unknown ({for_terminal(exc)}); "
            "soup loop watch --pre-wired refuses until it is known"
        )
    try:
        state, path = init_state(
            served_model=served_model,
            eval_suite=eval_suite,
            baseline=baseline,
            monthly_budget_usd=budget_usd,
            max_runs_per_day=max_runs_per_day,
            pre_wired=pre_wired,
            thumbs_db=thumbs_db,
            base_model=resolved_base,
            trace_dir=trace_dir,
            verify=verify_method,
            deploy_traffic_pct=deploy_pct,
            min_new_pairs=min_new_pairs,
            min_new_edits=min_new_edits,
            min_new_rated=min_new_rated,
            feedback_log=feedback_log,
            train_method=method,
            sft_recipe=sft_recipe,
            paraphraser_endpoint=paraphraser_endpoint,
            paraphraser_model=paraphraser_model,
            canary_max_wait_sec=max_wait_sec,
            canary_on_timeout=timeout_policy,
            canary_on_timeout_explicit=canary_on_timeout is not None,
            production_adapter=(
                served_model
                if os.path.isfile(os.path.join(served_model, "adapter_config.json"))
                else None
            ),
            force=force,
        )
    except (FileExistsError, FileNotFoundError, TypeError, ValueError) as exc:
        console.print(f"[red]init failed:[/] {for_terminal(str(exc))}")
        raise typer.Exit(code=2)
    if pre_wired:
        # The first version of the loop's version log (docs/loop-formats.md). A loop that is
        # not pre-wired gets it on its first watch cycle, as does one whose lines failed here.
        log = LoopLog()
        started = start_log(state, log, actor="user")
        if started is not state:
            write_state(started, path)
        _warn_log(log, hint="; soup loop watch tries again")
    console.print(
        Panel.fit(
            f"loop state created at [bold]{for_terminal(os.path.relpath(path))}[/]\n"
            f"served_model: [bold]{for_terminal(state.served_model)}[/]\n"
            f"eval_suite:   [bold]{for_terminal(state.eval_suite)}[/]"
            f" (verify: {for_terminal(state.verify or 'eval_gate')})\n"
            f"baseline:     [bold]{for_terminal(state.baseline)}[/]\n"
            f"trace_dir:    [bold]{for_terminal(_trace_dir_label(state))}[/]\n"
            f"feedback_log: [bold]{for_terminal(_feedback_log_label(state))}[/]\n"
            f"train_method: [bold]{for_terminal(state.train_method)}[/]\n"
            f"{_train_method_note(state)}"
            f"{_paraphraser_line(state)}"
            f"canary_max_wait: [bold]{format_wait_duration(state.canary_max_wait_sec)}[/]\n"
            f"canary_on_timeout: [bold]{for_terminal(state.canary_on_timeout)}[/]\n"
            f"base_model:   [bold]{for_terminal(state.base_model or 'unknown')}[/]"
            + (f" (from {for_terminal(base_source)})" if base_source else ""),
            title="soup loop init",
        )
    )


@app.command("status")
def status_cmd() -> None:
    """Show counters and current status."""
    state = _safe_read()
    # Show the spend and run count the next iteration will see (#2390): run_once resets both
    # from the saved last_run_date, so between iterations the stored values can be a past
    # month's or day's. One clock read for both, as in run_once; loop.yaml is not written.
    now = datetime.now(timezone.utc)
    try:
        spent = reset_monthly_spend_if_new_month(
            state.spent_this_month_usd, state.last_run_date, now=now
        )
        runs_today, _ = reset_daily_counter_if_new_day(
            state.runs_today, state.last_run_date, now=now
        )
    except ValueError as exc:
        console.print(f"[red]loop state invalid:[/] {for_terminal(str(exc))}")
        raise typer.Exit(code=2)
    table = Table(title="soup loop status", show_header=False)
    table.add_column("field", style="bold")
    table.add_column("value")
    table.add_row("status", f"[bold]{for_terminal(state.status)}[/]")
    # The PID file is read, never written or removed; a stale one is only reported.
    table.add_row("daemon", for_terminal(describe_daemon()))
    table.add_row("served_model", for_terminal(state.served_model))
    table.add_row("eval_suite", for_terminal(state.eval_suite))
    table.add_row("baseline", for_terminal(state.baseline))
    table.add_row("base_model", for_terminal(state.base_model or "unknown"))
    table.add_row("trace_dir", for_terminal(_trace_dir_label(state)))
    table.add_row("feedback_log", for_terminal(_feedback_log_label(state)))
    table.add_row("train_method", for_terminal(state.train_method))
    if state.train_method == "sft":
        table.add_row("sft_recipe", for_terminal(_sft_recipe_label(state)))
    if state.paraphraser_endpoint:
        table.add_row("paraphraser", for_terminal(_paraphraser_label(state)))
    table.add_row("verify", for_terminal(state.verify or "eval_gate"))
    table.add_row(
        "production",
        for_terminal(
            f"{state.production_name or 'served model'} "
            f"({state.production_adapter or 'no adapter'})"
        ),
    )
    table.add_row("canary_traffic", f"{state.deploy_traffic_pct:g}% per new canary")
    table.add_row("canary_max_wait", format_wait_duration(state.canary_max_wait_sec))
    table.add_row("canary_on_timeout", for_terminal(state.canary_on_timeout))
    table.add_row("new_data_since", _cursor_label(state))
    if state.waiting_note:
        # The last cycle that waited for corrections said this (not a live count).
        table.add_row("waiting", for_terminal(state.waiting_note))
    if state.last_failure:
        # The exit code and stderr tail of a failed training run, or why a deploy was refused.
        table.add_row("last_failure", for_terminal(state.last_failure))
    if state.pending_deploy:
        table.add_row(
            "pending_deploy",
            f"attempt {state.pending_deploy.get('attempts', 0)} refused; retried next cycle",
        )
    table.add_row("min_new_pairs", str(state.min_new_pairs))
    table.add_row("min_new_edits", str(state.min_new_edits))
    table.add_row("min_new_rated", str(state.min_new_rated))
    table.add_row("pre_wired", "yes" if state.pre_wired else "no")
    table.add_row("traces_collected", str(state.traces_collected))
    table.add_row("pairs_distilled", str(state.pairs_distilled))
    table.add_row("runs_gated", str(state.runs_gated))
    table.add_row("adapters_shipped", str(state.adapters_shipped))
    table.add_row("iteration_count", str(state.iteration_count))
    if state.canary_active:
        table.add_row(
            "canary",
            f"{for_terminal(state.canary_active)} @ "
            f"{state.canary_traffic_pct or 0:.1f}%",
        )
    if state.monthly_budget_usd is not None:
        table.add_row(
            "budget",
            f"${spent:.2f} / ${state.monthly_budget_usd:.2f}",
        )
    if state.max_runs_per_day is not None:
        table.add_row(
            "runs_today",
            f"{runs_today} / {state.max_runs_per_day}",
        )
    console.print(table)


@app.command("pause")
def pause_cmd() -> None:
    """Pause the watch daemon at the next iteration boundary."""
    state = _safe_read()
    if state.status == "stopped":
        console.print("[yellow]loop is already stopped[/]")
        raise typer.Exit(code=0)
    _update(lambda fresh: fresh.with_status("paused"))
    console.print("[green]loop paused[/]")


@app.command("resume")
def resume_cmd() -> None:
    """Resume a paused loop (next watch iteration picks up automatically)."""
    state = _safe_read()
    if state.status != "paused":
        console.print(f"[yellow]loop is {state.status}, not paused[/]")
        raise typer.Exit(code=0)
    _update(
        lambda fresh: fresh.with_status("running") if fresh.status == "paused" else fresh
    )
    console.print("[green]loop resumed[/]")


#: The settings ``soup loop set`` changes (what ``soup loop init`` needed ``--force`` for).
SETTABLE_KEYS = ("base_model", "trace_dir", "verify")


@app.command("set")
def set_cmd(
    key: str = typer.Argument(..., help="base_model, trace_dir or verify."),
    value: str = typer.Argument(..., help="The new value."),
) -> None:
    """Change ONE setting of an existing loop: base_model, trace_dir or verify.

    ``soup loop init`` on an existing loop needs ``--force`` and starts the state over
    (production, canary, cursors, counters, versions). This changes only the one key, under the
    state lock, and leaves everything else and the version log as they are.
    """
    if key not in SETTABLE_KEYS:
        console.print(
            f"[red]cannot set {for_terminal(repr(key))}:[/] soup loop set changes "
            f"{', '.join(SETTABLE_KEYS)}"
        )
        raise typer.Exit(code=2)
    state = _safe_read()
    if key == "base_model":
        if not value.strip():
            console.print("[red]base_model must not be empty[/]")
            raise typer.Exit(code=2)
        from soup_cli.utils.loop_stages import resolve_base_model_source

        try:
            new_value, _source = resolve_base_model_source(state.served_model, value)
        except ValueError as exc:
            console.print(f"[red]invalid base_model:[/] {for_terminal(exc)}")
            raise typer.Exit(code=2)
    elif key == "trace_dir":
        if not value.strip():
            console.print("[red]trace_dir must not be empty[/]")
            raise typer.Exit(code=2)
        if os.path.exists(value) and not os.path.isdir(value):
            console.print(f"[red]trace_dir:[/] {for_terminal(value)} is not a directory")
            raise typer.Exit(code=2)
        if not os.path.isdir(value):
            console.print(
                f"[yellow]warning:[/] no conversation logs at {for_terminal(value)} yet; "
                "soup serve --trace-log <dir>/<file>.jsonl writes them"
            )
        new_value = value
    else:
        from soup_cli.utils.loop_stages import check_verify_eval

        new_value = value.strip().lower().replace("-", "_")
        if new_value not in VERIFY_METHODS:
            console.print(
                f"[red]invalid verify {for_terminal(repr(value))}:[/] one of "
                f"{', '.join(VERIFY_METHODS)}"
            )
            raise typer.Exit(code=2)
        try:
            check_verify_eval(new_value, state.eval_suite)
        except ValueError as exc:
            if state.pre_wired:
                console.print(f"[red]invalid verify / eval:[/] {for_terminal(exc)}")
                raise typer.Exit(code=2)
            console.print(f"[yellow]warning:[/] {for_terminal(exc)}")
    _update(lambda fresh: replace(fresh, **{key: new_value}))
    console.print(f"[green]{key} set:[/] {for_terminal(new_value)}")


@app.command("watch")
def watch_cmd(
    foreground: bool = typer.Option(
        False,
        "--foreground",
        help="Run in foreground (default).",
    ),
    detach: bool = typer.Option(
        False,
        "--detach",
        help=(
            "Spawn a background subprocess running --foreground. Its pid goes to "
            ".soup/loop.pid and its output to .soup/loop.log (appended; rotated at 10 MB, "
            "one old file kept as .soup/loop.log.1); refused while that pid is alive."
        ),
    ),
    max_iterations: Optional[int] = typer.Option(
        None,
        "--max-iterations",
        help="Stop after N iterations (test/demo use).",
    ),
    poll_interval: float = typer.Option(
        60.0, "--poll-interval", help="Seconds between iterations [1, 3600]."
    ),
    pre_wired: bool = typer.Option(
        False, "--pre-wired",
        help="Force the pre-wired production stages even if loop.yaml didn't set it.",
    ),
    pack_cans: bool = typer.Option(
        False, "--pack-cans",
        help="Pack each iteration as a v0.26 Soup Can + Registry entry (v0.71.4).",
    ),
    tool_auth_token: Optional[str] = typer.Option(
        None,
        "--tool-auth-token",
        envvar="SOUP_TOOL_AUTH_TOKEN",
        help=(
            "The soup serve --tool-auth-token value, sent as a Bearer token when the "
            "deploy stage activates an adapter. Defaults to $SOUP_TOOL_AUTH_TOKEN, "
            "which keeps it out of the shell history."
        ),
    ),
) -> None:
    """Run the harvest → train → gate → deploy daemon."""
    if detach and foreground:
        console.print("[red]--detach and --foreground are mutually exclusive[/]")
        raise typer.Exit(code=2)
    detached_child = os.environ.get(DETACHED_ENV) == "1" and not detach
    if detached_child:
        # The child of `--detach`: remove the PID file on the way out (any exit path),
        # but only if it still names this process. A plain foreground watch never gets
        # here, so it never touches the file.
        atexit.register(remove_pid_file_if_own, os.getpid(), pid_file_path())
    state = _safe_read()  # ensure state exists before forking
    if tool_auth_token:
        # The deploy stage reads the token from the environment, as it reads
        # SOUP_LOOP_SERVE_ENDPOINT; a --detach child inherits it from there
        # rather than from argv, where any local user could read it (#1139).
        from soup_cli.utils.loop_stages import TOOL_AUTH_TOKEN_ENV

        os.environ[TOOL_AUTH_TOKEN_ENV] = tool_auth_token
    if pre_wired or state.pre_wired:
        # #1440: a configured thumbs database the harvest would refuse stops the
        # watch here, before it starts or detaches, instead of mid-iteration.
        from soup_cli.utils.loop_stages import (
            check_verify_eval,
            checked_thumbs_db,
            feedback_log_notice,
            resolve_base_model,
            trace_dir_notice,
            verify_method,
        )

        try:
            checked_thumbs_db(state)
        except ValueError as exc:
            console.print(f"[red]invalid thumbs database:[/] {for_terminal(exc)}")
            raise typer.Exit(code=2)
        try:
            notice = trace_dir_notice(state)
        except ValueError as exc:
            console.print(f"[red]invalid trace directory:[/] {for_terminal(exc)}")
            raise typer.Exit(code=2)
        if notice:
            console.print(f"[yellow]warning:[/] {for_terminal(notice)}")
        try:
            notice = feedback_log_notice(state)
        except ValueError as exc:
            console.print(f"[red]invalid feedback log:[/] {for_terminal(exc)}")
            raise typer.Exit(code=2)
        if notice:
            console.print(f"[yellow]warning:[/] {for_terminal(notice)}")
        # A check that cannot run skips every iteration's rollout: refuse now.
        try:
            check_verify_eval(verify_method(state), state.eval_suite)
        except ValueError as exc:
            console.print(f"[red]invalid --eval / --verify:[/] {for_terminal(exc)}")
            raise typer.Exit(code=2)
        # The train stage would skip every iteration; say so before starting.
        try:
            resolve_base_model(state.served_model, state.base_model)
        except ValueError as exc:
            console.print(f"[red]cannot determine the base model:[/] {for_terminal(exc)}")
            raise typer.Exit(code=2)
    elif os.environ.get("SOUP_LOOP_THUMBS_DB") or state.thumbs_db:
        console.print(
            "[yellow]warning:[/] a thumbs database is configured but the stub stages "
            "never read it; run `soup loop watch --pre-wired` to harvest it"
        )
    if detach:
        # #1808: the child validates these in WatchConfig.__post_init__ and exits
        # 2 with its output on /dev/null, while the parent printed "watch
        # detached". Run the same checks here, before anything is spawned.
        try:
            WatchConfig(poll_interval_sec=poll_interval, max_iterations=max_iterations)
        except (TypeError, ValueError) as exc:
            console.print(f"[red]invalid watch config:[/] {for_terminal(str(exc))}")
            raise typer.Exit(code=2)
        argv = [
            sys.executable,
            "-m",
            "soup_cli.cli",
            "loop",
            "watch",
            "--foreground",
            "--poll-interval",
            str(poll_interval),
        ]
        if max_iterations is not None:
            argv.extend(["--max-iterations", str(max_iterations)])
        if pre_wired:
            argv.append("--pre-wired")
        if pack_cans:
            argv.append("--pack-cans")
        # One daemon per loop.yaml. The PID file is CLAIMED (created exclusively, holding this
        # process's pid) before anything is spawned: two parallel --detach cannot both see
        # "no daemon". A live holder refuses the spawn; a stale file is replaced.
        pid_path = pid_file_path()
        try:
            replaced = claim_pid_file(pid_path)
        except PidFileHeldError as exc:
            console.print(
                f"[red]a loop daemon is already running:[/] pid {exc.pid}, "
                f"PID file {for_terminal(display_path(pid_path))} "
                "(delete the file if that pid is not a loop daemon)"
            )
            raise typer.Exit(code=2)
        except ValueError as exc:
            console.print(f"[red]cannot detach:[/] {for_terminal(exc)}")
            raise typer.Exit(code=2)
        if replaced is not None:
            console.print(
                f"[yellow]notice:[/] replacing stale PID file "
                f"{for_terminal(display_path(pid_path))} ({for_terminal(replaced.reason)})"
            )
        log_path = log_file_path()
        try:
            try:
                # A log that reached the limit moves to loop.log.1 first, so the child starts
                # on a fresh one; the running child rotates it from then on (rotate_open_log).
                if rotate_log_if_needed(log_path):
                    console.print(
                        f"[yellow]notice:[/] log rotated: "
                        f"{for_terminal(display_path(log_path))} reached its size limit, "
                        f"moved to {for_terminal(display_path(rotated_log_path(log_path)))}"
                    )
                log_handle = open_log(log_path)
            except (ValueError, OSError) as exc:
                console.print(f"[red]cannot detach:[/] {for_terminal(exc)}")
                raise typer.Exit(code=2)
            try:
                proc = subprocess.Popen(  # noqa: S603 — argv is internal, no shell
                    argv,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    close_fds=True,
                    env={**os.environ, DETACHED_ENV: "1"},
                    **detached_popen_kwargs(),
                )
            finally:
                # Popen hands the child its own duplicate; the parent keeps none.
                log_handle.close()
        except BaseException:
            # Nothing was spawned: give the claim back (only if it still holds our pid).
            remove_pid_file_if_own(os.getpid(), pid_path)
            raise
        try:
            write_pid_file(proc.pid, pid_path)
        except (ValueError, OSError) as exc:
            console.print(
                f"[red]watch detached pid={proc.pid} but its PID file could not be "
                f"written:[/] {for_terminal(exc)}"
            )
            raise typer.Exit(code=1)
        console.print(f"[green]watch detached[/] pid={proc.pid}")
        console.print(
            f"  PID file {for_terminal(display_path(pid_path))}, "
            f"log {for_terminal(display_path(log_path))}"
        )
        return
    use_prewired = pre_wired or state.pre_wired
    base_model = state.base_model or (
        state.served_model
        if state.served_model and not state.served_model.startswith("registry://")
        else "unknown"
    )
    try:
        if use_prewired:
            from soup_cli.utils.loop_stages import build_prewired_watch_config

            # Checked before the fork above, so it resolves here.
            base_model = resolve_base_model(state.served_model, state.base_model)
            cfg = build_prewired_watch_config(
                poll_interval_sec=float(poll_interval),
                max_iterations=max_iterations,
                pack_iterations=pack_cans,
                served_model=state.served_model,
                base_model=base_model,
            )
        else:
            cfg = WatchConfig(
                poll_interval_sec=float(poll_interval),
                max_iterations=max_iterations,
                pack_iterations=pack_cans,
                served_model=state.served_model,
                base_model=base_model,
            )
    except (TypeError, ValueError) as exc:
        console.print(f"[red]invalid watch config:[/] {for_terminal(str(exc))}")
        raise typer.Exit(code=2)
    if detached_child:
        # Only the --detach child writes to .soup/loop.log; a foreground watch may be
        # writing to a terminal and rotates nothing.
        cfg = replace(cfg, on_cycle=rotate_open_log)
    try:
        final_state, ran = watch(cfg)
    except WatchAbortedError as exc:
        console.print(
            f"[red]watch stopped after {exc.iterations} iteration(s):[/] {for_terminal(exc)}"
        )
        raise typer.Exit(code=1)
    console.print(
        f"[green]watch exited[/] iterations={ran} status={for_terminal(final_state.status)}"
    )


@app.command("canary")
def canary_cmd(
    new_adapter: str = typer.Argument(..., help="Adapter id/path to canary."),
    traffic: str = typer.Option(
        "5%", "--traffic", help='Traffic share, e.g. "5%" or "5".'
    ),
    autoroll_on_regress: bool = typer.Option(
        True,
        "--autoroll-on-regress/--no-autoroll-on-regress",
        help="Roll back automatically on MAJOR verdict.",
    ),
) -> None:
    """Promote ``new_adapter`` as the canary with a traffic split."""
    pct = _parse_traffic(traffic)
    state = _safe_read()
    # CanaryPolicy validates name shape + cross-fields; reuse here so the
    # state file can never disagree with the live router schema.
    try:
        policy = CanaryPolicy(
            stable=state.stable_name(),
            canary=new_adapter,
            traffic_pct=pct,
        )
    except (TypeError, ValueError) as exc:
        console.print(f"[red]invalid canary policy:[/] {for_terminal(str(exc))}")
        raise typer.Exit(code=2)
    # ``write_state`` refreshes updated_at on every persist, so we
    # explicitly route through replace() rather than mutate in place.
    new_state = replace(
        state,
        canary_active=policy.canary,
        canary_traffic_pct=policy.traffic_pct,
        canary_autoroll_on_regress=autoroll_on_regress,
        canary_rollout_id=uuid.uuid4().hex,
        canary_started_ts=time.time(),
        canary_cursor_ts=None,  # no iteration started it: its promotion leaves the data cursor
        canary_iteration=None,  # no iteration started this canary
    )
    # The log lines come first, as the daemon's do. A canary started by hand changes no
    # version; a running canary it replaces is withdrawn.
    log = LoopLog()
    line = withdrawn_line(state, f"replaced by soup loop canary {policy.canary}")
    if line is not None:
        log.iteration(line)
    log.progress(progress_line(
        None, "canary", "start", done=0, total=CANARY_MIN_SAMPLES,
        message=f"{policy.traffic_pct:g}% of keyed traffic; verdict after "
        f"{CANARY_MIN_SAMPLES} canary answers",
    ))
    owned = {
        name: getattr(new_state, name)
        for name in (
            "canary_active", "canary_traffic_pct", "canary_autoroll_on_regress",
            "canary_rollout_id", "canary_started_ts", "canary_cursor_ts",
            "canary_iteration",
        )
    }
    # Only the canary fields are this command's: what a daemon wrote meanwhile is kept.
    new_state = _update(lambda fresh: replace(fresh, **owned))
    console.print(
        f"[green]canary set:[/] {for_terminal(policy.canary or '')} @ "
        f"{policy.traffic_pct:.1f}% (autoroll={autoroll_on_regress})"
    )
    _warn_log(log)


@app.command("rollback")
def rollback_cmd() -> None:
    """Return to the version that was production before the last promotion.

    Points the server's unkeyed traffic back (SOUP_LOOP_SERVE_ENDPOINT, with
    --tool-auth-token / SOUP_TOOL_AUTH_TOKEN when the server needs it), then
    makes it the canary router's stable side again. A running canary is
    withdrawn too. With nothing promoted yet, a running canary is withdrawn
    and the served model keeps all traffic; with no canary either, refuses.
    """
    state = _safe_read()
    if state.production_name is None:
        if state.canary_active is None:
            console.print(
                "[red]nothing to roll back to:[/] no canary has been promoted and no canary "
                "is running; production is still the served model"
            )
            raise typer.Exit(code=2)
        # The server's unkeyed traffic already goes to the served model and its
        # router reads the canary from loop.yaml: clearing the canary is the rollback.
        # No version changes; the canary's iteration, if one started it, ends rolled_back.
        log = LoopLog()
        _log_withdrawal(log, state)
        _update(
            lambda fresh: replace(
                fresh,
                canary_active=None,
                canary_traffic_pct=None,
                canary_rollout_id=None,
                canary_adapter=None,
                canary_started_ts=None,
                canary_cursor_ts=None,
                canary_iteration=None,
            )
        )
        console.print(
            f"[green]canary withdrawn:[/] {for_terminal(state.canary_active)}; the served model "
            "keeps all traffic"
        )
        _warn_log(log)
        return
    from soup_cli.utils.loop_stages import switch_served_adapter

    target = state.previous_production_name
    ok, note = switch_served_adapter(target)
    if not ok:
        console.print(f"[red]rollback failed:[/] {for_terminal(note)}; loop.yaml is unchanged")
        raise typer.Exit(code=1)
    # The version log: a person went back a version (a running canary is withdrawn with it).
    log = LoopLog()
    state = start_log(state, log, actor="loop")  # an older loop.yaml: the log begins here
    changes = rollback_changes(state)
    _log_withdrawal(log, state)
    log.version(version_line(
        from_version=state.production_version, to_version=changes["production_version"],
        cause="rollback", actor="user", iteration_id=state.production_iteration_id,
        served_as=target, reason="soup loop rollback",
    ))
    log.progress(progress_line(
        state.production_iteration_id, "rollback", "end",
        message=f"production back to version {changes['production_version']}",
    ))
    cursor_changes = rollback_cursor_changes(state)
    _update(
        lambda fresh: replace(
            fresh,
            production_name=target,
            production_adapter=state.previous_production_adapter,
            previous_production_name=None,
            previous_production_adapter=None,
            canary_active=None,
            canary_traffic_pct=None,
            canary_rollout_id=None,
            canary_adapter=None,
            canary_started_ts=None,
            **changes,
            # the data cursor goes back with production (a running canary's own is withdrawn)
            **cursor_changes,
        )
    )
    label = target or "the served model"
    console.print(
        f"[green]rolled back:[/] {for_terminal(state.production_name)} -> {for_terminal(label)}"
        + (f" ([yellow]{for_terminal(note)}[/])" if note else "")
    )
    _warn_log(log)


@app.command("replay")
def replay_cmd(
    iteration_id: Optional[str] = typer.Argument(
        None, help="Iteration id (omit to list all)."
    ),
    extract: Optional[str] = typer.Option(
        None, "--extract",
        help="Extract the iteration's .can to this directory for a what-if re-run (v0.71.4).",
    ),
) -> None:
    """Replay a recorded loop iteration manifest."""
    if iteration_id is None:
        ids = list_iterations()
        if not ids:
            console.print("[yellow]no iterations recorded yet[/]")
            return
        console.print("\n".join(for_terminal(i) for i in ids))
        return
    try:
        record = read_iteration(iteration_id)
    except (FileNotFoundError, TypeError, ValueError) as exc:
        console.print(f"[red]replay failed:[/] {for_terminal(str(exc))}")
        raise typer.Exit(code=2)

    if extract is not None:
        from soup_cli.cans.unpack import extract_can
        from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink

        can_path = os.path.join(".soup-loops", iteration_id, "iteration.can")
        if not os.path.isfile(can_path):
            console.print(
                f"[red]no .can for {for_terminal(iteration_id)} "
                "(was it run with `watch --pack-cans`?)[/]"
            )
            raise typer.Exit(code=1)
        try:
            enforce_under_cwd_and_no_symlink(extract, "extract")
            dest = extract_can(can_path, extract)
        except (FileNotFoundError, ValueError, TypeError) as exc:
            console.print(f"[red]extract failed:[/] {for_terminal(str(exc))}")
            raise typer.Exit(code=2)
        console.print(
            f"[green]extracted {for_terminal(iteration_id)} -> {for_terminal(str(dest))}[/]"
        )
        return

    table = Table(title=f"replay {for_terminal(record.iteration_id)}", show_header=False)
    table.add_column("field", style="bold")
    table.add_column("value")
    for k, v in record.to_dict().items():
        table.add_row(for_terminal(str(k)), for_terminal(str(v)))
    console.print(table)


def _trace_dir_label(state: LoopState) -> str:
    """The trace directory the pre-wired harvest reads, and where it came from."""
    from soup_cli.utils.loop_stages import trace_dir_source

    path, source = trace_dir_source(state)
    return f"{path} (from {'--trace-dir' if source == 'trace_dir' else source})"


def _feedback_log_label(state: LoopState) -> str:
    """The feedback log the pre-wired harvest reads, and where it came from."""
    from soup_cli.utils.loop_stages import feedback_log_source

    path, source = feedback_log_source(state)
    return f"{path} (from {'--feedback-log' if source == 'feedback_log' else source})"


def _cursor_label(state: LoopState) -> str:
    """When the feedback the version in production was trained on ends (UTC), or "the start"."""
    if state.harvest_cursor_ts is None:
        return "the start"
    moment = datetime.fromtimestamp(state.harvest_cursor_ts, timezone.utc)
    return moment.replace(microsecond=0).isoformat()


def _parse_wait(raw: str, *, flag: str) -> float:
    """Seconds from ``"90m"`` / ``"24h"`` / ``"7d"`` / ``"3600"``; messages name ``flag``."""
    try:
        return parse_wait_duration(raw)
    except ValueError as exc:
        console.print(f"[red]invalid {flag}:[/] {for_terminal(exc)}")
        raise typer.Exit(code=2)


def _parse_traffic(raw: str, *, flag: str = "--traffic") -> float:
    """Parse ``"5%"`` / ``"5"`` / ``"  5.5 %"`` into a percent float; messages name ``flag``."""
    if not isinstance(raw, str):
        console.print(f"[red]{flag} must be a string, got {type(raw).__name__}[/]")
        raise typer.Exit(code=2)
    txt = raw.strip()
    if txt.endswith("%"):
        txt = txt[:-1].strip()
    try:
        pct = float(txt)
    except ValueError:
        console.print(f"[red]invalid {flag}:[/] {for_terminal(raw)}")
        raise typer.Exit(code=2)
    if not (0.0 <= pct <= 100.0):
        console.print(f"[red]{flag} must be in [0, 100][/]")
        raise typer.Exit(code=2)
    return pct
