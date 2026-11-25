"""Machine-readable results for a parsed, opt-in local train dry run."""

from __future__ import annotations

import inspect
import os
import re
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager, redirect_stdout
from contextvars import ContextVar
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from time import monotonic
from types import UnionType
from typing import TYPE_CHECKING, Any, Union, get_args, get_origin

import typer
from pydantic import BaseModel, ValidationError
from rich.console import Console

from soup_cli import __version__
from soup_cli.config.loader import ConfigLoadExit
from soup_cli.config.schema import SoupConfig
from soup_cli.utils.command_result import (
    CommandResult,
    PreflightEstimateBasis,
    PreflightInputDigests,
    PreflightResolved,
    PreflightVram,
    ResultReason,
    TrainPreflightData,
    encode_command_result,
)
from soup_cli.utils.paths import is_under
from soup_cli.utils.terminal import for_terminal, strip_control
from soup_cli.utils.typer_compat import ParameterSource, get_current_context

if TYPE_CHECKING:
    from soup_cli.utils.hardware_fit import HardwareFitReport

_UNSUPPORTED_OPTIONS = frozenset({
    "cloud", "cloud_submit", "gpu", "gpus", "nodes", "node_rank", "master_addr", "master_port",
    "no_reexec", "deepspeed", "fsdp", "find_lr", "find_lr_start", "find_lr_end",
    "find_lr_steps", "find_lr_output", "hf_resume", "resume", "push_as", "push_public",
    "wandb", "tensorboard", "tracker", "profile_run", "capture_activations", "capture_prompts",
    "annex_xi", "repro_receipt", "track_energy", "energy_country", "energy_out",
    "diagnose_gate", "diagnose_gate_require_all_modes", "trust_remote_code",
})
_UNSUPPORTED = "soup.train.preflight_unsupported"
_INTERNAL_ERROR = "soup.cli.execution_failed"
_INTERRUPTED = "soup.train.interrupted"
_INCOMPLETE = "soup.train.preflight_incomplete"
_SERIALIZATION_ERROR = "soup.cli.result_serialization_failed"
_REFUSED = "soup.train.preflight_refused"
_FIT_WARNING = "soup.train.hardware_fit_warning"
_REASON_TEXT = {
    _UNSUPPORTED: (
        "This result mode requires a supported local dry run.",
        "Use --dry-run --result-json without launch-only options.",
    ),
    "soup.config.missing": (
        "The configuration file was not found.", "Supply an existing config or run soup init.",
    ),
    "soup.config.invalid": (
        "The configuration could not be loaded.", "Correct the YAML or the listed fields.",
    ),
    "soup.data.invalid": (
        "Training data could not be loaded.", "Check the source, format and installed readers.",
    ),
    "soup.data.empty_train": (
        "No usable training rows were loaded.",
        "Supply rows that survive validation and the configured split.",
    ),
    "soup.train.hardware_fit": (
        "Predicted GPU memory exceeds capacity.",
        "Reduce batch or context memory use, or select supported capacity.",
    ),
    _FIT_WARNING: (
        "Hardware-fit validation reported a warning.", "Review the local estimate and warning.",
    ),
    _REFUSED: ("Training preflight refused the request.", "Review the local refusal diagnostic."),
    _INCOMPLETE: (
        "Preflight ended before validation completed.", "Check the local diagnostic and retry.",
    ),
    _INTERRUPTED: ("Training preflight was interrupted.", "Run preflight again when intended."),
    _INTERNAL_ERROR: (
        "The command could not complete.",
        "Check the local diagnostic and correct the environment or input.",
    ),
    _SERIALIZATION_ERROR: (
        "The command result could not be encoded.", "Check the local diagnostic and retry.",
    ),
}
_MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*(?:/[A-Za-z0-9][A-Za-z0-9._-]*)?")
_ERROR_TYPE = re.compile(r"[a-z][a-z0-9_.]{0,95}")
_UNKNOWN_KEY = "<key>"


def _safe_text(value: str | None) -> str | None:
    if value is None:
        return None
    return strip_control(value, single_line=True).strip()[:1024] or None


def _model_reference(value: str) -> str | None:
    safe = _safe_text(value)
    if safe != value:
        return None
    path = Path(value)
    try:
        if path.exists():
            root = Path.cwd()
            if not is_under(path, root):
                return None
            relative = Path(os.path.relpath(os.path.realpath(path), os.path.realpath(root)))
            return relative.as_posix() if relative.parts else None
    except (OSError, ValueError):
        return None
    return value if _MODEL_ID.fullmatch(value) else None


def _declared_fields(annotation: Any) -> dict[str, Any]:
    if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
        return {name: field.annotation for name, field in annotation.model_fields.items()}
    fields = {}
    if get_origin(annotation) in (UnionType, Union):
        for member in get_args(annotation):
            fields.update(_declared_fields(member))
    return fields


def _safe_location(location: tuple) -> list[str | int]:
    annotation: Any = SoupConfig
    path: list[str | int] = []
    for segment in location[:16]:
        if type(segment) is int and segment >= 0:
            path.append(segment)
            args = get_args(annotation)
            annotation = args[0] if args and get_origin(annotation) in (list, tuple) else None
            continue
        fields = _declared_fields(annotation)
        if isinstance(segment, str) and segment in fields:
            path.append(segment)
            annotation = fields[segment]
        else:
            path.append(_UNKNOWN_KEY)
            annotation = None
    return path


def _config_details(cause: BaseException | None) -> dict[str, Any]:
    if isinstance(cause, ConfigLoadExit):
        errors = [{"loc": [_UNKNOWN_KEY], "type": cause.error_type}] if cause.error_type else []
        return {"category": cause.category, "errors": errors}
    underlying = (cause.__cause__ or cause.__context__ or cause) if cause else None
    if isinstance(underlying, ValidationError):
        errors = underlying.errors(include_url=False, include_context=False, include_input=False)
        return {
            "category": "validation",
            "errors": [
                {
                    "loc": _safe_location(item["loc"]),
                    "type": (
                        item["type"] if _ERROR_TYPE.fullmatch(item["type"]) else "validation_error"
                    ),
                }
                for item in errors
            ],
        }
    return {"category": "validation", "errors": []}


def _reason(code: str) -> ResultReason:
    message, fix = _REASON_TEXT[code]
    return ResultReason(code=code, message=message, fix=fix)


class TrainPreflight:
    """Collect data already observed by the train command before its dry-run boundary."""

    def __init__(self) -> None:
        self._cfg: SoupConfig | None = None
        self._device_name: str | None = None
        self._vram: PreflightVram | None = None
        self._training_rows: int | None = None
        self._validation_rows: int | None = None
        self._before_hash: str | None = None
        self._data_hash: str | None = None
        self._warned_fit = False
        self._refusal: str | None = None
        self._cause: BaseException | None = None
        self._ready = False
        self._started_at = datetime.now(UTC)
        self._started = monotonic()

    def config(self, cfg: SoupConfig) -> None:
        self._cfg = cfg

    def device(self, name: str) -> None:
        self._device_name = _safe_text(name)

    def fit(self, report: HardwareFitReport, margin_fraction: float) -> None:
        self._vram = PreflightVram(
            estimated_peak_gb=report.peak_vram_gb,
            available_gb=report.available_vram_gb,
            margin_fraction=margin_fraction,
            required_gb=report.required_with_margin_gb,
        )

    def fit_warning(self) -> None:
        self._warned_fit = True

    def before_load(self, cfg: SoupConfig) -> None:
        from soup_cli.utils.training_identity import capture_training_data_hash

        self._before_hash = capture_training_data_hash(cfg)

    def loaded(
        self, cfg: SoupConfig, training_rows: int | None, validation_rows: int | None,
    ) -> None:
        from soup_cli.utils.training_identity import capture_training_data_hash

        self._training_rows = training_rows
        self._validation_rows = validation_rows
        if self._before_hash is not None and self._before_hash == capture_training_data_hash(cfg):
            self._data_hash = self._before_hash

    def refuse(self, code: str, cause: BaseException | None = None) -> None:
        self._refusal = code
        self._cause = cause

    def complete(self) -> None:
        self._ready = True

    def _payload(self) -> TrainPreflightData | None:
        cfg = self._cfg
        if cfg is None:
            return None
        training = cfg.training
        batch = training.batch_size if type(training.batch_size) is int else None
        known_single_process_batch = (
            cfg.task == "sft" and cfg.backend == "transformers"
            and not getattr(training, "packing", False) and not training.multipack
        )
        full_tuning = bool(
            training.unfrozen_parameters or training.lisa_enabled or training.lora.r == 0
        )
        return TrainPreflightData(
            dry_run=True,
            training_rows=self._training_rows,
            validation_rows=self._validation_rows,
            vram=self._vram or PreflightVram(),
            estimate_basis=PreflightEstimateBasis(
                vram="analytical_model" if self._vram is not None else "unavailable",
            ),
            resolved=PreflightResolved(
                base_model=_model_reference(cfg.base),
                task=cfg.task,
                backend=cfg.backend,
                device_name=self._device_name,
                epochs=training.epochs,
                batch_size_per_device=batch,
                effective_batch_size=(
                    batch * training.gradient_accumulation_steps
                    if batch is not None and known_single_process_batch else None
                ),
                context_length=cfg.data.max_length,
                adapter_rank=training.lora.r if not full_tuning and cfg.task == "sft" else None,
            ),
            input_digests=PreflightInputDigests(data_sha256=self._data_hash),
        )

    def _outcome(self, exit_code: int, terminal_reason: str | None) -> tuple[str, str | None]:
        if terminal_reason == _INTERRUPTED:
            return "cancelled", terminal_reason
        if terminal_reason:
            return "failed", terminal_reason
        if exit_code == 0 and self._ready:
            return "ok", None
        if exit_code == 0:
            return "failed", _INCOMPLETE
        return "refused", self._refusal or _REFUSED

    def _details(self, code: str) -> dict[str, Any] | None:
        if code == "soup.config.invalid":
            return _config_details(self._cause)
        if code == "soup.train.hardware_fit":
            return {
                "required_gb": self._vram.required_gb if self._vram is not None else None,
                "available_gb": self._vram.available_gb if self._vram is not None else None,
            }
        return None

    def encode(self, exit_code: int, terminal_reason: str | None) -> bytes:
        identity = {
            "schema": "soup.command_result",
            "schema_version": 1,
            "soup_version": __version__,
            "command": "train",
            "phase": "preflight",
            "exit_code": exit_code,
            "started_at": self._started_at,
            "duration_s": monotonic() - self._started,
        }
        status, code = self._outcome(exit_code, terminal_reason)
        try:
            reasons = []
            if code is not None:
                reason = _reason(code)
                reasons.append({
                    "code": reason.code, "message": reason.message, "fix": reason.fix,
                    "details": self._details(code),
                })
            result = CommandResult(
                **identity, status=status, data=self._payload(), reasons=reasons,
                warnings=[_reason(_FIT_WARNING)] if self._warned_fit else [],
            )
        except (TypeError, ValueError, OSError, OverflowError):
            result = CommandResult(
                **identity, status="failed", reasons=[_reason(_SERIALIZATION_ERROR)],
            )
        return encode_command_result(result)


_ACTIVE_PREFLIGHT: ContextVar[TrainPreflight | None] = ContextVar(
    "train_preflight_result", default=None,
)


def active_preflight() -> TrainPreflight | None:
    return _ACTIVE_PREFLIGHT.get()


@contextmanager
def _diagnostics_to_stderr() -> Iterator[None]:
    """Route Python output and native standard output until the JSON write."""
    try:
        error_fd = sys.stderr.fileno()
    except (AttributeError, OSError, ValueError):
        error_fd = 2

    saved = None
    try:
        sys.stdout.flush()
        saved = os.dup(1)
        os.dup2(error_fd, 1)
    except (AttributeError, OSError, ValueError):
        if saved is not None:
            os.close(saved)
            saved = None

    try:
        with redirect_stdout(sys.stderr):
            yield
    finally:
        if saved is not None:
            try:
                try:
                    sys.stderr.flush()
                except (AttributeError, OSError, ValueError):
                    pass
                os.dup2(saved, 1)
            finally:
                os.close(saved)


def _unsupported(arguments: dict[str, Any]) -> bool:
    from soup_cli.utils.launcher import is_in_distributed

    if arguments.get("dry_run") is not True or is_in_distributed():
        return True
    if os.environ.get("WORLD_SIZE", "") not in {"", "0", "1"} or "LOCAL_RANK" in os.environ:
        return True
    ctx = get_current_context(silent=True)
    if ctx is None:
        return bool(_UNSUPPORTED_OPTIONS.intersection(arguments))
    return any(
        ctx.get_parameter_source(name) not in (None, ParameterSource.DEFAULT)
        for name in _UNSUPPORTED_OPTIONS
    )


def train_preflight_result(command: Callable[..., Any]) -> Callable[..., Any]:
    """Reserve stdout only for parsed invocations that request a result."""
    signature = inspect.signature(command)
    result_position = list(signature.parameters).index("result_json")

    @wraps(command)
    def invoke(*args: Any, **kwargs: Any) -> Any:
        requested = kwargs.get(
            "result_json", args[result_position] if len(args) > result_position else False,
        )
        if not requested:
            return command(*args, **kwargs)

        arguments = signature.bind(*args, **kwargs).arguments
        receipt = TrainPreflight()
        token = _ACTIVE_PREFLIGHT.set(receipt)
        exit_code = 0
        terminal_reason = None
        output = Console(stderr=True)
        try:
            with _diagnostics_to_stderr():
                try:
                    if _unsupported(arguments):
                        reason = _reason(_UNSUPPORTED)
                        output.print(f"{reason.message} {reason.fix}", style="red", markup=False)
                        receipt.refuse(_UNSUPPORTED)
                        raise typer.Exit(2)
                    command(*args, **kwargs)
                except typer.Exit as exc:
                    exit_code = exc.exit_code
                except SystemExit as exc:
                    exit_code = exc.code if type(exc.code) is int else (1 if exc.code else 0)
                except KeyboardInterrupt:
                    exit_code = 130
                    terminal_reason = _INTERRUPTED
                    output.print("Training preflight interrupted.", style="yellow")
                except Exception as exc:
                    exit_code = 1
                    if receipt._refusal is not None:
                        detail = for_terminal(str(exc), single_line=True)[:1024]
                        output.print(
                            detail or _reason(receipt._refusal).message,
                            style="red", markup=False,
                        )
                    else:
                        terminal_reason = _INTERNAL_ERROR
                        name = for_terminal(type(exc).__name__, single_line=True)
                        output.print(
                            f"Training preflight failed unexpectedly ({name}).",
                            style="red", markup=False,
                        )
        finally:
            _ACTIVE_PREFLIGHT.reset(token)

        # A transport failure must not trigger a second write.
        sys.stdout.buffer.write(receipt.encode(exit_code, terminal_reason))
        if exit_code:
            raise typer.Exit(exit_code)

    return invoke
