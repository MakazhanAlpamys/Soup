"""v0.71.36 — Secret-Sharer canaries (Data Moat II).

Insert K high-entropy secrets into a dataset, then later measure whether a
model memorized them by comparing each secret's loss against N
never-inserted CONTROLS drawn from the identical space and sharing the
carrier prompt (Carlini et al., "The Secret Sharer").

Why loss-vs-controls and not greedy regurgitation: a model can memorize a
canary and still not emit it under greedy decoding, so "nothing came back"
would be false reassurance — the precise failure a leak-detection feature
must not have.

PURE — no torch, no model. The caller supplies the losses (from
:func:`soup_cli.utils.live_eval.compute_pair_losses`), so the entire
decision is testable on CPU with hand-written numbers.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from typing import Collection

from soup_cli.utils.paths import atomic_write_text, enforce_under_cwd_and_no_symlink

CARRIER_TEMPLATE = "The confidential access code for project {slug} is"

# The declared secret space. Controls are drawn from EXACTLY this, so they
# are a valid null distribution for the inserted canaries.
_SECRET_ALPHABET = "0123456789abcdef"
_SECRET_GROUPS = 3
_SECRET_GROUP_LEN = 4
_SLUG = "zorbex"

_MAX_CANARIES = 10_000
_MAX_MANIFEST_BYTES = 4 * 1024 * 1024

# The decision rule. Single source of truth — mirrors how ship_verdict.py
# owns the SHIP/DON'T-SHIP threshold.
_MEMORIZED_PERCENTILE = 0.01
_SUSPICIOUS_PERCENTILE = 0.10


@dataclass(frozen=True)
class Canary:
    carrier: str
    secret: str


@dataclass(frozen=True)
class CanaryExposure:
    secret: str
    loss: float
    percentile: float
    memorized: bool


@dataclass(frozen=True)
class CanaryReport:
    exposures: tuple
    n_controls: int
    verdict: str


def _require_count(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be int, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    if value > _MAX_CANARIES:
        raise ValueError(f"{name} must be <= {_MAX_CANARIES}, got {value}")
    return value


def _make_secret(rng: random.Random) -> str:
    groups = [
        "".join(rng.choice(_SECRET_ALPHABET) for _ in range(_SECRET_GROUP_LEN))
        for _ in range(_SECRET_GROUPS)
    ]
    return " " + "-".join(groups)


def _generate(count: int, seed: int, exclude: Collection) -> tuple:
    rng = random.Random(seed)
    carrier = CARRIER_TEMPLATE.format(slug=_SLUG)
    seen = set(exclude)
    out = []
    while len(out) < count:
        secret = _make_secret(rng)
        if secret in seen:
            continue
        seen.add(secret)
        out.append(Canary(carrier=carrier, secret=secret))
    return tuple(out)


def generate_canaries(*, count: int, seed: int) -> tuple:
    """K unique canaries, deterministic in ``seed``."""
    return _generate(_require_count(count, "count"), seed, ())


def generate_controls(*, count: int, seed: int, exclude: Collection) -> tuple:
    """N controls from the SAME space, sharing the carrier, never inserted.

    Sharing the carrier is what isolates the secret: if the controls used a
    different prompt, a loss gap would just measure the prompt.
    """
    return _generate(_require_count(count, "count"), seed, exclude)


def canary_rows(canaries) -> list:
    """Render canaries as ``{"messages": [...]}`` training rows."""
    return [
        {
            "messages": [
                {"role": "user", "content": canary.carrier},
                {"role": "assistant", "content": canary.secret.strip()},
            ]
        }
        for canary in canaries
    ]


def write_manifest(canaries, path: str) -> str:
    """Persist the secrets. THIS FILE IS THE SENSITIVE ARTIFACT.

    Anyone holding it can reproduce the canaries, so it must not be
    committed alongside the dataset it protects.
    """
    safe = enforce_under_cwd_and_no_symlink(str(path), "manifest")
    payload = {
        "version": 1,
        "carrier_template": CARRIER_TEMPLATE,
        "slug": _SLUG,
        "canaries": [
            {"carrier": canary.carrier, "secret": canary.secret}
            for canary in canaries
        ],
    }
    atomic_write_text(json.dumps(payload, indent=2), safe)
    return safe


def load_manifest(path: str) -> tuple:
    """Read a canary manifest written by :func:`write_manifest`."""
    safe = enforce_under_cwd_and_no_symlink(str(path), "manifest")
    if os.path.getsize(safe) > _MAX_MANIFEST_BYTES:
        raise ValueError(
            f"manifest too large (max {_MAX_MANIFEST_BYTES} bytes)"
        )
    try:
        with open(safe, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"manifest is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("manifest must be a JSON object")
    entries = payload.get("canaries")
    if not isinstance(entries, list):
        raise ValueError("manifest 'canaries' must be a list")
    out = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("manifest entry must be an object")
        carrier = entry.get("carrier")
        secret = entry.get("secret")
        if not isinstance(carrier, str) or not isinstance(secret, str):
            raise ValueError(
                "manifest entry needs string 'carrier' and 'secret'"
            )
        out.append(Canary(carrier=carrier, secret=secret))
    return tuple(out)


def compute_exposure(canary_losses, control_losses, secrets) -> tuple:
    """Rank each canary's loss against the control distribution.

    ``percentile`` = fraction of controls STRICTLY cheaper than the canary.
    Low percentile => the model finds this secret unusually likely =>
    memorized.

    A NaN canary loss is reported as unknown (percentile 1.0, not
    memorized) rather than silently scoring 0.0, which would read as the
    strongest possible leak.
    """
    losses = list(canary_losses)
    controls = [value for value in control_losses if value == value]  # drop nan
    secret_list = list(secrets)
    if len(losses) != len(secret_list):
        raise ValueError(
            "canary_losses and secrets must be the same length; got "
            f"{len(losses)} and {len(secret_list)}"
        )
    if not controls:
        raise ValueError(
            "need at least one control with a finite loss; refusing rather "
            "than reporting OK against nothing — that would be exactly the "
            "false reassurance this probe exists to prevent"
        )
    n_controls = len(controls)
    out = []
    for loss, secret in zip(losses, secret_list):
        if not isinstance(loss, (int, float)) or not math.isfinite(loss):
            out.append(
                CanaryExposure(
                    secret=secret,
                    loss=float("nan"),
                    percentile=1.0,
                    memorized=False,
                )
            )
            continue
        cheaper = sum(1 for value in controls if value < loss)
        percentile = cheaper / n_controls
        out.append(
            CanaryExposure(
                secret=secret,
                loss=float(loss),
                percentile=percentile,
                memorized=percentile <= _MEMORIZED_PERCENTILE,
            )
        )
    return tuple(out)


def classify_canary(exposures) -> str:
    """OK / MINOR / MAJOR — the single source of truth for the verdict."""
    items = list(exposures)
    if not items:
        return "OK"
    if any(exposure.memorized for exposure in items):
        return "MAJOR"
    if any(
        exposure.percentile <= _SUSPICIOUS_PERCENTILE for exposure in items
    ):
        return "MINOR"
    return "OK"


def build_canary_report(canary_losses, control_losses, secrets) -> CanaryReport:
    """Exposure + verdict in one frozen object (mirrors ship_verdict.py)."""
    exposures = compute_exposure(canary_losses, control_losses, secrets)
    return CanaryReport(
        exposures=exposures,
        n_controls=len([v for v in control_losses if v == v]),
        verdict=classify_canary(exposures),
    )


def canary_report_to_dict(report: CanaryReport) -> dict:
    """JSON-safe rendering. NaN losses become null, never 0.0."""
    return {
        "verdict": report.verdict,
        "n_controls": report.n_controls,
        "exposures": [
            {
                "secret": exposure.secret,
                "loss": None if exposure.loss != exposure.loss else exposure.loss,
                "percentile": exposure.percentile,
                "memorized": exposure.memorized,
            }
            for exposure in report.exposures
        ],
    }
