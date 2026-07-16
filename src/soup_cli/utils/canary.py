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
from typing import Any, Collection, Sequence

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
# Significance level for "more low-percentile canaries than chance explains".
_ALPHA = 0.05


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
    exposures: tuple[CanaryExposure, ...]
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


def _generate(
    count: int, seed: int, exclude: Collection[str]
) -> tuple[Canary, ...]:
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


def generate_canaries(*, count: int, seed: int) -> tuple[Canary, ...]:
    """K unique canaries, deterministic in ``seed``."""
    return _generate(_require_count(count, "count"), seed, ())


def generate_controls(
    *, count: int, seed: int, exclude: Collection[str]
) -> tuple[Canary, ...]:
    """N controls from the SAME space, sharing the carrier, never inserted.

    Sharing the carrier is what isolates the secret: if the controls used a
    different prompt, a loss gap would just measure the prompt.
    """
    return _generate(_require_count(count, "count"), seed, exclude)


def canary_rows(canaries: Sequence[Canary]) -> list[dict]:
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


def write_manifest(canaries: Sequence[Canary], path: str) -> str:
    """Persist the secrets. THIS FILE IS THE SENSITIVE ARTIFACT.

    Anyone holding it can reproduce the canaries, so it must not be
    committed alongside the dataset it protects, and it is chmod-600'd on
    POSIX: under the usual 022 umask a generic write lands 0644, letting
    any local user on a shared box read every canary without ever running
    ``check``. Mirrors registry/store.py, adapter_sign.py, audit_log.py.
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
    _harden_permissions(safe)
    return safe


def _harden_permissions(path: str) -> None:
    """Best-effort 0600 on a secret-bearing file (POSIX; no-op on Windows)."""
    if os.name != "nt":
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass


def load_manifest(path: str) -> tuple[Canary, ...]:
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
    if len(entries) > _MAX_CANARIES:
        # `check` runs one model forward pass per entry; the 4 MB size cap
        # alone still admits tens of thousands of minimal entries.
        raise ValueError(
            f"too many canaries in manifest ({len(entries)}); "
            f"max {_MAX_CANARIES}"
        )
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


def compute_exposure(
    canary_losses: Sequence[float],
    control_losses: Sequence[float],
    secrets: Sequence[str],
) -> tuple[CanaryExposure, ...]:
    """Rank each canary's loss against the control distribution.

    ``percentile`` = fraction of controls STRICTLY cheaper than the canary.
    Low percentile => the model finds this secret unusually likely =>
    memorized.

    Ties are counted as not-strictly-less, so an exact tie against every
    control yields percentile 0.0 and flags MAJOR. On real losses exact
    ties are vanishingly unlikely (they are float cross-entropies); where
    they do occur — e.g. a degenerate model emitting a constant loss — a
    false MAJOR is the safe direction for a leak detector, which must not
    under-report. Do not "fix" this by switching to ``<=``: that would let
    a genuinely memorized canary tying the cheapest control read as
    typical.

    A NaN canary loss is reported as unknown (percentile 1.0, not
    memorized) rather than silently scoring 0.0, which would read as the
    strongest possible leak.
    """
    losses = list(canary_losses)
    controls = [value for value in control_losses if not math.isnan(value)]
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


def _binomial_tail(k: int, n: int, prob: float) -> float:
    """P(X >= k) for X ~ Binomial(n, prob). Exact; n is small (<= 10k)."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return sum(
        math.comb(n, i) * (prob ** i) * ((1.0 - prob) ** (n - i))
        for i in range(k, n + 1)
    )


def classify_canary(exposures: Sequence[CanaryExposure]) -> str:
    """OK / MINOR / MAJOR — the single source of truth for the verdict.

    The verdict asks whether MORE canaries look memorized than chance alone
    explains, NOT whether any single one dipped low.

    Why: under the null (no memorization) each canary's percentile is
    ~Uniform(0,1), so P(percentile <= 0.01) = 0.01 PER CANARY — but with K
    canaries the chance that at least one dips below is 1 - 0.99^K. At the
    default K=16 that is ~15%, so an "any single canary" rule would fire
    MAJOR on a perfectly clean model roughly one run in seven — and MAJOR
    exits 2, which blocks a CI gate. Measured live on SmolLM2-135M: an
    untrained model produced percentiles spanning 1.6%-93%, with two under
    10%, purely from noise.

    So the count is tested against a binomial tail: a verdict fires only
    when observing that many low-percentile canaries would be unlikely
    (<5%) by chance. A genuinely memorized set is unmistakable — every
    canary lands at percentile 0.0 — so this costs no real sensitivity.
    """
    items = list(exposures)
    if not items:
        return "OK"
    n_canaries = len(items)
    n_memorized = sum(1 for e in items if e.memorized)
    n_suspicious = sum(
        1 for e in items if e.percentile <= _SUSPICIOUS_PERCENTILE
    )
    if _binomial_tail(n_memorized, n_canaries, _MEMORIZED_PERCENTILE) < _ALPHA:
        return "MAJOR"
    if _binomial_tail(n_suspicious, n_canaries, _SUSPICIOUS_PERCENTILE) < _ALPHA:
        return "MINOR"
    return "OK"


def build_canary_report(
    canary_losses: Sequence[float],
    control_losses: Sequence[float],
    secrets: Sequence[str],
) -> CanaryReport:
    """Exposure + verdict in one frozen object (mirrors ship_verdict.py)."""
    exposures = compute_exposure(canary_losses, control_losses, secrets)
    return CanaryReport(
        exposures=exposures,
        n_controls=len([v for v in control_losses if not math.isnan(v)]),
        verdict=classify_canary(exposures),
    )


def canary_report_to_dict(report: CanaryReport) -> dict[str, Any]:
    """JSON-safe rendering. NaN losses become null, never 0.0."""
    return {
        "verdict": report.verdict,
        "n_controls": report.n_controls,
        "exposures": [
            {
                "secret": exposure.secret,
                "loss": (
                    None if math.isnan(exposure.loss) else exposure.loss
                ),
                "percentile": exposure.percentile,
                "memorized": exposure.memorized,
            }
            for exposure in report.exposures
        ],
    }
