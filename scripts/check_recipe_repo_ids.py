#!/usr/bin/env python3
"""Resolve every shipped recipe's Hugging Face repo id, anonymously (#677).

The catalog names a repo in two independent places per recipe -- ``RecipeMeta.model``
and the YAML ``base:`` -- and until now nothing checked that either resolves.
#661 found two that did not; both had shipped and been released, because
``validate-recipes`` parses the YAML through the real schema and a nonexistent
repo id is a perfectly well-formed string.

Three things this deliberately does NOT do, each of which would make the guard
worse than nothing:

**It does not treat a status code as evidence.** An invented repo id returns the
same ``401`` from the Hub API as a real private one -- the mistake made in #661
before that report was worth filing. The authoritative signal is
``huggingface_hub``'s ``RepositoryNotFoundError``.

**It does not report gated repos.** A gated repo (``meta-llama/*``, ``google/gemma-*``)
resolves for an authorised user and refuses an anonymous one. Reporting those
would fire forever on correct recipes, and a guard that cries wolf gets muted.
``model_info`` exposes ``gated``; this reads it rather than inferring.

**It does not report a transient failure as a missing repo.** Timeouts, 5xx and
rate limits classify as UNVERIFIED -- "could not check" -- after retries. A bad
minute at the Hub must not become a bug report.

Run by ``.github/workflows/recipe-repo-ids.yml`` on a schedule. Never on pull
requests: 163 network calls per PR would make every unrelated change depend on
Hub availability.
"""

from __future__ import annotations

import enum
import os
import sys
import time
from dataclasses import dataclass
from typing import Any

_DEFAULT_ATTEMPTS = 3
_DEFAULT_BACKOFF = 2.0


class Status(enum.Enum):
    """Four outcomes, not two. Collapsing GATED or UNVERIFIED into MISSING is
    what turns this from a guard into noise."""

    EXISTS = "exists"
    GATED = "gated"
    MISSING = "missing"
    UNVERIFIED = "unverified"


@dataclass(frozen=True)
class RepoCheck:
    repo_id: str
    status: Status
    detail: str = ""


@dataclass(frozen=True)
class SurfacePair:
    """The two places a recipe names its model. Both are checked because #666's
    mutation testing showed the snapshot guard sees only the YAML half, so a
    wrong ``RecipeMeta.model`` slips past everything else."""

    meta_model: str
    yaml_base: str


@dataclass(frozen=True)
class RecipeReport:
    meta: RepoCheck
    yaml: RepoCheck


def classify_repo(
    api: Any,
    repo_id: str,
    *,
    not_found: type[BaseException],
    attempts: int = _DEFAULT_ATTEMPTS,
    backoff: float = _DEFAULT_BACKOFF,
) -> RepoCheck:
    """Resolve one repo id into exactly one :class:`Status`.

    ``not_found`` is injected rather than imported at module scope so the
    classification logic is testable without ``huggingface_hub`` installed and
    without touching the network.

    A ``RepositoryNotFoundError`` is a definite answer and is **not** retried;
    retrying it would burn the attempt budget that transient failures need.
    """
    last_detail = ""
    for attempt in range(1, max(1, attempts) + 1):
        try:
            info = api.model_info(repo_id)
        except not_found as exc:
            return RepoCheck(repo_id, Status.MISSING, str(exc)[:200])
        except Exception as exc:  # noqa: BLE001 -- anything else may be transient
            last_detail = f"{type(exc).__name__}: {str(exc)[:160]}"
            if attempt < attempts:
                time.sleep(backoff * attempt)
            continue
        if getattr(info, "gated", False):
            return RepoCheck(repo_id, Status.GATED, "gated; resolves for an authorised user")
        return RepoCheck(repo_id, Status.EXISTS, "")
    return RepoCheck(repo_id, Status.UNVERIFIED, last_detail)


def collect_recipe_repo_ids() -> dict[str, SurfacePair]:
    """Both model-id surfaces for every recipe in the shipped catalog."""
    import yaml as _yaml

    from soup_cli.recipes.catalog import RECIPES

    out: dict[str, SurfacePair] = {}
    for name, meta in RECIPES.items():
        parsed = _yaml.safe_load(meta.yaml_str) or {}
        out[name] = SurfacePair(meta_model=meta.model, yaml_base=parsed.get("base", ""))
    return out


def check_surfaces(
    surfaces: dict[str, Any],
    *,
    api: Any,
    not_found: type[BaseException],
    attempts: int = _DEFAULT_ATTEMPTS,
    backoff: float = _DEFAULT_BACKOFF,
) -> dict[str, RecipeReport]:
    """Resolve every surface, caching by id so a shared base is fetched once."""
    cache: dict[str, RepoCheck] = {}

    def _check(repo_id: str) -> RepoCheck:
        if repo_id not in cache:
            cache[repo_id] = classify_repo(
                api, repo_id, not_found=not_found, attempts=attempts, backoff=backoff
            )
        return cache[repo_id]

    report: dict[str, RecipeReport] = {}
    for name, pair in surfaces.items():
        if isinstance(pair, SurfacePair):
            meta_id, yaml_id = pair.meta_model, pair.yaml_base
        else:                       # a plain (meta, yaml) tuple, used by tests
            meta_id, yaml_id = pair
        report[name] = RecipeReport(meta=_check(meta_id), yaml=_check(yaml_id))
    return report


def missing_only(report: dict[str, RecipeReport]) -> list[str]:
    """Recipes with a genuinely nonexistent id on either surface."""
    return sorted(
        name for name, rec in report.items()
        if Status.MISSING in (rec.meta.status, rec.yaml.status)
    )


def unverified_only(report: dict[str, RecipeReport]) -> list[str]:
    """Recipes the Hub could not answer for. Reported separately, never as
    missing, and never a failure on their own."""
    return sorted(
        name for name, rec in report.items()
        if Status.MISSING not in (rec.meta.status, rec.yaml.status)
        and Status.UNVERIFIED in (rec.meta.status, rec.yaml.status)
    )


def _anonymous_api():
    """A Hub client with authentication explicitly off.

    ``HF_TOKEN`` in the environment would silently resolve gated repos as
    EXISTS, so the guard would go quiet exactly where a wrong id is most likely
    to hide. Refusing is better than reporting a result that means something
    different from what it says.
    """
    from huggingface_hub import HfApi

    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        if os.environ.get(var):
            raise SystemExit(
                f"{var} is set. This check must run anonymously: a token turns "
                "every gated repo into EXISTS and hides the case the guard is for."
            )
    return HfApi(token=False)


def format_report(report: dict[str, RecipeReport]) -> str:
    missing, unverified = missing_only(report), unverified_only(report)
    counts: dict[Status, int] = {s: 0 for s in Status}
    for rec in report.values():
        for check in (rec.meta, rec.yaml):
            counts[check.status] += 1

    lines = [
        f"checked {len(report)} recipes / {sum(counts.values())} surfaces",
        "  " + "  ".join(f"{s.value}={counts[s]}" for s in Status),
        "",
    ]
    if missing:
        lines.append(f"MISSING — {len(missing)} recipe(s) name a repo that does not exist:")
        for name in missing:
            rec = report[name]
            for label, check in (("RecipeMeta.model", rec.meta), ("YAML base:", rec.yaml)):
                if check.status is Status.MISSING:
                    lines.append(f"  {name}: {label} -> {check.repo_id}")
    else:
        lines.append("MISSING — none")
    if unverified:
        lines += ["", f"COULD NOT CHECK — {len(unverified)} recipe(s), not a failure:"]
        lines += [f"  {name}: {report[name].meta.detail or report[name].yaml.detail}"
                  for name in unverified]
    return "\n".join(lines)


def main() -> int:
    from huggingface_hub.errors import RepositoryNotFoundError

    report = check_surfaces(
        collect_recipe_repo_ids(), api=_anonymous_api(), not_found=RepositoryNotFoundError
    )
    print(format_report(report))
    # Exit non-zero ONLY for genuinely missing repos. An unverified run is not a
    # failure: making it one would mean a Hub outage reads as a broken catalog.
    return 1 if missing_only(report) else 0


if __name__ == "__main__":
    sys.exit(main())
