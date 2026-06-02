"""GitHub-shaped adapter PR rendering (v0.67.0 Part D).

A PR for an adapter is the triple ``{base SHA, dataset diff, adapter
weights}`` plus an eval-delta report. This module renders it as a
review-friendly Markdown document with eval-delta tables and
side-by-side sample-output diffs, suitable for posting as a GitHub
PR comment via the v0.68.0 GitHub Action.

Public surface:

- ``EvalDelta`` / ``SampleDiff`` / ``AdapterPR`` frozen dataclasses
- ``build_adapter_pr(...)`` factory from raw dicts (CLI / API friendly)
- ``render_pr_markdown(pr)`` returns review-ready Markdown
- ``render_pr_json(pr)`` returns JSON for downstream consumers
- ``write_pr_markdown(pr, path)`` atomic cwd-contained write
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable, Mapping, Optional, Tuple

from soup_cli.utils.paths import atomic_write_text

# ---------------------------------------------------------------------------
# Bounds (closed)
# ---------------------------------------------------------------------------

MAX_TITLE_LEN = 256
MAX_METRIC_NAME_LEN = 256
MAX_OUTPUT_LEN = 32_768  # per-output cap (defense + keeps PRs reviewable)
MAX_DATASET_DIFF_LEN = 1_048_576  # 1 MiB
MAX_DELTAS = 64
MAX_SAMPLES = 256
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def _check_text_field(value: object, field: str, max_len: int) -> str:
    if isinstance(value, bool):
        raise TypeError(f"{field} must not be bool")
    if not isinstance(value, str):
        raise TypeError(f"{field} must be str")
    if "\x00" in value:
        raise ValueError(f"{field} must not contain null bytes")
    if len(value) > max_len:
        raise ValueError(
            f"{field} length {len(value)} > {max_len}"
        )
    return value


def _check_required_text(value: object, field: str, max_len: int) -> str:
    text = _check_text_field(value, field, max_len)
    if not text:
        raise ValueError(f"{field} must be non-empty")
    return text


def _check_finite(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field} must not be bool")
    if not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    val = float(value)
    if not math.isfinite(val):
        raise ValueError(f"{field} must be finite")
    return val


def _check_sha256(value: object, field: str) -> str:
    if isinstance(value, bool):
        raise TypeError(f"{field} must not be bool")
    if not isinstance(value, str):
        raise TypeError(f"{field} must be str")
    if not _SHA256_RE.match(value):
        raise ValueError(f"{field} must be 64 hex chars")
    return value


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalDelta:
    """One metric: baseline -> candidate.

    The ``delta`` field is a computed property (post-init).
    """

    metric: str
    baseline: float
    candidate: float

    def __post_init__(self) -> None:
        _check_required_text(self.metric, "metric", MAX_METRIC_NAME_LEN)
        _check_finite(self.baseline, "baseline")
        _check_finite(self.candidate, "candidate")

    @property
    def delta(self) -> float:
        return self.candidate - self.baseline


@dataclass(frozen=True)
class SampleDiff:
    """One prompt with baseline + candidate outputs.

    Long outputs are rejected (not truncated) so PR authors notice and
    pre-truncate intentionally — keeps the PR reviewable.
    """

    prompt: str
    baseline_output: str
    candidate_output: str

    def __post_init__(self) -> None:
        _check_required_text(self.prompt, "prompt", MAX_OUTPUT_LEN)
        _check_text_field(
            self.baseline_output, "baseline_output", MAX_OUTPUT_LEN
        )
        _check_text_field(
            self.candidate_output, "candidate_output", MAX_OUTPUT_LEN
        )


@dataclass(frozen=True)
class AdapterPR:
    """An adapter PR — the triple {base SHA, dataset diff, adapter path}
    plus eval deltas + sample diffs for human review."""

    title: str
    base_sha: str
    adapter_path: str
    dataset_diff: str
    deltas: Tuple[EvalDelta, ...]
    samples: Tuple[SampleDiff, ...]

    def __post_init__(self) -> None:
        _check_required_text(self.title, "title", MAX_TITLE_LEN)
        _check_sha256(self.base_sha, "base_sha")
        _check_required_text(self.adapter_path, "adapter_path", MAX_OUTPUT_LEN)
        _check_text_field(
            self.dataset_diff, "dataset_diff", MAX_DATASET_DIFF_LEN
        )
        if not isinstance(self.deltas, tuple):
            raise TypeError("deltas must be tuple")
        if len(self.deltas) > MAX_DELTAS:
            raise ValueError(
                f"too many deltas ({len(self.deltas)} > {MAX_DELTAS})"
            )
        for d in self.deltas:
            if not isinstance(d, EvalDelta):
                raise TypeError("deltas entries must be EvalDelta")
        if not isinstance(self.samples, tuple):
            raise TypeError("samples must be tuple")
        if len(self.samples) > MAX_SAMPLES:
            raise ValueError(
                f"too many samples ({len(self.samples)} > {MAX_SAMPLES})"
            )
        for s in self.samples:
            if not isinstance(s, SampleDiff):
                raise TypeError("samples entries must be SampleDiff")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_adapter_pr(
    *,
    title: str,
    base_sha: str,
    adapter_path: str,
    dataset_diff: str,
    deltas: Iterable[Mapping[str, Any]],
    samples: Iterable[Mapping[str, Any]],
) -> AdapterPR:
    """Build a frozen ``AdapterPR`` from raw dicts (CLI / API friendly)."""
    delta_objs = []
    for raw in deltas:
        if not isinstance(raw, Mapping):
            raise TypeError("each delta must be a mapping")
        delta_objs.append(
            EvalDelta(
                metric=raw.get("metric", ""),
                baseline=raw.get("baseline", 0.0),
                candidate=raw.get("candidate", 0.0),
            )
        )
    sample_objs = []
    for raw in samples:
        if not isinstance(raw, Mapping):
            raise TypeError("each sample must be a mapping")
        sample_objs.append(
            SampleDiff(
                prompt=raw.get("prompt", ""),
                baseline_output=raw.get("baseline_output", ""),
                candidate_output=raw.get("candidate_output", ""),
            )
        )
    return AdapterPR(
        title=title,
        base_sha=base_sha,
        adapter_path=adapter_path,
        dataset_diff=dataset_diff,
        deltas=tuple(delta_objs),
        samples=tuple(sample_objs),
    )


# ---------------------------------------------------------------------------
# Markdown rendering — escapes table-active and link-active characters
# ---------------------------------------------------------------------------


def _md_table_escape(text: str) -> str:
    """Neutralise markdown table-active characters in a cell.

    Mirrors v0.29.0 model card v2 + v0.59.0 Annex XI ``_md_escape``
    policy: a forged-heading or table-injection attack via crafted
    metric names / prompts must not leak into the rendered PR document.
    """
    if not isinstance(text, str):
        return ""
    # Order matters: \\ first
    out = text.replace("\\", "\\\\")
    out = out.replace("|", "\\|")
    out = out.replace("\n", " ")
    out = out.replace("\r", " ")
    out = out.replace("\t", " ")
    return out


def _md_body_escape(text: str) -> str:
    """Lighter escape for code-fence body: only null bytes + CR."""
    if not isinstance(text, str):
        return ""
    return text.replace("\x00", "").replace("\r\n", "\n")


def render_pr_markdown(pr: AdapterPR) -> str:
    """Render an ``AdapterPR`` as a GitHub-style PR Markdown document."""
    if not isinstance(pr, AdapterPR):
        raise TypeError("pr must be AdapterPR")

    lines: list[str] = []
    lines.append(f"# {_md_table_escape(pr.title)}")
    lines.append("")
    lines.append(f"**Base SHA:** `{pr.base_sha[:12]}…`")
    lines.append(f"**Adapter:** `{_md_table_escape(pr.adapter_path)}`")
    lines.append("")

    # Eval deltas table
    if pr.deltas:
        lines.append("## Eval deltas")
        lines.append("")
        lines.append("| Metric | Baseline | Candidate | Δ |")
        lines.append("|---|---|---|---|")
        for d in pr.deltas:
            metric = _md_table_escape(d.metric)
            sign = "+" if d.delta >= 0 else ""
            lines.append(
                f"| {metric} | {d.baseline:.4f} | {d.candidate:.4f} | "
                f"{sign}{d.delta:.4f} |"
            )
        lines.append("")

    # Dataset diff
    if pr.dataset_diff:
        lines.append("## Dataset diff")
        lines.append("")
        lines.append("```diff")
        lines.append(_md_body_escape(pr.dataset_diff))
        lines.append("```")
        lines.append("")

    # Sample diffs
    if pr.samples:
        lines.append("## Sample diffs")
        lines.append("")
        for idx, s in enumerate(pr.samples, 1):
            lines.append(f"### Sample {idx}")
            lines.append("")
            lines.append("**Prompt:**")
            lines.append("")
            lines.append("```")
            lines.append(_md_body_escape(s.prompt))
            lines.append("```")
            lines.append("")
            lines.append("**Baseline:**")
            lines.append("")
            lines.append("```")
            lines.append(_md_body_escape(s.baseline_output))
            lines.append("```")
            lines.append("")
            lines.append("**Candidate:**")
            lines.append("")
            lines.append("```")
            lines.append(_md_body_escape(s.candidate_output))
            lines.append("```")
            lines.append("")

    return "\n".join(lines)


def render_pr_json(pr: AdapterPR) -> str:
    """Render an ``AdapterPR`` as JSON (for downstream API consumers)."""
    if not isinstance(pr, AdapterPR):
        raise TypeError("pr must be AdapterPR")
    data = {
        "title": pr.title,
        "base_sha": pr.base_sha,
        "adapter_path": pr.adapter_path,
        "dataset_diff": pr.dataset_diff,
        "deltas": [
            {
                "metric": d.metric,
                "baseline": d.baseline,
                "candidate": d.candidate,
                "delta": d.delta,
            }
            for d in pr.deltas
        ],
        "samples": [asdict(s) for s in pr.samples],
    }
    return json.dumps(data, indent=2, sort_keys=True, allow_nan=False)


def write_pr_markdown(pr: AdapterPR, path: str) -> str:
    """Atomic cwd-contained write of the rendered PR markdown."""
    if not isinstance(pr, AdapterPR):
        raise TypeError("pr must be AdapterPR")
    text = render_pr_markdown(pr)
    return atomic_write_text(text, path, field="pr markdown path")


# ---------------------------------------------------------------------------
# GitHub PR publisher (v0.71.4 #223)
# ---------------------------------------------------------------------------

# owner/repo#<number> — owner + repo are GitHub name-safe (alnum + ._-),
# number is a positive integer.
_PR_TARGET_RE = re.compile(
    r"^([A-Za-z0-9][A-Za-z0-9._-]*)/([A-Za-z0-9][A-Za-z0-9._-]*)#([0-9]+)$"
)
_MAX_PR_BODY_BYTES = 60_000  # GitHub caps issue-comment bodies at 65_536 bytes

# Env keys passed through to the `gh` child — everything else (HF_TOKEN /
# OPENAI_API_KEY / ANTHROPIC_API_KEY / ...) is filtered out so a publish call
# never leaks unrelated secrets to the subprocess (v0.71.4 review HIGH fix,
# mirrors v0.44.0 _LLAMA_ENV_ALLOWLIST).
_GH_ENV_ALLOWLIST = frozenset(
    {
        "PATH", "HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA",
        "SYSTEMROOT", "SystemRoot", "TEMP", "TMP", "TMPDIR",
        "GH_TOKEN", "GITHUB_TOKEN", "GH_ENTERPRISE_TOKEN", "GITHUB_API_URL",
        "GH_HOST", "GH_CONFIG_DIR",
        "XDG_CONFIG_HOME", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY",
        "http_proxy", "https_proxy", "no_proxy",
    }
)


def parse_pr_target(target: object) -> Tuple[str, str, int]:
    """Parse ``"owner/repo#42"`` into ``(owner, repo, pr_number)``.

    Raises ``TypeError`` for non-strings and ``ValueError`` for any string
    that does not match the ``owner/repo#<positive-int>`` shape.
    """
    if isinstance(target, bool) or not isinstance(target, str):
        raise TypeError("target must be str")
    match = _PR_TARGET_RE.match(target.strip())
    if not match:
        raise ValueError(
            "target must be 'owner/repo#<number>' "
            "(e.g. MakazhanAlpamys/Soup#42)"
        )
    num = int(match.group(3))
    if num < 1:
        raise ValueError("PR number must be >= 1")
    return match.group(1), match.group(2), num


def resolve_github_token(env: Optional[Mapping[str, str]] = None) -> str:
    """Resolve a GitHub token from ``GITHUB_TOKEN`` / ``GH_TOKEN`` env.

    Mirrors the v0.29.0 HF token-resolution policy: env only, first-match
    wins, blank values treated as missing. Raises ``RuntimeError`` (a
    user-actionable error the CLI renders) when neither is set.
    """
    source = env if env is not None else os.environ
    for key in ("GITHUB_TOKEN", "GH_TOKEN"):
        val = source.get(key)
        if val and val.strip():
            return val.strip()
    raise RuntimeError(
        "no GitHub token found; set GITHUB_TOKEN (or GH_TOKEN) "
        "to publish a PR comment"
    )


def post_pr_comment(
    target: str,
    body: str,
    *,
    env: Optional[Mapping[str, str]] = None,
    runner: Optional[Callable[..., Any]] = None,
) -> str:
    """Post ``body`` as a comment on the GitHub PR named by ``target``.

    Uses ``gh api`` (no PyGithub dependency) with the body sent over
    JSON stdin so multiline / markdown content is never shell-interpolated.
    Auth resolves via ``GITHUB_TOKEN`` / ``GH_TOKEN`` (gh reads the same
    vars). Returns the created comment's ``html_url`` (best-effort, may be
    empty). ``runner`` is an injectable ``subprocess.run`` for testing.
    """
    owner, repo, num = parse_pr_target(target)
    if not isinstance(body, str):
        raise TypeError("body must be str")
    if not body.strip():
        raise ValueError("body must be non-empty")
    if "\x00" in body:
        raise ValueError("body must not contain null bytes")
    if len(body.encode("utf-8")) > _MAX_PR_BODY_BYTES:
        raise ValueError(
            f"body exceeds {_MAX_PR_BODY_BYTES} byte GitHub comment cap"
        )
    # Fail fast if no token before spawning the subprocess.
    token = resolve_github_token(env)

    import subprocess  # noqa: S404 — argv list mode, no shell

    argv = [
        "gh",
        "api",
        "--method",
        "POST",
        f"repos/{owner}/{repo}/issues/{num}/comments",
        "--input",
        "-",
    ]
    stdin = json.dumps({"body": body})
    # gh resolves GH_TOKEN / GITHUB_TOKEN from its own environment; thread
    # the resolved token through. When ``env`` is supplied (tests / explicit)
    # use it verbatim; otherwise build a MINIMAL env from an allowlist so we
    # never leak HF_TOKEN / OPENAI_API_KEY / ANTHROPIC_API_KEY etc. into the
    # gh child (mirrors the v0.44.0 _LLAMA_ENV_ALLOWLIST policy).
    if env is not None:
        base_env = dict(env)
    else:
        base_env = {
            k: v for k, v in os.environ.items() if k in _GH_ENV_ALLOWLIST
        }
    base_env.setdefault("GH_TOKEN", token)
    run = runner if runner is not None else subprocess.run
    try:
        result = run(
            argv,
            input=stdin,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            env=base_env,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "`gh` CLI not found; install GitHub CLI or drop --push "
            "and use --output to write the Markdown"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("gh api timed out posting the PR comment") from exc

    if getattr(result, "returncode", 1) != 0:
        stderr = (getattr(result, "stderr", "") or "").strip()[:512]
        raise RuntimeError(
            f"gh api failed (rc={result.returncode}): {stderr or 'no detail'}"
        )
    try:
        data = json.loads(getattr(result, "stdout", "") or "{}")
        return str(data.get("html_url") or "")
    except (ValueError, TypeError):
        return ""
