#!/usr/bin/env python3
"""#1017 — refuse a stale merge without writing to a contributor's branch.

A ``pull_request`` run pins ``refs/pull/N/merge`` at *run creation*.
``actions/checkout`` fetches that SHA; later movement of ``main`` is not
re-resolved. ``gh run rerun`` replays the frozen merge and cannot change the
answer. That is how #763 nearly merged a ``NameError`` on green marks, and how
sixteen PRs reported failures that were only true of an older ``main``.

This script is the check, not a rebase policy:

* Fail when the PR head's merge-base is more than ``MAX_BEHIND`` commits
  behind the live base (option 2 on #1017). That is the #853 class: a new
  gate on ``main`` that the PR never ran.
* Rebuild the merge locally (no ``gh pr update-branch``) and run ruff F821
  on it. That is the #763 class: a 1-commit deletion that any ``N > 0``
  lag window would still allow.
* On a push to ``main``, post those results onto each open PR's head SHA via
  the Checks API so GitHub re-evaluates without a contributor push.

It does not flip ``required_status_checks.strict`` (every merge would then
force a rebase, and strict still does not refresh a frozen merge SHA). It
does not replace the 13-job test matrix: a false-red *test* cell still needs
a push. The CONTRIBUTING note says so.
"""

from __future__ import annotations

import argparse
import enum
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

MAX_BEHIND = 10
CHECK_NAME = "merge-freshness"
RUFF_SELECT = "F821"
RUFF_TARGETS = ("src/soup_cli", "scripts", "tests", "benchmarks")
_API_VERSION = "2022-11-28"


class Conclusion(enum.Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    NEUTRAL = "neutral"


@dataclass(frozen=True)
class Verdict:
    conclusion: Conclusion
    title: str
    summary: str

    @property
    def exit_code(self) -> int:
        return 1 if self.conclusion is Conclusion.FAILURE else 0


def decide(
    *,
    behind: int,
    max_behind: int = MAX_BEHIND,
    f821_hits: tuple[str, ...] = (),
    merge_conflict: bool = False,
    fetch_failed: bool = False,
) -> Verdict:
    """Turn measurements into a Checks API conclusion.

    ``fetch_failed`` is NEUTRAL, not FAILURE: a missing ``pull/N/head`` ref
    must not paint every other PR red, and must not fail the ``main`` reeval
    job. The PR-event gate passes ``fetch_failed=False`` because that job
    already has the head checked out.
    """
    if behind < 0:
        raise ValueError(f"behind must be >= 0, got {behind}")
    if max_behind < 0:
        raise ValueError(f"max_behind must be >= 0, got {max_behind}")

    remedy = (
        "Merge or rebase onto current origin/main and push. "
        "`gh run rerun` will not refresh a frozen merge SHA."
    )

    if fetch_failed:
        return Verdict(
            Conclusion.NEUTRAL,
            "could not fetch PR head",
            "Skipped: GitHub did not expose pull/N/head for this PR.",
        )

    reasons: list[str] = []
    if merge_conflict:
        reasons.append("Live merge against current main has conflicts.")
    if f821_hits:
        listed = "\n".join(f"  {hit}" for hit in f821_hits)
        reasons.append(
            f"ruff F821 on the live merge ({len(f821_hits)} hit(s)):\n{listed}"
        )
    if behind > max_behind:
        reasons.append(
            f"Merge-base is {behind} commits behind main "
            f"(cap is {max_behind})."
        )

    if reasons:
        body = "\n".join(reasons) + "\n\n" + remedy
        if merge_conflict:
            title = "live merge has conflicts"
        elif f821_hits:
            title = "live merge has undefined names"
        else:
            title = f"{behind} commits behind main"
        return Verdict(Conclusion.FAILURE, title, body)

    return Verdict(
        Conclusion.SUCCESS,
        f"{behind} commits behind main (cap {max_behind})",
        f"Live merge is clean for F821 and within the {max_behind}-commit cap.",
    )


def commits_behind(repo: Path, base: str, head: str) -> int:
    """How many commits ``base`` has that are not ancestors of ``head``."""
    merge_base = _git(repo, "merge-base", base, head).stdout.strip()
    if not merge_base:
        raise RuntimeError(f"no merge-base between {base} and {head}")
    count = _git(
        repo, "rev-list", "--count", f"{merge_base}..{base}"
    ).stdout.strip()
    return int(count)


def rebuild_merge(repo: Path, base: str, head: str, dest: Path) -> bool:
    """Materialise ``base`` merged into ``head`` at ``dest``.

    Returns True on a clean merge, False on conflict. The caller runs ruff
    against ``dest``; this function never executes anything from ``dest``.
    """
    dest = dest.resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    _git(repo, "worktree", "add", "--detach", str(dest), head)
    merge = subprocess.run(
        ["git", "merge", "--no-edit", "--no-ff", base],
        cwd=dest,
        check=False,
        capture_output=True,
        text=True,
        env=_git_env(),
    )
    return merge.returncode == 0


def run_f821(
    tree: Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess] | None = None,
) -> tuple[str, ...]:
    """Return concise F821 hit lines from ``tree``. Empty if clean."""
    targets = [name for name in RUFF_TARGETS if (tree / name).exists()]
    if not targets:
        return ()
    run = runner or _run_ruff
    proc = run(
        [
            "ruff",
            "check",
            "--select",
            RUFF_SELECT,
            "--output-format",
            "concise",
            *targets,
        ],
        cwd=tree,
    )
    return tuple(
        line.strip()
        for line in (proc.stdout or "").splitlines()
        if "F821" in line
    )


def post_check_run(
    *,
    repo: str,
    sha: str,
    verdict: Verdict,
    token: str,
    api_url: str = "https://api.github.com",
    opener: Callable[..., Any] | None = None,
) -> None:
    """Create a completed check run on ``sha``. Inject ``opener`` in tests."""
    payload = json.dumps(
        {
            "name": CHECK_NAME,
            "head_sha": sha,
            "status": "completed",
            "conclusion": verdict.conclusion.value,
            "output": {"title": verdict.title, "summary": verdict.summary},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{api_url.rstrip('/')}/repos/{repo}/check-runs",
        data=payload,
        method="POST",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": _API_VERSION,
            "Content-Type": "application/json",
        },
    )
    open_url = opener or urllib.request.urlopen
    with open_url(request, timeout=30) as response:
        response.read()


def list_open_pr_heads(
    *,
    repo: str,
    token: str,
    api_url: str = "https://api.github.com",
    opener: Callable[..., Any] | None = None,
    per_page: int = 100,
    max_pages: int = 5,
) -> list[tuple[int, str]]:
    """``(number, head_sha)`` for open PRs targeting ``main``."""
    open_url = opener or urllib.request.urlopen
    found: list[tuple[int, str]] = []
    for page in range(1, max_pages + 1):
        request = urllib.request.Request(
            f"{api_url.rstrip('/')}/repos/{repo}/pulls"
            f"?state=open&base=main&per_page={per_page}&page={page}",
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {token}",
                "X-GitHub-Api-Version": _API_VERSION,
            },
        )
        with open_url(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not payload:
            break
        for item in payload:
            found.append((int(item["number"]), str(item["head"]["sha"])))
        if len(payload) < per_page:
            break
    return found


def evaluate_refs(
    repo: Path,
    base: str,
    head: str,
    *,
    max_behind: int = MAX_BEHIND,
    f821_runner: Callable[..., subprocess.CompletedProcess] | None = None,
) -> Verdict:
    """Measure one head against a live base and return the verdict."""
    behind = commits_behind(repo, base, head)
    tmp = tempfile.mkdtemp(prefix="soup-merge-freshness-")
    dest = Path(tmp) / "merge"
    try:
        clean = rebuild_merge(repo, base, head, dest)
        if not clean:
            return decide(
                behind=behind, max_behind=max_behind, merge_conflict=True
            )
        hits = run_f821(dest, runner=f821_runner)
        return decide(behind=behind, max_behind=max_behind, f821_hits=hits)
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(dest)],
            cwd=repo,
            check=False,
            capture_output=True,
            text=True,
            env=_git_env(),
        )
        shutil.rmtree(tmp, ignore_errors=True)


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        env=_git_env(),
    )


def _git_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("GIT_TERMINAL_PROMPT", "0")
    env.setdefault("GIT_MERGE_AUTOEDIT", "no")
    return env


def _run_ruff(argv: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    # ``python -m ruff`` survives a venv whose Scripts dir is not on PATH.
    cmd = [sys.executable, "-m", "ruff", *argv[1:]]
    return subprocess.run(
        cmd, cwd=cwd, check=False, capture_output=True, text=True
    )


def _fetch_pr_head(repo: Path, number: int) -> str | None:
    ref = f"refs/pr/{number}"
    fetch = subprocess.run(
        ["git", "fetch", "--no-tags", "origin", f"pull/{number}/head:{ref}"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
        env=_git_env(),
    )
    if fetch.returncode != 0:
        return None
    return ref


def _print_verdict(verdict: Verdict) -> None:
    print(f"{verdict.conclusion.value}: {verdict.title}")
    print(verdict.summary)


def _maybe_post(verdict: Verdict, sha: str | None) -> None:
    if not sha:
        return
    token = os.environ.get("GITHUB_TOKEN", "")
    repo_name = os.environ.get("GITHUB_REPOSITORY", "")
    if not token or not repo_name:
        print("skipping check run: GITHUB_TOKEN or GITHUB_REPOSITORY unset")
        return
    api_url = os.environ.get("GITHUB_API_URL", "https://api.github.com")
    try:
        post_check_run(
            repo=repo_name, sha=sha, verdict=verdict, token=token, api_url=api_url
        )
    except urllib.error.HTTPError as exc:
        # Fork PR tokens are read-only. The job status still shows; the
        # push-to-main reeval posts the named check with a write token.
        print(f"check run not posted ({exc.code}): {exc.reason}")


def _cmd_gate(args: argparse.Namespace) -> int:
    root = Path(args.repo_root).resolve()
    verdict = evaluate_refs(
        root, args.base, args.head, max_behind=args.max_behind
    )
    _print_verdict(verdict)
    if args.post_check:
        _maybe_post(verdict, args.head_sha)
    return verdict.exit_code


def _cmd_reeval(args: argparse.Namespace) -> int:
    token = os.environ.get("GITHUB_TOKEN", "")
    repo_slug = os.environ.get("GITHUB_REPOSITORY", "")
    if not token or not repo_slug:
        print(
            "GITHUB_TOKEN and GITHUB_REPOSITORY are required for reeval",
            file=sys.stderr,
        )
        return 1
    api_url = os.environ.get("GITHUB_API_URL", "https://api.github.com")
    root = Path(args.repo_root).resolve()
    heads = list_open_pr_heads(repo=repo_slug, token=token, api_url=api_url)
    print(f"reevaluating {len(heads)} open PR(s) against {args.base}")
    posted = 0
    for number, sha in heads:
        ref = _fetch_pr_head(root, number)
        if ref is None:
            verdict = decide(behind=0, fetch_failed=True)
        else:
            verdict = evaluate_refs(
                root, args.base, ref, max_behind=args.max_behind
            )
        print(f"#{number} {sha[:12]} {verdict.conclusion.value}: {verdict.title}")
        try:
            post_check_run(
                repo=repo_slug,
                sha=sha,
                verdict=verdict,
                token=token,
                api_url=api_url,
            )
            posted += 1
        except urllib.error.HTTPError as exc:
            print(
                f"#{number} check run failed ({exc.code}): {exc.reason}",
                file=sys.stderr,
            )
            return 1
    print(f"posted {posted} check run(s)")
    return 0


def _cmd_decide(args: argparse.Namespace) -> int:
    hits = tuple(args.f821_hit) if args.f821_hit else ()
    verdict = decide(
        behind=args.behind,
        max_behind=args.max_behind,
        f821_hits=hits,
        merge_conflict=args.merge_conflict,
        fetch_failed=args.fetch_failed,
    )
    _print_verdict(verdict)
    return verdict.exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-behind", type=int, default=MAX_BEHIND)
    parser.add_argument("--repo-root", default=".")
    sub = parser.add_subparsers(dest="cmd", required=True)

    gate = sub.add_parser(
        "gate", help="evaluate the current checkout against a live base"
    )
    gate.add_argument("--base", required=True)
    gate.add_argument("--head", required=True)
    gate.add_argument("--head-sha", default="")
    gate.add_argument("--post-check", action="store_true")
    gate.set_defaults(func=_cmd_gate)

    reeval = sub.add_parser(
        "reeval", help="post checks for every open PR targeting main"
    )
    reeval.add_argument("--base", default="origin/main")
    reeval.set_defaults(func=_cmd_reeval)

    decide_cmd = sub.add_parser(
        "decide", help="verdict from already-measured inputs"
    )
    decide_cmd.add_argument("--behind", type=int, required=True)
    decide_cmd.add_argument("--f821-hit", action="append", default=[])
    decide_cmd.add_argument("--merge-conflict", action="store_true")
    decide_cmd.add_argument("--fetch-failed", action="store_true")
    decide_cmd.set_defaults(func=_cmd_decide)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
