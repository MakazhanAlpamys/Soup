#!/usr/bin/env python3
"""#1017 — refuse a stale merge without writing to a contributor's branch.

A ``pull_request`` run pins ``refs/pull/N/merge`` at *run creation*.
``actions/checkout`` fetches that SHA; later movement of ``main`` is not
re-resolved. ``gh run rerun`` replays the frozen merge and cannot change the
answer. That is how #763 nearly merged a ``NameError`` on green marks, and how
sixteen PRs reported failures that were only true of an older ``main``.

This script is the check, not a rebase policy:

* Rebuild the merge with ``git merge-tree --write-tree`` (no committer
  identity, no commit) and fail on conflict or ruff F821. That is the #763
  class. A non-zero merge-tree exit that is not a conflict is an error, not
  a conflict — classifying every git failure as ``merge_conflict`` is how a
  missing identity painted every open PR red.
* Report lag above ``MAX_BEHIND`` as NEUTRAL, not FAILURE. At ~27 commits/day
  a 10-commit cap is ``strict: true`` with extra steps (~4 hours) and would
  block ~40% of open PRs pending a rebase. The broken-merge half is the part
  with no false positives.
* On a push to ``main``, post those results onto each open PR's head SHA via
  the Checks API so GitHub re-evaluates without a contributor push.

It does not flip ``required_status_checks.strict``. It does not replace the
13-job test matrix: a false-red *test* cell still needs a push.
"""

from __future__ import annotations

import argparse
import enum
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
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


class MergeStatus(enum.Enum):
    CLEAN = "clean"
    CONFLICT = "conflict"
    ERROR = "error"


@dataclass(frozen=True)
class MergeBuild:
    status: MergeStatus
    tree: str = ""
    detail: str = ""


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
    git_error: bool = False,
) -> Verdict:
    """Turn measurements into a Checks API conclusion.

    GitHub counts NEUTRAL as passing a required check. So:

    * ``fetch_failed`` and ``git_error`` are FAILURE — we could not evaluate,
      and a required check must not pass on "we did not look".
    * lag above ``max_behind`` with a clean merge is NEUTRAL — informational,
      not a rebase demand.
    * conflict and F821 are FAILURE. Those are the #763 class.
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
            Conclusion.FAILURE,
            "could not fetch PR head",
            "Could not fetch the listed head SHA for this PR. "
            "A required check must not pass on a skipped evaluation.\n\n" + remedy,
        )
    if git_error:
        return Verdict(
            Conclusion.FAILURE,
            "could not rebuild the live merge",
            "git failed before a merge result existed (not a conflict). "
            "Check that origin/main is fetched.\n\n" + remedy,
        )

    if merge_conflict:
        return Verdict(
            Conclusion.FAILURE,
            "live merge has conflicts",
            "Live merge against current main has conflicts.\n\n" + remedy,
        )
    if f821_hits:
        listed = "\n".join(f"  {hit}" for hit in f821_hits)
        return Verdict(
            Conclusion.FAILURE,
            "live merge has undefined names",
            f"ruff F821 on the live merge ({len(f821_hits)} hit(s)):\n{listed}"
            f"\n\n{remedy}",
        )
    if behind > max_behind:
        return Verdict(
            Conclusion.NEUTRAL,
            f"{behind} commits behind main (informational)",
            f"Merge-base is {behind} commits behind main (report threshold is "
            f"{max_behind}). The live merge is clean; this is not a failure.",
        )
    return Verdict(
        Conclusion.SUCCESS,
        f"{behind} commits behind main",
        f"Live merge is clean for F821 ({behind} commit(s) behind main).",
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


def write_merge_tree(repo: Path, base: str, head: str) -> MergeBuild:
    """Merge ``base`` into ``head`` as a tree. Needs no committer identity.

    ``git merge --no-ff`` on a GitHub runner has no user.name, exits 128, and
    used to be classified as a conflict. ``merge-tree --write-tree`` writes a
    tree and uses exit 0 / 1 / other for clean / conflict / error.
    """
    proc = subprocess.run(
        ["git", "merge-tree", "--write-tree", base, head],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
        env=_git_env(),
    )
    if proc.returncode == 0:
        tree = proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else ""
        if not tree:
            return MergeBuild(MergeStatus.ERROR, detail="merge-tree wrote no tree")
        return MergeBuild(MergeStatus.CLEAN, tree=tree)
    if proc.returncode == 1:
        return MergeBuild(
            MergeStatus.CONFLICT, detail=(proc.stderr or proc.stdout)[:500]
        )
    return MergeBuild(
        MergeStatus.ERROR,
        detail=(proc.stderr or proc.stdout or str(proc.returncode))[:500],
    )


def materialize_tree(repo: Path, tree: str, dest: Path) -> None:
    """Unpack ``tree`` into ``dest``. Never executes anything from ``dest``."""
    dest.mkdir(parents=True, exist_ok=True)
    archive = subprocess.run(
        ["git", "archive", "--format=tar", tree],
        cwd=repo,
        check=True,
        capture_output=True,
        env=_git_env(),
    )
    with tarfile.open(fileobj=io.BytesIO(archive.stdout), mode="r:") as bundle:
        try:
            bundle.extractall(dest, filter="data")
        except TypeError:
            bundle.extractall(dest)


def rebuild_merge(repo: Path, base: str, head: str, dest: Path) -> MergeBuild:
    """Materialise a live merge at ``dest``. Returns the merge-tree status."""
    build = write_merge_tree(repo, base, head)
    if build.status is MergeStatus.CLEAN:
        materialize_tree(repo, build.tree, dest)
    return build


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
            "--isolated",
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
    try:
        behind = commits_behind(repo, base, head)
    except subprocess.CalledProcessError:
        return decide(behind=0, max_behind=max_behind, git_error=True)
    tmp = tempfile.mkdtemp(prefix="soup-merge-freshness-")
    dest = Path(tmp) / "merge"
    try:
        build = rebuild_merge(repo, base, head, dest)
        if build.status is MergeStatus.CONFLICT:
            return decide(
                behind=behind, max_behind=max_behind, merge_conflict=True
            )
        if build.status is MergeStatus.ERROR:
            return decide(behind=behind, max_behind=max_behind, git_error=True)
        hits = run_f821(dest, runner=f821_runner)
        return decide(behind=behind, max_behind=max_behind, f821_hits=hits)
    finally:
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


def _scrub_token_env(env: dict[str, str]) -> dict[str, str]:
    """Drop credentials so a planted module cannot post check runs."""
    cleaned: dict[str, str] = {}
    for key, value in env.items():
        if key == "GITHUB_TOKEN" or key.startswith("GH_"):
            continue
        if key.startswith("ACTIONS_") and "TOKEN" in key:
            continue
        cleaned[key] = value
    return cleaned


def _ruff_invocation(argv: list[str], *, tree: Path) -> tuple[list[str], Path]:
    """Resolve ruff before cwd changes, and isolate it from the tree.

    ``python -m ruff`` with cwd=the PR tree puts that tree first on
    ``sys.path``, so a top-level ``ruff/`` package runs instead of ruff,
    with ``GITHUB_TOKEN`` still in the environment. Prefer the binary from
    ``PATH``. If it is missing (a user-site install with no Scripts dir),
    run ``python -m ruff`` from this trusted directory with absolute paths
    so the tree is never ``sys.path[0]``. ``python -I -m ruff`` would also
    isolate cwd, but it ignores user site and cannot see that install.
    ``--isolated`` so the tree's ``[tool.ruff]`` cannot ignore F821.
    """
    rest = list(argv[1:])
    if "--isolated" not in rest:
        if rest[:1] == ["check"]:
            rest = ["check", "--isolated", *rest[1:]]
        else:
            rest = ["--isolated", *rest]
    resolved = shutil.which("ruff")
    if resolved:
        return [resolved, *rest], tree
    abs_rest: list[str] = []
    for item in rest:
        candidate = tree / item
        if item.startswith("-") or item == "check" or not candidate.exists():
            abs_rest.append(item)
        else:
            abs_rest.append(str(candidate.resolve()))
    cmd = [sys.executable]
    if sys.version_info >= (3, 11):
        cmd.append("-P")
    cmd.extend(["-m", "ruff", *abs_rest])
    return cmd, Path(__file__).resolve().parent


def _run_ruff(argv: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    cmd, run_cwd = _ruff_invocation(argv, tree=cwd)
    env = _scrub_token_env(_git_env())
    env["PYTHONSAFEPATH"] = "1"
    return subprocess.run(
        cmd,
        cwd=run_cwd,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def _fetch_commit(repo: Path, sha: str) -> str | None:
    """Fetch the listed head SHA, not ``pull/N/head`` which can move."""
    ref = f"refs/pr-sha/{sha}"
    fetch = subprocess.run(
        ["git", "fetch", "--no-tags", "origin", f"{sha}:{ref}"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
        env=_git_env(),
    )
    if fetch.returncode != 0:
        return None
    got = _git(repo, "rev-parse", ref).stdout.strip()
    if got != sha:
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


def reeval_open_prs(
    *,
    repo_root: Path,
    base: str,
    token: str,
    repo_slug: str,
    api_url: str = "https://api.github.com",
    max_behind: int = MAX_BEHIND,
    opener: Callable[..., Any] | None = None,
    fetch_commit: Callable[[Path, str], str | None] | None = None,
    evaluate: Callable[..., Verdict] | None = None,
) -> int:
    """Evaluate every open PR and post ``merge-freshness`` on its listed SHA.

    Continues after a Checks API HTTPError so one 403 does not abandon the rest.
    """
    heads = list_open_pr_heads(
        repo=repo_slug, token=token, api_url=api_url, opener=opener
    )
    print(f"reevaluating {len(heads)} open PR(s) against {base}")
    fetch = fetch_commit or _fetch_commit
    measure = evaluate or evaluate_refs
    posted = 0
    for number, sha in heads:
        ref = fetch(repo_root, sha)
        if ref is None:
            verdict = decide(behind=0, fetch_failed=True)
        else:
            verdict = measure(
                repo_root, base, ref, max_behind=max_behind
            )
        print(f"#{number} {sha[:12]} {verdict.conclusion.value}: {verdict.title}")
        try:
            post_check_run(
                repo=repo_slug,
                sha=sha,
                verdict=verdict,
                token=token,
                api_url=api_url,
                opener=opener,
            )
            posted += 1
        except urllib.error.HTTPError as exc:
            print(
                f"#{number} check run failed ({exc.code}): {exc.reason}",
                file=sys.stderr,
            )
    print(f"posted {posted} check run(s)")
    return 0


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
    return reeval_open_prs(
        repo_root=Path(args.repo_root).resolve(),
        base=args.base,
        token=token,
        repo_slug=repo_slug,
        api_url=api_url,
        max_behind=args.max_behind,
    )


def _cmd_decide(args: argparse.Namespace) -> int:
    hits = tuple(args.f821_hit) if args.f821_hit else ()
    verdict = decide(
        behind=args.behind,
        max_behind=args.max_behind,
        f821_hits=hits,
        merge_conflict=args.merge_conflict,
        fetch_failed=args.fetch_failed,
        git_error=args.git_error,
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
    decide_cmd.add_argument("--git-error", action="store_true")
    decide_cmd.set_defaults(func=_cmd_decide)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
