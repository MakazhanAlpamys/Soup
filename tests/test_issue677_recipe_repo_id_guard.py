"""#677 — nothing checks that a shipped recipe's model id actually resolves.

#661 found two that did not, both released. `validate-recipes` stayed green
because a nonexistent repo id is a perfectly well-formed string.

Three constraints from the issue, each of which makes the guard worse than
nothing if got wrong, and each pinned below:

1. **A bare 401 is not evidence.** An invented repo returns the same 401 as a
   real private one — the mistake I made in #661 before it was worth filing.
   The authoritative signal is `huggingface_hub`'s `RepositoryNotFoundError`
   against an explicitly unauthenticated client.
2. **Missing and gated are different, and only one is a bug.** A guard that
   reports every gated Llama repo forever is one people mute.
3. **A transient failure is not a missing repo.** Reported as one, the job
   becomes noise and gets ignored — the #404 lesson.

No test here touches the network: the Hub client is a fake, so the
classification logic is pinned without depending on Hub availability.
"""

from __future__ import annotations


class _NotFoundError(Exception):
    """Stand-in for huggingface_hub.errors.RepositoryNotFoundError."""


class _Info:
    def __init__(self, gated=False, private=False):
        self.gated = gated
        self.private = private


class _FakeApi:
    """A Hub client with scripted answers. Records calls so a test can assert
    the checker retried rather than classified on the first blip."""

    def __init__(self, answers):
        self._answers = answers
        self.calls = []

    def model_info(self, repo_id, **kwargs):
        self.calls.append(repo_id)
        answer = self._answers[repo_id]
        if isinstance(answer, list):
            answer = answer.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


class TestClassification:
    def test_a_real_public_repo_is_exists(self):
        from scripts.check_recipe_repo_ids import Status, classify_repo

        api = _FakeApi({"org/model": _Info()})
        assert classify_repo(api, "org/model", not_found=_NotFoundError).status is Status.EXISTS

    def test_a_gated_repo_is_gated_not_missing(self):
        """The direction that would make the guard permanently noisy."""
        from scripts.check_recipe_repo_ids import Status, classify_repo

        api = _FakeApi({"meta-llama/Llama-3.1-8B-Instruct": _Info(gated=True)})
        result = classify_repo(api, "meta-llama/Llama-3.1-8B-Instruct", not_found=_NotFoundError)

        assert result.status is Status.GATED
        assert result.status is not Status.MISSING

    def test_repository_not_found_is_missing(self):
        from scripts.check_recipe_repo_ids import Status, classify_repo

        api = _FakeApi({"org/gone": _NotFoundError("404")})
        assert classify_repo(api, "org/gone", not_found=_NotFoundError).status is Status.MISSING

    def test_a_transient_failure_is_unverified_not_missing(self):
        """Constraint 3. A timeout reported as "missing" is how this becomes noise."""
        from scripts.check_recipe_repo_ids import Status, classify_repo

        api = _FakeApi({"org/flaky": TimeoutError("read timed out")})
        result = classify_repo(api, "org/flaky", not_found=_NotFoundError, attempts=2, backoff=0)

        assert result.status is Status.UNVERIFIED
        assert result.status is not Status.MISSING

    def test_a_transient_failure_is_retried_before_giving_up(self):
        """One bad minute must not be one bad report."""
        from scripts.check_recipe_repo_ids import Status, classify_repo

        api = _FakeApi({"org/blip": [TimeoutError("blip"), _Info()]})
        result = classify_repo(api, "org/blip", not_found=_NotFoundError, attempts=3, backoff=0)

        assert result.status is Status.EXISTS
        assert len(api.calls) == 2, "must retry a transient failure, not classify it"

    def test_not_found_is_not_retried(self):
        """A 404 is a definite answer; retrying it wastes the whole budget."""
        from scripts.check_recipe_repo_ids import classify_repo

        api = _FakeApi({"org/gone": _NotFoundError("404")})
        classify_repo(api, "org/gone", not_found=_NotFoundError, attempts=3, backoff=0)

        assert len(api.calls) == 1

    def test_negative_control_an_invented_id_is_missing(self):
        """The acceptance criterion: the checker must be provably able to fail."""
        from scripts.check_recipe_repo_ids import Status, classify_repo

        api = _FakeApi({"zz-invented/does-not-exist-xyz": _NotFoundError("404")})
        result = classify_repo(api, "zz-invented/does-not-exist-xyz", not_found=_NotFoundError)

        assert result.status is Status.MISSING


class TestBothSurfacesAreCovered:
    def test_every_recipe_contributes_meta_and_yaml_ids(self):
        from scripts.check_recipe_repo_ids import collect_recipe_repo_ids

        surfaces = collect_recipe_repo_ids()

        assert len(surfaces) > 100, "the whole catalog, not a sample"
        for name, pair in surfaces.items():
            assert pair.meta_model, f"{name} has no RecipeMeta.model"
            assert pair.yaml_base, f"{name} has no YAML base:"

    def test_a_wrong_meta_model_is_caught_independently_of_the_yaml(self, monkeypatch):
        """#666's mutation testing showed the snapshot guard sees only the YAML
        half, so a wrong `RecipeMeta.model` slips past everything else."""
        from scripts.check_recipe_repo_ids import Status, check_surfaces

        api = _FakeApi({"good/real": _Info(), "bad/invented": _NotFoundError("404")})
        report = check_surfaces(
            {"r1": ("bad/invented", "good/real")}, api=api, not_found=_NotFoundError
        )

        assert report["r1"].meta.status is Status.MISSING
        assert report["r1"].yaml.status is Status.EXISTS

    def test_a_wrong_yaml_base_is_caught_independently_of_the_meta(self):
        from scripts.check_recipe_repo_ids import Status, check_surfaces

        api = _FakeApi({"good/real": _Info(), "bad/invented": _NotFoundError("404")})
        report = check_surfaces(
            {"r1": ("good/real", "bad/invented")}, api=api, not_found=_NotFoundError
        )

        assert report["r1"].meta.status is Status.EXISTS
        assert report["r1"].yaml.status is Status.MISSING


class TestReporting:
    def test_only_missing_is_reported(self):
        from scripts.check_recipe_repo_ids import check_surfaces, missing_only

        api = _FakeApi({
            "ok/public": _Info(),
            "ok/gated": _Info(gated=True),
            "bad/gone": _NotFoundError("404"),
        })
        report = check_surfaces(
            {"a": ("ok/public", "ok/public"),
             "b": ("ok/gated", "ok/gated"),
             "c": ("bad/gone", "bad/gone")},
            api=api, not_found=_NotFoundError,
        )

        assert sorted(missing_only(report)) == ["c"], (
            "gated and existing recipes must not be reported"
        )

    def test_unverified_is_reported_separately_from_missing(self):
        from scripts.check_recipe_repo_ids import check_surfaces, missing_only, unverified_only

        api = _FakeApi({"ok/public": _Info(), "flaky/one": TimeoutError("t")})
        report = check_surfaces(
            {"a": ("ok/public", "ok/public"), "b": ("flaky/one", "flaky/one")},
            api=api, not_found=_NotFoundError, attempts=1, backoff=0,
        )

        assert missing_only(report) == [], "a timeout is not a missing repo"
        assert unverified_only(report) == ["b"]


class TestTheWorkflow:
    """Constraint 3: it must not run in the PR matrix."""

    def _workflow(self):
        from pathlib import Path

        path = Path(__file__).parents[1] / ".github/workflows/recipe-repo-ids.yml"
        assert path.is_file(), "the scheduled workflow is missing"
        return path.read_text(encoding="utf-8")

    def test_it_is_scheduled(self):
        assert "schedule:" in self._workflow()
        assert "cron:" in self._workflow()

    def test_it_can_be_run_by_hand(self):
        assert "workflow_dispatch:" in self._workflow()

    def test_it_does_not_run_on_pull_requests(self):
        """Asserted rather than trusted: 163 network calls per PR would make
        every unrelated change depend on Hub availability."""
        text = self._workflow()
        assert "pull_request" not in text, (
            "this job must never be a PR gate — a docs PR going red because the "
            "Hub had a bad minute trains people to re-run rather than to read"
        )
