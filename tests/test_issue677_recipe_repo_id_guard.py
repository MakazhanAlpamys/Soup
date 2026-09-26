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

import pytest


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

    def test_a_gated_repo_that_raises_is_still_gated(self):
        """`GatedRepoError` SUBCLASSES `RepositoryNotFoundError`, so
        `except not_found` swallows it and every gated repo classifies MISSING
        unless the gated handler comes first.

        The other gated test pins the path where `model_info` *returns* with
        `info.gated` truthy. A live sweep only exercises that path today,
        because the Hub currently serves gated metadata anonymously -- which is
        why this went unnoticed in a run that otherwise looked complete.
        """
        from scripts.check_recipe_repo_ids import Status, classify_repo

        class _GatedError(_NotFoundError):      # the real subclass relationship
            pass

        api = _FakeApi({"meta-llama/Llama-3.1-8B-Instruct": _GatedError("gated")})
        result = classify_repo(
            api, "meta-llama/Llama-3.1-8B-Instruct",
            not_found=_NotFoundError, gated=_GatedError, attempts=1, backoff=0,
        )

        assert result.status is Status.GATED, (
            "a raising gated repo classified MISSING: the gated handler must "
            "precede not_found, since GatedRepoError subclasses it"
        )

    def test_the_real_hub_classes_are_still_subclassed_that_way(self):
        """Pin the upstream fact the ordering depends on, against the installed
        library, so an un-nesting upstream says so rather than going stale."""
        hub_errors = pytest.importorskip("huggingface_hub.errors")

        assert issubclass(hub_errors.GatedRepoError, hub_errors.RepositoryNotFoundError), (
            "GatedRepoError no longer subclasses RepositoryNotFoundError; the "
            "except-ordering in classify_repo was written for that relationship"
        )

    def test_negative_control_an_invented_id_is_missing(self):
        """The acceptance criterion: the checker must be provably able to fail.

        Independent of `test_repository_not_found_is_missing` rather than a
        restatement of it -- this drives the id through the whole pipeline
        (collector shape, `check_surfaces`, `missing_only`), so a checker that
        classified correctly but reported nothing would still fail here.
        """
        from scripts.check_recipe_repo_ids import check_surfaces, missing_only

        api = _FakeApi({
            "real/model": _Info(),
            "zz-invented/does-not-exist-xyz": _NotFoundError("404"),
        })
        report = check_surfaces(
            {"fine": ("real/model", "real/model"),
             "broken": ("zz-invented/does-not-exist-xyz",
                        "zz-invented/does-not-exist-xyz")},
            api=api, not_found=_NotFoundError,
        )

        assert missing_only(report) == ["broken"]


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
    """Constraint 3: it must not run in the PR matrix.

    Parses the triggers rather than scanning the file for the substring, which
    fired on correct code: adding the comment "Never add a pull_request trigger
    here." failed the old version. A guard that goes red on a correct change is
    one that gets deleted — the #404 lesson. Follows the in-repo precedent at
    tests/test_recipes_v031.py, which reads `parsed.get(True) or parsed.get("on")`
    because YAML parses a bare `on:` key as the boolean True.
    """

    def _triggers(self):
        from pathlib import Path

        import yaml

        path = Path(__file__).parents[1] / ".github/workflows/recipe-repo-ids.yml"
        assert path.is_file(), "the scheduled workflow is missing"
        parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
        triggers = parsed.get(True) or parsed.get("on")
        assert isinstance(triggers, dict), f"unexpected trigger shape: {triggers!r}"
        return triggers

    def test_it_is_scheduled(self):
        triggers = self._triggers()
        assert "schedule" in triggers
        assert any("cron" in entry for entry in triggers["schedule"])

    def test_it_can_be_run_by_hand(self):
        assert "workflow_dispatch" in self._triggers()

    def test_it_does_not_run_on_pull_requests(self):
        """163 network calls per PR would make every unrelated change depend on
        Hub availability. `pull_request_target` and `push` are checked too —
        either would put this on the critical path just as effectively."""
        triggers = self._triggers()
        for forbidden in ("pull_request", "pull_request_target", "push"):
            assert forbidden not in triggers, (
                f"{forbidden!r} would put this sweep on the critical path of "
                "unrelated changes"
            )

    def test_it_asks_for_no_write_permissions(self):
        from pathlib import Path

        import yaml

        path = Path(__file__).parents[1] / ".github/workflows/recipe-repo-ids.yml"
        perms = yaml.safe_load(path.read_text(encoding="utf-8")).get("permissions", {})
        assert perms == {"contents": "read"}, (
            f"a read-only check must not hold write scopes: {perms!r}"
        )


class TestTheAnonymousClient:
    """`_anonymous_api()` — untested, and it is the guard on the guard."""

    @pytest.mark.parametrize(
        "var", ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"]
    )
    def test_it_refuses_when_a_token_is_in_scope(self, monkeypatch, var):
        """A token resolves gated repos as EXISTS, so the guard goes quiet
        exactly where a wrong id is most likely to hide. Refusing beats
        returning a result that means something different from what it says."""
        from scripts.check_recipe_repo_ids import _anonymous_api

        monkeypatch.setenv(var, "hf_notarealtoken")
        with pytest.raises(SystemExit, match=var):
            _anonymous_api()

    def test_the_client_it_returns_has_authentication_off(self, monkeypatch):
        """`token=False` also neutralises a cached ~/.cache/huggingface/token,
        which the env-var check alone would miss."""
        pytest.importorskip("huggingface_hub")
        from scripts.check_recipe_repo_ids import _anonymous_api

        for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
            monkeypatch.delenv(var, raising=False)

        assert _anonymous_api().token is False


class TestTheExitCode:
    """Requirement 7 — "fails the job" — and the property that a Hub outage
    must NOT fail it. Both were asserted only in a code comment."""

    def _run_main(self, monkeypatch, answers, surfaces):
        import scripts.check_recipe_repo_ids as mod

        monkeypatch.setattr(mod, "_anonymous_api", lambda: _FakeApi(answers))
        monkeypatch.setattr(mod, "collect_recipe_repo_ids", lambda: surfaces)
        monkeypatch.setattr(mod, "_DEFAULT_ATTEMPTS", 1)
        monkeypatch.setattr(mod, "_DEFAULT_BACKOFF", 0)

        class _GatedError(_NotFoundError):
            pass

        return mod.main(not_found=_NotFoundError, gated=_GatedError)

    def test_main_returns_1_when_a_repo_is_missing(self, monkeypatch):
        code = self._run_main(
            monkeypatch,
            {"bad/gone": _NotFoundError("404")},
            {"r": ("bad/gone", "bad/gone")},
        )
        assert code == 1, "a missing repo must fail the scheduled job"

    def test_main_returns_0_when_everything_resolves(self, monkeypatch):
        code = self._run_main(
            monkeypatch, {"ok/real": _Info()}, {"r": ("ok/real", "ok/real")}
        )
        assert code == 0

    def test_an_unverified_only_run_does_not_fail(self, monkeypatch):
        """The load-bearing property. If a timeout fails the job, a bad minute
        at the Hub reads as a broken catalog and the guard gets muted."""
        code = self._run_main(
            monkeypatch,
            {"flaky/one": TimeoutError("read timed out")},
            {"r": ("flaky/one", "flaky/one")},
        )
        assert code == 0, "a Hub outage must not read as a broken catalog"


class TestTheReport:
    """`format_report` — every line of it was uncovered."""

    def _report(self):
        from scripts.check_recipe_repo_ids import check_surfaces

        api = _FakeApi({
            "ok/real": _Info(),
            "bad/gone": _NotFoundError("404"),
            "flaky/one": TimeoutError("read timed out"),
        })
        return check_surfaces(
            {"good-recipe": ("ok/real", "ok/real"),
             "broken-recipe": ("bad/gone", "ok/real"),
             "flaky-recipe": ("flaky/one", "ok/real")},
            api=api, not_found=_NotFoundError, attempts=1, backoff=0,
        )

    def test_it_names_the_broken_recipe_and_the_surface(self):
        from scripts.check_recipe_repo_ids import format_report

        text = format_report(self._report())
        assert "broken-recipe" in text, "a report that omits the name is unusable"
        assert "RecipeMeta.model" in text, "the reader must know WHICH surface"
        assert "bad/gone" in text

    def test_it_does_not_name_a_healthy_recipe_as_missing(self):
        """Reject-everything control."""
        from scripts.check_recipe_repo_ids import format_report

        missing_section = format_report(self._report()).split("COULD NOT CHECK")[0]
        assert "good-recipe" not in missing_section

    def test_it_reports_could_not_check_separately(self):
        from scripts.check_recipe_repo_ids import format_report

        text = format_report(self._report())
        assert "COULD NOT CHECK" in text
        assert "flaky-recipe" in text.split("COULD NOT CHECK")[1]


class TestTheProductionCollectorPath:
    """Every other test feeds plain tuples, so `check_surfaces`' real
    `SurfacePair` branch — the one production uses — was uncovered."""

    def test_real_surface_pairs_flow_through_check_surfaces(self):
        from scripts.check_recipe_repo_ids import (
            Status,
            check_surfaces,
            collect_recipe_repo_ids,
        )

        surfaces = collect_recipe_repo_ids()
        ids = {p.meta_model for p in surfaces.values()} | {p.yaml_base for p in surfaces.values()}
        api = _FakeApi({repo_id: _Info() for repo_id in ids})

        report = check_surfaces(surfaces, api=api, not_found=_NotFoundError)

        assert len(report) == len(surfaces)
        assert all(r.meta.status is Status.EXISTS for r in report.values())

    def test_a_divergent_yaml_base_is_actually_read(self, monkeypatch):
        """Kills the mutant that mirrors `meta.model` into `yaml_base`.

        Every shipped recipe currently has `meta.model == yaml base`, so that
        mutation changes no output today — but it would mean the collector had
        stopped reading the YAML at all, and divergence is the exact case the
        two-surface design exists for (#661 found it on the meta surface).
        """
        from scripts.check_recipe_repo_ids import collect_recipe_repo_ids

        class _Meta:
            model = "org/from-meta"
            yaml_str = "base: org/from-yaml\ntask: sft\n"

        import soup_cli.recipes.catalog as catalog

        monkeypatch.setattr(catalog, "RECIPES", {"synthetic": _Meta()})
        pair = collect_recipe_repo_ids()["synthetic"]

        assert pair.meta_model == "org/from-meta"
        assert pair.yaml_base == "org/from-yaml", (
            "the collector mirrored RecipeMeta.model instead of parsing the YAML"
        )
