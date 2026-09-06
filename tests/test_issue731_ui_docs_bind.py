"""Tests for Issue #731: FastAPI docs routes on a non-loopback `soup ui` bind.

`soup ui --public` serves `/openapi.json`, `/docs`, `/docs/oauth2-redirect` and
`/redoc` without a token, so anyone on the LAN can enumerate every route,
parameter and schema. No values leak -- every endpoint the schema describes
answers 401 -- so this is reconnaissance rather than disclosure, which is why
the fix removes the routes on a non-loopback bind rather than gating them.

Gating was rejected deliberately: `/docs` is a browser navigation and Swagger
cannot attach a Bearer header to it, so `Depends(_verify_token)` would make the
page unusable for the developer while `/openapi.json` stayed reachable to
anything that speaks curl. Removing the routes (`openapi_url=None`) leaves no
handler to reach at all, and costs the loopback developer nothing.
"""

import pytest
from fastapi.testclient import TestClient

from soup_cli.ui.app import create_app, get_auth_token

# The four routes #731 is about. `/docs/oauth2-redirect` is derived from
# `swagger_ui_oauth2_redirect_url`, not from `docs_url`, so it is listed
# separately -- disabling `/docs` alone would leave it serving.
DOC_ROUTES = ["/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"]

# Controls. #687 called these out explicitly: the dashboard cannot load if the
# SPA root or the health probe start answering 401/404, so an over-broad guard
# has to fail a test rather than pass one.
ALWAYS_OPEN = ["/", "/api/health"]

NONLOOPBACK = "0.0.0.0"
LOOPBACK = "127.0.0.1"


def _auth_headers():
    return {"Authorization": f"Bearer {get_auth_token()}"}


class TestDocRoutesAreGoneOnANonLoopbackBind:
    """The acceptance criterion: not served to an unauthenticated LAN client."""

    @pytest.mark.parametrize("route", DOC_ROUTES)
    def test_unauthenticated_client_cannot_read_them(self, route):
        client = TestClient(create_app(host=NONLOOPBACK))
        resp = client.get(route)
        assert resp.status_code == 404, (
            f"{route} served {resp.status_code} to an unauthenticated client "
            f"on a {NONLOOPBACK} bind"
        )

    @pytest.mark.parametrize("route", DOC_ROUTES)
    def test_the_route_is_absent_not_merely_refused(self, route):
        """404, not 401.

        A 401 would mean the surface is still described -- the route exists,
        and a token holder or a future dependency bug re-exposes it. The fix
        is removal, and this pins that distinction so a later rewrite into
        `Depends(_verify_token)` fails here instead of passing quietly.
        """
        client = TestClient(create_app(host=NONLOOPBACK))
        assert client.get(route, headers=_auth_headers()).status_code == 404, (
            f"{route} still exists on a {NONLOOPBACK} bind; it answers to a "
            "token holder, so the schema surface was gated rather than removed"
        )

    def test_no_openapi_schema_is_generated_at_all(self):
        """Belt and braces: the app object itself must not carry the schema URL."""
        app = create_app(host=NONLOOPBACK)
        assert app.openapi_url is None
        assert app.docs_url is None
        assert app.redoc_url is None
        assert app.swagger_ui_oauth2_redirect_url is None


class TestTheBeltAndBracesParametersDoRealWork:
    """Why all four parameters are passed, not just `openapi_url`.

    Upstream registers every docs route behind the schema URL --
    `fastapi/applications.py:1121` is `if self.openapi_url and self.docs_url:`,
    `:1149` is the same for `redoc_url`, and `:1139` (oauth2-redirect) is nested
    inside the `/docs` block. So `openapi_url=None` alone already removes all
    four, and a mutation that re-enables only `docs_url` or `redoc_url` changes
    no behaviour today.

    That makes the other three parameters defence-in-depth against a future edit
    that restores the schema route -- which is a real risk, because the schema is
    the one a developer is most likely to want back. These tests pin the upstream
    contract those parameters depend on, so if FastAPI ever registers `/docs`
    without a schema URL, we find out here rather than on a LAN bind.
    """

    def test_upstream_gates_every_docs_route_behind_the_schema_url(self):
        """The contract `openapi_url=None` relies on, pinned against FastAPI."""
        from fastapi import FastAPI

        bare = FastAPI(openapi_url=None)
        client = TestClient(bare)
        for route in DOC_ROUTES:
            assert client.get(route).status_code == 404, (
                f"FastAPI registered {route} with openapi_url=None; the fix in "
                "create_app relies on it not doing that"
            )

    @pytest.mark.parametrize(
        "kwargs, route",
        [
            ({"docs_url": None}, "/docs"),
            ({"redoc_url": None}, "/redoc"),
            ({"swagger_ui_oauth2_redirect_url": None}, "/docs/oauth2-redirect"),
        ],
    )
    def test_each_parameter_independently_removes_its_own_route(self, kwargs, route):
        """With the schema restored, each parameter must still hold on its own.

        This is the future regression the belt-and-braces arguments exist for: a
        later edit re-enables `openapi_url`, and the remaining three have to keep
        their routes off by themselves.
        """
        from fastapi import FastAPI

        app = FastAPI(openapi_url="/openapi.json", **kwargs)
        assert TestClient(app).get(route).status_code == 404
        # Control: the schema really is back, so a 404 above means the named
        # parameter did the work rather than the schema still being absent.
        assert TestClient(app).get("/openapi.json").status_code == 200


class TestLoopbackKeepsTheDeveloperConvenience:
    """The asymmetry is a decision, not an accident -- so it is pinned."""

    @pytest.mark.parametrize("route", DOC_ROUTES)
    def test_docs_still_serve_on_loopback(self, route):
        client = TestClient(create_app(host=LOOPBACK))
        assert client.get(route).status_code == 200, (
            f"{route} stopped serving on a {LOOPBACK} bind; the fix is scoped "
            "to non-loopback binds and must not take the local docs with it"
        )

    def test_localhost_is_treated_as_loopback_too(self):
        """`_is_loopback` accepts the name, so the docs must survive it."""
        client = TestClient(create_app(host="localhost"))
        assert client.get("/openapi.json").status_code == 200


class TestTheGuardDidNotOverreach:
    """Controls: an over-broad fix breaks the dashboard, and must be caught."""

    @pytest.mark.parametrize("route", ALWAYS_OPEN)
    def test_public_routes_survive_on_a_nonloopback_bind(self, route):
        client = TestClient(create_app(host=NONLOOPBACK))
        assert client.get(route).status_code == 200, (
            f"{route} must stay reachable unauthenticated -- #687 requires it "
            "for the SPA to load and for the health probe to answer"
        )

    def test_private_reads_are_still_gated_on_a_nonloopback_bind(self):
        """The 401 story from #707 is untouched by removing the doc routes."""
        client = TestClient(create_app(host=NONLOOPBACK))
        assert client.get("/api/runs").status_code == 401

    def test_authenticated_reads_still_work_without_the_schema_routes(self):
        """Removing `openapi_url` must not disturb route registration."""
        client = TestClient(create_app(host=NONLOOPBACK))
        assert client.get("/api/system", headers=_auth_headers()).status_code == 200
