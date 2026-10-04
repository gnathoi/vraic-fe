import pytest
from fastapi import HTTPException, Response
from fastapi.security import HTTPBasicCredentials
from starlette.requests import Request

from jfe import api


def req(funnel=False, ip="203.0.113.7"):
    headers = [(b"tailscale-funnel-request", b"?1")] if funnel else []
    return Request({"type": "http", "method": "POST", "path": "/", "headers": headers, "client": (ip, 40000)})


@pytest.fixture(autouse=True)
def codes(monkeypatch):
    monkeypatch.setattr(api, "OPERATOR_CODE", "admin-code")
    monkeypatch.setattr(api, "EVENT_CODE", "event-code")
    monkeypatch.setattr(api.db, "mode", lambda: "OPEN_DEMO")
    monkeypatch.setattr(api.db, "new_id", lambda prefix: prefix + "_1")
    monkeypatch.setattr(api.db, "q", lambda *a: [])
    monkeypatch.setattr(api.db, "one", lambda *a: {"1": 1})  # any session cookie would be an admin session
    monkeypatch.setattr(api.db, "event", lambda *a, **k: None)
    api.hits.clear()


def start(code, **kw):
    return api.create_session(api.SessionIn(code=code), req(**kw), Response())


def status(fn, *a, **kw):
    with pytest.raises(HTTPException) as e:
        fn(*a, **kw)
    return e.value.status_code


def test_admin_and_operator_refuse_funnel_requests():
    creds = HTTPBasicCredentials(username="x", password="admin-code")
    assert api.admin(req(), creds) == "admin"
    assert status(api.admin, req(funnel=True), creds) == 404
    op = {"is_operator": True}
    assert api.operator(req(), op) is op
    assert status(api.operator, req(funnel=True), op) == 404


def test_admin_code_gives_no_operator_session_via_funnel():
    assert start("admin-code")["operator"] is True
    assert status(start, "admin-code", funnel=True) == 403
    assert start("event-code", funnel=True)["operator"] is False


def test_unset_event_code_admits_only_the_admin_code(monkeypatch):
    monkeypatch.setattr(api, "EVENT_CODE", "")
    for code in ("", " ", "JERSEY"):
        assert status(start, code) == 403
    assert start("admin-code")["operator"] is True


def test_wrong_codes_are_rate_limited_per_ip_and_ipv6_64():
    for _ in range(20):
        start("event-code")  # valid codes do not count
        assert status(start, "wrong") == 403
    assert status(start, "wrong") == 429
    assert status(start, "event-code") == 429
    assert start("event-code", ip="203.0.113.8")["operator"] is False
    for i in range(20):
        assert status(start, "wrong", ip=f"2001:db8::{i:x}") == 403
    assert status(start, "wrong", ip="2001:db8::ffff:1") == 429
    assert status(start, "wrong", ip="2001:db8:0:1::1") == 403
