"""Tests for password login, sessions and the request gate.

The security properties here are the ones that fail silently: a session that
outlives its timeout, a cookie missing HttpOnly, a gate that lets an unsigned
request through to an endpoint holding the Docker socket.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from webui import auth as auth_mod  # noqa: E402
from webui import server as server_mod  # noqa: E402

PASSWORD = "correct horse battery staple"


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------

class TestPasswordHashing:
    def test_roundtrip(self):
        stored = auth_mod.hash_password(PASSWORD)
        assert auth_mod.verify_password(stored, PASSWORD)

    def test_wrong_password_fails(self):
        stored = auth_mod.hash_password(PASSWORD)
        assert not auth_mod.verify_password(stored, PASSWORD + "x")

    def test_salt_is_per_call(self):
        assert auth_mod.hash_password(PASSWORD)["salt"] != \
               auth_mod.hash_password(PASSWORD)["salt"]

    def test_plaintext_is_never_stored(self):
        stored = auth_mod.hash_password(PASSWORD)
        assert PASSWORD not in json.dumps(stored)

    @pytest.mark.parametrize("broken", [{}, {"salt": "!!", "hash": "!!"},
                                        {"salt": "AAAA"}, {"hash": "AAAA"}])
    def test_malformed_record_fails_closed(self, broken):
        assert not auth_mod.verify_password(broken, PASSWORD)


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

@pytest.fixture
def auth(tmp_path):
    return auth_mod.Auth(str(tmp_path), enabled=True, password=PASSWORD)


class TestSessions:
    def test_login_then_valid(self, auth):
        sid = auth.login(PASSWORD, "1.1.1.1")
        assert sid and auth.is_valid(sid)

    def test_wrong_password_gives_no_session(self, auth):
        assert auth.login("nope", "1.1.1.1") is None

    @pytest.mark.parametrize("bad", ["", None, 123, {}])
    def test_non_string_password_refused(self, auth, bad):
        assert auth.login(bad, "1.1.1.1") is None

    def test_unknown_session_invalid(self, auth):
        assert not auth.is_valid("made-up")
        assert not auth.is_valid(None)

    def test_logout_revokes(self, auth):
        sid = auth.login(PASSWORD, "1.1.1.1")
        auth.logout(sid)
        assert not auth.is_valid(sid)

    def test_sessions_are_distinct(self, auth):
        a = auth.login(PASSWORD, "1.1.1.1")
        b = auth.login(PASSWORD, "1.1.1.1")
        assert a != b
        auth.logout(a)
        assert auth.is_valid(b)

    def test_idle_timeout(self, tmp_path):
        a = auth_mod.Auth(str(tmp_path), enabled=True, password=PASSWORD, idle_s=0.05)
        sid = a.login(PASSWORD, "1.1.1.1")
        time.sleep(0.1)
        assert not a.is_valid(sid)

    def test_absolute_timeout(self, tmp_path):
        a = auth_mod.Auth(str(tmp_path), enabled=True, password=PASSWORD,
                          absolute_s=0.05)
        sid = a.login(PASSWORD, "1.1.1.1")
        time.sleep(0.1)
        assert not a.is_valid(sid)

    def test_activity_extends_idle_window(self, tmp_path):
        a = auth_mod.Auth(str(tmp_path), enabled=True, password=PASSWORD, idle_s=0.3)
        sid = a.login(PASSWORD, "1.1.1.1")
        for _ in range(4):
            time.sleep(0.1)
            assert a.is_valid(sid)

    def test_disabled_auth_lets_everything_through(self, tmp_path):
        a = auth_mod.Auth(str(tmp_path), enabled=False)
        assert a.is_valid(None) and a.is_valid("anything")


class TestBackoff:
    def test_no_delay_before_threshold(self, auth):
        for _ in range(auth_mod.BACKOFF_AFTER - 1):
            auth.login("wrong", "9.9.9.9")
        assert auth.retry_after("9.9.9.9") == 0.0

    def test_delay_grows_after_threshold(self, auth):
        for _ in range(auth_mod.BACKOFF_AFTER + 2):
            auth.login("wrong", "9.9.9.9")
        assert auth.retry_after("9.9.9.9") > 0

    def test_backoff_is_per_client(self, auth):
        for _ in range(auth_mod.BACKOFF_AFTER + 2):
            auth.login("wrong", "9.9.9.9")
        assert auth.retry_after("8.8.8.8") == 0.0

    def test_success_clears_backoff(self, auth):
        for _ in range(auth_mod.BACKOFF_AFTER + 2):
            auth.login("wrong", "5.5.5.5")
        auth.login(PASSWORD, "5.5.5.5")
        assert auth.retry_after("5.5.5.5") == 0.0


class TestCookies:
    def test_flags(self, auth):
        cookie = auth.cookie("abc")
        assert "HttpOnly" in cookie
        assert "SameSite=Strict" in cookie
        assert "Path=/" in cookie
        assert "Secure" not in cookie

    def test_secure_when_behind_tls(self, tmp_path):
        a = auth_mod.Auth(str(tmp_path), enabled=True, password=PASSWORD,
                          secure_cookies=True)
        assert "Secure" in a.cookie("abc")

    def test_expired_cookie_clears(self, auth):
        assert "Max-Age=0" in auth.expired_cookie()

    @pytest.mark.parametrize("header,expected", [
        ("vb_session=abc", "abc"),
        ("other=1; vb_session=xyz; more=2", "xyz"),
        ("other=1", None),
        ("", None),
        (None, None),
        ("vb_session=", None),
    ])
    def test_parsing(self, header, expected):
        assert auth_mod.Auth.session_from_header(header) == expected


class TestCredentialStore:
    def test_generated_on_first_run_and_reused(self, tmp_path):
        first = auth_mod.Auth(str(tmp_path), enabled=True)
        assert first.generated_password
        assert os.path.isfile(first.credentials_path)

        second = auth_mod.Auth(str(tmp_path), enabled=True)
        assert second.generated_password is None
        assert second.login(first.generated_password, "1.1.1.1")

    def test_stored_file_holds_no_plaintext(self, tmp_path):
        a = auth_mod.Auth(str(tmp_path), enabled=True)
        with open(a.credentials_path) as fh:
            assert a.generated_password not in fh.read()

    def test_supplied_password_is_not_persisted(self, tmp_path):
        auth_mod.Auth(str(tmp_path), enabled=True, password=PASSWORD)
        assert not os.path.exists(
            os.path.join(str(tmp_path), "state", "webui", "credentials.json"))


# ---------------------------------------------------------------------------
# The request gate
# ---------------------------------------------------------------------------

def request(url, *, method="GET", body=None, headers=None, cookie=None):
    hdrs = dict(headers or {})
    if cookie:
        hdrs["Cookie"] = cookie
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        hdrs["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    opener = urllib.request.build_opener(urllib.request.HTTPRedirectHandler)
    try:
        with opener.open(req, timeout=5) as res:
            return res.status, res.read(), res.headers
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), exc.headers


@pytest.fixture
def secured(tmp_path):
    auth = auth_mod.Auth(str(tmp_path), enabled=True, password=PASSWORD)
    server = server_mod.make_server(str(tmp_path), "127.0.0.1", 0, True, auth)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    yield base, auth
    server.shutdown()
    server.server_close()


def sign_in(base):
    status, _body, headers = request(f"{base}/api/login", method="POST",
                                     body={"password": PASSWORD})
    assert status == 200
    return headers["Set-Cookie"].split(";")[0]


class TestGate:
    @pytest.mark.parametrize("path", [
        "/api/runs", "/api/profiles", "/api/jobs", "/api/engines", "/api/datasets"])
    def test_api_requires_sign_in(self, secured, path):
        base, _auth = secured
        status, _b, _h = request(base + path)
        assert status == 401

    def test_page_redirects_to_login(self, secured):
        base, _auth = secured
        # No redirect following: the 302 itself is the assertion.
        req = urllib.request.Request(base + "/")

        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *a, **k):
                return None

        opener = urllib.request.build_opener(NoRedirect)
        try:
            with opener.open(req, timeout=5) as res:
                assert res.status == 302
        except urllib.error.HTTPError as exc:
            assert exc.code == 302
            assert exc.headers["Location"] == "/login.html"

    @pytest.mark.parametrize("path", ["/login.html", "/login.js", "/style.css"])
    def test_login_assets_are_public(self, secured, path):
        base, _auth = secured
        status, _b, _h = request(base + path)
        assert status == 200

    def test_health_is_public_but_says_nothing(self, secured):
        base, _auth = secured
        status, body, _h = request(f"{base}/api/health")
        payload = json.loads(body)
        assert status == 200
        assert payload["auth_enabled"] is True
        assert payload["authenticated"] is False
        assert "root" not in payload
        assert "control_enabled" not in payload

    def test_sign_in_then_reach_the_api(self, secured):
        base, _auth = secured
        cookie = sign_in(base)
        status, _b, _h = request(f"{base}/api/runs", cookie=cookie)
        assert status == 200

    def test_wrong_password_refused(self, secured):
        base, _auth = secured
        status, _b, _h = request(f"{base}/api/login", method="POST",
                                 body={"password": "nope"})
        assert status == 401

    def test_cookie_carries_the_flags(self, secured):
        base, _auth = secured
        _s, _b, headers = request(f"{base}/api/login", method="POST",
                                  body={"password": PASSWORD})
        cookie = headers["Set-Cookie"]
        assert "HttpOnly" in cookie and "SameSite=Strict" in cookie

    def test_logout_ends_the_session(self, secured):
        base, _auth = secured
        cookie = sign_in(base)
        status, _b, _h = request(f"{base}/api/logout", method="POST", cookie=cookie)
        assert status == 200
        status, _b, _h = request(f"{base}/api/runs", cookie=cookie)
        assert status == 401

    def test_forged_cookie_refused(self, secured):
        base, _auth = secured
        status, _b, _h = request(f"{base}/api/runs", cookie="vb_session=forged")
        assert status == 401

    def test_cross_site_mutation_refused(self, secured):
        base, _auth = secured
        cookie = sign_in(base)
        status, _b, _h = request(f"{base}/api/jobs", method="POST",
                                 body={"profile": "dev"}, cookie=cookie,
                                 headers={"Origin": "https://evil.example"})
        assert status == 403

    def test_same_origin_mutation_allowed_through_the_gate(self, secured):
        base, _auth = secured
        cookie = sign_in(base)
        host = base.split("//", 1)[1]
        status, _b, _h = request(f"{base}/api/jobs", method="POST",
                                 body={"profile": "nope"}, cookie=cookie,
                                 headers={"Origin": f"http://{host}"})
        # Rejected on its merits (no such profile), not by the origin check.
        assert status == 400

    def test_repeated_failures_get_throttled(self, secured):
        base, _auth = secured
        for _ in range(auth_mod.BACKOFF_AFTER + 1):
            request(f"{base}/api/login", method="POST", body={"password": "bad"})
        status, body, _h = request(f"{base}/api/login", method="POST",
                                   body={"password": "bad"})
        assert status == 429
        assert "retry in" in json.loads(body)["error"]


class TestExposureDefaults:
    @pytest.mark.parametrize("host,loopback", [
        ("127.0.0.1", True), ("localhost", True), ("::1", True), ("", True),
        ("0.0.0.0", False), ("192.168.1.10", False),
    ])
    def test_is_loopback(self, host, loopback):
        assert server_mod.is_loopback(host) is loopback

    def test_contradictory_flags_refused(self):
        assert server_mod.main(["--host", "0.0.0.0", "--auth", "--no-auth"]) == 2

    @pytest.mark.parametrize("argv,expected", [
        # Loopback stays frictionless.
        (["--host", "127.0.0.1"], False),
        (["--host", "127.0.0.1", "--auth"], True),
        # A non-loopback binding turns auth on by itself.
        (["--host", "0.0.0.0"], True),
        (["--host", "192.168.1.10"], True),
        (["--host", "0.0.0.0", "--no-auth"], False),
        # In a container the bind address is always 0.0.0.0; what decides is
        # the address it is actually published on.
        (["--host", "0.0.0.0", "--published-host", "127.0.0.1"], False),
        (["--host", "127.0.0.1", "--published-host", "192.168.1.10"], True),
    ])
    def test_auth_default_follows_reachability(self, argv, expected, monkeypatch):
        seen = {}

        def fake_serve(*_a, **kw):
            seen.update(kw)
            return 0

        monkeypatch.setattr(server_mod, "serve", fake_serve)
        assert server_mod.main(argv) == 0
        assert seen["auth_enabled"] is expected
