"""HTTP server for the vector-bench web UI.

Standard library only. Binds to the loopback interface by default: the app can
start benchmark runs, so it is reached over an SSH port-forward rather than
exposed. A Host-header allowlist blocks DNS-rebinding against that binding.
"""

from __future__ import annotations

import json
import mimetypes
import os
import posixpath
import sys
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional, Tuple
from urllib.parse import parse_qs, unquote, urlparse

VB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if VB_ROOT not in sys.path:
    sys.path.insert(0, VB_ROOT)

from webui import api as api_mod  # noqa: E402
from webui import auth as auth_mod  # noqa: E402
from webui import control as control_mod  # noqa: E402
from webui import runs as runs_mod  # noqa: E402

api_mod.extend(control_mod.ROUTES)

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
LOOPBACK_HOSTS = ("localhost", "127.0.0.1", "[::1]", "::1")

# Reachable before signing in: the login page itself, what it needs to render,
# and the endpoints it posts to.
PUBLIC_PATHS = frozenset({
    "/login.html", "/login.js", "/style.css", "/favicon.ico",
    "/api/login", "/api/health",
})


def _safe_join(base: str, relative: str) -> Optional[str]:
    """Resolve relative under base, or None if it escapes."""
    parts = [p for p in posixpath.normpath("/" + relative).split("/") if p and p != ".."]
    candidate = os.path.realpath(os.path.join(base, *parts))
    if candidate != os.path.realpath(base) and not candidate.startswith(
            os.path.realpath(base) + os.sep):
        return None
    return candidate


class Handler(BaseHTTPRequestHandler):
    server_version = "vector-bench-webui"
    protocol_version = "HTTP/1.1"

    # Injected by serve().
    api: api_mod.Api
    auth: auth_mod.Auth
    allowed_hosts: tuple

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("[webui] %s %s\n" % (self.address_string(), fmt % args))

    # -- helpers ---------------------------------------------------------

    def _host_allowed(self) -> bool:
        if not self.allowed_hosts:
            return True
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0]
        return host in self.allowed_hosts or host == ""

    def _session(self) -> Optional[str]:
        return self.auth.session_from_header(self.headers.get("Cookie"))

    def _authenticated(self) -> bool:
        return self.auth.is_valid(self._session())

    def _origin_ok(self) -> bool:
        """Reject a cross-site mutation.

        SameSite=Strict already stops the cookie travelling on a cross-site
        request; this refuses the request outright when an Origin says it came
        from somewhere else, so a misconfigured proxy cannot quietly undo that.
        """
        origin = self.headers.get("Origin")
        if not origin:
            return True
        host = self.headers.get("Host") or ""
        return origin.split("//", 1)[-1] == host

    def _send(self, status: int, body: bytes, content_type: str,
              cache: bool = False, cookie: Optional[str] = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        if cookie:
            self.send_header("Set-Cookie", cookie)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        if not cache:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_json(self, status: int, payload, cookie: Optional[str] = None) -> None:
        body = json.dumps(payload, default=str).encode()
        self._send(status, body, "application/json; charset=utf-8", cookie=cookie)

    def _send_file(self, path: str) -> None:
        try:
            with open(path, "rb") as fh:
                body = fh.read()
        except OSError:
            self._send_json(404, {"error": "not found"})
            return
        ctype, _ = mimetypes.guess_type(path)
        self._send(200, body, ctype or "application/octet-stream")

    # -- routing ---------------------------------------------------------

    def _handle(self, method: str) -> None:
        if not self._host_allowed():
            self._send_json(403, {"error": "host not allowed"})
            return

        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        query = parse_qs(parsed.query)

        if method in ("POST", "PUT", "DELETE") and not self._origin_ok():
            self._send_json(403, {"error": "cross-site request refused"})
            return

        if path == "/api/login" and method == "POST":
            self._handle_login()
            return
        if path == "/api/logout" and method == "POST":
            self.auth.logout(self._session())
            self._send_json(200, {"ok": True}, cookie=self.auth.expired_cookie())
            return

        if self.auth.enabled and path not in PUBLIC_PATHS and not self._authenticated():
            if path.startswith("/api/"):
                self._send_json(401, {"error": "not signed in"})
            else:
                self.send_response(302)
                self.send_header("Location", "/login.html")
                self.send_header("Content-Length", "0")
                self.end_headers()
            return

        if path == "/api/health":
            if self.auth.enabled and not self._authenticated():
                self._send_json(200, {"ok": True, "auth_enabled": True,
                                      "authenticated": False})
            else:
                self._send_json(200, {"ok": True, "auth_enabled": self.auth.enabled,
                                      "authenticated": True,
                                      "root": self.api.root,
                                      "control_enabled": self.api.allow_control})
            return

        if path.startswith("/api/"):
            body = None
            if method in ("POST", "PUT"):
                body = self.read_body()
                if body is None:
                    self._send_json(400, {"error": "invalid JSON body"})
                    return
            result = api_mod.dispatch(self.api, method, path, query, body)
            if result is None:
                self._send_json(404, {"error": f"no such endpoint: {path}"})
            else:
                self._send_json(result[0], result[1])
            return

        if method not in ("GET", "HEAD"):
            self._send_json(405, {"error": f"method not allowed: {method}"})
            return

        served = (self._serve_bundle(path) or self._serve_run_asset(path)
                  or self._serve_static(path))
        if not served:
            self._send_json(404, {"error": f"not found: {path}"})

    def _serve_bundle(self, path: str) -> bool:
        """Package a run and send it as a download.

        Built per request rather than cached: a run directory is small, and a
        stale bundle beside a regenerated report is worse than a second of tar.
        """
        prefix, suffix = "/runs/", "/bundle"
        if not (path.startswith(prefix) and path.endswith(suffix)):
            return False
        run_id = path[len(prefix):-len(suffix)]
        run_dir = runs_mod.resolve_run_dir(self.api.results_dir, run_id)
        if run_dir is None:
            return False

        from orchestrator.export import bundle_filename, write_bundle

        with tempfile.TemporaryDirectory() as work:
            out = os.path.join(work, bundle_filename(run_id))
            ok, detail = write_bundle(run_dir, out)
            if not ok:
                self._send_json(500, {"error": detail})
                return True
            with open(out, "rb") as fh:
                body = fh.read()

        self.send_response(200)
        self.send_header("Content-Type", "application/gzip")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Disposition",
                         f'attachment; filename="{bundle_filename(run_id)}"')
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)
        return True

    def _serve_run_asset(self, path: str) -> bool:
        """Serve results/<run_id>/report/... so report.html and its charts open."""
        prefix = "/runs/"
        if not path.startswith(prefix):
            return False
        remainder = path[len(prefix):]
        run_id, _, relative = remainder.partition("/")
        run_dir = runs_mod.resolve_run_dir(self.api.results_dir, run_id)
        if run_dir is None or not relative:
            return False
        target = _safe_join(run_dir, relative)
        if target is None or not os.path.isfile(target):
            return False
        self._send_file(target)
        return True

    def _serve_static(self, path: str) -> bool:
        relative = "index.html" if path == "/" else path.lstrip("/")
        target = _safe_join(STATIC_DIR, relative)
        if target is None or not os.path.isfile(target):
            return False
        self._send_file(target)
        return True

    def do_GET(self) -> None:
        self._handle("GET")

    def do_HEAD(self) -> None:
        self._handle("HEAD")

    def do_POST(self) -> None:
        self._handle("POST")

    def do_PUT(self) -> None:
        self._handle("PUT")

    def do_DELETE(self) -> None:
        self._handle("DELETE")

    def _handle_login(self) -> None:
        if not self.auth.enabled:
            self._send_json(400, {"error": "authentication is not enabled"})
            return

        client = self.client_address[0] if self.client_address else "?"
        wait = self.auth.retry_after(client)
        if wait > 0:
            self._send_json(429, {"error": f"too many attempts; retry in "
                                           f"{wait:.0f}s"})
            return

        body = self.read_body()
        if body is None:
            self._send_json(400, {"error": "invalid JSON body"})
            return

        session_id = self.auth.login(str(body.get("password") or ""), client)
        if not session_id:
            self._send_json(401, {"error": "wrong password"})
            return
        self._send_json(200, {"ok": True}, cookie=self.auth.cookie(session_id))

    def read_body(self) -> Optional[dict]:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return None
        if length <= 0:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None


def is_loopback(host: str) -> bool:
    return host in ("127.0.0.1", "::1", "localhost", "")


def make_server(root: str, host: str, port: int, allow_control: bool,
                auth: Optional[auth_mod.Auth] = None,
                restrict_host_header: bool = True) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {
        "api": api_mod.Api(root, allow_control=allow_control),
        "auth": auth or auth_mod.Auth(root, enabled=False),
        # Only meaningful for a loopback binding, where it blocks DNS
        # rebinding. A deliberately exposed server is reached by its own
        # hostname, which we cannot know here.
        "allowed_hosts": LOOPBACK_HOSTS if restrict_host_header else (),
    })
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server


def serve(root: str = VB_ROOT, host: str = "127.0.0.1", port: int = 8080,
          allow_control: bool = False, auth_enabled: bool = False,
          password: Optional[str] = None, behind_proxy: bool = False,
          published_host: Optional[str] = None) -> int:
    # In a container the bind address is always 0.0.0.0 and says nothing about
    # who can reach the service -- Docker's publish address decides that. The
    # caller that knows the real one passes it, so the exposure warning
    # describes the deployment rather than the socket.
    reachable = published_host if published_host is not None else host
    loopback = is_loopback(reachable)
    auth = auth_mod.Auth(root, enabled=auth_enabled,
                         password=password,
                         secure_cookies=behind_proxy)
    server = make_server(root, host, port, allow_control, auth,
                         restrict_host_header=loopback)

    control = "enabled" if allow_control else "disabled (read-only)"
    scheme = "https" if behind_proxy else "http"
    # Only claim a URL when this process is the thing you connect to. Behind a
    # container publish the port here is the internal one, and printing it beside
    # the real address is worse than not printing it: the caller that set up the
    # publish knows the address and announces it.
    if published_host is None:
        where = f"on {scheme}://{host}:{port}"
    else:
        where = f"(bound {host}:{port} in-container)"
    print(f"[webui] serving {root} {where}  control: {control}  "
          f"auth: {'on' if auth_enabled else 'off'}", flush=True)
    if auth.generated_password:
        print("\n[webui] generated a password (shown once, stored hashed in "
              "state/webui/credentials.json):\n"
              f"\n    {auth.generated_password}\n", flush=True)
    if not loopback and not behind_proxy:
        print("[webui] WARNING: bound to a non-loopback address without "
              "--behind-proxy. The password and session cookie cross the "
              "network in cleartext unless something in front is terminating "
              "TLS.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[webui] stopped", flush=True)
    finally:
        server.server_close()
    return 0


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(prog="vb-webui")
    p.add_argument("--root", default=VB_ROOT)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--allow-control", action="store_true",
                   help="enable profile editing and run launching")
    p.add_argument("--auth", action="store_true",
                   help="require a password (implied by a non-loopback --host)")
    p.add_argument("--no-auth", action="store_true",
                   help="bind a non-loopback address without a password. "
                        "Only for a network you already trust")
    p.add_argument("--behind-proxy", action="store_true",
                   help="TLS is terminated in front; marks cookies Secure")
    p.add_argument("--published-host", default=None,
                   help="address users actually reach this on, when it differs "
                        "from --host (a container binds 0.0.0.0 regardless)")
    args = p.parse_args(argv)

    # A non-loopback binding exposes an endpoint that can start containers, so
    # it requires a password unless the operator says otherwise in as many
    # words. Getting this wrong silently is the failure worth preventing.
    reachable = args.published_host if args.published_host is not None else args.host
    auth_enabled = args.auth
    if not is_loopback(reachable) and not args.no_auth:
        auth_enabled = True
    if auth_enabled and args.no_auth:
        print("--auth and --no-auth are contradictory", file=sys.stderr)
        return 2

    return serve(args.root, args.host, args.port, args.allow_control,
                 auth_enabled=auth_enabled,
                 password=os.environ.get("VB_WEB_PASSWORD"),
                 behind_proxy=args.behind_proxy,
                 published_host=args.published_host)


if __name__ == "__main__":
    raise SystemExit(main())
