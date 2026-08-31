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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional, Tuple
from urllib.parse import parse_qs, unquote, urlparse

VB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if VB_ROOT not in sys.path:
    sys.path.insert(0, VB_ROOT)

from webui import api as api_mod  # noqa: E402
from webui import control as control_mod  # noqa: E402
from webui import runs as runs_mod  # noqa: E402

api_mod.extend(control_mod.ROUTES)

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
ALLOWED_HOSTS = ("localhost", "127.0.0.1", "[::1]", "::1")


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

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("[webui] %s %s\n" % (self.address_string(), fmt % args))

    # -- helpers ---------------------------------------------------------

    def _host_allowed(self) -> bool:
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0]
        return host in ALLOWED_HOSTS or host == ""

    def _send(self, status: int, body: bytes, content_type: str,
              cache: bool = False) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        if not cache:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_json(self, status: int, payload) -> None:
        body = json.dumps(payload, default=str).encode()
        self._send(status, body, "application/json; charset=utf-8")

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

        served = self._serve_run_asset(path) or self._serve_static(path)
        if not served:
            self._send_json(404, {"error": f"not found: {path}"})

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


def make_server(root: str, host: str, port: int,
                allow_control: bool) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,),
                   {"api": api_mod.Api(root, allow_control=allow_control)})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server


def serve(root: str = VB_ROOT, host: str = "127.0.0.1", port: int = 8080,
          allow_control: bool = False) -> int:
    server = make_server(root, host, port, allow_control)
    control = "enabled" if allow_control else "disabled (read-only)"
    print(f"[webui] serving {root} on http://{host}:{port}  control: {control}",
          flush=True)
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
    args = p.parse_args(argv)
    return serve(args.root, args.host, args.port, args.allow_control)


if __name__ == "__main__":
    raise SystemExit(main())
