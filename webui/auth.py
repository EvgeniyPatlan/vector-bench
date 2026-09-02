"""Password login with server-side sessions.

Not JWT. This is one process on one machine, so the only thing statelessness
would buy is the inability to revoke a token without adding a blocklist -- which
is the state back again. A random session id in an HttpOnly cookie, checked
against a dict this process owns, is smaller and can actually be revoked.

Sessions live in memory, so restarting the server signs everyone out. That is
the intended behaviour rather than a limitation.

This layer authenticates. It does NOT encrypt: over plain HTTP the password and
the cookie cross the network in cleartext. Terminate TLS in front of it.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from typing import Dict, Optional, Tuple

SESSION_COOKIE = "vb_session"
CREDENTIALS_NAME = "credentials.json"

# scrypt parameters. Interactive-login cost: ~100 ms on a modern core.
SCRYPT_N = 2 ** 14
SCRYPT_R = 8
SCRYPT_P = 1
SCRYPT_LEN = 32

DEFAULT_IDLE_S = float(os.environ.get("VB_WEB_IDLE_TIMEOUT", 24 * 3600))
DEFAULT_ABSOLUTE_S = float(os.environ.get("VB_WEB_SESSION_LIFETIME", 7 * 24 * 3600))

# Failed logins back off per client address: no lockout (which would let anyone
# lock the owner out), just an increasing delay.
BACKOFF_AFTER = 3
BACKOFF_BASE_S = 2.0
BACKOFF_MAX_S = 300.0


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


def _unb64(text: str) -> bytes:
    return base64.b64decode(text.encode())


def hash_password(password: str, salt: Optional[bytes] = None) -> Dict[str, str]:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=SCRYPT_N, r=SCRYPT_R,
                            p=SCRYPT_P, dklen=SCRYPT_LEN)
    return {"algo": "scrypt", "n": SCRYPT_N, "r": SCRYPT_R, "p": SCRYPT_P,
            "salt": _b64(salt), "hash": _b64(digest)}


def verify_password(stored: Dict[str, str], candidate: str) -> bool:
    """Constant-time check. A malformed or partial record fails closed."""
    try:
        expected = _unb64(stored["hash"])
        digest = hashlib.scrypt(
            candidate.encode(), salt=_unb64(stored["salt"]),
            n=int(stored.get("n", SCRYPT_N)), r=int(stored.get("r", SCRYPT_R)),
            p=int(stored.get("p", SCRYPT_P)), dklen=SCRYPT_LEN)
    except (KeyError, ValueError, TypeError, AttributeError, binascii.Error):
        return False
    return hmac.compare_digest(digest, expected)


class Auth:
    """Password check, session store and login backoff.

    `enabled` False makes every check pass, which is the loopback default.
    """

    def __init__(self, root: str, enabled: bool = False,
                 password: Optional[str] = None,
                 secure_cookies: bool = False,
                 idle_s: float = DEFAULT_IDLE_S,
                 absolute_s: float = DEFAULT_ABSOLUTE_S):
        self.enabled = enabled
        self.root = root
        self.secure_cookies = secure_cookies
        self.idle_s = idle_s
        self.absolute_s = absolute_s
        self.credentials_path = os.path.join(root, "state", "webui", CREDENTIALS_NAME)

        self._lock = threading.Lock()
        self._sessions: Dict[str, Dict[str, float]] = {}
        self._failures: Dict[str, Tuple[int, float]] = {}
        self._stored: Optional[Dict[str, str]] = None
        self.generated_password: Optional[str] = None

        if enabled:
            self._stored = self._resolve_credentials(password)

    # -- credentials -----------------------------------------------------

    def _resolve_credentials(self, password: Optional[str]) -> Dict[str, str]:
        if password:
            # Supplied for this process only; never written to disk.
            return hash_password(password)

        try:
            with open(self.credentials_path) as fh:
                stored = json.load(fh)
            if stored.get("salt") and stored.get("hash"):
                return stored
        except (OSError, json.JSONDecodeError):
            pass

        generated = secrets.token_urlsafe(18)
        stored = hash_password(generated)
        self.generated_password = generated
        self._write_credentials(stored)
        return stored

    def _write_credentials(self, stored: Dict[str, str]) -> None:
        directory = os.path.dirname(self.credentials_path)
        try:
            os.makedirs(directory, exist_ok=True)
            temporary = f"{self.credentials_path}.tmp"
            with open(temporary, "w") as fh:
                json.dump({**stored, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                              time.gmtime())}, fh, indent=1)
            os.chmod(temporary, 0o600)
            os.replace(temporary, self.credentials_path)
        except OSError:
            pass

    # -- backoff ---------------------------------------------------------

    def retry_after(self, client: str) -> float:
        """Seconds this client must wait before another attempt is accepted."""
        with self._lock:
            failures, next_allowed = self._failures.get(client, (0, 0.0))
        if failures < BACKOFF_AFTER:
            return 0.0
        return max(0.0, next_allowed - time.time())

    def _record_failure(self, client: str) -> None:
        with self._lock:
            failures, _next = self._failures.get(client, (0, 0.0))
            failures += 1
            delay = 0.0
            if failures >= BACKOFF_AFTER:
                delay = min(BACKOFF_MAX_S,
                            BACKOFF_BASE_S * (2 ** (failures - BACKOFF_AFTER)))
            self._failures[client] = (failures, time.time() + delay)

    def _clear_failures(self, client: str) -> None:
        with self._lock:
            self._failures.pop(client, None)

    # -- login -----------------------------------------------------------

    def login(self, password: str, client: str) -> Optional[str]:
        """Session id on success, None on failure."""
        if not self.enabled:
            return None
        if not isinstance(password, str) or not password:
            self._record_failure(client)
            return None
        if not verify_password(self._stored or {}, password):
            self._record_failure(client)
            return None

        self._clear_failures(client)
        session_id = secrets.token_urlsafe(32)
        now = time.time()
        with self._lock:
            self._sessions[session_id] = {"created": now, "last_seen": now}
        return session_id

    def logout(self, session_id: Optional[str]) -> None:
        if not session_id:
            return
        with self._lock:
            self._sessions.pop(session_id, None)

    def is_valid(self, session_id: Optional[str]) -> bool:
        if not self.enabled:
            return True
        if not session_id:
            return False
        now = time.time()
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return False
            if now - session["created"] > self.absolute_s or \
                    now - session["last_seen"] > self.idle_s:
                self._sessions.pop(session_id, None)
                return False
            session["last_seen"] = now
        return True

    # -- cookies ---------------------------------------------------------

    def cookie(self, session_id: str) -> str:
        parts = [f"{SESSION_COOKIE}={session_id}", "Path=/", "HttpOnly",
                 "SameSite=Strict", f"Max-Age={int(self.absolute_s)}"]
        if self.secure_cookies:
            parts.append("Secure")
        return "; ".join(parts)

    def expired_cookie(self) -> str:
        parts = [f"{SESSION_COOKIE}=", "Path=/", "HttpOnly", "SameSite=Strict",
                 "Max-Age=0"]
        if self.secure_cookies:
            parts.append("Secure")
        return "; ".join(parts)

    @staticmethod
    def session_from_header(cookie_header: Optional[str]) -> Optional[str]:
        for chunk in (cookie_header or "").split(";"):
            name, _, value = chunk.strip().partition("=")
            if name == SESSION_COOKIE and value:
                return value
        return None
