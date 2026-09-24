"""
app/security.py
---------------
Optional HTTP Basic authentication for every HTTP and WebSocket endpoint.

Enabled when both AUTH_USERNAME and AUTH_PASSWORD are set. Browsers attach
Basic credentials automatically to fetch(), <img> MJPEG feeds and same-origin
WebSocket handshakes once the user has logged in, so the WebUI needs no changes.
"""

import base64
import binascii
import logging
import os
import secrets
from typing import Optional, Tuple

logger = logging.getLogger("vision_jev.security")

AUTH_REALM = "Vision-Jev Guard"


def load_credentials() -> Optional[Tuple[str, str]]:
    username = os.getenv("AUTH_USERNAME", "")
    password = os.getenv("AUTH_PASSWORD", "")
    if username and password:
        return username, password
    return None


def _check_basic_header(header_value: str, credentials: Tuple[str, str]) -> bool:
    scheme, _, encoded = header_value.partition(" ")
    if scheme.lower() != "basic" or not encoded:
        return False
    try:
        decoded = base64.b64decode(encoded.strip(), validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError):
        return False
    username, sep, password = decoded.partition(":")
    if not sep:
        return False
    user_ok = secrets.compare_digest(username.encode(), credentials[0].encode())
    pass_ok = secrets.compare_digest(password.encode(), credentials[1].encode())
    return user_ok and pass_ok


class BasicAuthMiddleware:
    """Pure ASGI middleware so that WebSocket handshakes are protected as well as HTTP."""

    def __init__(self, app, credentials: Optional[Tuple[str, str]] = None):
        self.app = app
        self.credentials = credentials

    async def __call__(self, scope, receive, send):
        if self.credentials is None or scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        auth_header = headers.get(b"authorization", b"").decode("latin-1")
        if _check_basic_header(auth_header, self.credentials):
            await self.app(scope, receive, send)
            return

        if scope["type"] == "websocket":
            # Reject the handshake before it is accepted
            await send({"type": "websocket.close", "code": 1008})
            return

        await send({
            "type": "http.response.start",
            "status": 401,
            "headers": [
                (b"www-authenticate", f'Basic realm="{AUTH_REALM}", charset="UTF-8"'.encode()),
                (b"content-type", b"text/plain; charset=utf-8"),
            ],
        })
        await send({"type": "http.response.body", "body": b"Authentication required"})


def validate_http_url(url: str) -> str:
    """Allows only absolute http(s) URLs (blocks file://, gopher://, etc.)."""
    from urllib.parse import urlparse

    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("URL must be an absolute http:// or https:// URL")
    return url.strip()
