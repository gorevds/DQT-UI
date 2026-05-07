"""Authentication primitives — opt-in OIDC.

.. warning::
   **Security note (v1.x):** the id_token signature is *not verified*
   in this stub. DQT trusts the TLS channel + the OIDC state cookie.
   Do not use this module against an untrusted issuer in production —
   a real deployment must drop in PyJWT against the issuer's JWKS, or
   front DQT with a verified-OIDC reverse proxy.

Activated when ``DQT_AUTH=oidc``. The Dash app's Flask server gains a
``/auth/login`` redirect to the configured OIDC provider, an
``/auth/callback`` that exchanges the code for an id_token, and a
``before_request`` hook that redirects unauthenticated requests.

Configuration env vars:

* ``DQT_AUTH=oidc`` — activate.
* ``DQT_OIDC_ISSUER`` — OIDC issuer URL (.well-known/openid-configuration).
* ``DQT_OIDC_CLIENT_ID`` / ``DQT_OIDC_CLIENT_SECRET``.
* ``DQT_AUTH_SECRET_KEY`` — Flask session secret. Required when auth is on.
* ``DQT_AUTH_ALLOWED_DOMAINS`` — comma-separated email domains that may sign in.

This module deliberately keeps SAML out of scope — banks that need it
plug a SAML proxy (e.g. Keycloak / Authentik) in front of the OIDC
endpoint and DQT keeps its OIDC-only profile clean.
"""
from __future__ import annotations

import json
import logging
import os
import secrets
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Optional

_log = logging.getLogger(__name__)
_ISSUER_CACHE: dict[str, dict] = {}


def is_active() -> bool:
    return (os.environ.get("DQT_AUTH") or "").strip().lower() == "oidc"


def issuer_metadata(issuer: str) -> dict:
    """Fetch and cache the OIDC discovery document for ``issuer``."""
    cache = _ISSUER_CACHE.get(issuer)
    if cache is not None:
        return cache
    url = issuer.rstrip("/") + "/.well-known/openid-configuration"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"OIDC discovery failed for {issuer}: {exc}") from exc
    _ISSUER_CACHE[issuer] = data
    return data


def register(flask_app: Any) -> None:
    """Wire Flask routes for OIDC login / logout / callback.

    No-op if ``DQT_AUTH`` is not ``oidc``. Failures during registration
    log an error and leave the app un-protected — we'd rather serve a
    Dash error than refuse to boot, since misconfigured auth in
    production is the most common DQT bring-up mistake.
    """
    if not is_active():
        return
    try:
        from flask import redirect, request, session, url_for
    except ImportError:  # pragma: no cover
        _log.warning("Flask not available; OIDC auth not mounted")
        return

    secret = os.environ.get("DQT_AUTH_SECRET_KEY")
    if not secret:
        _log.error("DQT_AUTH=oidc but DQT_AUTH_SECRET_KEY not set; auth disabled")
        return
    flask_app.secret_key = secret
    # Tighten session cookie defaults. ``Secure`` only fires when the
    # response is delivered over TLS — fine for production behind nginx,
    # noise-free in dev when the dev server stays on plain HTTP.
    flask_app.config.update(
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=True,
        SESSION_COOKIE_HTTPONLY=True,
    )

    issuer = os.environ.get("DQT_OIDC_ISSUER")
    client_id = os.environ.get("DQT_OIDC_CLIENT_ID")
    client_secret = os.environ.get("DQT_OIDC_CLIENT_SECRET")
    allowed = {d.strip().lower() for d in
                (os.environ.get("DQT_AUTH_ALLOWED_DOMAINS") or "").split(",")
                if d.strip()}
    if not (issuer and client_id and client_secret):
        _log.error("OIDC requires DQT_OIDC_ISSUER / CLIENT_ID / CLIENT_SECRET")
        return

    @flask_app.route("/auth/login")
    def _login():
        meta = issuer_metadata(issuer)
        state = secrets.token_urlsafe(24)
        session["oidc_state"] = state
        params = {
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": url_for("_callback", _external=True),
            "scope": "openid email profile",
            "state": state,
        }
        return redirect(meta["authorization_endpoint"] + "?" + urllib.parse.urlencode(params))

    @flask_app.route("/auth/callback")
    def _callback():
        if request.args.get("state") != session.get("oidc_state"):
            return ("invalid state", 400)
        code = request.args.get("code")
        if not code:
            return ("missing code", 400)
        meta = issuer_metadata(issuer)
        body = urllib.parse.urlencode({
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": url_for("_callback", _external=True),
            "client_id": client_id,
            "client_secret": client_secret,
        }).encode("utf-8")
        req = urllib.request.Request(
            meta["token_endpoint"], data=body, method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                token = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as exc:
            return (f"OIDC token exchange failed: {exc}", 502)
        # Decode the id_token claims without verifying the signature: a
        # production deployment would verify against the JWKS endpoint;
        # the v1 stub trusts the TLS channel + the state cookie. Banks
        # that need full RFC-7519 verification should drop in PyJWT.
        claims = _decode_id_token_unverified(token.get("id_token") or "")
        email = (claims.get("email") or "").lower()
        if allowed and email.split("@", 1)[-1] not in allowed:
            return (f"email domain not allowed: {email}", 403)
        session["user"] = {"email": email, "sub": claims.get("sub")}
        from dqt import audit
        audit.record("auth.login", {"email": email})
        return redirect("/")

    @flask_app.route("/auth/logout")
    def _logout():
        user = session.pop("user", None)
        from dqt import audit
        audit.record("auth.logout", {"email": (user or {}).get("email")})
        return redirect("/")

    @flask_app.before_request
    def _guard():
        # Allow:
        # * auth routes (login / callback / logout);
        # * the healthz probe and OpenAPI access without session;
        # * Dash's own internals (/_dash-*, /assets/*, /_favicon.ico) — without
        #   these the SPA can't render even the login button;
        # * read-only share endpoints — the whole point of share tokens is
        #   that they are unguessable URLs that work without an account.
        path = request.path or ""
        if (
            path.startswith("/auth/")
            or path == "/api/v1/healthz"
            or path.startswith("/_dash-")
            or path.startswith("/assets/")
            or path == "/_favicon.ico"
            or path.startswith("/api/v1/share/")
        ):
            return None
        if "user" not in session:
            return redirect("/auth/login")
        return None


def _decode_id_token_unverified(jwt_text: str) -> dict:
    """Parse the claims segment of a JWT *without verifying the signature*.

    ``v1`` stub semantics — see module docstring.
    """
    import base64

    parts = (jwt_text or "").split(".")
    if len(parts) < 2:
        return {}
    raw = parts[1] + "=" * (-len(parts[1]) % 4)
    try:
        decoded = base64.urlsafe_b64decode(raw.encode("ascii")).decode("utf-8")
        return json.loads(decoded)
    except (ValueError, json.JSONDecodeError):
        return {}


def current_user(session_obj: Optional[dict]) -> Optional[dict]:
    """Convenience accessor used by Dash callbacks; returns the dict
    saved by the OIDC callback, or ``None`` for un-authenticated."""
    if session_obj is None:
        return None
    return session_obj.get("user")
