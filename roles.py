"""
JWT role decoding for role-aware search.

Deliberately the same token shape and secret convention as the JWT services
in interview-poc-sharepoint (JwtValidator.java) and interview-poc-rust
(auth/mod.rs): HS256, a "role" claim, and a JWT_SECRET env var that must be
at least 32 bytes (RFC 7518 SS3.2). A token minted by either of those two
services' /api/tokens endpoint decodes here unchanged, as long as JWT_SECRET
matches - one auth convention, three codebases, not three separate ones that
happen to look similar.
"""
import os
import time

import jwt

MIN_SECRET_BYTES = 32
ALGORITHM = "HS256"


class TokenError(Exception):
    """Raised for any missing/invalid/expired/undersized-secret token problem."""


def _secret() -> str:
    secret = os.getenv("JWT_SECRET")
    if not secret:
        raise TokenError("JWT_SECRET is not set")
    if len(secret.encode("utf-8")) < MIN_SECRET_BYTES:
        raise TokenError(
            f"JWT_SECRET is {len(secret.encode('utf-8'))} bytes; "
            f"HMAC-SHA signing requires at least {MIN_SECRET_BYTES} bytes (256 bits)"
        )
    return secret


def role_from_token(token: str) -> str:
    """Decode a Bearer/raw JWT and return its role claim, upper-cased."""
    if not token or not token.strip():
        raise TokenError("Missing token")

    raw = token.strip()
    if raw.lower().startswith("bearer "):
        raw = raw[len("bearer "):].strip()

    try:
        claims = jwt.decode(raw, _secret(), algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError as e:
        raise TokenError("Token expired") from e
    except jwt.InvalidTokenError as e:
        raise TokenError(f"Invalid token: {e}") from e

    role = claims.get("role")
    if not role:
        raise TokenError("Token has no role claim")
    return str(role).upper()


def generate_demo_token(client_id: str, role: str, expiry_seconds: int = 3600) -> str:
    """
    Mint a token in the same shape JwtValidator.generateToken() / auth::JwtValidator
    issue, for local demo use only (there's no real login flow in this project).
    """
    now = int(time.time())
    claims = {
        "sub": client_id,
        "role": role.upper(),
        "iat": now,
        "exp": now + expiry_seconds,
    }
    return jwt.encode(claims, _secret(), algorithm=ALGORITHM)
