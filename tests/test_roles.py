import os

import jwt
import pytest

import roles

VALID_SECRET = "CHANGE-ME-LOCAL-DEV-JWT-SECRET-32BYTES-MIN"


@pytest.fixture(autouse=True)
def set_secret(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", VALID_SECRET)


def test_generate_then_decode_round_trips_the_role():
    token = roles.generate_demo_token("client-1", "admin")
    assert roles.role_from_token(token) == "ADMIN"


def test_role_from_token_accepts_a_raw_bearer_header_value():
    token = roles.generate_demo_token("client-1", "read")
    assert roles.role_from_token(f"Bearer {token}") == "READ"


def test_role_from_token_rejects_a_missing_token():
    with pytest.raises(roles.TokenError):
        roles.role_from_token("")
    with pytest.raises(roles.TokenError):
        roles.role_from_token("   ")


def test_role_from_token_rejects_a_token_signed_with_a_different_key():
    other_token = jwt.encode(
        {"sub": "client-1", "role": "ADMIN"},
        "a-completely-different-32-byte-plus-signing-key",
        algorithm="HS256",
    )
    with pytest.raises(roles.TokenError):
        roles.role_from_token(other_token)


def test_role_from_token_rejects_a_token_with_no_role_claim():
    token = jwt.encode({"sub": "client-1"}, VALID_SECRET, algorithm="HS256")
    with pytest.raises(roles.TokenError):
        roles.role_from_token(token)


def test_generate_demo_token_rejects_an_undersized_secret(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "too-short")
    with pytest.raises(roles.TokenError, match="9 bytes"):
        roles.generate_demo_token("client-1", "read")


def test_role_from_token_requires_jwt_secret_to_be_set(monkeypatch):
    monkeypatch.delenv("JWT_SECRET", raising=False)
    with pytest.raises(roles.TokenError):
        roles.role_from_token("anything")
