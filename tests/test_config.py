"""Unit tests for portal.config.Settings derived properties."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from portal.config import Settings


def test_debug_defaults_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEBUG", raising=False)
    monkeypatch.delenv("debug", raising=False)
    assert Settings(_env_file=None).debug is False


@pytest.fixture
def example_env(monkeypatch: pytest.MonkeyPatch) -> Path:
    # Exercise the committed example, not a developer's .env or shell overrides.
    for name in tuple(os.environ):
        if name.lower() in Settings.model_fields:
            monkeypatch.delenv(name)
    return Path(__file__).resolve().parents[1] / ".env.example"


def test_env_example_disables_debug_by_default(example_env: Path) -> None:
    assert Settings(_env_file=example_env).debug is False


def test_env_example_allows_explicit_local_debug(example_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEBUG", "true")
    assert Settings(_env_file=example_env).debug is True


def test_env_example_enforces_production_secret_checks(example_env: Path) -> None:
    settings = Settings(_env_file=example_env)
    with pytest.raises(RuntimeError, match="SECRET_KEY"):
        settings.validate_production_secrets()


def test_env_example_accepts_configured_secrets(example_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("SECRET_KEY", "API_KEY_ENCRYPTION_KEY", "MEDIAMTX_AUTH_HOOK_SECRET", "BOOTH_ACCESS_TOKEN"):
        monkeypatch.setenv(name, f"test-only-{name.lower()}-not-a-real-secret-00000000")
    settings = Settings(_env_file=example_env)
    assert settings.debug is False
    settings.validate_production_secrets()


def test_effective_jitsi_internal_base_fallback():
    s = Settings(jitsi_internal_base="", jitsi_base_url="http://jitsi.local")
    assert s.effective_jitsi_internal_base == "http://jitsi.local"


def test_effective_jitsi_internal_base_override():
    s = Settings(jitsi_internal_base="http://internal.jitsi", jitsi_base_url="http://jitsi.local")
    assert s.effective_jitsi_internal_base == "http://internal.jitsi"


def test_legacy_admin_password_environment_variable_is_ignored(monkeypatch):
    monkeypatch.setenv("ADMIN_PASSWORD", "legacy-shared-secret")

    settings = Settings(_env_file=None)

    assert "admin_password" not in type(settings).model_fields


def test_production_requires_booth_access_token():
    settings = Settings(
        _env_file=None,
        debug=False,
        secret_key="a-secure-secret-key-that-is-long-enough",
        mediamtx_auth_hook_secret="a-secure-hook-secret",
        booth_access_token="",
    )

    try:
        settings.validate_production_secrets()
    except RuntimeError as exc:
        assert "BOOTH_ACCESS_TOKEN" in str(exc)
    else:
        raise AssertionError("empty BOOTH_ACCESS_TOKEN must fail production validation")


def test_production_rejects_known_placeholder_and_short_booth_tokens():
    for token in ("change-me", "secret", "short-token"):
        settings = Settings(
            _env_file=None,
            debug=False,
            secret_key="a-secure-secret-key-that-is-long-enough",
            mediamtx_auth_hook_secret="a-secure-hook-secret",
            booth_access_token=token,
        )
        try:
            settings.validate_production_secrets()
        except RuntimeError as exc:
            assert "BOOTH_ACCESS_TOKEN" in str(exc)
        else:
            raise AssertionError(f"weak BOOTH_ACCESS_TOKEN {token!r} must fail production validation")


def test_production_accepts_long_booth_access_token():
    settings = Settings(
        _env_file=None,
        debug=False,
        secret_key="a-secure-secret-key-that-is-long-enough",
        mediamtx_auth_hook_secret="a-secure-hook-secret",
        booth_access_token="b" * 32,
    )
    settings.validate_production_secrets()
