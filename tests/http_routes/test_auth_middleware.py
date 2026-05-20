"""HTTP integration tests for the auth middleware."""

import pytest


class TestAuthMiddleware:
    """When --enable-api-auth is set, /v1/* requires a Bearer token."""

    @pytest.mark.asyncio
    async def test_v1_without_auth_returns_401(self, auth_client):
        resp = await auth_client.post("/v1/sessions", json={})
        assert resp.status == 401

    @pytest.mark.asyncio
    async def test_v1_with_wrong_token_returns_401(self, auth_client):
        resp = await auth_client.post(
            "/v1/sessions",
            json={},
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert resp.status == 401

    @pytest.mark.asyncio
    async def test_v1_with_correct_token_succeeds(self, auth_client):
        resp = await auth_client.post(
            "/v1/sessions",
            json={},
            headers={"Authorization": "Bearer secret-test-token"},
        )
        assert resp.status == 200

    @pytest.mark.asyncio
    async def test_non_v1_routes_bypass_auth(self, auth_client):
        """Routes outside /v1/ should work without auth even when enabled."""
        resp = await auth_client.get("/healthz")
        assert resp.status == 200

        resp = await auth_client.get("/config")
        assert resp.status == 200

        resp = await auth_client.get("/status")
        assert resp.status == 200
