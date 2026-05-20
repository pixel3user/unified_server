"""HTTP integration tests for /healthz, /config, /status endpoints."""

import pytest


class TestHealthz:
    """The /healthz endpoint reflects server health."""

    @pytest.mark.asyncio
    async def test_healthy_returns_200(self, client):
        resp = await client.get("/healthz")
        assert resp.status == 200
        data = await resp.json()
        assert data["ok"] is True
        assert data["healthy"] is True

    @pytest.mark.asyncio
    async def test_unhealthy_returns_503(self, client):
        """When _healthy is False, /healthz should 503."""
        app_state = client.app["_app_state"]
        app_state._healthy = False

        resp = await client.get("/healthz")
        assert resp.status == 503
        data = await resp.json()
        assert data["ok"] is False
        assert data["healthy"] is False

    @pytest.mark.asyncio
    async def test_includes_uptime(self, client):
        resp = await client.get("/healthz")
        data = await resp.json()
        assert "uptime_seconds" in data
        assert data["uptime_seconds"] >= 0


class TestConfig:
    """The /config endpoint returns RTC configuration for the browser."""

    @pytest.mark.asyncio
    async def test_returns_ice_servers(self, client):
        resp = await client.get("/config")
        assert resp.status == 200
        data = await resp.json()
        assert "iceServers" in data
        assert "iceTransportPolicy" in data

    @pytest.mark.asyncio
    async def test_v1_config_includes_metadata(self, client):
        resp = await client.get("/v1/config")
        assert resp.status == 200
        data = await resp.json()
        assert "rtc_config" in data
        assert "single_session_mode" in data
        assert "session_token_header" in data
        assert data["session_token_header"] == "x-session-token"


class TestStatus:
    """The /status endpoint returns comprehensive diagnostics."""

    @pytest.mark.asyncio
    async def test_status_structure(self, client):
        resp = await client.get("/status")
        assert resp.status == 200
        data = await resp.json()
        assert "uptime_seconds" in data
        assert "mode" in data
        assert data["mode"] == "web_test_only"
        assert "personaplex" in data
        assert "mirror" in data
        assert "engine" in data
        assert "webrtc" in data
        assert "debug" in data

    @pytest.mark.asyncio
    async def test_status_webrtc_section(self, client):
        resp = await client.get("/status")
        data = await resp.json()
        webrtc = data["webrtc"]
        assert "peer_count" in webrtc
        assert "session_count" in webrtc
        assert "single_session_mode" in webrtc
        assert webrtc["session_count"] == 0



class TestReadyz:
    """The /readyz endpoint reflects model readiness, not just process liveness."""

    @pytest.mark.asyncio
    async def test_ready_in_web_test_only_mode(self, client):
        """In web_test_only mode (no engine), /readyz should be 200 immediately."""
        resp = await client.get("/readyz")
        assert resp.status == 200
        data = await resp.json()
        assert data["ready"] is True
        assert data["reasons"] == []

    @pytest.mark.asyncio
    async def test_readyz_returns_503_when_unhealthy(self, client):
        """If the process is unhealthy, /readyz should also be 503."""
        app_state = client.app["_app_state"]
        app_state._healthy = False

        resp = await client.get("/readyz")
        assert resp.status == 503
        data = await resp.json()
        assert data["ready"] is False
        assert any("unhealthy" in r for r in data["reasons"])

    @pytest.mark.asyncio
    async def test_readyz_includes_uptime(self, client):
        resp = await client.get("/readyz")
        data = await resp.json()
        assert "uptime_seconds" in data



class TestMetrics:
    """The /metrics endpoint serves Prometheus exposition format."""

    @pytest.mark.asyncio
    async def test_metrics_returns_501_without_prometheus_client(self, client):
        """Without prometheus_client installed, /metrics returns 501."""
        # prometheus_client may or may not be installed in test env.
        # Either way, the endpoint should be reachable.
        resp = await client.get("/metrics")
        # 501 = not installed, 200 = installed and serving metrics
        assert resp.status in (200, 501)

    @pytest.mark.asyncio
    async def test_metrics_endpoint_is_registered(self, client):
        """The /metrics route exists and doesn't 404."""
        resp = await client.get("/metrics")
        assert resp.status != 404
