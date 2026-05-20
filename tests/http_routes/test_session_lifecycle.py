"""HTTP integration tests for session create/delete/stats lifecycle."""

import pytest


class TestCreateSession:
    """POST /v1/sessions creates a new session."""

    @pytest.mark.asyncio
    async def test_creates_session(self, client):
        resp = await client.post("/v1/sessions", json={})
        assert resp.status == 200
        data = await resp.json()
        assert "session_id" in data
        assert "token" in data
        assert data["single_session_mode"] is True

    @pytest.mark.asyncio
    async def test_second_session_returns_409_in_single_mode(self, client):
        """In single-session mode, a second create without replace=true → 409."""
        resp1 = await client.post("/v1/sessions", json={})
        assert resp1.status == 200

        resp2 = await client.post("/v1/sessions", json={})
        assert resp2.status == 409
        data = await resp2.json()
        assert "error" in data
        assert "active_session" in data

    @pytest.mark.asyncio
    async def test_replace_true_closes_existing(self, client):
        """replace=true closes the old session and creates a new one."""
        resp1 = await client.post("/v1/sessions", json={})
        data1 = await resp1.json()
        old_id = data1["session_id"]

        resp2 = await client.post("/v1/sessions", json={"replace": True})
        assert resp2.status == 200
        data2 = await resp2.json()
        assert data2["session_id"] != old_id


class TestDeleteSession:
    """DELETE /v1/sessions/{id} closes a session."""

    @pytest.mark.asyncio
    async def test_delete_with_valid_token(self, client):
        resp = await client.post("/v1/sessions", json={})
        data = await resp.json()
        sid = data["session_id"]
        token = data["token"]

        del_resp = await client.delete(
            f"/v1/sessions/{sid}",
            headers={"x-session-token": token},
        )
        assert del_resp.status == 200
        del_data = await del_resp.json()
        assert del_data["ok"] is True

    @pytest.mark.asyncio
    async def test_delete_without_token_returns_401(self, client):
        resp = await client.post("/v1/sessions", json={})
        data = await resp.json()
        sid = data["session_id"]

        del_resp = await client.delete(f"/v1/sessions/{sid}")
        assert del_resp.status == 401

    @pytest.mark.asyncio
    async def test_delete_wrong_token_returns_401(self, client):
        resp = await client.post("/v1/sessions", json={})
        data = await resp.json()
        sid = data["session_id"]

        del_resp = await client.delete(
            f"/v1/sessions/{sid}",
            headers={"x-session-token": "wrong-token"},
        )
        assert del_resp.status == 401

    @pytest.mark.asyncio
    async def test_delete_nonexistent_returns_404(self, client):
        del_resp = await client.delete(
            "/v1/sessions/nonexistent-id",
            headers={"x-session-token": "any"},
        )
        assert del_resp.status == 404


class TestSessionStats:
    """GET /v1/sessions/{id}/stats returns session details."""

    @pytest.mark.asyncio
    async def test_stats_returns_session_info(self, client):
        resp = await client.post("/v1/sessions", json={})
        data = await resp.json()
        sid = data["session_id"]
        token = data["token"]

        stats_resp = await client.get(
            f"/v1/sessions/{sid}/stats",
            headers={"x-session-token": token},
        )
        assert stats_resp.status == 200
        stats = await stats_resp.json()
        assert stats["session_id"] == sid
        assert "created_epoch" in stats
        assert "pc_state" in stats


class TestActiveSession:
    """GET /v1/session returns the current active session."""

    @pytest.mark.asyncio
    async def test_no_active_session(self, client):
        resp = await client.get("/v1/session")
        assert resp.status == 200
        data = await resp.json()
        assert data["active_session"] is None

    @pytest.mark.asyncio
    async def test_active_session_after_create(self, client):
        await client.post("/v1/sessions", json={})
        resp = await client.get("/v1/session")
        data = await resp.json()
        assert data["active_session"] is not None
        assert "session_id" in data["active_session"]
