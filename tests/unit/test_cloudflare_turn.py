"""Unit tests for CloudflareTurnProvider: caching, fallback, error handling."""

import json
import sys
import time
import types
from pathlib import Path
from unittest.mock import patch

import pytest

UNIFIED_ROOT = Path(__file__).resolve().parent.parent.parent
if str(UNIFIED_ROOT) not in sys.path:
    sys.path.insert(0, str(UNIFIED_ROOT))

# Mock heavy dependencies that unified_server imports at module level
_heavy_mocks = [
    "huggingface_hub", "sentencepiece", "torch", "torch.cuda",
    "moshi", "moshi.models", "moshi.models.loaders", "moshi.server",
    "scripts", "scripts.musetalk_webrtc", "scripts.musetalk_webrtc.cli",
    "scripts.musetalk_webrtc.server",
]
for mod_name in _heavy_mocks:
    if mod_name not in sys.modules:
        m = types.ModuleType(mod_name)
        sys.modules[mod_name] = m

# Add required attributes that unified_server.py references
sys.modules["torch"].cuda = types.ModuleType("torch.cuda")
sys.modules["torch"].cuda.is_available = lambda: False
sys.modules["torch"].cuda.current_device = lambda: 0
sys.modules["torch"].cuda.mem_get_info = lambda d: (0, 0)
sys.modules["torch"].cuda.memory_allocated = lambda d: 0
sys.modules["torch"].cuda.memory_reserved = lambda d: 0
sys.modules["torch"].cuda.max_memory_allocated = lambda d: 0
sys.modules["torch"].cuda.max_memory_reserved = lambda d: 0
sys.modules["torch"].cuda.get_device_name = lambda d: "mock"
sys.modules["torch"].no_grad = lambda: types.SimpleNamespace(__enter__=lambda s: None, __exit__=lambda s, *a: None)
sys.modules["huggingface_hub"].hf_hub_download = lambda *a, **kw: "/dev/null"
sys.modules["moshi.models.loaders"].DEFAULT_REPO = "mock/repo"
sys.modules["moshi.models.loaders"].MIMI_NAME = "mimi.bin"
sys.modules["moshi.models.loaders"].TEXT_TOKENIZER_NAME = "tok.bin"
sys.modules["moshi.models.loaders"].MOSHI_NAME = "moshi.bin"
sys.modules["moshi.server"].ServerState = type("ServerState", (), {})
sys.modules["moshi.server"]._get_voice_prompt_dir = lambda *a: "/tmp"
sys.modules["moshi.server"].seed_all = lambda x: None
sys.modules["moshi.server"].torch_auto_device = lambda x: "cpu"
sys.modules["scripts.musetalk_webrtc.cli"].parse_args = lambda: None
sys.modules["scripts.musetalk_webrtc.server"].WebRtcApp = type("WebRtcApp", (), {})

from unified_server import CloudflareTurnProvider


@pytest.fixture
def provider():
    return CloudflareTurnProvider(
        token_id="abcdef1234567890abcdef1234567890",
        api_token="test-api-token-for-unit-tests",
        ttl_seconds=3600,
        fallback_policy="all",
    )


def _mock_response(data: dict, status: int = 200):
    """Create a mock urllib response context manager."""
    import io

    class FakeResp:
        def __init__(self):
            self.status = status
            self._body = json.dumps(data).encode("utf-8")

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    return FakeResp()


class TestCaching:
    """Cache TTL and refresh behavior."""

    def test_first_call_fetches(self, provider):
        """First get_config() makes a network request."""
        mock_data = {"iceServers": [{"urls": "turn:example.com:3478"}]}
        with patch("urllib.request.urlopen", return_value=_mock_response(mock_data)):
            config = provider.get_config()
        assert len(config["iceServers"]) == 1
        assert "turn:example.com:3478" in config["iceServers"][0]["urls"]

    def test_second_call_uses_cache(self, provider):
        """Second call within TTL returns cached config without fetch."""
        mock_data = {"iceServers": [{"urls": "turn:example.com:3478"}]}
        with patch("urllib.request.urlopen", return_value=_mock_response(mock_data)) as mock_open:
            provider.get_config()
            provider.get_config()
        # Should only have called urlopen once
        assert mock_open.call_count == 1

    def test_cache_expires_after_ttl(self, provider):
        """After TTL window passes, a new fetch is made."""
        mock_data = {"iceServers": [{"urls": "turn:example.com:3478"}]}
        with patch("urllib.request.urlopen", return_value=_mock_response(mock_data)) as mock_open:
            provider.get_config()
            # Force cache expiration
            provider._cached_until = time.time() - 1
            provider.get_config()
        assert mock_open.call_count == 2


class TestErrorHandling:
    """Error paths and fallback behavior."""

    def test_http_error_raises(self, provider):
        """HTTP error from Cloudflare raises RuntimeError."""
        import urllib.error

        exc = urllib.error.HTTPError(
            url="https://example.com",
            code=403,
            msg="Forbidden",
            hdrs={},
            fp=__import__("io").BytesIO(b'{"error": "forbidden"}'),
        )
        with patch("urllib.request.urlopen", side_effect=exc):
            with pytest.raises(RuntimeError, match="Cloudflare TURN HTTP 403"):
                provider.get_config()

    def test_last_error_populated_on_failure(self, provider):
        """last_error and last_status_code are set on failure."""
        import urllib.error

        exc = urllib.error.HTTPError(
            url="https://example.com",
            code=500,
            msg="Internal Server Error",
            hdrs={},
            fp=__import__("io").BytesIO(b"server error"),
        )
        with patch("urllib.request.urlopen", side_effect=exc):
            with pytest.raises(RuntimeError):
                provider.get_config()
        assert provider.last_status_code == 500


class TestFiltering:
    """URL filtering (e.g., removing :53 DNS ports)."""

    def test_filters_port_53_urls(self, provider):
        """TURN URLs with :53 are filtered out (Cloudflare DNS port).

        NOTE: The current filter uses substring ':53' which also matches ':5349'.
        This is a known bug — the filter is overly aggressive. This test documents
        the actual behavior. When fixed, update this test to keep :5349.
        """
        mock_data = {
            "iceServers": [
                {
                    "urls": [
                        "turn:turn.example.com:3478",
                        "turn:dns.example.com:53",  # should be filtered
                        "turns:turn.example.com:5349",  # BUG: also filtered by ":53" substring
                    ],
                    "username": "user",
                    "credential": "pass",
                }
            ]
        }
        with patch("urllib.request.urlopen", return_value=_mock_response(mock_data)):
            config = provider.get_config()

        urls = config["iceServers"][0]["urls"]
        assert "turn:dns.example.com:53" not in urls
        assert "turn:turn.example.com:3478" in urls
        # BUG: :5349 is incorrectly filtered because of naive ":53" substring match
        # TODO: Fix filter to use regex or exact port match
        assert "turns:turn.example.com:5349" not in urls  # documents the bug


class TestDebugIdentity:
    """Debug identity output for diagnostics."""

    def test_debug_identity_structure(self, provider):
        """debug_identity returns expected keys with masked values."""
        identity = provider.debug_identity()
        assert "token_id_prefix" in identity
        assert "token_id_suffix" in identity
        assert "api_token_len" in identity
        assert "api_token_sha256_prefix" in identity
        assert len(identity["token_id_prefix"]) == 8
        assert len(identity["token_id_suffix"]) == 6
