"""Unit tests for the BackoffPolicy / compute_backoff helper.

These are pure-function tests with no I/O and a seeded RNG, so they're
deterministic and run in milliseconds.
"""

from __future__ import annotations

import importlib
import random

import pytest


@pytest.fixture
def backoff_mod():
    return importlib.import_module("scripts.musetalk_webrtc.backoff")


class TestBackoffPolicyValidation:
    """The dataclass should reject obviously broken inputs."""

    def test_rejects_negative_base(self, backoff_mod):
        with pytest.raises(ValueError, match="base_seconds"):
            backoff_mod.BackoffPolicy(base_seconds=-1.0)

    def test_rejects_cap_below_base(self, backoff_mod):
        with pytest.raises(ValueError, match="cap_seconds"):
            backoff_mod.BackoffPolicy(base_seconds=10.0, cap_seconds=5.0)

    def test_rejects_multiplier_below_one(self, backoff_mod):
        with pytest.raises(ValueError, match="multiplier"):
            backoff_mod.BackoffPolicy(multiplier=0.5)

    def test_accepts_zero_base(self, backoff_mod):
        # A zero base is degenerate but valid; it means delay starts at 0.
        p = backoff_mod.BackoffPolicy(base_seconds=0.0, cap_seconds=10.0)
        assert p.deterministic_max(0) == 0.0


class TestDeterministicMax:
    """Verify the upper-bound schedule before jitter is applied."""

    def test_doubles_per_attempt(self, backoff_mod):
        p = backoff_mod.BackoffPolicy(base_seconds=1.0, cap_seconds=1000.0, multiplier=2.0)
        assert p.deterministic_max(0) == 1.0
        assert p.deterministic_max(1) == 2.0
        assert p.deterministic_max(2) == 4.0
        assert p.deterministic_max(3) == 8.0

    def test_caps_at_cap_seconds(self, backoff_mod):
        p = backoff_mod.BackoffPolicy(base_seconds=1.0, cap_seconds=5.0, multiplier=2.0)
        # 1, 2, 4, 8 — but 8 > 5 so it caps
        assert p.deterministic_max(3) == 5.0
        assert p.deterministic_max(10) == 5.0
        assert p.deterministic_max(100) == 5.0

    def test_handles_huge_attempt_without_overflow(self, backoff_mod):
        p = backoff_mod.BackoffPolicy(base_seconds=1.0, cap_seconds=30.0)
        # 2**1000 would overflow; the helper must clamp safely.
        assert p.deterministic_max(1000) == 30.0
        assert p.deterministic_max(10**6) == 30.0

    def test_negative_attempt_treated_as_zero(self, backoff_mod):
        p = backoff_mod.BackoffPolicy(base_seconds=1.0, cap_seconds=30.0)
        assert p.deterministic_max(-1) == 1.0
        assert p.deterministic_max(-100) == 1.0


class TestComputeBackoffJitter:
    """Verify the jitter behavior with a seeded RNG."""

    def test_returns_value_within_bounds(self, backoff_mod):
        """For any attempt, the returned delay must be in [0, deterministic_max]."""
        p = backoff_mod.BackoffPolicy(base_seconds=1.0, cap_seconds=10.0)
        rng = random.Random(42)
        for attempt in range(20):
            delay = backoff_mod.compute_backoff(attempt, p, rng=rng)
            assert 0 <= delay <= p.deterministic_max(attempt)

    def test_seeded_rng_is_deterministic(self, backoff_mod):
        """Same seed → same sequence of delays."""
        p = backoff_mod.BackoffPolicy(base_seconds=1.0, cap_seconds=10.0)
        rng_a = random.Random(123)
        rng_b = random.Random(123)
        delays_a = [backoff_mod.compute_backoff(i, p, rng=rng_a) for i in range(10)]
        delays_b = [backoff_mod.compute_backoff(i, p, rng=rng_b) for i in range(10)]
        assert delays_a == delays_b

    def test_non_zero_delays_eventually_seen(self, backoff_mod):
        """With a real RNG, we expect a spread of values, not all zero."""
        p = backoff_mod.BackoffPolicy(base_seconds=1.0, cap_seconds=10.0)
        rng = random.Random(0)
        delays = [backoff_mod.compute_backoff(2, p, rng=rng) for _ in range(50)]
        assert min(delays) >= 0.0
        assert max(delays) <= 4.0  # deterministic_max(2) = 4
        # Variance should be substantial — not all clumped near zero
        assert max(delays) - min(delays) > 1.0

    def test_zero_upper_returns_zero(self, backoff_mod):
        """If base is 0, every delay should be exactly 0."""
        p = backoff_mod.BackoffPolicy(base_seconds=0.0, cap_seconds=10.0)
        rng = random.Random(1)
        for attempt in range(5):
            assert backoff_mod.compute_backoff(attempt, p, rng=rng) == 0.0


class TestThunderingHerdProperty:
    """The whole point of jitter: spread retries across time."""

    def test_50_clients_with_same_attempt_have_spread_of_delays(self, backoff_mod):
        """If 50 clients fail at the same time, their next-retry timestamps should differ.

        This is the property that prevents a thundering-herd reconnect storm
        on PersonaPlex restart.
        """
        p = backoff_mod.BackoffPolicy(base_seconds=1.0, cap_seconds=30.0)
        # Each client uses an independent rng (simulates 50 separate processes)
        delays = [backoff_mod.compute_backoff(3, p, rng=random.Random(seed)) for seed in range(50)]
        # deterministic_max(3) = 8 seconds; jitter spread should cover most of [0, 8]
        assert max(delays) > 5.0  # at least one client waits "a long time"
        assert min(delays) < 2.0  # at least one client retries quickly
