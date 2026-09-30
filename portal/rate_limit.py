from __future__ import annotations

import asyncio
import time
from threading import Lock

# Simple in-memory rate limiter for auth endpoints.
# Format: { "action:identifier": (window_seconds, [timestamp1, timestamp2, ...]) }
# The window is stored per key so expired keys can be swept without knowing which
# call site produced them (call sites pass different windows: 60s and 3600s).
_rates: dict[str, tuple[float, list[float]]] = {}
_lock = Lock()

# Keys are keyed on caller-supplied identifiers (email addresses, client IPs), so an
# unauthenticated caller controls how many exist. Sweeping on a timer keeps the dict
# proportional to live traffic; the hard cap keeps it bounded even under a flood.
_SWEEP_INTERVAL_SECONDS = 60.0
_MAX_TRACKED_KEYS = 10_000
_last_sweep = 0.0


def _evict_expired(store: dict[str, tuple[float, list[float]]], now: float, *, reserve: int = 0) -> None:
    """Drop entries whose window has fully elapsed, then cap the total size.

    Once every timestamp for a key is older than that key's window, the key can
    never deny a request again, so retaining it only costs memory. If sweeping
    alone does not get us under ``_MAX_TRACKED_KEYS`` we evict the
    least-recently-seen keys as a memory backstop.

    ``reserve`` is the number of keys the caller is about to write, so the cap
    still holds after the caller's own insert.
    """
    for key, (window, timestamps) in list(store.items()):
        if not timestamps or now - timestamps[-1] >= window:
            del store[key]

    overflow = len(store) + reserve - _MAX_TRACKED_KEYS
    if overflow > 0:
        # Timestamps are appended in wall-clock order, so [-1] is the most recent hit.
        stale_first = sorted(store.items(), key=lambda item: item[1][1][-1])
        for key, _ in stale_first[:overflow]:
            del store[key]


def check_rate_limit(action: str, identifier: str, max_requests: int, window_seconds: int = 3600) -> bool:
    """Check if the given action/identifier has exceeded the rate limit.

    Returns True if allowed, False if rate limited.
    """
    global _last_sweep

    key = f"{action}:{identifier}"
    now = time.time()
    cutoff = now - window_seconds

    with _lock:
        if now - _last_sweep >= _SWEEP_INTERVAL_SECONDS or len(_rates) >= _MAX_TRACKED_KEYS:
            _last_sweep = now
            _evict_expired(_rates, now, reserve=1)

        # Filter out old requests
        timestamps = [ts for ts in _rates.get(key, (window_seconds, []))[1] if ts > cutoff]

        if len(timestamps) >= max_requests:
            _rates[key] = (window_seconds, timestamps)
            return False

        timestamps.append(now)
        _rates[key] = (window_seconds, timestamps)
        return True


class InMemoryRateLimiter:
    """
    Async-safe in-memory sliding window rate limiter.

    Suitable for single-process ASGI event loops. Note that this state
    is lost on restart and does not scale horizontally. If VoxBento scales
    out, this should be replaced with a Redis-backed token bucket.

    Expired keys are swept during ``is_rate_limited``, so callers do not need to
    schedule ``cleanup()`` for memory to stay bounded.
    """

    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        # key -> (window_seconds, timestamps of the hits still inside the window)
        self._store: dict[str, tuple[float, list[float]]] = {}
        self._lock = asyncio.Lock()
        self._last_sweep = 0.0

    async def is_rate_limited(self, key: str) -> bool:
        now = time.time()
        cutoff = now - self.window_seconds

        async with self._lock:
            if now - self._last_sweep >= _SWEEP_INTERVAL_SECONDS or len(self._store) >= _MAX_TRACKED_KEYS:
                self._last_sweep = now
                _evict_expired(self._store, now, reserve=1)

            # Drop hits that have slid out of the window, so a caller cannot get a
            # second full allowance by straddling a window boundary.
            timestamps = [ts for ts in self._store.get(key, (self.window_seconds, []))[1] if ts > cutoff]

            if len(timestamps) >= self.max_requests:
                self._store[key] = (self.window_seconds, timestamps)
                return True

            timestamps.append(now)
            self._store[key] = (self.window_seconds, timestamps)
            return False

    async def cleanup(self):
        """Periodic cleanup to prevent unbounded memory growth.

        ``is_rate_limited`` already sweeps on a timer; this remains available for
        callers that want to reclaim memory immediately (for example on an idle
        tick or in tests).
        """
        now = time.time()
        async with self._lock:
            self._last_sweep = now
            _evict_expired(self._store, now)


# Global instances
# 10 requests per minute per IP for authorization
auth_rate_limiter = InMemoryRateLimiter(max_requests=10, window_seconds=60)

# 100 requests per minute per client_id for token exchange
token_rate_limiter = InMemoryRateLimiter(max_requests=100, window_seconds=60)
