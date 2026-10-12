"""Common time base and a simple multichannel ring buffer."""
from __future__ import annotations

import time

import numpy as np


class Clock:
    """Host monotonic clock in seconds. All sources stamp packets with this
    (or map device time onto it) so radar/vibration/camera can be fused."""

    def now(self) -> float:
        return time.monotonic()


class RingBuffer:
    """Fixed-capacity buffer of shape (capacity, n_channels)."""

    def __init__(self, capacity: int, n_channels: int = 1) -> None:
        if capacity <= 0 or n_channels <= 0:
            raise ValueError("capacity and n_channels must be > 0")
        self._buf = np.zeros((capacity, n_channels), dtype=np.float64)
        self._cap = capacity
        self._n = 0       # valid samples
        self._head = 0    # next write index

    def __len__(self) -> int:
        return self._n

    def clear(self) -> None:
        self._n = 0
        self._head = 0

    def append(self, block: np.ndarray) -> None:
        block = np.atleast_2d(np.asarray(block, dtype=np.float64))
        if block.shape[1] != self._buf.shape[1]:
            raise ValueError("channel count mismatch")
        if block.shape[0] >= self._cap:
            block = block[-self._cap:]
        k = block.shape[0]
        end = self._head + k
        if end <= self._cap:
            self._buf[self._head:end] = block
        else:
            first = self._cap - self._head
            self._buf[self._head:] = block[:first]
            self._buf[: k - first] = block[first:]
        self._head = end % self._cap
        self._n = min(self._cap, self._n + k)

    def latest(self, n: int) -> np.ndarray:
        """Most recent n samples in chronological order (fewer if not available)."""
        n = min(n, self._n)
        idx = (self._head - n + np.arange(n)) % self._cap
        return self._buf[idx].copy()
