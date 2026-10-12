"""Abstract sensor interfaces. Swap implementations without touching the pipeline."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from backend.core.types import PersonDetection, RadarCaps, RadarPacket, VibrationPacket


class _Closeable(ABC):
    @abstractmethod
    def close(self) -> None: ...

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class RadarSource(_Closeable):
    @abstractmethod
    def capabilities(self) -> RadarCaps:
        """Downstream code MUST check this (e.g. has_phase) before using respiration features."""

    @abstractmethod
    def read(self) -> List[RadarPacket]:
        """Non-blocking: return all packets completed since the last call (may be empty)."""


class VibrationSource(_Closeable):
    @abstractmethod
    def read(self) -> List[VibrationPacket]: ...


class PersonSource(_Closeable):
    """Visible-person detections (camera). Used to build the exclusion map."""

    @abstractmethod
    def read(self) -> List[PersonDetection]: ...
