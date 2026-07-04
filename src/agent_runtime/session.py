from __future__ import annotations

from enum import Enum


class RunKind(Enum):
    FRESH = "fresh"
    RESUME = "resume"


__all__ = [
    "RunKind",
]
