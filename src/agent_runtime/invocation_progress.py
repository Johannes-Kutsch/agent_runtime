from __future__ import annotations

import enum


class InvocationProgress(enum.Enum):
    NOT_STARTED = "not_started"
    STARTED = "started"


__all__ = ["InvocationProgress"]
