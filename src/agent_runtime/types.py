from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class StageSelection:
    model: str = ""
    effort: str = ""
    service: str = ""
    fallback: StageSelection | None = None


StageOverride = StageSelection


__all__ = ["StageSelection"]
