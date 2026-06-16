from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class StageOverride:
    model: str = ""
    effort: str = ""
    service: str = ""
    fallback: StageOverride | None = None


__all__ = ["StageOverride"]
