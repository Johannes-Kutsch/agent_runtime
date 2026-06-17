from __future__ import annotations

import dataclasses

from .identity import validate_runtime_identity_label


@dataclasses.dataclass(frozen=True)
class StageSelection:
    model: str = ""
    effort: str = ""
    service: str = ""
    fallback: StageSelection | None = None

    def __post_init__(self) -> None:
        validate_runtime_identity_label(
            self.service,
            kind="StageSelection service",
        )


StageOverride = StageSelection


__all__ = ["StageSelection"]
