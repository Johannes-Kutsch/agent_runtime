from __future__ import annotations

import dataclasses

from .identity import validate_runtime_identity_label


@dataclasses.dataclass(frozen=True, slots=True)
class UsageLimitScope:
    value: str

    def __post_init__(self) -> None:
        validate_runtime_identity_label(
            self.value,
            kind="UsageLimitScope value",
        )


__all__ = ["UsageLimitScope"]
