from __future__ import annotations

import dataclasses

from .identity import validate_runtime_identity_label


@dataclasses.dataclass(frozen=True, slots=True)
class InvocationRole:
    value: str

    def __post_init__(self) -> None:
        validate_runtime_identity_label(
            self.value,
            kind="InvocationRole value",
        )


__all__ = ["InvocationRole"]
