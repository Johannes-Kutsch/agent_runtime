from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True, slots=True)
class InvocationRole:
    value: str

    def __post_init__(self) -> None:
        if not self.value:
            raise ValueError("InvocationRole value must not be empty")
        if any(character.isspace() for character in self.value):
            raise ValueError("InvocationRole value must not contain whitespace")
        if "/" in self.value or "\\" in self.value:
            raise ValueError("InvocationRole value must not contain path separators")
        if self.value in {".", ".."}:
            raise ValueError("InvocationRole value must not be path traversal-like")


__all__ = ["InvocationRole"]
