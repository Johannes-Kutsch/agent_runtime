from __future__ import annotations


def validate_runtime_identity_label(
    value: str,
    *,
    kind: str,
    allow_empty: bool = False,
) -> str:
    if value == "":
        if allow_empty:
            return value
        raise ValueError(f"{kind} must not be empty")
    if not value.strip():
        raise ValueError(f"{kind} must not be whitespace-only")
    if any(character.isspace() for character in value):
        raise ValueError(f"{kind} must not contain whitespace")
    if "/" in value or "\\" in value:
        raise ValueError(f"{kind} must not contain path separators")
    if value in {".", ".."}:
        raise ValueError(f"{kind} must not be path traversal-like")
    return value


__all__ = [
    "validate_runtime_identity_label",
]
