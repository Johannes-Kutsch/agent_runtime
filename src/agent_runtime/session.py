from __future__ import annotations

from enum import Enum

from .identity import validate_runtime_identity_label, validate_session_namespace


class RunKind(Enum):
    FRESH = "fresh"
    RESUME = "resume"


def provider_state_relpath(
    role: str,
    provider_name: str,
    namespace: str = "",
    *,
    session_root: str = "",
) -> str:
    validate_runtime_identity_label(
        provider_name,
        kind="Provider state service name",
    )
    validate_session_namespace(namespace)
    base = f"{role}/{provider_name}/"
    if namespace:
        base = f"{role}/{namespace}/{provider_name}/"
    return f"{session_root}/{base}" if session_root else base


__all__ = [
    "RunKind",
    "provider_state_relpath",
]
