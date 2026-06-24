from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, NewType

from .identity import validate_runtime_identity_label

if TYPE_CHECKING:
    from ._runtime_lifecycle import ProviderAuth

ClaudeCodeOAuthToken = NewType("ClaudeCodeOAuthToken", str)


@dataclasses.dataclass(frozen=True)
class ProviderSelection:
    service: str
    model: str
    effort: str
    auth: ProviderAuth | None = None

    def __post_init__(self) -> None:
        validate_provider_selection(self)

    def __repr__(self) -> str:
        return (
            "ProviderSelection("
            f"service={self.service!r}, "
            f"model={self.model!r}, "
            f"effort={self.effort!r}, "
            f"auth={self.auth!r})"
        )


@dataclasses.dataclass(frozen=True)
class ResolvedProvider:
    """Credential-free identity of the provider actually run."""

    service: str
    model: str
    effort: str


def validate_provider_selection(selection: ProviderSelection) -> None:
    _require_provider_value("service", selection.service)
    _require_provider_value("model", selection.model)
    _require_provider_value("effort", selection.effort)


def _require_provider_value(field_name: str, value: str) -> None:
    if value.strip():
        if field_name == "service":
            validate_runtime_identity_label(
                value,
                kind="ProviderSelection service",
            )
        return
    raise ValueError(f"ProviderSelection requires a non-empty {field_name}.")


__all__ = ["ClaudeCodeOAuthToken", "ProviderSelection", "ResolvedProvider"]
