from __future__ import annotations

import dataclasses
from typing import Any

from ._runtime_lifecycle import Continuation
from .contracts import ToolAccess


@dataclasses.dataclass(frozen=True)
class PortableContinuationPayload:
    service_name: str
    model: str
    effort: str
    tool_access: ToolAccess
    provider_resume_state: dict[str, Any]

    @classmethod
    def from_continuation(
        cls,
        continuation: Continuation,
    ) -> PortableContinuationPayload:
        provider_resume_state = continuation.provider_resume_state
        if not isinstance(provider_resume_state, dict):
            raise TypeError("Continuation provider_resume_state must be a JSON object.")
        return cls(
            service_name=continuation.selected_service,
            model=continuation.selected_model,
            effort=continuation.selected_effort,
            tool_access=continuation.tool_access,
            provider_resume_state=provider_resume_state,
        )

    def to_continuation(self) -> Continuation:
        return Continuation(
            selected_service=self.service_name,
            selected_model=self.model,
            selected_effort=self.effort,
            tool_access=self.tool_access,
            provider_resume_state=self.provider_resume_state,
        )


def read_portable_continuation_payload(
    continuation: Continuation,
) -> PortableContinuationPayload:
    return PortableContinuationPayload.from_continuation(continuation)


def create_portable_continuation_payload(
    *,
    service_name: str,
    model: str,
    effort: str,
    tool_access: ToolAccess,
    provider_resume_state: dict[str, Any],
) -> PortableContinuationPayload:
    return PortableContinuationPayload(
        service_name=service_name,
        model=model,
        effort=effort,
        tool_access=tool_access,
        provider_resume_state=provider_resume_state,
    )
