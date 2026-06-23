from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Protocol

from .identity import validate_session_namespace
from .session import ProviderSessionState, ProviderSessionStateRequest


@dataclasses.dataclass(frozen=True)
class ProviderSessionPlanningRequest:
    worktree: Path
    namespace: str

    def __post_init__(self) -> None:
        validate_session_namespace(self.namespace)


@dataclasses.dataclass(frozen=True)
class ProviderSessionPlanningFacts:
    state_dir_relpath: str | None
    provider_state_dir: Path | None
    has_resumable_provider_state: bool


class ProviderSessionAdapter(Protocol):
    @property
    def service_name(self) -> str: ...

    def provider_session_planning_facts(
        self, request: ProviderSessionPlanningRequest
    ) -> ProviderSessionPlanningFacts: ...

    def provider_session_state(
        self, request: ProviderSessionStateRequest
    ) -> ProviderSessionState: ...

    def prepare_local_provider_run_state(
        self,
        provider_state_dir: Path | None,
    ) -> None: ...


__all__ = [
    "ProviderSessionAdapter",
    "ProviderSessionPlanningFacts",
    "ProviderSessionPlanningRequest",
]
