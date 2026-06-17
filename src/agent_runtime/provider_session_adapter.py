from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Protocol

from .contracts import ProviderSessionRecordingStore, ProviderStatePreparationAction
from .identity import validate_session_namespace
from .roles import InvocationRole
from .session import (
    ProviderSessionPreferences,
    ProviderSessionPreferencesRequest,
    ProviderSessionState,
    ProviderSessionStateRequest,
)


@dataclasses.dataclass(frozen=True)
class ProviderSessionPlanningRequest:
    worktree: Path
    role: InvocationRole
    namespace: str

    def __post_init__(self) -> None:
        validate_session_namespace(self.namespace)


@dataclasses.dataclass(frozen=True)
class ProviderSessionPlanningFacts:
    state_dir_relpath: str | None
    provider_state_dir: Path | None
    has_resumable_provider_state: bool


class ProviderSessionService(Protocol):
    @property
    def name(self) -> str: ...

    def state_dir_relpath(
        self, role: InvocationRole, namespace: str = ""
    ) -> str | None: ...

    def is_resumable(self, state_dir: Path) -> bool: ...

    def provider_session_preferences(
        self, request: ProviderSessionPreferencesRequest
    ) -> ProviderSessionPreferences: ...

    def provider_session_state(
        self, request: ProviderSessionStateRequest
    ) -> ProviderSessionState: ...


class ProviderSessionAdapter(Protocol):
    @property
    def service_name(self) -> str: ...

    def provider_session_planning_facts(
        self, request: ProviderSessionPlanningRequest
    ) -> ProviderSessionPlanningFacts: ...

    def provider_session_preferences(
        self, request: ProviderSessionPreferencesRequest
    ) -> ProviderSessionPreferences: ...

    def provider_session_state(
        self, request: ProviderSessionStateRequest
    ) -> ProviderSessionState: ...

    def prepare_local_provider_run_state(
        self,
        provider_state_dir: Path | None,
        auth_seed_action: ProviderStatePreparationAction | None = None,
    ) -> None: ...

    def record_provider_session_id(
        self,
        *,
        role_session: ProviderSessionRecordingStore,
        provider_session_id: str,
        service_state_dir: Path | None = None,
    ) -> None: ...

    def recover_provider_session_id(
        self,
        provider_state_dir: Path | None,
    ) -> str | None: ...

    def is_exact_resumable_provider_session(
        self,
        *,
        provider_session_id: str | None,
        provider_state_dir: Path | None,
    ) -> bool: ...


__all__ = [
    "ProviderSessionAdapter",
    "ProviderSessionPlanningFacts",
    "ProviderSessionPlanningRequest",
    "ProviderSessionService",
]
