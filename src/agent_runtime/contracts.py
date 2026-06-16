from __future__ import annotations

import dataclasses
import enum
from collections.abc import Callable, Iterable, Iterator
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from .provider_errors import ProviderErrorObservation
from .roles import AgentRole
from .session import RunKind


@dataclasses.dataclass
class AssistantTurn:
    text: str


@dataclasses.dataclass
class PromptTokens:
    count: int


@dataclasses.dataclass
class UnsupportedTokens:
    count: int
    source: str


@dataclasses.dataclass
class Result:
    text: str


@dataclasses.dataclass
class UsageLimit:
    reset_time: datetime | None
    raw_message: str | None = None
    is_permanent: bool = False


@dataclasses.dataclass
class TransientError:
    status_code: int | None
    raw_message: str
    observations: tuple[ProviderErrorObservation, ...] = dataclasses.field(
        default=(),
        compare=False,
    )


@dataclasses.dataclass
class HardError:
    status_code: int
    raw_message: str
    classification: str | None = None
    observations: tuple[ProviderErrorObservation, ...] = dataclasses.field(
        default=(),
        compare=False,
    )


@dataclasses.dataclass
class CredentialFailure:
    raw_message: str
    service_name: str
    source_observations: tuple[ProviderErrorObservation, ...] = dataclasses.field(
        compare=False,
    )
    status_code: int | None = None
    classification: str | None = None


ParsedTurn = (
    AssistantTurn
    | PromptTokens
    | UnsupportedTokens
    | Result
    | UsageLimit
    | TransientError
    | HardError
    | CredentialFailure
)


class ToolPolicy(enum.Enum):
    RESTRICTED = "restricted"
    PARTIAL = "partial"
    FULL = "full"


class ProviderStatePreparationAction(Protocol):
    def apply(self) -> None: ...


class ProviderSessionRecordingStore(Protocol):
    def save_service_session_id(self, service_name: str, session_id: str) -> None: ...


class ServiceSelectionProvider(Protocol):
    def is_available(self, now: datetime | None = None) -> bool: ...

    def next_wake_time(self) -> datetime: ...


class ResumabilityProvider(Protocol):
    def is_resumable(self, state_dir: Path) -> bool: ...


class ExecutionService(Protocol):
    @property
    def name(self) -> str: ...

    def build_command(
        self,
        role: AgentRole,
        model: str,
        effort: str,
        run_kind: RunKind,
        session_uuid: str | None,
        *,
        tool_policy: Any | None = None,
    ) -> str: ...

    def build_env(
        self,
        state_dir_container_path: str | None = None,
        token: str | None = None,
    ) -> dict[str, str]: ...

    def run(
        self,
        lines: Iterable[str],
        on_provider_session_id: Callable[[str], None] | None = None,
    ) -> Iterator[ParsedTurn]: ...

    def mark_exhausted(self, reset_time: datetime | None) -> None: ...


class ExecutionProvider(ExecutionService, Protocol):
    """Compatibility alias for the focused execution service seam."""


class ResidentExecutionProvider(
    ResumabilityProvider,
    ExecutionProvider,
    Protocol,
):
    pass


class SessionPlanningProvider(
    ResumabilityProvider,
    Protocol,
):
    @property
    def name(self) -> str: ...


class AgentService(
    ServiceSelectionProvider,
    ResidentExecutionProvider,
    Protocol,
):
    def state_dir_relpath(self, role: AgentRole, namespace: str = "") -> str | None: ...

    def is_resumable(self, state_dir: Path) -> bool: ...

    def valid_models(self) -> frozenset[str]: ...

    def valid_efforts(self) -> frozenset[str]: ...


__all__ = [
    "AgentService",
    "AssistantTurn",
    "CredentialFailure",
    "ExecutionService",
    "ExecutionProvider",
    "HardError",
    "ParsedTurn",
    "PromptTokens",
    "ResidentExecutionProvider",
    "ProviderSessionRecordingStore",
    "ProviderStatePreparationAction",
    "Result",
    "ResumabilityProvider",
    "ServiceSelectionProvider",
    "SessionPlanningProvider",
    "ToolPolicy",
    "TransientError",
    "UnsupportedTokens",
    "UsageLimit",
]
