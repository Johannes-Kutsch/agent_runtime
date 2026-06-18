from __future__ import annotations

import dataclasses
import enum
from collections.abc import Callable, Iterable, Iterator
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from .provider_errors import ProviderErrorObservation
from .roles import InvocationRole
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
class ModelActivity:
    pass


@dataclasses.dataclass
class UsageLimit:
    reset_time: datetime | None
    raw_message: str | None = None
    is_permanent: bool = False


@dataclasses.dataclass
class TransientError:
    status_code: int | None
    raw_message: str
    classification: str | None = None
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
    | ModelActivity
    | UsageLimit
    | TransientError
    | HardError
    | CredentialFailure
)


@dataclasses.dataclass(frozen=True)
class ToolPolicyProfile:
    allowed_tools: tuple[str, ...] | None = None
    disallowed_tools: tuple[str, ...] = ()
    strict_mcp_config: bool = True


class ToolPolicy(enum.Enum):
    RESTRICTED = "restricted"
    PARTIAL = "partial"
    FULL = "full"

    @property
    def profile(self) -> ToolPolicyProfile:
        if self is ToolPolicy.RESTRICTED:
            return ToolPolicyProfile(allowed_tools=("Read", "Glob"))
        if self is ToolPolicy.PARTIAL:
            return ToolPolicyProfile(disallowed_tools=("Edit", "Write", "NotebookEdit"))
        return ToolPolicyProfile()


_NO_TOOLS_POLICY = ToolPolicyProfile(
    allowed_tools=("none",),
    disallowed_tools=("all",),
)


@dataclasses.dataclass(frozen=True, init=False)
class ToolAccess:
    kind: str
    workspace: Path | None
    _tool_policy: ToolPolicy | ToolPolicyProfile

    def __init__(
        self,
        *,
        kind: str,
        workspace: Path | None,
        tool_policy: ToolPolicy | ToolPolicyProfile,
    ) -> None:
        if kind not in {"none", "workspace_backed"}:
            raise ValueError(f"Unsupported tool access kind: {kind}")
        if kind == "none" and workspace is not None:
            raise ValueError("ToolAccess.no_tools() cannot carry a workspace.")
        if kind == "none" and tool_policy != _NO_TOOLS_POLICY:
            raise ValueError(
                "ToolAccess.no_tools() must forbid provider tool access with the closed no-tools policy."
            )
        if kind == "workspace_backed" and workspace is None:
            raise ValueError("ToolAccess.workspace_backed() requires a workspace path.")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "workspace", workspace)
        object.__setattr__(self, "_tool_policy", tool_policy)

    @classmethod
    def no_tools(cls) -> ToolAccess:
        return cls(
            kind="none",
            workspace=None,
            tool_policy=_NO_TOOLS_POLICY,
        )

    @classmethod
    def workspace_backed(
        cls,
        workspace: Path,
        *,
        tool_policy: ToolPolicy | ToolPolicyProfile = ToolPolicy.FULL,
    ) -> ToolAccess:
        return cls(
            kind="workspace_backed",
            workspace=workspace,
            tool_policy=tool_policy,
        )

    @property
    def tool_policy(self) -> ToolPolicy | ToolPolicyProfile:
        return self._tool_policy

    def require_workspace(
        self,
        workspace: Path | None,
        *,
        context: str,
    ) -> None:
        if self.kind != "workspace_backed":
            return
        if self.workspace == workspace:
            return
        raise ValueError(
            f"{context} workspace-backed tool access requires worktree {self.workspace}, got {workspace}."
        )


class ProviderStatePreparationAction(Protocol):
    def apply(self) -> None: ...


class ServiceSelectionProvider(Protocol):
    def is_available(self, now: datetime | None = None) -> bool: ...

    def next_wake_time(self) -> datetime: ...

    def mark_exhausted(self, reset_time: datetime | None) -> None: ...


class ResumabilityProvider(Protocol):
    def is_resumable(self, state_dir: Path) -> bool: ...


class ExecutionProvider(Protocol):
    @property
    def name(self) -> str: ...

    def build_command(
        self,
        role: InvocationRole,
        model: str,
        effort: str,
        run_kind: RunKind,
        session_uuid: str | None,
        *,
        tool_policy: ToolPolicy | ToolPolicyProfile | Any | None = None,
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


class ResumableExecutionProvider(
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


__all__ = [
    "AssistantTurn",
    "CredentialFailure",
    "ExecutionProvider",
    "HardError",
    "ModelActivity",
    "ParsedTurn",
    "PromptTokens",
    "ResumableExecutionProvider",
    "ProviderStatePreparationAction",
    "Result",
    "ResumabilityProvider",
    "ServiceSelectionProvider",
    "SessionPlanningProvider",
    "ToolPolicy",
    "ToolAccess",
    "ToolPolicyProfile",
    "TransientError",
    "UnsupportedTokens",
    "UsageLimit",
]
