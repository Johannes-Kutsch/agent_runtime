from __future__ import annotations

import asyncio
import dataclasses
import inspect
import math
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, Mapping, cast

from .contracts import ToolAccess, ToolPolicy, ToolPolicyProfile
from .provider_usage import ProviderUsage
from . import _lifecycle_request_facts as _lifecycle_request_facts_module
from .types import ProviderSelection, ResolvedProvider
from .errors import ProviderUnavailableReason

__all__ = [
    "AgentEvent",
    "Cancelled",
    "Completed",
    "Continuation",
    "EphemeralRunRequest",
    "NewSessionRunRequest",
    "ProviderUnavailable",
    "ProviderUsage",
    "ProviderAuth",
    "ResolvedProvider",
    "ResumedSessionRunRequest",
    "RunResult",
    "RuntimeOutcome",
    "TimedOut",
    "UsageLimited",
]

if TYPE_CHECKING:
    from ._portable_continuation_payload import PortableContinuationPayload

_MISSING_TOOL_POLICY = object()
_DEFAULT_EPHEMERAL_SESSION_NAMESPACE = ""
_PUBLIC_INVOCATION_DIR_NAME = "invocation_dir"


@dataclasses.dataclass
class CancellationToken:
    _event: asyncio.Event = dataclasses.field(
        default_factory=asyncio.Event,
        init=False,
        repr=False,
    )

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> None:
        self._event.set()


def _redacted_credential_value(value: str | None) -> str:
    return "None" if value is None else "'<redacted>'"


def _resolve_public_invocation_dir(
    invocation_dir: Path | None,
    compatibility_kwargs: dict[str, Any],
    *,
    context: str,
) -> Path:
    legacy_worktree = compatibility_kwargs.pop("worktree", None)
    if compatibility_kwargs:
        unexpected_argument = next(iter(compatibility_kwargs))
        raise TypeError(
            f"{context} got an unexpected keyword argument '{unexpected_argument}'."
        )
    if invocation_dir is not None and legacy_worktree is not None:
        raise TypeError(
            f"{context} received conflicting `{_PUBLIC_INVOCATION_DIR_NAME}` and `worktree` values."
        )
    resolved_invocation_dir = (
        invocation_dir if invocation_dir is not None else legacy_worktree
    )
    if resolved_invocation_dir is None:
        raise TypeError(f"{context} requires an `{_PUBLIC_INVOCATION_DIR_NAME}` value.")
    return resolved_invocation_dir


def _public_request_signature(
    *parameter_names: str,
) -> inspect.Signature:
    return inspect.Signature(
        [
            inspect.Parameter(
                name,
                kind=(
                    inspect.Parameter.KEYWORD_ONLY
                    if name == "override"
                    else inspect.Parameter.POSITIONAL_OR_KEYWORD
                ),
            )
            for name in parameter_names
        ]
    )


def _require_json_compatible_resume_state(
    value: Any,
    *,
    path: str = "provider_resume_state",
) -> None:
    if value is None or isinstance(value, str | bool | int):
        return
    if isinstance(value, float):
        if math.isfinite(value):
            return
        raise TypeError("Continuation provider_resume_state must be JSON-compatible.")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _require_json_compatible_resume_state(item, path=f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    "Continuation provider_resume_state must be JSON-compatible."
                )
            _require_json_compatible_resume_state(item, path=f"{path}.{key}")
        return
    raise TypeError("Continuation provider_resume_state must be JSON-compatible.")


@dataclasses.dataclass(frozen=True, init=False)
class Continuation:
    serialized: str

    def __init__(
        self,
        serialized: str | None = None,
        *,
        selected_service: str | None = None,
        selected_model: str | None = None,
        selected_effort: str | None = None,
        tool_access: ToolAccess | None = None,
        provider_resume_state: Any = None,
    ) -> None:
        if serialized is None:
            if (
                selected_service is None
                or selected_model is None
                or selected_effort is None
                or tool_access is None
                or provider_resume_state is None
            ):
                raise TypeError(
                    "Continuation requires either `serialized` or all legacy "
                    "continuation fields."
                )
            _require_json_compatible_resume_state(provider_resume_state)
            from ._portable_continuation_payload import (
                create_portable_continuation_payload,
            )

            serialized = create_portable_continuation_payload(
                service_name=selected_service,
                model=selected_model,
                effort=selected_effort,
                tool_access=tool_access,
                provider_resume_state=provider_resume_state,
            ).serialized

        object.__setattr__(self, "serialized", serialized)

    @property
    def provider_resume_state(self) -> Any:
        return self._payload().provider_resume_state

    @property
    def tool_access(self) -> ToolAccess:
        return self._payload().tool_access

    def _payload(self) -> "PortableContinuationPayload":
        from ._portable_continuation_payload import PortableContinuationPayload

        return PortableContinuationPayload.from_serialized(self.serialized)

    @property
    def serialized_payload(self) -> "PortableContinuationPayload":
        return self._payload()


@dataclasses.dataclass(frozen=True)
class ProviderAuth:
    claude_code_oauth_token: str | None = None
    opencode_api_key: str | None = None

    def __repr__(self) -> str:
        return (
            "ProviderAuth("
            "claude_code_oauth_token="
            f"{_redacted_credential_value(self.claude_code_oauth_token)}, "
            f"opencode_api_key={_redacted_credential_value(self.opencode_api_key)})"
        )


@dataclasses.dataclass(frozen=True)
class AgentEvent:
    type: Literal["agent_message", "agent_tool_call", "other"]
    display_message: str
    raw_provider_output: str


@dataclasses.dataclass(frozen=True)
class RunResult:
    """Run facts carried by every ``RuntimeOutcome``, present even after interruption."""

    output: str
    usage: ProviderUsage | None
    continuation: Continuation | None
    selected: ResolvedProvider


@dataclasses.dataclass(frozen=True)
class Completed:
    pass


@dataclasses.dataclass(frozen=True)
class UsageLimited:
    reset_time: datetime | None


@dataclasses.dataclass(frozen=True)
class ProviderUnavailable:
    reason: ProviderUnavailableReason
    detail: str


@dataclasses.dataclass(frozen=True)
class Cancelled:
    pass


@dataclasses.dataclass(frozen=True)
class TimedOut:
    pass


OutcomeKind = Completed | UsageLimited | ProviderUnavailable | Cancelled | TimedOut


@dataclasses.dataclass(frozen=True)
class RuntimeOutcome:
    kind: OutcomeKind
    result: RunResult


@dataclasses.dataclass(frozen=True, init=False)
class EphemeralRunRequest:
    prompt: str
    invocation_dir: Path
    provider_selection: ProviderSelection
    tool_access: ToolAccess
    timeout_seconds: int = 300
    on_live_output: Callable[[AgentEvent], None] | None = None
    token: CancellationToken | None = None
    argv_transform: (
        Callable[[tuple[str, ...], Path, Mapping[str, str]], tuple[str, ...]] | None
    ) = None

    def __init__(
        self,
        prompt: str,
        invocation_dir: Path | None = None,
        provider_selection: ProviderSelection | None = None,
        tool_policy: ToolPolicy | ToolPolicyProfile | object = _MISSING_TOOL_POLICY,
        tool_access: ToolAccess | object = _MISSING_TOOL_POLICY,
        timeout_seconds: int = 300,
        token: CancellationToken | None = None,
        on_live_output: Callable[[AgentEvent], None] | None = None,
        **compatibility_kwargs: Any,
    ) -> None:
        normalized_request = _lifecycle_request_facts_module._ephemeral_run_request_facts(
            invocation_dir=invocation_dir,
            compatibility_kwargs=compatibility_kwargs,
            provider_selection=provider_selection,
            tool_access=tool_access,
            tool_policy=tool_policy,
            missing_sentinel=_MISSING_TOOL_POLICY,
            session_namespace=_DEFAULT_EPHEMERAL_SESSION_NAMESPACE,
            context="EphemeralRunRequest",
            missing_message="EphemeralRunRequest requires an explicit `tool_policy` value.",
            public_invocation_dir_name=_PUBLIC_INVOCATION_DIR_NAME,
        )

        object.__setattr__(self, "prompt", prompt)
        object.__setattr__(
            self,
            _PUBLIC_INVOCATION_DIR_NAME,
            normalized_request.invocation_dir,
        )
        object.__setattr__(
            self,
            "provider_selection",
            normalized_request.provider_selection,
        )
        object.__setattr__(self, "tool_access", normalized_request.tool_access)
        object.__setattr__(self, "timeout_seconds", timeout_seconds)
        object.__setattr__(self, "on_live_output", on_live_output)
        object.__setattr__(self, "token", token)
        object.__setattr__(self, "argv_transform", normalized_request.argv_transform)

    @property
    def mount_path(self) -> Path:
        return self.invocation_dir

    @property
    def tool_policy(self) -> ToolPolicy | ToolPolicyProfile:
        return self.tool_access.tool_policy


@dataclasses.dataclass(frozen=True, init=False)
class NewSessionRunRequest:
    prompt: str
    invocation_dir: Path
    provider_selection: ProviderSelection
    tool_access: ToolAccess
    timeout_seconds: int = 300
    name: str = "Runtime Agent"
    status_display: Any = None
    work_body: str = ""
    on_live_output: Callable[[AgentEvent], None] | None = None
    token: CancellationToken | None = None
    argv_transform: (
        Callable[[tuple[str, ...], Path, Mapping[str, str]], tuple[str, ...]] | None
    ) = None

    if TYPE_CHECKING:
        session_store: Path | None = None
        _session_namespace: str = ""

    def __init__(
        self,
        prompt: str,
        invocation_dir: Path | None = None,
        provider_selection: ProviderSelection | None = None,
        tool_policy: ToolPolicy | ToolPolicyProfile | object = _MISSING_TOOL_POLICY,
        tool_access: ToolAccess | object = _MISSING_TOOL_POLICY,
        session_store: Path | None = None,
        _session_namespace: str = "",
        timeout_seconds: int = 300,
        name: str = "Runtime Agent",
        status_display: Any = None,
        work_body: str = "",
        on_live_output: Callable[[AgentEvent], None] | None = None,
        token: CancellationToken | None = None,
        **compatibility_kwargs: Any,
    ) -> None:
        normalized_request = _lifecycle_request_facts_module._new_session_run_request_facts(
            invocation_dir=invocation_dir,
            compatibility_kwargs=compatibility_kwargs,
            provider_selection=provider_selection,
            tool_access=tool_access,
            tool_policy=tool_policy,
            session_store=session_store,
            session_namespace=_session_namespace,
            missing_sentinel=_MISSING_TOOL_POLICY,
            context="NewSessionRunRequest",
            missing_message="NewSessionRunRequest requires an explicit `tool_policy` value.",
            public_invocation_dir_name=_PUBLIC_INVOCATION_DIR_NAME,
        )

        object.__setattr__(self, "prompt", prompt)
        object.__setattr__(
            self,
            _PUBLIC_INVOCATION_DIR_NAME,
            normalized_request.invocation_dir,
        )
        object.__setattr__(self, "session_store", normalized_request.session_store)
        object.__setattr__(
            self,
            "provider_selection",
            normalized_request.provider_selection,
        )
        object.__setattr__(self, "tool_access", normalized_request.tool_access)
        object.__setattr__(
            self, "_session_namespace", normalized_request.session_namespace
        )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "status_display", status_display)
        object.__setattr__(self, "work_body", work_body)
        object.__setattr__(self, "timeout_seconds", timeout_seconds)
        object.__setattr__(self, "on_live_output", on_live_output)
        object.__setattr__(self, "token", token)
        object.__setattr__(self, "argv_transform", normalized_request.argv_transform)

    @property
    def mount_path(self) -> Path:
        return self.invocation_dir

    @property
    def tool_policy(self) -> ToolPolicy | ToolPolicyProfile:
        return self.tool_access.tool_policy

    @property
    def _runtime_state_dir(self) -> Path | None:
        return self.session_store


@dataclasses.dataclass(frozen=True, init=False)
class ResumedSessionRunRequest:
    prompt: str
    invocation_dir: Path
    model: str
    effort: str
    continuation: Continuation | None
    provider_auth: ProviderAuth | None
    tool_access: ToolAccess
    timeout_seconds: int = 300
    name: str = "Runtime Agent"
    status_display: Any = None
    work_body: str = ""
    on_live_output: Callable[[AgentEvent], None] | None = None
    token: CancellationToken | None = None
    argv_transform: (
        Callable[[tuple[str, ...], Path, Mapping[str, str]], tuple[str, ...]] | None
    ) = None

    if TYPE_CHECKING:
        session_store: Path | None = None
        _session_namespace: str = ""

    def __init__(
        self,
        prompt: str,
        invocation_dir: Path | None = None,
        continuation: Continuation | None = None,
        provider_auth: ProviderAuth | None = None,
        session_store: Path | None = None,
        _session_namespace: str = "",
        tool_access: ToolAccess | object = _MISSING_TOOL_POLICY,
        timeout_seconds: int = 300,
        name: str = "Runtime Agent",
        status_display: Any = None,
        work_body: str = "",
        on_live_output: Callable[[AgentEvent], None] | None = None,
        token: CancellationToken | None = None,
        **compatibility_kwargs: Any,
    ) -> None:
        argv_transform = compatibility_kwargs.pop("argv_transform", None)
        compatibility_session_namespace = compatibility_kwargs.pop(
            "session_namespace",
            _session_namespace,
        )
        compatibility_runtime_state_dir = compatibility_kwargs.pop(
            "runtime_state_dir",
            session_store,
        )
        if _session_namespace and compatibility_session_namespace != _session_namespace:
            raise TypeError(
                "ResumedSessionRunRequest received conflicting `session_namespace` and `_session_namespace` values."
            )
        if (
            session_store is not None
            and compatibility_runtime_state_dir != session_store
        ):
            raise TypeError(
                "ResumedSessionRunRequest received conflicting `runtime_state_dir` and `session_store` values."
            )
        _session_namespace = compatibility_session_namespace
        session_store = compatibility_runtime_state_dir
        resolved_invocation_dir = _resolve_public_invocation_dir(
            invocation_dir,
            compatibility_kwargs,
            context="ResumedSessionRunRequest",
        )
        if continuation is None:
            raise TypeError("ResumedSessionRunRequest requires a `continuation` value.")
        if "tool_policy" in compatibility_kwargs or isinstance(tool_access, ToolAccess):
            raise TypeError(
                "ResumedSessionRunRequest derives fixed tool access from `continuation` and does not accept `tool_access` or `tool_policy` overrides."
            )
        normalized_request = (
            _lifecycle_request_facts_module._resumed_session_request_facts(
                continuation=continuation,
                worktree=resolved_invocation_dir,
                session_namespace=_session_namespace,
                context="ResumedSessionRunRequest",
                workspace_name=_PUBLIC_INVOCATION_DIR_NAME,
            )
        )

        object.__setattr__(self, "prompt", prompt)
        object.__setattr__(
            self,
            _PUBLIC_INVOCATION_DIR_NAME,
            normalized_request.invocation_dir,
        )
        object.__setattr__(self, "session_store", session_store)
        object.__setattr__(self, "model", normalized_request.model)
        object.__setattr__(self, "effort", normalized_request.effort)
        object.__setattr__(
            self,
            "_session_namespace",
            normalized_request.session_namespace,
        )
        object.__setattr__(self, "continuation", continuation)
        object.__setattr__(self, "provider_auth", provider_auth)
        object.__setattr__(self, "tool_access", normalized_request.tool_access)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "status_display", status_display)
        object.__setattr__(self, "work_body", work_body)
        object.__setattr__(self, "timeout_seconds", timeout_seconds)
        object.__setattr__(self, "on_live_output", on_live_output)
        object.__setattr__(self, "token", token)
        object.__setattr__(self, "argv_transform", argv_transform)

    @property
    def mount_path(self) -> Path:
        return self.invocation_dir

    @property
    def tool_policy(self) -> ToolPolicy | ToolPolicyProfile:
        return self.tool_access.tool_policy

    @property
    def _runtime_state_dir(self) -> Path | None:
        return self.session_store


cast(Any, EphemeralRunRequest).__signature__ = _public_request_signature(
    "prompt",
    "invocation_dir",
    "provider_selection",
    "tool_policy",
    "timeout_seconds",
    "token",
    "on_live_output",
    "argv_transform",
)
cast(Any, NewSessionRunRequest).__signature__ = _public_request_signature(
    "prompt",
    "invocation_dir",
    "provider_selection",
    "tool_policy",
    "session_store",
    "timeout_seconds",
    "name",
    "status_display",
    "work_body",
    "token",
    "on_live_output",
    "argv_transform",
)
cast(Any, ResumedSessionRunRequest).__signature__ = _public_request_signature(
    "prompt",
    "invocation_dir",
    "continuation",
    "provider_auth",
    "session_store",
    "timeout_seconds",
    "on_live_output",
    "token",
    "argv_transform",
)
