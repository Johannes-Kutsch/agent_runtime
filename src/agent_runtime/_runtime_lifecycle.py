from __future__ import annotations

import dataclasses
import inspect
import math
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, cast

from .contracts import ToolAccess, ToolPolicy, ToolPolicyProfile
from .execution_contracts import CancellationToken, WorktreeMount
from .invocation_progress import InvocationProgress
from .provider_usage import ProviderUsage
from ._request_normalization import (
    normalize_continuation_request,
    normalize_provider_selection_request,
    normalize_session_plan_request,
    require_invocation_role,
)
from .provider_session_adapter import ProviderSessionAdapter
from .roles import InvocationRole
from .session import RunKind
from .session_planning import ResumableSessionPlan
from .types import ProviderSelection
from .errors import RuntimeConfigurationError

__all__ = [
    "AgentMessageTurn",
    "Continuation",
    "EphemeralRunRequest",
    "EphemeralRunResult",
    "EphemeralResultMetadata",
    "EphemeralRuntimeMetadata",
    "InvocationRecord",
    "NewSessionRunRequest",
    "ProviderUsage",
    "ProviderAuth",
    "ResumedSessionRunRequest",
    "RuntimeOutcome",
    "SessionRunResult",
    "SessionRuntimeMetadata",
]

if TYPE_CHECKING:
    from ._portable_continuation_payload import PortableContinuationPayload

_MISSING_TOOL_POLICY = object()
_DEFAULT_EPHEMERAL_ROLE = InvocationRole("implementer")
_DEFAULT_EPHEMERAL_SESSION_NAMESPACE = ""
_PUBLIC_INVOCATION_DIR_NAME = "invocation_dir"


def _redacted_credential_value(value: str | None) -> str:
    return "None" if value is None else "'<redacted>'"


def _resolve_public_invocation_dir(
    invocation_dir: Path | WorktreeMount | None,
    compatibility_kwargs: dict[str, Any],
    *,
    context: str,
) -> Path | WorktreeMount:
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
class AgentMessageTurn:
    text: str
    service_name: str


@dataclasses.dataclass(frozen=True)
class InvocationRecord:
    run_kind: RunKind
    service_name: str
    provider_session_id: str | None
    prompt: str
    provider_output: bytes | None = None
    usage: ProviderUsage | None = None


@dataclasses.dataclass(frozen=True)
class RuntimeOutcome:
    kind: str
    output: str
    result: EphemeralRunResult | SessionRunResult | None = None
    service_name: str | None = None
    account_label: str | None = None
    reset_time: datetime | None = None
    invocation_progress: InvocationProgress | None = None
    continuation: Continuation | None = None
    usage: ProviderUsage | None = None
    invocation_records: tuple[InvocationRecord, ...] = ()

    @classmethod
    def completed(
        cls,
        *,
        output: str,
        result: EphemeralRunResult | SessionRunResult,
        invocation_records: tuple[InvocationRecord, ...] = (),
        usage: ProviderUsage | None = None,
    ) -> RuntimeOutcome:
        return cls(
            kind="completed",
            output=output,
            result=result,
            usage=usage,
            invocation_records=invocation_records,
        )

    @classmethod
    def usage_limited(
        cls,
        *,
        output: str,
        service_name: str | None,
        reset_time: datetime | None,
        account_label: str | None = None,
        invocation_progress: InvocationProgress,
        continuation: Continuation | None = None,
        invocation_records: tuple[InvocationRecord, ...] = (),
        usage: ProviderUsage | None = None,
    ) -> RuntimeOutcome:
        return cls(
            kind="usage_limited",
            output=output,
            service_name=service_name,
            account_label=account_label,
            reset_time=reset_time,
            invocation_progress=invocation_progress,
            continuation=continuation,
            usage=usage,
            invocation_records=invocation_records,
        )

    @classmethod
    def no_service_available(
        cls,
        *,
        output: str,
        reset_time: datetime | None,
        invocation_progress: InvocationProgress,
        continuation: Continuation | None = None,
        invocation_records: tuple[InvocationRecord, ...] = (),
        usage: ProviderUsage | None = None,
    ) -> RuntimeOutcome:
        return cls(
            kind="no_service_available",
            output=output,
            reset_time=reset_time,
            invocation_progress=invocation_progress,
            continuation=continuation,
            usage=usage,
            invocation_records=invocation_records,
        )

    @classmethod
    def cancelled(
        cls,
        *,
        output: str,
        invocation_progress: InvocationProgress,
        continuation: Continuation | None = None,
        invocation_records: tuple[InvocationRecord, ...] = (),
        usage: ProviderUsage | None = None,
    ) -> RuntimeOutcome:
        return cls(
            kind="cancelled",
            output=output,
            invocation_progress=invocation_progress,
            continuation=continuation,
            usage=usage,
            invocation_records=invocation_records,
        )

    @classmethod
    def timed_out(
        cls,
        *,
        output: str,
        invocation_progress: InvocationProgress,
        continuation: Continuation | None = None,
        invocation_records: tuple[InvocationRecord, ...] = (),
        usage: ProviderUsage | None = None,
    ) -> RuntimeOutcome:
        return cls(
            kind="timed_out",
            output=output,
            invocation_progress=invocation_progress,
            continuation=continuation,
            usage=usage,
            invocation_records=invocation_records,
        )

    @classmethod
    def retryable_provider_failure(
        cls,
        *,
        output: str,
        service_name: str,
        invocation_progress: InvocationProgress,
        continuation: Continuation | None = None,
        invocation_records: tuple[InvocationRecord, ...] = (),
        usage: ProviderUsage | None = None,
    ) -> RuntimeOutcome:
        return cls(
            kind="retryable_provider_failure",
            output=output,
            service_name=service_name,
            invocation_progress=invocation_progress,
            continuation=continuation,
            usage=usage,
            invocation_records=invocation_records,
        )

    @property
    def runtime_metadata(
        self,
    ) -> EphemeralRuntimeMetadata | SessionRuntimeMetadata:
        result = self.result
        if result is None:
            raise AttributeError("Only completed outcomes carry runtime metadata.")
        if isinstance(result, EphemeralRunResult):
            return result.runtime_metadata
        return result.runtime_metadata

    @property
    def metadata(self) -> EphemeralResultMetadata:
        result = self.result
        if not isinstance(result, EphemeralRunResult):
            raise AttributeError("Completed outcome does not carry ephemeral metadata.")
        return result.metadata

    @property
    def selected_service_path(self) -> tuple[str, ...]:
        result = self.result
        if not isinstance(result, EphemeralRunResult):
            raise AttributeError("Completed outcome does not carry selection metadata.")
        return result.selected_service_path

    @property
    def selected_service(self) -> str:
        result = self.result
        if not isinstance(result, EphemeralRunResult):
            raise AttributeError("Completed outcome does not carry selection metadata.")
        return result.selected_service

    @property
    def selected_model(self) -> str:
        result = self.result
        if not isinstance(result, EphemeralRunResult):
            raise AttributeError("Completed outcome does not carry selection metadata.")
        return result.selected_model

    @property
    def selected_effort(self) -> str:
        result = self.result
        if not isinstance(result, EphemeralRunResult):
            raise AttributeError("Completed outcome does not carry selection metadata.")
        return result.selected_effort

    @property
    def used_fallback(self) -> bool:
        result = self.result
        if not isinstance(result, EphemeralRunResult):
            raise AttributeError("Completed outcome does not carry selection metadata.")
        return result.used_fallback

    @property
    def tool_access(self) -> ToolAccess:
        result = self.result
        if not isinstance(result, EphemeralRunResult):
            raise AttributeError("Completed outcome does not carry tool access.")
        return result.tool_access

    @property
    def raw_output(self) -> str:
        return self.output


@dataclasses.dataclass(frozen=True)
class EphemeralRuntimeMetadata:
    run_kind: RunKind


@dataclasses.dataclass(frozen=True)
class EphemeralResultMetadata:
    selected_service_path: tuple[str, ...]
    runtime: EphemeralRuntimeMetadata


@dataclasses.dataclass(frozen=True)
class EphemeralRunResult:
    output: str
    selected_service: str
    selected_model: str
    selected_effort: str
    tool_access: ToolAccess
    used_fallback: bool
    metadata: EphemeralResultMetadata
    usage: ProviderUsage | None = None

    @property
    def selected_service_path(self) -> tuple[str, ...]:
        return self.metadata.selected_service_path

    @property
    def runtime_metadata(self) -> EphemeralRuntimeMetadata:
        return self.metadata.runtime

    @property
    def raw_output(self) -> str:
        return self.output


@dataclasses.dataclass(frozen=True, init=False)
class EphemeralRunRequest:
    prompt: str
    invocation_dir: Path
    provider_selection: ProviderSelection
    tool_access: ToolAccess
    on_live_output: Callable[[AgentMessageTurn], None] | None = None
    token: CancellationToken | None = None

    def __init__(
        self,
        prompt: str,
        invocation_dir: Path | WorktreeMount | None = None,
        provider_selection: ProviderSelection | None = None,
        tool_policy: ToolPolicy | ToolPolicyProfile | object = _MISSING_TOOL_POLICY,
        tool_access: ToolAccess | object = _MISSING_TOOL_POLICY,
        token: CancellationToken | None = None,
        on_live_output: Callable[[AgentMessageTurn], None] | None = None,
        **compatibility_kwargs: Any,
    ) -> None:
        resolved_invocation_dir = _resolve_public_invocation_dir(
            invocation_dir,
            compatibility_kwargs,
            context="EphemeralRunRequest",
        )
        normalized_request = normalize_provider_selection_request(
            provider_selection=provider_selection,
            role=_DEFAULT_EPHEMERAL_ROLE,
            worktree=resolved_invocation_dir,
            tool_access=tool_access,
            tool_policy=tool_policy,
            missing_sentinel=_MISSING_TOOL_POLICY,
            session_namespace=_DEFAULT_EPHEMERAL_SESSION_NAMESPACE,
            context="EphemeralRunRequest",
            missing_message="EphemeralRunRequest requires an explicit `tool_policy` value.",
            workspace_name=_PUBLIC_INVOCATION_DIR_NAME,
        )

        object.__setattr__(self, "prompt", prompt)
        object.__setattr__(
            self,
            _PUBLIC_INVOCATION_DIR_NAME,
            normalized_request.worktree.path,
        )
        object.__setattr__(
            self,
            "provider_selection",
            normalized_request.provider_selection,
        )
        object.__setattr__(self, "tool_access", normalized_request.tool_access)
        object.__setattr__(self, "on_live_output", on_live_output)
        object.__setattr__(self, "token", token)

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
    role: InvocationRole
    session_store: Any
    provider_session_adapter: ProviderSessionAdapter
    tool_access: ToolAccess
    name: str = "Runtime Agent"
    status_display: Any = None
    work_body: str = ""
    on_live_output: Callable[[AgentMessageTurn], None] | None = None
    token: CancellationToken | None = None

    if TYPE_CHECKING:
        _runtime_state_dir: Path | None = None
        _session_namespace: str = ""

    def __init__(
        self,
        prompt: str,
        invocation_dir: Path | WorktreeMount | None = None,
        provider_selection: ProviderSelection | None = None,
        role: InvocationRole | None = None,
        session_store: Any | None = None,
        provider_session_adapter: ProviderSessionAdapter | None = None,
        tool_policy: ToolPolicy | ToolPolicyProfile | object = _MISSING_TOOL_POLICY,
        tool_access: ToolAccess | object = _MISSING_TOOL_POLICY,
        _runtime_state_dir: Path | None = None,
        _session_namespace: str = "",
        name: str = "Runtime Agent",
        status_display: Any = None,
        work_body: str = "",
        on_live_output: Callable[[AgentMessageTurn], None] | None = None,
        token: CancellationToken | None = None,
        **compatibility_kwargs: Any,
    ) -> None:
        compatibility_session_namespace = compatibility_kwargs.pop(
            "session_namespace",
            _session_namespace,
        )
        compatibility_runtime_state_dir = compatibility_kwargs.pop(
            "runtime_state_dir",
            _runtime_state_dir,
        )
        if _session_namespace and compatibility_session_namespace != _session_namespace:
            raise TypeError(
                "NewSessionRunRequest received conflicting `session_namespace` and `_session_namespace` values."
            )
        if (
            _runtime_state_dir is not None
            and compatibility_runtime_state_dir != _runtime_state_dir
        ):
            raise TypeError(
                "NewSessionRunRequest received conflicting `runtime_state_dir` and `_runtime_state_dir` values."
            )
        _session_namespace = compatibility_session_namespace
        _runtime_state_dir = compatibility_runtime_state_dir
        resolved_invocation_dir = _resolve_public_invocation_dir(
            invocation_dir,
            compatibility_kwargs,
            context="NewSessionRunRequest",
        )
        normalized_request = normalize_provider_selection_request(
            provider_selection=provider_selection,
            role=role or _DEFAULT_EPHEMERAL_ROLE,
            worktree=resolved_invocation_dir,
            tool_access=tool_access,
            tool_policy=tool_policy,
            missing_sentinel=_MISSING_TOOL_POLICY,
            session_namespace=_session_namespace,
            context="NewSessionRunRequest",
            missing_message="NewSessionRunRequest requires an explicit `tool_policy` value.",
            workspace_name=_PUBLIC_INVOCATION_DIR_NAME,
        )

        object.__setattr__(self, "prompt", prompt)
        object.__setattr__(
            self,
            _PUBLIC_INVOCATION_DIR_NAME,
            normalized_request.worktree.path,
        )
        object.__setattr__(self, "_runtime_state_dir", _runtime_state_dir)
        object.__setattr__(
            self,
            "provider_selection",
            normalized_request.provider_selection,
        )
        object.__setattr__(self, "role", normalized_request.role)
        object.__setattr__(self, "session_store", session_store)
        object.__setattr__(
            self,
            "provider_session_adapter",
            provider_session_adapter,
        )
        object.__setattr__(self, "tool_access", normalized_request.tool_access)
        object.__setattr__(
            self, "_session_namespace", normalized_request.session_namespace
        )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "status_display", status_display)
        object.__setattr__(self, "work_body", work_body)
        object.__setattr__(self, "on_live_output", on_live_output)
        object.__setattr__(self, "token", token)

    @property
    def mount_path(self) -> Path:
        return self.invocation_dir

    @property
    def tool_policy(self) -> ToolPolicy | ToolPolicyProfile:
        return self.tool_access.tool_policy


@dataclasses.dataclass(frozen=True)
class SessionRuntimeMetadata:
    service_name: str
    provider_session_id: str | None
    run_kind: RunKind
    session_namespace: str
    exact_transcript_match: bool
    selected_model: str = dataclasses.field(default="", compare=False)
    selected_effort: str = dataclasses.field(default="", compare=False)
    tool_policy: ToolPolicy | ToolPolicyProfile = dataclasses.field(
        default=ToolPolicy.UNRESTRICTED,
        compare=False,
    )


@dataclasses.dataclass(frozen=True)
class SessionRunResult:
    output: str
    runtime_metadata: SessionRuntimeMetadata
    continuation: Continuation | None = dataclasses.field(
        default=None,
        compare=False,
    )


@dataclasses.dataclass(frozen=True, init=False)
class ResumedSessionRunRequest:
    prompt: str
    invocation_dir: WorktreeMount
    model: str
    effort: str
    role: InvocationRole
    session_plan: ResumableSessionPlan | None
    continuation: Continuation | None
    provider_auth: ProviderAuth | None
    tool_access: ToolAccess
    name: str = "Runtime Agent"
    status_display: Any = None
    work_body: str = ""
    on_live_output: Callable[[AgentMessageTurn], None] | None = None
    token: CancellationToken | None = None

    if TYPE_CHECKING:
        _runtime_state_dir: Path | None = None
        _session_namespace: str = ""

    def __init__(
        self,
        prompt: str,
        invocation_dir: Path | WorktreeMount | None = None,
        model: str | None = None,
        effort: str | None = None,
        session_plan: ResumableSessionPlan | None = None,
        continuation: Continuation | None = None,
        role: InvocationRole | None = None,
        provider_auth: ProviderAuth | None = None,
        _runtime_state_dir: Path | None = None,
        _session_namespace: str = "",
        tool_policy: ToolPolicy | object = _MISSING_TOOL_POLICY,
        tool_access: ToolAccess | object = _MISSING_TOOL_POLICY,
        name: str = "Runtime Agent",
        status_display: Any = None,
        work_body: str = "",
        on_live_output: Callable[[AgentMessageTurn], None] | None = None,
        token: CancellationToken | None = None,
        **compatibility_kwargs: Any,
    ) -> None:
        compatibility_session_namespace = compatibility_kwargs.pop(
            "session_namespace",
            _session_namespace,
        )
        compatibility_runtime_state_dir = compatibility_kwargs.pop(
            "runtime_state_dir",
            _runtime_state_dir,
        )
        if _session_namespace and compatibility_session_namespace != _session_namespace:
            raise TypeError(
                "ResumedSessionRunRequest received conflicting `session_namespace` and `_session_namespace` values."
            )
        if (
            _runtime_state_dir is not None
            and compatibility_runtime_state_dir != _runtime_state_dir
        ):
            raise TypeError(
                "ResumedSessionRunRequest received conflicting `runtime_state_dir` and `_runtime_state_dir` values."
            )
        _session_namespace = compatibility_session_namespace
        _runtime_state_dir = compatibility_runtime_state_dir
        resolved_invocation_dir = _resolve_public_invocation_dir(
            invocation_dir,
            compatibility_kwargs,
            context="ResumedSessionRunRequest",
        )
        if continuation is not None and session_plan is not None:
            raise TypeError(
                "ResumedSessionRunRequest received conflicting `session_plan` and `continuation` values."
            )
        if (
            isinstance(tool_access, ToolAccess)
            and tool_policy is not _MISSING_TOOL_POLICY
        ):
            raise TypeError(
                "ResumedSessionRunRequest received conflicting `tool_access` and `tool_policy` values."
            )
        if continuation is not None:
            resolved_role = require_invocation_role(
                role or _DEFAULT_EPHEMERAL_ROLE,
                context="ResumedSessionRunRequest",
                message=(
                    "ResumedSessionRunRequest requires a `role` value when "
                    "constructed from a continuation."
                ),
            )
            if tool_policy is not _MISSING_TOOL_POLICY or isinstance(
                tool_access, ToolAccess
            ):
                raise TypeError(
                    "ResumedSessionRunRequest derives fixed tool access from `continuation` and does not accept `tool_access` or `tool_policy` overrides."
                )
            from ._portable_continuation_payload import (
                read_portable_continuation_payload,
            )

            try:
                continuation_payload = read_portable_continuation_payload(continuation)
                normalized_request = normalize_continuation_request(
                    role=resolved_role,
                    worktree=resolved_invocation_dir,
                    tool_access=continuation_payload.tool_access,
                    session_namespace=_session_namespace,
                    context="ResumedSessionRunRequest",
                    role_message=(
                        "ResumedSessionRunRequest requires a `role` value when "
                        "constructed from a continuation."
                    ),
                    workspace_name=_PUBLIC_INVOCATION_DIR_NAME,
                )
                resolved_model = continuation_payload.model if model is None else model
                resolved_effort = (
                    continuation_payload.effort if effort is None else effort
                )
            except TypeError as exc:
                raise RuntimeConfigurationError(str(exc)) from exc
        else:
            if session_plan is None:
                raise TypeError(
                    "ResumedSessionRunRequest requires either a `session_plan` or `continuation` value."
                )
            if model is None or effort is None:
                raise TypeError(
                    "ResumedSessionRunRequest requires `model` and `effort` when constructed from a session plan."
                )
            if role is not None:
                raise TypeError(
                    "ResumedSessionRunRequest does not accept request-level `role` when `session_plan` is supplied."
                )
            normalized_request = normalize_session_plan_request(
                role=session_plan.role,
                worktree=resolved_invocation_dir,
                tool_access=tool_access,
                tool_policy=tool_policy,
                missing_sentinel=_MISSING_TOOL_POLICY,
                session_namespace=session_plan.namespace,
                context="ResumedSessionRunRequest",
                missing_message="ResumedSessionRunRequest requires an explicit `tool_policy` value.",
                workspace_name=_PUBLIC_INVOCATION_DIR_NAME,
            )
            resolved_model = model
            resolved_effort = effort

        object.__setattr__(self, "prompt", prompt)
        object.__setattr__(
            self,
            _PUBLIC_INVOCATION_DIR_NAME,
            normalized_request.worktree.mount,
        )
        object.__setattr__(self, "_runtime_state_dir", _runtime_state_dir)
        object.__setattr__(self, "model", resolved_model)
        object.__setattr__(self, "effort", resolved_effort)
        object.__setattr__(self, "role", normalized_request.role)
        object.__setattr__(
            self,
            "_session_namespace",
            normalized_request.session_namespace,
        )
        object.__setattr__(self, "session_plan", session_plan)
        object.__setattr__(self, "continuation", continuation)
        object.__setattr__(self, "provider_auth", provider_auth)
        object.__setattr__(self, "tool_access", normalized_request.tool_access)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "status_display", status_display)
        object.__setattr__(self, "work_body", work_body)
        object.__setattr__(self, "on_live_output", on_live_output)
        object.__setattr__(self, "token", token)

    @property
    def mount_path(self) -> Any:
        return self.invocation_dir.host_path

    @property
    def tool_policy(self) -> ToolPolicy | ToolPolicyProfile:
        return self.tool_access.tool_policy


cast(Any, EphemeralRunRequest).__signature__ = _public_request_signature(
    "prompt",
    "invocation_dir",
    "provider_selection",
    "tool_policy",
    "token",
    "on_live_output",
)
cast(Any, NewSessionRunRequest).__signature__ = _public_request_signature(
    "prompt",
    "invocation_dir",
    "provider_selection",
    "role",
    "session_store",
    "provider_session_adapter",
    "tool_policy",
    "name",
    "status_display",
    "work_body",
    "token",
    "on_live_output",
)
cast(Any, ResumedSessionRunRequest).__signature__ = _public_request_signature(
    "prompt",
    "invocation_dir",
    "continuation",
    "provider_auth",
    "model",
    "effort",
    "on_live_output",
    "token",
)
