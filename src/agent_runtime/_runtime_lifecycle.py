from __future__ import annotations

import dataclasses
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from .contracts import ToolAccess, ToolPolicy, ToolPolicyProfile
from .execution_contracts import CancellationToken, WorktreeMount
from .identity import validate_session_namespace
from .invocation_progress import InvocationProgress
from .provider_usage import ProviderUsage
from .provider_session_adapter import ProviderSessionAdapter
from .roles import InvocationRole
from .session import RunKind
from .session_planning import ResumableSessionPlan
from .types import StageSelection, validate_stage_selection
from .usage_limit_scope import UsageLimitScope

__all__ = [
    "Continuation",
    "EphemeralRunRequest",
    "EphemeralRunResult",
    "EphemeralResultMetadata",
    "EphemeralRuntimeMetadata",
    "NewSessionRunRequest",
    "ProviderUsage",
    "ProviderAuth",
    "ResumedSessionRunRequest",
    "RuntimeOutcome",
    "SessionRunResult",
    "SessionRuntimeMetadata",
]

_MISSING_TOOL_POLICY = object()


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
    selected_service: str
    selected_model: str
    selected_effort: str
    tool_access: ToolAccess
    _provider_resume_state_json: str = dataclasses.field(
        repr=False,
    )

    def __init__(
        self,
        *,
        selected_service: str,
        selected_model: str,
        selected_effort: str,
        tool_access: ToolAccess,
        provider_resume_state: Any,
    ) -> None:
        _require_json_compatible_resume_state(provider_resume_state)
        object.__setattr__(self, "selected_service", selected_service)
        object.__setattr__(self, "selected_model", selected_model)
        object.__setattr__(self, "selected_effort", selected_effort)
        object.__setattr__(self, "tool_access", tool_access)
        object.__setattr__(
            self,
            "_provider_resume_state_json",
            json.dumps(
                provider_resume_state,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
        )

    @property
    def provider_resume_state(self) -> Any:
        return json.loads(self._provider_resume_state_json)


@dataclasses.dataclass(frozen=True)
class ProviderAuth:
    claude_code_oauth_token: str | None = None
    opencode_api_key: str | None = None


@dataclasses.dataclass(frozen=True)
class RuntimeOutcome:
    kind: str
    output: str
    result: EphemeralRunResult | SessionRunResult | None = None
    service_name: str | None = None
    reset_time: datetime | None = None
    usage_limit_scope: UsageLimitScope | None = None
    invocation_progress: InvocationProgress | None = None
    continuation: Continuation | None = None
    usage: ProviderUsage | None = None

    @classmethod
    def completed(
        cls,
        *,
        output: str,
        result: EphemeralRunResult | SessionRunResult,
        usage: ProviderUsage | None = None,
    ) -> RuntimeOutcome:
        return cls(kind="completed", output=output, result=result, usage=usage)

    @classmethod
    def usage_limited(
        cls,
        *,
        output: str,
        service_name: str | None,
        reset_time: datetime | None,
        usage_limit_scope: UsageLimitScope | None,
        invocation_progress: InvocationProgress,
        continuation: Continuation | None = None,
        usage: ProviderUsage | None = None,
    ) -> RuntimeOutcome:
        return cls(
            kind="usage_limited",
            output=output,
            service_name=service_name,
            reset_time=reset_time,
            usage_limit_scope=usage_limit_scope,
            invocation_progress=invocation_progress,
            continuation=continuation,
            usage=usage,
        )

    @classmethod
    def no_service_available(
        cls,
        *,
        output: str,
        reset_time: datetime | None,
        usage_limit_scope: UsageLimitScope | None = None,
        invocation_progress: InvocationProgress,
        continuation: Continuation | None = None,
        usage: ProviderUsage | None = None,
    ) -> RuntimeOutcome:
        return cls(
            kind="no_service_available",
            output=output,
            reset_time=reset_time,
            usage_limit_scope=usage_limit_scope,
            invocation_progress=invocation_progress,
            continuation=continuation,
            usage=usage,
        )

    @classmethod
    def cancelled(
        cls,
        *,
        output: str,
        invocation_progress: InvocationProgress,
        continuation: Continuation | None = None,
        usage: ProviderUsage | None = None,
    ) -> RuntimeOutcome:
        return cls(
            kind="cancelled",
            output=output,
            invocation_progress=invocation_progress,
            continuation=continuation,
            usage=usage,
        )

    @classmethod
    def timed_out(
        cls,
        *,
        output: str,
        invocation_progress: InvocationProgress,
        continuation: Continuation | None = None,
        usage: ProviderUsage | None = None,
    ) -> RuntimeOutcome:
        return cls(
            kind="timed_out",
            output=output,
            invocation_progress=invocation_progress,
            continuation=continuation,
            usage=usage,
        )

    @classmethod
    def retryable_provider_failure(
        cls,
        *,
        output: str,
        service_name: str,
        invocation_progress: InvocationProgress,
        continuation: Continuation | None = None,
        usage: ProviderUsage | None = None,
    ) -> RuntimeOutcome:
        return cls(
            kind="retryable_provider_failure",
            output=output,
            service_name=service_name,
            invocation_progress=invocation_progress,
            continuation=continuation,
            usage=usage,
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
    session_namespace: str


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
    worktree: Path
    logs_dir: Path | None
    stage: StageSelection
    role: InvocationRole
    tool_access: ToolAccess
    usage_limit_scope: UsageLimitScope | None = None
    session_namespace: str = ""
    token: CancellationToken | None = None
    auth: ProviderAuth | None = None

    def __init__(
        self,
        prompt: str,
        worktree: Path | WorktreeMount,
        logs_dir: Path | None = None,
        stage: StageSelection | None = None,
        role: InvocationRole | None = None,
        usage_limit_scope: UsageLimitScope | None = None,
        tool_policy: ToolPolicy | ToolPolicyProfile | object = _MISSING_TOOL_POLICY,
        tool_access: ToolAccess | object = _MISSING_TOOL_POLICY,
        session_namespace: str = "",
        token: CancellationToken | None = None,
        auth: ProviderAuth | None = None,
        *,
        override: StageSelection | None = None,
    ) -> None:
        if stage is None:
            stage = override
        elif override is not None and override != stage:
            raise TypeError(
                "EphemeralRunRequest received conflicting `stage` and `override` values."
            )
        if stage is None:
            raise TypeError("EphemeralRunRequest requires a `stage` value.")
        if role is None:
            raise TypeError("EphemeralRunRequest requires a `role` value.")
        worktree_path = (
            worktree.host_path if isinstance(worktree, WorktreeMount) else worktree
        )
        if (
            isinstance(tool_access, ToolAccess)
            and tool_policy is not _MISSING_TOOL_POLICY
        ):
            raise TypeError(
                "EphemeralRunRequest received conflicting `tool_access` and `tool_policy` values."
            )
        if isinstance(tool_access, ToolAccess):
            resolved_tool_access = tool_access
        elif tool_policy is not _MISSING_TOOL_POLICY:
            resolved_tool_access = ToolAccess.workspace_backed(
                worktree_path,
                tool_policy=cast(ToolPolicy | ToolPolicyProfile, tool_policy),
            )
        else:
            raise TypeError(
                "EphemeralRunRequest requires an explicit `tool_access` value."
            )
        resolved_tool_access.require_workspace(
            worktree_path,
            context="EphemeralRunRequest",
        )
        validate_stage_selection(stage)
        validate_session_namespace(session_namespace)

        object.__setattr__(self, "prompt", prompt)
        object.__setattr__(self, "worktree", worktree_path)
        object.__setattr__(self, "logs_dir", logs_dir)
        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "tool_access", resolved_tool_access)
        object.__setattr__(self, "usage_limit_scope", usage_limit_scope)
        object.__setattr__(self, "session_namespace", session_namespace)
        object.__setattr__(self, "token", token)
        object.__setattr__(self, "auth", auth)

    @property
    def mount_path(self) -> Path:
        return self.worktree

    @property
    def override(self) -> StageSelection:
        return self.stage

    @property
    def tool_policy(self) -> ToolPolicy | ToolPolicyProfile:
        return self.tool_access.tool_policy


@dataclasses.dataclass(frozen=True, init=False)
class NewSessionRunRequest:
    prompt: str
    worktree: Path
    runtime_state_dir: Path | None
    logs_dir: Path | None
    stage: StageSelection
    role: InvocationRole
    provider_auth: ProviderAuth | None
    session_store: Any
    provider_session_adapter: ProviderSessionAdapter
    tool_access: ToolAccess
    usage_limit_scope: UsageLimitScope | None = None
    session_namespace: str = ""
    name: str = "Runtime Agent"
    status_display: Any = None
    work_body: str = ""
    token: CancellationToken | None = None

    def __init__(
        self,
        prompt: str,
        worktree: Path | WorktreeMount,
        runtime_state_dir: Path | None = None,
        logs_dir: Path | None = None,
        stage: StageSelection | None = None,
        role: InvocationRole | None = None,
        provider_auth: ProviderAuth | None = None,
        session_store: Any | None = None,
        provider_session_adapter: ProviderSessionAdapter | None = None,
        usage_limit_scope: UsageLimitScope | None = None,
        tool_policy: ToolPolicy | ToolPolicyProfile | object = _MISSING_TOOL_POLICY,
        tool_access: ToolAccess | object = _MISSING_TOOL_POLICY,
        session_namespace: str = "",
        name: str = "Runtime Agent",
        status_display: Any = None,
        work_body: str = "",
        token: CancellationToken | None = None,
        *,
        override: StageSelection | None = None,
    ) -> None:
        if stage is None:
            stage = override
        elif override is not None and override != stage:
            raise TypeError(
                "NewSessionRunRequest received conflicting `stage` and `override` values."
            )
        if stage is None:
            raise TypeError("NewSessionRunRequest requires a `stage` value.")
        if role is None:
            raise TypeError("NewSessionRunRequest requires a `role` value.")
        worktree_path = (
            worktree.host_path if isinstance(worktree, WorktreeMount) else worktree
        )
        if (
            isinstance(tool_access, ToolAccess)
            and tool_policy is not _MISSING_TOOL_POLICY
        ):
            raise TypeError(
                "NewSessionRunRequest received conflicting `tool_access` and `tool_policy` values."
            )
        if isinstance(tool_access, ToolAccess):
            resolved_tool_access = tool_access
        elif tool_policy is not _MISSING_TOOL_POLICY:
            resolved_tool_access = ToolAccess.workspace_backed(
                worktree_path,
                tool_policy=cast(ToolPolicy | ToolPolicyProfile, tool_policy),
            )
        else:
            raise TypeError(
                "NewSessionRunRequest requires an explicit `tool_access` value."
            )
        resolved_tool_access.require_workspace(
            worktree_path,
            context="NewSessionRunRequest",
        )
        validate_stage_selection(stage)
        validate_session_namespace(session_namespace)

        object.__setattr__(self, "prompt", prompt)
        object.__setattr__(self, "worktree", worktree_path)
        object.__setattr__(self, "runtime_state_dir", runtime_state_dir)
        object.__setattr__(self, "logs_dir", logs_dir)
        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "provider_auth", provider_auth)
        object.__setattr__(self, "session_store", session_store)
        object.__setattr__(
            self,
            "provider_session_adapter",
            provider_session_adapter,
        )
        object.__setattr__(self, "tool_access", resolved_tool_access)
        object.__setattr__(self, "usage_limit_scope", usage_limit_scope)
        object.__setattr__(self, "session_namespace", session_namespace)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "status_display", status_display)
        object.__setattr__(self, "work_body", work_body)
        object.__setattr__(self, "token", token)

    @property
    def mount_path(self) -> Path:
        return self.worktree

    @property
    def override(self) -> StageSelection:
        return self.stage

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
    worktree: WorktreeMount
    runtime_state_dir: Path | None
    logs_dir: Path | None
    model: str
    effort: str
    role: InvocationRole
    session_namespace: str
    session_plan: ResumableSessionPlan | None
    continuation: Continuation | None
    provider_auth: ProviderAuth | None
    tool_access: ToolAccess
    usage_limit_scope: UsageLimitScope | None = None
    name: str = "Runtime Agent"
    status_display: Any = None
    work_body: str = ""
    token: CancellationToken | None = None

    def __init__(
        self,
        prompt: str,
        worktree: Path | WorktreeMount,
        runtime_state_dir: Path | None = None,
        logs_dir: Path | None = None,
        model: str | None = None,
        effort: str | None = None,
        session_plan: ResumableSessionPlan | None = None,
        continuation: Continuation | None = None,
        role: InvocationRole | None = None,
        provider_auth: ProviderAuth | None = None,
        session_namespace: str = "",
        usage_limit_scope: UsageLimitScope | None = None,
        tool_policy: ToolPolicy | object = _MISSING_TOOL_POLICY,
        tool_access: ToolAccess | object = _MISSING_TOOL_POLICY,
        name: str = "Runtime Agent",
        status_display: Any = None,
        work_body: str = "",
        token: CancellationToken | None = None,
    ) -> None:
        if continuation is not None and session_plan is not None:
            raise TypeError(
                "ResumedSessionRunRequest received conflicting `session_plan` and `continuation` values."
            )
        worktree_path = (
            worktree.host_path if isinstance(worktree, WorktreeMount) else worktree
        )
        if (
            isinstance(tool_access, ToolAccess)
            and tool_policy is not _MISSING_TOOL_POLICY
        ):
            raise TypeError(
                "ResumedSessionRunRequest received conflicting `tool_access` and `tool_policy` values."
            )
        if continuation is not None:
            if role is None:
                raise TypeError(
                    "ResumedSessionRunRequest requires a `role` value when constructed from a continuation."
                )
            if tool_policy is not _MISSING_TOOL_POLICY or isinstance(
                tool_access, ToolAccess
            ):
                raise TypeError(
                    "ResumedSessionRunRequest derives fixed tool access from `continuation` and does not accept `tool_access` or `tool_policy` overrides."
                )
            validate_session_namespace(session_namespace)
            resolved_model = continuation.selected_model if model is None else model
            resolved_effort = continuation.selected_effort if effort is None else effort
            resolved_tool_access = continuation.tool_access
            resolved_role = role
            resolved_session_namespace = session_namespace
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
            if isinstance(tool_access, ToolAccess):
                resolved_tool_access = tool_access
            elif tool_policy is not _MISSING_TOOL_POLICY:
                resolved_tool_access = ToolAccess.workspace_backed(
                    worktree_path,
                    tool_policy=cast(ToolPolicy | ToolPolicyProfile, tool_policy),
                )
            else:
                raise TypeError(
                    "ResumedSessionRunRequest requires an explicit `tool_policy` value."
                )
            resolved_model = model
            resolved_effort = effort
            resolved_role = session_plan.role
            resolved_session_namespace = session_plan.namespace
            usage_limit_scope = session_plan.usage_limit_scope
        resolved_tool_access.require_workspace(
            worktree_path,
            context="ResumedSessionRunRequest",
        )
        resolved_worktree = (
            worktree if isinstance(worktree, WorktreeMount) else WorktreeMount(worktree)
        )

        object.__setattr__(self, "prompt", prompt)
        object.__setattr__(self, "worktree", resolved_worktree)
        object.__setattr__(self, "runtime_state_dir", runtime_state_dir)
        object.__setattr__(self, "logs_dir", logs_dir)
        object.__setattr__(self, "model", resolved_model)
        object.__setattr__(self, "effort", resolved_effort)
        object.__setattr__(self, "role", resolved_role)
        object.__setattr__(self, "session_namespace", resolved_session_namespace)
        object.__setattr__(self, "session_plan", session_plan)
        object.__setattr__(self, "continuation", continuation)
        object.__setattr__(self, "provider_auth", provider_auth)
        object.__setattr__(self, "tool_access", resolved_tool_access)
        object.__setattr__(self, "usage_limit_scope", usage_limit_scope)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "status_display", status_display)
        object.__setattr__(self, "work_body", work_body)
        object.__setattr__(self, "token", token)

    @property
    def mount_path(self) -> Any:
        return self.worktree.host_path

    @property
    def tool_policy(self) -> ToolPolicy | ToolPolicyProfile:
        return self.tool_access.tool_policy
