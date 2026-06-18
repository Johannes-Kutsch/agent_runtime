from __future__ import annotations

import asyncio
import dataclasses
import json
import math
import os
import re
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

from . import _time as _time_module
from .contracts import (
    AssistantTurn,
    CredentialFailure,
    HardError,
    PromptTokens,
    Result,
    ToolAccess,
    ToolPolicy,
    ToolPolicyProfile,
    TransientError,
    UsageLimit,
)
from .execution_contracts import (
    CancellationToken,
    PromptRunRequest as _PromptRunRequest,
    PromptRuntimeExecutionAdapter as _PromptRuntimeExecutionAdapter,
    RunSessionPlan,
    TextOutputAdapter,
    WorkInvocationPresentation,
    WorkInvocationRequest,
    WorktreeMount,
)
from .errors import (
    AgentCancelledError,
    AgentCredentialFailureError,
    NoServiceAvailableError,
    AgentTimeoutError,
    RetryableProviderFailureError,
    RuntimeConfigurationError,
    UsageLimitError,
)
from .identity import validate_session_namespace
from .invocation_progress import InvocationProgress
from .provider_session_adapter import ProviderSessionAdapter
from .provider_output import reduce_text_output_events
from .roles import InvocationRole
from .service_registry import ServiceRegistry
from .session import RunKind
from .session_planning import (
    ResumableSessionPlan,
    ResumableSessionPlanRequest,
    plan_resumable_session,
)
from .stage_priority_chain import iter_stage_chain
from .types import StageSelection, validate_stage_selection
from .usage_limit_scope import UsageLimitScope
from .work import invoke_work

__all__ = [
    "Continuation",
    "EphemeralRunRequest",
    "EphemeralRunResult",
    "EphemeralResultMetadata",
    "EphemeralRuntime",
    "EphemeralRuntimeExecutionAdapter",
    "EphemeralRuntimeMetadata",
    "NewSessionRunRequest",
    "NewSessionRuntime",
    "NewSessionRuntimeExecutionAdapter",
    "InvocationProgress",
    "ProviderAuth",
    "ResumedSessionRunRequest",
    "ResumedSessionRuntime",
    "ResumedSessionRuntimeExecutionAdapter",
    "RuntimeClient",
    "RuntimeOutcome",
    "SessionRunResult",
    "SessionRuntimeMetadata",
    "ToolAccess",
    "ToolPolicy",
    "ToolPolicyProfile",
    "WorktreeMount",
]

EphemeralRuntimeExecutionAdapter = _PromptRuntimeExecutionAdapter
NewSessionRuntimeExecutionAdapter = _PromptRuntimeExecutionAdapter
ResumedSessionRuntimeExecutionAdapter = _PromptRuntimeExecutionAdapter
_MISSING_TOOL_POLICY = object()

_DEFAULT_RUNTIME_NAME = "Runtime Agent"
_CLAUDE_VALID_MODELS = frozenset({"haiku", "sonnet", "opus"})
_CLAUDE_VALID_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})
_CLAUDE_SUBSCRIPTION_ACCESS_DENIAL_PHRASE = (
    "disabled Claude subscription access for Claude Code"
)
_CLAUDE_RESET_PATTERN = re.compile(
    r"resets\s+"
    r"(?:(?P<month>[A-Za-z]+)\s+(?P<day>\d{1,2}),\s+)?"
    r"(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?(?P<ampm>am|pm)\s+\(UTC\)",
    re.IGNORECASE,
)
_CLAUDE_MONTHS = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sept": 9,
    "sep": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}


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

    @classmethod
    def completed(
        cls,
        *,
        output: str,
        result: EphemeralRunResult | SessionRunResult,
    ) -> RuntimeOutcome:
        return cls(kind="completed", output=output, result=result)

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
    ) -> RuntimeOutcome:
        return cls(
            kind="usage_limited",
            output=output,
            service_name=service_name,
            reset_time=reset_time,
            usage_limit_scope=usage_limit_scope,
            invocation_progress=invocation_progress,
            continuation=continuation,
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
    ) -> RuntimeOutcome:
        return cls(
            kind="no_service_available",
            output=output,
            reset_time=reset_time,
            usage_limit_scope=usage_limit_scope,
            invocation_progress=invocation_progress,
            continuation=continuation,
        )

    @classmethod
    def cancelled(
        cls,
        *,
        output: str,
        invocation_progress: InvocationProgress,
        continuation: Continuation | None = None,
    ) -> RuntimeOutcome:
        return cls(
            kind="cancelled",
            output=output,
            invocation_progress=invocation_progress,
            continuation=continuation,
        )

    @classmethod
    def timed_out(
        cls,
        *,
        output: str,
        invocation_progress: InvocationProgress,
        continuation: Continuation | None = None,
    ) -> RuntimeOutcome:
        return cls(
            kind="timed_out",
            output=output,
            invocation_progress=invocation_progress,
            continuation=continuation,
        )

    @classmethod
    def retryable_provider_failure(
        cls,
        *,
        output: str,
        service_name: str,
        invocation_progress: InvocationProgress,
        continuation: Continuation | None = None,
    ) -> RuntimeOutcome:
        return cls(
            kind="retryable_provider_failure",
            output=output,
            service_name=service_name,
            invocation_progress=invocation_progress,
            continuation=continuation,
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
    stage: StageSelection
    role: InvocationRole
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
        stage: StageSelection | None = None,
        role: InvocationRole | None = None,
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
        if session_store is None:
            raise TypeError("NewSessionRunRequest requires a `session_store` value.")
        if provider_session_adapter is None:
            raise TypeError(
                "NewSessionRunRequest requires a `provider_session_adapter` value."
            )
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
        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "role", role)
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
    model: str
    effort: str
    role: InvocationRole
    session_namespace: str
    session_plan: ResumableSessionPlan | None
    continuation: Continuation | None
    tool_access: ToolAccess
    usage_limit_scope: UsageLimitScope | None = None
    name: str = "Runtime Agent"
    status_display: Any = None
    work_body: str = ""
    token: CancellationToken | None = None

    def __init__(
        self,
        prompt: str,
        worktree: WorktreeMount,
        model: str | None = None,
        effort: str | None = None,
        session_plan: ResumableSessionPlan | None = None,
        continuation: Continuation | None = None,
        role: InvocationRole | None = None,
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
                    worktree.host_path,
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
            worktree.host_path,
            context="ResumedSessionRunRequest",
        )

        object.__setattr__(self, "prompt", prompt)
        object.__setattr__(self, "worktree", worktree)
        object.__setattr__(self, "model", resolved_model)
        object.__setattr__(self, "effort", resolved_effort)
        object.__setattr__(self, "role", resolved_role)
        object.__setattr__(self, "session_namespace", resolved_session_namespace)
        object.__setattr__(self, "session_plan", session_plan)
        object.__setattr__(self, "continuation", continuation)
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


@dataclasses.dataclass(frozen=True)
class _RuntimeIntent:
    run_session: RunSessionPlan
    model: str
    effort: str
    output_adapter: Any = dataclasses.field(repr=False)
    dependencies: Any = dataclasses.field(repr=False)
    presentation: WorkInvocationPresentation = dataclasses.field(
        default_factory=WorkInvocationPresentation
    )
    token: CancellationToken | None = None
    allow_non_typed_resume_retry: bool = False


@dataclasses.dataclass
class _EphemeralPreparedProviderRunSession:
    run_kind: RunKind = RunKind.FRESH
    provider_session_id: str | None = None

    def record_provider_session_id(self, provider_session_id: str) -> None:
        self.provider_session_id = provider_session_id

    def record_successful_run(self) -> None:
        return None


class _EphemeralPreparedRunSessionState:
    provider_state_dir_container_path: str | None = None

    def __init__(self) -> None:
        self._provider_run_session = _EphemeralPreparedProviderRunSession()

    def prepare_for_run(self) -> None:
        return None

    def initial_provider_run_session(self) -> _EphemeralPreparedProviderRunSession:
        return self._provider_run_session

    def resumable_provider_run_session(self) -> _EphemeralPreparedProviderRunSession:
        return self._provider_run_session

    def protocol_reprompt_provider_run_session(self) -> None:
        return None


@dataclasses.dataclass
class _TrackedPreparedSessionState:
    _prepared_session: Any
    latest_provider_run_session: Any | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._prepared_session, name)


def _selected_service_path(
    override: StageSelection,
    *,
    selected_service: str,
) -> tuple[str, ...]:
    path: list[str] = []
    for node in iter_stage_chain(override):
        if not node.service:
            continue
        path.append(node.service)
        if node.service == selected_service:
            return tuple(path)
    return (selected_service,)


def _validate_claude_stage(stage: StageSelection) -> None:
    if stage.model not in _CLAUDE_VALID_MODELS:
        raise RuntimeConfigurationError(f"Unsupported Claude model {stage.model!r}.")
    if stage.effort not in _CLAUDE_VALID_EFFORTS:
        raise RuntimeConfigurationError(f"Unsupported Claude effort {stage.effort!r}.")


def _claude_command(
    *,
    model: str,
    effort: str,
    tool_access: ToolAccess,
) -> str:
    flags = (
        "--verbose --dangerously-skip-permissions --output-format stream-json -p -"
        " --disable-slash-commands --exclude-dynamic-system-prompt-sections"
    )
    if tool_access.kind == "none":
        flags += ' --tools none --disallowedTools "all"'
    flags += " --strict-mcp-config --mcp-config '{\"mcpServers\":{}}'"
    flags += f" --model {model}"
    flags += f" --effort {effort}"
    return f"claude {flags} < /tmp/.pycastle_prompt"


def _claude_env(
    *,
    auth: ProviderAuth | None,
    state_dir_container_path: str | None = None,
) -> dict[str, str]:
    env = dict(os.environ)
    token = None if auth is None else auth.claude_code_oauth_token
    if token:
        env["CLAUDE_CODE_OAUTH_TOKEN"] = token
    if state_dir_container_path:
        env["CLAUDE_CONFIG_DIR"] = state_dir_container_path
    return env


def _is_claude_subscription_access_denial(event: dict[str, Any]) -> bool:
    result = event.get("result")
    return (
        event.get("is_error") is True
        and event.get("api_error_status") == 403
        and isinstance(result, str)
        and _CLAUDE_SUBSCRIPTION_ACCESS_DENIAL_PHRASE.lower() in result.lower()
    )


def _parse_claude_event(line: str) -> list[Any]:
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return []
    if not isinstance(event, dict):
        return []
    if event.get("api_error_status") == 429:
        return [
            UsageLimit(
                reset_time=_parse_claude_reset_time(event.get("result")),
                raw_message=line,
            )
        ]
    if _is_claude_subscription_access_denial(event):
        return [
            CredentialFailure(
                raw_message=line,
                service_name="claude",
                source_observations=(),
                status_code=403,
            )
        ]
    if event.get("is_error") and event.get("type") == "result":
        status = event.get("api_error_status")
        if status is None or (isinstance(status, int) and status >= 500):
            return [
                TransientError(
                    status_code=status if isinstance(status, int) else None,
                    raw_message=line,
                )
            ]
        if isinstance(status, int) and 400 <= status < 500:
            return [HardError(status_code=status, raw_message=line)]
        return []
    if event.get("type") == "assistant":
        message = event.get("message") or {}
        content = message.get("content") or []
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        usage = message.get("usage") or {}
        total_tokens = (
            int(usage.get("input_tokens") or 0)
            + int(usage.get("cache_creation_input_tokens") or 0)
            + int(usage.get("cache_read_input_tokens") or 0)
        )
        parsed_events: list[Any] = []
        if total_tokens > 0:
            parsed_events.append(PromptTokens(count=total_tokens))
        if parts:
            parsed_events.append(AssistantTurn(text="\n\n".join(parts)))
        return parsed_events
    if event.get("type") == "result" and isinstance(event.get("result"), str):
        return [Result(text=cast(str, event["result"]))]
    return []


def _reduce_claude_stream(lines: list[str]) -> str:
    parsed_events: list[Any] = []
    for line in lines:
        parsed_events.extend(_parse_claude_event(line))
    return reduce_text_output_events(
        parsed_events,
        lambda _turn: None,
        provider="claude",
    )


def _parse_claude_reset_time(retry_text: object) -> datetime | None:
    if not isinstance(retry_text, str):
        return None
    match = _CLAUDE_RESET_PATTERN.search(retry_text)
    if match is None:
        return None
    hour = int(match.group("hour"))
    if not 1 <= hour <= 12:
        return None
    ampm = match.group("ampm").lower()
    if ampm == "pm" and hour != 12:
        hour += 12
    elif ampm == "am" and hour == 12:
        hour = 0
    minute = int(match.group("minute") or 0)
    if not 0 <= minute <= 59:
        return None
    now_local = _time_module.now_local()
    utc_now = now_local.astimezone(timezone.utc)
    month_text = match.group("month")
    day_text = match.group("day")
    if month_text is not None or day_text is not None:
        if month_text is None or day_text is None:
            return None
        month = _CLAUDE_MONTHS.get(month_text.lower())
        if month is None:
            return None
        utc_dt = datetime(
            utc_now.year,
            month,
            int(day_text),
            hour,
            minute,
            tzinfo=timezone.utc,
        )
        local_dt = utc_dt.astimezone(now_local.tzinfo)
        if local_dt < now_local - timedelta(days=31):
            return datetime(
                utc_dt.year + 1,
                month,
                int(day_text),
                hour,
                minute,
                tzinfo=timezone.utc,
            ).astimezone(now_local.tzinfo)
        return local_dt
    utc_dt = datetime.combine(
        utc_now.date(),
        datetime.min.time(),
        tzinfo=timezone.utc,
    ).replace(hour=hour, minute=minute)
    if utc_dt < utc_now - timedelta(minutes=2):
        utc_dt += timedelta(days=1)
    return utc_dt.astimezone(now_local.tzinfo)


def _select_builtin_stage(stage: StageSelection) -> StageSelection:
    for candidate in iter_stage_chain(stage):
        if candidate.service == "claude":
            return candidate
    raise RuntimeConfigurationError(
        "RuntimeClient requires at least one supported built-in service candidate."
    )


def _require_execution_adapter_method(
    adapter: _PromptRuntimeExecutionAdapter,
    method_name: str,
) -> Any:
    method = getattr(adapter, method_name, None)
    if callable(method):
        return method
    raise RuntimeConfigurationError(
        f"Prompt runtime requires an execution adapter with callable `{method_name}()`."
    )


def _build_run_session(
    *,
    mount_path: Any,
    role: InvocationRole,
    session_namespace: str,
    service: Any,
    container_workspace: str,
    usage_limit_scope: UsageLimitScope | None = None,
    run_kind: RunKind = RunKind.FRESH,
    provider_session_id: str | None = None,
    provider_resume_state: Any = None,
    provider_state_dir_container_path: str | None = None,
    exact_transcript_match: bool = False,
) -> RunSessionPlan:
    return RunSessionPlan(
        mount_path=mount_path,
        role=role,
        session_namespace=session_namespace,
        service=service,
        container_workspace=container_workspace,
        usage_limit_scope=usage_limit_scope,
        run_kind=run_kind,
        provider_session_id=provider_session_id,
        provider_resume_state=provider_resume_state,
        provider_state_dir_container_path=provider_state_dir_container_path,
        exact_transcript_match=exact_transcript_match,
    )


def _latest_provider_run_session(prepared_session: Any) -> Any:
    provider_run_session = getattr(
        prepared_session,
        "latest_provider_run_session",
        None,
    )
    if provider_run_session is not None:
        return provider_run_session
    return prepared_session.initial_provider_run_session()


async def _invoke_runtime_intent(intent: _RuntimeIntent) -> Any:
    return await invoke_work(
        WorkInvocationRequest(
            run_session=intent.run_session,
            model=intent.model,
            effort=intent.effort,
            output_adapter=intent.output_adapter,
            dependencies=intent.dependencies,
            presentation=intent.presentation,
            token=intent.token,
            allow_non_typed_resume_retry=intent.allow_non_typed_resume_retry,
        )
    )


class EphemeralRuntime:
    def __init__(
        self,
        *,
        execution_adapter: EphemeralRuntimeExecutionAdapter,
        service_registry: ServiceRegistry | dict[str, Any] | None = None,
    ) -> None:
        registry = (
            service_registry
            if isinstance(service_registry, ServiceRegistry)
            else ServiceRegistry(service_registry or {})
        )
        self._service_registry = registry
        self._execution_adapter = execution_adapter

    async def run_ephemeral(self, request: EphemeralRunRequest) -> RuntimeOutcome:
        try:
            result = await _run_ephemeral(
                runner=self._execution_adapter,
                service_registry=self._service_registry,
                request=request,
            )
        except AgentCancelledError as exc:
            return RuntimeOutcome.cancelled(
                output="",
                invocation_progress=exc.invocation_progress,
                continuation=exc.continuation,
            )
        except AgentTimeoutError as exc:
            return RuntimeOutcome.timed_out(
                output="",
                invocation_progress=exc.invocation_progress,
                continuation=exc.continuation,
            )
        except NoServiceAvailableError as exc:
            return RuntimeOutcome.no_service_available(
                output="",
                reset_time=exc.reset_time,
                usage_limit_scope=exc.usage_limit_scope,
                invocation_progress=exc.invocation_progress,
            )
        except RetryableProviderFailureError as exc:
            return RuntimeOutcome.retryable_provider_failure(
                output="",
                service_name=exc.service_name,
                invocation_progress=exc.invocation_progress,
                continuation=exc.continuation,
            )
        except UsageLimitError as exc:
            return RuntimeOutcome.usage_limited(
                output="",
                service_name=exc.service_name,
                reset_time=exc.reset_time,
                usage_limit_scope=exc.usage_limit_scope,
                invocation_progress=exc.invocation_progress,
                continuation=exc.continuation,
            )
        return RuntimeOutcome.completed(output=result.output, result=result)


class NewSessionRuntime:
    def __init__(
        self,
        *,
        execution_adapter: NewSessionRuntimeExecutionAdapter,
        service_registry: ServiceRegistry | dict[str, Any] | None = None,
    ) -> None:
        registry = (
            service_registry
            if isinstance(service_registry, ServiceRegistry)
            else ServiceRegistry(service_registry or {})
        )
        self._service_registry = registry
        self._execution_adapter = execution_adapter

    async def run_new_session(self, request: NewSessionRunRequest) -> RuntimeOutcome:
        try:
            result = await _run_new_session(
                runner=self._execution_adapter,
                service_registry=self._service_registry,
                request=request,
            )
        except AgentCancelledError as exc:
            return RuntimeOutcome.cancelled(
                output="",
                invocation_progress=exc.invocation_progress,
                continuation=exc.continuation,
            )
        except AgentTimeoutError as exc:
            return RuntimeOutcome.timed_out(
                output="",
                invocation_progress=exc.invocation_progress,
                continuation=exc.continuation,
            )
        except NoServiceAvailableError as exc:
            return RuntimeOutcome.no_service_available(
                output="",
                reset_time=exc.reset_time,
                usage_limit_scope=exc.usage_limit_scope,
                invocation_progress=exc.invocation_progress,
            )
        except RetryableProviderFailureError as exc:
            return RuntimeOutcome.retryable_provider_failure(
                output="",
                service_name=exc.service_name,
                invocation_progress=exc.invocation_progress,
                continuation=exc.continuation,
            )
        except UsageLimitError as exc:
            return RuntimeOutcome.usage_limited(
                output="",
                service_name=exc.service_name,
                reset_time=exc.reset_time,
                usage_limit_scope=exc.usage_limit_scope,
                invocation_progress=exc.invocation_progress,
                continuation=exc.continuation,
            )
        return RuntimeOutcome.completed(output=result.output, result=result)


async def _run_resumed_session_outcome(
    *,
    runner: ResumedSessionRuntimeExecutionAdapter,
    request: ResumedSessionRunRequest,
) -> RuntimeOutcome:
    try:
        result = await _run_resumed_session(
            runner=runner,
            request=request,
        )
    except AgentCancelledError as exc:
        return RuntimeOutcome.cancelled(
            output="",
            invocation_progress=exc.invocation_progress,
            continuation=exc.continuation,
        )
    except AgentTimeoutError as exc:
        return RuntimeOutcome.timed_out(
            output="",
            invocation_progress=exc.invocation_progress,
            continuation=exc.continuation,
        )
    except NoServiceAvailableError as exc:
        return RuntimeOutcome.no_service_available(
            output="",
            reset_time=exc.reset_time,
            usage_limit_scope=exc.usage_limit_scope,
            invocation_progress=exc.invocation_progress,
            continuation=exc.continuation,
        )
    except RetryableProviderFailureError as exc:
        return RuntimeOutcome.retryable_provider_failure(
            output="",
            service_name=exc.service_name,
            invocation_progress=exc.invocation_progress,
            continuation=exc.continuation,
        )
    except UsageLimitError as exc:
        return RuntimeOutcome.usage_limited(
            output="",
            service_name=exc.service_name,
            reset_time=exc.reset_time,
            usage_limit_scope=exc.usage_limit_scope,
            invocation_progress=exc.invocation_progress,
            continuation=exc.continuation,
        )
    return RuntimeOutcome.completed(output=result.output, result=result)


class ResumedSessionRuntime:
    def __init__(
        self,
        *,
        execution_adapter: ResumedSessionRuntimeExecutionAdapter,
    ) -> None:
        self._execution_adapter = execution_adapter

    async def run_resumed_session(
        self,
        request: ResumedSessionRunRequest,
    ) -> RuntimeOutcome:
        return await _run_resumed_session_outcome(
            runner=self._execution_adapter,
            request=request,
        )


class RuntimeClient:
    def run_ephemeral(self, request: EphemeralRunRequest) -> RuntimeOutcome:
        try:
            result = asyncio.run(_run_builtin_ephemeral(request))
        except UsageLimitError as exc:
            return RuntimeOutcome.usage_limited(
                output="",
                service_name=exc.service_name,
                reset_time=exc.reset_time,
                usage_limit_scope=exc.usage_limit_scope
                or UsageLimitScope(request.role.value),
                invocation_progress=exc.invocation_progress,
                continuation=exc.continuation,
            )
        return RuntimeOutcome.completed(output=result.output, result=result)


async def _run_builtin_ephemeral(request: EphemeralRunRequest) -> EphemeralRunResult:
    selected_stage = _select_builtin_stage(request.stage)
    _validate_claude_stage(selected_stage)
    if request.auth is None or not request.auth.claude_code_oauth_token:
        raise AgentCredentialFailureError(
            message="Missing Claude Code OAuth token.",
            service_name="claude",
            observations=(),
        )
    process = subprocess.Popen(
        _claude_command(
            model=selected_stage.model,
            effort=selected_stage.effort,
            tool_access=request.tool_access,
        ),
        shell=True,
        cwd=request.worktree,
        env=_claude_env(auth=request.auth),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout_lines = [] if process.stdout is None else list(process.stdout)
    result_text = _reduce_claude_stream(stdout_lines)
    process.wait()
    selected_service_path = _selected_service_path(
        request.stage,
        selected_service="claude",
    )
    result = EphemeralRunResult(
        output=result_text,
        selected_service="claude",
        selected_model=selected_stage.model,
        selected_effort=selected_stage.effort,
        tool_access=request.tool_access,
        used_fallback=len(selected_service_path) > 1,
        metadata=EphemeralResultMetadata(
            selected_service_path=selected_service_path,
            runtime=EphemeralRuntimeMetadata(
                run_kind=RunKind.FRESH,
                session_namespace=request.session_namespace,
            ),
        ),
    )
    return result


async def _run_prompt(
    *,
    runner: _PromptRuntimeExecutionAdapter,
    service_registry: ServiceRegistry,
    request: _PromptRunRequest,
) -> str:
    resolved_override = service_registry.resolve(
        request.stage,
        _time_module.now_local(),
    )
    role = request.role
    resolve_service = _require_execution_adapter_method(runner, "resolve_service")
    build_work_dependencies = _require_execution_adapter_method(
        runner,
        "build_work_dependencies",
    )
    resolved_service = resolve_service(resolved_override.service)
    dependencies = build_work_dependencies(
        name=request.name,
        model=resolved_override.model,
        effort=resolved_override.effort,
        service=resolved_service,
    )
    return await _invoke_runtime_intent(
        _RuntimeIntent(
            run_session=_build_run_session(
                mount_path=request.mount_path,
                role=role,
                session_namespace=request.session_namespace,
                service=resolved_service,
                container_workspace=dependencies.execution.container_workspace,
                usage_limit_scope=request.usage_limit_scope,
            ),
            model=resolved_override.model,
            effort=resolved_override.effort,
            output_adapter=TextOutputAdapter(
                prompt=request.prompt,
                tool_access=request.tool_access,
                workspace=request.worktree.host_path,
            ),
            dependencies=dependencies,
            presentation=WorkInvocationPresentation(
                name=request.name,
                status_display=request.status_display,
                work_body=request.work_body,
            ),
            token=request.token,
        )
    )


async def _run_ephemeral(
    *,
    runner: EphemeralRuntimeExecutionAdapter,
    service_registry: ServiceRegistry,
    request: EphemeralRunRequest,
) -> EphemeralRunResult:
    if not service_registry.has_configured_candidate(request.stage):
        raise RuntimeConfigurationError(
            "Ephemeral runtime requires at least one configured service candidate."
        )

    role = request.role
    resolve_service = _require_execution_adapter_method(runner, "resolve_service")
    build_work_dependencies = _require_execution_adapter_method(
        runner,
        "build_work_dependencies",
    )

    while True:
        now = _time_module.now_local()
        if request.token is not None and request.token.is_cancelled:
            raise AgentCancelledError(
                invocation_progress=InvocationProgress.NOT_STARTED,
            )
        if not service_registry.has_available_for(request.stage, now):
            next_wake_time = service_registry.next_wake_time_for(
                request.stage,
                now,
            )
            raise NoServiceAvailableError(
                reset_time=next_wake_time,
                usage_limit_scope=request.usage_limit_scope
                or UsageLimitScope(role.value),
            )

        resolved_override = service_registry.resolve(request.stage, now)
        resolved_service = resolve_service(resolved_override.service)
        dependencies = build_work_dependencies(
            name=_DEFAULT_RUNTIME_NAME,
            model=resolved_override.model,
            effort=resolved_override.effort,
            service=resolved_service,
        )
        try:
            raw_output = await _invoke_runtime_intent(
                _RuntimeIntent(
                    run_session=_build_run_session(
                        mount_path=request.mount_path,
                        role=role,
                        session_namespace=request.session_namespace,
                        service=resolved_service,
                        container_workspace=dependencies.execution.container_workspace,
                        usage_limit_scope=request.usage_limit_scope,
                    ),
                    model=resolved_override.model,
                    effort=resolved_override.effort,
                    output_adapter=TextOutputAdapter(
                        prompt=request.prompt,
                        tool_access=request.tool_access,
                        workspace=request.worktree,
                    ),
                    dependencies=dataclasses.replace(
                        dependencies,
                        execution=dataclasses.replace(
                            dependencies.execution,
                            prepare_session=lambda _run_session: cast(
                                Any,
                                _EphemeralPreparedRunSessionState(),
                            ),
                        ),
                    ),
                    presentation=WorkInvocationPresentation(
                        name=_DEFAULT_RUNTIME_NAME,
                    ),
                    token=request.token,
                )
            )
        except Exception as exc:
            if isinstance(exc, UsageLimitError):
                raise
            raise

        selected_service_path = _selected_service_path(
            request.stage,
            selected_service=resolved_service.name,
        )
        return EphemeralRunResult(
            output=raw_output if isinstance(raw_output, str) else str(raw_output),
            selected_service=resolved_service.name,
            selected_model=resolved_override.model,
            selected_effort=resolved_override.effort,
            tool_access=request.tool_access,
            used_fallback=len(selected_service_path) > 1,
            metadata=EphemeralResultMetadata(
                selected_service_path=selected_service_path,
                runtime=EphemeralRuntimeMetadata(
                    run_kind=RunKind.FRESH,
                    session_namespace=request.session_namespace,
                ),
            ),
        )


async def _run_new_session(
    *,
    runner: NewSessionRuntimeExecutionAdapter,
    service_registry: ServiceRegistry,
    request: NewSessionRunRequest,
) -> SessionRunResult:
    if not service_registry.has_configured_candidate(request.stage):
        raise RuntimeConfigurationError(
            "New-session runtime requires at least one configured service candidate."
        )
    while True:
        now = _time_module.now_local()
        if request.token is not None and request.token.is_cancelled:
            raise AgentCancelledError(
                invocation_progress=InvocationProgress.NOT_STARTED,
            )
        if not service_registry.has_available_for(request.stage, now):
            raise NoServiceAvailableError(
                reset_time=service_registry.next_wake_time_for(request.stage, now),
                usage_limit_scope=request.usage_limit_scope
                or UsageLimitScope(request.role.value),
            )

        resolved_override = service_registry.resolve(request.stage, now)
        resolve_service = _require_execution_adapter_method(runner, "resolve_service")
        resolved_service = resolve_service(resolved_override.service)
        session_plan = plan_resumable_session(
            ResumableSessionPlanRequest(
                worktree=request.worktree,
                role=request.role,
                namespace=request.session_namespace,
                service=resolved_service,
                session_store=request.session_store,
                provider_session_adapter=request.provider_session_adapter,
                usage_limit_scope=request.usage_limit_scope,
            )
        )
        try:
            return await _run_resumed_session(
                runner=runner,
                request=ResumedSessionRunRequest(
                    prompt=request.prompt,
                    worktree=WorktreeMount(request.worktree),
                    model=resolved_override.model,
                    effort=resolved_override.effort,
                    session_plan=session_plan,
                    tool_access=request.tool_access,
                    name=request.name,
                    status_display=request.status_display,
                    work_body=request.work_body,
                    token=request.token,
                ),
            )
        except UsageLimitError as exc:
            if exc.invocation_progress is not InvocationProgress.NOT_STARTED:
                raise
            service_registry.mark_exhausted(
                resolved_override.service,
                reset_time=exc.reset_time,
            )
            exhausted_now = _time_module.now_local()
            if not service_registry.has_available_for(request.stage, exhausted_now):
                raise NoServiceAvailableError(
                    reset_time=service_registry.next_wake_time_for(
                        request.stage,
                        exhausted_now,
                    ),
                    usage_limit_scope=exc.usage_limit_scope,
                    invocation_progress=exc.invocation_progress,
                ) from exc


async def _run_resumed_session(
    *,
    runner: ResumedSessionRuntimeExecutionAdapter,
    request: ResumedSessionRunRequest,
) -> SessionRunResult:
    resolve_service = _require_execution_adapter_method(runner, "resolve_service")
    build_work_dependencies = _require_execution_adapter_method(
        runner,
        "build_work_dependencies",
    )
    if request.continuation is not None:
        continuation = request.continuation
        service_name = continuation.selected_service
        provider_resume_state = _continuation_resume_state(continuation)
        try:
            service = resolve_service(service_name)
        except NoServiceAvailableError as exc:
            exc.continuation = continuation
            raise
        run_kind = RunKind.RESUME
        provider_session_id = cast(
            str | None,
            provider_resume_state.get("provider_session_id"),
        )
        provider_state_dir_relpath = cast(
            str | None,
            provider_resume_state.get("provider_state_dir_relpath"),
        )
        exact_transcript_match = bool(
            provider_resume_state.get("exact_transcript_match", False)
        )
    else:
        plan = cast(ResumableSessionPlan, request.session_plan)
        service = plan.service
        service_name = service.name
        run_kind = plan.run_kind
        provider_session_id = plan.provider_session_id
        provider_state_dir_relpath = getattr(
            plan,
            "_provider_state_dir_relpath",
            None,
        )
        exact_transcript_match = plan.exact_transcript_match
    dependencies = build_work_dependencies(
        name=request.name,
        model=request.model,
        effort=request.effort,
        service=service,
    )
    prepared_session: Any = None

    def _prepare_session(run_session: RunSessionPlan) -> Any:
        nonlocal prepared_session
        if prepared_session is None:
            prepared_session = _TrackedPreparedSessionState(
                dependencies.execution.prepare_session(run_session)
            )
        return prepared_session

    resumable_dependencies = dataclasses.replace(
        dependencies,
        execution=dataclasses.replace(
            dependencies.execution,
            prepare_session=_prepare_session,
        ),
    )
    run_session = _build_run_session(
        mount_path=request.worktree.host_path,
        role=request.role,
        session_namespace=request.session_namespace,
        service=service,
        container_workspace=dependencies.execution.container_workspace,
        usage_limit_scope=request.usage_limit_scope,
        run_kind=run_kind,
        provider_session_id=provider_session_id,
        provider_resume_state=(
            provider_resume_state if request.continuation is not None else None
        ),
        provider_state_dir_container_path=_provider_state_dir_container_path(
            worktree=request.worktree.host_path,
            provider_state_dir=(
                None if request.continuation is not None else plan.provider_state_dir
            ),
            provider_state_dir_relpath=provider_state_dir_relpath,
            container_workspace=dependencies.execution.container_workspace,
        ),
        exact_transcript_match=exact_transcript_match,
    )
    try:
        output = await _invoke_runtime_intent(
            _RuntimeIntent(
                run_session=run_session,
                model=request.model,
                effort=request.effort,
                output_adapter=TextOutputAdapter(
                    prompt=request.prompt,
                    tool_access=request.tool_access,
                    workspace=request.worktree.host_path,
                ),
                dependencies=resumable_dependencies,
                presentation=WorkInvocationPresentation(
                    name=request.name,
                    status_display=request.status_display,
                    work_body=request.work_body,
                ),
                token=request.token,
            )
        )
    except (
        AgentCancelledError,
        AgentTimeoutError,
        RetryableProviderFailureError,
        UsageLimitError,
    ) as exc:
        exc.continuation = (
            request.continuation
            if request.continuation is not None
            else _interruption_continuation(
                request=request,
                service_name=service_name,
                run_kind=run_kind,
                provider_state_dir_relpath=provider_state_dir_relpath,
                exact_transcript_match=exact_transcript_match,
                prepared_session=prepared_session,
                prepare_session=resumable_dependencies.execution.prepare_session,
                run_session=run_session,
                invocation_progress=exc.invocation_progress,
            )
        )
        raise
    if prepared_session is None:
        prepared_session = resumable_dependencies.execution.prepare_session(run_session)
    provider_run_session = _latest_provider_run_session(prepared_session)
    return SessionRunResult(
        output=output,
        runtime_metadata=SessionRuntimeMetadata(
            service_name=service_name,
            provider_session_id=provider_run_session.provider_session_id,
            run_kind=run_kind,
            session_namespace=request.session_namespace,
            exact_transcript_match=exact_transcript_match,
        ),
        continuation=_build_continuation(
            service_name=service_name,
            model=request.model,
            effort=request.effort,
            tool_access=request.tool_access,
            run_kind=run_kind,
            provider_session_id=provider_run_session.provider_session_id,
            provider_state_dir_relpath=provider_state_dir_relpath,
            exact_transcript_match=exact_transcript_match,
            prepared_session=prepared_session,
            provider_run_session=provider_run_session,
        ),
    )


def _provider_state_dir_container_path(
    *,
    worktree: Path,
    provider_state_dir: Path | None,
    provider_state_dir_relpath: str | None,
    container_workspace: str,
) -> str | None:
    if provider_state_dir is None:
        return (
            None
            if provider_state_dir_relpath is None
            else f"{container_workspace}/{provider_state_dir_relpath}"
        )
    try:
        container_relpath = provider_state_dir.relative_to(worktree)
    except ValueError:
        return (
            None
            if provider_state_dir_relpath is None
            else f"{container_workspace}/{provider_state_dir_relpath}"
        )
    return f"{container_workspace}/{container_relpath.as_posix()}/"


def _continuation_resume_state(continuation: Continuation) -> dict[str, Any]:
    provider_resume_state = continuation.provider_resume_state
    if not isinstance(provider_resume_state, dict):
        raise RuntimeConfigurationError(
            "Continuation provider_resume_state must be a JSON object."
        )
    return provider_resume_state


def _build_continuation(
    *,
    service_name: str,
    model: str,
    effort: str,
    tool_access: ToolAccess,
    run_kind: RunKind,
    provider_session_id: str | None,
    provider_state_dir_relpath: str | None,
    exact_transcript_match: bool,
    prepared_session: Any | None = None,
    provider_run_session: Any | None = None,
) -> Continuation:
    if provider_run_session is not None and hasattr(
        provider_run_session, "latest_provider_resume_state"
    ):
        provider_resume_state = getattr(
            provider_run_session,
            "latest_provider_resume_state",
        )
    elif provider_run_session is not None and hasattr(
        provider_run_session, "provider_resume_state"
    ):
        provider_resume_state = getattr(
            provider_run_session,
            "provider_resume_state",
        )
    elif prepared_session is not None and hasattr(
        prepared_session, "provider_resume_state"
    ):
        provider_resume_state = getattr(prepared_session, "provider_resume_state")
    else:
        provider_resume_state = {
            "run_kind": run_kind.value,
            "provider_session_id": provider_session_id,
            "provider_state_dir_relpath": provider_state_dir_relpath,
            "exact_transcript_match": exact_transcript_match,
        }
    return Continuation(
        selected_service=service_name,
        selected_model=model,
        selected_effort=effort,
        tool_access=tool_access,
        provider_resume_state=provider_resume_state,
    )


def _interruption_continuation(
    *,
    request: ResumedSessionRunRequest,
    service_name: str,
    run_kind: RunKind,
    provider_state_dir_relpath: str | None,
    exact_transcript_match: bool,
    prepared_session: Any,
    prepare_session: Any,
    run_session: RunSessionPlan,
    invocation_progress: InvocationProgress,
) -> Continuation | None:
    if invocation_progress is not InvocationProgress.STARTED:
        return None
    if prepared_session is None:
        prepared_session = prepare_session(run_session)
    provider_run_session = _latest_provider_run_session(prepared_session)
    return _build_continuation(
        service_name=service_name,
        model=request.model,
        effort=request.effort,
        tool_access=request.tool_access,
        run_kind=run_kind,
        provider_session_id=provider_run_session.provider_session_id,
        provider_state_dir_relpath=provider_state_dir_relpath,
        exact_transcript_match=exact_transcript_match,
        prepared_session=prepared_session,
        provider_run_session=provider_run_session,
    )
