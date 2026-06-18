from __future__ import annotations

from typing import Any

from . import _time as _time_module
from . import _builtin_runtime_client as _builtin_runtime_client_module
from .contracts import (
    ToolAccess,
    ToolPolicy,
    ToolPolicyProfile,
)
from .execution_contracts import (
    PromptRunRequest as _PromptRunRequest,
    PromptRuntimeExecutionAdapter as _PromptRuntimeExecutionAdapter,
    TextOutputAdapter,
    WorkInvocationPresentation,
    WorktreeMount,
)
from .errors import (
    UsageLimitError,
)
from .invocation_progress import InvocationProgress
from . import _runtime_facade_lifecycle as _runtime_facade_lifecycle_module
from ._runtime_lifecycle import (
    Continuation,
    EphemeralResultMetadata,
    EphemeralRunRequest,
    EphemeralRunResult,
    EphemeralRuntimeMetadata,
    NewSessionRunRequest,
    ProviderAuth,
    ResumedSessionRunRequest,
    RuntimeOutcome,
    SessionRunResult,
    SessionRuntimeMetadata,
)
from .service_registry import ServiceRegistry
from .usage_limit_scope import UsageLimitScope

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

_RuntimeIntent = _runtime_facade_lifecycle_module._RuntimeIntent
_EphemeralPreparedProviderRunSession = (
    _runtime_facade_lifecycle_module._EphemeralPreparedProviderRunSession
)
_EphemeralPreparedRunSessionState = (
    _runtime_facade_lifecycle_module._EphemeralPreparedRunSessionState
)
_TrackedPreparedSessionState = (
    _runtime_facade_lifecycle_module._TrackedPreparedSessionState
)
_require_execution_adapter_method = (
    _runtime_facade_lifecycle_module._require_execution_adapter_method
)
_build_run_session = _runtime_facade_lifecycle_module._build_run_session
_latest_provider_run_session = (
    _runtime_facade_lifecycle_module._latest_provider_run_session
)
_invoke_runtime_intent = _runtime_facade_lifecycle_module._invoke_runtime_intent
_run_ephemeral = _runtime_facade_lifecycle_module._run_ephemeral
_run_new_session = _runtime_facade_lifecycle_module._run_new_session
_run_resumed_session = _runtime_facade_lifecycle_module._run_resumed_session
_run_resumed_session_outcome = (
    _runtime_facade_lifecycle_module._run_resumed_session_outcome
)
_provider_state_dir_container_path = (
    _runtime_facade_lifecycle_module._provider_state_dir_container_path
)
_continuation_resume_state = _runtime_facade_lifecycle_module._continuation_resume_state
_build_continuation = _runtime_facade_lifecycle_module._build_continuation
_interruption_continuation = _runtime_facade_lifecycle_module._interruption_continuation
_coerce_service_registry = _runtime_facade_lifecycle_module._coerce_service_registry
_run_ephemeral_outcome = _runtime_facade_lifecycle_module._run_ephemeral_outcome
_run_new_session_outcome = _runtime_facade_lifecycle_module._run_new_session_outcome

for _runtime_export in (
    Continuation,
    EphemeralResultMetadata,
    EphemeralRunRequest,
    EphemeralRunResult,
    EphemeralRuntimeMetadata,
    NewSessionRunRequest,
    ProviderAuth,
    ResumedSessionRunRequest,
    RuntimeOutcome,
    SessionRunResult,
    SessionRuntimeMetadata,
):
    _runtime_export.__module__ = __name__

_selected_service_path = _builtin_runtime_client_module._selected_service_path
_validate_claude_stage = _builtin_runtime_client_module._validate_claude_stage
_claude_command = _builtin_runtime_client_module._claude_command
_claude_env = _builtin_runtime_client_module._claude_env
_is_claude_subscription_access_denial = (
    _builtin_runtime_client_module._is_claude_subscription_access_denial
)
_parse_claude_reset_time = _builtin_runtime_client_module._parse_claude_reset_time
_select_builtin_stage = _builtin_runtime_client_module._select_builtin_stage


def _parse_claude_event(line: str) -> list[Any]:
    return _builtin_runtime_client_module._parse_claude_event_with_dependencies(
        line,
        parse_claude_reset_time=_parse_claude_reset_time,
        is_claude_subscription_access_denial=_is_claude_subscription_access_denial,
    )


def _reduce_claude_stream(lines: list[str]) -> str:
    return _builtin_runtime_client_module._reduce_claude_stream_with_dependencies(
        lines,
        parse_claude_event=_parse_claude_event,
    )


class EphemeralRuntime:
    def __init__(
        self,
        *,
        execution_adapter: EphemeralRuntimeExecutionAdapter,
        service_registry: ServiceRegistry | dict[str, Any] | None = None,
    ) -> None:
        self._service_registry = _coerce_service_registry(service_registry)
        self._execution_adapter = execution_adapter

    async def run_ephemeral(self, request: EphemeralRunRequest) -> RuntimeOutcome:
        return await _run_ephemeral_outcome(
            runner=self._execution_adapter,
            service_registry=self._service_registry,
            request=request,
        )


class NewSessionRuntime:
    def __init__(
        self,
        *,
        execution_adapter: NewSessionRuntimeExecutionAdapter,
        service_registry: ServiceRegistry | dict[str, Any] | None = None,
    ) -> None:
        self._service_registry = _coerce_service_registry(service_registry)
        self._execution_adapter = execution_adapter

    async def run_new_session(self, request: NewSessionRunRequest) -> RuntimeOutcome:
        return await _run_new_session_outcome(
            runner=self._execution_adapter,
            service_registry=self._service_registry,
            request=request,
        )


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
            result = _run_builtin_ephemeral(request)
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


def _run_builtin_ephemeral(request: EphemeralRunRequest) -> EphemeralRunResult:
    return _builtin_runtime_client_module._run_builtin_ephemeral(
        request,
        select_builtin_stage=_select_builtin_stage,
        validate_claude_stage=_validate_claude_stage,
        claude_command=_claude_command,
        claude_env=_claude_env,
        reduce_claude_stream=_reduce_claude_stream,
        selected_service_path=_selected_service_path,
    )


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
