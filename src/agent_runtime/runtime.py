from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from . import _time as _time_module
from . import _builtin_runtime_client as _builtin_runtime_client_module
from .contracts import ToolPolicy
from .execution_contracts import (
    PromptRunRequest as _PromptRunRequest,
    PromptRuntimeExecutionAdapter as _PromptRuntimeExecutionAdapter,
    TextOutputAdapter,
    WorkInvocationPresentation,
    WorktreeMount,
)
from .errors import (
    AgentCancelledError,
    AgentTimeoutError,
    NoServiceAvailableError,
    RetryableProviderFailureError,
    RuntimeConfigurationError,
    UsageLimitError,
)
from .invocation_progress import InvocationProgress
from . import _runtime_facade_lifecycle as _runtime_facade_lifecycle_module
from ._runtime_lifecycle import (
    AgentMessageTurn,
    Continuation,
    EphemeralResultMetadata,
    EphemeralRunRequest,
    EphemeralRunResult,
    EphemeralRuntimeMetadata,
    InvocationRecord,
    NewSessionRunRequest,
    ProviderAuth,
    ProviderUsage,
    ResumedSessionRunRequest,
    RuntimeOutcome,
    SessionRunResult,
    SessionRuntimeMetadata,
)
from .service_registry import ServiceRegistry
from .types import ProviderSelection

if TYPE_CHECKING:
    from ._provider_invocation import ProviderInvocationAdapter

__all__ = [
    "Continuation",
    "EphemeralRunRequest",
    "EphemeralRunResult",
    "EphemeralResultMetadata",
    "EphemeralRuntimeMetadata",
    "NewSessionRunRequest",
    "InvocationProgress",
    "InvocationRecord",
    "AgentMessageTurn",
    "ProviderAuth",
    "ProviderSelection",
    "ProviderUsage",
    "ResumedSessionRunRequest",
    "RuntimeClient",
    "RuntimeOutcome",
    "SessionRunResult",
    "SessionRuntimeMetadata",
    "ToolPolicy",
    "WorktreeMount",
]

_REMOVED_RUNTIME_PUBLIC_SURFACE_NAMES = {
    "ToolAccess",
    "ToolPolicyProfile",
    "InvocationRole",
    "UsageLimitScope",
}

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
_run_ephemeral_outcome = _runtime_facade_lifecycle_module._run_ephemeral_outcome
_run_new_session_outcome = _runtime_facade_lifecycle_module._run_new_session_outcome
for _runtime_export in (
    AgentMessageTurn,
    Continuation,
    EphemeralResultMetadata,
    EphemeralRunRequest,
    EphemeralRunResult,
    EphemeralRuntimeMetadata,
    InvocationRecord,
    NewSessionRunRequest,
    ProviderAuth,
    ProviderSelection,
    ProviderUsage,
    ResumedSessionRunRequest,
    RuntimeOutcome,
    SessionRunResult,
    SessionRuntimeMetadata,
):
    _runtime_export.__module__ = __name__

_selected_service_path = _builtin_runtime_client_module._selected_service_path
_validate_claude_stage = _builtin_runtime_client_module._validate_claude_stage
_validate_opencode_stage = _builtin_runtime_client_module._validate_opencode_stage
_claude_command = _builtin_runtime_client_module._claude_command
_claude_env = _builtin_runtime_client_module._claude_env
_opencode_command = _builtin_runtime_client_module._opencode_command
_opencode_env = _builtin_runtime_client_module._opencode_env
_is_claude_subscription_access_denial = (
    _builtin_runtime_client_module._is_claude_subscription_access_denial
)
_parse_claude_reset_time = _builtin_runtime_client_module._parse_claude_reset_time
_parse_opencode_reset_time = _builtin_runtime_client_module._parse_opencode_reset_time
_select_builtin_stage = _builtin_runtime_client_module._select_builtin_stage
_supported_builtin_provider_selection = (
    _builtin_runtime_client_module.supported_builtin_provider_selection
)
_supported_builtin_stage = _builtin_runtime_client_module.supported_builtin_stage
_BuiltInAvailabilityState = _builtin_runtime_client_module.BuiltInAvailabilityState
_run_builtin_new_session = _builtin_runtime_client_module._run_builtin_new_session
_run_builtin_resumed_session = (
    _builtin_runtime_client_module._run_builtin_resumed_session
)


def _exception_invocation_records(exc: object) -> tuple[InvocationRecord, ...]:
    return tuple(getattr(exc, "_runtime_invocation_records", ()))


def _parse_claude_event(line: str) -> list[Any]:
    return _builtin_runtime_client_module._parse_claude_event_with_dependencies(
        line,
        parse_claude_reset_time=_parse_claude_reset_time,
        is_claude_subscription_access_denial=_is_claude_subscription_access_denial,
    )


def _reduce_claude_stream(
    lines: list[str],
    on_live_output: Callable[[AgentMessageTurn], None] | None = None,
) -> tuple[str, ProviderUsage | None]:
    return _builtin_runtime_client_module._reduce_claude_stream_with_dependencies(
        lines,
        parse_claude_event=_parse_claude_event,
        on_live_output=on_live_output,
    )


def _reduce_opencode_stream(
    lines: list[str],
    on_live_output: Callable[[AgentMessageTurn], None] | None = None,
) -> str:
    return _builtin_runtime_client_module._reduce_opencode_stream(
        lines,
        on_live_output=on_live_output,
    )


def _run_builtin_session_outcome(
    call: Any,
) -> RuntimeOutcome:
    try:
        return call()
    except AgentCancelledError as exc:
        return RuntimeOutcome.cancelled(
            output="",
            invocation_progress=exc.invocation_progress,
            continuation=exc.continuation,
            usage=exc.usage,
        )
    except AgentTimeoutError as exc:
        return RuntimeOutcome.timed_out(
            output="",
            invocation_progress=exc.invocation_progress,
            continuation=exc.continuation,
            usage=exc.usage,
        )
    except NoServiceAvailableError as exc:
        return RuntimeOutcome.no_service_available(
            output="",
            reset_time=exc.reset_time,
            invocation_progress=exc.invocation_progress,
            continuation=exc.continuation,
            usage=exc.usage,
        )
    except RetryableProviderFailureError as exc:
        if getattr(exc, "_is_live_output_exception", False):
            raise
        return RuntimeOutcome.retryable_provider_failure(
            output="",
            service_name=exc.service_name,
            invocation_progress=exc.invocation_progress,
            continuation=exc.continuation,
            usage=exc.usage,
            invocation_records=_exception_invocation_records(exc),
        )
    except UsageLimitError as exc:
        if getattr(exc, "_is_live_output_exception", False):
            raise
        return RuntimeOutcome.usage_limited(
            output="",
            service_name=exc.service_name,
            account_label=exc.account_label,
            reset_time=exc.reset_time,
            invocation_progress=exc.invocation_progress,
            continuation=exc.continuation,
            usage=exc.usage,
            invocation_records=_exception_invocation_records(exc),
        )


class RuntimeClient:
    def __init__(self) -> None:
        self._availability = _BuiltInAvailabilityState()

    def run_ephemeral(self, request: EphemeralRunRequest) -> RuntimeOutcome:
        if _supported_builtin_provider_selection(request.provider_selection) is None:
            raise RuntimeConfigurationError(
                "RuntimeClient requires at least one supported built-in service candidate."
            )
        while True:
            now = _time_module.now_local()
            selected_provider_selection = self._availability.first_available_stage(
                request.provider_selection, now=now
            )
            if selected_provider_selection is None:
                return RuntimeOutcome.no_service_available(
                    output="",
                    reset_time=self._availability.next_wake_time(
                        request.provider_selection,
                        now=now,
                    ),
                    invocation_progress=InvocationProgress.NOT_STARTED,
                )
            try:
                result = _run_builtin_ephemeral(
                    request,
                    select_builtin_stage=lambda _stage: selected_provider_selection,
                )
            except UsageLimitError as exc:
                if getattr(exc, "_is_live_output_exception", False):
                    raise
                exhausted_now = _time_module.now_local()
                service_name = exc.service_name or selected_provider_selection.service
                self._availability.mark_exhausted(
                    service_name,
                    reset_time=exc.reset_time,
                    now=exhausted_now,
                )
                if self._availability.has_available_stage(
                    request.provider_selection,
                    now=exhausted_now,
                ):
                    continue
                return RuntimeOutcome.no_service_available(
                    output="",
                    reset_time=self._availability.next_wake_time(
                        request.provider_selection,
                        now=exhausted_now,
                    )
                    or exc.reset_time,
                    invocation_progress=exc.invocation_progress,
                    continuation=exc.continuation,
                    usage=exc.usage,
                )
            return RuntimeOutcome.completed(
                output=result.output,
                result=result,
                usage=result.usage,
            )

    async def run_new_session(self, request: NewSessionRunRequest) -> RuntimeOutcome:
        return _run_builtin_session_outcome(lambda: _run_builtin_new_session(request))

    async def run_resumed_session(
        self,
        request: ResumedSessionRunRequest,
    ) -> RuntimeOutcome:
        return _run_builtin_session_outcome(
            lambda: _run_builtin_resumed_session(request)
        )


def _run_builtin_ephemeral(
    request: EphemeralRunRequest,
    *,
    provider_invocation_adapter: ProviderInvocationAdapter | None = None,
    select_builtin_stage: Any = _select_builtin_stage,
) -> EphemeralRunResult:
    return _builtin_runtime_client_module._run_builtin_ephemeral(
        request,
        provider_invocation_adapter=provider_invocation_adapter,
        select_builtin_stage=select_builtin_stage,
        validate_claude_stage=_validate_claude_stage,
        validate_opencode_stage=_validate_opencode_stage,
        claude_command=_claude_command,
        claude_env=_claude_env,
        reduce_claude_stream=_reduce_claude_stream,
        opencode_command=_opencode_command,
        opencode_env=_opencode_env,
        reduce_opencode_stream=_reduce_opencode_stream,
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


def __getattr__(name: str) -> object:
    if name in _REMOVED_RUNTIME_PUBLIC_SURFACE_NAMES:
        raise AttributeError(
            f"{name} is not part of the Runtime Public Surface; "
            "import compatibility contracts from `agent_runtime.contracts`."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
