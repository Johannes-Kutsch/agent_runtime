from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from . import _time
from . import _builtin_provider_stream_interpretation as _stream_interpretation_module
from . import _builtin_runtime_client as _builtin_runtime_client_module
from ._live_runtime_output_exceptions import is_live_runtime_output_exception
from . import (
    _live_runtime_output_timeout_context as _live_runtime_output_timeout_context_module,
)
from ._session_backed_provider_execution import (
    _run_builtin_new_session,
    _run_builtin_resumed_session,
    _session_backed_service_name,
)
from .contracts import ToolPolicy
from .errors import (
    AgentCancelledError,
    AgentTimeoutError,
    ProviderUnavailableError,
    RuntimeConfigurationError,
    UsageLimitError,
)
from ._runtime_lifecycle import (
    AgentEvent,
    Cancelled,
    Completed,
    Continuation,
    EphemeralRunRequest,
    NewSessionRunRequest,
    ProviderUnavailable,
    ProviderAuth,
    ProviderUsage,
    ResumedSessionRunRequest,
    RunResult,
    RuntimeOutcome,
    TimedOut,
    UsageLimited,
)
from .types import ProviderSelection, ResolvedProvider

if TYPE_CHECKING:
    from ._provider_invocation import ProviderInvocationAdapter

_time_module = _time

__all__ = [
    "AgentEvent",
    "Cancelled",
    "Completed",
    "Continuation",
    "EphemeralRunRequest",
    "NewSessionRunRequest",
    "ProviderUnavailable",
    "ProviderAuth",
    "ProviderSelection",
    "ProviderUsage",
    "ResolvedProvider",
    "ResumedSessionRunRequest",
    "RunResult",
    "RuntimeClient",
    "RuntimeOutcome",
    "TimedOut",
    "UsageLimited",
    "ToolPolicy",
]

_REMOVED_RUNTIME_PUBLIC_SURFACE_NAMES = {
    "ToolAccess",
    "ToolPolicyProfile",
    "InvocationRole",
    "UsageLimitScope",
}

for _runtime_export in (
    AgentEvent,
    Cancelled,
    Completed,
    Continuation,
    EphemeralRunRequest,
    NewSessionRunRequest,
    ProviderUnavailable,
    ProviderAuth,
    ProviderSelection,
    ProviderUsage,
    ResolvedProvider,
    ResumedSessionRunRequest,
    RunResult,
    RuntimeOutcome,
    TimedOut,
    UsageLimited,
):
    _runtime_export.__module__ = __name__

_validate_claude_stage = _builtin_runtime_client_module._validate_claude_stage
_validate_opencode_stage = _builtin_runtime_client_module._validate_opencode_stage
_is_claude_subscription_access_denial = (
    _stream_interpretation_module.is_claude_subscription_access_denial
)
_parse_claude_reset_time = _stream_interpretation_module.parse_claude_reset_time
_parse_opencode_reset_time = _stream_interpretation_module.parse_opencode_reset_time
_select_builtin_stage = _builtin_runtime_client_module._select_builtin_stage
_supported_builtin_provider_selection = (
    _builtin_runtime_client_module.supported_builtin_provider_selection
)


def _interrupted_result(exc: Any, selected: ResolvedProvider) -> RunResult:
    return RunResult(
        output="",
        usage=exc.usage,
        continuation=exc.continuation,
        selected=selected,
    )


def _parse_claude_event(line: str) -> list[Any]:
    return _stream_interpretation_module.parse_claude_event_with_dependencies(
        line,
        parse_claude_reset_time=_parse_claude_reset_time,
        is_claude_subscription_access_denial=_is_claude_subscription_access_denial,
    )


def _reduce_claude_stream(
    lines: list[str],
    on_live_output: Callable[[AgentEvent], None] | None = None,
) -> tuple[str, ProviderUsage | None]:
    return _stream_interpretation_module.reduce_claude_stream_with_dependencies(
        lines,
        parse_claude_event=_parse_claude_event,
        on_live_output=on_live_output,
    )


def _reduce_opencode_stream(
    lines: list[str],
    on_live_output: Callable[[AgentEvent], None] | None = None,
) -> tuple[str, ProviderUsage | None]:
    return _stream_interpretation_module.reduce_opencode_stream(
        lines,
        on_live_output=on_live_output,
    )


def _raise_if_live_output_exception(exc: BaseException) -> None:
    if is_live_runtime_output_exception(exc):
        raise exc


def _run_builtin_session_outcome(
    call: Any,
    *,
    service_name: str = "",
    selected_model: str = "",
    selected_effort: str = "",
) -> RuntimeOutcome:
    def _selected(service: str | None = None) -> ResolvedProvider:
        return ResolvedProvider(
            service=service or service_name,
            model=selected_model,
            effort=selected_effort,
        )

    try:
        return RuntimeOutcome(kind=Completed(), result=call())
    except AgentCancelledError as exc:
        _raise_if_live_output_exception(exc)
        return RuntimeOutcome(
            kind=Cancelled(),
            result=_interrupted_result(exc, _selected()),
        )
    except AgentTimeoutError as exc:
        _raise_if_live_output_exception(exc)
        return RuntimeOutcome(
            kind=TimedOut(),
            result=_interrupted_result(exc, _selected()),
        )
    except ProviderUnavailableError as exc:
        _raise_if_live_output_exception(exc)
        return RuntimeOutcome(
            kind=ProviderUnavailable(reason=exc.reason, detail=str(exc)),
            result=_interrupted_result(exc, _selected(exc.service_name)),
        )
    except UsageLimitError as exc:
        _raise_if_live_output_exception(exc)
        return RuntimeOutcome(
            kind=UsageLimited(reset_time=exc.reset_time),
            result=_interrupted_result(exc, _selected(exc.service_name)),
        )


class RuntimeClient:
    async def run_ephemeral(self, request: EphemeralRunRequest) -> RuntimeOutcome:
        selected_provider_selection = _supported_builtin_provider_selection(
            request.provider_selection
        )
        if selected_provider_selection is None:
            raise RuntimeConfigurationError(
                "RuntimeClient requires at least one supported built-in service candidate."
            )
        selected = ResolvedProvider(
            service=selected_provider_selection.service,
            model=selected_provider_selection.model,
            effort=selected_provider_selection.effort,
        )
        try:
            result = _run_builtin_ephemeral(request)
        except AgentTimeoutError as exc:
            _raise_if_live_output_exception(exc)
            return RuntimeOutcome(
                kind=TimedOut(),
                result=_interrupted_result(exc, selected),
            )
        except UsageLimitError as exc:
            _raise_if_live_output_exception(exc)
            return RuntimeOutcome(
                kind=UsageLimited(reset_time=exc.reset_time),
                result=_interrupted_result(exc, selected),
            )
        return RuntimeOutcome(kind=Completed(), result=result)

    async def run_new_session(self, request: NewSessionRunRequest) -> RuntimeOutcome:
        return _live_runtime_output_timeout_context_module._run_with_live_runtime_output_timeout_context(
            request.on_live_output,
            request.timeout_seconds,
            lambda on_live_output: _run_builtin_session_outcome(
                lambda: _run_builtin_new_session(
                    request,
                    on_live_output=on_live_output,
                ),
                service_name=request.provider_selection.service,
                selected_model=request.provider_selection.model,
                selected_effort=request.provider_selection.effort,
            ),
        )

    async def run_resumed_session(
        self,
        request: ResumedSessionRunRequest,
    ) -> RuntimeOutcome:
        return _live_runtime_output_timeout_context_module._run_with_live_runtime_output_timeout_context(
            request.on_live_output,
            request.timeout_seconds,
            lambda on_live_output: _run_builtin_session_outcome(
                lambda: _run_builtin_resumed_session(
                    request,
                    on_live_output=on_live_output,
                ),
                service_name=_session_backed_service_name(request),
                selected_model=request.model,
                selected_effort=request.effort,
            ),
        )


def _run_builtin_ephemeral(
    request: EphemeralRunRequest,
    *,
    provider_invocation_adapter: ProviderInvocationAdapter | None = None,
    select_builtin_stage: Any = _select_builtin_stage,
) -> RunResult:
    return _builtin_runtime_client_module._run_builtin_ephemeral(
        request,
        provider_invocation_adapter=provider_invocation_adapter,
        select_builtin_stage=select_builtin_stage,
        reduce_claude_stream=_reduce_claude_stream,
        reduce_opencode_stream=_reduce_opencode_stream,
    )


def __getattr__(name: str) -> object:
    if name in _REMOVED_RUNTIME_PUBLIC_SURFACE_NAMES:
        raise AttributeError(
            f"{name} is not part of the Runtime Public Surface; "
            "import compatibility contracts from `agent_runtime.contracts`."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
