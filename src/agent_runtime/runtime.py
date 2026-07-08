from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, cast

from . import _time
from . import _builtin_provider_stream_interpretation as _stream_interpretation_module
from . import _builtin_runtime_client as _builtin_runtime_client_module
from ._session_backed_provider_execution import (
    _run_builtin_new_session,
    _run_builtin_resumed_session,
)
from .contracts import ToolPolicy
from .errors import (
    RuntimeConfigurationError,
)
from ._runtime_lifecycle import (
    AgentEvent,
    Cancelled,
    Completed,
    Continuation,
    EphemeralRunRequest,
    ModelNotAvailable,
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
from ._runtime_outcome_folding import (
    _fold_runtime_outcome,
)

if TYPE_CHECKING:
    from ._provider_invocation import ProviderInvocationAdapter

_time_module = _time

__all__ = [
    "AgentEvent",
    "Cancelled",
    "Completed",
    "Continuation",
    "EphemeralRunRequest",
    "ModelNotAvailable",
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
_select_builtin_stage = _builtin_runtime_client_module._select_builtin_stage
_supported_builtin_provider_selection = (
    _builtin_runtime_client_module.supported_builtin_provider_selection
)


def _reduce_opencode_stream(
    lines: list[str],
    on_live_output: Callable[[AgentEvent], None] | None = None,
) -> tuple[str, ProviderUsage | None]:
    return _stream_interpretation_module.reduce_opencode_stream(
        lines,
        on_live_output=on_live_output,
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
        return _fold_runtime_outcome(
            lambda: _run_builtin_ephemeral(
                request,
            ),
            selected_provider=selected,
            preserve_continuation=False,
        )

    async def run_new_session(self, request: NewSessionRunRequest) -> RuntimeOutcome:
        if request.session_store is None:
            raise RuntimeConfigurationError(
                "RuntimeClient Start Session Run requires a `session_store`."
            )
        return _fold_runtime_outcome(
            lambda: _run_builtin_new_session(
                request,
                on_live_output=request.on_live_output,
            ),
            selected_provider=ResolvedProvider(
                service=request.provider_selection.service,
                model=request.provider_selection.model,
                effort=request.provider_selection.effort,
            ),
        )

    async def run_resumed_session(
        self,
        request: ResumedSessionRunRequest,
    ) -> RuntimeOutcome:
        if request.session_store is None:
            raise RuntimeConfigurationError(
                "RuntimeClient Resume Session Run requires a `session_store`."
            )
        return _fold_runtime_outcome(
            lambda: _run_builtin_resumed_session(
                request,
                on_live_output=request.on_live_output,
            ),
            selected_provider=lambda: (
                cast(
                    Continuation,
                    request.continuation,
                ).resume_facts.selected
            ),
        )


def _run_builtin_ephemeral(
    request: EphemeralRunRequest,
    provider_invocation_adapter: ProviderInvocationAdapter | None = None,
    select_builtin_stage: Any = _select_builtin_stage,
) -> RunResult:
    return _builtin_runtime_client_module._run_builtin_ephemeral(
        request,
        provider_invocation_adapter=provider_invocation_adapter,
        select_builtin_stage=select_builtin_stage,
        reduce_opencode_stream=_reduce_opencode_stream,
    )


def __getattr__(name: str) -> object:
    if name in _REMOVED_RUNTIME_PUBLIC_SURFACE_NAMES:
        raise AttributeError(
            f"{name} is not part of the Runtime Public Surface; "
            "import compatibility contracts from `agent_runtime.contracts`."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
