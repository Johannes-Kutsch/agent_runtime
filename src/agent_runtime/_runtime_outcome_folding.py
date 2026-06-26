from __future__ import annotations

from typing import Callable

from ._live_runtime_output_exceptions import is_live_runtime_output_exception
from ._runtime_lifecycle import (
    Cancelled,
    Completed,
    ProviderUnavailable,
    RunResult,
    RuntimeOutcome,
    TimedOut,
    UsageLimited,
)
from .errors import (
    AgentCancelledError,
    AgentTimeoutError,
    ProviderUnavailableError,
    UsageLimitError,
)
from .types import ResolvedProvider

_SelectedProviderFacts = ResolvedProvider | Callable[[], ResolvedProvider]
_Call = Callable[[], RunResult]


def _raise_if_live_runtime_output_exception(exc: BaseException) -> None:
    if is_live_runtime_output_exception(exc):
        raise exc


def _fold_runtime_outcome(
    call: _Call,
    *,
    selected_provider: _SelectedProviderFacts,
) -> RuntimeOutcome:
    if isinstance(selected_provider, ResolvedProvider):

        def resolved_provider() -> ResolvedProvider:
            return selected_provider

    else:
        resolved_provider = selected_provider

    def selected(service_name: str | None = None) -> ResolvedProvider:
        selected_value = resolved_provider()
        return ResolvedProvider(
            service=service_name or selected_value.service,
            model=selected_value.model,
            effort=selected_value.effort,
        )

    def interrupted_result(
        exc: AgentCancelledError
        | AgentTimeoutError
        | UsageLimitError
        | ProviderUnavailableError,
        *,
        service_name: str | None = None,
    ) -> RunResult:
        return RunResult(
            output="",
            usage=exc.usage,
            continuation=exc.continuation,
            selected=selected(service_name),
        )

    try:
        return RuntimeOutcome(kind=Completed(), result=call())
    except AgentCancelledError as exc:
        _raise_if_live_runtime_output_exception(exc)
        return RuntimeOutcome(
            kind=Cancelled(),
            result=interrupted_result(exc),
        )
    except AgentTimeoutError as exc:
        _raise_if_live_runtime_output_exception(exc)
        return RuntimeOutcome(
            kind=TimedOut(),
            result=interrupted_result(exc),
        )
    except ProviderUnavailableError as exc:
        _raise_if_live_runtime_output_exception(exc)
        return RuntimeOutcome(
            kind=ProviderUnavailable(reason=exc.reason, detail=str(exc)),
            result=interrupted_result(exc, service_name=exc.service_name),
        )
    except UsageLimitError as exc:
        _raise_if_live_runtime_output_exception(exc)
        return RuntimeOutcome(
            kind=UsageLimited(reset_time=exc.reset_time),
            result=interrupted_result(exc, service_name=exc.service_name),
        )
