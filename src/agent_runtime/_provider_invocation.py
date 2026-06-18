from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol

from .agent_log import LogicalAgentInvocationLog, WorkInvocationLog
from .errors import RetryableProviderFailureError, UsageLimitError
from .provider_usage import ProviderUsage
from .roles import InvocationRole
from .session import RunKind
from .usage_limit_scope import UsageLimitScope

_FAILURE_STDOUT_LINES_ATTR = "_provider_invocation_stdout_lines"
_FAILURE_PROVIDER_SESSION_ID_ATTR = "_provider_invocation_provider_session_id"

ProviderOutputReducer = Callable[[list[str]], tuple[str, ProviderUsage | None]]
ProviderLoggedOutputReducer = Callable[
    [list[str], WorkInvocationLog], tuple[str, ProviderUsage | None]
]


@dataclasses.dataclass(frozen=True, slots=True)
class ProviderInvocationPrompt:
    content: str
    path: Path | None = None
    cleanup_path: bool = False


@dataclasses.dataclass(frozen=True, slots=True)
class ProviderOutputReductionHooks:
    reduce_output: ProviderOutputReducer
    reduce_logged_output: ProviderLoggedOutputReducer | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class ProviderInvocationLogContext:
    invocation_log: LogicalAgentInvocationLog
    role: InvocationRole
    usage_limit_scope: UsageLimitScope | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class ProviderInvocationRequest:
    command: str
    worktree: Path
    environment: Mapping[str, str]
    prompt: ProviderInvocationPrompt
    run_kind: RunKind
    role: InvocationRole
    usage_limit_scope: UsageLimitScope | None
    log_context: ProviderInvocationLogContext | None
    provider_session_id: str | None
    output_hooks: ProviderOutputReductionHooks


@dataclasses.dataclass(frozen=True, slots=True)
class ProviderInvocationResult:
    output: str
    usage: ProviderUsage | None = None
    stdout_lines: tuple[str, ...] = ()
    provider_session_id: str | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class ProviderInvocationFailure:
    error: UsageLimitError | RetryableProviderFailureError
    stdout_lines: tuple[str, ...] = ()
    provider_session_id: str | None = None


class ProviderInvocationAdapter(Protocol):
    def execute(
        self,
        request: ProviderInvocationRequest,
    ) -> ProviderInvocationResult: ...


def record_provider_invocation_failure_facts(
    error: UsageLimitError | RetryableProviderFailureError,
    *,
    stdout_lines: tuple[str, ...] = (),
    provider_session_id: str | None = None,
) -> None:
    setattr(error, _FAILURE_STDOUT_LINES_ATTR, stdout_lines)
    setattr(error, _FAILURE_PROVIDER_SESSION_ID_ATTR, provider_session_id)


def provider_invocation_failure_stdout_lines(
    error: UsageLimitError | RetryableProviderFailureError,
) -> tuple[str, ...]:
    return tuple(getattr(error, _FAILURE_STDOUT_LINES_ATTR, ()))


def provider_invocation_failure_provider_session_id(
    error: UsageLimitError | RetryableProviderFailureError,
) -> str | None:
    return getattr(error, _FAILURE_PROVIDER_SESSION_ID_ATTR, None)


@dataclasses.dataclass(slots=True)
class InMemoryProviderInvocationAdapter:
    prepared_invocations: list[ProviderInvocationResult | ProviderInvocationFailure] = (
        dataclasses.field(default_factory=list)
    )
    recorded_requests: list[ProviderInvocationRequest] = dataclasses.field(
        default_factory=list
    )

    def execute(
        self,
        request: ProviderInvocationRequest,
    ) -> ProviderInvocationResult:
        self.recorded_requests.append(request)
        if not self.prepared_invocations:
            raise AssertionError("No prepared provider invocation remains.")
        prepared = self.prepared_invocations.pop(0)
        if isinstance(prepared, ProviderInvocationFailure):
            record_provider_invocation_failure_facts(
                prepared.error,
                stdout_lines=prepared.stdout_lines,
                provider_session_id=prepared.provider_session_id,
            )
            raise prepared.error
        return prepared
