from __future__ import annotations

import dataclasses
import subprocess
from collections.abc import Callable, Mapping
from contextlib import nullcontext
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
ProviderSessionIdExtractor = Callable[[list[str]], str | None]


@dataclasses.dataclass(frozen=True, slots=True)
class ProviderInvocationPrompt:
    content: str
    path: Path | None = None
    cleanup_path: bool = False


@dataclasses.dataclass(frozen=True, slots=True)
class ProviderOutputReductionHooks:
    reduce_output: ProviderOutputReducer
    reduce_logged_output: ProviderLoggedOutputReducer | None = None
    extract_provider_session_id: ProviderSessionIdExtractor | None = None


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
class ProviderInvocationPreparedStream:
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


class ProductionProviderInvocationAdapter:
    def execute(
        self,
        request: ProviderInvocationRequest,
    ) -> ProviderInvocationResult:
        prompt_path = request.prompt.path
        if prompt_path is not None:
            prompt_path.write_text(request.prompt.content, encoding="utf-8")

        work_invocation_context = (
            nullcontext()
            if request.log_context is None
            else request.log_context.invocation_log.open_work_invocation(
                role=request.log_context.role,
                run_kind=request.run_kind,
                session_uuid=request.provider_session_id,
                prompt=request.prompt.content,
                usage_limit_scope=request.log_context.usage_limit_scope,
            )
        )

        try:
            with work_invocation_context as work_invocation_log:
                process = subprocess.Popen(
                    request.command,
                    shell=True,
                    cwd=request.worktree,
                    env=dict(request.environment),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                stdout_lines = [] if process.stdout is None else list(process.stdout)
                process.wait()

                def _observed_provider_session_id() -> str | None:
                    provider_session_id = request.provider_session_id
                    if request.output_hooks.extract_provider_session_id is not None:
                        provider_session_id = (
                            request.output_hooks.extract_provider_session_id(
                                stdout_lines
                            )
                            or provider_session_id
                        )
                    return provider_session_id

                if work_invocation_log is None:
                    try:
                        output, usage = request.output_hooks.reduce_output(stdout_lines)
                    except Exception as exc:
                        observed_provider_session_id = _observed_provider_session_id()
                        setattr(
                            exc, "provider_session_id", observed_provider_session_id
                        )
                        if isinstance(
                            exc, (UsageLimitError, RetryableProviderFailureError)
                        ):
                            record_provider_invocation_failure_facts(
                                exc,
                                stdout_lines=tuple(stdout_lines),
                                provider_session_id=observed_provider_session_id,
                            )
                        raise
                else:
                    work_invocation_log.append_provider_chunk(
                        "".join(stdout_lines).encode()
                    )
                    reducer = request.output_hooks.reduce_logged_output or (
                        lambda lines, _work_invocation_log: (
                            request.output_hooks.reduce_output(lines)
                        )
                    )
                    try:
                        output, usage = reducer(stdout_lines, work_invocation_log)
                    except Exception as exc:
                        observed_provider_session_id = _observed_provider_session_id()
                        setattr(
                            exc, "provider_session_id", observed_provider_session_id
                        )
                        if isinstance(
                            exc, (UsageLimitError, RetryableProviderFailureError)
                        ):
                            record_provider_invocation_failure_facts(
                                exc,
                                stdout_lines=tuple(stdout_lines),
                                provider_session_id=observed_provider_session_id,
                            )
                        raise
                return ProviderInvocationResult(
                    output=output,
                    usage=usage,
                    stdout_lines=tuple(stdout_lines),
                    provider_session_id=_observed_provider_session_id(),
                )
        finally:
            if request.prompt.cleanup_path and prompt_path is not None:
                prompt_path.unlink(missing_ok=True)


@dataclasses.dataclass(slots=True)
class InMemoryProviderInvocationAdapter:
    prepared_invocations: list[
        ProviderInvocationResult
        | ProviderInvocationFailure
        | ProviderInvocationPreparedStream
    ] = dataclasses.field(default_factory=list)
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
        if isinstance(prepared, ProviderInvocationPreparedStream):
            stdout_lines = list(prepared.stdout_lines)

            def _observed_provider_session_id() -> str | None:
                provider_session_id = (
                    prepared.provider_session_id or request.provider_session_id
                )
                if request.output_hooks.extract_provider_session_id is not None:
                    provider_session_id = (
                        request.output_hooks.extract_provider_session_id(stdout_lines)
                        or provider_session_id
                    )
                return provider_session_id

            work_invocation_context = (
                nullcontext()
                if request.log_context is None
                else request.log_context.invocation_log.open_work_invocation(
                    role=request.log_context.role,
                    run_kind=request.run_kind,
                    session_uuid=request.provider_session_id,
                    prompt=request.prompt.content,
                    usage_limit_scope=request.log_context.usage_limit_scope,
                )
            )
            with work_invocation_context as work_invocation_log:
                if work_invocation_log is None:

                    def reducer() -> tuple[str, ProviderUsage | None]:
                        return request.output_hooks.reduce_output(stdout_lines)
                else:
                    work_invocation_log.append_provider_chunk(
                        "".join(stdout_lines).encode()
                    )
                    logged_reducer = request.output_hooks.reduce_logged_output or (
                        lambda lines, _work_invocation_log: (
                            request.output_hooks.reduce_output(lines)
                        )
                    )

                    def reducer() -> tuple[str, ProviderUsage | None]:
                        return logged_reducer(stdout_lines, work_invocation_log)

                try:
                    output, usage = reducer()
                except Exception as exc:
                    observed_provider_session_id = _observed_provider_session_id()
                    setattr(exc, "provider_session_id", observed_provider_session_id)
                    if isinstance(
                        exc, (UsageLimitError, RetryableProviderFailureError)
                    ):
                        record_provider_invocation_failure_facts(
                            exc,
                            stdout_lines=tuple(stdout_lines),
                            provider_session_id=observed_provider_session_id,
                        )
                    raise
            return ProviderInvocationResult(
                output=output,
                usage=usage,
                stdout_lines=tuple(stdout_lines),
                provider_session_id=_observed_provider_session_id(),
            )
        return prepared
