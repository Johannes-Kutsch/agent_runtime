from __future__ import annotations

import dataclasses
import subprocess
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Protocol

from .agent_log import LogicalAgentInvocationLog, WorkInvocationLog
from .provider_usage import ProviderUsage
from .roles import InvocationRole
from .session import RunKind
from .usage_limit_scope import UsageLimitScope

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


class ProviderInvocationAdapter(Protocol):
    def execute(
        self,
        request: ProviderInvocationRequest,
    ) -> ProviderInvocationResult: ...


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
                if work_invocation_log is None:
                    output, usage = request.output_hooks.reduce_output(stdout_lines)
                else:
                    work_invocation_log.append_provider_chunk(
                        "".join(stdout_lines).encode()
                    )
                    reducer = request.output_hooks.reduce_logged_output or (
                        lambda lines, _work_invocation_log: (
                            request.output_hooks.reduce_output(lines)
                        )
                    )
                    output, usage = reducer(stdout_lines, work_invocation_log)
                provider_session_id = request.provider_session_id
                if request.output_hooks.extract_provider_session_id is not None:
                    provider_session_id = (
                        request.output_hooks.extract_provider_session_id(stdout_lines)
                        or provider_session_id
                    )
                return ProviderInvocationResult(
                    output=output,
                    usage=usage,
                    stdout_lines=tuple(stdout_lines),
                    provider_session_id=provider_session_id,
                )
        finally:
            if request.prompt.cleanup_path and prompt_path is not None:
                prompt_path.unlink(missing_ok=True)
