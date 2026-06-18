from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
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


class ProviderInvocationAdapter(Protocol):
    def execute(
        self,
        request: ProviderInvocationRequest,
    ) -> ProviderInvocationResult: ...
