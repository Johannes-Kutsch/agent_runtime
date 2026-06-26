from __future__ import annotations

from ._import_isolation import (
    assert_runtime_import_isolation as _assert_runtime_import_isolation,
)

from .contracts import ToolPolicy
from .errors import (
    AgentCredentialFailureError,
    AgentRuntimeError,
    AgentTimeoutError,
    HardAgentError,
    RuntimeConfigurationError,
    TransientAgentError,
    UsageLimitError,
)
from .runtime import (
    AgentEvent,
    Cancelled,
    Completed,
    Continuation,
    ProviderUnavailable,
    ProviderSelection,
    ProviderAuth,
    ProviderUsage,
    RunResult,
    RuntimeClient,
    RuntimeOutcome,
    TimedOut,
    UsageLimited,
)
from .session import RunKind
from .types import ClaudeCodeOAuthToken, ResolvedProvider

__all__ = [
    "AgentCredentialFailureError",
    "AgentEvent",
    "AgentRuntimeError",
    "AgentTimeoutError",
    "Cancelled",
    "ClaudeCodeOAuthToken",
    "Completed",
    "Continuation",
    "HardAgentError",
    "ProviderUnavailable",
    "ProviderAuth",
    "ProviderSelection",
    "ProviderUsage",
    "ResolvedProvider",
    "RunResult",
    "RuntimeClient",
    "RuntimeConfigurationError",
    "RuntimeOutcome",
    "RunKind",
    "TimedOut",
    "ToolPolicy",
    "TransientAgentError",
    "UsageLimited",
    "UsageLimitError",
]


_REMOVED_RUNTIME_PUBLIC_SURFACE_NAMES = {
    "ToolAccess",
    "ToolPolicyProfile",
    "InvocationRole",
    "UsageLimitScope",
}


def __getattr__(name: str) -> object:
    if name in _REMOVED_RUNTIME_PUBLIC_SURFACE_NAMES:
        raise AttributeError(
            f"{name} is not part of the Runtime Public Surface; "
            "import compatibility contracts from `agent_runtime.contracts`."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


_assert_runtime_import_isolation(importer=__name__)
