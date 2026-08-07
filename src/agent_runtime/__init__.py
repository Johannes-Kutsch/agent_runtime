from __future__ import annotations

from ._import_isolation import (
    assert_runtime_import_isolation as _assert_runtime_import_isolation,
)
from ._provider_invocation import (
    InvocationFailureKind,
    ProviderInvocationAdapter,
    ProviderInvocationFailure,
    ProviderInvocationRequest,
    ProviderInvocationResult,
    consume_provider_stdout_lines,
)
from .contracts import ToolPolicy
from .errors import (
    AgentCredentialFailureError,
    AgentRuntimeError,
    AgentTimeoutError,
    HardAgentError,
    RuntimeConfigurationError,
    UsageLimitError,
)
from .runtime import (
    AgentEvent,
    Cancelled,
    Completed,
    Continuation,
    ModelNotAvailable,
    ProviderAuth,
    ProviderSelection,
    ProviderUnavailable,
    ProviderUsage,
    RunResult,
    RuntimeClient,
    RuntimeOutcome,
    TimedOut,
    UsageLimited,
)
from .session import RunKind
from .types import ClaudeCodeOAuthToken, ResolvedProvider

InvocationFailureKind.__module__ = __name__
ProviderInvocationAdapter.__module__ = __name__
ProviderInvocationFailure.__module__ = __name__
ProviderInvocationRequest.__module__ = __name__
ProviderInvocationResult.__module__ = __name__

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
    "InvocationFailureKind",
    "ModelNotAvailable",
    "ProviderAuth",
    "ProviderInvocationAdapter",
    "ProviderInvocationFailure",
    "ProviderInvocationRequest",
    "ProviderInvocationResult",
    "ProviderSelection",
    "ProviderUnavailable",
    "ProviderUsage",
    "ResolvedProvider",
    "RunKind",
    "RunResult",
    "RuntimeClient",
    "RuntimeConfigurationError",
    "RuntimeOutcome",
    "TimedOut",
    "ToolPolicy",
    "UsageLimitError",
    "UsageLimited",
    "consume_provider_stdout_lines",
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
