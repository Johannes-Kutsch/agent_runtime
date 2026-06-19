from __future__ import annotations

from ._import_isolation import (
    assert_runtime_import_isolation as _assert_runtime_import_isolation,
)

from .contracts import ToolPolicy
from .errors import (
    AgentCredentialFailureError,
    AgentFailedError,
    AgentRuntimeError,
    AgentTimeoutError,
    HardAgentError,
    RuntimeConfigurationError,
    TransientAgentError,
    UsageLimitError,
)
from .invocation_progress import InvocationProgress
from .runtime import (
    Continuation,
    ProviderAuth,
    ProviderUsage,
    RuntimeClient,
    RuntimeOutcome,
)
from .session import RunKind
from .types import StageSelection

__all__ = [
    "AgentCredentialFailureError",
    "AgentFailedError",
    "AgentRuntimeError",
    "AgentTimeoutError",
    "Continuation",
    "HardAgentError",
    "InvocationProgress",
    "ProviderAuth",
    "ProviderUsage",
    "RuntimeClient",
    "RuntimeConfigurationError",
    "RuntimeOutcome",
    "RunKind",
    "StageSelection",
    "ToolPolicy",
    "TransientAgentError",
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
