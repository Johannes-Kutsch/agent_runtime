from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from .identity import validate_runtime_identity_label
from .invocation_progress import InvocationProgress
from .provider_usage import ProviderUsage


class AgentRuntimeError(RuntimeError):
    pass


class RuntimeConfigurationError(AgentRuntimeError):
    pass


class AgentCancelledError(AgentRuntimeError):
    def __init__(
        self,
        *,
        invocation_progress: InvocationProgress = InvocationProgress.NOT_STARTED,
        continuation: Any | None = None,
        usage: ProviderUsage | None = None,
    ) -> None:
        self.invocation_progress = invocation_progress
        self.continuation = continuation
        self.usage = usage
        super().__init__("Agent run was cancelled.")


class AgentTimeoutError(AgentRuntimeError, TimeoutError):
    def __init__(
        self,
        message: str = "",
        invocation_role: str = "",
        worktree_path: Path | None = None,
        invocation_progress: InvocationProgress = InvocationProgress.NOT_STARTED,
        continuation: Any | None = None,
        usage: ProviderUsage | None = None,
    ) -> None:
        self.invocation_role = invocation_role
        self.worktree_path = worktree_path
        self.invocation_progress = invocation_progress
        self.continuation = continuation
        self.usage = usage
        super().__init__(message)


class ProviderUnavailableReason(str, Enum):
    SERVICE_NOT_AVAILABLE = "SERVICE_NOT_AVAILABLE"
    TRANSIENT_API_ERROR = "TRANSIENT_API_ERROR"


class ProviderUnavailableError(AgentRuntimeError):
    def __init__(
        self,
        message: str = "",
        *,
        reason: ProviderUnavailableReason,
        service_name: str,
        invocation_progress: InvocationProgress = InvocationProgress.NOT_STARTED,
        continuation: Any | None = None,
        usage: ProviderUsage | None = None,
    ) -> None:
        validate_runtime_identity_label(
            service_name,
            kind="ProviderUnavailableError service name",
        )
        self.reason = reason
        self.service_name = service_name
        self.invocation_progress = invocation_progress
        self.continuation = continuation
        self.usage = usage
        super().__init__(message)


class UsageLimitError(AgentRuntimeError):
    def __init__(
        self,
        reset_time: datetime | None = None,
        raw_message: str | None = None,
        service_name: str | None = None,
        *,
        is_permanent: bool = False,
        invocation_progress: InvocationProgress = InvocationProgress.NOT_STARTED,
        continuation: Any | None = None,
        usage: ProviderUsage | None = None,
    ) -> None:
        self.reset_time = reset_time
        self.raw_message = raw_message
        if service_name is not None:
            validate_runtime_identity_label(
                service_name,
                kind="UsageLimitError service name",
            )
        self.service_name = service_name
        self.is_permanent = is_permanent
        self.invocation_progress = invocation_progress
        self.continuation = continuation
        self.usage = usage
        super().__init__(
            f"Usage limit reached (reset_time={reset_time.isoformat() if reset_time else None})"
        )


class ModelNotAvailableError(AgentRuntimeError):
    def __init__(
        self,
        message: str = "",
        *,
        service_name: str,
        raw_message: str | None = None,
        invocation_progress: InvocationProgress = InvocationProgress.NOT_STARTED,
        continuation: Any | None = None,
        usage: ProviderUsage | None = None,
    ) -> None:
        validate_runtime_identity_label(
            service_name,
            kind="ModelNotAvailableError service name",
        )
        self.service_name = service_name
        self.raw_message = raw_message
        self.invocation_progress = invocation_progress
        self.continuation = continuation
        self.usage = usage
        super().__init__(message)


class HardAgentError(AgentRuntimeError):
    def __init__(
        self,
        message: str = "",
        service_name: str = "",
        classification: str | None = None,
    ) -> None:
        self.caller = ""
        if service_name:
            validate_runtime_identity_label(
                service_name,
                kind="HardAgentError service name",
            )
        self.service_name = service_name
        self.classification = classification
        super().__init__(message)


class ContinuationUnrecoverableError(AgentRuntimeError):
    def __init__(
        self,
        message: str = "",
        *,
        service_name: str,
        classification: str | None = None,
        raw_message: str | None = None,
    ) -> None:
        validate_runtime_identity_label(
            service_name,
            kind="ContinuationUnrecoverableError service_name",
        )
        self.service_name = service_name
        self.classification = classification
        self.raw_message = raw_message
        super().__init__(message)


class AgentCredentialFailureError(HardAgentError):
    def __init__(
        self,
        message: str = "",
        *,
        service_name: str,
        classification: str | None = None,
    ) -> None:
        self.is_operator_actionable = True
        super().__init__(
            message=message,
            service_name=service_name,
            classification=classification,
        )


__all__ = [
    "AgentCancelledError",
    "AgentCredentialFailureError",
    "AgentRuntimeError",
    "AgentTimeoutError",
    "ContinuationUnrecoverableError",
    "HardAgentError",
    "ModelNotAvailableError",
    "ProviderUnavailableError",
    "ProviderUnavailableReason",
    "RuntimeConfigurationError",
    "UsageLimitError",
]
