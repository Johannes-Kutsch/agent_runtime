from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from .identity import validate_runtime_identity_label, validate_session_namespace
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


class TransientAgentError(AgentRuntimeError):
    def __init__(self, message: str = "", status_code: int | None = None) -> None:
        self.status_code = status_code
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


class AgentFailedError(AgentRuntimeError):
    def __init__(
        self,
        invocation_role: str,
        worktree_path: Path,
        namespace: str = "",
        failure_class: str = "",
        service_name: str = "",
        provider_session_path: str | None = None,
        session_root: str = "",
    ) -> None:
        validate_session_namespace(namespace)
        if service_name:
            validate_runtime_identity_label(
                service_name,
                kind="AgentFailedError service name",
            )
        super().__init__(f"Agent {invocation_role!r} failed irrecoverably")
        self.invocation_role = invocation_role
        self.worktree_path = worktree_path
        self.namespace = namespace
        self.failure_class = failure_class
        self.service_name = service_name
        self.provider_session_path = provider_session_path
        self.session_root = session_root

    @property
    def session_dir(self) -> str:
        if self.provider_session_path is not None:
            return self.provider_session_path
        parts = [
            self.session_root,
            self.invocation_role,
            self.namespace,
            self.service_name,
        ]
        return "/".join(part for part in parts if part)


__all__ = [
    "AgentCancelledError",
    "AgentCredentialFailureError",
    "AgentFailedError",
    "AgentRuntimeError",
    "AgentTimeoutError",
    "HardAgentError",
    "ProviderUnavailableError",
    "ProviderUnavailableReason",
    "RuntimeConfigurationError",
    "TransientAgentError",
    "UsageLimitError",
]
