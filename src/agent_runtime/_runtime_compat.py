from __future__ import annotations

from typing import Any

from . import _runtime_facade_lifecycle as _runtime_facade_lifecycle_module
from .execution_contracts import (
    PromptRuntimeExecutionAdapter as _PromptRuntimeExecutionAdapter,
)
from .runtime import (
    EphemeralRunRequest,
    NewSessionRunRequest,
    ResumedSessionRunRequest,
    RuntimeOutcome,
    _run_ephemeral_outcome,
    _run_new_session_outcome,
    _run_resumed_session_outcome,
)
from .service_registry import ServiceRegistry

__all__ = [
    "EphemeralRuntime",
    "EphemeralRuntimeExecutionAdapter",
    "NewSessionRuntime",
    "NewSessionRuntimeExecutionAdapter",
    "ResumedSessionRuntime",
    "ResumedSessionRuntimeExecutionAdapter",
]

EphemeralRuntimeExecutionAdapter = _PromptRuntimeExecutionAdapter
NewSessionRuntimeExecutionAdapter = _PromptRuntimeExecutionAdapter
ResumedSessionRuntimeExecutionAdapter = _PromptRuntimeExecutionAdapter

_coerce_service_registry = _runtime_facade_lifecycle_module._coerce_service_registry


class EphemeralRuntime:
    def __init__(
        self,
        *,
        execution_adapter: EphemeralRuntimeExecutionAdapter,
        service_registry: ServiceRegistry | dict[str, Any] | None = None,
    ) -> None:
        self._service_registry = _coerce_service_registry(service_registry)
        self._execution_adapter = execution_adapter

    async def run_ephemeral(self, request: EphemeralRunRequest) -> RuntimeOutcome:
        return await _run_ephemeral_outcome(
            runner=self._execution_adapter,
            service_registry=self._service_registry,
            request=request,
        )


class NewSessionRuntime:
    def __init__(
        self,
        *,
        execution_adapter: NewSessionRuntimeExecutionAdapter,
        service_registry: ServiceRegistry | dict[str, Any] | None = None,
    ) -> None:
        self._service_registry = _coerce_service_registry(service_registry)
        self._execution_adapter = execution_adapter

    async def run_new_session(self, request: NewSessionRunRequest) -> RuntimeOutcome:
        return await _run_new_session_outcome(
            runner=self._execution_adapter,
            service_registry=self._service_registry,
            request=request,
        )


class ResumedSessionRuntime:
    def __init__(
        self,
        *,
        execution_adapter: ResumedSessionRuntimeExecutionAdapter,
    ) -> None:
        self._execution_adapter = execution_adapter

    async def run_resumed_session(
        self,
        request: ResumedSessionRunRequest,
    ) -> RuntimeOutcome:
        return await _run_resumed_session_outcome(
            runner=self._execution_adapter,
            request=request,
        )
