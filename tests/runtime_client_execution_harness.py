from __future__ import annotations

import dataclasses

import pytest

import agent_runtime._provider_invocation as provider_invocation_runtime
import agent_runtime.runtime as prompt_runtime


@dataclasses.dataclass(slots=True)
class RuntimeClientExecutionHarness:
    _adapter: provider_invocation_runtime.InMemoryProviderInvocationAdapter

    @classmethod
    def install(
        cls,
        monkeypatch: pytest.MonkeyPatch,
    ) -> RuntimeClientExecutionHarness:
        adapter = provider_invocation_runtime.InMemoryProviderInvocationAdapter()
        monkeypatch.setattr(
            prompt_runtime._builtin_runtime_client_module,
            "_default_provider_invocation_adapter",
            lambda: adapter,
        )
        return cls(_adapter=adapter)

    @property
    def adapter(self) -> provider_invocation_runtime.InMemoryProviderInvocationAdapter:
        return self._adapter

    @property
    def recorded_requests(
        self,
    ) -> list[provider_invocation_runtime.ProviderInvocationRequest]:
        return self._adapter.recorded_requests

    def prepare_result(
        self,
        result: provider_invocation_runtime.ProviderInvocationResult,
    ) -> provider_invocation_runtime.ProviderInvocationResult:
        self._adapter.prepared_invocations.append(result)
        return result

    def prepare_failure(
        self,
        failure: provider_invocation_runtime.ProviderInvocationFailure,
    ) -> provider_invocation_runtime.ProviderInvocationFailure:
        self._adapter.prepared_invocations.append(failure)
        return failure

    def prepare_prepared_stream(
        self,
        prepared_stream: provider_invocation_runtime.ProviderInvocationPreparedStream,
    ) -> provider_invocation_runtime.ProviderInvocationPreparedStream:
        self._adapter.prepared_invocations.append(prepared_stream)
        return prepared_stream

    def execute(
        self,
        request: provider_invocation_runtime.ProviderInvocationRequest,
    ) -> (
        provider_invocation_runtime.ProviderInvocationResult
        | provider_invocation_runtime.ProviderInvocationFailure
    ):
        return self._adapter.execute(request)
