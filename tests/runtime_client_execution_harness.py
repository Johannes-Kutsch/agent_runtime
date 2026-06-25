from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, TypeVar, cast

import pytest

import agent_runtime._provider_invocation as provider_invocation_runtime
import agent_runtime.runtime as prompt_runtime

ProviderSelectionT = TypeVar("ProviderSelectionT")


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
    def recorded_requests(
        self,
    ) -> list[provider_invocation_runtime.ProviderInvocationRequest]:
        return self._adapter.recorded_requests

    @staticmethod
    def attach_provider_auth(
        provider_selection: ProviderSelectionT,
        auth: prompt_runtime.ProviderAuth,
    ) -> ProviderSelectionT:
        return cast(
            ProviderSelectionT,
            dataclasses.replace(cast(Any, provider_selection), auth=auth),
        )

    @staticmethod
    def install_local_codex_host_auth(
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        *,
        auth_file_content: str = "{}",
    ) -> Path:
        host_home = tmp_path / "host-home"
        host_auth_path = host_home / ".codex" / "auth.json"
        host_auth_path.parent.mkdir(parents=True, exist_ok=True)
        host_auth_path.write_text(auth_file_content, encoding="utf-8")
        monkeypatch.setattr(
            prompt_runtime._builtin_runtime_client_module.Path,
            "home",
            lambda: host_home,
        )
        return host_auth_path

    def prepare(
        self,
        prepared_invocation: (
            provider_invocation_runtime.ProviderInvocationResult
            | provider_invocation_runtime.ProviderInvocationFailure
            | provider_invocation_runtime.ProviderInvocationPreparedStream
        ),
    ) -> RuntimeClientExecutionHarness:
        self._adapter.prepared_invocations.append(prepared_invocation)
        return self

    def prepare_all(
        self,
        *prepared_invocations: (
            provider_invocation_runtime.ProviderInvocationResult
            | provider_invocation_runtime.ProviderInvocationFailure
            | provider_invocation_runtime.ProviderInvocationPreparedStream
        ),
    ) -> RuntimeClientExecutionHarness:
        for prepared_invocation in prepared_invocations:
            self.prepare(prepared_invocation)
        return self

    def prepare_result(
        self,
        result: provider_invocation_runtime.ProviderInvocationResult,
    ) -> provider_invocation_runtime.ProviderInvocationResult:
        self.prepare(result)
        return result

    def prepare_failure(
        self,
        failure: provider_invocation_runtime.ProviderInvocationFailure,
    ) -> provider_invocation_runtime.ProviderInvocationFailure:
        self.prepare(failure)
        return failure

    def prepare_prepared_stream(
        self,
        prepared_stream: provider_invocation_runtime.ProviderInvocationPreparedStream,
    ) -> provider_invocation_runtime.ProviderInvocationPreparedStream:
        self.prepare(prepared_stream)
        return prepared_stream
