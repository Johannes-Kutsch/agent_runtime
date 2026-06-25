from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar, cast

import pytest

import agent_runtime.contracts as contracts_runtime
import agent_runtime._provider_invocation as provider_invocation_runtime
import agent_runtime.runtime as prompt_runtime

ProviderSelectionT = TypeVar("ProviderSelectionT")
PreparedInvocation = (
    provider_invocation_runtime.ProviderInvocationResult
    | provider_invocation_runtime.ProviderInvocationFailure
    | provider_invocation_runtime.ProviderInvocationPreparedStream
)


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

    @classmethod
    def ephemeral_run_request(
        cls,
        *,
        prompt: str = "already rendered prompt",
        invocation_dir: Path,
        provider_selection: Any,
        provider_auth: prompt_runtime.ProviderAuth | None = None,
        tool_policy: prompt_runtime.ToolPolicy = prompt_runtime.ToolPolicy.NONE,
        tool_access: contracts_runtime.ToolAccess | None = None,
        timeout_seconds: int = 300,
        token: Any = None,
        on_live_output: Callable[[prompt_runtime.AgentEvent], None] | None = None,
    ) -> prompt_runtime.EphemeralRunRequest:
        if provider_auth is not None:
            provider_selection = cls.attach_provider_auth(
                provider_selection,
                provider_auth,
            )

        request_kwargs: dict[str, Any] = {
            "prompt": prompt,
            "invocation_dir": invocation_dir,
            "provider_selection": provider_selection,
            "timeout_seconds": timeout_seconds,
            "token": token,
            "on_live_output": on_live_output,
        }
        if tool_access is None:
            request_kwargs["tool_policy"] = tool_policy
        else:
            request_kwargs["tool_access"] = tool_access

        return prompt_runtime.EphemeralRunRequest(**request_kwargs)

    @staticmethod
    def runtime_state_dir(
        invocation_dir: Path,
        *,
        dirname: str = ".agent-runtime",
    ) -> Path:
        return invocation_dir / dirname / "state"

    @classmethod
    def prepare_runtime_state_dir(
        cls,
        invocation_dir: Path,
        *,
        dirname: str = ".agent-runtime",
    ) -> Path:
        runtime_state_dir = cls.runtime_state_dir(invocation_dir, dirname=dirname)
        runtime_state_dir.mkdir(parents=True, exist_ok=True)
        return runtime_state_dir

    @staticmethod
    def provider_state_dir(
        runtime_state_dir: Path,
        *,
        session_namespace: str = "main",
        service: str,
    ) -> Path:
        return runtime_state_dir / "implementer" / session_namespace / service

    @classmethod
    def start_session_run_request(
        cls,
        *,
        prompt: str = "already rendered prompt",
        invocation_dir: Path,
        provider_selection: Any,
        runtime_state_dir: Path | None = None,
        session_namespace: str = "main",
        provider_auth: prompt_runtime.ProviderAuth | None = None,
        tool_policy: prompt_runtime.ToolPolicy = prompt_runtime.ToolPolicy.NONE,
        tool_access: contracts_runtime.ToolAccess | None = None,
        timeout_seconds: int = 300,
        token: Any = None,
        on_live_output: Callable[[prompt_runtime.AgentEvent], None] | None = None,
    ) -> prompt_runtime.NewSessionRunRequest:
        if provider_auth is not None:
            provider_selection = cls.attach_provider_auth(
                provider_selection,
                provider_auth,
            )

        request_kwargs: dict[str, Any] = {
            "prompt": prompt,
            "invocation_dir": invocation_dir,
            "runtime_state_dir": runtime_state_dir,
            "provider_selection": provider_selection,
            "session_namespace": session_namespace,
            "timeout_seconds": timeout_seconds,
            "token": token,
            "on_live_output": on_live_output,
        }
        if tool_access is None:
            request_kwargs["tool_policy"] = tool_policy
        else:
            request_kwargs["tool_access"] = tool_access

        return prompt_runtime.NewSessionRunRequest(**request_kwargs)

    @classmethod
    def resume_session_run_request(
        cls,
        *,
        prompt: str = "already rendered prompt",
        invocation_dir: Path,
        continuation: prompt_runtime.Continuation,
        runtime_state_dir: Path | None = None,
        session_namespace: str = "main",
        provider_auth: prompt_runtime.ProviderAuth | None = None,
        timeout_seconds: int = 300,
        token: Any = None,
        on_live_output: Callable[[prompt_runtime.AgentEvent], None] | None = None,
    ) -> prompt_runtime.ResumedSessionRunRequest:
        return prompt_runtime.ResumedSessionRunRequest(
            prompt=prompt,
            invocation_dir=invocation_dir,
            runtime_state_dir=runtime_state_dir,
            continuation=continuation,
            session_namespace=session_namespace,
            provider_auth=provider_auth,
            timeout_seconds=timeout_seconds,
            token=token,
            on_live_output=on_live_output,
        )

    @staticmethod
    def codex_continuation(
        *,
        model: str = "gpt-5.4",
        effort: str = "medium",
        tool_access: contracts_runtime.ToolAccess | None = None,
        provider_session_id: str = "selected-thread",
        provider_state_dir_relpath: str = "implementer/main/codex/",
        exact_transcript_match: bool = False,
    ) -> prompt_runtime.Continuation:
        return prompt_runtime.Continuation(
            selected_service="codex",
            selected_model=model,
            selected_effort=effort,
            tool_access=tool_access or contracts_runtime.ToolAccess.no_tools(),
            provider_resume_state={
                "run_kind": "resume",
                "provider_session_id": provider_session_id,
                "provider_state_dir_relpath": provider_state_dir_relpath,
                "exact_transcript_match": exact_transcript_match,
            },
        )

    @staticmethod
    def opencode_continuation(
        *,
        model: str = "glm-5.2",
        effort: str = "medium",
        tool_access: contracts_runtime.ToolAccess | None = None,
        provider_session_id: str = "persisted-session-1",
        provider_state: dict[str, str] | None = None,
        exact_transcript_match: bool = True,
    ) -> prompt_runtime.Continuation:
        return prompt_runtime.Continuation(
            selected_service="opencode",
            selected_model=model,
            selected_effort=effort,
            tool_access=tool_access or contracts_runtime.ToolAccess.no_tools(),
            provider_resume_state={
                "provider_session_id": provider_session_id,
                "provider_state": provider_state
                if provider_state is not None
                else {
                    "session_id": provider_session_id,
                    "resume_jsonl": "[]",
                },
                "exact_transcript_match": exact_transcript_match,
            },
        )

    @staticmethod
    def prepare_codex_rollout_state(
        provider_state_dir: Path,
        *thread_ids: str,
        date_path: tuple[str, str, str] = ("2026", "05", "30"),
        filename: str = "rollout-001.jsonl",
    ) -> Path:
        rollout_dir = (
            provider_state_dir / "sessions" / date_path[0] / date_path[1] / date_path[2]
        )
        rollout_dir.mkdir(parents=True, exist_ok=True)
        rollout_path = rollout_dir / filename
        rollout_path.write_text(
            "\n".join(
                json.dumps({"type": "thread.started", "thread_id": thread_id})
                for thread_id in thread_ids
            )
            + "\n",
            encoding="utf-8",
        )
        return rollout_path

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
        prepared_invocation: PreparedInvocation,
    ) -> RuntimeClientExecutionHarness:
        self._adapter.prepared_invocations.append(prepared_invocation)
        return self

    def prepare_all(
        self,
        *prepared_invocations: PreparedInvocation,
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
