from __future__ import annotations

import asyncio
import os
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import pytest

import agent_runtime as runtime
import agent_runtime._provider_invocation as provider_invocation
import agent_runtime.contracts as contracts_runtime
import agent_runtime._runtime_compat as compat_runtime
import agent_runtime.runtime as prompt_runtime
from agent_runtime.contracts import ExecutionProvider, TransientError
from agent_runtime.errors import (
    AgentCancelledError,
    AgentCredentialFailureError,
    AgentTimeoutError,
    HardAgentError,
    RetryableProviderFailureError,
    TransientAgentError,
    UsageLimitError,
)
from agent_runtime._execution_contracts import (
    CancellationToken,
    PreparedRunSessionState,
    TextOutputAdapter,
    WorkExecutionAdapter,
    WorkExecutionDependencies,
    WorkFailureHandling,
    WorkInvocationDependencies,
    WorkPresentationDependencies,
)
from agent_runtime.provider_errors import ProviderErrorObservation
from agent_runtime.provider_output import reduce_text_output_events
from agent_runtime.roles import InvocationRole
from agent_runtime._service_registry import ServiceRegistry
from agent_runtime.session import RunKind

from tests.runtime_boundary_fakes import ExecutionServiceFake as _ExecutionService


@dataclass
class _ProviderRunSession:
    run_kind: RunKind = RunKind.FRESH
    provider_session_id: str | None = None

    def record_provider_session_id(self, provider_session_id: str) -> None:
        self.provider_session_id = provider_session_id

    def record_successful_run(self) -> None:
        return None


class _PreparedRunSession:
    provider_state_dir_container_path: str | None = None

    def __init__(self) -> None:
        self._provider_run_session = _ProviderRunSession()

    def prepare_for_run(self) -> None:
        return None

    def initial_provider_run_session(self) -> _ProviderRunSession:
        return self._provider_run_session

    def resumable_provider_run_session(self) -> _ProviderRunSession:
        return self._provider_run_session

    def protocol_reprompt_provider_run_session(self) -> None:
        return None


class _Session:
    def __init__(self, provider_state_dir: str | None = None) -> None:
        self.provider_state_dir = provider_state_dir


def _selection_with_auth(selection: Any, auth: Any) -> Any:
    return replace(selection, auth=auth)


def _codex_executable() -> str:
    return "codex.cmd" if os.name == "nt" else "codex"


def _observed_command_text(command: str | tuple[str, ...]) -> str:
    return command if isinstance(command, str) else " ".join(command)


class _RoleAwareEphemeralCompatWorkRunner:
    async def setup(self, git_name: str, git_email: str, work_body: str = "") -> None:
        del git_name, git_email, work_body

    async def work(
        self,
        role: InvocationRole,
        prompt: str,
        *,
        run_kind: RunKind = RunKind.FRESH,
        session_uuid: str | None = None,
        on_provider_session_id: Any = None,
    ) -> str:
        assert callable(on_provider_session_id)
        assert run_kind is RunKind.FRESH
        assert session_uuid is None

        on_provider_session_id(f"provider-{role.value}")
        return f"{role.value}:{prompt}"

    async def work_text(
        self,
        prompt: str,
        *,
        role: InvocationRole = InvocationRole("implementer"),
        tool_policy: Any = runtime.ToolPolicy.UNRESTRICTED,
        run_kind: RunKind = RunKind.FRESH,
        session_uuid: str | None = None,
        on_provider_session_id: Any = None,
    ) -> str:
        del tool_policy
        result = await self.work(
            role,
            prompt,
            run_kind=run_kind,
            session_uuid=session_uuid,
            on_provider_session_id=on_provider_session_id,
        )
        return str(result)


class _EphemeralExecutionAdapter:
    def __init__(self) -> None:
        self.prepare_session_calls = 0

    def resolve_service(self, service_name: str = "") -> ExecutionProvider:
        return _ExecutionService(service_name)

    def build_work_dependencies(
        self,
        *,
        name: str,
        model: str,
        effort: str,
        service: ExecutionProvider,
    ) -> WorkInvocationDependencies:
        del name, model, effort, service

        def _prepare_session(_run_session: Any) -> PreparedRunSessionState:
            self.prepare_session_calls += 1
            return cast(PreparedRunSessionState, _PreparedRunSession())

        return WorkInvocationDependencies(
            execution=WorkExecutionDependencies(
                container_workspace="/workspace",
                prepare_session=_prepare_session,
                build_session=lambda mount_path, service, provider_state_dir: (
                    _Session()
                ),
                build_runner=lambda session, status_display: cast(
                    WorkExecutionAdapter,
                    _RoleAwareEphemeralCompatWorkRunner(),
                ),
                get_git_identity=lambda: ("Runtime Test", "runtime@example.com"),
            ),
            failure_handling=WorkFailureHandling(timeout_retries=0),
            presentation=WorkPresentationDependencies(),
        )


class _RawSetupFailureEphemeralRunner:
    async def setup(self, git_name: str, git_email: str, work_body: str = "") -> None:
        del git_name, git_email, work_body
        raise RuntimeError("missing auth")

    async def work_text(
        self,
        prompt: str,
        *,
        role: InvocationRole = InvocationRole("implementer"),
        tool_policy: Any = runtime.ToolPolicy.UNRESTRICTED,
        run_kind: RunKind = RunKind.FRESH,
        session_uuid: str | None = None,
        on_provider_session_id: Any = None,
    ) -> str:
        del prompt, role, tool_policy, run_kind, session_uuid, on_provider_session_id
        raise AssertionError("setup failure should stop execution before work_text")


class _SetupTranslatedEphemeralExecutionAdapter:
    def resolve_service(self, service_name: str = "") -> ExecutionProvider:
        return _ExecutionService(service_name)

    def build_work_dependencies(
        self,
        *,
        name: str,
        model: str,
        effort: str,
        service: ExecutionProvider,
    ) -> WorkInvocationDependencies:
        del name, model, effort, service
        return WorkInvocationDependencies(
            execution=WorkExecutionDependencies(
                container_workspace="/workspace",
                prepare_session=lambda _run_session: cast(
                    PreparedRunSessionState, _PreparedRunSession()
                ),
                build_session=lambda mount_path, service, provider_state_dir: (
                    _Session()
                ),
                build_runner=lambda session, status_display: cast(
                    WorkExecutionAdapter,
                    _RawSetupFailureEphemeralRunner(),
                ),
                get_git_identity=lambda: ("Runtime Test", "runtime@example.com"),
            ),
            failure_handling=WorkFailureHandling(
                timeout_retries=0,
                translate_setup_failure=lambda role, exc: AgentCredentialFailureError(
                    str(exc),
                    service_name="claude",
                    classification="credential",
                    observations=(),
                ),
            ),
            presentation=WorkPresentationDependencies(),
        )


class _UsageLimitThenSuccessEphemeralRunner(_RoleAwareEphemeralCompatWorkRunner):
    def __init__(self) -> None:
        self._attempts = 0

    async def work_text(
        self,
        prompt: str,
        *,
        role: InvocationRole = InvocationRole("implementer"),
        tool_policy: Any = runtime.ToolPolicy.UNRESTRICTED,
        run_kind: RunKind = RunKind.FRESH,
        session_uuid: str | None = None,
        on_provider_session_id: Any = None,
    ) -> str:
        self._attempts += 1
        if self._attempts == 1:
            raise UsageLimitError(
                reset_time=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
                service_name="codex",
                invocation_progress=runtime.InvocationProgress.STARTED,
            )
        return await super().work_text(
            prompt,
            role=role,
            tool_policy=tool_policy,
            run_kind=run_kind,
            session_uuid=session_uuid,
            on_provider_session_id=on_provider_session_id,
        )


class _UsageLimitThenSuccessEphemeralExecutionAdapter:
    def __init__(self) -> None:
        self._runner = _UsageLimitThenSuccessEphemeralRunner()

    def resolve_service(self, service_name: str = "") -> ExecutionProvider:
        return _ExecutionService(service_name)

    def build_work_dependencies(
        self,
        *,
        name: str,
        model: str,
        effort: str,
        service: ExecutionProvider,
    ) -> WorkInvocationDependencies:
        del name, model, effort, service
        return WorkInvocationDependencies(
            execution=WorkExecutionDependencies(
                container_workspace="/workspace",
                prepare_session=lambda _run_session: cast(
                    PreparedRunSessionState, _PreparedRunSession()
                ),
                build_session=lambda mount_path, service, provider_state_dir: (
                    _Session()
                ),
                build_runner=lambda session, status_display: cast(
                    WorkExecutionAdapter,
                    self._runner,
                ),
                get_git_identity=lambda: ("Runtime Test", "runtime@example.com"),
            ),
            failure_handling=WorkFailureHandling(timeout_retries=0),
            presentation=WorkPresentationDependencies(),
        )


class _TimeoutEphemeralRunner(_RoleAwareEphemeralCompatWorkRunner):
    async def work_text(
        self,
        prompt: str,
        *,
        role: InvocationRole = InvocationRole("implementer"),
        tool_policy: Any = runtime.ToolPolicy.UNRESTRICTED,
        run_kind: RunKind = RunKind.FRESH,
        session_uuid: str | None = None,
        on_provider_session_id: Any = None,
    ) -> str:
        del prompt, role, tool_policy, run_kind, session_uuid, on_provider_session_id
        raise AgentTimeoutError("timed out")


class _TimeoutEphemeralExecutionAdapter:
    def resolve_service(self, service_name: str = "") -> ExecutionProvider:
        return _ExecutionService(service_name)

    def build_work_dependencies(
        self,
        *,
        name: str,
        model: str,
        effort: str,
        service: ExecutionProvider,
    ) -> WorkInvocationDependencies:
        del name, model, effort, service
        return WorkInvocationDependencies(
            execution=WorkExecutionDependencies(
                container_workspace="/workspace",
                prepare_session=lambda _run_session: cast(
                    PreparedRunSessionState, _PreparedRunSession()
                ),
                build_session=lambda mount_path, service, provider_state_dir: (
                    _Session()
                ),
                build_runner=lambda session, status_display: cast(
                    WorkExecutionAdapter,
                    _TimeoutEphemeralRunner(),
                ),
                get_git_identity=lambda: ("Runtime Test", "runtime@example.com"),
            ),
            failure_handling=WorkFailureHandling(timeout_retries=0),
            presentation=WorkPresentationDependencies(),
        )


class _RetryableProviderFailureEphemeralRunner(_RoleAwareEphemeralCompatWorkRunner):
    async def work_text(
        self,
        prompt: str,
        *,
        role: InvocationRole = InvocationRole("implementer"),
        tool_policy: Any = runtime.ToolPolicy.UNRESTRICTED,
        run_kind: RunKind = RunKind.FRESH,
        session_uuid: str | None = None,
        on_provider_session_id: Any = None,
    ) -> str:
        del prompt, role, tool_policy, run_kind, session_uuid, on_provider_session_id
        return reduce_text_output_events(
            [
                TransientError(
                    status_code=503,
                    raw_message="retry later",
                    classification="retryable",
                )
            ],
            lambda _turn: None,
            provider="codex",
        )


class _RetryableProviderFailureEphemeralExecutionAdapter:
    def resolve_service(self, service_name: str = "") -> ExecutionProvider:
        return _ExecutionService(service_name)

    def build_work_dependencies(
        self,
        *,
        name: str,
        model: str,
        effort: str,
        service: ExecutionProvider,
    ) -> WorkInvocationDependencies:
        del name, model, effort, service
        return WorkInvocationDependencies(
            execution=WorkExecutionDependencies(
                container_workspace="/workspace",
                prepare_session=lambda _run_session: cast(
                    PreparedRunSessionState, _PreparedRunSession()
                ),
                build_session=lambda mount_path, service, provider_state_dir: (
                    _Session()
                ),
                build_runner=lambda session, status_display: cast(
                    WorkExecutionAdapter,
                    _RetryableProviderFailureEphemeralRunner(),
                ),
                get_git_identity=lambda: ("Runtime Test", "runtime@example.com"),
            ),
            failure_handling=WorkFailureHandling(timeout_retries=0),
            presentation=WorkPresentationDependencies(),
        )


class _HardFailureEphemeralRunner(_RoleAwareEphemeralCompatWorkRunner):
    async def work_text(
        self,
        prompt: str,
        *,
        role: InvocationRole = InvocationRole("implementer"),
        tool_policy: Any = runtime.ToolPolicy.UNRESTRICTED,
        run_kind: RunKind = RunKind.FRESH,
        session_uuid: str | None = None,
        on_provider_session_id: Any = None,
    ) -> str:
        del prompt, role, tool_policy, run_kind, session_uuid, on_provider_session_id
        raise HardAgentError("hard failure", service_name="codex")


class _HardFailureEphemeralExecutionAdapter:
    def resolve_service(self, service_name: str = "") -> ExecutionProvider:
        return _ExecutionService(service_name)

    def build_work_dependencies(
        self,
        *,
        name: str,
        model: str,
        effort: str,
        service: ExecutionProvider,
    ) -> WorkInvocationDependencies:
        del name, model, effort, service
        return WorkInvocationDependencies(
            execution=WorkExecutionDependencies(
                container_workspace="/workspace",
                prepare_session=lambda _run_session: cast(
                    PreparedRunSessionState, _PreparedRunSession()
                ),
                build_session=lambda mount_path, service, provider_state_dir: (
                    _Session()
                ),
                build_runner=lambda session, status_display: cast(
                    WorkExecutionAdapter,
                    _HardFailureEphemeralRunner(),
                ),
                get_git_identity=lambda: ("Runtime Test", "runtime@example.com"),
            ),
            failure_handling=WorkFailureHandling(timeout_retries=0),
            presentation=WorkPresentationDependencies(),
        )


class _TransientProviderFailureEphemeralRunner(_RoleAwareEphemeralCompatWorkRunner):
    async def work_text(
        self,
        prompt: str,
        *,
        role: InvocationRole = InvocationRole("implementer"),
        tool_policy: Any = runtime.ToolPolicy.UNRESTRICTED,
        run_kind: RunKind = RunKind.FRESH,
        session_uuid: str | None = None,
        on_provider_session_id: Any = None,
    ) -> str:
        del prompt, role, tool_policy, run_kind, session_uuid, on_provider_session_id
        return reduce_text_output_events(
            [TransientError(status_code=503, raw_message="retry later")],
            lambda _turn: None,
            provider="codex",
        )


class _TransientProviderFailureEphemeralExecutionAdapter:
    def resolve_service(self, service_name: str = "") -> ExecutionProvider:
        return _ExecutionService(service_name)

    def build_work_dependencies(
        self,
        *,
        name: str,
        model: str,
        effort: str,
        service: ExecutionProvider,
    ) -> WorkInvocationDependencies:
        del name, model, effort, service
        return WorkInvocationDependencies(
            execution=WorkExecutionDependencies(
                container_workspace="/workspace",
                prepare_session=lambda _run_session: cast(
                    PreparedRunSessionState, _PreparedRunSession()
                ),
                build_session=lambda mount_path, service, provider_state_dir: (
                    _Session()
                ),
                build_runner=lambda session, status_display: cast(
                    WorkExecutionAdapter,
                    _TransientProviderFailureEphemeralRunner(),
                ),
                get_git_identity=lambda: ("Runtime Test", "runtime@example.com"),
            ),
            failure_handling=WorkFailureHandling(timeout_retries=0),
            presentation=WorkPresentationDependencies(),
        )


class _InterruptedEphemeralRunner(_RoleAwareEphemeralCompatWorkRunner):
    def __init__(self, error: Exception) -> None:
        self._error = error

    async def work_text(
        self,
        prompt: str,
        *,
        role: InvocationRole = InvocationRole("implementer"),
        tool_policy: Any = runtime.ToolPolicy.UNRESTRICTED,
        run_kind: RunKind = RunKind.FRESH,
        session_uuid: str | None = None,
        on_provider_session_id: Any = None,
    ) -> str:
        del prompt, role, tool_policy, run_kind, session_uuid, on_provider_session_id
        raise self._error


class _InterruptedEphemeralExecutionAdapter:
    def __init__(self, error: Exception) -> None:
        self._runner = _InterruptedEphemeralRunner(error)

    def resolve_service(self, service_name: str = "") -> ExecutionProvider:
        return _ExecutionService(service_name)

    def build_work_dependencies(
        self,
        *,
        name: str,
        model: str,
        effort: str,
        service: ExecutionProvider,
    ) -> WorkInvocationDependencies:
        del name, model, effort, service
        return WorkInvocationDependencies(
            execution=WorkExecutionDependencies(
                container_workspace="/workspace",
                prepare_session=lambda _run_session: cast(
                    PreparedRunSessionState, _PreparedRunSession()
                ),
                build_session=lambda mount_path, service, provider_state_dir: (
                    _Session()
                ),
                build_runner=lambda session, status_display: cast(
                    WorkExecutionAdapter,
                    self._runner,
                ),
                get_git_identity=lambda: ("Runtime Test", "runtime@example.com"),
            ),
            failure_handling=WorkFailureHandling(timeout_retries=0),
            presentation=WorkPresentationDependencies(),
        )


def _tool_policy_effect_text(tool_policy: Any) -> str:
    profile = (
        tool_policy.profile
        if isinstance(tool_policy, runtime.ToolPolicy)
        else tool_policy
    )
    allowed_tools = profile.allowed_tools or ()
    disallowed_tools = profile.disallowed_tools or ()
    allowed = ",".join(allowed_tools) or "all"
    disallowed = ",".join(disallowed_tools) or "none"
    return f"allowed={allowed};disallowed={disallowed}"


_TOOL_POLICY_CASES = [
    pytest.param(policy, id=policy.value) for policy in runtime.ToolPolicy
] + [
    pytest.param(
        contracts_runtime.ToolPolicyProfile(
            allowed_tools=("Read", "Bash"),
            disallowed_tools=("Edit",),
        ),
        id="custom-profile",
    )
]


class _ToolPolicyRenderingPromptRunner:
    async def setup(self, git_name: str, git_email: str, work_body: str = "") -> None:
        del git_name, git_email, work_body

    async def work_text(
        self,
        prompt: str,
        *,
        role: InvocationRole = InvocationRole("implementer"),
        tool_policy: Any,
        run_kind: RunKind = RunKind.FRESH,
        session_uuid: str | None = None,
        on_provider_session_id: Any = None,
    ) -> str:
        del prompt, role, run_kind, session_uuid
        assert callable(on_provider_session_id)
        on_provider_session_id("provider-session")
        return _tool_policy_effect_text(tool_policy)


def _seed_codex_host_auth(monkeypatch: pytest.MonkeyPatch, home_dir: Path) -> None:
    auth_dir = home_dir / ".codex"
    auth_dir.mkdir(parents=True)
    (auth_dir / "auth.json").write_text('{"access_token":"token"}')
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module,
        "_codex_host_auth_path",
        lambda: auth_dir / "auth.json",
        raising=False,
    )


def _seed_empty_codex_host_auth(
    monkeypatch: pytest.MonkeyPatch,
    home_dir: Path,
) -> None:
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module,
        "_codex_host_auth_path",
        lambda: home_dir / ".codex" / "auth.json",
        raising=False,
    )


def _stub_codex_prompt_path(
    monkeypatch: pytest.MonkeyPatch,
    *,
    on_write: Callable[[str], None] | None = None,
    on_unlink: Callable[[], None] | None = None,
) -> None:
    prompt_path = Path("/tmp/.provider_prompt")
    original_write_text = Path.write_text
    original_unlink = Path.unlink

    def _fake_write_text(self: Path, data: str, *args: Any, **kwargs: Any) -> int:
        if self == prompt_path:
            if on_write is not None:
                on_write(data)
            return len(data)
        return original_write_text(self, data, *args, **kwargs)

    def _fake_unlink(self: Path, *args: Any, **kwargs: Any) -> None:
        if self == prompt_path:
            if on_unlink is not None:
                on_unlink()
            return None
        return original_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", _fake_write_text)
    monkeypatch.setattr(Path, "unlink", _fake_unlink)


def test_runtime_client_runs_codex_stage_with_pycastle_command_and_env_semantics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stage_selection_factory: Callable[..., runtime.ProviderSelection],
) -> None:
    home_dir = tmp_path / "home"
    _seed_codex_host_auth(monkeypatch, home_dir)

    observed: dict[str, Any] = {}

    class _Stdin:
        def write(self, data: str) -> None:
            observed["prompt"] = data

        def close(self) -> None:
            observed["prompt_deleted"] = True

    class _FakeProcess:
        def __init__(
            self,
            command: str | tuple[str, ...],
            *,
            cwd: Path,
            env: dict[str, str],
            stdout: Any,
        ) -> None:
            observed["command"] = _observed_command_text(command)
            observed["cwd"] = cwd
            observed["env"] = env
            self.stdin = _Stdin()
            self.stdout = stdout

        def wait(self) -> int:
            return 0

    def _fake_popen(
        command: str | tuple[str, ...],
        *,
        shell: bool,
        cwd: Path,
        env: dict[str, str],
        stdout: Any,
        stderr: Any,
        text: bool,
        stdin: Any | None = None,
    ) -> _FakeProcess:
        del shell, stderr, text, stdin
        return _FakeProcess(
            command,
            cwd=cwd,
            env=env,
            stdout=iter(
                [
                    '{"type":"thread.started","thread_id":"thread-123"}\n',
                    '{"type":"item.completed","item":{"type":"agent_message","text":"hello from codex"}}\n',
                    '{"type":"turn.completed"}\n',
                ]
            ),
        )

    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module.subprocess,
        "Popen",
        _fake_popen,
    )

    outcome = prompt_runtime.RuntimeClient().run_ephemeral(
        prompt_runtime.EphemeralRunRequest(
            prompt="already rendered prompt",
            invocation_dir=tmp_path,
            provider_selection=stage_selection_factory(
                service="codex",
                model="gpt-5.4",
                effort="high",
            ),
            tool_access=contracts_runtime.ToolAccess.workspace_backed(tmp_path),
        )
    )

    assert outcome == prompt_runtime.RuntimeOutcome.completed(
        output="hello from codex",
        result=prompt_runtime.EphemeralRunResult(
            output="hello from codex",
            selected_service="codex",
            selected_model="gpt-5.4",
            selected_effort="high",
            tool_access=contracts_runtime.ToolAccess.workspace_backed(tmp_path),
            metadata=prompt_runtime.EphemeralResultMetadata(
                runtime=prompt_runtime.EphemeralRuntimeMetadata(
                    run_kind=RunKind.FRESH,
                ),
            ),
        ),
    )
    assert observed["command"] == (
        f"{_codex_executable()} exec -m gpt-5.4 -c model_reasoning_effort=high "
        "-c approval_policy=never --sandbox danger-full-access --json"
    )
    assert observed["prompt"] == "already rendered prompt"
    assert observed["prompt_deleted"] is True
    assert observed["cwd"] == tmp_path
    assert observed["env"] == {"TZ": "UTC"}


def test_runtime_client_exposes_codex_usage_on_completed_outcome(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stage_selection_factory: Callable[..., runtime.ProviderSelection],
) -> None:
    home_dir = tmp_path / "home"
    _seed_codex_host_auth(monkeypatch, home_dir)
    _stub_codex_prompt_path(monkeypatch)

    class _FakeProcess:
        stdout = iter(
            [
                '{"type":"item.completed","item":{"type":"agent_message","text":"hello from codex"}}\n',
                (
                    '{"type":"turn.completed","usage":{"input_tokens":120,'
                    '"cached_tokens":30,"output_tokens":45,"reasoning_tokens":15}}\n'
                ),
            ]
        )

        def wait(self) -> int:
            return 0

    def _fake_popen(
        command: str,
        *,
        shell: bool,
        cwd: Path,
        env: dict[str, str],
        stdout: Any,
        stderr: Any,
        text: bool,
    ) -> _FakeProcess:
        del command, shell, cwd, env, stdout, stderr, text
        return _FakeProcess()

    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module.subprocess,
        "Popen",
        _fake_popen,
    )

    outcome = prompt_runtime.RuntimeClient().run_ephemeral(
        prompt_runtime.EphemeralRunRequest(
            prompt="already rendered prompt",
            worktree=tmp_path,
            provider_selection=stage_selection_factory(
                service="codex",
                model="gpt-5.4",
                effort="high",
            ),
            tool_access=contracts_runtime.ToolAccess.workspace_backed(tmp_path),
        )
    )

    assert outcome.usage == runtime.ProviderUsage(
        input_tokens=120,
        output_tokens=45,
        cache_read_input_tokens=30,
        cache_creation_input_tokens=None,
        cost_usd=None,
        duration_seconds=None,
    )


def test_runtime_client_writes_ephemeral_invocation_log_only_when_logs_dir_is_supplied(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stage_selection_factory: Callable[..., runtime.ProviderSelection],
) -> None:
    logs_dir = tmp_path / "runtime-logs"
    _stub_codex_prompt_path(monkeypatch)

    class _FakeProcess:
        stdout = iter(
            [
                (
                    '{"type":"text","sessionID":"observed-session","part":'
                    '{"type":"text","time":{"end":"2026-01-01T00:00:00Z"},'
                    '"text":"hello from opencode"}}\n'
                ),
                (
                    '{"type":"session.status","sessionID":"observed-session",'
                    '"status":{"type":"idle"}}\n'
                ),
            ]
        )

        def wait(self) -> int:
            return 0

    def _fake_popen(
        command: str,
        *,
        shell: bool,
        cwd: Path,
        env: dict[str, str],
        stdout: Any,
        stderr: Any,
        text: bool,
    ) -> _FakeProcess:
        del command, shell, cwd, env, stdout, stderr, text
        return _FakeProcess()

    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module.subprocess,
        "Popen",
        _fake_popen,
    )

    outcome = prompt_runtime.RuntimeClient().run_ephemeral(
        prompt_runtime.EphemeralRunRequest(
            prompt="already rendered prompt",
            worktree=tmp_path,
            provider_selection=_selection_with_auth(
                stage_selection_factory(
                    service="opencode",
                    model="glm-5",
                    effort="medium",
                    auth=prompt_runtime.ProviderAuth(opencode_api_key="token"),
                ),
                prompt_runtime.ProviderAuth(opencode_api_key="token"),
            ),
            tool_access=contracts_runtime.ToolAccess.no_tools(),
        )
    )

    assert outcome.output == "hello from opencode"
    assert list(logs_dir.glob("*.log")) == []


def test_runtime_client_does_not_create_ephemeral_invocation_log_when_dispatch_never_starts(
    tmp_path: Path,
    stage_selection_factory: Callable[..., runtime.ProviderSelection],
) -> None:
    logs_dir = tmp_path / "runtime-logs"

    with pytest.raises(AgentCredentialFailureError):
        prompt_runtime.RuntimeClient().run_ephemeral(
            prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                worktree=tmp_path,
                provider_selection=stage_selection_factory(
                    service="opencode",
                    model="glm-5",
                    effort="medium",
                ),
                tool_access=contracts_runtime.ToolAccess.no_tools(),
            )
        )

    assert list(logs_dir.glob("*.log")) == []


def test_runtime_client_exposes_claude_usage_on_completed_outcome(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stage_selection_factory: Callable[..., runtime.ProviderSelection],
) -> None:
    class _FakeProcess:
        stdout = iter(
            [
                (
                    '{"type":"assistant","message":{"content":[{"type":"text",'
                    '"text":"hello from claude"}],"usage":{"input_tokens":100,'
                    '"cache_creation_input_tokens":20,'
                    '"cache_read_input_tokens":30}}}\n'
                ),
                '{"type":"result","result":"hello from claude"}\n',
            ]
        )

        def wait(self) -> int:
            return 0

    def _fake_popen(
        command: str,
        *,
        shell: bool,
        cwd: Path,
        env: dict[str, str],
        stdout: Any,
        stderr: Any,
        text: bool,
    ) -> _FakeProcess:
        del command, shell, cwd, env, stdout, stderr, text
        return _FakeProcess()

    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module.subprocess,
        "Popen",
        _fake_popen,
    )

    outcome = prompt_runtime.RuntimeClient().run_ephemeral(
        prompt_runtime.EphemeralRunRequest(
            prompt="already rendered prompt",
            worktree=tmp_path,
            provider_selection=_selection_with_auth(
                stage_selection_factory(
                    service="claude",
                    model="sonnet",
                    effort="medium",
                    auth=prompt_runtime.ProviderAuth(claude_code_oauth_token="token"),
                ),
                prompt_runtime.ProviderAuth(claude_code_oauth_token="token"),
            ),
            tool_access=contracts_runtime.ToolAccess.no_tools(),
        )
    )

    assert outcome.usage == runtime.ProviderUsage(
        input_tokens=100,
        output_tokens=None,
        cache_read_input_tokens=30,
        cache_creation_input_tokens=20,
        cost_usd=None,
        duration_seconds=None,
    )


def test_runtime_client_keeps_latest_claude_usage_facts_on_completed_outcome(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stage_selection_factory: Callable[..., runtime.ProviderSelection],
) -> None:
    class _FakeProcess:
        stdout = iter(
            [
                (
                    '{"type":"assistant","message":{"content":[{"type":"text",'
                    '"text":"hello from claude"}],"usage":{"input_tokens":100}}}\n'
                ),
                (
                    '{"type":"assistant","message":{"content":[{"type":"text",'
                    '"text":"hello from claude"}],"usage":{"input_tokens":100,'
                    '"cache_creation_input_tokens":20,'
                    '"cache_read_input_tokens":30}}}\n'
                ),
                '{"type":"result","result":"hello from claude"}\n',
            ]
        )

        def wait(self) -> int:
            return 0

    def _fake_popen(
        command: str,
        *,
        shell: bool,
        cwd: Path,
        env: dict[str, str],
        stdout: Any,
        stderr: Any,
        text: bool,
    ) -> _FakeProcess:
        del command, shell, cwd, env, stdout, stderr, text
        return _FakeProcess()

    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module.subprocess,
        "Popen",
        _fake_popen,
    )

    outcome = prompt_runtime.RuntimeClient().run_ephemeral(
        prompt_runtime.EphemeralRunRequest(
            prompt="already rendered prompt",
            worktree=tmp_path,
            provider_selection=_selection_with_auth(
                stage_selection_factory(
                    service="claude",
                    model="sonnet",
                    effort="medium",
                    auth=prompt_runtime.ProviderAuth(claude_code_oauth_token="token"),
                ),
                prompt_runtime.ProviderAuth(claude_code_oauth_token="token"),
            ),
            tool_access=contracts_runtime.ToolAccess.no_tools(),
        )
    )

    assert outcome.usage == runtime.ProviderUsage(
        input_tokens=100,
        output_tokens=None,
        cache_read_input_tokens=30,
        cache_creation_input_tokens=20,
        cost_usd=None,
        duration_seconds=None,
    )


def test_runtime_client_preserves_claude_usage_before_usage_limit_interruption(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stage_selection_factory: Callable[..., runtime.ProviderSelection],
) -> None:
    class _FakeProcess:
        stdout = iter(
            [
                (
                    '{"type":"assistant","message":{"content":[{"type":"text",'
                    '"text":"partial answer"}],"usage":{"input_tokens":40,'
                    '"cache_creation_input_tokens":5,'
                    '"cache_read_input_tokens":8}}}\n'
                ),
                (
                    '{"type":"result","is_error":true,"api_error_status":429,'
                    '"result":"usage limit resets January 2, 5pm (UTC)"}\n'
                ),
            ]
        )

        def wait(self) -> int:
            return 0

    def _fake_popen(
        command: str,
        *,
        shell: bool,
        cwd: Path,
        env: dict[str, str],
        stdout: Any,
        stderr: Any,
        text: bool,
    ) -> _FakeProcess:
        del command, shell, cwd, env, stdout, stderr, text
        return _FakeProcess()

    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module.subprocess,
        "Popen",
        _fake_popen,
    )
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module._time_module,
        "now_local",
        lambda: datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
    )

    outcome = prompt_runtime.RuntimeClient().run_ephemeral(
        prompt_runtime.EphemeralRunRequest(
            prompt="already rendered prompt",
            worktree=tmp_path,
            provider_selection=_selection_with_auth(
                stage_selection_factory(
                    service="claude",
                    model="sonnet",
                    effort="medium",
                    auth=prompt_runtime.ProviderAuth(claude_code_oauth_token="token"),
                ),
                prompt_runtime.ProviderAuth(claude_code_oauth_token="token"),
            ),
            tool_access=contracts_runtime.ToolAccess.no_tools(),
        )
    )

    assert outcome == prompt_runtime.RuntimeOutcome.usage_limited(
        output="",
        service_name="claude",
        reset_time=datetime(2026, 1, 2, 17, 0, tzinfo=timezone.utc),
        invocation_progress=prompt_runtime.InvocationProgress.STARTED,
        usage=runtime.ProviderUsage(
            input_tokens=40,
            output_tokens=None,
            cache_read_input_tokens=8,
            cache_creation_input_tokens=5,
            cost_usd=None,
            duration_seconds=None,
        ),
    )


def test_runtime_client_reports_missing_codex_host_auth_before_subprocess_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stage_selection_factory: Callable[..., runtime.ProviderSelection],
) -> None:
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    _seed_empty_codex_host_auth(monkeypatch, home_dir)

    def _unexpected_popen(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise AssertionError("subprocess should not start without host auth")

    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module.subprocess,
        "Popen",
        _unexpected_popen,
    )

    with pytest.raises(AgentCredentialFailureError) as exc_info:
        prompt_runtime.RuntimeClient().run_ephemeral(
            prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                worktree=tmp_path,
                provider_selection=stage_selection_factory(
                    service="codex",
                    model="gpt-5.4",
                    effort="medium",
                ),
                tool_access=contracts_runtime.ToolAccess.workspace_backed(tmp_path),
            )
        )

    assert str(exc_info.value) == (
        "Codex authentication missing: run `codex login` on the host."
    )
    assert exc_info.value.service_name == "codex"
    assert exc_info.value.status_code == 401
    assert exc_info.value.observations == (
        ProviderErrorObservation(
            service_name="codex",
            raw_provider_text=(
                "Codex authentication missing: run `codex login` on the host."
            ),
            source_stream="pre-dispatch host check",
            status_code=401,
        ),
    )


def test_runtime_client_reports_isolated_missing_codex_host_auth_before_subprocess_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stage_selection_factory: Callable[..., runtime.ProviderSelection],
) -> None:
    host_home_with_login = tmp_path / "host-home-with-login"
    (host_home_with_login / ".codex").mkdir(parents=True, exist_ok=True)
    (host_home_with_login / ".codex" / "auth.json").write_text(
        '{"access_token":"token"}',
        encoding="utf-8",
    )

    isolated_home_without_login = tmp_path / "isolated-home-without-login"

    _stub_codex_prompt_path(monkeypatch)

    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module.Path,
        "home",
        lambda: host_home_with_login,
    )
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module,
        "_codex_host_auth_path",
        lambda: isolated_home_without_login / ".codex" / "auth.json",
        raising=False,
    )

    def _unexpected_popen(
        command: str,
        *,
        shell: bool,
        cwd: Path,
        env: dict[str, str],
        stdout: Any,
        stderr: Any,
        text: bool,
    ) -> None:
        del command, shell, cwd, env, stdout, stderr, text
        raise AssertionError(
            "subprocess should not run when lookup path is missing auth"
        )

    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module.subprocess,
        "Popen",
        _unexpected_popen,
    )

    with pytest.raises(AgentCredentialFailureError) as exc_info:
        runtime.RuntimeClient().run_ephemeral(
            prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                worktree=tmp_path,
                provider_selection=stage_selection_factory(
                    service="codex",
                    model="gpt-5.4",
                    effort="medium",
                ),
                tool_access=contracts_runtime.ToolAccess.workspace_backed(tmp_path),
            )
        )

    assert str(exc_info.value) == (
        "Codex authentication missing: run `codex login` on the host."
    )
    assert exc_info.value.service_name == "codex"
    assert exc_info.value.status_code == 401
    assert exc_info.value.observations == (
        ProviderErrorObservation(
            service_name="codex",
            raw_provider_text=(
                "Codex authentication missing: run `codex login` on the host."
            ),
            source_stream="pre-dispatch host check",
            status_code=401,
        ),
    )


def test_runtime_client_runs_codex_with_isolated_present_host_auth(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stage_selection_factory: Callable[..., runtime.ProviderSelection],
) -> None:
    host_home_without_login = tmp_path / "host-home-without-login"
    host_home_without_login.mkdir()
    host_auth_home = tmp_path / "isolated-home-with-login"
    (host_auth_home / ".codex").mkdir(parents=True, exist_ok=True)
    (host_auth_home / ".codex" / "auth.json").write_text(
        '{"access_token":"token"}',
        encoding="utf-8",
    )

    _stub_codex_prompt_path(monkeypatch)

    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module.Path,
        "home",
        lambda: host_home_without_login,
    )
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module,
        "_codex_host_auth_path",
        lambda: host_auth_home / ".codex" / "auth.json",
        raising=False,
    )

    class _FakeProcess:
        stdout = iter(
            (
                '{"type":"item.completed","item":{"type":"agent_message","text":"hello from codex"}}\n',
                '{"type":"turn.completed"}\n',
            )
        )

        def wait(self) -> int:
            return 0

    def _fake_popen(
        command: str,
        *,
        shell: bool,
        cwd: Path,
        env: dict[str, str],
        stdout: Any,
        stderr: Any,
        text: bool,
    ) -> _FakeProcess:
        del command, shell, cwd, env, stdout, stderr, text
        return _FakeProcess()

    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module.subprocess,
        "Popen",
        _fake_popen,
    )

    outcome = runtime.RuntimeClient().run_ephemeral(
        prompt_runtime.EphemeralRunRequest(
            prompt="already rendered prompt",
            worktree=tmp_path,
            provider_selection=stage_selection_factory(
                service="codex",
                model="gpt-5.4",
                effort="medium",
            ),
            tool_access=contracts_runtime.ToolAccess.workspace_backed(tmp_path),
        )
    )

    assert outcome == prompt_runtime.RuntimeOutcome.completed(
        output="hello from codex",
        result=prompt_runtime.EphemeralRunResult(
            output="hello from codex",
            selected_service="codex",
            selected_model="gpt-5.4",
            selected_effort="medium",
            tool_access=contracts_runtime.ToolAccess.workspace_backed(tmp_path),
            metadata=prompt_runtime.EphemeralResultMetadata(
                runtime=prompt_runtime.EphemeralRuntimeMetadata(
                    run_kind=RunKind.FRESH,
                ),
            ),
        ),
    )


@pytest.mark.parametrize(
    ("tool_access", "expected_flag"),
    [
        (
            contracts_runtime.ToolAccess.no_tools(),
            "--sandbox read-only",
        ),
        (
            contracts_runtime.ToolAccess.workspace_backed(
                Path("."), tool_policy=runtime.ToolPolicy.NONE
            ),
            "--sandbox read-only",
        ),
        (
            contracts_runtime.ToolAccess.workspace_backed(
                Path("."), tool_policy=runtime.ToolPolicy.INSPECT_ONLY
            ),
            "--sandbox read-only",
        ),
        (
            contracts_runtime.ToolAccess.workspace_backed(
                Path("."), tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION
            ),
            "--sandbox read-only",
        ),
        (
            contracts_runtime.ToolAccess.workspace_backed(
                Path("."), tool_policy=runtime.ToolPolicy.UNRESTRICTED
            ),
            "--sandbox danger-full-access",
        ),
    ],
)
def test_runtime_client_preserves_pycastle_codex_sandbox_and_bypass_flag_selection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stage_selection_factory: Callable[..., runtime.ProviderSelection],
    tool_access: contracts_runtime.ToolAccess,
    expected_flag: str,
) -> None:
    home_dir = tmp_path / "home"
    _seed_codex_host_auth(monkeypatch, home_dir)
    _stub_codex_prompt_path(monkeypatch)

    observed_commands: list[str] = []

    class _FakeProcess:
        stdout = iter(['{"type":"turn.completed"}\n'])

        def wait(self) -> int:
            return 0

    def _fake_popen(
        command: str | tuple[str, ...],
        *,
        shell: bool,
        cwd: Path,
        env: dict[str, str],
        stdout: Any,
        stderr: Any,
        text: bool,
        stdin: Any | None = None,
    ) -> _FakeProcess:
        del shell, cwd, env, stdout, stderr, text, stdin
        observed_commands.append(_observed_command_text(command))
        return _FakeProcess()

    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module.subprocess,
        "Popen",
        _fake_popen,
    )

    request_tool_access = (
        tool_access
        if tool_access.kind == "none"
        else contracts_runtime.ToolAccess.workspace_backed(
            tmp_path,
            tool_policy=tool_access.tool_policy,
        )
    )
    prompt_runtime.RuntimeClient().run_ephemeral(
        prompt_runtime.EphemeralRunRequest(
            prompt="already rendered prompt",
            worktree=tmp_path,
            provider_selection=stage_selection_factory(
                service="codex",
                model="gpt-5.4",
                effort="medium",
            ),
            tool_access=request_tool_access,
        )
    )

    assert expected_flag in observed_commands[0]


def test_runtime_client_classifies_codex_refresh_token_reuse_prose_as_credential_lineage_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stage_selection_factory: Callable[..., runtime.ProviderSelection],
) -> None:
    home_dir = tmp_path / "home"
    _seed_codex_host_auth(monkeypatch, home_dir)
    _stub_codex_prompt_path(monkeypatch)

    class _FakeProcess:
        stdout = iter(
            [
                "not json\n",
                '{"type":"error","message":"Access token could not be refreshed because the refresh token was already used."}\n',
            ]
        )

        def wait(self) -> int:
            return 0

    def _fake_popen(
        command: str,
        *,
        shell: bool,
        cwd: Path,
        env: dict[str, str],
        stdout: Any,
        stderr: Any,
        text: bool,
    ) -> _FakeProcess:
        del command, shell, cwd, env, stdout, stderr, text
        return _FakeProcess()

    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module.subprocess,
        "Popen",
        _fake_popen,
    )

    with pytest.raises(AgentCredentialFailureError) as exc_info:
        prompt_runtime.RuntimeClient().run_ephemeral(
            prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                worktree=tmp_path,
                provider_selection=stage_selection_factory(
                    service="codex",
                    model="gpt-5.4",
                    effort="medium",
                ),
                tool_access=contracts_runtime.ToolAccess.workspace_backed(tmp_path),
            )
        )

    assert exc_info.value.classification == "codex_auth_lineage_exhausted"
    assert exc_info.value.service_name == "codex"
    assert exc_info.value.status_code == 401
    assert len(exc_info.value.observations) == 1
    assert exc_info.value.observations[0] == ProviderErrorObservation(
        service_name="codex",
        raw_provider_text=(
            "Access token could not be refreshed because the refresh token was already used."
        ),
        source_stream="json_event.error",
        status_code=401,
    )


def test_runtime_client_returns_usage_limit_outcome_with_parsed_codex_reset_time(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stage_selection_factory: Callable[..., runtime.ProviderSelection],
) -> None:
    home_dir = tmp_path / "home"
    _seed_codex_host_auth(monkeypatch, home_dir)
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module._time_module,
        "now_local",
        lambda: datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
    )
    _stub_codex_prompt_path(monkeypatch)

    class _FakeProcess:
        stdout = iter(
            [
                '{"type":"turn.failed","error":{"message":"You\'ve hit your usage limit. Try again at January 2, 5pm (UTC)."}}\n',
            ]
        )

        def wait(self) -> int:
            return 0

    def _fake_popen(
        command: str,
        *,
        shell: bool,
        cwd: Path,
        env: dict[str, str],
        stdout: Any,
        stderr: Any,
        text: bool,
    ) -> _FakeProcess:
        del command, shell, cwd, env, stdout, stderr, text
        return _FakeProcess()

    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module.subprocess,
        "Popen",
        _fake_popen,
    )

    outcome = prompt_runtime.RuntimeClient().run_ephemeral(
        prompt_runtime.EphemeralRunRequest(
            prompt="already rendered prompt",
            worktree=tmp_path,
            provider_selection=stage_selection_factory(
                service="codex",
                model="gpt-5.4",
                effort="medium",
            ),
            tool_access=contracts_runtime.ToolAccess.workspace_backed(tmp_path),
        )
    )

    assert outcome == prompt_runtime.RuntimeOutcome.usage_limited(
        output="",
        service_name="codex",
        reset_time=datetime(2026, 1, 2, 17, 0, tzinfo=timezone.utc),
        invocation_progress=prompt_runtime.InvocationProgress.NOT_STARTED,
    )


def test_runtime_client_rolls_codex_usage_limit_reset_time_into_next_year_when_needed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stage_selection_factory: Callable[..., runtime.ProviderSelection],
) -> None:
    home_dir = tmp_path / "home"
    _seed_codex_host_auth(monkeypatch, home_dir)
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module._time_module,
        "now_local",
        lambda: datetime(2026, 12, 31, 23, 0, tzinfo=timezone.utc),
    )
    _stub_codex_prompt_path(monkeypatch)

    class _FakeProcess:
        stdout = iter(
            [
                '{"type":"turn.failed","error":{"message":"You\'ve hit your usage limit. Try again at January 2, 5pm (UTC)."}}\n',
            ]
        )

        def wait(self) -> int:
            return 0

    def _fake_popen(
        command: str,
        *,
        shell: bool,
        cwd: Path,
        env: dict[str, str],
        stdout: Any,
        stderr: Any,
        text: bool,
    ) -> _FakeProcess:
        del command, shell, cwd, env, stdout, stderr, text
        return _FakeProcess()

    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module.subprocess,
        "Popen",
        _fake_popen,
    )

    outcome = prompt_runtime.RuntimeClient().run_ephemeral(
        prompt_runtime.EphemeralRunRequest(
            prompt="already rendered prompt",
            worktree=tmp_path,
            provider_selection=stage_selection_factory(
                service="codex",
                model="gpt-5.4",
                effort="medium",
            ),
            tool_access=contracts_runtime.ToolAccess.workspace_backed(tmp_path),
        )
    )

    assert outcome == prompt_runtime.RuntimeOutcome.usage_limited(
        output="",
        service_name="codex",
        reset_time=datetime(2027, 1, 2, 17, 0, tzinfo=timezone.utc),
        invocation_progress=prompt_runtime.InvocationProgress.NOT_STARTED,
    )


def test_runtime_client_reuses_selected_builtin_after_usage_limited_call(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stage_selection_factory: Callable[..., runtime.ProviderSelection],
) -> None:
    home_dir = tmp_path / "home"
    _seed_codex_host_auth(monkeypatch, home_dir)
    _stub_codex_prompt_path(monkeypatch)
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module._time_module,
        "now_local",
        lambda: datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
    )

    observed_commands: list[str] = []

    class _FakeProcess:
        def __init__(self, stdout: Any) -> None:
            self.stdout = stdout

        def wait(self) -> int:
            return 0

    def _fake_popen(
        command: str,
        *,
        shell: bool,
        cwd: Path,
        env: dict[str, str],
        stdout: Any,
        stderr: Any,
        text: bool,
    ) -> _FakeProcess:
        del shell, cwd, env, stdout, stderr, text
        command_text = _observed_command_text(command)
        observed_commands.append(command_text)
        if command_text.startswith(f"{_codex_executable()} exec"):
            return _FakeProcess(
                iter(
                    [
                        '{"type":"turn.failed","error":{"message":"You\'ve hit your usage limit. Try again at January 2, 5pm (UTC)."}}\n',
                    ]
                )
            )
        return _FakeProcess(
            iter(
                [
                    '{"type":"assistant","message":{"content":[{"type":"text","text":"hello from claude"}]}}\n',
                    '{"type":"result","result":"hello from claude"}\n',
                ]
            )
        )

    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module.subprocess,
        "Popen",
        _fake_popen,
    )

    client = prompt_runtime.RuntimeClient()
    first_stage = stage_selection_factory(
        service="codex",
        model="gpt-5.4",
        effort="medium",
    )
    second_stage = stage_selection_factory(
        service="codex",
        model="gpt-5.4",
        effort="medium",
        fallback=stage_selection_factory(
            service="claude",
            model="sonnet",
            effort="medium",
            auth=prompt_runtime.ProviderAuth(claude_code_oauth_token="token"),
        ),
    )

    first_outcome = client.run_ephemeral(
        prompt_runtime.EphemeralRunRequest(
            prompt="already rendered prompt",
            worktree=tmp_path,
            provider_selection=first_stage,
            tool_access=contracts_runtime.ToolAccess.workspace_backed(tmp_path),
        )
    )
    second_outcome = client.run_ephemeral(
        prompt_runtime.EphemeralRunRequest(
            prompt="already rendered prompt",
            worktree=tmp_path,
            provider_selection=second_stage,
            tool_access=contracts_runtime.ToolAccess.workspace_backed(tmp_path),
        )
    )

    assert first_outcome == prompt_runtime.RuntimeOutcome.usage_limited(
        output="",
        service_name="codex",
        reset_time=datetime(2026, 1, 2, 17, 0, tzinfo=timezone.utc),
        invocation_progress=prompt_runtime.InvocationProgress.NOT_STARTED,
    )
    assert second_outcome == prompt_runtime.RuntimeOutcome.usage_limited(
        output="",
        service_name="codex",
        reset_time=datetime(2026, 1, 2, 17, 0, tzinfo=timezone.utc),
        invocation_progress=prompt_runtime.InvocationProgress.NOT_STARTED,
    )
    assert observed_commands == [
        (
            f"{_codex_executable()} exec -m gpt-5.4 -c model_reasoning_effort=medium "
            "-c approval_policy=never --sandbox danger-full-access --json"
        ),
        (
            f"{_codex_executable()} exec -m gpt-5.4 -c model_reasoning_effort=medium "
            "-c approval_policy=never --sandbox danger-full-access --json"
        ),
    ]


def test_runtime_client_instances_keep_independent_builtin_availability_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stage_selection_factory: Callable[..., runtime.ProviderSelection],
) -> None:
    home_dir = tmp_path / "home"
    _seed_codex_host_auth(monkeypatch, home_dir)
    _stub_codex_prompt_path(monkeypatch)
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module._time_module,
        "now_local",
        lambda: datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
    )

    observed_commands: list[str] = []

    class _FakeProcess:
        def __init__(self, stdout: Any) -> None:
            self.stdout = stdout

        def wait(self) -> int:
            return 0

    def _fake_popen(
        command: str,
        *,
        shell: bool,
        cwd: Path,
        env: dict[str, str],
        stdout: Any,
        stderr: Any,
        text: bool,
    ) -> _FakeProcess:
        del shell, cwd, env, stdout, stderr, text
        command_text = _observed_command_text(command)
        observed_commands.append(command_text)
        if command_text.startswith(f"{_codex_executable()} exec"):
            return _FakeProcess(
                iter(
                    [
                        '{"type":"turn.failed","error":{"message":"You\'ve hit your usage limit. Try again at January 2, 5pm (UTC)."}}\n',
                    ]
                )
            )
        return _FakeProcess(
            iter(
                [
                    '{"type":"assistant","message":{"content":[{"type":"text","text":"hello from claude"}]}}\n',
                    '{"type":"result","result":"hello from claude"}\n',
                ]
            )
        )

    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module.subprocess,
        "Popen",
        _fake_popen,
    )

    first_client = prompt_runtime.RuntimeClient()
    second_client = prompt_runtime.RuntimeClient()

    first_outcome = first_client.run_ephemeral(
        prompt_runtime.EphemeralRunRequest(
            prompt="already rendered prompt",
            worktree=tmp_path,
            provider_selection=stage_selection_factory(
                service="codex",
                model="gpt-5.4",
                effort="medium",
            ),
            tool_access=contracts_runtime.ToolAccess.workspace_backed(tmp_path),
        )
    )
    second_outcome = second_client.run_ephemeral(
        prompt_runtime.EphemeralRunRequest(
            prompt="already rendered prompt",
            worktree=tmp_path,
            provider_selection=_selection_with_auth(
                stage_selection_factory(
                    service="codex",
                    model="gpt-5.4",
                    effort="medium",
                    fallback=stage_selection_factory(
                        service="claude",
                        model="sonnet",
                        effort="medium",
                        auth=prompt_runtime.ProviderAuth(
                            claude_code_oauth_token="token"
                        ),
                    ),
                ),
                prompt_runtime.ProviderAuth(claude_code_oauth_token="token"),
            ),
            tool_access=contracts_runtime.ToolAccess.workspace_backed(tmp_path),
        )
    )

    assert first_outcome == prompt_runtime.RuntimeOutcome.usage_limited(
        output="",
        service_name="codex",
        reset_time=datetime(2026, 1, 2, 17, 0, tzinfo=timezone.utc),
        invocation_progress=prompt_runtime.InvocationProgress.NOT_STARTED,
    )
    assert second_outcome == prompt_runtime.RuntimeOutcome.usage_limited(
        output="",
        service_name="codex",
        reset_time=datetime(2026, 1, 2, 17, 0, tzinfo=timezone.utc),
        invocation_progress=prompt_runtime.InvocationProgress.NOT_STARTED,
    )
    assert observed_commands == [
        (
            f"{_codex_executable()} exec -m gpt-5.4 -c model_reasoning_effort=medium "
            "-c approval_policy=never --sandbox danger-full-access --json"
        ),
        (
            f"{_codex_executable()} exec -m gpt-5.4 -c model_reasoning_effort=medium "
            "-c approval_policy=never --sandbox danger-full-access --json"
        ),
    ]


@pytest.mark.parametrize(
    ("tool_policy", "expected_flags"),
    [
        (runtime.ToolPolicy.NONE, ("--disallowedTools all",)),
        (runtime.ToolPolicy.INSPECT_ONLY, ("--tools Read Glob",)),
        (
            runtime.ToolPolicy.NO_FILE_MUTATION,
            ("--disallowedTools Edit Write NotebookEdit",),
        ),
        (runtime.ToolPolicy.UNRESTRICTED, tuple()),
    ],
)
def test_runtime_client_runs_claude_ephemeral_with_tool_policy_commands(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stage_selection_factory: Callable[..., runtime.ProviderSelection],
    tool_policy: runtime.ToolPolicy,
    expected_flags: tuple[str, ...],
) -> None:
    _stub_codex_prompt_path(monkeypatch)

    observed_commands: list[str] = []

    class _FakeProcess:
        def __init__(self, stdout: Any) -> None:
            self.stdout = stdout

        def wait(self) -> int:
            return 0

    def _fake_popen(
        command: str,
        *,
        shell: bool,
        cwd: Path,
        env: dict[str, str],
        stdout: Any,
        stderr: Any,
        text: bool,
    ) -> _FakeProcess:
        del shell, cwd, env, stdout, stderr, text
        observed_commands.append(_observed_command_text(command))
        return _FakeProcess(
            iter(
                [
                    '{"type":"assistant","message":{"content":[{"type":"text","text":"hello from claude"}]}}\n',
                    '{"type":"result","result":"hello from claude"}\n',
                ]
            )
        )

    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module.subprocess,
        "Popen",
        _fake_popen,
    )

    outcome = prompt_runtime.RuntimeClient().run_ephemeral(
        prompt_runtime.EphemeralRunRequest(
            prompt="already rendered prompt",
            worktree=tmp_path,
            provider_selection=_selection_with_auth(
                stage_selection_factory(
                    service="claude",
                    model="sonnet",
                    effort="medium",
                    auth=prompt_runtime.ProviderAuth(claude_code_oauth_token="token"),
                ),
                prompt_runtime.ProviderAuth(claude_code_oauth_token="token"),
            ),
            tool_access=(
                contracts_runtime.ToolAccess.no_tools()
                if tool_policy is runtime.ToolPolicy.NONE
                else contracts_runtime.ToolAccess.workspace_backed(
                    tmp_path,
                    tool_policy=tool_policy,
                )
            ),
        )
    )

    assert outcome == prompt_runtime.RuntimeOutcome.completed(
        output="hello from claude",
        result=prompt_runtime.EphemeralRunResult(
            output="hello from claude",
            selected_service="claude",
            selected_model="sonnet",
            selected_effort="medium",
            tool_access=(
                contracts_runtime.ToolAccess.no_tools()
                if tool_policy is runtime.ToolPolicy.NONE
                else contracts_runtime.ToolAccess.workspace_backed(
                    tmp_path,
                    tool_policy=tool_policy,
                )
            ),
            metadata=prompt_runtime.EphemeralResultMetadata(
                runtime=prompt_runtime.EphemeralRuntimeMetadata(
                    run_kind=RunKind.FRESH,
                ),
            ),
        ),
        usage=None,
    )
    command = observed_commands[0]
    if tool_policy is runtime.ToolPolicy.NONE:
        assert "--tools none" not in command
    if tool_policy is runtime.ToolPolicy.INSPECT_ONLY:
        assert "--tools none" not in command
        assert '--disallowedTools "' not in command
    elif tool_policy is runtime.ToolPolicy.NO_FILE_MUTATION:
        assert "--tools" not in command
    elif tool_policy is runtime.ToolPolicy.UNRESTRICTED:
        assert "--tools" not in command
        assert "--disallowedTools" not in command
    for flag in expected_flags:
        assert flag in command


def test_run_builtin_ephemeral_prefers_argv_for_claude_with_windows_style_prompt_path(
    stage_selection_factory: Callable[..., runtime.ProviderSelection],
) -> None:
    invocation_dir = Path(r"C:\Users\Test User\Prompt Dir")
    adapter = provider_invocation.InMemoryProviderInvocationAdapter(
        prepared_invocations=[
            provider_invocation.ProviderInvocationPreparedStream(
                stdout_lines=(
                    '{"type":"assistant","message":{"content":[{"type":"text","text":"hello from claude"}]}}\n',
                    '{"type":"result","result":"hello from claude"}\n',
                )
            )
        ]
    )

    result = prompt_runtime._run_builtin_ephemeral(
        prompt_runtime.EphemeralRunRequest(
            prompt="already rendered prompt",
            worktree=invocation_dir,
            provider_selection=_selection_with_auth(
                stage_selection_factory(
                    service="claude",
                    model="sonnet",
                    effort="medium",
                    auth=prompt_runtime.ProviderAuth(claude_code_oauth_token="token"),
                ),
                prompt_runtime.ProviderAuth(claude_code_oauth_token="token"),
            ),
            tool_access=contracts_runtime.ToolAccess.workspace_backed(invocation_dir),
        ),
        provider_invocation_adapter=adapter,
    )

    assert result == prompt_runtime.EphemeralRunResult(
        output="hello from claude",
        selected_service="claude",
        selected_model="sonnet",
        selected_effort="medium",
        tool_access=contracts_runtime.ToolAccess.workspace_backed(invocation_dir),
        metadata=prompt_runtime.EphemeralResultMetadata(
            runtime=prompt_runtime.EphemeralRuntimeMetadata(
                run_kind=RunKind.FRESH,
            ),
        ),
        usage=None,
    )
    recorded_request = adapter.recorded_requests[0]
    assert recorded_request.argv == (
        "claude",
        "--verbose",
        "--dangerously-skip-permissions",
        "--output-format",
        "stream-json",
        "-p",
        "-",
        "--disable-slash-commands",
        "--exclude-dynamic-system-prompt-sections",
        "--strict-mcp-config",
        "--mcp-config",
        '{"mcpServers":{}}',
        "--model",
        "sonnet",
        "--effort",
        "medium",
    )
    assert recorded_request.prefer_argv is True
    assert recorded_request.command == (
        "claude --verbose --dangerously-skip-permissions --output-format "
        "stream-json -p - --disable-slash-commands "
        "--exclude-dynamic-system-prompt-sections --strict-mcp-config "
        "--mcp-config '{\"mcpServers\":{}}' --model sonnet --effort medium < "
        f"'{recorded_request.prompt.path}'"
    )
    assert recorded_request.prompt.path == invocation_dir / ".provider_prompt"


@pytest.mark.parametrize(
    ("service_name", "model", "stdout_lines"),
    [
        (
            "codex",
            "gpt-5.4",
            (
                '{"type":"thread.started","thread_id":"thread-123"}\n',
                '{"type":"item.completed","item":{"type":"agent_message","text":"hello from codex"}}\n',
                '{"type":"turn.completed"}\n',
            ),
        ),
        (
            "opencode",
            "glm-5",
            (
                (
                    '{"type":"text","sessionID":"observed-session","part":'
                    '{"type":"text","time":{"end":"2026-01-01T00:00:00Z"},'
                    '"text":"hello from opencode"}}\n'
                ),
                (
                    '{"type":"session.status","sessionID":"observed-session",'
                    '"status":{"type":"idle"}}\n'
                ),
            ),
        ),
    ],
)
def test_run_builtin_ephemeral_non_claude_uses_runtime_neutral_temp_prompt_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stage_selection_factory: Callable[..., runtime.ProviderSelection],
    service_name: str,
    model: str,
    stdout_lines: tuple[str, ...],
) -> None:
    if service_name == "codex":
        _seed_codex_host_auth(monkeypatch, tmp_path / "home")
        auth = None
    else:
        auth = prompt_runtime.ProviderAuth(opencode_api_key="token")

    adapter = provider_invocation.InMemoryProviderInvocationAdapter(
        prepared_invocations=[
            provider_invocation.ProviderInvocationPreparedStream(
                stdout_lines=stdout_lines
            )
        ]
    )

    prompt_runtime._run_builtin_ephemeral(
        prompt_runtime.EphemeralRunRequest(
            prompt="already rendered prompt",
            worktree=tmp_path,
            provider_selection=_selection_with_auth(
                stage_selection_factory(
                    service=service_name,
                    model=model,
                    effort="medium",
                ),
                auth,
            ),
            tool_access=contracts_runtime.ToolAccess.workspace_backed(tmp_path),
        ),
        provider_invocation_adapter=adapter,
    )

    recorded_request = adapter.recorded_requests[0]
    assert recorded_request.prompt.path == Path("/tmp/.provider_prompt")


@pytest.mark.parametrize(
    ("tool_access", "expected_flag"),
    [
        (
            contracts_runtime.ToolAccess.no_tools(),
            "--sandbox read-only",
        ),
        (
            contracts_runtime.ToolAccess.workspace_backed(
                Path("."), tool_policy=runtime.ToolPolicy.INSPECT_ONLY
            ),
            "--sandbox read-only",
        ),
        (
            contracts_runtime.ToolAccess.workspace_backed(
                Path("."), tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION
            ),
            "--sandbox read-only",
        ),
        (
            contracts_runtime.ToolAccess.workspace_backed(
                Path("."), tool_policy=runtime.ToolPolicy.UNRESTRICTED
            ),
            "--sandbox danger-full-access",
        ),
    ],
)
def test_runtime_client_falls_back_within_stage_chain_after_usage_limited_builtin(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stage_selection_factory: Callable[..., runtime.ProviderSelection],
    tool_access: contracts_runtime.ToolAccess,
    expected_flag: str,
) -> None:
    home_dir = tmp_path / "home"
    _seed_codex_host_auth(monkeypatch, home_dir)
    _stub_codex_prompt_path(monkeypatch)
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module._time_module,
        "now_local",
        lambda: datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
    )

    observed_commands: list[str] = []

    class _FakeProcess:
        def __init__(self, stdout: Any) -> None:
            self.stdout = stdout

        def wait(self) -> int:
            return 0

    def _fake_popen(
        command: str,
        *,
        shell: bool,
        cwd: Path,
        env: dict[str, str],
        stdout: Any,
        stderr: Any,
        text: bool,
    ) -> _FakeProcess:
        del shell, cwd, env, stdout, stderr, text
        command_text = _observed_command_text(command)
        observed_commands.append(command_text)
        if command_text.startswith(f"{_codex_executable()} exec"):
            return _FakeProcess(
                iter(
                    [
                        '{"type":"turn.failed","error":{"message":"You\'ve hit your usage limit. Try again at January 2, 5pm (UTC)."}}\n',
                    ]
                )
            )
        return _FakeProcess(
            iter(
                [
                    '{"type":"assistant","message":{"content":[{"type":"text","text":"hello from claude"}]}}\n',
                    '{"type":"result","result":"hello from claude"}\n',
                ]
            )
        )

    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module.subprocess,
        "Popen",
        _fake_popen,
    )

    outcome = prompt_runtime.RuntimeClient().run_ephemeral(
        prompt_runtime.EphemeralRunRequest(
            prompt="already rendered prompt",
            worktree=tmp_path,
            provider_selection=_selection_with_auth(
                stage_selection_factory(
                    service="codex",
                    model="gpt-5.4",
                    effort="medium",
                    fallback=stage_selection_factory(
                        service="claude",
                        model="sonnet",
                        effort="medium",
                        auth=prompt_runtime.ProviderAuth(
                            claude_code_oauth_token="token"
                        ),
                    ),
                ),
                prompt_runtime.ProviderAuth(claude_code_oauth_token="token"),
            ),
            tool_access=(
                tool_access
                if tool_access.kind == "none"
                else contracts_runtime.ToolAccess.workspace_backed(
                    tmp_path, tool_policy=tool_access.tool_policy
                )
            ),
        )
    )

    assert outcome == prompt_runtime.RuntimeOutcome.usage_limited(
        output="",
        service_name="codex",
        reset_time=datetime(2026, 1, 2, 17, 0, tzinfo=timezone.utc),
        invocation_progress=prompt_runtime.InvocationProgress.NOT_STARTED,
    )
    assert observed_commands[0] == (
        f"{_codex_executable()} exec -m gpt-5.4 -c model_reasoning_effort=medium "
        f"-c approval_policy=never {expected_flag} --json"
    )
    assert len(observed_commands) == 1


def test_runtime_client_reports_no_service_available_when_every_reachable_builtin_is_exhausted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stage_selection_factory: Callable[..., runtime.ProviderSelection],
) -> None:
    home_dir = tmp_path / "home"
    _seed_codex_host_auth(monkeypatch, home_dir)
    _stub_codex_prompt_path(monkeypatch)
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module._time_module,
        "now_local",
        lambda: datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
    )

    observed_commands: list[str] = []

    class _FakeProcess:
        def __init__(self, stdout: Any) -> None:
            self.stdout = stdout

        def wait(self) -> int:
            return 0

    def _fake_popen(
        command: str,
        *,
        shell: bool,
        cwd: Path,
        env: dict[str, str],
        stdout: Any,
        stderr: Any,
        text: bool,
    ) -> _FakeProcess:
        del shell, cwd, env, stdout, stderr, text
        command_text = _observed_command_text(command)
        observed_commands.append(command_text)
        if command_text.startswith(f"{_codex_executable()} exec"):
            return _FakeProcess(
                iter(
                    [
                        '{"type":"turn.failed","error":{"message":"You\'ve hit your usage limit. Try again at January 3, 5pm (UTC)."}}\n',
                    ]
                )
            )
        return _FakeProcess(
            iter(
                [
                    (
                        '{"type":"result","is_error":true,"api_error_status":429,'
                        '"result":"usage limit resets January 2, 5pm (UTC)"}\n'
                    ),
                ]
            )
        )

    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module.subprocess,
        "Popen",
        _fake_popen,
    )

    outcome = prompt_runtime.RuntimeClient().run_ephemeral(
        prompt_runtime.EphemeralRunRequest(
            prompt="already rendered prompt",
            worktree=tmp_path,
            provider_selection=_selection_with_auth(
                stage_selection_factory(
                    service="codex",
                    model="gpt-5.4",
                    effort="medium",
                    fallback=stage_selection_factory(
                        service="claude",
                        model="sonnet",
                        effort="medium",
                        auth=prompt_runtime.ProviderAuth(
                            claude_code_oauth_token="token"
                        ),
                    ),
                ),
                prompt_runtime.ProviderAuth(claude_code_oauth_token="token"),
            ),
            tool_access=contracts_runtime.ToolAccess.workspace_backed(tmp_path),
        )
    )

    assert outcome == prompt_runtime.RuntimeOutcome.usage_limited(
        output="",
        service_name="codex",
        reset_time=datetime(2026, 1, 3, 17, 0, tzinfo=timezone.utc),
        invocation_progress=prompt_runtime.InvocationProgress.NOT_STARTED,
    )
    assert observed_commands == [
        (
            f"{_codex_executable()} exec -m gpt-5.4 -c model_reasoning_effort=medium "
            "-c approval_policy=never --sandbox danger-full-access --json"
        ),
    ]


def test_runtime_client_does_not_fallback_or_mark_availability_on_credential_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stage_selection_factory: Callable[..., runtime.ProviderSelection],
) -> None:
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    _seed_empty_codex_host_auth(monkeypatch, home_dir)

    stage = stage_selection_factory(
        service="codex",
        model="gpt-5.4",
        effort="medium",
        fallback=stage_selection_factory(
            service="claude",
            model="sonnet",
            effort="medium",
            auth=prompt_runtime.ProviderAuth(claude_code_oauth_token="token"),
        ),
    )
    client = prompt_runtime.RuntimeClient()

    with pytest.raises(AgentCredentialFailureError):
        client.run_ephemeral(
            prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                worktree=tmp_path,
                provider_selection=stage,
                tool_access=contracts_runtime.ToolAccess.workspace_backed(tmp_path),
            )
        )

    _seed_codex_host_auth(monkeypatch, home_dir)
    _stub_codex_prompt_path(monkeypatch)
    observed_commands: list[str] = []

    class _FakeProcess:
        stdout = iter(
            [
                '{"type":"item.completed","item":{"type":"agent_message","text":"hello from codex"}}\n',
                '{"type":"turn.completed"}\n',
            ]
        )

        def wait(self) -> int:
            return 0

    def _fake_popen(
        command: str | tuple[str, ...],
        *,
        shell: bool,
        cwd: Path,
        env: dict[str, str],
        stdout: Any,
        stderr: Any,
        text: bool,
        stdin: Any | None = None,
    ) -> _FakeProcess:
        del shell, cwd, env, stdout, stderr, text, stdin
        observed_commands.append(_observed_command_text(command))
        return _FakeProcess()

    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module.subprocess,
        "Popen",
        _fake_popen,
    )

    outcome = client.run_ephemeral(
        prompt_runtime.EphemeralRunRequest(
            prompt="already rendered prompt",
            worktree=tmp_path,
            provider_selection=stage,
            tool_access=contracts_runtime.ToolAccess.workspace_backed(tmp_path),
        )
    )

    assert outcome == prompt_runtime.RuntimeOutcome.completed(
        output="hello from codex",
        result=prompt_runtime.EphemeralRunResult(
            output="hello from codex",
            selected_service="codex",
            selected_model="gpt-5.4",
            selected_effort="medium",
            tool_access=contracts_runtime.ToolAccess.workspace_backed(tmp_path),
            metadata=prompt_runtime.EphemeralResultMetadata(
                runtime=prompt_runtime.EphemeralRuntimeMetadata(
                    run_kind=RunKind.FRESH,
                ),
            ),
        ),
    )
    assert observed_commands == [
        (
            f"{_codex_executable()} exec -m gpt-5.4 -c model_reasoning_effort=medium "
            "-c approval_policy=never --sandbox danger-full-access --json"
        )
    ]


def test_runtime_client_does_not_fallback_or_mark_availability_on_hard_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stage_selection_factory: Callable[..., runtime.ProviderSelection],
) -> None:
    home_dir = tmp_path / "home"
    _seed_codex_host_auth(monkeypatch, home_dir)
    _stub_codex_prompt_path(monkeypatch)

    stage = stage_selection_factory(
        service="codex",
        model="gpt-5.4",
        effort="medium",
        fallback=stage_selection_factory(
            service="claude",
            model="sonnet",
            effort="medium",
        ),
    )
    client = prompt_runtime.RuntimeClient()

    class _HardFailureProcess:
        stdout = iter(['{"type":"error","message":"invalid token"}\n'])

        def wait(self) -> int:
            return 0

    def _hard_failure_popen(
        command: str,
        *,
        shell: bool,
        cwd: Path,
        env: dict[str, str],
        stdout: Any,
        stderr: Any,
        text: bool,
    ) -> _HardFailureProcess:
        del command, shell, cwd, env, stdout, stderr, text
        return _HardFailureProcess()

    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module.subprocess,
        "Popen",
        _hard_failure_popen,
    )

    with pytest.raises(HardAgentError):
        client.run_ephemeral(
            prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                worktree=tmp_path,
                provider_selection=stage,
                tool_access=contracts_runtime.ToolAccess.workspace_backed(tmp_path),
            )
        )

    observed_commands: list[str] = []

    class _SuccessProcess:
        stdout = iter(
            [
                '{"type":"item.completed","item":{"type":"agent_message","text":"hello from codex"}}\n',
                '{"type":"turn.completed"}\n',
            ]
        )

        def wait(self) -> int:
            return 0

    def _success_popen(
        command: str | tuple[str, ...],
        *,
        shell: bool,
        cwd: Path,
        env: dict[str, str],
        stdout: Any,
        stderr: Any,
        text: bool,
        stdin: Any | None = None,
    ) -> _SuccessProcess:
        del shell, cwd, env, stdout, stderr, text, stdin
        observed_commands.append(_observed_command_text(command))
        return _SuccessProcess()

    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module.subprocess,
        "Popen",
        _success_popen,
    )

    outcome = client.run_ephemeral(
        prompt_runtime.EphemeralRunRequest(
            prompt="already rendered prompt",
            worktree=tmp_path,
            provider_selection=stage,
            tool_access=contracts_runtime.ToolAccess.workspace_backed(tmp_path),
        )
    )

    assert outcome == prompt_runtime.RuntimeOutcome.completed(
        output="hello from codex",
        result=prompt_runtime.EphemeralRunResult(
            output="hello from codex",
            selected_service="codex",
            selected_model="gpt-5.4",
            selected_effort="medium",
            tool_access=contracts_runtime.ToolAccess.workspace_backed(tmp_path),
            metadata=prompt_runtime.EphemeralResultMetadata(
                runtime=prompt_runtime.EphemeralRuntimeMetadata(
                    run_kind=RunKind.FRESH,
                ),
            ),
        ),
    )
    assert observed_commands == [
        (
            f"{_codex_executable()} exec -m gpt-5.4 -c model_reasoning_effort=medium "
            "-c approval_policy=never --sandbox danger-full-access --json"
        )
    ]


def test_runtime_client_reuses_selected_builtin_after_concurrent_usage_limit_update(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stage_selection_factory: Callable[..., runtime.ProviderSelection],
) -> None:
    home_dir = tmp_path / "home"
    _seed_codex_host_auth(monkeypatch, home_dir)
    _stub_codex_prompt_path(monkeypatch)
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module._time_module,
        "now_local",
        lambda: datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
    )

    observed_commands: list[str] = []
    codex_started = threading.Event()
    release_codex = threading.Event()
    codex_calls = 0

    class _FakeProcess:
        def __init__(self, stdout: Any) -> None:
            self.stdout = stdout

        def wait(self) -> int:
            return 0

    def _fake_popen(
        command: str,
        *,
        shell: bool,
        cwd: Path,
        env: dict[str, str],
        stdout: Any,
        stderr: Any,
        text: bool,
    ) -> _FakeProcess:
        nonlocal codex_calls
        del shell, cwd, env, stdout, stderr, text
        command_text = _observed_command_text(command)
        observed_commands.append(command_text)
        if command_text.startswith(f"{_codex_executable()} exec"):
            codex_calls += 1
            codex_started.set()
            release_codex.wait(timeout=2)
            return _FakeProcess(
                iter(
                    [
                        '{"type":"turn.failed","error":{"message":"You\'ve hit your usage limit. Try again at January 2, 5pm (UTC)."}}\n',
                    ]
                )
            )
        return _FakeProcess(
            iter(
                [
                    '{"type":"assistant","message":{"content":[{"type":"text","text":"hello from claude"}]}}\n',
                    '{"type":"result","result":"hello from claude"}\n',
                ]
            )
        )

    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module.subprocess,
        "Popen",
        _fake_popen,
    )

    client = prompt_runtime.RuntimeClient()
    first_result: list[prompt_runtime.RuntimeOutcome] = []

    def _run_first_call() -> None:
        first_result.append(
            client.run_ephemeral(
                prompt_runtime.EphemeralRunRequest(
                    prompt="already rendered prompt",
                    worktree=tmp_path,
                    provider_selection=stage_selection_factory(
                        service="codex",
                        model="gpt-5.4",
                        effort="medium",
                    ),
                    tool_access=contracts_runtime.ToolAccess.workspace_backed(tmp_path),
                )
            )
        )

    thread = threading.Thread(target=_run_first_call)
    thread.start()
    assert codex_started.wait(timeout=2)
    release_codex.set()
    thread.join(timeout=2)
    assert not thread.is_alive()

    second_outcome = client.run_ephemeral(
        prompt_runtime.EphemeralRunRequest(
            prompt="already rendered prompt",
            worktree=tmp_path,
            provider_selection=_selection_with_auth(
                stage_selection_factory(
                    service="codex",
                    model="gpt-5.4",
                    effort="medium",
                    fallback=stage_selection_factory(
                        service="claude",
                        model="sonnet",
                        effort="medium",
                        auth=prompt_runtime.ProviderAuth(
                            claude_code_oauth_token="token"
                        ),
                    ),
                ),
                prompt_runtime.ProviderAuth(claude_code_oauth_token="token"),
            ),
            tool_access=contracts_runtime.ToolAccess.workspace_backed(tmp_path),
        )
    )

    assert first_result == [
        prompt_runtime.RuntimeOutcome.usage_limited(
            output="",
            service_name="codex",
            reset_time=datetime(2026, 1, 2, 17, 0, tzinfo=timezone.utc),
            invocation_progress=prompt_runtime.InvocationProgress.NOT_STARTED,
        )
    ]
    assert second_outcome == prompt_runtime.RuntimeOutcome.usage_limited(
        output="",
        service_name="codex",
        reset_time=datetime(2026, 1, 2, 17, 0, tzinfo=timezone.utc),
        invocation_progress=prompt_runtime.InvocationProgress.NOT_STARTED,
    )
    assert codex_calls == 2
    assert observed_commands == [
        (
            f"{_codex_executable()} exec -m gpt-5.4 -c model_reasoning_effort=medium "
            "-c approval_policy=never --sandbox danger-full-access --json"
        ),
        (
            f"{_codex_executable()} exec -m gpt-5.4 -c model_reasoning_effort=medium "
            "-c approval_policy=never --sandbox danger-full-access --json"
        ),
    ]


def test_runtime_client_ignores_malformed_codex_lines_before_classifying_hard_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stage_selection_factory: Callable[..., runtime.ProviderSelection],
) -> None:
    home_dir = tmp_path / "home"
    _seed_codex_host_auth(monkeypatch, home_dir)
    _stub_codex_prompt_path(monkeypatch)

    class _FakeProcess:
        stdout = iter(
            [
                '"not a dict"\n',
                "not json\n",
                '{"type":"error","message":"invalid token"}\n',
            ]
        )

        def wait(self) -> int:
            return 0

    def _fake_popen(
        command: str,
        *,
        shell: bool,
        cwd: Path,
        env: dict[str, str],
        stdout: Any,
        stderr: Any,
        text: bool,
    ) -> _FakeProcess:
        del command, shell, cwd, env, stdout, stderr, text
        return _FakeProcess()

    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module.subprocess,
        "Popen",
        _fake_popen,
    )

    with pytest.raises(HardAgentError) as exc_info:
        prompt_runtime.RuntimeClient().run_ephemeral(
            prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                worktree=tmp_path,
                provider_selection=stage_selection_factory(
                    service="codex",
                    model="gpt-5.4",
                    effort="medium",
                ),
                tool_access=contracts_runtime.ToolAccess.workspace_backed(tmp_path),
            )
        )

    assert str(exc_info.value) == "invalid token"
    assert exc_info.value.service_name == "codex"
    assert exc_info.value.status_code == 401
    assert exc_info.value.observations == (
        ProviderErrorObservation(
            service_name="codex",
            raw_provider_text="invalid token",
            source_stream="json_event.error",
            status_code=401,
        ),
    )


def test_ephemeral_runtime_runs_prompt_without_preparing_or_returning_continuation_state(
    stage_selection_factory: Callable[..., runtime.ProviderSelection],
    service_registry_factory: Callable[..., ServiceRegistry],
) -> None:
    execution_adapter = _EphemeralExecutionAdapter()

    result = asyncio.run(
        compat_runtime.EphemeralRuntime(
            execution_adapter=execution_adapter,
            service_registry=service_registry_factory("claude"),
        ).run_ephemeral(
            prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                worktree=Path("."),
                provider_selection=stage_selection_factory(
                    service="claude",
                    model="gpt-5",
                    effort="medium",
                ),
                tool_access=contracts_runtime.ToolAccess.no_tools(),
            )
        )
    )

    assert result.output == "implementer:already rendered prompt"
    assert execution_adapter.prepare_session_calls == 0
    assert not hasattr(result.result, "continuation")


def test_ephemeral_runtime_requires_selected_configured_service(
    stage_selection_factory: Callable[..., runtime.ProviderSelection],
    service_registry_factory: Callable[..., ServiceRegistry],
) -> None:
    with pytest.raises(runtime.RuntimeConfigurationError):
        asyncio.run(
            compat_runtime.EphemeralRuntime(
                execution_adapter=_EphemeralExecutionAdapter(),
                service_registry=service_registry_factory("codex", "claude"),
            ).run_ephemeral(
                prompt_runtime.EphemeralRunRequest(
                    prompt="already rendered prompt",
                    worktree=Path("."),
                    provider_selection=stage_selection_factory(
                        service="missing",
                        fallback=stage_selection_factory(
                            service="claude",
                            model="sonnet",
                            effort="high",
                        ),
                    ),
                    tool_access=contracts_runtime.ToolAccess.no_tools(),
                )
            )
        )


def test_ephemeral_runtime_applies_runtime_setup_failure_translation(
    stage_selection_factory: Callable[..., runtime.ProviderSelection],
    service_registry_factory: Callable[..., ServiceRegistry],
) -> None:
    with pytest.raises(AgentCredentialFailureError) as exc_info:
        asyncio.run(
            compat_runtime.EphemeralRuntime(
                execution_adapter=_SetupTranslatedEphemeralExecutionAdapter(),
                service_registry=service_registry_factory("claude"),
            ).run_ephemeral(
                prompt_runtime.EphemeralRunRequest(
                    prompt="already rendered prompt",
                    worktree=Path("."),
                    provider_selection=stage_selection_factory(
                        service="claude",
                        model="gpt-5",
                        effort="medium",
                    ),
                    tool_access=contracts_runtime.ToolAccess.no_tools(),
                )
            )
        )

    assert str(exc_info.value) == "missing auth"
    assert exc_info.value.service_name == "claude"


def test_ephemeral_runtime_returns_usage_limited_outcome_for_usage_limit_conditions(
    stage_selection_factory: Callable[..., runtime.ProviderSelection],
    service_registry_factory: Callable[..., ServiceRegistry],
) -> None:
    result = asyncio.run(
        compat_runtime.EphemeralRuntime(
            execution_adapter=_UsageLimitThenSuccessEphemeralExecutionAdapter(),
            service_registry=service_registry_factory("codex"),
        ).run_ephemeral(
            prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                worktree=Path("."),
                provider_selection=stage_selection_factory(
                    service="codex",
                    model="gpt-5",
                    effort="medium",
                ),
                tool_access=contracts_runtime.ToolAccess.no_tools(),
            )
        )
    )

    assert result == prompt_runtime.RuntimeOutcome.usage_limited(
        output="",
        service_name="codex",
        reset_time=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
        invocation_progress=prompt_runtime.InvocationProgress.STARTED,
    )
    assert result.result is None


def test_ephemeral_runtime_returns_no_service_available_outcome_for_temporarily_unavailable_services(
    stage_selection_factory: Callable[..., runtime.ProviderSelection],
    service_registry_factory: Callable[..., ServiceRegistry],
) -> None:
    result = asyncio.run(
        compat_runtime.EphemeralRuntime(
            execution_adapter=_EphemeralExecutionAdapter(),
            service_registry=service_registry_factory(
                "codex",
                "claude",
                unavailable={"codex"},
            ),
        ).run_ephemeral(
            prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                worktree=Path("."),
                provider_selection=stage_selection_factory(
                    service="codex",
                    model="gpt-5",
                    effort="medium",
                    fallback=stage_selection_factory(
                        service="claude",
                        model="sonnet",
                        effort="high",
                    ),
                ),
                tool_access=contracts_runtime.ToolAccess.no_tools(),
            )
        )
    )

    assert result == prompt_runtime.RuntimeOutcome.no_service_available(
        output="",
        reset_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        invocation_progress=prompt_runtime.InvocationProgress.NOT_STARTED,
    )
    assert result.result is None


def test_ephemeral_runtime_returns_cancelled_outcome_for_caller_cancellation(
    stage_selection_factory: Callable[..., runtime.ProviderSelection],
    service_registry_factory: Callable[..., ServiceRegistry],
) -> None:
    cancelled_token = CancellationToken()
    cancelled_token.cancel()

    result = asyncio.run(
        compat_runtime.EphemeralRuntime(
            execution_adapter=_EphemeralExecutionAdapter(),
            service_registry=service_registry_factory("codex"),
        ).run_ephemeral(
            prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                worktree=Path("."),
                provider_selection=stage_selection_factory(
                    service="codex",
                    model="gpt-5",
                    effort="medium",
                ),
                tool_access=contracts_runtime.ToolAccess.no_tools(),
                token=cancelled_token,
            )
        )
    )

    assert result == prompt_runtime.RuntimeOutcome.cancelled(
        output="",
        invocation_progress=prompt_runtime.InvocationProgress.NOT_STARTED,
    )
    assert result.result is None


def test_ephemeral_runtime_returns_timed_out_outcome_for_timeout_conditions(
    stage_selection_factory: Callable[..., runtime.ProviderSelection],
    service_registry_factory: Callable[..., ServiceRegistry],
) -> None:
    result = asyncio.run(
        compat_runtime.EphemeralRuntime(
            execution_adapter=_TimeoutEphemeralExecutionAdapter(),
            service_registry=service_registry_factory("codex"),
        ).run_ephemeral(
            prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                worktree=Path("."),
                provider_selection=stage_selection_factory(
                    service="codex",
                    model="gpt-5",
                    effort="medium",
                ),
                tool_access=contracts_runtime.ToolAccess.no_tools(),
            )
        )
    )

    assert result == prompt_runtime.RuntimeOutcome.timed_out(
        output="",
        invocation_progress=prompt_runtime.InvocationProgress.NOT_STARTED,
    )
    assert result.result is None


def test_ephemeral_runtime_returns_retryable_provider_failure_outcome_for_retryable_provider_failures(
    stage_selection_factory: Callable[..., runtime.ProviderSelection],
    service_registry_factory: Callable[..., ServiceRegistry],
) -> None:
    result = asyncio.run(
        compat_runtime.EphemeralRuntime(
            execution_adapter=_RetryableProviderFailureEphemeralExecutionAdapter(),
            service_registry=service_registry_factory("codex"),
        ).run_ephemeral(
            prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                worktree=Path("."),
                provider_selection=stage_selection_factory(
                    service="codex",
                    model="gpt-5",
                    effort="medium",
                ),
                tool_access=contracts_runtime.ToolAccess.no_tools(),
            )
        )
    )

    assert result == prompt_runtime.RuntimeOutcome.retryable_provider_failure(
        output="",
        service_name="codex",
        invocation_progress=prompt_runtime.InvocationProgress.NOT_STARTED,
    )
    assert result.result is None


@pytest.mark.parametrize(
    ("error", "expected_outcome"),
    [
        pytest.param(
            UsageLimitError(
                reset_time=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
                service_name="codex",
                invocation_progress=runtime.InvocationProgress.STARTED,
                usage=runtime.ProviderUsage(input_tokens=10, output_tokens=4),
            ),
            prompt_runtime.RuntimeOutcome.usage_limited(
                output="",
                service_name="codex",
                reset_time=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
                invocation_progress=prompt_runtime.InvocationProgress.STARTED,
                usage=runtime.ProviderUsage(input_tokens=10, output_tokens=4),
            ),
            id="usage-limited",
        ),
        pytest.param(
            AgentCancelledError(
                invocation_progress=runtime.InvocationProgress.STARTED,
                usage=runtime.ProviderUsage(input_tokens=7),
            ),
            prompt_runtime.RuntimeOutcome.cancelled(
                output="",
                invocation_progress=prompt_runtime.InvocationProgress.STARTED,
                usage=runtime.ProviderUsage(input_tokens=7),
            ),
            id="cancelled",
        ),
        pytest.param(
            AgentTimeoutError(
                "timed out",
                invocation_progress=runtime.InvocationProgress.STARTED,
                usage=runtime.ProviderUsage(input_tokens=8),
            ),
            prompt_runtime.RuntimeOutcome.timed_out(
                output="",
                invocation_progress=prompt_runtime.InvocationProgress.STARTED,
                usage=runtime.ProviderUsage(input_tokens=8),
            ),
            id="timed-out",
        ),
        pytest.param(
            RetryableProviderFailureError(
                "retry later",
                service_name="codex",
                invocation_progress=runtime.InvocationProgress.STARTED,
                usage=runtime.ProviderUsage(input_tokens=9, output_tokens=2),
            ),
            prompt_runtime.RuntimeOutcome.retryable_provider_failure(
                output="",
                service_name="codex",
                invocation_progress=prompt_runtime.InvocationProgress.STARTED,
                usage=runtime.ProviderUsage(input_tokens=9, output_tokens=2),
            ),
            id="retryable-provider-failure",
        ),
    ],
)
def test_ephemeral_runtime_preserves_observed_usage_on_interrupted_outcomes(
    stage_selection_factory: Callable[..., runtime.ProviderSelection],
    service_registry_factory: Callable[..., ServiceRegistry],
    error: Exception,
    expected_outcome: prompt_runtime.RuntimeOutcome,
) -> None:
    result = asyncio.run(
        compat_runtime.EphemeralRuntime(
            execution_adapter=_InterruptedEphemeralExecutionAdapter(error),
            service_registry=service_registry_factory("codex"),
        ).run_ephemeral(
            prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                worktree=Path("."),
                provider_selection=stage_selection_factory(
                    service="codex",
                    model="gpt-5",
                    effort="medium",
                ),
                tool_access=contracts_runtime.ToolAccess.no_tools(),
            )
        )
    )

    assert result == expected_outcome


def test_ephemeral_runtime_keeps_exceptional_failures_exceptional(
    stage_selection_factory: Callable[..., runtime.ProviderSelection],
    service_registry_factory: Callable[..., ServiceRegistry],
) -> None:
    with pytest.raises(runtime.RuntimeConfigurationError):
        asyncio.run(
            compat_runtime.EphemeralRuntime(
                execution_adapter=_EphemeralExecutionAdapter(),
                service_registry=service_registry_factory("codex"),
            ).run_ephemeral(
                prompt_runtime.EphemeralRunRequest(
                    prompt="already rendered prompt",
                    worktree=Path("."),
                    provider_selection=stage_selection_factory(
                        service="missing",
                        fallback=stage_selection_factory(
                            service="also-missing",
                            model="sonnet",
                            effort="high",
                        ),
                    ),
                    tool_access=contracts_runtime.ToolAccess.no_tools(),
                )
            )
        )

    with pytest.raises(AgentCredentialFailureError):
        asyncio.run(
            compat_runtime.EphemeralRuntime(
                execution_adapter=_SetupTranslatedEphemeralExecutionAdapter(),
                service_registry=service_registry_factory("codex"),
            ).run_ephemeral(
                prompt_runtime.EphemeralRunRequest(
                    prompt="already rendered prompt",
                    worktree=Path("."),
                    provider_selection=stage_selection_factory(
                        service="codex",
                        model="gpt-5",
                        effort="medium",
                    ),
                    tool_access=contracts_runtime.ToolAccess.no_tools(),
                )
            )
        )

    with pytest.raises(HardAgentError):
        asyncio.run(
            compat_runtime.EphemeralRuntime(
                execution_adapter=_HardFailureEphemeralExecutionAdapter(),
                service_registry=service_registry_factory("codex"),
            ).run_ephemeral(
                prompt_runtime.EphemeralRunRequest(
                    prompt="already rendered prompt",
                    worktree=Path("."),
                    provider_selection=stage_selection_factory(
                        service="codex",
                        model="gpt-5",
                        effort="medium",
                    ),
                    tool_access=contracts_runtime.ToolAccess.no_tools(),
                )
            )
        )

    with pytest.raises(TransientAgentError):
        asyncio.run(
            compat_runtime.EphemeralRuntime(
                execution_adapter=_TransientProviderFailureEphemeralExecutionAdapter(),
                service_registry=service_registry_factory("codex"),
            ).run_ephemeral(
                prompt_runtime.EphemeralRunRequest(
                    prompt="already rendered prompt",
                    worktree=Path("."),
                    provider_selection=stage_selection_factory(
                        service="codex",
                        model="gpt-5",
                        effort="medium",
                    ),
                    tool_access=contracts_runtime.ToolAccess.no_tools(),
                )
            )
        )


@pytest.mark.parametrize("tool_policy", _TOOL_POLICY_CASES)
def test_text_output_adapter_exposes_tool_policy_effects_through_public_adapter_seam(
    tool_policy: runtime.ToolPolicy | contracts_runtime.ToolPolicyProfile,
) -> None:
    output = asyncio.run(
        TextOutputAdapter(
            prompt="already rendered prompt",
            tool_policy=tool_policy,
        ).invoke(
            runner=cast(WorkExecutionAdapter, _ToolPolicyRenderingPromptRunner()),
            prompt="already rendered prompt",
            role=InvocationRole("implementer"),
            run_kind=RunKind.FRESH,
            session_uuid=None,
            on_provider_session_id=lambda _provider_session_id: None,
        )
    )

    assert output == _tool_policy_effect_text(tool_policy)


def test_text_output_adapter_explicit_no_tools_forbids_provider_tool_access() -> None:
    output = asyncio.run(
        TextOutputAdapter(
            prompt="already rendered prompt",
            tool_access=contracts_runtime.ToolAccess.no_tools(),
        ).invoke(
            runner=cast(WorkExecutionAdapter, _ToolPolicyRenderingPromptRunner()),
            prompt="already rendered prompt",
            role=InvocationRole("implementer"),
            run_kind=RunKind.FRESH,
            session_uuid=None,
            on_provider_session_id=lambda _provider_session_id: None,
        )
    )

    assert output == "allowed=none;disallowed=all"


def test_text_output_adapter_uses_workspace_backed_tool_access_through_public_adapter_seam() -> (
    None
):
    output = asyncio.run(
        TextOutputAdapter(
            prompt="already rendered prompt",
            tool_access=contracts_runtime.ToolAccess.workspace_backed(
                Path("/repo"),
                tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
            ),
            workspace=Path("/repo"),
        ).invoke(
            runner=cast(WorkExecutionAdapter, _ToolPolicyRenderingPromptRunner()),
            prompt="already rendered prompt",
            role=InvocationRole("implementer"),
            run_kind=RunKind.FRESH,
            session_uuid=None,
            on_provider_session_id=lambda _provider_session_id: None,
        )
    )

    assert output == _tool_policy_effect_text(runtime.ToolPolicy.NO_FILE_MUTATION)


def test_text_output_adapter_prefers_tool_access_over_compatibility_tool_policy() -> (
    None
):
    output = asyncio.run(
        TextOutputAdapter(
            prompt="already rendered prompt",
            tool_policy=runtime.ToolPolicy.UNRESTRICTED,
            tool_access=contracts_runtime.ToolAccess.workspace_backed(
                Path("/repo"),
                tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
            ),
            workspace=Path("/repo"),
        ).invoke(
            runner=cast(WorkExecutionAdapter, _ToolPolicyRenderingPromptRunner()),
            prompt="already rendered prompt",
            role=InvocationRole("implementer"),
            run_kind=RunKind.FRESH,
            session_uuid=None,
            on_provider_session_id=lambda _provider_session_id: None,
        )
    )

    assert output == _tool_policy_effect_text(runtime.ToolPolicy.NO_FILE_MUTATION)


def test_text_output_adapter_rejects_workspace_backed_tool_access_without_workspace_context() -> (
    None
):
    with pytest.raises(
        ValueError,
        match=re.escape(
            "TextOutputAdapter workspace-backed tool access requires worktree /repo, got None."
        ),
    ):
        TextOutputAdapter(
            prompt="already rendered prompt",
            tool_access=contracts_runtime.ToolAccess.workspace_backed(Path("/repo")),
        )
