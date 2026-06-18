from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import pytest

import agent_runtime as runtime
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
from agent_runtime.execution_contracts import (
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
from agent_runtime.service_registry import ServiceRegistry
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
        tool_policy: Any = runtime.ToolPolicy.FULL,
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
        tool_policy: Any = runtime.ToolPolicy.FULL,
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
        tool_policy: Any = runtime.ToolPolicy.FULL,
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
        tool_policy: Any = runtime.ToolPolicy.FULL,
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
        tool_policy: Any = runtime.ToolPolicy.FULL,
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
        tool_policy: Any = runtime.ToolPolicy.FULL,
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
        tool_policy: Any = runtime.ToolPolicy.FULL,
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
        tool_policy: Any = runtime.ToolPolicy.FULL,
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
        runtime.ToolPolicyProfile(
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
    monkeypatch.setenv("HOME", str(home_dir))


def _stub_codex_prompt_path(
    monkeypatch: pytest.MonkeyPatch,
    *,
    on_write: Callable[[str], None] | None = None,
    on_unlink: Callable[[], None] | None = None,
) -> None:
    prompt_path = Path("/tmp/.pycastle_prompt")
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
    stage_selection_factory: Callable[..., runtime.StageSelection],
) -> None:
    home_dir = tmp_path / "home"
    _seed_codex_host_auth(monkeypatch, home_dir)

    observed: dict[str, Any] = {}
    _stub_codex_prompt_path(
        monkeypatch,
        on_write=lambda data: observed.__setitem__("prompt", data),
        on_unlink=lambda: observed.__setitem__("prompt_deleted", True),
    )

    class _FakeProcess:
        def __init__(
            self,
            command: str,
            *,
            cwd: Path,
            env: dict[str, str],
            stdout: Any,
        ) -> None:
            observed["command"] = command
            observed["cwd"] = cwd
            observed["env"] = env
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
        del shell, stderr, text
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
            worktree=tmp_path,
            stage=stage_selection_factory(
                service="codex",
                model="gpt-5.4",
                effort="high",
            ),
            role=InvocationRole("implementer"),
            tool_access=runtime.ToolAccess.workspace_backed(tmp_path),
        )
    )

    assert outcome == prompt_runtime.RuntimeOutcome.completed(
        output="hello from codex",
        result=prompt_runtime.EphemeralRunResult(
            output="hello from codex",
            selected_service="codex",
            selected_model="gpt-5.4",
            selected_effort="high",
            tool_access=runtime.ToolAccess.workspace_backed(tmp_path),
            used_fallback=False,
            metadata=prompt_runtime.EphemeralResultMetadata(
                selected_service_path=("codex",),
                runtime=prompt_runtime.EphemeralRuntimeMetadata(
                    run_kind=RunKind.FRESH,
                    session_namespace="",
                ),
            ),
        ),
    )
    assert observed["command"] == (
        "codex exec -m gpt-5.4 -c model_reasoning_effort=high "
        "-c approval_policy=never --sandbox danger-full-access "
        "--json < /tmp/.pycastle_prompt"
    )
    assert observed["prompt"] == "already rendered prompt"
    assert observed["prompt_deleted"] is True
    assert observed["cwd"] == tmp_path
    assert observed["env"] == {"TZ": "UTC"}


def test_runtime_client_exposes_codex_usage_on_completed_outcome(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stage_selection_factory: Callable[..., runtime.StageSelection],
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
            stage=stage_selection_factory(
                service="codex",
                model="gpt-5.4",
                effort="high",
            ),
            role=InvocationRole("implementer"),
            tool_access=runtime.ToolAccess.workspace_backed(tmp_path),
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


def test_runtime_client_exposes_claude_usage_on_completed_outcome(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stage_selection_factory: Callable[..., runtime.StageSelection],
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
            stage=stage_selection_factory(
                service="claude",
                model="sonnet",
                effort="medium",
            ),
            role=InvocationRole("implementer"),
            tool_access=runtime.ToolAccess.no_tools(),
            auth=prompt_runtime.ProviderAuth(claude_code_oauth_token="token"),
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
    stage_selection_factory: Callable[..., runtime.StageSelection],
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
            stage=stage_selection_factory(
                service="claude",
                model="sonnet",
                effort="medium",
            ),
            role=InvocationRole("implementer"),
            tool_access=runtime.ToolAccess.no_tools(),
            auth=prompt_runtime.ProviderAuth(claude_code_oauth_token="token"),
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
    stage_selection_factory: Callable[..., runtime.StageSelection],
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
            stage=stage_selection_factory(
                service="claude",
                model="sonnet",
                effort="medium",
            ),
            role=InvocationRole("implementer"),
            tool_access=runtime.ToolAccess.no_tools(),
            auth=prompt_runtime.ProviderAuth(claude_code_oauth_token="token"),
        )
    )

    assert outcome == prompt_runtime.RuntimeOutcome.usage_limited(
        output="",
        service_name="claude",
        reset_time=datetime(2026, 1, 2, 17, 0, tzinfo=timezone.utc),
        usage_limit_scope=runtime.UsageLimitScope("implementer"),
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
    stage_selection_factory: Callable[..., runtime.StageSelection],
) -> None:
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    monkeypatch.setenv("HOME", str(home_dir))

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
                stage=stage_selection_factory(
                    service="codex",
                    model="gpt-5.4",
                    effort="medium",
                ),
                role=InvocationRole("implementer"),
                tool_access=runtime.ToolAccess.workspace_backed(tmp_path),
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


@pytest.mark.parametrize(
    ("tool_access", "expected_flag"),
    [
        (
            runtime.ToolAccess.workspace_backed(
                Path("."), tool_policy=runtime.ToolPolicy.RESTRICTED
            ),
            "--sandbox danger-full-access",
        ),
        (
            runtime.ToolAccess.workspace_backed(
                Path("."), tool_policy=runtime.ToolPolicy.PARTIAL
            ),
            "--dangerously-bypass-approvals-and-sandbox",
        ),
        (
            runtime.ToolAccess.no_tools(),
            "--sandbox danger-full-access",
        ),
    ],
)
def test_runtime_client_preserves_pycastle_codex_sandbox_and_bypass_flag_selection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stage_selection_factory: Callable[..., runtime.StageSelection],
    tool_access: runtime.ToolAccess,
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
        observed_commands.append(command)
        return _FakeProcess()

    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module.subprocess,
        "Popen",
        _fake_popen,
    )

    request_tool_access = (
        tool_access
        if tool_access.kind == "none"
        else runtime.ToolAccess.workspace_backed(
            tmp_path,
            tool_policy=tool_access.tool_policy,
        )
    )
    prompt_runtime.RuntimeClient().run_ephemeral(
        prompt_runtime.EphemeralRunRequest(
            prompt="already rendered prompt",
            worktree=tmp_path,
            stage=stage_selection_factory(
                service="codex",
                model="gpt-5.4",
                effort="medium",
            ),
            role=InvocationRole("implementer"),
            tool_access=request_tool_access,
        )
    )

    assert expected_flag in observed_commands[0]


def test_runtime_client_classifies_codex_refresh_token_reuse_prose_as_credential_lineage_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stage_selection_factory: Callable[..., runtime.StageSelection],
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
                stage=stage_selection_factory(
                    service="codex",
                    model="gpt-5.4",
                    effort="medium",
                ),
                role=InvocationRole("implementer"),
                tool_access=runtime.ToolAccess.workspace_backed(tmp_path),
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
    stage_selection_factory: Callable[..., runtime.StageSelection],
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
            stage=stage_selection_factory(
                service="codex",
                model="gpt-5.4",
                effort="medium",
            ),
            role=InvocationRole("implementer"),
            tool_access=runtime.ToolAccess.workspace_backed(tmp_path),
        )
    )

    assert outcome == prompt_runtime.RuntimeOutcome.usage_limited(
        output="",
        service_name="codex",
        reset_time=datetime(2026, 1, 2, 17, 0, tzinfo=timezone.utc),
        usage_limit_scope=runtime.UsageLimitScope("implementer"),
        invocation_progress=prompt_runtime.InvocationProgress.NOT_STARTED,
    )


def test_runtime_client_rolls_codex_usage_limit_reset_time_into_next_year_when_needed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stage_selection_factory: Callable[..., runtime.StageSelection],
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
            stage=stage_selection_factory(
                service="codex",
                model="gpt-5.4",
                effort="medium",
            ),
            role=InvocationRole("implementer"),
            tool_access=runtime.ToolAccess.workspace_backed(tmp_path),
        )
    )

    assert outcome == prompt_runtime.RuntimeOutcome.usage_limited(
        output="",
        service_name="codex",
        reset_time=datetime(2027, 1, 2, 17, 0, tzinfo=timezone.utc),
        usage_limit_scope=runtime.UsageLimitScope("implementer"),
        invocation_progress=prompt_runtime.InvocationProgress.NOT_STARTED,
    )


def test_runtime_client_ignores_malformed_codex_lines_before_classifying_hard_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stage_selection_factory: Callable[..., runtime.StageSelection],
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
                stage=stage_selection_factory(
                    service="codex",
                    model="gpt-5.4",
                    effort="medium",
                ),
                role=InvocationRole("implementer"),
                tool_access=runtime.ToolAccess.workspace_backed(tmp_path),
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
    stage_selection_factory: Callable[..., runtime.StageSelection],
    service_registry_factory: Callable[..., ServiceRegistry],
) -> None:
    execution_adapter = _EphemeralExecutionAdapter()

    result = asyncio.run(
        prompt_runtime.EphemeralRuntime(
            execution_adapter=execution_adapter,
            service_registry=service_registry_factory("claude"),
        ).run_ephemeral(
            prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                worktree=Path("."),
                stage=stage_selection_factory(
                    service="claude",
                    model="gpt-5",
                    effort="medium",
                ),
                role=InvocationRole("implementer"),
                tool_access=runtime.ToolAccess.no_tools(),
            )
        )
    )

    assert result.output == "implementer:already rendered prompt"
    assert execution_adapter.prepare_session_calls == 0
    assert not hasattr(result.result, "continuation")


def test_ephemeral_runtime_preserves_fallback_selection_metadata_on_completed_outcome(
    stage_selection_factory: Callable[..., runtime.StageSelection],
    service_registry_factory: Callable[..., ServiceRegistry],
) -> None:
    result = asyncio.run(
        prompt_runtime.EphemeralRuntime(
            execution_adapter=_EphemeralExecutionAdapter(),
            service_registry=service_registry_factory("codex", "claude"),
        ).run_ephemeral(
            prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                worktree=Path("."),
                stage=stage_selection_factory(
                    service="missing",
                    fallback=stage_selection_factory(
                        service="claude",
                        model="sonnet",
                        effort="high",
                    ),
                ),
                role=InvocationRole("implementer"),
                tool_access=runtime.ToolAccess.no_tools(),
            )
        )
    )

    assert result.selected_service_path == ("missing", "claude")
    assert result.used_fallback is True


def test_ephemeral_runtime_applies_runtime_setup_failure_translation(
    stage_selection_factory: Callable[..., runtime.StageSelection],
    service_registry_factory: Callable[..., ServiceRegistry],
) -> None:
    with pytest.raises(AgentCredentialFailureError) as exc_info:
        asyncio.run(
            prompt_runtime.EphemeralRuntime(
                execution_adapter=_SetupTranslatedEphemeralExecutionAdapter(),
                service_registry=service_registry_factory("claude"),
            ).run_ephemeral(
                prompt_runtime.EphemeralRunRequest(
                    prompt="already rendered prompt",
                    worktree=Path("."),
                    stage=stage_selection_factory(
                        service="claude",
                        model="gpt-5",
                        effort="medium",
                    ),
                    role=InvocationRole("implementer"),
                    tool_access=runtime.ToolAccess.no_tools(),
                )
            )
        )

    assert str(exc_info.value) == "missing auth"
    assert exc_info.value.service_name == "claude"


def test_ephemeral_runtime_returns_usage_limited_outcome_for_usage_limit_conditions(
    stage_selection_factory: Callable[..., runtime.StageSelection],
    service_registry_factory: Callable[..., ServiceRegistry],
) -> None:
    result = asyncio.run(
        prompt_runtime.EphemeralRuntime(
            execution_adapter=_UsageLimitThenSuccessEphemeralExecutionAdapter(),
            service_registry=service_registry_factory("codex"),
        ).run_ephemeral(
            prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                worktree=Path("."),
                stage=stage_selection_factory(
                    service="codex",
                    model="gpt-5",
                    effort="medium",
                ),
                role=InvocationRole("implementer"),
                tool_access=runtime.ToolAccess.no_tools(),
            )
        )
    )

    assert result == prompt_runtime.RuntimeOutcome.usage_limited(
        output="",
        service_name="codex",
        reset_time=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
        usage_limit_scope=runtime.UsageLimitScope("implementer"),
        invocation_progress=prompt_runtime.InvocationProgress.STARTED,
    )
    assert result.result is None


def test_ephemeral_runtime_returns_no_service_available_outcome_for_temporarily_unavailable_services(
    stage_selection_factory: Callable[..., runtime.StageSelection],
    service_registry_factory: Callable[..., ServiceRegistry],
) -> None:
    result = asyncio.run(
        prompt_runtime.EphemeralRuntime(
            execution_adapter=_EphemeralExecutionAdapter(),
            service_registry=service_registry_factory(
                "codex",
                unavailable={"codex"},
            ),
        ).run_ephemeral(
            prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                worktree=Path("."),
                stage=stage_selection_factory(
                    service="codex",
                    model="gpt-5",
                    effort="medium",
                ),
                role=InvocationRole("implementer"),
                tool_access=runtime.ToolAccess.no_tools(),
            )
        )
    )

    assert result == prompt_runtime.RuntimeOutcome.no_service_available(
        output="",
        reset_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        usage_limit_scope=runtime.UsageLimitScope("implementer"),
        invocation_progress=prompt_runtime.InvocationProgress.NOT_STARTED,
    )
    assert result.result is None


def test_ephemeral_runtime_returns_cancelled_outcome_for_caller_cancellation(
    stage_selection_factory: Callable[..., runtime.StageSelection],
    service_registry_factory: Callable[..., ServiceRegistry],
) -> None:
    cancelled_token = CancellationToken()
    cancelled_token.cancel()

    result = asyncio.run(
        prompt_runtime.EphemeralRuntime(
            execution_adapter=_EphemeralExecutionAdapter(),
            service_registry=service_registry_factory("codex"),
        ).run_ephemeral(
            prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                worktree=Path("."),
                stage=stage_selection_factory(
                    service="codex",
                    model="gpt-5",
                    effort="medium",
                ),
                role=InvocationRole("implementer"),
                tool_access=runtime.ToolAccess.no_tools(),
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
    stage_selection_factory: Callable[..., runtime.StageSelection],
    service_registry_factory: Callable[..., ServiceRegistry],
) -> None:
    result = asyncio.run(
        prompt_runtime.EphemeralRuntime(
            execution_adapter=_TimeoutEphemeralExecutionAdapter(),
            service_registry=service_registry_factory("codex"),
        ).run_ephemeral(
            prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                worktree=Path("."),
                stage=stage_selection_factory(
                    service="codex",
                    model="gpt-5",
                    effort="medium",
                ),
                role=InvocationRole("implementer"),
                tool_access=runtime.ToolAccess.no_tools(),
            )
        )
    )

    assert result == prompt_runtime.RuntimeOutcome.timed_out(
        output="",
        invocation_progress=prompt_runtime.InvocationProgress.NOT_STARTED,
    )
    assert result.result is None


def test_ephemeral_runtime_returns_retryable_provider_failure_outcome_for_retryable_provider_failures(
    stage_selection_factory: Callable[..., runtime.StageSelection],
    service_registry_factory: Callable[..., ServiceRegistry],
) -> None:
    result = asyncio.run(
        prompt_runtime.EphemeralRuntime(
            execution_adapter=_RetryableProviderFailureEphemeralExecutionAdapter(),
            service_registry=service_registry_factory("codex"),
        ).run_ephemeral(
            prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                worktree=Path("."),
                stage=stage_selection_factory(
                    service="codex",
                    model="gpt-5",
                    effort="medium",
                ),
                role=InvocationRole("implementer"),
                tool_access=runtime.ToolAccess.no_tools(),
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
                usage_limit_scope=runtime.UsageLimitScope("implementer"),
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
    stage_selection_factory: Callable[..., runtime.StageSelection],
    service_registry_factory: Callable[..., ServiceRegistry],
    error: Exception,
    expected_outcome: prompt_runtime.RuntimeOutcome,
) -> None:
    result = asyncio.run(
        prompt_runtime.EphemeralRuntime(
            execution_adapter=_InterruptedEphemeralExecutionAdapter(error),
            service_registry=service_registry_factory("codex"),
        ).run_ephemeral(
            prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                worktree=Path("."),
                stage=stage_selection_factory(
                    service="codex",
                    model="gpt-5",
                    effort="medium",
                ),
                role=InvocationRole("implementer"),
                tool_access=runtime.ToolAccess.no_tools(),
            )
        )
    )

    assert result == expected_outcome


def test_ephemeral_runtime_keeps_exceptional_failures_exceptional(
    stage_selection_factory: Callable[..., runtime.StageSelection],
    service_registry_factory: Callable[..., ServiceRegistry],
) -> None:
    with pytest.raises(runtime.RuntimeConfigurationError):
        asyncio.run(
            prompt_runtime.EphemeralRuntime(
                execution_adapter=_EphemeralExecutionAdapter(),
                service_registry=service_registry_factory("codex"),
            ).run_ephemeral(
                prompt_runtime.EphemeralRunRequest(
                    prompt="already rendered prompt",
                    worktree=Path("."),
                    stage=stage_selection_factory(
                        service="missing",
                        fallback=stage_selection_factory(
                            service="also-missing",
                            model="sonnet",
                            effort="high",
                        ),
                    ),
                    role=InvocationRole("implementer"),
                    tool_access=runtime.ToolAccess.no_tools(),
                )
            )
        )

    with pytest.raises(AgentCredentialFailureError):
        asyncio.run(
            prompt_runtime.EphemeralRuntime(
                execution_adapter=_SetupTranslatedEphemeralExecutionAdapter(),
                service_registry=service_registry_factory("codex"),
            ).run_ephemeral(
                prompt_runtime.EphemeralRunRequest(
                    prompt="already rendered prompt",
                    worktree=Path("."),
                    stage=stage_selection_factory(
                        service="codex",
                        model="gpt-5",
                        effort="medium",
                    ),
                    role=InvocationRole("implementer"),
                    tool_access=runtime.ToolAccess.no_tools(),
                )
            )
        )

    with pytest.raises(HardAgentError):
        asyncio.run(
            prompt_runtime.EphemeralRuntime(
                execution_adapter=_HardFailureEphemeralExecutionAdapter(),
                service_registry=service_registry_factory("codex"),
            ).run_ephemeral(
                prompt_runtime.EphemeralRunRequest(
                    prompt="already rendered prompt",
                    worktree=Path("."),
                    stage=stage_selection_factory(
                        service="codex",
                        model="gpt-5",
                        effort="medium",
                    ),
                    role=InvocationRole("implementer"),
                    tool_access=runtime.ToolAccess.no_tools(),
                )
            )
        )

    with pytest.raises(TransientAgentError):
        asyncio.run(
            prompt_runtime.EphemeralRuntime(
                execution_adapter=_TransientProviderFailureEphemeralExecutionAdapter(),
                service_registry=service_registry_factory("codex"),
            ).run_ephemeral(
                prompt_runtime.EphemeralRunRequest(
                    prompt="already rendered prompt",
                    worktree=Path("."),
                    stage=stage_selection_factory(
                        service="codex",
                        model="gpt-5",
                        effort="medium",
                    ),
                    role=InvocationRole("implementer"),
                    tool_access=runtime.ToolAccess.no_tools(),
                )
            )
        )


@pytest.mark.parametrize("tool_policy", _TOOL_POLICY_CASES)
def test_text_output_adapter_exposes_tool_policy_effects_through_public_adapter_seam(
    tool_policy: runtime.ToolPolicy | runtime.ToolPolicyProfile,
) -> None:
    output = asyncio.run(
        TextOutputAdapter(
            prompt="already rendered prompt",
            tool_policy=tool_policy,
        ).invoke(
            runner=cast(WorkExecutionAdapter, _ToolPolicyRenderingPromptRunner()),
            role=InvocationRole("implementer"),
            prompt="already rendered prompt",
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
            tool_access=runtime.ToolAccess.no_tools(),
        ).invoke(
            runner=cast(WorkExecutionAdapter, _ToolPolicyRenderingPromptRunner()),
            role=InvocationRole("implementer"),
            prompt="already rendered prompt",
            run_kind=RunKind.FRESH,
            session_uuid=None,
            on_provider_session_id=lambda _provider_session_id: None,
        )
    )

    assert output == "allowed=none;disallowed=all"


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
            tool_access=runtime.ToolAccess.workspace_backed(Path("/repo")),
        )
