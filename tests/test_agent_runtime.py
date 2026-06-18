from __future__ import annotations

import asyncio
import json
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import pytest

import agent_runtime as runtime
import agent_runtime.provider_session_adapter as provider_session_adapter_runtime
import agent_runtime.runtime as prompt_runtime
import agent_runtime.session as session_runtime
from agent_runtime.agent_log import AgentInvocationLog
from agent_runtime._import_isolation import assert_runtime_import_isolation
from agent_runtime.contracts import (
    AssistantTurn,
    CredentialFailure,
    ExecutionProvider,
    HardError,
    ModelActivity,
    PromptTokens,
    Result,
    ServiceSelectionProvider,
    TransientError,
    UnsupportedTokens,
    UsageLimit,
)
from agent_runtime.provider_session_adapter import ProviderSessionPlanningRequest
from agent_runtime.errors import (
    AgentCancelledError,
    AgentCredentialFailureError,
    AgentFailedError,
    AgentRuntimeError,
    AgentTimeoutError,
    HardAgentError,
    RetryableProviderFailureError,
    TransientAgentError,
    UsageLimitError,
)
from agent_runtime.execution_contracts import (
    PreparedRunSessionState,
    PromptRunRequest,
    PromptRunSession,
    TextOutputAdapter,
    WorkExecutionAdapter,
    WorkExecutionDependencies,
    WorkFailureHandling,
    WorkInvocationDependencies,
    WorkPresentationDependencies,
    WorktreeMount,
)
from agent_runtime.provider_errors import ProviderErrorObservation
from agent_runtime.provider_output import reduce_text_output_events
from agent_runtime.roles import InvocationRole
from agent_runtime.service_registry import ServiceRegistry
from agent_runtime.session import (
    ProviderSessionSelection,
    RunKind,
    is_exact_resumable_service_session,
    normalize_state_dir_relpath,
    provider_state_relpath,
    provider_state_session_id_path,
    select_resumable_provider_session_id,
)
from agent_runtime.stage_priority_chain import (
    chain_entries,
    render_chain_label,
    select_configured_candidate_chain,
)
from agent_runtime.usage_limit_decision import (
    SleepUntil,
    Stop,
    UsageLimitOutcome,
    decide_usage_limit_continuation,
)
from agent_runtime.session_planning import ResumableSessionPlan
from agent_runtime.session_planning import (
    AuthSeedingRequirement,
    ResumableSessionPlanRequest,
)
from tests.runtime_boundary_fakes import (
    ExecutionServiceFake as _ExecutionService,
    ResidentPlanningProviderSessionAdapterFake as _ResidentPlanningProviderSessionAdapter,
    SelectionServiceFake as _Service,
    SessionStoreFake as _SessionStore,
)


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

    def __enter__(self) -> _Session:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb

    def exec_simple(self, cmd: str) -> str:
        del cmd
        return ""


@pytest.fixture
def provider_error_observation() -> ProviderErrorObservation:
    return ProviderErrorObservation(
        service_name="codex",
        raw_provider_text="bad credential",
        source_stream="stderr",
    )


class _EphemeralCompatWorkRunner:
    def __init__(
        self,
        service: _ExecutionService,
        *,
        attempts_by_service: dict[str, int],
    ) -> None:
        self._service = service
        self._attempts_by_service = attempts_by_service

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
        assert role == InvocationRole("implementer")
        assert run_kind is RunKind.FRESH
        assert session_uuid is None

        service_name = self._service.name
        attempt_count = self._attempts_by_service.get(service_name, 0) + 1
        self._attempts_by_service[service_name] = attempt_count

        if service_name == "codex":
            if attempt_count > 1:
                raise AssertionError("ephemeral retried the exhausted primary service")
            raise UsageLimitError(
                reset_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
                service_name=service_name,
            )

        assert callable(on_provider_session_id)
        on_provider_session_id(f"provider-{service_name}")
        return f"{service_name}:{prompt}"

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

    async def prompt_only(
        self,
        prompt: str,
        *,
        role: InvocationRole = InvocationRole("implementer"),
        run_kind: RunKind = RunKind.FRESH,
        session_uuid: str | None = None,
        on_provider_session_id: Any = None,
    ) -> str:
        return await self.work(
            role,
            prompt,
            run_kind=run_kind,
            session_uuid=session_uuid,
            on_provider_session_id=on_provider_session_id,
        )


class _EphemeralCompatExecutionAdapter:
    def __init__(self) -> None:
        self._attempts_by_service: dict[str, int] = {}

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
        del name, model, effort
        execution_service = cast(_ExecutionService, service)
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
                    _EphemeralCompatWorkRunner(
                        execution_service,
                        attempts_by_service=self._attempts_by_service,
                    ),
                ),
                get_git_identity=lambda: ("Runtime Test", "runtime@example.com"),
            ),
            failure_handling=WorkFailureHandling(timeout_retries=0),
            presentation=WorkPresentationDependencies(),
        )


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


class _ToolPolicyRenderingEphemeralExecutionAdapter:
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
                    _ToolPolicyRenderingPromptRunner(),
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

    async def prompt_only(
        self,
        prompt: str,
        *,
        role: InvocationRole = InvocationRole("implementer"),
        run_kind: RunKind = RunKind.FRESH,
        session_uuid: str | None = None,
        on_provider_session_id: Any = None,
    ) -> str:
        return await self.work(
            role,
            prompt,
            run_kind=run_kind,
            session_uuid=session_uuid,
            on_provider_session_id=on_provider_session_id,
        )


class _RoleAwareEphemeralCompatExecutionAdapter:
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
                prepare_session=lambda run_session: cast(
                    PreparedRunSessionState, _PreparedRunSession()
                ),
                build_session=lambda mount_path, service, provider_state_dir: _Session(
                    provider_state_dir
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


class _UsageLimitWithoutMappingRunner(_RoleAwareEphemeralCompatWorkRunner):
    async def work(
        self,
        role: InvocationRole,
        prompt: str,
        *,
        run_kind: RunKind = RunKind.FRESH,
        session_uuid: str | None = None,
        on_provider_session_id: Any = None,
    ) -> str:
        del role, prompt, run_kind, session_uuid, on_provider_session_id
        raise UsageLimitError(reset_time=None, service_name="codex")


class _UsageLimitWithoutMappingExecutionAdapter(
    _RoleAwareEphemeralCompatExecutionAdapter
):
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
                prepare_session=lambda run_session: cast(
                    PreparedRunSessionState, _PreparedRunSession()
                ),
                build_session=lambda mount_path, service, provider_state_dir: _Session(
                    provider_state_dir
                ),
                build_runner=lambda session, status_display: cast(
                    WorkExecutionAdapter,
                    _UsageLimitWithoutMappingRunner(),
                ),
                get_git_identity=lambda: ("Runtime Test", "runtime@example.com"),
            ),
            failure_handling=WorkFailureHandling(timeout_retries=0),
            presentation=WorkPresentationDependencies(),
        )


class _StartedUsageLimitEphemeralCompatRunner(_RoleAwareEphemeralCompatWorkRunner):
    async def prompt_only(
        self,
        prompt: str,
        *,
        role: InvocationRole = InvocationRole("implementer"),
        run_kind: RunKind = RunKind.FRESH,
        session_uuid: str | None = None,
        on_provider_session_id: Any = None,
    ) -> str:
        del prompt, role, run_kind, session_uuid, on_provider_session_id
        raise UsageLimitError(
            reset_time=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
            service_name="codex",
            invocation_progress=runtime.InvocationProgress.STARTED,
        )


class _ModelActivityUsageLimitEphemeralCompatRunner(
    _RoleAwareEphemeralCompatWorkRunner
):
    async def prompt_only(
        self,
        prompt: str,
        *,
        role: InvocationRole = InvocationRole("implementer"),
        run_kind: RunKind = RunKind.FRESH,
        session_uuid: str | None = None,
        on_provider_session_id: Any = None,
    ) -> str:
        del prompt, role, run_kind, session_uuid, on_provider_session_id
        return reduce_text_output_events(
            [
                ModelActivity(),
                UsageLimit(
                    reset_time=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
                ),
            ],
            lambda _turn: None,
            provider="codex",
        )


class _ModelActivityUsageLimitEphemeralCompatExecutionAdapter(
    _RoleAwareEphemeralCompatExecutionAdapter
):
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
                prepare_session=lambda run_session: cast(
                    PreparedRunSessionState, _PreparedRunSession()
                ),
                build_session=lambda mount_path, service, provider_state_dir: _Session(
                    provider_state_dir
                ),
                build_runner=lambda session, status_display: cast(
                    WorkExecutionAdapter,
                    _ModelActivityUsageLimitEphemeralCompatRunner(),
                ),
                get_git_identity=lambda: ("Runtime Test", "runtime@example.com"),
            ),
            failure_handling=WorkFailureHandling(timeout_retries=0),
            presentation=WorkPresentationDependencies(),
        )


class _StartedUsageLimitEphemeralCompatExecutionAdapter(
    _RoleAwareEphemeralCompatExecutionAdapter
):
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
                prepare_session=lambda run_session: cast(
                    PreparedRunSessionState, _PreparedRunSession()
                ),
                build_session=lambda mount_path, service, provider_state_dir: _Session(
                    provider_state_dir
                ),
                build_runner=lambda session, status_display: cast(
                    WorkExecutionAdapter,
                    _StartedUsageLimitEphemeralCompatRunner(),
                ),
                get_git_identity=lambda: ("Runtime Test", "runtime@example.com"),
            ),
            failure_handling=WorkFailureHandling(timeout_retries=0),
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


class _RetryableProviderFailureEphemeralCompatRunner(
    _RoleAwareEphemeralCompatWorkRunner
):
    async def prompt_only(
        self,
        prompt: str,
        *,
        role: InvocationRole = InvocationRole("implementer"),
        run_kind: RunKind = RunKind.FRESH,
        session_uuid: str | None = None,
        on_provider_session_id: Any = None,
    ) -> str:
        del prompt, role, run_kind, session_uuid, on_provider_session_id
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


class _RetryableProviderFailureEphemeralCompatExecutionAdapter(
    _RoleAwareEphemeralCompatExecutionAdapter
):
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
                prepare_session=lambda run_session: cast(
                    PreparedRunSessionState, _PreparedRunSession()
                ),
                build_session=lambda mount_path, service, provider_state_dir: _Session(
                    provider_state_dir
                ),
                build_runner=lambda session, status_display: cast(
                    WorkExecutionAdapter,
                    _RetryableProviderFailureEphemeralCompatRunner(),
                ),
                get_git_identity=lambda: ("Runtime Test", "runtime@example.com"),
            ),
            failure_handling=WorkFailureHandling(timeout_retries=0),
            presentation=WorkPresentationDependencies(),
        )


class _StartedRetryableProviderFailureEphemeralCompatRunner(
    _RoleAwareEphemeralCompatWorkRunner
):
    async def prompt_only(
        self,
        prompt: str,
        *,
        role: InvocationRole = InvocationRole("implementer"),
        run_kind: RunKind = RunKind.FRESH,
        session_uuid: str | None = None,
        on_provider_session_id: Any = None,
    ) -> str:
        del prompt, role, run_kind, session_uuid, on_provider_session_id
        return reduce_text_output_events(
            [
                AssistantTurn("hello"),
                TransientError(
                    status_code=503,
                    raw_message="retry later",
                    classification="retryable",
                ),
            ],
            lambda _turn: None,
            provider="codex",
        )


class _StartedRetryableProviderFailureEphemeralCompatExecutionAdapter(
    _RoleAwareEphemeralCompatExecutionAdapter
):
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
                prepare_session=lambda run_session: cast(
                    PreparedRunSessionState, _PreparedRunSession()
                ),
                build_session=lambda mount_path, service, provider_state_dir: _Session(
                    provider_state_dir
                ),
                build_runner=lambda session, status_display: cast(
                    WorkExecutionAdapter,
                    _StartedRetryableProviderFailureEphemeralCompatRunner(),
                ),
                get_git_identity=lambda: ("Runtime Test", "runtime@example.com"),
            ),
            failure_handling=WorkFailureHandling(timeout_retries=0),
            presentation=WorkPresentationDependencies(),
        )


class _TransientProviderFailureEphemeralCompatRunner(
    _RoleAwareEphemeralCompatWorkRunner
):
    async def prompt_only(
        self,
        prompt: str,
        *,
        role: InvocationRole = InvocationRole("implementer"),
        run_kind: RunKind = RunKind.FRESH,
        session_uuid: str | None = None,
        on_provider_session_id: Any = None,
    ) -> str:
        del prompt, role, run_kind, session_uuid, on_provider_session_id
        return reduce_text_output_events(
            [TransientError(status_code=503, raw_message="retry later")],
            lambda _turn: None,
            provider="codex",
        )


class _TransientProviderFailureEphemeralCompatExecutionAdapter(
    _RoleAwareEphemeralCompatExecutionAdapter
):
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
                prepare_session=lambda run_session: cast(
                    PreparedRunSessionState, _PreparedRunSession()
                ),
                build_session=lambda mount_path, service, provider_state_dir: _Session(
                    provider_state_dir
                ),
                build_runner=lambda session, status_display: cast(
                    WorkExecutionAdapter,
                    _TransientProviderFailureEphemeralCompatRunner(),
                ),
                get_git_identity=lambda: ("Runtime Test", "runtime@example.com"),
            ),
            failure_handling=WorkFailureHandling(timeout_retries=0),
            presentation=WorkPresentationDependencies(),
        )


class _StartedCancellationEphemeralCompatRunner(_RoleAwareEphemeralCompatWorkRunner):
    async def prompt_only(
        self,
        prompt: str,
        *,
        role: InvocationRole = InvocationRole("implementer"),
        run_kind: RunKind = RunKind.FRESH,
        session_uuid: str | None = None,
        on_provider_session_id: Any = None,
    ) -> str:
        del prompt, role, run_kind, session_uuid, on_provider_session_id
        raise AgentCancelledError(
            invocation_progress=runtime.InvocationProgress.STARTED,
        )


class _StartedCancellationEphemeralCompatExecutionAdapter(
    _RoleAwareEphemeralCompatExecutionAdapter
):
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
                prepare_session=lambda run_session: cast(
                    PreparedRunSessionState, _PreparedRunSession()
                ),
                build_session=lambda mount_path, service, provider_state_dir: _Session(
                    provider_state_dir
                ),
                build_runner=lambda session, status_display: cast(
                    WorkExecutionAdapter,
                    _StartedCancellationEphemeralCompatRunner(),
                ),
                get_git_identity=lambda: ("Runtime Test", "runtime@example.com"),
            ),
            failure_handling=WorkFailureHandling(timeout_retries=0),
            presentation=WorkPresentationDependencies(),
        )


class _RecordingStatusDisplay:
    def __init__(self) -> None:
        self.removals: list[tuple[str, str, str]] = []

    def register(
        self,
        caller: str,
        kind: str,
        startup_message: str = "started",
        work_body: str = "",
        initial_phase: str = "Setup",
        color_key: int | None = None,
        model_display: Any = None,
    ) -> None:
        del (
            caller,
            kind,
            startup_message,
            work_body,
            initial_phase,
            color_key,
            model_display,
        )

    def update_phase(self, name: str, phase: str) -> None:
        del name, phase

    def reset_idle_timer(self, name: str) -> None:
        del name

    def update_tokens(self, name: str, current_tokens: int) -> None:
        del name, current_tokens

    def remove(
        self,
        caller: str,
        shutdown_message: str = "finished",
        shutdown_style: str = "success",
    ) -> None:
        self.removals.append((caller, shutdown_message, shutdown_style))

    def print(self, caller: str, message: object, style: str | None = None) -> None:
        del caller, message, style


class _StartedCancellationStatusEphemeralCompatExecutionAdapter(
    _RoleAwareEphemeralCompatExecutionAdapter
):
    def __init__(self, status_display: _RecordingStatusDisplay) -> None:
        self._status_display = status_display

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
                prepare_session=lambda run_session: cast(
                    PreparedRunSessionState, _PreparedRunSession()
                ),
                build_session=lambda mount_path, service, provider_state_dir: _Session(
                    provider_state_dir
                ),
                build_runner=lambda session, status_display: cast(
                    WorkExecutionAdapter,
                    _StartedCancellationEphemeralCompatRunner(),
                ),
                get_git_identity=lambda: ("Runtime Test", "runtime@example.com"),
            ),
            failure_handling=WorkFailureHandling(timeout_retries=0),
            presentation=WorkPresentationDependencies(
                status_display_factory=lambda: self._status_display
            ),
        )


class _TimeoutEphemeralCompatRunner(_RoleAwareEphemeralCompatWorkRunner):
    async def prompt_only(
        self,
        prompt: str,
        *,
        role: InvocationRole = InvocationRole("implementer"),
        run_kind: RunKind = RunKind.FRESH,
        session_uuid: str | None = None,
        on_provider_session_id: Any = None,
    ) -> str:
        del prompt, role, run_kind, session_uuid, on_provider_session_id
        raise AgentTimeoutError("timed out")


class _TimeoutEphemeralCompatExecutionAdapter(
    _RoleAwareEphemeralCompatExecutionAdapter
):
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
                prepare_session=lambda run_session: cast(
                    PreparedRunSessionState, _PreparedRunSession()
                ),
                build_session=lambda mount_path, service, provider_state_dir: _Session(
                    provider_state_dir
                ),
                build_runner=lambda session, status_display: cast(
                    WorkExecutionAdapter,
                    _TimeoutEphemeralCompatRunner(),
                ),
                get_git_identity=lambda: ("Runtime Test", "runtime@example.com"),
            ),
            failure_handling=WorkFailureHandling(timeout_retries=0),
            presentation=WorkPresentationDependencies(),
        )


class _StartedTimeoutEphemeralCompatRunner(_RoleAwareEphemeralCompatWorkRunner):
    async def prompt_only(
        self,
        prompt: str,
        *,
        role: InvocationRole = InvocationRole("implementer"),
        run_kind: RunKind = RunKind.FRESH,
        session_uuid: str | None = None,
        on_provider_session_id: Any = None,
    ) -> str:
        del prompt, role, run_kind, session_uuid, on_provider_session_id
        raise AgentTimeoutError(
            "timed out",
            invocation_progress=runtime.InvocationProgress.STARTED,
        )


class _StartedTimeoutEphemeralCompatExecutionAdapter(
    _RoleAwareEphemeralCompatExecutionAdapter
):
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
                prepare_session=lambda run_session: cast(
                    PreparedRunSessionState, _PreparedRunSession()
                ),
                build_session=lambda mount_path, service, provider_state_dir: _Session(
                    provider_state_dir
                ),
                build_runner=lambda session, status_display: cast(
                    WorkExecutionAdapter,
                    _StartedTimeoutEphemeralCompatRunner(),
                ),
                get_git_identity=lambda: ("Runtime Test", "runtime@example.com"),
            ),
            failure_handling=WorkFailureHandling(timeout_retries=0),
            presentation=WorkPresentationDependencies(),
        )


class _PromptOnlyEphemeralCompatWorkRunner:
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
        del role, prompt, run_kind, session_uuid, on_provider_session_id
        raise AssertionError("ephemeral used tool-capable work invocation")

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
        raise AssertionError("ephemeral used tool-capable work_text invocation")

    async def prompt_only(
        self,
        prompt: str,
        *,
        role: InvocationRole,
        run_kind: RunKind = RunKind.FRESH,
        session_uuid: str | None = None,
        on_provider_session_id: Any = None,
    ) -> str:
        assert callable(on_provider_session_id)
        assert run_kind is RunKind.FRESH
        assert session_uuid is None
        on_provider_session_id("provider-prompt-only")
        return f"{role.value}:{prompt}:prompt_only"


class _PromptOnlyEphemeralCompatExecutionAdapter:
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
                    _PromptOnlyEphemeralCompatWorkRunner(),
                ),
                get_git_identity=lambda: ("Runtime Test", "runtime@example.com"),
            ),
            failure_handling=WorkFailureHandling(timeout_retries=0),
            presentation=WorkPresentationDependencies(),
        )


class _NormalizedPromptOnlyEphemeralCompatWorkRunner:
    async def setup(self, git_name: str, git_email: str, work_body: str = "") -> None:
        del git_name, git_email, work_body

    async def prompt_only(
        self,
        prompt: str,
        *,
        role: InvocationRole,
        run_kind: RunKind = RunKind.FRESH,
        session_uuid: str | None = None,
        on_provider_session_id: Any = None,
    ) -> str:
        assert callable(on_provider_session_id)
        assert role == InvocationRole("implementer")
        assert run_kind is RunKind.FRESH
        assert session_uuid is None
        on_provider_session_id("provider-normalized")
        return f"normalized:{prompt}"


class _NormalizedPromptOnlyEphemeralCompatExecutionAdapter:
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
                    _NormalizedPromptOnlyEphemeralCompatWorkRunner(),
                ),
                get_git_identity=lambda: ("Runtime Test", "runtime@example.com"),
            ),
            failure_handling=WorkFailureHandling(timeout_retries=0),
            presentation=WorkPresentationDependencies(),
        )


class _ToolCapableOnlyEphemeralCompatWorkRunner:
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
    ) -> dict[str, str]:
        del role, prompt, run_kind, session_uuid, on_provider_session_id
        raise AssertionError("ephemeral fell back to tool-capable work invocation")

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
        raise AssertionError("ephemeral fell back to tool-capable work_text invocation")


class _MissingPromptOnlyEphemeralCompatExecutionAdapter:
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
                    _ToolCapableOnlyEphemeralCompatWorkRunner(),
                ),
                get_git_identity=lambda: ("Runtime Test", "runtime@example.com"),
            ),
            failure_handling=WorkFailureHandling(timeout_retries=0),
            presentation=WorkPresentationDependencies(),
        )


@dataclass
class _ResidentAdapterPreparedRunSession:
    provider_state_dir_container_path: str | None
    run_kind: RunKind
    provider_session_id: str | None

    def prepare_for_run(self) -> None:
        return None

    def initial_provider_run_session(self) -> _ProviderRunSession:
        return _ProviderRunSession(
            run_kind=self.run_kind,
            provider_session_id=self.provider_session_id,
        )

    def resumable_provider_run_session(self) -> _ProviderRunSession:
        return self.initial_provider_run_session()

    def protocol_reprompt_provider_run_session(self) -> None:
        return None


class _ResidentSeamRunner:
    def __init__(self, session: _Session) -> None:
        self._session = session

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
        del role, prompt, on_provider_session_id
        return f"{run_kind.value}:{session_uuid}:{self._session.provider_state_dir}"

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
        return await self.work(
            role,
            prompt,
            run_kind=run_kind,
            session_uuid=session_uuid,
            on_provider_session_id=on_provider_session_id,
        )


class _ResidentSeamExecutionAdapter:
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

        def _prepare_session(run_session: Any) -> _ResidentAdapterPreparedRunSession:
            return _ResidentAdapterPreparedRunSession(
                provider_state_dir_container_path="/workspace/runtime-state/",
                run_kind=run_session.run_kind,
                provider_session_id=f"prepared:{run_session.provider_session_id}",
            )

        return WorkInvocationDependencies(
            execution=WorkExecutionDependencies(
                container_workspace="/workspace",
                prepare_session=cast(Any, _prepare_session),
                build_session=lambda mount_path, service, provider_state_dir: _Session(
                    provider_state_dir
                ),
                build_runner=lambda session, status_display: cast(
                    WorkExecutionAdapter, _ResidentSeamRunner(cast(_Session, session))
                ),
                get_git_identity=lambda: ("Runtime Test", "runtime@example.com"),
            ),
            failure_handling=WorkFailureHandling(timeout_retries=0),
            presentation=WorkPresentationDependencies(),
        )


class _RuntimePlannedPathResidentExecutionAdapter:
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

        def _prepare_session(run_session: Any) -> _ResidentAdapterPreparedRunSession:
            return _ResidentAdapterPreparedRunSession(
                provider_state_dir_container_path=(
                    run_session.provider_state_dir_container_path
                ),
                run_kind=run_session.run_kind,
                provider_session_id=f"prepared:{run_session.provider_session_id}",
            )

        return WorkInvocationDependencies(
            execution=WorkExecutionDependencies(
                container_workspace="/workspace",
                prepare_session=cast(Any, _prepare_session),
                build_session=lambda mount_path, service, provider_state_dir: _Session(
                    provider_state_dir
                ),
                build_runner=lambda session, status_display: cast(
                    WorkExecutionAdapter, _ResidentSeamRunner(cast(_Session, session))
                ),
                get_git_identity=lambda: ("Runtime Test", "runtime@example.com"),
            ),
            failure_handling=WorkFailureHandling(timeout_retries=0),
            presentation=WorkPresentationDependencies(),
        )


class _ContinuationBoundServiceResidentExecutionAdapter(
    _RuntimePlannedPathResidentExecutionAdapter
):
    def resolve_service(self, service_name: str = "") -> ExecutionProvider:
        if service_name != "bound-service":
            raise AssertionError(f"expected continuation service, got {service_name!r}")
        return _ExecutionService("resolved-service")


class _NamedExternalStateResidentPlanningProviderSessionAdapter:
    def __init__(self, service_name: str) -> None:
        self._service_name = service_name
        self._state_dir_relpath = f"{service_name}-runtime-state/"
        self._provider_state_dir = Path(f"/host/{service_name}-runtime-state")

    @property
    def service_name(self) -> str:
        return self._service_name

    def provider_session_planning_facts(
        self,
        request: ProviderSessionPlanningRequest,
    ) -> Any:
        del request
        return provider_session_adapter_runtime.ProviderSessionPlanningFacts(
            state_dir_relpath=self._state_dir_relpath,
            provider_state_dir=self._provider_state_dir,
            has_resumable_provider_state=True,
        )

    def provider_session_state(self, request: Any) -> Any:
        del request
        return session_runtime.ProviderSessionState(
            run_kind=RunKind.RESUME,
            provider_session_id=f"recovered-{self._service_name}",
            state_dir_relpath=self._state_dir_relpath,
            state_dir_path=self._provider_state_dir,
        )

    def prepare_local_provider_run_state(
        self,
        provider_state_dir: Path | None,
        auth_seed_action: Any | None = None,
    ) -> None:
        del provider_state_dir, auth_seed_action

    def record_provider_session_id(
        self,
        *,
        session_store: Any,
        provider_session_id: str,
        service_state_dir: Path | None = None,
    ) -> None:
        del service_state_dir
        session_store.save_service_session_id(self._service_name, provider_session_id)


class _UsageLimitedThenFallbackNewSessionRunner(_ResidentSeamRunner):
    def __init__(
        self,
        session: _Session,
        *,
        service_name: str,
        attempts_by_service: dict[str, int],
    ) -> None:
        super().__init__(session)
        self._service_name = service_name
        self._attempts_by_service = attempts_by_service

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
        attempt_count = self._attempts_by_service.get(self._service_name, 0) + 1
        self._attempts_by_service[self._service_name] = attempt_count
        if self._service_name == "codex":
            if attempt_count > 1:
                raise AssertionError(
                    "new-session runtime retried the exhausted primary service"
                )
            raise UsageLimitError(
                reset_time=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
                service_name=self._service_name,
            )
        return await super().work_text(
            prompt,
            role=role,
            tool_policy=tool_policy,
            run_kind=run_kind,
            session_uuid=session_uuid,
            on_provider_session_id=on_provider_session_id,
        )


class _UsageLimitedThenFallbackNewSessionExecutionAdapter:
    def __init__(self) -> None:
        self._attempts_by_service: dict[str, int] = {}

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
        del name, model, effort
        execution_service = cast(_ExecutionService, service)

        def _prepare_session(run_session: Any) -> _ResidentAdapterPreparedRunSession:
            return _ResidentAdapterPreparedRunSession(
                provider_state_dir_container_path=(
                    run_session.provider_state_dir_container_path
                ),
                run_kind=run_session.run_kind,
                provider_session_id=f"prepared:{run_session.provider_session_id}",
            )

        return WorkInvocationDependencies(
            execution=WorkExecutionDependencies(
                container_workspace="/workspace",
                prepare_session=cast(Any, _prepare_session),
                build_session=lambda mount_path, service, provider_state_dir: _Session(
                    provider_state_dir
                ),
                build_runner=lambda session, status_display: cast(
                    WorkExecutionAdapter,
                    _UsageLimitedThenFallbackNewSessionRunner(
                        cast(_Session, session),
                        service_name=execution_service.name,
                        attempts_by_service=self._attempts_by_service,
                    ),
                ),
                get_git_identity=lambda: ("Runtime Test", "runtime@example.com"),
            ),
            failure_handling=WorkFailureHandling(timeout_retries=0),
            presentation=WorkPresentationDependencies(),
        )


class _StartedUsageLimitNewSessionRunner(_ResidentSeamRunner):
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
        raise UsageLimitError(
            reset_time=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
            service_name="codex",
            invocation_progress=runtime.InvocationProgress.STARTED,
        )


class _StartedUsageLimitNewSessionExecutionAdapter:
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

        def _prepare_session(run_session: Any) -> _ResidentAdapterPreparedRunSession:
            return _ResidentAdapterPreparedRunSession(
                provider_state_dir_container_path=(
                    run_session.provider_state_dir_container_path
                ),
                run_kind=run_session.run_kind,
                provider_session_id=f"prepared:{run_session.provider_session_id}",
            )

        return WorkInvocationDependencies(
            execution=WorkExecutionDependencies(
                container_workspace="/workspace",
                prepare_session=cast(Any, _prepare_session),
                build_session=lambda mount_path, service, provider_state_dir: _Session(
                    provider_state_dir
                ),
                build_runner=lambda session, status_display: cast(
                    WorkExecutionAdapter,
                    _StartedUsageLimitNewSessionRunner(cast(_Session, session)),
                ),
                get_git_identity=lambda: ("Runtime Test", "runtime@example.com"),
            ),
            failure_handling=WorkFailureHandling(timeout_retries=0),
            presentation=WorkPresentationDependencies(),
        )


class _NotStartedUsageLimitNewSessionRunner(_ResidentSeamRunner):
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
        raise UsageLimitError(
            reset_time=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
            service_name="codex",
            invocation_progress=runtime.InvocationProgress.NOT_STARTED,
        )


class _PreparedNotStartedUsageLimitNewSessionExecutionAdapter:
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

        def _prepare_session(run_session: Any) -> _ResidentAdapterPreparedRunSession:
            self.prepare_session_calls += 1
            return _ResidentAdapterPreparedRunSession(
                provider_state_dir_container_path=(
                    run_session.provider_state_dir_container_path
                ),
                run_kind=run_session.run_kind,
                provider_session_id=f"prepared:{run_session.provider_session_id}",
            )

        return WorkInvocationDependencies(
            execution=WorkExecutionDependencies(
                container_workspace="/workspace",
                prepare_session=cast(Any, _prepare_session),
                build_session=lambda mount_path, service, provider_state_dir: _Session(
                    provider_state_dir
                ),
                build_runner=lambda session, status_display: cast(
                    WorkExecutionAdapter,
                    _NotStartedUsageLimitNewSessionRunner(cast(_Session, session)),
                ),
                get_git_identity=lambda: ("Runtime Test", "runtime@example.com"),
            ),
            failure_handling=WorkFailureHandling(timeout_retries=0),
            presentation=WorkPresentationDependencies(),
        )


class _StartedCancellationNewSessionRunner(_ResidentSeamRunner):
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
        raise AgentCancelledError(
            invocation_progress=runtime.InvocationProgress.STARTED,
        )


class _StartedCancellationNewSessionExecutionAdapter:
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

        def _prepare_session(run_session: Any) -> _ResidentAdapterPreparedRunSession:
            return _ResidentAdapterPreparedRunSession(
                provider_state_dir_container_path=(
                    run_session.provider_state_dir_container_path
                ),
                run_kind=run_session.run_kind,
                provider_session_id=f"prepared:{run_session.provider_session_id}",
            )

        return WorkInvocationDependencies(
            execution=WorkExecutionDependencies(
                container_workspace="/workspace",
                prepare_session=cast(Any, _prepare_session),
                build_session=lambda mount_path, service, provider_state_dir: _Session(
                    provider_state_dir
                ),
                build_runner=lambda session, status_display: cast(
                    WorkExecutionAdapter,
                    _StartedCancellationNewSessionRunner(cast(_Session, session)),
                ),
                get_git_identity=lambda: ("Runtime Test", "runtime@example.com"),
            ),
            failure_handling=WorkFailureHandling(timeout_retries=0),
            presentation=WorkPresentationDependencies(),
        )


class _NotStartedCancellationNewSessionRunner(_ResidentSeamRunner):
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
        raise AgentCancelledError(
            invocation_progress=runtime.InvocationProgress.NOT_STARTED,
        )


class _PreparedNotStartedCancellationNewSessionExecutionAdapter:
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

        def _prepare_session(run_session: Any) -> _ResidentAdapterPreparedRunSession:
            self.prepare_session_calls += 1
            return _ResidentAdapterPreparedRunSession(
                provider_state_dir_container_path=(
                    run_session.provider_state_dir_container_path
                ),
                run_kind=run_session.run_kind,
                provider_session_id=f"prepared:{run_session.provider_session_id}",
            )

        return WorkInvocationDependencies(
            execution=WorkExecutionDependencies(
                container_workspace="/workspace",
                prepare_session=cast(Any, _prepare_session),
                build_session=lambda mount_path, service, provider_state_dir: _Session(
                    provider_state_dir
                ),
                build_runner=lambda session, status_display: cast(
                    WorkExecutionAdapter,
                    _NotStartedCancellationNewSessionRunner(cast(_Session, session)),
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


class _ToolPolicyRenderingResidentRunner(_ResidentSeamRunner):
    def __init__(
        self,
        session: _Session,
    ) -> None:
        super().__init__(session)

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
        del prompt, role, run_kind, session_uuid, on_provider_session_id
        return _tool_policy_effect_text(tool_policy)


class _ToolPolicyRenderingResidentExecutionAdapter:
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

        def _prepare_session(run_session: Any) -> _ResidentAdapterPreparedRunSession:
            return _ResidentAdapterPreparedRunSession(
                provider_state_dir_container_path="/workspace/runtime-state/",
                run_kind=run_session.run_kind,
                provider_session_id=f"prepared:{run_session.provider_session_id}",
            )

        return WorkInvocationDependencies(
            execution=WorkExecutionDependencies(
                container_workspace="/workspace",
                prepare_session=cast(Any, _prepare_session),
                build_session=lambda mount_path, service, provider_state_dir: _Session(
                    provider_state_dir
                ),
                build_runner=lambda session, status_display: cast(
                    WorkExecutionAdapter,
                    _ToolPolicyRenderingResidentRunner(cast(_Session, session)),
                ),
                get_git_identity=lambda: ("Runtime Test", "runtime@example.com"),
            ),
            failure_handling=WorkFailureHandling(timeout_retries=0),
            presentation=WorkPresentationDependencies(),
        )


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


class _StartedUsageLimitResidentRunner(_ResidentSeamRunner):
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
                AssistantTurn("hello"),
                UsageLimit(reset_time=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)),
            ],
            lambda _turn: None,
            provider="codex",
        )


class _ModelActivityUsageLimitResidentRunner(_ResidentSeamRunner):
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
                ModelActivity(),
                UsageLimit(reset_time=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)),
            ],
            lambda _turn: None,
            provider="codex",
        )


class _ModelActivityUsageLimitResidentExecutionAdapter:
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

        def _prepare_session(run_session: Any) -> _ResidentAdapterPreparedRunSession:
            return _ResidentAdapterPreparedRunSession(
                provider_state_dir_container_path="/workspace/runtime-state/",
                run_kind=run_session.run_kind,
                provider_session_id=f"prepared:{run_session.provider_session_id}",
            )

        return WorkInvocationDependencies(
            execution=WorkExecutionDependencies(
                container_workspace="/workspace",
                prepare_session=cast(Any, _prepare_session),
                build_session=lambda mount_path, service, provider_state_dir: _Session(
                    provider_state_dir
                ),
                build_runner=lambda session, status_display: cast(
                    WorkExecutionAdapter,
                    _ModelActivityUsageLimitResidentRunner(cast(_Session, session)),
                ),
                get_git_identity=lambda: ("Runtime Test", "runtime@example.com"),
            ),
            failure_handling=WorkFailureHandling(timeout_retries=0),
            presentation=WorkPresentationDependencies(),
        )


class _StartedUsageLimitResidentExecutionAdapter:
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

        def _prepare_session(run_session: Any) -> _ResidentAdapterPreparedRunSession:
            return _ResidentAdapterPreparedRunSession(
                provider_state_dir_container_path="/workspace/runtime-state/",
                run_kind=run_session.run_kind,
                provider_session_id=f"prepared:{run_session.provider_session_id}",
            )

        return WorkInvocationDependencies(
            execution=WorkExecutionDependencies(
                container_workspace="/workspace",
                prepare_session=cast(Any, _prepare_session),
                build_session=lambda mount_path, service, provider_state_dir: _Session(
                    provider_state_dir
                ),
                build_runner=lambda session, status_display: cast(
                    WorkExecutionAdapter,
                    _StartedUsageLimitResidentRunner(cast(_Session, session)),
                ),
                get_git_identity=lambda: ("Runtime Test", "runtime@example.com"),
            ),
            failure_handling=WorkFailureHandling(timeout_retries=0),
            presentation=WorkPresentationDependencies(),
        )


class _ContinuationBoundStartedUsageLimitResidentExecutionAdapter(
    _StartedUsageLimitResidentExecutionAdapter
):
    def resolve_service(self, service_name: str = "") -> ExecutionProvider:
        if service_name != "bound-service":
            raise AssertionError(f"expected continuation service, got {service_name!r}")
        return _ExecutionService("resolved-service")


class _RetryableProviderFailureResidentRunner(_ResidentSeamRunner):
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


class _RetryableProviderFailureResidentExecutionAdapter:
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

        def _prepare_session(run_session: Any) -> _ResidentAdapterPreparedRunSession:
            return _ResidentAdapterPreparedRunSession(
                provider_state_dir_container_path="/workspace/runtime-state/",
                run_kind=run_session.run_kind,
                provider_session_id=f"prepared:{run_session.provider_session_id}",
            )

        return WorkInvocationDependencies(
            execution=WorkExecutionDependencies(
                container_workspace="/workspace",
                prepare_session=cast(Any, _prepare_session),
                build_session=lambda mount_path, service, provider_state_dir: _Session(
                    provider_state_dir
                ),
                build_runner=lambda session, status_display: cast(
                    WorkExecutionAdapter,
                    _RetryableProviderFailureResidentRunner(cast(_Session, session)),
                ),
                get_git_identity=lambda: ("Runtime Test", "runtime@example.com"),
            ),
            failure_handling=WorkFailureHandling(timeout_retries=0),
            presentation=WorkPresentationDependencies(),
        )


class _StartedRetryableProviderFailureResidentRunner(_ResidentSeamRunner):
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
                AssistantTurn("hello"),
                TransientError(
                    status_code=503,
                    raw_message="retry later",
                    classification="retryable",
                ),
            ],
            lambda _turn: None,
            provider="codex",
        )


class _StartedRetryableProviderFailureResidentExecutionAdapter:
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

        def _prepare_session(run_session: Any) -> _ResidentAdapterPreparedRunSession:
            return _ResidentAdapterPreparedRunSession(
                provider_state_dir_container_path="/workspace/runtime-state/",
                run_kind=run_session.run_kind,
                provider_session_id=f"prepared:{run_session.provider_session_id}",
            )

        return WorkInvocationDependencies(
            execution=WorkExecutionDependencies(
                container_workspace="/workspace",
                prepare_session=cast(Any, _prepare_session),
                build_session=lambda mount_path, service, provider_state_dir: _Session(
                    provider_state_dir
                ),
                build_runner=lambda session, status_display: cast(
                    WorkExecutionAdapter,
                    _StartedRetryableProviderFailureResidentRunner(
                        cast(_Session, session)
                    ),
                ),
                get_git_identity=lambda: ("Runtime Test", "runtime@example.com"),
            ),
            failure_handling=WorkFailureHandling(timeout_retries=0),
            presentation=WorkPresentationDependencies(),
        )


class _TransientProviderFailureResidentRunner(_ResidentSeamRunner):
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


class _TransientProviderFailureResidentExecutionAdapter:
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

        def _prepare_session(run_session: Any) -> _ResidentAdapterPreparedRunSession:
            return _ResidentAdapterPreparedRunSession(
                provider_state_dir_container_path="/workspace/runtime-state/",
                run_kind=run_session.run_kind,
                provider_session_id=f"prepared:{run_session.provider_session_id}",
            )

        return WorkInvocationDependencies(
            execution=WorkExecutionDependencies(
                container_workspace="/workspace",
                prepare_session=cast(Any, _prepare_session),
                build_session=lambda mount_path, service, provider_state_dir: _Session(
                    provider_state_dir
                ),
                build_runner=lambda session, status_display: cast(
                    WorkExecutionAdapter,
                    _TransientProviderFailureResidentRunner(cast(_Session, session)),
                ),
                get_git_identity=lambda: ("Runtime Test", "runtime@example.com"),
            ),
            failure_handling=WorkFailureHandling(timeout_retries=0),
            presentation=WorkPresentationDependencies(),
        )


class _TimeoutResidentRunner(_ResidentSeamRunner):
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


class _TimeoutResidentExecutionAdapter:
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

        def _prepare_session(run_session: Any) -> _ResidentAdapterPreparedRunSession:
            return _ResidentAdapterPreparedRunSession(
                provider_state_dir_container_path="/workspace/runtime-state/",
                run_kind=run_session.run_kind,
                provider_session_id=f"prepared:{run_session.provider_session_id}",
            )

        return WorkInvocationDependencies(
            execution=WorkExecutionDependencies(
                container_workspace="/workspace",
                prepare_session=cast(Any, _prepare_session),
                build_session=lambda mount_path, service, provider_state_dir: _Session(
                    provider_state_dir
                ),
                build_runner=lambda session, status_display: cast(
                    WorkExecutionAdapter,
                    _TimeoutResidentRunner(cast(_Session, session)),
                ),
                get_git_identity=lambda: ("Runtime Test", "runtime@example.com"),
            ),
            failure_handling=WorkFailureHandling(timeout_retries=0),
            presentation=WorkPresentationDependencies(),
        )


class _StartedTimeoutResidentRunner(_ResidentSeamRunner):
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
        raise AgentTimeoutError(
            "timed out",
            invocation_progress=runtime.InvocationProgress.STARTED,
        )


class _StartedTimeoutResidentExecutionAdapter:
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

        def _prepare_session(run_session: Any) -> _ResidentAdapterPreparedRunSession:
            return _ResidentAdapterPreparedRunSession(
                provider_state_dir_container_path="/workspace/runtime-state/",
                run_kind=run_session.run_kind,
                provider_session_id=f"prepared:{run_session.provider_session_id}",
            )

        return WorkInvocationDependencies(
            execution=WorkExecutionDependencies(
                container_workspace="/workspace",
                prepare_session=cast(Any, _prepare_session),
                build_session=lambda mount_path, service, provider_state_dir: _Session(
                    provider_state_dir
                ),
                build_runner=lambda session, status_display: cast(
                    WorkExecutionAdapter,
                    _StartedTimeoutResidentRunner(cast(_Session, session)),
                ),
                get_git_identity=lambda: ("Runtime Test", "runtime@example.com"),
            ),
            failure_handling=WorkFailureHandling(timeout_retries=0),
            presentation=WorkPresentationDependencies(),
        )


class _CredentialFailureEphemeralCompatRunner(_RoleAwareEphemeralCompatWorkRunner):
    async def prompt_only(
        self,
        prompt: str,
        *,
        role: InvocationRole = InvocationRole("implementer"),
        run_kind: RunKind = RunKind.FRESH,
        session_uuid: str | None = None,
        on_provider_session_id: Any = None,
    ) -> str:
        del prompt, role, run_kind, session_uuid, on_provider_session_id
        raise AgentCredentialFailureError(
            "missing auth",
            service_name="codex",
            classification="credential",
            observations=(),
        )


class _CredentialFailureEphemeralCompatExecutionAdapter(
    _RoleAwareEphemeralCompatExecutionAdapter
):
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
                prepare_session=lambda run_session: cast(
                    PreparedRunSessionState, _PreparedRunSession()
                ),
                build_session=lambda mount_path, service, provider_state_dir: _Session(
                    provider_state_dir
                ),
                build_runner=lambda session, status_display: cast(
                    WorkExecutionAdapter,
                    _CredentialFailureEphemeralCompatRunner(),
                ),
                get_git_identity=lambda: ("Runtime Test", "runtime@example.com"),
            ),
            failure_handling=WorkFailureHandling(timeout_retries=0),
            presentation=WorkPresentationDependencies(),
        )


class _HardFailureEphemeralCompatRunner(_RoleAwareEphemeralCompatWorkRunner):
    async def prompt_only(
        self,
        prompt: str,
        *,
        role: InvocationRole = InvocationRole("implementer"),
        run_kind: RunKind = RunKind.FRESH,
        session_uuid: str | None = None,
        on_provider_session_id: Any = None,
    ) -> str:
        del prompt, role, run_kind, session_uuid, on_provider_session_id
        raise HardAgentError("hard failure", service_name="codex")


class _HardFailureEphemeralCompatExecutionAdapter(
    _RoleAwareEphemeralCompatExecutionAdapter
):
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
                prepare_session=lambda run_session: cast(
                    PreparedRunSessionState, _PreparedRunSession()
                ),
                build_session=lambda mount_path, service, provider_state_dir: _Session(
                    provider_state_dir
                ),
                build_runner=lambda session, status_display: cast(
                    WorkExecutionAdapter,
                    _HardFailureEphemeralCompatRunner(),
                ),
                get_git_identity=lambda: ("Runtime Test", "runtime@example.com"),
            ),
            failure_handling=WorkFailureHandling(timeout_retries=0),
            presentation=WorkPresentationDependencies(),
        )


class _CredentialFailureResidentRunner(_ResidentSeamRunner):
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
        raise AgentCredentialFailureError(
            "missing auth",
            service_name="codex",
            classification="credential",
            observations=(),
        )


class _CredentialFailureResidentExecutionAdapter:
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

        def _prepare_session(run_session: Any) -> _ResidentAdapterPreparedRunSession:
            return _ResidentAdapterPreparedRunSession(
                provider_state_dir_container_path="/workspace/runtime-state/",
                run_kind=run_session.run_kind,
                provider_session_id=f"prepared:{run_session.provider_session_id}",
            )

        return WorkInvocationDependencies(
            execution=WorkExecutionDependencies(
                container_workspace="/workspace",
                prepare_session=cast(Any, _prepare_session),
                build_session=lambda mount_path, service, provider_state_dir: _Session(
                    provider_state_dir
                ),
                build_runner=lambda session, status_display: cast(
                    WorkExecutionAdapter,
                    _CredentialFailureResidentRunner(cast(_Session, session)),
                ),
                get_git_identity=lambda: ("Runtime Test", "runtime@example.com"),
            ),
            failure_handling=WorkFailureHandling(timeout_retries=0),
            presentation=WorkPresentationDependencies(),
        )


class _HardFailureResidentRunner(_ResidentSeamRunner):
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


class _HardFailureResidentExecutionAdapter:
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

        def _prepare_session(run_session: Any) -> _ResidentAdapterPreparedRunSession:
            return _ResidentAdapterPreparedRunSession(
                provider_state_dir_container_path="/workspace/runtime-state/",
                run_kind=run_session.run_kind,
                provider_session_id=f"prepared:{run_session.provider_session_id}",
            )

        return WorkInvocationDependencies(
            execution=WorkExecutionDependencies(
                container_workspace="/workspace",
                prepare_session=cast(Any, _prepare_session),
                build_session=lambda mount_path, service, provider_state_dir: _Session(
                    provider_state_dir
                ),
                build_runner=lambda session, status_display: cast(
                    WorkExecutionAdapter,
                    _HardFailureResidentRunner(cast(_Session, session)),
                ),
                get_git_identity=lambda: ("Runtime Test", "runtime@example.com"),
            ),
            failure_handling=WorkFailureHandling(timeout_retries=0),
            presentation=WorkPresentationDependencies(),
        )


class _UnclassifiedProviderFailureResidentRunner(_ResidentSeamRunner):
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
        del prompt, tool_policy, run_kind, session_uuid, on_provider_session_id
        raise AgentFailedError(role.value, Path("/repo"), service_name="codex")


class _UnclassifiedProviderFailureResidentExecutionAdapter:
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

        def _prepare_session(run_session: Any) -> _ResidentAdapterPreparedRunSession:
            return _ResidentAdapterPreparedRunSession(
                provider_state_dir_container_path="/workspace/runtime-state/",
                run_kind=run_session.run_kind,
                provider_session_id=f"prepared:{run_session.provider_session_id}",
            )

        return WorkInvocationDependencies(
            execution=WorkExecutionDependencies(
                container_workspace="/workspace",
                prepare_session=cast(Any, _prepare_session),
                build_session=lambda mount_path, service, provider_state_dir: _Session(
                    provider_state_dir
                ),
                build_runner=lambda session, status_display: cast(
                    WorkExecutionAdapter,
                    _UnclassifiedProviderFailureResidentRunner(cast(_Session, session)),
                ),
                get_git_identity=lambda: ("Runtime Test", "runtime@example.com"),
            ),
            failure_handling=WorkFailureHandling(timeout_retries=0),
            presentation=WorkPresentationDependencies(),
        )


class _UnexpectedFailureResidentRunner(_ResidentSeamRunner):
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
        raise RuntimeError("unexpected failure")


class _UnexpectedFailureResidentExecutionAdapter:
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

        def _prepare_session(run_session: Any) -> _ResidentAdapterPreparedRunSession:
            return _ResidentAdapterPreparedRunSession(
                provider_state_dir_container_path="/workspace/runtime-state/",
                run_kind=run_session.run_kind,
                provider_session_id=f"prepared:{run_session.provider_session_id}",
            )

        return WorkInvocationDependencies(
            execution=WorkExecutionDependencies(
                container_workspace="/workspace",
                prepare_session=cast(Any, _prepare_session),
                build_session=lambda mount_path, service, provider_state_dir: _Session(
                    provider_state_dir
                ),
                build_runner=lambda session, status_display: cast(
                    WorkExecutionAdapter,
                    _UnexpectedFailureResidentRunner(cast(_Session, session)),
                ),
                get_git_identity=lambda: ("Runtime Test", "runtime@example.com"),
            ),
            failure_handling=WorkFailureHandling(timeout_retries=0),
            presentation=WorkPresentationDependencies(),
        )


@pytest.mark.parametrize(
    ("request_factory", "expected_message"),
    [
        (
            lambda: PromptRunRequest(
                prompt="already rendered prompt",
                worktree=WorktreeMount(Path(".")),
                stage=runtime.StageSelection(
                    service="codex",
                    model="gpt-5.4",
                    effort="medium",
                ),
                role=InvocationRole("implementer"),
            ),
            "PromptRunRequest requires an explicit `tool_policy` value.",
        ),
        (
            lambda: prompt_runtime.ResumedSessionRunRequest(
                prompt="already rendered prompt",
                worktree=WorktreeMount(Path(".")),
                model="gpt-5.4",
                effort="medium",
                session_plan=ResumableSessionPlan(
                    role=InvocationRole("reviewer"),
                    worktree=Path("."),
                    namespace="main",
                    service=cast(ExecutionProvider, _ExecutionService("codex")),
                    run_kind=RunKind.FRESH,
                    provider_state_dir=None,
                    provider_session_id=None,
                    auth_seeding_requirement=AuthSeedingRequirement.NOT_REQUIRED,
                ),
            ),
            "ResumedSessionRunRequest requires an explicit `tool_policy` value.",
        ),
    ],
)
def test_tool_capable_requests_require_explicit_tool_policy(
    request_factory: Callable[[], object],
    expected_message: str,
) -> None:
    with pytest.raises(TypeError, match=re.escape(expected_message)):
        request_factory()


def test_text_output_adapter_requires_explicit_tool_policy() -> None:
    with pytest.raises(
        TypeError,
        match=re.escape("TextOutputAdapter requires an explicit `tool_policy` value."),
    ):
        TextOutputAdapter(prompt="already rendered prompt")


@pytest.mark.parametrize("label", ["", "has space", "a/b", "../escape"])
def test_invocation_role_rejects_unsafe_labels(label: str) -> None:
    with pytest.raises(ValueError):
        runtime.InvocationRole(label)


@pytest.mark.parametrize("label", ["", "has space", "a/b", "../escape"])
def test_usage_limit_scope_rejects_unsafe_labels(label: str) -> None:
    with pytest.raises(ValueError):
        runtime.UsageLimitScope(label)


@pytest.mark.parametrize("label", ["", " ", "a/b", "../escape"])
def test_runtime_service_identities_reject_unsafe_labels(label: str) -> None:
    with pytest.raises(ValueError):
        runtime.StageSelection(
            service=label,
            model="provider model / ../ still allowed",
            effort="high effort / ../ still allowed",
        )

    with pytest.raises(ValueError):
        ServiceRegistry(
            {
                label: cast(
                    ServiceSelectionProvider,
                    _Service(
                        "codex",
                        available=True,
                        wake_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
                    ),
                )
            }
        )


@pytest.mark.parametrize("label", [" ", "a/b", "../escape"])
def test_prompt_run_session_namespace_preserves_empty_default_and_rejects_unsafe_non_empty_values(
    label: str,
) -> None:
    assert PromptRunSession().namespace == ""
    assert PromptRunSession(namespace="").namespace == ""

    with pytest.raises(ValueError):
        PromptRunSession(namespace=label)


@pytest.mark.parametrize("label", [" ", "a/b", "../escape"])
def test_provider_session_namespace_seams_preserve_empty_default_and_reject_unsafe_non_empty_values(
    label: str,
    execution_service_factory: Callable[..., ExecutionProvider],
    session_store_factory: Callable[..., _SessionStore],
    resident_provider_session_adapter: _ResidentPlanningProviderSessionAdapter,
) -> None:
    assert (
        ProviderSessionPlanningRequest(
            worktree=Path("."),
            role=InvocationRole("implementer"),
            namespace="",
        ).namespace
        == ""
    )
    assert (
        ResumableSessionPlanRequest(
            worktree=Path("."),
            role=InvocationRole("implementer"),
            namespace="",
            service=execution_service_factory(),
            session_store=session_store_factory(),
            provider_session_adapter=resident_provider_session_adapter,
        ).namespace
        == ""
    )

    with pytest.raises(ValueError):
        ProviderSessionPlanningRequest(
            worktree=Path("."),
            role=InvocationRole("implementer"),
            namespace=label,
        )

    with pytest.raises(ValueError):
        ResumableSessionPlanRequest(
            worktree=Path("."),
            role=InvocationRole("implementer"),
            namespace=label,
            service=execution_service_factory(),
            session_store=session_store_factory(),
            provider_session_adapter=resident_provider_session_adapter,
        )


def test_agent_failed_error_rejects_unsafe_session_namespace_before_building_diagnostics() -> (
    None
):
    with pytest.raises(ValueError):
        AgentFailedError(
            invocation_role="implementer",
            worktree_path=Path("."),
            namespace="../escape",
        )


def test_agent_failed_error_rejects_unsafe_service_name_before_building_diagnostics() -> (
    None
):
    with pytest.raises(ValueError):
        AgentFailedError(
            invocation_role="implementer",
            worktree_path=Path("."),
            service_name="a/b",
        )


def test_agent_timeout_error_exposes_invocation_role_metadata() -> None:
    timeout = AgentTimeoutError(
        "timed out",
        invocation_role="reviewer",
        worktree_path=Path("worktree"),
    )

    assert timeout.invocation_role == "reviewer"


def test_agent_failed_error_exposes_invocation_role_metadata() -> None:
    failed = AgentFailedError(
        invocation_role="reviewer",
        worktree_path=Path("worktree"),
    )

    assert failed.invocation_role == "reviewer"


def test_agent_failed_error_builds_session_dir_from_namespace_metadata() -> None:
    failed = AgentFailedError(
        invocation_role="reviewer",
        worktree_path=Path("worktree"),
        namespace="main",
    )

    assert failed.session_dir == "reviewer/main"


def test_agent_failed_error_builds_session_dir_from_namespace_and_service_name_metadata() -> (
    None
):
    failed = AgentFailedError(
        invocation_role="reviewer",
        worktree_path=Path("worktree"),
        namespace="main",
        service_name="codex",
    )

    assert failed.session_dir == "reviewer/main/codex"


def test_usage_limit_error_exposes_usage_limit_scope_metadata() -> None:
    error = UsageLimitError(
        reset_time=None,
        usage_limit_scope=runtime.UsageLimitScope("quota-review"),
    )

    assert error.usage_limit_scope == runtime.UsageLimitScope("quota-review")


def test_permanent_usage_limit_account_label_remains_diagnostic_metadata() -> None:
    decision = decide_usage_limit_continuation(
        UsageLimitOutcome(
            reset_time=None,
            service_name=None,
            account_label="team account",
            is_permanent=True,
        ),
        stage_override=None,
        service_registry=None,
        now=datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc),
        compute_wake_time=lambda reset_time, current_time: (current_time, False),
    )

    assert isinstance(decision, Stop)
    assert decision.message is not None
    assert "team account" in decision.message
    assert "claude" not in decision.message.lower()


@pytest.mark.parametrize("service_name", [" ", "a/b", "../escape"])
def test_hard_agent_error_rejects_unsafe_runtime_service_labels_before_recording_diagnostics(
    service_name: str,
) -> None:
    with pytest.raises(ValueError):
        HardAgentError("hard", service_name=service_name)


@pytest.mark.parametrize("service_name", ["", " ", "a/b", "../escape"])
def test_provider_state_path_helpers_reject_unsafe_runtime_service_labels(
    service_name: str,
) -> None:
    role = InvocationRole("implementer")

    with pytest.raises(ValueError):
        provider_state_relpath(role, service_name, namespace="main")

    with pytest.raises(ValueError):
        normalize_state_dir_relpath(
            role,
            "main",
            service_name,
            ".runtime-session/implementer/main/codex/",
        )


def test_model_and_effort_values_remain_provider_execution_parameters(
    ephemeral_request_factory: Callable[..., prompt_runtime.EphemeralRunRequest],
    service_registry_factory: Callable[..., ServiceRegistry],
    stage_selection_factory: Callable[..., runtime.StageSelection],
) -> None:
    result = asyncio.run(
        prompt_runtime.EphemeralRuntime(
            execution_adapter=_RoleAwareEphemeralCompatExecutionAdapter(),
            service_registry=service_registry_factory("codex"),
        ).run_ephemeral(
            ephemeral_request_factory(
                stage=stage_selection_factory(
                    service="codex",
                    model="../gpt 5 / provider specific",
                    effort="very-high / provider specific",
                )
            )
        )
    )

    assert result.selected_model == "../gpt 5 / provider specific"
    assert result.selected_effort == "very-high / provider specific"


def test_import_isolation_helper_reports_forbidden_modules() -> None:
    with pytest.raises(ImportError) as excinfo:
        assert_runtime_import_isolation(
            importer="agent_runtime",
            newly_loaded_modules={"allowed.mod", "forbidden.pkg", "forbidden.pkg.sub"},
            forbidden_prefixes=("forbidden.pkg",),
        )

    assert "forbidden.pkg" in str(excinfo.value)


def test_agent_invocation_log_uses_invocation_role_header_key(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "agent.log"
    invocation_log = AgentInvocationLog(
        now_local=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc)
    )

    with invocation_log.open_work_invocation(
        log_path=log_path,
        role=InvocationRole("implementer"),
        run_kind=RunKind.FRESH,
        session_uuid=None,
        prompt="already rendered prompt",
    ):
        pass

    header = json.loads(log_path.read_text().splitlines()[0])

    assert header["invocation_role"] == "implementer"
    assert "role" not in header


def test_agent_invocation_log_uses_log_name_and_logs_dir_parameters(
    tmp_path: Path,
) -> None:
    invocation_log = AgentInvocationLog(
        now_local=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc)
    )

    reserved_path = invocation_log.reserve(
        log_name="Issue 51 Review",
        logs_dir=tmp_path,
    )
    logical_log = invocation_log.start_logical_session(
        log_name="Issue 51 Review",
        logs_dir=tmp_path,
    )

    assert reserved_path.name == "issue-51-review-20260101T0000.log"
    assert logical_log.log_path.name == "issue-51-review-20260101T0000-2.log"


def test_agent_invocation_log_omits_default_usage_limit_scope(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "agent.log"
    invocation_log = AgentInvocationLog(
        now_local=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc)
    )

    with invocation_log.open_work_invocation(
        log_path=log_path,
        role=InvocationRole("implementer"),
        usage_limit_scope=runtime.UsageLimitScope("implementer"),
        run_kind=RunKind.FRESH,
        session_uuid=None,
        prompt="same scope as role",
    ):
        pass

    header = json.loads(log_path.read_text().splitlines()[0])

    assert "usage_limit_scope" not in header


def test_agent_invocation_log_records_non_default_usage_limit_scope(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "agent.log"
    invocation_log = AgentInvocationLog(
        now_local=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc)
    )

    with invocation_log.open_work_invocation(
        log_path=log_path,
        role=InvocationRole("implementer"),
        usage_limit_scope=runtime.UsageLimitScope("repo-write"),
        run_kind=RunKind.RESUME,
        session_uuid=None,
        prompt="different scope from role",
    ):
        pass

    header = json.loads(log_path.read_text().splitlines()[0])

    assert header["invocation_role"] == "implementer"
    assert header["usage_limit_scope"] == "repo-write"


def test_agent_invocation_log_records_provider_session_id_in_header(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "agent.log"
    invocation_log = AgentInvocationLog(
        now_local=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc)
    )

    with invocation_log.open_work_invocation(
        log_path=log_path,
        role=InvocationRole("implementer"),
        usage_limit_scope=runtime.UsageLimitScope("repo-write"),
        run_kind=RunKind.RESUME,
        session_uuid=None,
        prompt="different scope from role",
    ) as work_invocation:
        work_invocation.record_provider_session_id("provider-session")

    header = json.loads(log_path.read_text().splitlines()[0])

    assert header["provider_session_id"] == "provider-session"


def test_stage_chain_resolution_prefers_first_available_configured_service(
    stage_selection_factory: Callable[..., runtime.StageSelection],
) -> None:
    override = stage_selection_factory(
        service="missing",
        model="ignored",
        effort="medium",
        fallback=stage_selection_factory(
            service="codex",
            fallback=stage_selection_factory(
                service="claude",
                model="sonnet",
                effort="high",
            ),
        ),
    )

    selection = select_configured_candidate_chain(
        override,
        configured_service_names=("codex", "claude"),
        available_service_names=("claude",),
    )

    assert selection.has_configured_candidate is True
    assert selection.selected_chain == stage_selection_factory(
        service="claude",
        model="sonnet",
        effort="high",
    )
    assert render_chain_label(override) == "missing -> codex -> claude"
    assert [entry.service for entry in chain_entries(override)] == [
        "missing",
        "codex",
        "claude",
    ]


def test_public_stage_selection_requires_non_empty_candidate_configuration() -> None:
    with pytest.raises(ValueError, match="service"):
        runtime.StageSelection(
            service="",
            model="gpt-5.4",
            effort="medium",
        )

    with pytest.raises(ValueError, match="model"):
        runtime.StageSelection(
            service="codex",
            model="",
            effort="medium",
        )

    with pytest.raises(ValueError, match="effort"):
        runtime.StageSelection(
            service="codex",
            model="gpt-5.4",
            effort="",
        )

    with pytest.raises(ValueError, match="model"):
        runtime.StageSelection(
            service="codex",
            model="gpt-5.4",
            effort="medium",
            fallback=runtime.StageSelection(
                service="claude",
                model="",
                effort="high",
            ),
        )


def test_public_stage_selection_rejects_path_like_service_name() -> None:
    with pytest.raises(ValueError, match="StageSelection service"):
        runtime.StageSelection(
            service="bad/name",
            model="gpt-5.4",
            effort="medium",
        )


def test_public_stage_selection_rejects_invalid_fallback_effort() -> None:
    with pytest.raises(ValueError, match="effort"):
        runtime.StageSelection(
            service="codex",
            model="gpt-5.4",
            effort="medium",
            fallback=runtime.StageSelection(
                service="claude",
                model="sonnet",
                effort="",
            ),
        )


def test_service_registry_resolve_and_wake_time(
    service_registry_factory: Callable[..., ServiceRegistry],
    stage_selection_factory: Callable[..., runtime.StageSelection],
) -> None:
    registry = service_registry_factory(
        "codex",
        "claude",
        unavailable={"codex"},
        wake_times={
            "codex": datetime(2026, 1, 1, tzinfo=timezone.utc),
            "claude": datetime(2026, 1, 2, tzinfo=timezone.utc),
        },
    )
    override = stage_selection_factory(
        service="codex",
        fallback=stage_selection_factory(
            service="claude",
            model="sonnet",
            effort="high",
        ),
    )

    resolved = registry.resolve(override, datetime(2026, 1, 1, tzinfo=timezone.utc))

    assert resolved == stage_selection_factory(
        service="claude",
        model="sonnet",
        effort="high",
    )
    assert registry.has_available(datetime(2026, 1, 1, tzinfo=timezone.utc)) is True
    assert registry.next_wake_time(
        datetime(2026, 1, 1, tzinfo=timezone.utc)
    ) == datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_service_registry_rejects_invalid_public_service_name_configuration() -> None:
    with pytest.raises(ValueError, match="ServiceRegistry service name"):
        ServiceRegistry(
            {
                "bad/name": cast(
                    ServiceSelectionProvider,
                    _Service(
                        "bad/name",
                        available=True,
                        wake_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
                    ),
                ),
            }
        )


def test_application_can_render_service_availability_summary_from_registry(
    service_registry_factory: Callable[..., ServiceRegistry],
) -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    registry = service_registry_factory(
        "codex",
        "claude",
        unavailable={"codex"},
        wake_times={
            "codex": datetime(2026, 1, 2, tzinfo=timezone.utc),
            "claude": datetime(2026, 1, 3, tzinfo=timezone.utc),
        },
    )

    summary_lines = [
        f"{name}: {'available' if service.is_available(now=now) else 'unavailable'}"
        for name, service in registry.services.items()
    ]

    assert summary_lines == [
        "codex: unavailable",
        "claude: available",
    ]


def test_public_stage_selection_rejects_invalid_fallback_service_name() -> None:
    with pytest.raises(ValueError, match="service"):
        runtime.StageSelection(
            service="codex",
            model="gpt-5.4",
            effort="medium",
            fallback=runtime.StageSelection(
                service="",
                model="sonnet",
                effort="high",
            ),
        )


def test_service_registry_preserves_per_candidate_configuration_on_filtered_chain(
    service_registry_factory: Callable[..., ServiceRegistry],
    stage_selection_factory: Callable[..., runtime.StageSelection],
) -> None:
    registry = service_registry_factory(
        "codex",
        "claude",
        "gemini",
        unavailable={"codex"},
    )
    override = stage_selection_factory(
        service="codex",
        fallback=stage_selection_factory(
            service="missing",
            model="ignored",
            effort="low",
            fallback=stage_selection_factory(
                service="claude",
                model="sonnet",
                effort="high",
                fallback=stage_selection_factory(
                    service="gemini",
                    model="2.5-pro",
                    effort="low",
                ),
            ),
        ),
    )

    assert registry.resolve(
        override,
        datetime(2026, 1, 1, tzinfo=timezone.utc),
    ) == stage_selection_factory(
        service="claude",
        model="sonnet",
        effort="high",
        fallback=stage_selection_factory(
            service="gemini",
            model="2.5-pro",
            effort="low",
        ),
    )


def test_usage_limit_continuation_exposes_selected_usage_limit_scope() -> None:
    now = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    wake_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

    decision = decide_usage_limit_continuation(
        UsageLimitOutcome(
            reset_time=None,
            service_name="codex",
            usage_limit_scope=runtime.UsageLimitScope("quota-review"),
        ),
        stage_override=None,
        service_registry=None,
        now=now,
        compute_wake_time=lambda reset_time, current_time: (wake_time, False),
    )

    assert decision == SleepUntil(
        wake_time=wake_time,
        message="Usage limit reached. Sleeping until 12:00. Press Ctrl+C to abort.",
        is_estimated=False,
        usage_limit_scope=runtime.UsageLimitScope("quota-review"),
    )


def test_prompt_run_request_accepts_explicit_no_tools_tool_access() -> None:
    request = PromptRunRequest(
        prompt="already rendered prompt",
        worktree=WorktreeMount(Path("/repo")),
        stage=runtime.StageSelection(
            service="codex",
            model="gpt-5.4",
            effort="medium",
        ),
        role=InvocationRole("implementer"),
        tool_access=runtime.ToolAccess.no_tools(),
    )

    assert request.tool_access == runtime.ToolAccess.no_tools()
    assert request.tool_policy == runtime.ToolAccess.no_tools().tool_policy


def test_tool_access_none_rejects_non_toolless_policy() -> None:
    with pytest.raises(
        ValueError,
        match=re.escape(
            "ToolAccess.no_tools() must forbid provider tool access with the closed no-tools policy."
        ),
    ):
        runtime.ToolAccess(
            kind="none",
            workspace=None,
            tool_policy=runtime.ToolPolicy.FULL,
        )


def test_provider_state_helpers_normalize_legacy_layout_and_build_session_id_path() -> (
    None
):
    legacy = ".runtime-session/implementer/main/codex/"

    assert (
        provider_state_relpath(
            InvocationRole("implementer"),
            "codex",
            session_root=".runtime-session",
        )
        == ".runtime-session/implementer/codex/"
    )
    assert (
        normalize_state_dir_relpath(
            InvocationRole("implementer"),
            "main",
            "codex",
            legacy,
        )
        == ".runtime-session/implementer/main/codex/"
    )
    assert provider_state_session_id_path(Path("state"), "codex") == Path(
        "state/thread_id"
    )


def test_select_resumable_provider_session_id_recovers_and_persists_state(
    session_store_factory: Callable[..., _SessionStore],
) -> None:
    state_dir = Path("state")
    session_store = session_store_factory()

    selection = select_resumable_provider_session_id(
        session_store,
        "codex",
        provider_state_dir=state_dir,
        has_resumable_provider_state=True,
        recover_provider_session_id=lambda path: (
            "provider-session" if path == state_dir else None
        ),
    )

    assert selection == ProviderSessionSelection(
        provider_session_id="provider-session",
        persist_provider_session_id=True,
    )
    assert session_store.service_session_id("codex") == "provider-session"


def test_select_resumable_provider_session_id_prefers_session_store_over_recovery(
    session_store_factory: Callable[..., _SessionStore],
) -> None:
    recover_calls = 0
    session_store = session_store_factory(service_sessions={"codex": "stored-session"})

    def recover_provider_session_id(_path: Path | None) -> str | None:
        nonlocal recover_calls
        recover_calls += 1
        return "recovered-session"

    selection = select_resumable_provider_session_id(
        session_store,
        "codex",
        provider_state_dir=Path("state"),
        has_resumable_provider_state=True,
        recover_provider_session_id=recover_provider_session_id,
    )

    assert selection == ProviderSessionSelection(
        provider_session_id="stored-session",
        persist_provider_session_id=False,
    )
    assert recover_calls == 0
    assert session_store.service_session_id("codex") == "stored-session"


def test_exact_resumable_service_session_requires_matching_metadata_and_maybe_matcher(
    session_store_factory: Callable[..., _SessionStore],
) -> None:
    session_store = session_store_factory(
        service_sessions={"codex": "provider-session"},
        service_metadata={"codex": {"provider_session_id": "provider-session"}},
        exact_transcript_service="codex",
    )

    assert (
        is_exact_resumable_service_session(
            session_store,
            "codex",
            provider_session_id="provider-session",
            provider_state_dir=Path("state"),
        )
        is True
    )
    assert (
        is_exact_resumable_service_session(
            session_store,
            "codex",
            provider_session_id="provider-session",
            provider_state_dir=Path("state"),
            exact_provider_session_matcher=lambda *_args: False,
        )
        is False
    )


def test_provider_output_reduction_returns_result() -> None:
    turns: list[str] = []

    result = reduce_text_output_events(
        [
            PromptTokens(2),
            UnsupportedTokens(3, "source"),
            AssistantTurn("hello"),
            Result("done"),
        ],
        turns.append,
        provider="codex",
    )

    assert result == "done"
    assert turns == ["hello"]


def test_provider_output_reduction_reports_prompt_tokens() -> None:
    token_counts: list[int] = []

    result = reduce_text_output_events(
        [PromptTokens(2)],
        lambda _turn: None,
        token_counts.append,
        provider="codex",
    )

    assert result == ""
    assert token_counts == [2]


def test_provider_output_reduction_maps_usage_limit() -> None:
    with pytest.raises(UsageLimitError) as exc_info:
        reduce_text_output_events(
            [UsageLimit(reset_time=None)], lambda _turn: None, provider="codex"
        )

    assert exc_info.value.service_name == "codex"
    assert exc_info.value.reset_time is None


def test_provider_output_reduction_accepts_explicit_model_activity_for_usage_limit() -> (
    None
):
    with pytest.raises(UsageLimitError) as exc_info:
        reduce_text_output_events(
            [ModelActivity(), UsageLimit(reset_time=None)],
            lambda _turn: None,
            provider="codex",
        )

    assert exc_info.value.invocation_progress is runtime.InvocationProgress.STARTED


def test_provider_output_reduction_keeps_unknown_activity_usage_limits_not_started() -> (
    None
):
    with pytest.raises(UsageLimitError) as exc_info:
        reduce_text_output_events(
            [
                PromptTokens(2),
                UnsupportedTokens(3, "source"),
                UsageLimit(reset_time=None),
            ],
            lambda _turn: None,
            provider="codex",
        )

    assert exc_info.value.invocation_progress is runtime.InvocationProgress.NOT_STARTED


def test_provider_output_reduction_maps_transient_error() -> None:
    with pytest.raises(TransientAgentError) as exc_info:
        reduce_text_output_events(
            [TransientError(status_code=503, raw_message="retry")],
            lambda _turn: None,
            provider="codex",
        )

    assert exc_info.value.status_code == 503
    assert str(exc_info.value) == "retry"


def test_provider_output_reduction_maps_retryable_provider_failure() -> None:
    with pytest.raises(RetryableProviderFailureError) as exc_info:
        reduce_text_output_events(
            [
                TransientError(
                    status_code=503,
                    raw_message="retry",
                    classification="retryable",
                )
            ],
            lambda _turn: None,
            provider="codex",
        )

    assert exc_info.value.service_name == "codex"
    assert exc_info.value.status_code == 503
    assert exc_info.value.invocation_progress is runtime.InvocationProgress.NOT_STARTED
    assert str(exc_info.value) == "retry"


def test_provider_output_reduction_reports_started_progress_for_retryable_provider_failure() -> (
    None
):
    with pytest.raises(RetryableProviderFailureError) as exc_info:
        reduce_text_output_events(
            [
                AssistantTurn("hello"),
                TransientError(
                    status_code=503,
                    raw_message="retry",
                    classification="retryable",
                ),
            ],
            lambda _turn: None,
            provider="codex",
        )

    assert exc_info.value.invocation_progress is runtime.InvocationProgress.STARTED


def test_provider_output_reduction_accepts_explicit_model_activity_for_retryable_provider_failure() -> (
    None
):
    with pytest.raises(RetryableProviderFailureError) as exc_info:
        reduce_text_output_events(
            [
                ModelActivity(),
                TransientError(
                    status_code=503,
                    raw_message="retry",
                    classification="retryable",
                ),
            ],
            lambda _turn: None,
            provider="codex",
        )

    assert exc_info.value.invocation_progress is runtime.InvocationProgress.STARTED


def test_provider_output_reduction_maps_hard_error(
    provider_error_observation: ProviderErrorObservation,
) -> None:
    with pytest.raises(HardAgentError) as exc_info:
        reduce_text_output_events(
            [
                HardError(
                    status_code=400,
                    raw_message="bad",
                    observations=(provider_error_observation,),
                )
            ],
            lambda _turn: None,
            provider="codex",
        )

    assert exc_info.value.service_name == "codex"
    assert exc_info.value.status_code == 400
    assert exc_info.value.observations == (provider_error_observation,)


def test_provider_output_reduction_maps_credential_failure(
    provider_error_observation: ProviderErrorObservation,
) -> None:
    with pytest.raises(AgentCredentialFailureError) as exc_info:
        reduce_text_output_events(
            [
                CredentialFailure(
                    raw_message="missing auth",
                    service_name="codex",
                    source_observations=(provider_error_observation,),
                )
            ],
            lambda _turn: None,
            provider="codex",
        )

    assert exc_info.value.service_name == "codex"
    assert str(exc_info.value) == "missing auth"
    assert exc_info.value.observations == (provider_error_observation,)


def test_provider_output_reduction_joins_assistant_turns_without_result() -> None:
    turns: list[str] = []

    result = reduce_text_output_events(
        [
            PromptTokens(2),
            AssistantTurn("hello"),
            UnsupportedTokens(3, "source"),
            AssistantTurn("world"),
        ],
        turns.append,
        provider="codex",
    )

    assert result == "hello\nworld"
    assert turns == ["hello", "world"]


def test_provider_output_reduction_stops_after_result() -> None:
    turns: list[str] = []
    token_counts: list[int] = []

    result = reduce_text_output_events(
        [
            AssistantTurn("hello"),
            Result("done"),
            PromptTokens(99),
            AssistantTurn("ignored"),
        ],
        turns.append,
        token_counts.append,
        provider="codex",
    )

    assert result == "done"
    assert turns == ["hello"]


def test_runtime_client_runs_claude_ephemeral_stage_through_builtin_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}

    class _ClaudeProcess:
        def __init__(self) -> None:
            self.stdout = iter(
                [
                    json.dumps(
                        {
                            "type": "assistant",
                            "message": {
                                "content": [{"type": "text", "text": "intermediate"}],
                                "usage": {
                                    "input_tokens": 5,
                                    "cache_creation_input_tokens": 0,
                                    "cache_read_input_tokens": 0,
                                },
                            },
                        }
                    )
                    + "\n",
                    json.dumps({"type": "result", "result": "final output"}) + "\n",
                ]
            )
            self.stderr = iter(())
            self.returncode = 0

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
    ) -> _ClaudeProcess:
        captured["command"] = command
        captured["shell"] = shell
        captured["cwd"] = cwd
        captured["env"] = env
        captured["stdout"] = stdout
        captured["stderr"] = stderr
        captured["text"] = text
        return _ClaudeProcess()

    monkeypatch.setattr(subprocess, "Popen", _fake_popen)

    outcome = runtime.RuntimeClient().run_ephemeral(
        prompt_runtime.EphemeralRunRequest(
            prompt="already rendered prompt",
            worktree=tmp_path,
            stage=runtime.StageSelection(
                service="claude",
                model="sonnet",
                effort="medium",
            ),
            role=InvocationRole("implementer"),
            tool_access=runtime.ToolAccess.no_tools(),
            auth=runtime.ProviderAuth(claude_code_oauth_token="oauth-token"),
        )
    )

    assert outcome == prompt_runtime.RuntimeOutcome.completed(
        output="final output",
        result=prompt_runtime.EphemeralRunResult(
            output="final output",
            selected_service="claude",
            selected_model="sonnet",
            selected_effort="medium",
            tool_access=runtime.ToolAccess.no_tools(),
            used_fallback=False,
            metadata=prompt_runtime.EphemeralResultMetadata(
                selected_service_path=("claude",),
                runtime=prompt_runtime.EphemeralRuntimeMetadata(
                    run_kind=RunKind.FRESH,
                    session_namespace="",
                ),
            ),
        ),
    )
    assert captured["cwd"] == tmp_path
    assert captured["env"]["CLAUDE_CODE_OAUTH_TOKEN"] == "oauth-token"
    assert "--output-format stream-json" in captured["command"]


def test_runtime_client_runs_rendered_prompt_through_claude_subprocess(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    claude_path = tmp_path / "fake-claude"
    claude_path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import json",
                "import sys",
                "prompt = sys.stdin.read()",
                'print(json.dumps({"type": "result", "result": prompt}))',
            ]
        )
        + "\n"
    )
    claude_path.chmod(0o755)
    monkeypatch.setattr(
        prompt_runtime,
        "_claude_command",
        lambda **kwargs: f"{claude_path} < {kwargs['prompt_path']}",
    )

    outcome = runtime.RuntimeClient().run_ephemeral(
        prompt_runtime.EphemeralRunRequest(
            prompt="already rendered prompt",
            worktree=tmp_path,
            stage=runtime.StageSelection(
                service="claude",
                model="sonnet",
                effort="medium",
            ),
            role=InvocationRole("implementer"),
            tool_access=runtime.ToolAccess.no_tools(),
            auth=runtime.ProviderAuth(claude_code_oauth_token="oauth-token"),
        )
    )

    assert outcome == prompt_runtime.RuntimeOutcome.completed(
        output="already rendered prompt",
        result=prompt_runtime.EphemeralRunResult(
            output="already rendered prompt",
            selected_service="claude",
            selected_model="sonnet",
            selected_effort="medium",
            tool_access=runtime.ToolAccess.no_tools(),
            used_fallback=False,
            metadata=prompt_runtime.EphemeralResultMetadata(
                selected_service_path=("claude",),
                runtime=prompt_runtime.EphemeralRuntimeMetadata(
                    run_kind=RunKind.FRESH,
                    session_namespace="",
                ),
            ),
        ),
    )


def test_runtime_client_passes_only_claude_specific_env_to_subprocess(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("SHOULD_NOT_LEAK", "host-value")
    captured: dict[str, Any] = {}

    class _ClaudeProcess:
        def __init__(self) -> None:
            self.stdout = iter(
                [json.dumps({"type": "result", "result": "final output"}) + "\n"]
            )
            self.stderr = iter(())
            self.returncode = 0

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
    ) -> _ClaudeProcess:
        captured["env"] = env
        return _ClaudeProcess()

    monkeypatch.setattr(subprocess, "Popen", _fake_popen)

    runtime.RuntimeClient().run_ephemeral(
        prompt_runtime.EphemeralRunRequest(
            prompt="already rendered prompt",
            worktree=tmp_path,
            stage=runtime.StageSelection(
                service="claude",
                model="sonnet",
                effort="medium",
            ),
            role=InvocationRole("implementer"),
            tool_access=runtime.ToolAccess.no_tools(),
            auth=runtime.ProviderAuth(claude_code_oauth_token="oauth-token"),
        )
    )

    assert captured["env"] == {"CLAUDE_CODE_OAUTH_TOKEN": "oauth-token"}


def test_runtime_client_maps_claude_usage_limit_stream_to_usage_limited_outcome(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _ClaudeProcess:
        def __init__(self) -> None:
            self.stdout = iter(
                [
                    json.dumps(
                        {
                            "type": "result",
                            "is_error": True,
                            "api_error_status": 429,
                            "result": "Claude usage limit reached.",
                        }
                    )
                    + "\n"
                ]
            )
            self.stderr = iter(())
            self.returncode = 0

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *args, **kwargs: _ClaudeProcess(),
    )

    outcome = runtime.RuntimeClient().run_ephemeral(
        prompt_runtime.EphemeralRunRequest(
            prompt="already rendered prompt",
            worktree=tmp_path,
            stage=runtime.StageSelection(
                service="claude",
                model="sonnet",
                effort="medium",
            ),
            role=InvocationRole("implementer"),
            tool_access=runtime.ToolAccess.no_tools(),
            auth=runtime.ProviderAuth(claude_code_oauth_token="oauth-token"),
        )
    )

    assert outcome == prompt_runtime.RuntimeOutcome.usage_limited(
        output="",
        service_name="claude",
        reset_time=None,
        usage_limit_scope=runtime.UsageLimitScope("implementer"),
        invocation_progress=prompt_runtime.InvocationProgress.NOT_STARTED,
    )


def test_runtime_client_reachable_claude_stage_requires_token_without_falling_through(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    subprocess_calls = 0

    def _unexpected_popen(*args: Any, **kwargs: Any) -> Any:
        nonlocal subprocess_calls
        subprocess_calls += 1
        raise AssertionError("subprocess should not start without Claude auth")

    monkeypatch.setattr(subprocess, "Popen", _unexpected_popen)

    with pytest.raises(AgentCredentialFailureError) as exc_info:
        runtime.RuntimeClient().run_ephemeral(
            prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                worktree=tmp_path,
                stage=runtime.StageSelection(
                    service="missing",
                    model="ignored",
                    effort="low",
                    fallback=runtime.StageSelection(
                        service="claude",
                        model="sonnet",
                        effort="medium",
                        fallback=runtime.StageSelection(
                            service="codex",
                            model="gpt-5",
                            effort="medium",
                        ),
                    ),
                ),
                role=InvocationRole("implementer"),
                tool_access=runtime.ToolAccess.no_tools(),
                auth=runtime.ProviderAuth(),
            )
        )

    assert exc_info.value.service_name == "claude"
    assert subprocess_calls == 0


def test_runtime_client_maps_claude_transient_error_stream_to_transient_exception(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _ClaudeProcess:
        def __init__(self) -> None:
            self.stdout = iter(
                [
                    json.dumps(
                        {
                            "type": "result",
                            "is_error": True,
                            "api_error_status": 500,
                            "result": "temporary Claude failure",
                        }
                    )
                    + "\n"
                ]
            )
            self.stderr = iter(())
            self.returncode = 0

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *args, **kwargs: _ClaudeProcess(),
    )

    with pytest.raises(TransientAgentError) as exc_info:
        runtime.RuntimeClient().run_ephemeral(
            prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                worktree=tmp_path,
                stage=runtime.StageSelection(
                    service="claude",
                    model="sonnet",
                    effort="medium",
                ),
                role=InvocationRole("implementer"),
                tool_access=runtime.ToolAccess.no_tools(),
                auth=runtime.ProviderAuth(claude_code_oauth_token="oauth-token"),
            )
        )

    assert exc_info.value.status_code == 500


def test_runtime_client_parses_claude_usage_limit_reset_time(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _ClaudeProcess:
        def __init__(self) -> None:
            self.stdout = iter(
                [
                    json.dumps(
                        {
                            "type": "result",
                            "is_error": True,
                            "api_error_status": 429,
                            "result": "Claude usage limit reached, resets Jan 2, 4pm (UTC).",
                        }
                    )
                    + "\n"
                ]
            )
            self.stderr = iter(())
            self.returncode = 0

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *args, **kwargs: _ClaudeProcess(),
    )
    monkeypatch.setattr(
        prompt_runtime._time_module,
        "now_local",
        lambda: datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
    )

    outcome = runtime.RuntimeClient().run_ephemeral(
        prompt_runtime.EphemeralRunRequest(
            prompt="already rendered prompt",
            worktree=tmp_path,
            stage=runtime.StageSelection(
                service="claude",
                model="sonnet",
                effort="medium",
            ),
            role=InvocationRole("implementer"),
            tool_access=runtime.ToolAccess.no_tools(),
            auth=runtime.ProviderAuth(claude_code_oauth_token="oauth-token"),
        )
    )

    assert outcome == prompt_runtime.RuntimeOutcome.usage_limited(
        output="",
        service_name="claude",
        reset_time=datetime(2026, 1, 2, 16, 0, tzinfo=timezone.utc),
        usage_limit_scope=runtime.UsageLimitScope("implementer"),
        invocation_progress=prompt_runtime.InvocationProgress.NOT_STARTED,
    )


def test_runtime_module_claude_event_parser_keeps_runtime_reset_time_override() -> None:
    reset_time = datetime(2026, 1, 2, 16, 0, tzinfo=timezone.utc)

    def _fake_parse_claude_reset_time(_text: object) -> datetime | None:
        return reset_time

    original_parse_reset_time = prompt_runtime._parse_claude_reset_time
    prompt_runtime._parse_claude_reset_time = cast(Any, _fake_parse_claude_reset_time)
    try:
        events = prompt_runtime._parse_claude_event(
            json.dumps(
                {
                    "type": "result",
                    "is_error": True,
                    "api_error_status": 429,
                    "result": "Claude usage limit reached.",
                }
            )
        )
    finally:
        prompt_runtime._parse_claude_reset_time = original_parse_reset_time

    assert events == [UsageLimit(reset_time=reset_time, raw_message=None)]


def test_runtime_client_keeps_runtime_selected_service_path_override_in_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _ClaudeProcess:
        def __init__(self) -> None:
            self.stdout = iter(
                [json.dumps({"type": "result", "result": "final output"}) + "\n"]
            )
            self.stderr = iter(())
            self.returncode = 0

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *args, **kwargs: _ClaudeProcess(),
    )
    monkeypatch.setattr(
        prompt_runtime,
        "_selected_service_path",
        lambda *_args, **_kwargs: ("patched", "claude"),
    )

    outcome = runtime.RuntimeClient().run_ephemeral(
        prompt_runtime.EphemeralRunRequest(
            prompt="already rendered prompt",
            worktree=tmp_path,
            stage=runtime.StageSelection(
                service="claude",
                model="sonnet",
                effort="medium",
            ),
            role=InvocationRole("implementer"),
            tool_access=runtime.ToolAccess.no_tools(),
            auth=runtime.ProviderAuth(claude_code_oauth_token="oauth-token"),
        )
    )

    assert outcome == prompt_runtime.RuntimeOutcome.completed(
        output="final output",
        result=prompt_runtime.EphemeralRunResult(
            output="final output",
            selected_service="claude",
            selected_model="sonnet",
            selected_effort="medium",
            tool_access=runtime.ToolAccess.no_tools(),
            used_fallback=True,
            metadata=prompt_runtime.EphemeralResultMetadata(
                selected_service_path=("patched", "claude"),
                runtime=prompt_runtime.EphemeralRuntimeMetadata(
                    run_kind=RunKind.FRESH,
                    session_namespace="",
                ),
            ),
        ),
    )


def test_runtime_client_preserves_claude_credential_failure_observations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    denial_message = "Disabled Claude subscription access for Claude Code."

    class _ClaudeProcess:
        def __init__(self) -> None:
            self.stdout = iter(
                [
                    json.dumps(
                        {
                            "type": "result",
                            "is_error": True,
                            "api_error_status": 403,
                            "result": denial_message,
                        }
                    )
                    + "\n"
                ]
            )
            self.stderr = iter(())
            self.returncode = 0

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *args, **kwargs: _ClaudeProcess(),
    )

    with pytest.raises(AgentCredentialFailureError) as exc_info:
        runtime.RuntimeClient().run_ephemeral(
            prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                worktree=tmp_path,
                stage=runtime.StageSelection(
                    service="claude",
                    model="sonnet",
                    effort="medium",
                ),
                role=InvocationRole("implementer"),
                tool_access=runtime.ToolAccess.no_tools(),
                auth=runtime.ProviderAuth(claude_code_oauth_token="oauth-token"),
            )
        )

    assert exc_info.value.observations == (
        ProviderErrorObservation(
            service_name="claude",
            raw_provider_text=denial_message,
            source_stream="json_event.result",
            status_code=403,
        ),
    )


def test_agent_timeout_error_is_an_agent_runtime_error() -> None:
    timeout = AgentTimeoutError("timed out")

    assert isinstance(timeout, AgentRuntimeError)


def test_usage_limit_error_defaults_service_name_metadata_to_none() -> None:
    usage_limit = UsageLimitError(reset_time=None)

    assert usage_limit.service_name is None


def test_transient_agent_error_exposes_status_code_metadata() -> None:
    transient = TransientAgentError("transient", status_code=502)

    assert transient.status_code == 502


def test_hard_agent_error_exposes_service_name_metadata() -> None:
    hard = HardAgentError("hard", status_code=400, service_name="codex")

    assert hard.service_name == "codex"


def test_agent_failed_error_builds_session_dir_from_service_name_metadata() -> None:
    failed = AgentFailedError("implementer", Path("worktree"), service_name="codex")

    assert failed.session_dir == "implementer/codex"
