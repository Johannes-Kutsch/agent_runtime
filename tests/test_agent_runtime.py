from __future__ import annotations

import asyncio
import importlib
import json
import re
from collections.abc import Callable
from dataclasses import FrozenInstanceError, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import pytest

import agent_runtime as runtime
import agent_runtime.provider_session_adapter as provider_session_adapter_runtime
import agent_runtime.runtime as prompt_runtime
import agent_runtime.session as session_runtime
import agent_runtime.session_planning as session_planning_runtime
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
    ResumabilityProvider,
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
    NoServiceAvailableError,
    RetryableProviderFailureError,
    TransientAgentError,
    UsageLimitError,
)
from agent_runtime.execution_contracts import (
    CancellationToken,
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
    plan_resumable_session,
)
from tests.runtime_boundary_fakes import (
    ExecutionServiceFake as _ExecutionService,
    ExternalStateResidentPlanningProviderSessionAdapterFake as _ExternalStateResidentPlanningProviderSessionAdapter,
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


class _RoleAwareResidentSeamRunner(_ResidentSeamRunner):
    def __init__(
        self,
        session: _Session,
        observed_roles: list[InvocationRole],
    ) -> None:
        super().__init__(session)
        self._observed_roles = observed_roles

    async def work(
        self,
        role: InvocationRole,
        prompt: str,
        *,
        run_kind: RunKind = RunKind.FRESH,
        session_uuid: str | None = None,
        on_provider_session_id: Any = None,
    ) -> str:
        self._observed_roles.append(role)
        return await super().work(
            role,
            prompt,
            run_kind=run_kind,
            session_uuid=session_uuid,
            on_provider_session_id=on_provider_session_id,
        )


class _RoleAwareResidentSeamExecutionAdapter:
    def __init__(self) -> None:
        self.observed_roles: list[InvocationRole] = []

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
                    _RoleAwareResidentSeamRunner(
                        cast(_Session, session),
                        self.observed_roles,
                    ),
                ),
                get_git_identity=lambda: ("Runtime Test", "runtime@example.com"),
            ),
            failure_handling=WorkFailureHandling(timeout_retries=0),
            presentation=WorkPresentationDependencies(),
        )


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


def test_package_exports_runtime_surface() -> None:
    assert runtime.__all__ == [
        "AgentCredentialFailureError",
        "AgentFailedError",
        "AgentRuntimeError",
        "AgentTimeoutError",
        "Continuation",
        "HardAgentError",
        "ExecutionProvider",
        "InvocationRole",
        "InvocationProgress",
        "ProviderSessionAdapter",
        "RuntimeConfigurationError",
        "RuntimeOutcome",
        "RunKind",
        "StageSelection",
        "ToolAccess",
        "ToolPolicy",
        "ToolPolicyProfile",
        "TransientAgentError",
        "UsageLimitError",
        "UsageLimitScope",
    ]
    assert runtime.StageSelection.__module__.startswith("agent_runtime")
    assert not hasattr(runtime, "StageOverride")
    assert runtime.AgentRuntimeError is AgentRuntimeError
    assert runtime.RuntimeOutcome is prompt_runtime.RuntimeOutcome
    assert not hasattr(runtime, "assert_runtime_import_isolation")
    assert not hasattr(runtime, "run_prompt")
    assert not hasattr(runtime, "ServiceRegistry")
    assert not hasattr(runtime, "ProviderSessionPreferences")
    assert not hasattr(runtime, "ProviderSessionPreferencesRequest")
    assert not hasattr(runtime, "ProviderSessionState")
    assert not hasattr(runtime, "ProviderSessionStateRequest")
    assert not hasattr(prompt_runtime, "PromptRuntime")
    assert not hasattr(prompt_runtime, "PromptRunRequest")
    assert not hasattr(prompt_runtime, "PromptRuntimeExecutionAdapter")
    assert not hasattr(prompt_runtime, "run_ephemeral")
    assert not hasattr(prompt_runtime, "run_prompt")
    assert not hasattr(prompt_runtime, "run_resumable_prompt")
    assert not hasattr(prompt_runtime, "ResidentRunRequest")
    assert not hasattr(prompt_runtime, "ResidentRunResult")
    assert not hasattr(prompt_runtime, "ResidentRuntime")
    assert not hasattr(prompt_runtime, "ResidentRuntimeExecutionAdapter")
    assert not hasattr(prompt_runtime, "ResidentRuntimeMetadata")
    assert {
        "EphemeralRunRequest",
        "EphemeralRunResult",
        "EphemeralResultMetadata",
        "EphemeralRuntime",
        "EphemeralRuntimeExecutionAdapter",
        "EphemeralRuntimeMetadata",
        "Continuation",
        "ResumedSessionRunRequest",
        "SessionRunResult",
        "ResumedSessionRuntime",
        "ResumedSessionRuntimeExecutionAdapter",
        "SessionRuntimeMetadata",
    } <= set(prompt_runtime.__all__)
    assert not hasattr(prompt_runtime, "ResumableRunResult")
    assert not hasattr(prompt_runtime, "ResumableRuntimeMetadata")
    assert "ResumableRunRequest" not in prompt_runtime.__all__
    assert not hasattr(prompt_runtime, "ResumableRunRequest")
    assert "ResumableRuntime" not in prompt_runtime.__all__
    assert "ResumableRuntimeExecutionAdapter" not in prompt_runtime.__all__
    assert "OneShotRunRequest" not in prompt_runtime.__all__
    assert "OneShotRunResult" not in prompt_runtime.__all__
    assert "OneShotResultMetadata" not in prompt_runtime.__all__
    assert "OneShotRuntime" not in prompt_runtime.__all__
    assert "OneShotRuntimeExecutionAdapter" not in prompt_runtime.__all__
    assert "OneShotRuntimeMetadata" not in prompt_runtime.__all__


def test_runtime_star_import_uses_lifecycle_surface_while_removed_legacy_aliases_fail_direct_import() -> (
    None
):
    exported_names: dict[str, object] = {}

    exec("from agent_runtime.runtime import *", {}, exported_names)

    assert "EphemeralRuntime" in exported_names
    assert "EphemeralRunRequest" in exported_names
    assert "ResumedSessionRuntime" in exported_names
    assert "ResumedSessionRunRequest" in exported_names
    assert "ResumableRuntime" not in exported_names
    assert "ResumableRunRequest" not in exported_names
    assert "OneShotRuntime" not in exported_names
    assert "OneShotRunRequest" not in exported_names
    with pytest.raises(ImportError):
        exec("from agent_runtime.runtime import ResumableRunRequest", {}, {})
    with pytest.raises(ImportError):
        exec("from agent_runtime.runtime import OneShotRuntime", {}, {})
    with pytest.raises(ImportError):
        exec("from agent_runtime.runtime import OneShotRunRequest", {}, {})


def test_runtime_direct_import_rejects_removed_legacy_names() -> None:
    with pytest.raises(AttributeError):
        getattr(prompt_runtime, "OneShotRuntime")
    with pytest.raises(AttributeError):
        getattr(prompt_runtime, "OneShotRunRequest")
    with pytest.raises(AttributeError):
        getattr(prompt_runtime, "OneShotRunResult")
    with pytest.raises(AttributeError):
        getattr(prompt_runtime, "OneShotResultMetadata")
    with pytest.raises(AttributeError):
        getattr(prompt_runtime, "OneShotRuntimeExecutionAdapter")
    with pytest.raises(AttributeError):
        getattr(prompt_runtime, "OneShotRuntimeMetadata")


def test_runtime_direct_import_rejects_removed_resumable_completed_result_names() -> (
    None
):
    with pytest.raises(AttributeError):
        getattr(prompt_runtime, "ResumableRuntime")
    with pytest.raises(AttributeError):
        getattr(prompt_runtime, "ResumableRuntimeExecutionAdapter")
    with pytest.raises(ImportError):
        exec("from agent_runtime.runtime import ResumableRuntime", {}, {})
    with pytest.raises(ImportError):
        exec(
            "from agent_runtime.runtime import ResumableRuntimeExecutionAdapter", {}, {}
        )
    with pytest.raises(ImportError):
        exec("from agent_runtime.runtime import ResumableRunResult", {}, {})
    with pytest.raises(ImportError):
        exec("from agent_runtime.runtime import ResumableRuntimeMetadata", {}, {})


def test_types_module_exposes_stage_selection_as_the_only_stage_chain_value() -> None:
    types_module = importlib.import_module("agent_runtime.types")

    assert types_module.StageSelection.__module__.startswith("agent_runtime")
    assert not hasattr(types_module, "StageOverride")
    with pytest.raises(ImportError, match="StageOverride"):
        exec("from agent_runtime.types import StageOverride", {})


def test_runtime_surface_exposes_resumed_session_lifecycle_names() -> None:
    assert {
        "NewSessionRunRequest",
        "NewSessionRuntime",
        "ResumedSessionRunRequest",
        "ResumedSessionRuntime",
    } <= set(prompt_runtime.__all__)
    assert hasattr(prompt_runtime, "ResumedSessionRunRequest")
    assert hasattr(prompt_runtime, "ResumedSessionRuntime")
    assert prompt_runtime.ResumedSessionRunRequest.__name__ == (
        "ResumedSessionRunRequest"
    )


def test_resumed_session_runtime_exposes_only_lifecycle_resume_method() -> None:
    runtime_instance = prompt_runtime.ResumedSessionRuntime(
        execution_adapter=cast(
            prompt_runtime.ResumedSessionRuntimeExecutionAdapter,
            object(),
        )
    )

    assert hasattr(runtime_instance, "run_resumed_session")
    assert not hasattr(runtime_instance, "run_resumable_prompt")


def test_readme_guides_consumers_to_lifecycle_session_entrypoints() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "ResumedSessionRuntime" in readme
    assert "ResumedSessionRunRequest" in readme
    assert "run_resumed_session" in readme
    assert "run_resumable_prompt" not in readme


def test_contracts_expose_execution_provider_as_canonical_public_protocol_name() -> (
    None
):
    contracts = importlib.import_module("agent_runtime.contracts")

    assert "ExecutionProvider" in contracts.__all__
    assert "ResumableExecutionProvider" in contracts.__all__
    assert not hasattr(contracts, "ExecutionService")
    assert not hasattr(contracts, "ResidentExecutionProvider")
    assert runtime.ExecutionProvider is contracts.ExecutionProvider


def test_session_planning_surface_uses_resumable_vocabulary() -> None:
    assert not hasattr(session_planning_runtime, "ResidentSessionPlan")
    assert not hasattr(session_planning_runtime, "ResidentSessionPlanRequest")
    assert not hasattr(session_planning_runtime, "plan_resident_session")
    assert {
        "ResumableSessionPlan",
        "ResumableSessionPlanRequest",
        "plan_resumable_session",
    } <= set(session_planning_runtime.__all__)


def test_provider_session_planning_surface_exposes_immutable_decision_only() -> None:
    assert session_planning_runtime.ProviderSessionPlanRequest.__name__ == (
        "ProviderSessionPlanRequest"
    )
    assert not hasattr(session_planning_runtime, "ProviderRunStatePlan")
    assert not hasattr(session_planning_runtime, "plan_provider_run_state")
    assert not hasattr(
        session_planning_runtime,
        "record_observed_provider_session_id",
    )
    assert not hasattr(
        session_planning_runtime,
        "record_successful_provider_session_metadata",
    )
    assert {
        "ProviderSessionDecision",
        "ProviderSessionPlanRequest",
        "plan_provider_session",
    } <= set(session_planning_runtime.__all__)


def test_provider_session_planning_returns_immutable_decision_value(
    execution_service_factory: Callable[..., ExecutionProvider],
    session_store_factory: Callable[..., _SessionStore],
    resident_provider_session_adapter: _ResidentPlanningProviderSessionAdapter,
) -> None:
    provider_session_decision = session_planning_runtime.plan_provider_session(
        session_planning_runtime.ProviderSessionPlanRequest(
            worktree=Path("."),
            role=InvocationRole("implementer"),
            namespace="main",
            resumability_service=cast(
                ResumabilityProvider, execution_service_factory()
            ),
            session_store=session_store_factory(),
            provider_session_adapter=resident_provider_session_adapter,
        )
    )

    assert (
        provider_session_decision
        == session_planning_runtime.ProviderSessionDecision(
            run_kind=RunKind.RESUME,
            provider_session_id="recovered-session",
            state_dir_relpath="state/",
            state_dir_path=Path("state"),
            recovered_session_id_persistence=(
                session_planning_runtime.RecoveredSessionIdPersistence.SKIP
            ),
            service_state_dir=Path("state"),
            exact_transcript_match=False,
            auth_seeding_requirement=AuthSeedingRequirement.NOT_REQUIRED,
            auth_seed_action=None,
            use_service_state_dir_for_container=False,
        )
    )
    with pytest.raises(FrozenInstanceError):
        setattr(provider_session_decision, "provider_session_id", "other-session")


def test_resumable_session_plan_exposes_public_value_fields_only(
    execution_service_factory: Callable[..., ExecutionProvider],
    session_store_factory: Callable[..., _SessionStore],
    resident_provider_session_adapter: _ResidentPlanningProviderSessionAdapter,
) -> None:
    service = execution_service_factory()

    session_plan = plan_resumable_session(
        ResumableSessionPlanRequest(
            worktree=Path("."),
            role=InvocationRole("implementer"),
            namespace="main",
            service=service,
            session_store=session_store_factory(),
            provider_session_adapter=resident_provider_session_adapter,
        )
    )

    assert session_plan.role == InvocationRole("implementer")
    assert session_plan.worktree == Path(".")
    assert session_plan.namespace == "main"
    assert session_plan.service is service
    assert session_plan.run_kind is RunKind.RESUME
    assert session_plan.provider_state_dir == Path("state")
    assert session_plan.provider_session_id == "recovered-session"
    assert session_plan.auth_seeding_requirement is AuthSeedingRequirement.NOT_REQUIRED
    assert session_plan.auth_seed_action is None
    assert session_plan.exact_transcript_match is False
    assert session_plan.usage_limit_scope is None
    with pytest.raises(FrozenInstanceError):
        setattr(session_plan, "provider_state_dir", Path("other-state"))


def test_resumable_session_plan_hides_container_state_selection_metadata(
    execution_service_factory: Callable[..., ExecutionProvider],
    session_store_factory: Callable[..., _SessionStore],
    resident_provider_session_adapter: _ResidentPlanningProviderSessionAdapter,
) -> None:
    service = execution_service_factory()

    session_plan = plan_resumable_session(
        ResumableSessionPlanRequest(
            worktree=Path("."),
            role=InvocationRole("implementer"),
            namespace="main",
            service=service,
            session_store=session_store_factory(),
            provider_session_adapter=resident_provider_session_adapter,
        )
    )

    field_names = {field.name for field in fields(session_plan)}

    assert "service_state_dir" not in field_names
    assert "use_service_state_dir_for_container" not in field_names


def test_provider_session_dtos_remain_on_focused_session_seam() -> None:
    assert session_runtime.ProviderSessionState.__module__ == "agent_runtime.session"
    assert (
        session_runtime.ProviderSessionStateRequest.__module__
        == "agent_runtime.session"
    )


def test_provider_session_seams_consolidate_public_session_store_vocabulary() -> None:
    assert "SessionStore" in session_runtime.__all__
    assert not hasattr(session_runtime, "ServiceResumeIdentityStore")
    assert not hasattr(
        importlib.import_module("agent_runtime.contracts"),
        "ProviderSessionRecordingStore",
    )


def test_provider_session_adapter_public_seam_stays_narrow() -> None:
    assert provider_session_adapter_runtime.__all__ == [
        "ProviderSessionAdapter",
        "ProviderSessionPlanningFacts",
        "ProviderSessionPlanningRequest",
    ]
    adapter_members = provider_session_adapter_runtime.ProviderSessionAdapter.__dict__

    assert "provider_session_planning_facts" in adapter_members
    assert "provider_session_state" in adapter_members
    assert "prepare_local_provider_run_state" in adapter_members
    assert "record_provider_session_id" in adapter_members
    assert "provider_session_preferences" not in adapter_members
    assert "recover_provider_session_id" not in adapter_members
    assert "is_exact_resumable_provider_session" not in adapter_members
    assert not hasattr(provider_session_adapter_runtime, "ProviderSessionService")


def test_provider_session_public_dtos_expose_only_runtime_planning_fields() -> None:
    assert [
        field.name for field in fields(session_runtime.ProviderSessionStateRequest)
    ] == [
        "session_store",
        "provider_state_dir",
        "has_resumable_provider_state",
        "state_dir_relpath",
        "require_exact_transcript_match",
    ]
    assert [field.name for field in fields(session_runtime.ProviderSessionState)] == [
        "run_kind",
        "provider_session_id",
        "state_dir_relpath",
        "state_dir_path",
        "exact_transcript_match",
        "persist_provider_session_id",
        "auth_seeding_requirement",
        "auth_seed_action",
        "use_service_state_dir_for_container",
    ]


def test_package_surface_exposes_invocation_role_value_object() -> None:
    role = runtime.InvocationRole("implementer")

    assert role.value == "implementer"


def test_package_surface_exposes_usage_limit_scope_value_object() -> None:
    usage_limit_scope = runtime.UsageLimitScope("quota-review")

    assert usage_limit_scope.value == "quota-review"


def test_tool_policy_restricted_resolves_to_provider_neutral_profile() -> None:
    profile = runtime.ToolPolicy.RESTRICTED.profile

    assert profile.allowed_tools == ("Read", "Glob")
    assert profile.disallowed_tools == ()
    assert profile.strict_mcp_config is True


def test_runtime_surface_exposes_tool_policy_profiles_for_partial_and_full() -> None:
    partial = runtime.ToolPolicy.PARTIAL.profile
    full = runtime.ToolPolicy.FULL.profile

    assert isinstance(partial, prompt_runtime.ToolPolicyProfile)
    assert partial.allowed_tools is None
    assert partial.disallowed_tools == ("Edit", "Write", "NotebookEdit")
    assert partial.strict_mcp_config is True
    assert isinstance(full, prompt_runtime.ToolPolicyProfile)
    assert full.allowed_tools is None
    assert full.disallowed_tools == ()
    assert full.strict_mcp_config is True


def test_tool_policy_profiles_stay_provider_neutral() -> None:
    for policy in runtime.ToolPolicy:
        profile = policy.profile
        rendered_values = (profile.allowed_tools or ()) + profile.disallowed_tools

        assert profile.strict_mcp_config is True
        assert all(not value.startswith("-") for value in rendered_values)
        assert all(
            provider not in value.lower()
            for value in rendered_values
            for provider in ("claude", "codex", "opencode")
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


def test_ephemeral_runtime_returns_completed_outcome_with_selected_runtime_metadata_and_tool_access(
    stage_selection_factory: Callable[..., runtime.StageSelection],
    service_registry_factory: Callable[..., ServiceRegistry],
) -> None:
    tool_access = runtime.ToolAccess.workspace_backed(
        Path("/repo"),
        tool_policy=runtime.ToolPolicy.PARTIAL,
    )

    result = asyncio.run(
        prompt_runtime.EphemeralRuntime(
            execution_adapter=_ToolPolicyRenderingEphemeralExecutionAdapter(),
            service_registry=service_registry_factory("claude"),
        ).run_ephemeral(
            prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                worktree=Path("/repo"),
                stage=stage_selection_factory(
                    service="claude",
                    model="gpt-5",
                    effort="medium",
                ),
                role=InvocationRole("implementer"),
                tool_access=tool_access,
            )
        )
    )

    assert result == prompt_runtime.RuntimeOutcome.completed(
        output=_tool_policy_effect_text(runtime.ToolPolicy.PARTIAL),
        result=prompt_runtime.EphemeralRunResult(
            output=_tool_policy_effect_text(runtime.ToolPolicy.PARTIAL),
            selected_service="claude",
            selected_model="gpt-5",
            selected_effort="medium",
            tool_access=tool_access,
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
    assert result.tool_access == tool_access


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


def test_resumed_session_runtime_preserves_resumed_session_behavior_through_run_session_seam(
    execution_service_factory: Callable[..., ExecutionProvider],
    session_store_factory: Callable[..., _SessionStore],
    resident_provider_session_adapter: _ResidentPlanningProviderSessionAdapter,
) -> None:
    service = execution_service_factory()
    session_plan = plan_resumable_session(
        ResumableSessionPlanRequest(
            worktree=Path("."),
            role=InvocationRole("implementer"),
            namespace="main",
            service=service,
            session_store=session_store_factory(),
            provider_session_adapter=resident_provider_session_adapter,
        )
    )

    assert session_plan == ResumableSessionPlan(
        role=InvocationRole("implementer"),
        worktree=Path("."),
        namespace="main",
        service=service,
        run_kind=RunKind.RESUME,
        provider_state_dir=Path("state"),
        provider_session_id="recovered-session",
        auth_seeding_requirement=AuthSeedingRequirement.NOT_REQUIRED,
    )

    result = asyncio.run(
        prompt_runtime.ResumedSessionRuntime(
            execution_adapter=_RuntimePlannedPathResidentExecutionAdapter()
        ).run_resumed_session(
            prompt_runtime.ResumedSessionRunRequest(
                prompt="already rendered prompt",
                worktree=WorktreeMount(Path(".")),
                model="gpt-5.4",
                effort="medium",
                session_plan=session_plan,
                tool_policy=runtime.ToolPolicy.FULL,
            )
        )
    )

    assert result == prompt_runtime.RuntimeOutcome.completed(
        output="resume:prepared:recovered-session:/workspace/state/",
        result=prompt_runtime.SessionRunResult(
            output="resume:prepared:recovered-session:/workspace/state/",
            runtime_metadata=prompt_runtime.SessionRuntimeMetadata(
                service_name="codex",
                provider_session_id="prepared:recovered-session",
                run_kind=RunKind.RESUME,
                session_namespace="main",
                exact_transcript_match=False,
            ),
        ),
    )


def test_resumed_session_runtime_uses_invocation_role_from_session_plan(
    execution_service_factory: Callable[..., ExecutionProvider],
    session_store_factory: Callable[..., _SessionStore],
    resident_provider_session_adapter: _ResidentPlanningProviderSessionAdapter,
) -> None:
    role = InvocationRole("reviewer")
    service = execution_service_factory()
    session_plan = plan_resumable_session(
        ResumableSessionPlanRequest(
            worktree=Path("."),
            role=role,
            namespace="main",
            service=service,
            session_store=session_store_factory(),
            provider_session_adapter=resident_provider_session_adapter,
        )
    )
    execution_adapter = _RoleAwareResidentSeamExecutionAdapter()

    asyncio.run(
        prompt_runtime.ResumedSessionRuntime(
            execution_adapter=execution_adapter
        ).run_resumed_session(
            prompt_runtime.ResumedSessionRunRequest(
                prompt="already rendered prompt",
                worktree=WorktreeMount(Path(".")),
                model="gpt-5.4",
                effort="medium",
                session_plan=session_plan,
                tool_policy=runtime.ToolPolicy.FULL,
            )
        )
    )

    assert execution_adapter.observed_roles == [role]


def test_resumed_session_runtime_preserves_planned_relative_provider_state_path(
    execution_service_factory: Callable[..., ExecutionProvider],
    session_store_factory: Callable[..., _SessionStore],
    external_state_provider_session_adapter: _ExternalStateResidentPlanningProviderSessionAdapter,
) -> None:
    worktree = Path("/repo")
    service = execution_service_factory()
    provider_session_decision = session_planning_runtime.plan_provider_session(
        session_planning_runtime.ProviderSessionPlanRequest(
            worktree=worktree,
            role=InvocationRole("implementer"),
            namespace="main",
            resumability_service=cast(ResumabilityProvider, service),
            session_store=session_store_factory(),
            provider_session_adapter=external_state_provider_session_adapter,
        )
    )
    session_plan = plan_resumable_session(
        ResumableSessionPlanRequest(
            worktree=worktree,
            role=InvocationRole("implementer"),
            namespace="main",
            service=service,
            session_store=session_store_factory(),
            provider_session_adapter=external_state_provider_session_adapter,
        )
    )

    assert (
        provider_session_decision
        == session_planning_runtime.ProviderSessionDecision(
            run_kind=RunKind.RESUME,
            provider_session_id="recovered-session",
            state_dir_relpath="runtime-state/",
            state_dir_path=Path("/host/runtime-state"),
            recovered_session_id_persistence=(
                session_planning_runtime.RecoveredSessionIdPersistence.SKIP
            ),
            service_state_dir=Path("/host/runtime-state"),
            exact_transcript_match=False,
            auth_seeding_requirement=AuthSeedingRequirement.NOT_REQUIRED,
            auth_seed_action=None,
            use_service_state_dir_for_container=False,
        )
    )
    assert session_plan.provider_state_dir == Path("/host/runtime-state")
    assert session_plan.provider_session_id == "recovered-session"
    assert (
        provider_session_decision.container_state_dir_path(
            worktree=worktree,
            container_workspace="/workspace",
        )
        == "/workspace/runtime-state/"
    )

    result = asyncio.run(
        prompt_runtime.ResumedSessionRuntime(
            execution_adapter=_RuntimePlannedPathResidentExecutionAdapter()
        ).run_resumed_session(
            prompt_runtime.ResumedSessionRunRequest(
                prompt="already rendered prompt",
                worktree=WorktreeMount(worktree),
                model="gpt-5.4",
                effort="medium",
                session_plan=session_plan,
                tool_policy=runtime.ToolPolicy.FULL,
            )
        )
    )

    assert result == prompt_runtime.RuntimeOutcome.completed(
        output="resume:prepared:recovered-session:/workspace/runtime-state/",
        result=prompt_runtime.SessionRunResult(
            output="resume:prepared:recovered-session:/workspace/runtime-state/",
            runtime_metadata=prompt_runtime.SessionRuntimeMetadata(
                service_name="codex",
                provider_session_id="prepared:recovered-session",
                run_kind=RunKind.RESUME,
                session_namespace="main",
                exact_transcript_match=False,
            ),
        ),
    )


def test_resumed_session_runtime_returns_portable_continuation_resume_data(
    execution_service_factory: Callable[..., ExecutionProvider],
    session_store_factory: Callable[..., _SessionStore],
    external_state_provider_session_adapter: _ExternalStateResidentPlanningProviderSessionAdapter,
) -> None:
    worktree = Path("/repo")
    service = execution_service_factory()
    session_plan = plan_resumable_session(
        ResumableSessionPlanRequest(
            worktree=worktree,
            role=InvocationRole("implementer"),
            namespace="main",
            service=service,
            session_store=session_store_factory(),
            provider_session_adapter=external_state_provider_session_adapter,
        )
    )
    tool_access = runtime.ToolAccess.workspace_backed(
        worktree,
        tool_policy=runtime.ToolPolicy.PARTIAL,
    )

    result = asyncio.run(
        prompt_runtime.ResumedSessionRuntime(
            execution_adapter=_RuntimePlannedPathResidentExecutionAdapter()
        ).run_resumed_session(
            prompt_runtime.ResumedSessionRunRequest(
                prompt="already rendered prompt",
                worktree=WorktreeMount(worktree),
                model="gpt-5.4",
                effort="medium",
                session_plan=session_plan,
                tool_access=tool_access,
            )
        )
    )

    assert hasattr(runtime, "Continuation")
    assert hasattr(prompt_runtime, "Continuation")
    assert runtime.Continuation is prompt_runtime.Continuation
    assert result.result is not None
    assert getattr(result.result, "continuation") == prompt_runtime.Continuation(
        selected_service="codex",
        selected_model="gpt-5.4",
        selected_effort="medium",
        tool_access=tool_access,
        provider_resume_state={
            "run_kind": "resume",
            "provider_session_id": "prepared:recovered-session",
            "provider_state_dir_relpath": "runtime-state/",
            "exact_transcript_match": False,
        },
    )


def test_new_session_runtime_selects_fallback_service_before_binding_continuation(
    stage_selection_factory: Callable[..., runtime.StageSelection],
    service_registry_factory: Callable[..., ServiceRegistry],
    session_store_factory: Callable[..., _SessionStore],
) -> None:
    worktree = Path("/repo")
    tool_access = runtime.ToolAccess.workspace_backed(
        worktree,
        tool_policy=runtime.ToolPolicy.PARTIAL,
    )

    result = asyncio.run(
        prompt_runtime.NewSessionRuntime(
            execution_adapter=_RuntimePlannedPathResidentExecutionAdapter(),
            service_registry=service_registry_factory("claude"),
        ).run_new_session(
            prompt_runtime.NewSessionRunRequest(
                prompt="already rendered prompt",
                worktree=WorktreeMount(worktree),
                stage=stage_selection_factory(
                    service="missing",
                    fallback=stage_selection_factory(
                        service="claude",
                        model="sonnet",
                        effort="high",
                    ),
                ),
                role=InvocationRole("implementer"),
                session_namespace="main",
                session_store=session_store_factory(),
                provider_session_adapter=_NamedExternalStateResidentPlanningProviderSessionAdapter(
                    "claude"
                ),
                tool_access=tool_access,
            )
        )
    )

    assert (
        result.output
        == "resume:prepared:recovered-claude:/workspace/claude-runtime-state/"
    )
    assert isinstance(result.result, prompt_runtime.SessionRunResult)
    assert result.result.runtime_metadata.service_name == "claude"
    assert result.result.continuation == prompt_runtime.Continuation(
        selected_service="claude",
        selected_model="sonnet",
        selected_effort="high",
        tool_access=tool_access,
        provider_resume_state={
            "run_kind": "resume",
            "provider_session_id": "prepared:recovered-claude",
            "provider_state_dir_relpath": "claude-runtime-state/",
            "exact_transcript_match": False,
        },
    )


def test_new_session_runtime_retries_fallback_before_binding_continuation(
    stage_selection_factory: Callable[..., runtime.StageSelection],
    service_registry_factory: Callable[..., ServiceRegistry],
    session_store_factory: Callable[..., _SessionStore],
) -> None:
    worktree = Path("/repo")
    tool_access = runtime.ToolAccess.workspace_backed(
        worktree,
        tool_policy=runtime.ToolPolicy.PARTIAL,
    )

    result = asyncio.run(
        prompt_runtime.NewSessionRuntime(
            execution_adapter=_UsageLimitedThenFallbackNewSessionExecutionAdapter(),
            service_registry=service_registry_factory("codex", "claude"),
        ).run_new_session(
            prompt_runtime.NewSessionRunRequest(
                prompt="already rendered prompt",
                worktree=WorktreeMount(worktree),
                stage=stage_selection_factory(
                    service="codex",
                    model="gpt-5.4",
                    effort="medium",
                    fallback=stage_selection_factory(
                        service="claude",
                        model="sonnet",
                        effort="high",
                    ),
                ),
                role=InvocationRole("implementer"),
                session_namespace="main",
                session_store=session_store_factory(),
                provider_session_adapter=_NamedExternalStateResidentPlanningProviderSessionAdapter(
                    "claude"
                ),
                tool_access=tool_access,
            )
        )
    )

    assert result == prompt_runtime.RuntimeOutcome.completed(
        output="resume:prepared:recovered-claude:/workspace/claude-runtime-state/",
        result=prompt_runtime.SessionRunResult(
            output="resume:prepared:recovered-claude:/workspace/claude-runtime-state/",
            runtime_metadata=prompt_runtime.SessionRuntimeMetadata(
                service_name="claude",
                provider_session_id="prepared:recovered-claude",
                run_kind=RunKind.RESUME,
                session_namespace="main",
                exact_transcript_match=False,
            ),
        ),
    )
    assert result.result is not None
    assert isinstance(result.result, prompt_runtime.SessionRunResult)
    assert result.result.continuation == prompt_runtime.Continuation(
        selected_service="claude",
        selected_model="sonnet",
        selected_effort="high",
        tool_access=tool_access,
        provider_resume_state={
            "run_kind": "resume",
            "provider_session_id": "prepared:recovered-claude",
            "provider_state_dir_relpath": "claude-runtime-state/",
            "exact_transcript_match": False,
        },
    )


def test_new_session_runtime_keeps_started_usage_limit_outcome(
    stage_selection_factory: Callable[..., runtime.StageSelection],
    service_registry_factory: Callable[..., ServiceRegistry],
    session_store_factory: Callable[..., _SessionStore],
) -> None:
    worktree = Path("/repo")
    tool_access = runtime.ToolAccess.workspace_backed(
        worktree,
        tool_policy=runtime.ToolPolicy.PARTIAL,
    )

    result = asyncio.run(
        prompt_runtime.NewSessionRuntime(
            execution_adapter=_StartedUsageLimitNewSessionExecutionAdapter(),
            service_registry=service_registry_factory("codex", "claude"),
        ).run_new_session(
            prompt_runtime.NewSessionRunRequest(
                prompt="already rendered prompt",
                worktree=WorktreeMount(worktree),
                stage=stage_selection_factory(
                    service="codex",
                    model="gpt-5.4",
                    effort="medium",
                    fallback=stage_selection_factory(
                        service="claude",
                        model="sonnet",
                        effort="high",
                    ),
                ),
                role=InvocationRole("implementer"),
                session_namespace="main",
                session_store=session_store_factory(),
                provider_session_adapter=_NamedExternalStateResidentPlanningProviderSessionAdapter(
                    "codex"
                ),
                tool_access=tool_access,
            )
        )
    )

    assert result == prompt_runtime.RuntimeOutcome.usage_limited(
        output="",
        service_name="codex",
        reset_time=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
        usage_limit_scope=runtime.UsageLimitScope("implementer"),
        invocation_progress=runtime.InvocationProgress.STARTED,
        continuation=prompt_runtime.Continuation(
            selected_service="codex",
            selected_model="gpt-5.4",
            selected_effort="medium",
            tool_access=tool_access,
            provider_resume_state={
                "run_kind": "resume",
                "provider_session_id": "prepared:recovered-codex",
                "provider_state_dir_relpath": "codex-runtime-state/",
                "exact_transcript_match": False,
            },
        ),
    )


def test_new_session_runtime_returns_continuation_for_started_interruption(
    stage_selection_factory: Callable[..., runtime.StageSelection],
    service_registry_factory: Callable[..., ServiceRegistry],
    session_store_factory: Callable[..., _SessionStore],
) -> None:
    worktree = Path("/repo")
    tool_access = runtime.ToolAccess.workspace_backed(
        worktree,
        tool_policy=runtime.ToolPolicy.PARTIAL,
    )

    result = asyncio.run(
        prompt_runtime.NewSessionRuntime(
            execution_adapter=_StartedUsageLimitNewSessionExecutionAdapter(),
            service_registry=service_registry_factory("codex"),
        ).run_new_session(
            prompt_runtime.NewSessionRunRequest(
                prompt="already rendered prompt",
                worktree=WorktreeMount(worktree),
                stage=stage_selection_factory(
                    service="codex",
                    model="gpt-5.4",
                    effort="medium",
                ),
                role=InvocationRole("implementer"),
                session_namespace="main",
                session_store=session_store_factory(),
                provider_session_adapter=_NamedExternalStateResidentPlanningProviderSessionAdapter(
                    "codex"
                ),
                tool_access=tool_access,
            )
        )
    )

    assert result.continuation == prompt_runtime.Continuation(
        selected_service="codex",
        selected_model="gpt-5.4",
        selected_effort="medium",
        tool_access=tool_access,
        provider_resume_state={
            "run_kind": "resume",
            "provider_session_id": "prepared:recovered-codex",
            "provider_state_dir_relpath": "codex-runtime-state/",
            "exact_transcript_match": False,
        },
    )


def test_new_session_runtime_returns_adapter_owned_provider_resume_state(
    stage_selection_factory: Callable[..., runtime.StageSelection],
    service_registry_factory: Callable[..., ServiceRegistry],
    session_store_factory: Callable[..., _SessionStore],
) -> None:
    worktree = Path("/repo")
    tool_access = runtime.ToolAccess.workspace_backed(
        worktree,
        tool_policy=runtime.ToolPolicy.PARTIAL,
    )
    adapter_resume_state = {
        "adapter_session": {"id": "prepared:recovered-codex", "phase": "ready"},
        "workspace_mount": "/workspace/codex-runtime-state/",
        "attempts": [1, 2],
    }

    @dataclass
    class _AdapterOwnedPreparedRunSession:
        provider_state_dir_container_path: str | None
        run_kind: RunKind
        provider_session_id: str | None
        provider_resume_state: dict[str, Any]

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

    class _AdapterOwnedResumeStateExecutionAdapter:
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

            def _prepare_session(
                run_session: Any,
            ) -> _AdapterOwnedPreparedRunSession:
                return _AdapterOwnedPreparedRunSession(
                    provider_state_dir_container_path="/workspace/codex-runtime-state/",
                    run_kind=run_session.run_kind,
                    provider_session_id="prepared:recovered-codex",
                    provider_resume_state=adapter_resume_state,
                )

            return WorkInvocationDependencies(
                execution=WorkExecutionDependencies(
                    container_workspace="/workspace",
                    prepare_session=cast(Any, _prepare_session),
                    build_session=lambda mount_path, service, provider_state_dir: (
                        _Session(provider_state_dir)
                    ),
                    build_runner=lambda session, status_display: cast(
                        WorkExecutionAdapter,
                        _ResidentSeamRunner(cast(_Session, session)),
                    ),
                    get_git_identity=lambda: ("Runtime Test", "runtime@example.com"),
                ),
                failure_handling=WorkFailureHandling(timeout_retries=0),
                presentation=WorkPresentationDependencies(),
            )

    result = asyncio.run(
        prompt_runtime.NewSessionRuntime(
            execution_adapter=_AdapterOwnedResumeStateExecutionAdapter(),
            service_registry=service_registry_factory("codex"),
        ).run_new_session(
            prompt_runtime.NewSessionRunRequest(
                prompt="already rendered prompt",
                worktree=WorktreeMount(worktree),
                stage=stage_selection_factory(
                    service="codex",
                    model="gpt-5.4",
                    effort="medium",
                ),
                role=InvocationRole("implementer"),
                session_namespace="main",
                session_store=session_store_factory(),
                provider_session_adapter=_NamedExternalStateResidentPlanningProviderSessionAdapter(
                    "codex"
                ),
                tool_access=tool_access,
            )
        )
    )

    assert isinstance(result.result, prompt_runtime.SessionRunResult)
    assert result.result.continuation == prompt_runtime.Continuation(
        selected_service="codex",
        selected_model="gpt-5.4",
        selected_effort="medium",
        tool_access=tool_access,
        provider_resume_state=adapter_resume_state,
    )


def test_new_session_runtime_reports_not_started_progress_without_continuation(
    stage_selection_factory: Callable[..., runtime.StageSelection],
    service_registry_factory: Callable[..., ServiceRegistry],
    session_store_factory: Callable[..., _SessionStore],
) -> None:
    worktree = Path("/repo")
    tool_access = runtime.ToolAccess.workspace_backed(
        worktree,
        tool_policy=runtime.ToolPolicy.PARTIAL,
    )

    result = asyncio.run(
        prompt_runtime.NewSessionRuntime(
            execution_adapter=_PreparedNotStartedUsageLimitNewSessionExecutionAdapter(),
            service_registry=service_registry_factory("codex"),
        ).run_new_session(
            prompt_runtime.NewSessionRunRequest(
                prompt="already rendered prompt",
                worktree=WorktreeMount(worktree),
                stage=stage_selection_factory(
                    service="codex",
                    model="gpt-5.4",
                    effort="medium",
                ),
                role=InvocationRole("implementer"),
                session_namespace="main",
                session_store=session_store_factory(),
                provider_session_adapter=_NamedExternalStateResidentPlanningProviderSessionAdapter(
                    "codex"
                ),
                tool_access=tool_access,
            )
        )
    )

    assert result == prompt_runtime.RuntimeOutcome.no_service_available(
        output="",
        reset_time=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
        usage_limit_scope=runtime.UsageLimitScope("implementer"),
        invocation_progress=runtime.InvocationProgress.NOT_STARTED,
    )
    assert result.continuation is None


def test_new_session_runtime_does_not_create_continuation_from_session_allocation_alone(
    stage_selection_factory: Callable[..., runtime.StageSelection],
    service_registry_factory: Callable[..., ServiceRegistry],
    session_store_factory: Callable[..., _SessionStore],
) -> None:
    worktree = Path("/repo")
    tool_access = runtime.ToolAccess.workspace_backed(
        worktree,
        tool_policy=runtime.ToolPolicy.PARTIAL,
    )
    execution_adapter = _PreparedNotStartedUsageLimitNewSessionExecutionAdapter()

    result = asyncio.run(
        prompt_runtime.NewSessionRuntime(
            execution_adapter=execution_adapter,
            service_registry=service_registry_factory("codex"),
        ).run_new_session(
            prompt_runtime.NewSessionRunRequest(
                prompt="already rendered prompt",
                worktree=WorktreeMount(worktree),
                stage=stage_selection_factory(
                    service="codex",
                    model="gpt-5.4",
                    effort="medium",
                ),
                role=InvocationRole("implementer"),
                session_namespace="main",
                session_store=session_store_factory(),
                provider_session_adapter=_NamedExternalStateResidentPlanningProviderSessionAdapter(
                    "codex"
                ),
                tool_access=tool_access,
            )
        )
    )

    assert execution_adapter.prepare_session_calls == 1
    assert result.invocation_progress is runtime.InvocationProgress.NOT_STARTED
    assert result.continuation is None


def test_new_session_runtime_returns_continuation_for_started_cancellation(
    stage_selection_factory: Callable[..., runtime.StageSelection],
    service_registry_factory: Callable[..., ServiceRegistry],
    session_store_factory: Callable[..., _SessionStore],
) -> None:
    worktree = Path("/repo")
    tool_access = runtime.ToolAccess.workspace_backed(
        worktree,
        tool_policy=runtime.ToolPolicy.PARTIAL,
    )

    result = asyncio.run(
        prompt_runtime.NewSessionRuntime(
            execution_adapter=_StartedCancellationNewSessionExecutionAdapter(),
            service_registry=service_registry_factory("codex"),
        ).run_new_session(
            prompt_runtime.NewSessionRunRequest(
                prompt="already rendered prompt",
                worktree=WorktreeMount(worktree),
                stage=stage_selection_factory(
                    service="codex",
                    model="gpt-5.4",
                    effort="medium",
                ),
                role=InvocationRole("implementer"),
                session_namespace="main",
                session_store=session_store_factory(),
                provider_session_adapter=_NamedExternalStateResidentPlanningProviderSessionAdapter(
                    "codex"
                ),
                tool_access=tool_access,
            )
        )
    )

    assert result == prompt_runtime.RuntimeOutcome.cancelled(
        output="",
        invocation_progress=runtime.InvocationProgress.STARTED,
        continuation=prompt_runtime.Continuation(
            selected_service="codex",
            selected_model="gpt-5.4",
            selected_effort="medium",
            tool_access=tool_access,
            provider_resume_state={
                "run_kind": "resume",
                "provider_session_id": "prepared:recovered-codex",
                "provider_state_dir_relpath": "codex-runtime-state/",
                "exact_transcript_match": False,
            },
        ),
    )


def test_new_session_runtime_keeps_not_started_cancellation_without_continuation(
    stage_selection_factory: Callable[..., runtime.StageSelection],
    service_registry_factory: Callable[..., ServiceRegistry],
    session_store_factory: Callable[..., _SessionStore],
) -> None:
    worktree = Path("/repo")
    tool_access = runtime.ToolAccess.workspace_backed(
        worktree,
        tool_policy=runtime.ToolPolicy.PARTIAL,
    )
    execution_adapter = _PreparedNotStartedCancellationNewSessionExecutionAdapter()

    result = asyncio.run(
        prompt_runtime.NewSessionRuntime(
            execution_adapter=execution_adapter,
            service_registry=service_registry_factory("codex"),
        ).run_new_session(
            prompt_runtime.NewSessionRunRequest(
                prompt="already rendered prompt",
                worktree=WorktreeMount(worktree),
                stage=stage_selection_factory(
                    service="codex",
                    model="gpt-5.4",
                    effort="medium",
                ),
                role=InvocationRole("implementer"),
                session_namespace="main",
                session_store=session_store_factory(),
                provider_session_adapter=_NamedExternalStateResidentPlanningProviderSessionAdapter(
                    "codex"
                ),
                tool_access=tool_access,
            )
        )
    )

    assert execution_adapter.prepare_session_calls == 1
    assert result == prompt_runtime.RuntimeOutcome.cancelled(
        output="",
        invocation_progress=runtime.InvocationProgress.NOT_STARTED,
    )
    assert result.continuation is None


def test_new_session_runtime_returns_timed_out_outcome_with_continuation_after_model_activity(
    stage_selection_factory: Callable[..., runtime.StageSelection],
    service_registry_factory: Callable[..., ServiceRegistry],
    session_store_factory: Callable[..., _SessionStore],
) -> None:
    worktree = Path("/repo")
    tool_access = runtime.ToolAccess.workspace_backed(
        worktree,
        tool_policy=runtime.ToolPolicy.PARTIAL,
    )

    result = asyncio.run(
        prompt_runtime.NewSessionRuntime(
            execution_adapter=_StartedTimeoutResidentExecutionAdapter(),
            service_registry=service_registry_factory("codex"),
        ).run_new_session(
            prompt_runtime.NewSessionRunRequest(
                prompt="already rendered prompt",
                worktree=WorktreeMount(worktree),
                stage=stage_selection_factory(
                    service="codex",
                    model="gpt-5.4",
                    effort="medium",
                ),
                role=InvocationRole("implementer"),
                session_namespace="main",
                session_store=session_store_factory(),
                provider_session_adapter=_NamedExternalStateResidentPlanningProviderSessionAdapter(
                    "codex"
                ),
                tool_access=tool_access,
            )
        )
    )

    assert result == prompt_runtime.RuntimeOutcome.timed_out(
        output="",
        invocation_progress=runtime.InvocationProgress.STARTED,
        continuation=prompt_runtime.Continuation(
            selected_service="codex",
            selected_model="gpt-5.4",
            selected_effort="medium",
            tool_access=tool_access,
            provider_resume_state={
                "run_kind": "resume",
                "provider_session_id": "prepared:recovered-codex",
                "provider_state_dir_relpath": "codex-runtime-state/",
                "exact_transcript_match": False,
            },
        ),
    )


def test_new_session_runtime_returns_retryable_provider_failure_outcome_with_continuation_after_model_activity(
    stage_selection_factory: Callable[..., runtime.StageSelection],
    service_registry_factory: Callable[..., ServiceRegistry],
    session_store_factory: Callable[..., _SessionStore],
) -> None:
    worktree = Path("/repo")
    tool_access = runtime.ToolAccess.workspace_backed(
        worktree,
        tool_policy=runtime.ToolPolicy.PARTIAL,
    )

    result = asyncio.run(
        prompt_runtime.NewSessionRuntime(
            execution_adapter=_StartedRetryableProviderFailureResidentExecutionAdapter(),
            service_registry=service_registry_factory("codex"),
        ).run_new_session(
            prompt_runtime.NewSessionRunRequest(
                prompt="already rendered prompt",
                worktree=WorktreeMount(worktree),
                stage=stage_selection_factory(
                    service="codex",
                    model="gpt-5.4",
                    effort="medium",
                ),
                role=InvocationRole("implementer"),
                session_namespace="main",
                session_store=session_store_factory(),
                provider_session_adapter=_NamedExternalStateResidentPlanningProviderSessionAdapter(
                    "codex"
                ),
                tool_access=tool_access,
            )
        )
    )

    assert result == prompt_runtime.RuntimeOutcome.retryable_provider_failure(
        output="",
        service_name="codex",
        invocation_progress=runtime.InvocationProgress.STARTED,
        continuation=prompt_runtime.Continuation(
            selected_service="codex",
            selected_model="gpt-5.4",
            selected_effort="medium",
            tool_access=tool_access,
            provider_resume_state={
                "run_kind": "resume",
                "provider_session_id": "prepared:recovered-codex",
                "provider_state_dir_relpath": "codex-runtime-state/",
                "exact_transcript_match": False,
            },
        ),
    )


def test_new_session_runtime_keeps_not_started_timeout_without_continuation(
    stage_selection_factory: Callable[..., runtime.StageSelection],
    service_registry_factory: Callable[..., ServiceRegistry],
    session_store_factory: Callable[..., _SessionStore],
) -> None:
    worktree = Path("/repo")
    tool_access = runtime.ToolAccess.workspace_backed(
        worktree,
        tool_policy=runtime.ToolPolicy.PARTIAL,
    )
    execution_adapter = _TimeoutResidentExecutionAdapter()

    result = asyncio.run(
        prompt_runtime.NewSessionRuntime(
            execution_adapter=execution_adapter,
            service_registry=service_registry_factory("codex"),
        ).run_new_session(
            prompt_runtime.NewSessionRunRequest(
                prompt="already rendered prompt",
                worktree=WorktreeMount(worktree),
                stage=stage_selection_factory(
                    service="codex",
                    model="gpt-5.4",
                    effort="medium",
                ),
                role=InvocationRole("implementer"),
                session_namespace="main",
                session_store=session_store_factory(),
                provider_session_adapter=_NamedExternalStateResidentPlanningProviderSessionAdapter(
                    "codex"
                ),
                tool_access=tool_access,
            )
        )
    )

    assert result == prompt_runtime.RuntimeOutcome.timed_out(
        output="",
        invocation_progress=runtime.InvocationProgress.NOT_STARTED,
    )
    assert result.continuation is None


def test_new_session_runtime_keeps_not_started_retryable_provider_failure_without_continuation(
    stage_selection_factory: Callable[..., runtime.StageSelection],
    service_registry_factory: Callable[..., ServiceRegistry],
    session_store_factory: Callable[..., _SessionStore],
) -> None:
    worktree = Path("/repo")
    tool_access = runtime.ToolAccess.workspace_backed(
        worktree,
        tool_policy=runtime.ToolPolicy.PARTIAL,
    )

    result = asyncio.run(
        prompt_runtime.NewSessionRuntime(
            execution_adapter=_RetryableProviderFailureResidentExecutionAdapter(),
            service_registry=service_registry_factory("codex"),
        ).run_new_session(
            prompt_runtime.NewSessionRunRequest(
                prompt="already rendered prompt",
                worktree=WorktreeMount(worktree),
                stage=stage_selection_factory(
                    service="codex",
                    model="gpt-5.4",
                    effort="medium",
                ),
                role=InvocationRole("implementer"),
                session_namespace="main",
                session_store=session_store_factory(),
                provider_session_adapter=_NamedExternalStateResidentPlanningProviderSessionAdapter(
                    "codex"
                ),
                tool_access=tool_access,
            )
        )
    )

    assert result == prompt_runtime.RuntimeOutcome.retryable_provider_failure(
        output="",
        service_name="codex",
        invocation_progress=runtime.InvocationProgress.NOT_STARTED,
    )
    assert result.continuation is None


def test_new_session_runtime_keeps_exceptional_failures_exceptional(
    stage_selection_factory: Callable[..., runtime.StageSelection],
    service_registry_factory: Callable[..., ServiceRegistry],
    session_store_factory: Callable[..., _SessionStore],
) -> None:
    request = prompt_runtime.NewSessionRunRequest(
        prompt="already rendered prompt",
        worktree=WorktreeMount(Path("/repo")),
        stage=stage_selection_factory(
            service="codex",
            model="gpt-5.4",
            effort="medium",
        ),
        role=InvocationRole("implementer"),
        session_namespace="main",
        session_store=session_store_factory(),
        provider_session_adapter=_NamedExternalStateResidentPlanningProviderSessionAdapter(
            "codex"
        ),
        tool_access=runtime.ToolAccess.workspace_backed(
            Path("/repo"),
            tool_policy=runtime.ToolPolicy.PARTIAL,
        ),
    )

    with pytest.raises(runtime.RuntimeConfigurationError):
        asyncio.run(
            prompt_runtime.NewSessionRuntime(
                execution_adapter=cast(Any, object()),
                service_registry=service_registry_factory("codex"),
            ).run_new_session(request)
        )

    with pytest.raises(AgentCredentialFailureError):
        asyncio.run(
            prompt_runtime.NewSessionRuntime(
                execution_adapter=_CredentialFailureResidentExecutionAdapter(),
                service_registry=service_registry_factory("codex"),
            ).run_new_session(request)
        )

    with pytest.raises(HardAgentError):
        asyncio.run(
            prompt_runtime.NewSessionRuntime(
                execution_adapter=_HardFailureResidentExecutionAdapter(),
                service_registry=service_registry_factory("codex"),
            ).run_new_session(request)
        )

    with pytest.raises(TransientAgentError):
        asyncio.run(
            prompt_runtime.NewSessionRuntime(
                execution_adapter=_TransientProviderFailureResidentExecutionAdapter(),
                service_registry=service_registry_factory("codex"),
            ).run_new_session(request)
        )

    with pytest.raises(AgentFailedError):
        asyncio.run(
            prompt_runtime.NewSessionRuntime(
                execution_adapter=_UnclassifiedProviderFailureResidentExecutionAdapter(),
                service_registry=service_registry_factory("codex"),
            ).run_new_session(request)
        )

    with pytest.raises(RuntimeError, match="unexpected failure"):
        asyncio.run(
            prompt_runtime.NewSessionRuntime(
                execution_adapter=_UnexpectedFailureResidentExecutionAdapter(),
                service_registry=service_registry_factory("codex"),
            ).run_new_session(request)
        )


def test_continuation_provider_resume_state_requires_json_compatible_data() -> None:
    continuation = prompt_runtime.Continuation(
        selected_service="codex",
        selected_model="gpt-5.4",
        selected_effort="medium",
        tool_access=runtime.ToolAccess.no_tools(),
        provider_resume_state={
            "provider_session_id": "prepared:recovered-session",
            "attempts": [1, 2],
            "exact_transcript_match": False,
            "metadata": {"phase": "resume", "notes": None},
        },
    )

    assert continuation.provider_resume_state == {
        "provider_session_id": "prepared:recovered-session",
        "attempts": [1, 2],
        "exact_transcript_match": False,
        "metadata": {"phase": "resume", "notes": None},
    }

    with pytest.raises(
        TypeError,
        match=re.escape("Continuation provider_resume_state must be JSON-compatible."),
    ):
        prompt_runtime.Continuation(
            selected_service="codex",
            selected_model="gpt-5.4",
            selected_effort="medium",
            tool_access=runtime.ToolAccess.no_tools(),
            provider_resume_state={"provider_state_dir": Path("/repo/state")},
        )


def test_resumed_session_runtime_resumes_from_portable_continuation_data() -> None:
    worktree = Path("/repo")
    continuation = prompt_runtime.Continuation(
        selected_service="codex",
        selected_model="gpt-5.4",
        selected_effort="medium",
        tool_access=runtime.ToolAccess.workspace_backed(
            worktree,
            tool_policy=runtime.ToolPolicy.PARTIAL,
        ),
        provider_resume_state={
            "run_kind": "resume",
            "provider_session_id": "recovered-session",
            "provider_state_dir_relpath": "runtime-state/",
            "exact_transcript_match": False,
        },
    )

    result = asyncio.run(
        prompt_runtime.ResumedSessionRuntime(
            execution_adapter=_RuntimePlannedPathResidentExecutionAdapter()
        ).run_resumed_session(
            prompt_runtime.ResumedSessionRunRequest(
                prompt="already rendered prompt",
                worktree=WorktreeMount(worktree),
                role=InvocationRole("implementer"),
                session_namespace="main",
                continuation=continuation,
            )
        )
    )

    assert result == prompt_runtime.RuntimeOutcome.completed(
        output="resume:prepared:recovered-session:/workspace/runtime-state/",
        result=prompt_runtime.SessionRunResult(
            output="resume:prepared:recovered-session:/workspace/runtime-state/",
            runtime_metadata=prompt_runtime.SessionRuntimeMetadata(
                service_name="codex",
                provider_session_id="prepared:recovered-session",
                run_kind=RunKind.RESUME,
                session_namespace="main",
                exact_transcript_match=False,
            ),
        ),
    )
    assert isinstance(result.result, prompt_runtime.SessionRunResult)
    assert result.result.continuation == prompt_runtime.Continuation(
        selected_service="codex",
        selected_model="gpt-5.4",
        selected_effort="medium",
        tool_access=runtime.ToolAccess.workspace_backed(
            worktree,
            tool_policy=runtime.ToolPolicy.PARTIAL,
        ),
        provider_resume_state={
            "run_kind": "resume",
            "provider_session_id": "prepared:recovered-session",
            "provider_state_dir_relpath": "runtime-state/",
            "exact_transcript_match": False,
        },
    )


def test_resumed_session_runtime_passes_continuation_provider_resume_state_to_adapter() -> (
    None
):
    worktree = Path("/repo")
    provider_resume_state = {
        "resume_cursor": {"session": "recovered-session", "turn": 7},
        "provider_flags": ["exact", "tools"],
        "workspace_mount": "/workspace/runtime-state/",
    }

    @dataclass
    class _ObservedResumeStatePreparedRunSession:
        provider_state_dir_container_path: str | None
        run_kind: RunKind
        provider_session_id: str | None
        observed_resume_state: dict[str, Any]

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

    class _ResumeStateObservingRunner:
        def __init__(
            self,
            session: _Session,
            observed_resume_state: dict[str, Any],
        ) -> None:
            self._session = session
            self._observed_resume_state = observed_resume_state

        async def setup(
            self,
            git_name: str,
            git_email: str,
            work_body: str = "",
        ) -> None:
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
            del role, prompt, session_uuid, on_provider_session_id
            assert run_kind is RunKind.RESUME
            return json.dumps(
                {
                    "provider_resume_state": self._observed_resume_state,
                    "provider_state_dir": self._session.provider_state_dir,
                },
                sort_keys=True,
            )

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

    class _ContinuationResumeStateExecutionAdapter:
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
            observed_resume_state: dict[str, Any] = {}

            def _prepare_session(
                run_session: Any,
            ) -> _ObservedResumeStatePreparedRunSession:
                observed_resume_state.update(run_session.provider_resume_state)
                return _ObservedResumeStatePreparedRunSession(
                    provider_state_dir_container_path="/workspace/runtime-state/",
                    run_kind=run_session.run_kind,
                    provider_session_id="prepared:recovered-session",
                    observed_resume_state=observed_resume_state,
                )

            return WorkInvocationDependencies(
                execution=WorkExecutionDependencies(
                    container_workspace="/workspace",
                    prepare_session=cast(Any, _prepare_session),
                    build_session=lambda mount_path, service, provider_state_dir: (
                        _Session(provider_state_dir)
                    ),
                    build_runner=lambda session, status_display: cast(
                        WorkExecutionAdapter,
                        _ResumeStateObservingRunner(
                            cast(_Session, session),
                            observed_resume_state,
                        ),
                    ),
                    get_git_identity=lambda: ("Runtime Test", "runtime@example.com"),
                ),
                failure_handling=WorkFailureHandling(timeout_retries=0),
                presentation=WorkPresentationDependencies(),
            )

    result = asyncio.run(
        prompt_runtime.ResumedSessionRuntime(
            execution_adapter=_ContinuationResumeStateExecutionAdapter()
        ).run_resumed_session(
            prompt_runtime.ResumedSessionRunRequest(
                prompt="already rendered prompt",
                worktree=WorktreeMount(worktree),
                role=InvocationRole("implementer"),
                session_namespace="main",
                continuation=prompt_runtime.Continuation(
                    selected_service="codex",
                    selected_model="gpt-5.4",
                    selected_effort="medium",
                    tool_access=runtime.ToolAccess.workspace_backed(
                        worktree,
                        tool_policy=runtime.ToolPolicy.PARTIAL,
                    ),
                    provider_resume_state=provider_resume_state,
                ),
            )
        )
    )

    assert result.output == json.dumps(
        {
            "provider_resume_state": provider_resume_state,
            "provider_state_dir": "/workspace/runtime-state/",
        },
        sort_keys=True,
    )


def test_resumed_session_runtime_completed_outcome_keeps_service_bound_in_continuation() -> (
    None
):
    worktree = Path("/repo")
    continuation = prompt_runtime.Continuation(
        selected_service="bound-service",
        selected_model="gpt-5.4",
        selected_effort="medium",
        tool_access=runtime.ToolAccess.workspace_backed(worktree),
        provider_resume_state={
            "run_kind": "resume",
            "provider_session_id": "recovered-session",
            "provider_state_dir_relpath": "runtime-state/",
            "exact_transcript_match": False,
        },
    )

    result = asyncio.run(
        prompt_runtime.ResumedSessionRuntime(
            execution_adapter=_ContinuationBoundServiceResidentExecutionAdapter()
        ).run_resumed_session(
            prompt_runtime.ResumedSessionRunRequest(
                prompt="already rendered prompt",
                worktree=WorktreeMount(worktree),
                role=InvocationRole("implementer"),
                session_namespace="main",
                continuation=continuation,
            )
        )
    )

    assert isinstance(result.result, prompt_runtime.SessionRunResult)
    assert result.result.runtime_metadata.service_name == "bound-service"
    assert result.result.continuation == prompt_runtime.Continuation(
        selected_service="bound-service",
        selected_model="gpt-5.4",
        selected_effort="medium",
        tool_access=runtime.ToolAccess.workspace_backed(worktree),
        provider_resume_state={
            "run_kind": "resume",
            "provider_session_id": "prepared:recovered-session",
            "provider_state_dir_relpath": "runtime-state/",
            "exact_transcript_match": False,
        },
    )


def test_resumed_session_runtime_returns_latest_adapter_updated_provider_resume_state() -> (
    None
):
    worktree = Path("/repo")
    initial_resume_state = {
        "resume_cursor": {"session": "recovered-session", "turn": 7},
        "phase": "initial",
    }

    @dataclass
    class _UpdatingProviderRunSession:
        run_kind: RunKind
        provider_session_id: str | None
        latest_provider_resume_state: dict[str, Any]

        def record_provider_session_id(self, provider_session_id: str) -> None:
            self.provider_session_id = provider_session_id
            self.latest_provider_resume_state = {
                "resume_cursor": {"session": provider_session_id, "turn": 8},
                "phase": "running",
            }

        def record_successful_run(self) -> None:
            self.latest_provider_resume_state = {
                "resume_cursor": {
                    "session": self.provider_session_id,
                    "turn": 9,
                },
                "phase": "completed",
            }

    @dataclass
    class _UpdatingPreparedRunSession:
        provider_state_dir_container_path: str | None
        provider_resume_state: dict[str, Any]

        def prepare_for_run(self) -> None:
            return None

        def initial_provider_run_session(self) -> _UpdatingProviderRunSession:
            return _UpdatingProviderRunSession(
                run_kind=RunKind.RESUME,
                provider_session_id="recovered-session",
                latest_provider_resume_state={
                    "resume_cursor": {"session": "recovered-session", "turn": 7},
                    "phase": "initial",
                },
            )

        def resumable_provider_run_session(self) -> _UpdatingProviderRunSession:
            return self.initial_provider_run_session()

        def protocol_reprompt_provider_run_session(self) -> None:
            return None

    class _LatestResumeStateRunner:
        async def setup(
            self,
            git_name: str,
            git_email: str,
            work_body: str = "",
        ) -> None:
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
            del role, prompt, run_kind, session_uuid
            assert callable(on_provider_session_id)
            on_provider_session_id("prepared:next-session")
            return "completed"

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

    class _LatestResumeStateExecutionAdapter:
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

            def _prepare_session(run_session: Any) -> _UpdatingPreparedRunSession:
                return _UpdatingPreparedRunSession(
                    provider_state_dir_container_path="/workspace/runtime-state/",
                    provider_resume_state=dict(run_session.provider_resume_state),
                )

            return WorkInvocationDependencies(
                execution=WorkExecutionDependencies(
                    container_workspace="/workspace",
                    prepare_session=cast(Any, _prepare_session),
                    build_session=lambda mount_path, service, provider_state_dir: (
                        _Session(provider_state_dir)
                    ),
                    build_runner=lambda session, status_display: cast(
                        WorkExecutionAdapter,
                        _LatestResumeStateRunner(),
                    ),
                    get_git_identity=lambda: ("Runtime Test", "runtime@example.com"),
                ),
                failure_handling=WorkFailureHandling(timeout_retries=0),
                presentation=WorkPresentationDependencies(),
            )

    result = asyncio.run(
        prompt_runtime.ResumedSessionRuntime(
            execution_adapter=_LatestResumeStateExecutionAdapter()
        ).run_resumed_session(
            prompt_runtime.ResumedSessionRunRequest(
                prompt="already rendered prompt",
                worktree=WorktreeMount(worktree),
                role=InvocationRole("implementer"),
                session_namespace="main",
                continuation=prompt_runtime.Continuation(
                    selected_service="codex",
                    selected_model="gpt-5.4",
                    selected_effort="medium",
                    tool_access=runtime.ToolAccess.workspace_backed(worktree),
                    provider_resume_state=initial_resume_state,
                ),
            )
        )
    )

    assert isinstance(result.result, prompt_runtime.SessionRunResult)
    assert result.result.continuation == prompt_runtime.Continuation(
        selected_service="codex",
        selected_model="gpt-5.4",
        selected_effort="medium",
        tool_access=runtime.ToolAccess.workspace_backed(worktree),
        provider_resume_state={
            "resume_cursor": {"session": "prepared:next-session", "turn": 9},
            "phase": "completed",
        },
    )


def test_resumed_session_runtime_keeps_frozen_adapter_session_seam_unchanged() -> None:
    worktree = Path("/repo")
    initial_resume_state = {
        "resume_cursor": {"session": "recovered-session", "turn": 7},
        "phase": "initial",
    }
    initial_provider_run_session_calls: list[None] = []

    @dataclass
    class _UpdatingProviderRunSession:
        run_kind: RunKind
        provider_session_id: str | None
        latest_provider_resume_state: dict[str, Any]

        def record_provider_session_id(self, provider_session_id: str) -> None:
            self.provider_session_id = provider_session_id
            self.latest_provider_resume_state = {
                "resume_cursor": {"session": provider_session_id, "turn": 8},
                "phase": "running",
            }

        def record_successful_run(self) -> None:
            self.latest_provider_resume_state = {
                "resume_cursor": {
                    "session": self.provider_session_id,
                    "turn": 9,
                },
                "phase": "completed",
            }

    @dataclass(frozen=True, slots=True)
    class _FrozenPreparedRunSession:
        provider_state_dir_container_path: str | None
        provider_resume_state: dict[str, Any]
        provider_run_session: _UpdatingProviderRunSession

        def prepare_for_run(self) -> None:
            return None

        def initial_provider_run_session(self) -> _UpdatingProviderRunSession:
            initial_provider_run_session_calls.append(None)
            if len(initial_provider_run_session_calls) > 1:
                raise AssertionError(
                    "Runtime should not recreate the initial provider run session."
                )
            return self.provider_run_session

        def resumable_provider_run_session(self) -> _UpdatingProviderRunSession:
            return self.initial_provider_run_session()

        def protocol_reprompt_provider_run_session(self) -> None:
            return None

    class _FrozenSessionRunner:
        async def setup(
            self,
            git_name: str,
            git_email: str,
            work_body: str = "",
        ) -> None:
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
            del role, prompt, run_kind, session_uuid
            assert callable(on_provider_session_id)
            on_provider_session_id("prepared:next-session")
            return "completed"

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

    class _FrozenPreparedSessionExecutionAdapter:
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

            def _prepare_session(run_session: Any) -> _FrozenPreparedRunSession:
                return _FrozenPreparedRunSession(
                    provider_state_dir_container_path="/workspace/runtime-state/",
                    provider_resume_state=dict(run_session.provider_resume_state),
                    provider_run_session=_UpdatingProviderRunSession(
                        run_kind=RunKind.RESUME,
                        provider_session_id="recovered-session",
                        latest_provider_resume_state=dict(initial_resume_state),
                    ),
                )

            return WorkInvocationDependencies(
                execution=WorkExecutionDependencies(
                    container_workspace="/workspace",
                    prepare_session=cast(Any, _prepare_session),
                    build_session=lambda mount_path, service, provider_state_dir: (
                        _Session(provider_state_dir)
                    ),
                    build_runner=lambda session, status_display: cast(
                        WorkExecutionAdapter,
                        _FrozenSessionRunner(),
                    ),
                    get_git_identity=lambda: ("Runtime Test", "runtime@example.com"),
                ),
                failure_handling=WorkFailureHandling(timeout_retries=0),
                presentation=WorkPresentationDependencies(),
            )

    result = asyncio.run(
        prompt_runtime.ResumedSessionRuntime(
            execution_adapter=_FrozenPreparedSessionExecutionAdapter()
        ).run_resumed_session(
            prompt_runtime.ResumedSessionRunRequest(
                prompt="already rendered prompt",
                worktree=WorktreeMount(worktree),
                role=InvocationRole("implementer"),
                session_namespace="main",
                continuation=prompt_runtime.Continuation(
                    selected_service="codex",
                    selected_model="gpt-5.4",
                    selected_effort="medium",
                    tool_access=runtime.ToolAccess.workspace_backed(worktree),
                    provider_resume_state=initial_resume_state,
                ),
            )
        )
    )

    assert isinstance(result.result, prompt_runtime.SessionRunResult)
    assert initial_provider_run_session_calls == [None]
    assert result.result.continuation == prompt_runtime.Continuation(
        selected_service="codex",
        selected_model="gpt-5.4",
        selected_effort="medium",
        tool_access=runtime.ToolAccess.workspace_backed(worktree),
        provider_resume_state={
            "resume_cursor": {"session": "prepared:next-session", "turn": 9},
            "phase": "completed",
        },
    )


def test_resumable_run_request_from_continuation_rejects_tool_access_override() -> None:
    with pytest.raises(
        TypeError,
        match=re.escape(
            "ResumedSessionRunRequest derives fixed tool access from `continuation` and does not accept `tool_access` or `tool_policy` overrides."
        ),
    ):
        prompt_runtime.ResumedSessionRunRequest(
            prompt="already rendered prompt",
            worktree=WorktreeMount(Path("/repo")),
            role=InvocationRole("implementer"),
            session_namespace="main",
            continuation=prompt_runtime.Continuation(
                selected_service="codex",
                selected_model="gpt-5.4",
                selected_effort="medium",
                tool_access=runtime.ToolAccess.no_tools(),
                provider_resume_state={"run_kind": "resume"},
            ),
            tool_access=runtime.ToolAccess.workspace_backed(Path("/repo")),
        )


def test_resumable_run_request_from_continuation_requires_role() -> None:
    with pytest.raises(
        TypeError,
        match=re.escape(
            "ResumedSessionRunRequest requires a `role` value when constructed from a continuation."
        ),
    ):
        prompt_runtime.ResumedSessionRunRequest(
            prompt="already rendered prompt",
            worktree=WorktreeMount(Path("/repo")),
            session_namespace="main",
            continuation=prompt_runtime.Continuation(
                selected_service="codex",
                selected_model="gpt-5.4",
                selected_effort="medium",
                tool_access=runtime.ToolAccess.no_tools(),
                provider_resume_state={"run_kind": "resume"},
            ),
        )


def test_resumable_run_request_rejects_conflicting_continuation_and_session_plan(
    session_store_factory: Callable[..., _SessionStore],
    resident_provider_session_adapter: _ResidentPlanningProviderSessionAdapter,
) -> None:
    service = cast(ExecutionProvider, _ExecutionService("codex"))
    session_plan = plan_resumable_session(
        ResumableSessionPlanRequest(
            worktree=Path("/repo"),
            role=InvocationRole("implementer"),
            namespace="main",
            service=service,
            session_store=session_store_factory(),
            provider_session_adapter=resident_provider_session_adapter,
        )
    )

    with pytest.raises(
        TypeError,
        match=re.escape(
            "ResumedSessionRunRequest received conflicting `session_plan` and `continuation` values."
        ),
    ):
        prompt_runtime.ResumedSessionRunRequest(
            prompt="already rendered prompt",
            worktree=WorktreeMount(Path("/repo")),
            model="gpt-5.4",
            effort="medium",
            session_plan=session_plan,
            continuation=prompt_runtime.Continuation(
                selected_service="codex",
                selected_model="gpt-5.4",
                selected_effort="medium",
                tool_access=runtime.ToolAccess.no_tools(),
                provider_resume_state={"run_kind": "resume"},
            ),
            role=InvocationRole("implementer"),
            tool_policy=runtime.ToolPolicy.FULL,
        )


def test_resumable_run_request_rejects_conflicting_tool_access_and_tool_policy() -> (
    None
):
    with pytest.raises(
        TypeError,
        match=re.escape(
            "ResumedSessionRunRequest received conflicting `tool_access` and `tool_policy` values."
        ),
    ):
        prompt_runtime.ResumedSessionRunRequest(
            prompt="already rendered prompt",
            worktree=WorktreeMount(Path("/repo")),
            model="gpt-5.4",
            effort="medium",
            session_plan=ResumableSessionPlan(
                role=InvocationRole("reviewer"),
                worktree=Path("/repo"),
                namespace="main",
                service=cast(ExecutionProvider, _ExecutionService("codex")),
                run_kind=RunKind.FRESH,
                provider_state_dir=None,
                provider_session_id=None,
                auth_seeding_requirement=AuthSeedingRequirement.NOT_REQUIRED,
            ),
            tool_access=runtime.ToolAccess.no_tools(),
            tool_policy=runtime.ToolPolicy.FULL,
        )


def test_resumed_session_runtime_from_continuation_defaults_and_overrides_model_and_effort() -> (
    None
):
    worktree = Path("/repo")
    continuation = prompt_runtime.Continuation(
        selected_service="codex",
        selected_model="gpt-5.4",
        selected_effort="medium",
        tool_access=runtime.ToolAccess.workspace_backed(worktree),
        provider_resume_state={
            "run_kind": "resume",
            "provider_session_id": "recovered-session",
            "provider_state_dir_relpath": "runtime-state/",
            "exact_transcript_match": False,
        },
    )

    defaulted_result = asyncio.run(
        prompt_runtime.ResumedSessionRuntime(
            execution_adapter=_RuntimePlannedPathResidentExecutionAdapter()
        ).run_resumed_session(
            prompt_runtime.ResumedSessionRunRequest(
                prompt="already rendered prompt",
                worktree=WorktreeMount(worktree),
                role=InvocationRole("implementer"),
                session_namespace="main",
                continuation=continuation,
            )
        )
    )
    overridden_result = asyncio.run(
        prompt_runtime.ResumedSessionRuntime(
            execution_adapter=_RuntimePlannedPathResidentExecutionAdapter()
        ).run_resumed_session(
            prompt_runtime.ResumedSessionRunRequest(
                prompt="already rendered prompt",
                worktree=WorktreeMount(worktree),
                role=InvocationRole("implementer"),
                session_namespace="main",
                continuation=continuation,
                model="gpt-5.5",
                effort="high",
            )
        )
    )

    assert isinstance(defaulted_result.result, prompt_runtime.SessionRunResult)
    assert defaulted_result.result.continuation == prompt_runtime.Continuation(
        selected_service="codex",
        selected_model="gpt-5.4",
        selected_effort="medium",
        tool_access=runtime.ToolAccess.workspace_backed(worktree),
        provider_resume_state={
            "run_kind": "resume",
            "provider_session_id": "prepared:recovered-session",
            "provider_state_dir_relpath": "runtime-state/",
            "exact_transcript_match": False,
        },
    )
    assert isinstance(overridden_result.result, prompt_runtime.SessionRunResult)
    assert overridden_result.result.continuation == prompt_runtime.Continuation(
        selected_service="codex",
        selected_model="gpt-5.5",
        selected_effort="high",
        tool_access=runtime.ToolAccess.workspace_backed(worktree),
        provider_resume_state={
            "run_kind": "resume",
            "provider_session_id": "prepared:recovered-session",
            "provider_state_dir_relpath": "runtime-state/",
            "exact_transcript_match": False,
        },
    )


def test_resumed_session_runtime_started_usage_limit_keeps_service_bound_in_continuation() -> (
    None
):
    worktree = Path("/repo")
    continuation = _bound_service_resumed_continuation(worktree)

    result = asyncio.run(
        prompt_runtime.ResumedSessionRuntime(
            execution_adapter=_ContinuationBoundStartedUsageLimitResidentExecutionAdapter()
        ).run_resumed_session(
            prompt_runtime.ResumedSessionRunRequest(
                prompt="already rendered prompt",
                worktree=WorktreeMount(worktree),
                role=InvocationRole("implementer"),
                session_namespace="main",
                continuation=continuation,
            )
        )
    )

    assert result == prompt_runtime.RuntimeOutcome.usage_limited(
        output="",
        service_name="codex",
        reset_time=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
        usage_limit_scope=runtime.UsageLimitScope("implementer"),
        invocation_progress=prompt_runtime.InvocationProgress.STARTED,
        continuation=continuation,
    )


@pytest.mark.parametrize("tool_policy", _TOOL_POLICY_CASES)
def test_resumed_session_runtime_exposes_tool_policy_effects_through_runtime_result(
    tool_policy: runtime.ToolPolicy | runtime.ToolPolicyProfile,
    session_store_factory: Callable[..., _SessionStore],
    resident_provider_session_adapter: _ResidentPlanningProviderSessionAdapter,
) -> None:
    execution_adapter = _ToolPolicyRenderingResidentExecutionAdapter()
    service = cast(ExecutionProvider, _ExecutionService("codex"))
    session_plan = plan_resumable_session(
        ResumableSessionPlanRequest(
            worktree=Path("."),
            role=InvocationRole("implementer"),
            namespace="main",
            service=service,
            session_store=session_store_factory(),
            provider_session_adapter=resident_provider_session_adapter,
        )
    )

    result = asyncio.run(
        prompt_runtime.ResumedSessionRuntime(
            execution_adapter=execution_adapter
        ).run_resumed_session(
            prompt_runtime.ResumedSessionRunRequest(
                prompt="already rendered prompt",
                worktree=WorktreeMount(Path(".")),
                model="gpt-5.4",
                effort="medium",
                session_plan=session_plan,
                tool_policy=tool_policy,
            )
        )
    )

    assert result.output == _tool_policy_effect_text(tool_policy)


def test_resumed_session_runtime_reports_started_progress_for_usage_limited_outcome(
    session_store_factory: Callable[..., _SessionStore],
    resident_provider_session_adapter: _ResidentPlanningProviderSessionAdapter,
) -> None:
    service = cast(ExecutionProvider, _ExecutionService("codex"))
    session_plan = plan_resumable_session(
        ResumableSessionPlanRequest(
            worktree=Path("."),
            role=InvocationRole("implementer"),
            namespace="main",
            service=service,
            session_store=session_store_factory(),
            provider_session_adapter=resident_provider_session_adapter,
        )
    )

    result = asyncio.run(
        prompt_runtime.ResumedSessionRuntime(
            execution_adapter=_StartedUsageLimitResidentExecutionAdapter()
        ).run_resumed_session(
            prompt_runtime.ResumedSessionRunRequest(
                prompt="already rendered prompt",
                worktree=WorktreeMount(Path(".")),
                model="gpt-5.4",
                effort="medium",
                session_plan=session_plan,
                tool_policy=runtime.ToolPolicy.FULL,
            )
        )
    )

    assert result == prompt_runtime.RuntimeOutcome.usage_limited(
        output="",
        service_name="codex",
        reset_time=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
        usage_limit_scope=runtime.UsageLimitScope("implementer"),
        invocation_progress=prompt_runtime.InvocationProgress.STARTED,
        continuation=prompt_runtime.Continuation(
            selected_service="codex",
            selected_model="gpt-5.4",
            selected_effort="medium",
            tool_access=runtime.ToolAccess.workspace_backed(Path(".")),
            provider_resume_state={
                "run_kind": "resume",
                "provider_session_id": "prepared:recovered-session",
                "provider_state_dir_relpath": "state/",
                "exact_transcript_match": False,
            },
        ),
    )


def test_resumed_session_runtime_prefers_adapter_reported_model_activity_for_usage_limited_outcome(
    session_store_factory: Callable[..., _SessionStore],
    resident_provider_session_adapter: _ResidentPlanningProviderSessionAdapter,
) -> None:
    service = cast(ExecutionProvider, _ExecutionService("codex"))
    session_plan = plan_resumable_session(
        ResumableSessionPlanRequest(
            worktree=Path("."),
            role=InvocationRole("implementer"),
            namespace="main",
            service=service,
            session_store=session_store_factory(),
            provider_session_adapter=resident_provider_session_adapter,
        )
    )

    result = asyncio.run(
        prompt_runtime.ResumedSessionRuntime(
            execution_adapter=_ModelActivityUsageLimitResidentExecutionAdapter()
        ).run_resumed_session(
            prompt_runtime.ResumedSessionRunRequest(
                prompt="already rendered prompt",
                worktree=WorktreeMount(Path(".")),
                model="gpt-5.4",
                effort="medium",
                session_plan=session_plan,
                tool_policy=runtime.ToolPolicy.FULL,
            )
        )
    )

    assert result == prompt_runtime.RuntimeOutcome.usage_limited(
        output="",
        service_name="codex",
        reset_time=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
        usage_limit_scope=runtime.UsageLimitScope("implementer"),
        invocation_progress=prompt_runtime.InvocationProgress.STARTED,
        continuation=prompt_runtime.Continuation(
            selected_service="codex",
            selected_model="gpt-5.4",
            selected_effort="medium",
            tool_access=runtime.ToolAccess.workspace_backed(Path(".")),
            provider_resume_state={
                "run_kind": "resume",
                "provider_session_id": "prepared:recovered-session",
                "provider_state_dir_relpath": "state/",
                "exact_transcript_match": False,
            },
        ),
    )


def test_resumed_session_runtime_returns_no_service_available_outcome_for_bound_service_unavailability() -> (
    None
):
    worktree = Path("/repo")
    continuation = _bound_service_resumed_continuation(worktree)

    class _UnavailableBoundServiceExecutionAdapter:
        def resolve_service(self, service_name: str = "") -> ExecutionProvider:
            assert service_name == "bound-service"
            raise NoServiceAvailableError(
                reset_time=datetime(2026, 1, 2, tzinfo=timezone.utc),
                usage_limit_scope=runtime.UsageLimitScope("resume-scope"),
                invocation_progress=runtime.InvocationProgress.STARTED,
            )

        def build_work_dependencies(
            self,
            *,
            name: str,
            model: str,
            effort: str,
            service: ExecutionProvider,
        ) -> WorkInvocationDependencies:
            del name, model, effort, service
            raise AssertionError("bound service unavailability should stop before work")

    result = asyncio.run(
        prompt_runtime.ResumedSessionRuntime(
            execution_adapter=_UnavailableBoundServiceExecutionAdapter()
        ).run_resumed_session(
            prompt_runtime.ResumedSessionRunRequest(
                prompt="already rendered prompt",
                worktree=WorktreeMount(worktree),
                role=InvocationRole("implementer"),
                session_namespace="main",
                continuation=continuation,
            )
        )
    )

    assert result == prompt_runtime.RuntimeOutcome.no_service_available(
        output="",
        reset_time=datetime(2026, 1, 2, tzinfo=timezone.utc),
        usage_limit_scope=runtime.UsageLimitScope("resume-scope"),
        invocation_progress=runtime.InvocationProgress.STARTED,
        continuation=continuation,
    )


def test_resumed_session_runtime_returns_cancelled_outcome_with_input_continuation_after_model_activity() -> (
    None
):
    worktree = Path("/repo")
    continuation = _bound_service_resumed_continuation(worktree)

    class _StartedCancellationRunner(_ResidentSeamRunner):
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
            del (
                prompt,
                role,
                tool_policy,
                run_kind,
                session_uuid,
                on_provider_session_id,
            )
            raise AgentCancelledError(
                invocation_progress=runtime.InvocationProgress.STARTED,
            )

    class _StartedCancellationExecutionAdapter:
        def resolve_service(self, service_name: str = "") -> ExecutionProvider:
            assert service_name == "bound-service"
            return _ExecutionService("resolved-service")

        def build_work_dependencies(
            self,
            *,
            name: str,
            model: str,
            effort: str,
            service: ExecutionProvider,
        ) -> WorkInvocationDependencies:
            del name, model, effort, service

            def _prepare_session(
                run_session: Any,
            ) -> _ResidentAdapterPreparedRunSession:
                return _ResidentAdapterPreparedRunSession(
                    provider_state_dir_container_path="/workspace/runtime-state/",
                    run_kind=run_session.run_kind,
                    provider_session_id=f"prepared:{run_session.provider_session_id}",
                )

            return WorkInvocationDependencies(
                execution=WorkExecutionDependencies(
                    container_workspace="/workspace",
                    prepare_session=cast(Any, _prepare_session),
                    build_session=lambda mount_path, service, provider_state_dir: (
                        _Session(provider_state_dir)
                    ),
                    build_runner=lambda session, status_display: cast(
                        WorkExecutionAdapter,
                        _StartedCancellationRunner(cast(_Session, session)),
                    ),
                    get_git_identity=lambda: ("Runtime Test", "runtime@example.com"),
                ),
                failure_handling=WorkFailureHandling(timeout_retries=0),
                presentation=WorkPresentationDependencies(),
            )

    result = asyncio.run(
        prompt_runtime.ResumedSessionRuntime(
            execution_adapter=_StartedCancellationExecutionAdapter()
        ).run_resumed_session(
            prompt_runtime.ResumedSessionRunRequest(
                prompt="already rendered prompt",
                worktree=WorktreeMount(worktree),
                role=InvocationRole("implementer"),
                session_namespace="main",
                continuation=continuation,
            )
        )
    )

    assert result == prompt_runtime.RuntimeOutcome.cancelled(
        output="",
        invocation_progress=runtime.InvocationProgress.STARTED,
        continuation=continuation,
    )


def test_resumed_session_runtime_returns_timed_out_outcome_with_input_continuation_after_model_activity() -> (
    None
):
    worktree = Path("/repo")
    continuation = _bound_service_resumed_continuation(worktree)

    result = asyncio.run(
        prompt_runtime.ResumedSessionRuntime(
            execution_adapter=_StartedTimeoutResidentExecutionAdapter()
        ).run_resumed_session(
            prompt_runtime.ResumedSessionRunRequest(
                prompt="already rendered prompt",
                worktree=WorktreeMount(worktree),
                role=InvocationRole("implementer"),
                session_namespace="main",
                continuation=continuation,
            )
        )
    )

    assert result == prompt_runtime.RuntimeOutcome.timed_out(
        output="",
        invocation_progress=runtime.InvocationProgress.STARTED,
        continuation=continuation,
    )


def test_resumed_session_runtime_returns_retryable_provider_failure_outcome_with_input_continuation_after_model_activity() -> (
    None
):
    worktree = Path("/repo")
    continuation = _bound_service_resumed_continuation(worktree)

    result = asyncio.run(
        prompt_runtime.ResumedSessionRuntime(
            execution_adapter=_StartedRetryableProviderFailureResidentExecutionAdapter()
        ).run_resumed_session(
            prompt_runtime.ResumedSessionRunRequest(
                prompt="already rendered prompt",
                worktree=WorktreeMount(worktree),
                role=InvocationRole("implementer"),
                session_namespace="main",
                continuation=continuation,
            )
        )
    )

    assert result == prompt_runtime.RuntimeOutcome.retryable_provider_failure(
        output="",
        service_name="codex",
        invocation_progress=runtime.InvocationProgress.STARTED,
        continuation=continuation,
    )


def _bound_service_resumed_continuation(
    worktree: Path,
) -> prompt_runtime.Continuation:
    return prompt_runtime.Continuation(
        selected_service="bound-service",
        selected_model="gpt-5.4",
        selected_effort="medium",
        tool_access=runtime.ToolAccess.workspace_backed(worktree),
        provider_resume_state={
            "run_kind": "resume",
            "provider_session_id": "recovered-session",
            "provider_state_dir_relpath": "runtime-state/",
            "exact_transcript_match": False,
        },
    )


def test_resumed_session_runtime_returns_usage_limited_outcome_with_input_continuation_before_model_activity() -> (
    None
):
    worktree = Path("/repo")
    continuation = _bound_service_resumed_continuation(worktree)

    class _NotStartedUsageLimitExecutionAdapter:
        def resolve_service(self, service_name: str = "") -> ExecutionProvider:
            assert service_name == "bound-service"
            return _ExecutionService("resolved-service")

        def build_work_dependencies(
            self,
            *,
            name: str,
            model: str,
            effort: str,
            service: ExecutionProvider,
        ) -> WorkInvocationDependencies:
            del name, model, effort, service

            class _NotStartedUsageLimitRunner(_ResidentSeamRunner):
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
                    del (
                        prompt,
                        role,
                        tool_policy,
                        run_kind,
                        session_uuid,
                        on_provider_session_id,
                    )
                    raise UsageLimitError(
                        reset_time=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
                        service_name="codex",
                        invocation_progress=runtime.InvocationProgress.NOT_STARTED,
                    )

            def _prepare_session(
                run_session: Any,
            ) -> _ResidentAdapterPreparedRunSession:
                return _ResidentAdapterPreparedRunSession(
                    provider_state_dir_container_path="/workspace/runtime-state/",
                    run_kind=run_session.run_kind,
                    provider_session_id=f"prepared:{run_session.provider_session_id}",
                )

            return WorkInvocationDependencies(
                execution=WorkExecutionDependencies(
                    container_workspace="/workspace",
                    prepare_session=cast(Any, _prepare_session),
                    build_session=lambda mount_path, service, provider_state_dir: (
                        _Session(provider_state_dir)
                    ),
                    build_runner=lambda session, status_display: cast(
                        WorkExecutionAdapter,
                        _NotStartedUsageLimitRunner(cast(_Session, session)),
                    ),
                    get_git_identity=lambda: ("Runtime Test", "runtime@example.com"),
                ),
                failure_handling=WorkFailureHandling(timeout_retries=0),
                presentation=WorkPresentationDependencies(),
            )

    result = asyncio.run(
        prompt_runtime.ResumedSessionRuntime(
            execution_adapter=_NotStartedUsageLimitExecutionAdapter()
        ).run_resumed_session(
            prompt_runtime.ResumedSessionRunRequest(
                prompt="already rendered prompt",
                worktree=WorktreeMount(worktree),
                role=InvocationRole("implementer"),
                session_namespace="main",
                continuation=continuation,
            )
        )
    )

    assert result == prompt_runtime.RuntimeOutcome.usage_limited(
        output="",
        service_name="codex",
        reset_time=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
        usage_limit_scope=runtime.UsageLimitScope("implementer"),
        invocation_progress=runtime.InvocationProgress.NOT_STARTED,
        continuation=continuation,
    )


def test_resumed_session_runtime_returns_no_service_available_outcome_with_input_continuation_before_model_activity() -> (
    None
):
    worktree = Path("/repo")
    continuation = _bound_service_resumed_continuation(worktree)

    class _UnavailableBoundServiceExecutionAdapter:
        def resolve_service(self, service_name: str = "") -> ExecutionProvider:
            assert service_name == "bound-service"
            raise NoServiceAvailableError(
                reset_time=datetime(2026, 1, 2, tzinfo=timezone.utc),
                invocation_progress=runtime.InvocationProgress.NOT_STARTED,
            )

        def build_work_dependencies(
            self,
            *,
            name: str,
            model: str,
            effort: str,
            service: ExecutionProvider,
        ) -> WorkInvocationDependencies:
            del name, model, effort, service
            raise AssertionError("bound service unavailability should stop before work")

    result = asyncio.run(
        prompt_runtime.ResumedSessionRuntime(
            execution_adapter=_UnavailableBoundServiceExecutionAdapter()
        ).run_resumed_session(
            prompt_runtime.ResumedSessionRunRequest(
                prompt="already rendered prompt",
                worktree=WorktreeMount(worktree),
                role=InvocationRole("implementer"),
                session_namespace="main",
                continuation=continuation,
            )
        )
    )

    assert result == prompt_runtime.RuntimeOutcome.no_service_available(
        output="",
        reset_time=datetime(2026, 1, 2, tzinfo=timezone.utc),
        usage_limit_scope=None,
        invocation_progress=runtime.InvocationProgress.NOT_STARTED,
        continuation=continuation,
    )


def test_resumed_session_runtime_returns_cancelled_outcome_with_input_continuation_before_model_activity() -> (
    None
):
    worktree = Path("/repo")
    continuation = _bound_service_resumed_continuation(worktree)
    cancelled_token = CancellationToken()
    cancelled_token.cancel()

    result = asyncio.run(
        prompt_runtime.ResumedSessionRuntime(
            execution_adapter=_TimeoutResidentExecutionAdapter()
        ).run_resumed_session(
            prompt_runtime.ResumedSessionRunRequest(
                prompt="already rendered prompt",
                worktree=WorktreeMount(worktree),
                role=InvocationRole("implementer"),
                session_namespace="main",
                continuation=continuation,
                token=cancelled_token,
            )
        )
    )

    assert result == prompt_runtime.RuntimeOutcome.cancelled(
        output="",
        invocation_progress=runtime.InvocationProgress.NOT_STARTED,
        continuation=continuation,
    )


@pytest.mark.parametrize(
    ("execution_adapter", "expected_kind", "service_name"),
    [
        (
            _TimeoutResidentExecutionAdapter(),
            "timed_out",
            None,
        ),
        (
            _RetryableProviderFailureResidentExecutionAdapter(),
            "retryable_provider_failure",
            "codex",
        ),
    ],
)
def test_resumed_session_runtime_preserves_input_continuation_for_not_started_interruption_outcomes(
    execution_adapter: Any,
    expected_kind: str,
    service_name: str | None,
) -> None:
    worktree = Path("/repo")
    continuation = _bound_service_resumed_continuation(worktree)

    result = asyncio.run(
        prompt_runtime.ResumedSessionRuntime(
            execution_adapter=execution_adapter
        ).run_resumed_session(
            prompt_runtime.ResumedSessionRunRequest(
                prompt="already rendered prompt",
                worktree=WorktreeMount(worktree),
                role=InvocationRole("implementer"),
                session_namespace="main",
                continuation=continuation,
            )
        )
    )

    assert result == prompt_runtime.RuntimeOutcome(
        kind=expected_kind,
        output="",
        service_name=service_name,
        invocation_progress=runtime.InvocationProgress.NOT_STARTED,
        continuation=continuation,
    )


def test_resumed_session_runtime_returns_cancelled_outcome_for_pre_start_caller_cancellation(
    session_store_factory: Callable[..., _SessionStore],
    resident_provider_session_adapter: _ResidentPlanningProviderSessionAdapter,
) -> None:
    service = cast(ExecutionProvider, _ExecutionService("codex"))
    session_plan = plan_resumable_session(
        ResumableSessionPlanRequest(
            worktree=Path("."),
            role=InvocationRole("implementer"),
            namespace="main",
            service=service,
            session_store=session_store_factory(),
            provider_session_adapter=resident_provider_session_adapter,
        )
    )
    cancelled_token = CancellationToken()
    cancelled_token.cancel()

    result = asyncio.run(
        prompt_runtime.ResumedSessionRuntime(
            execution_adapter=_RuntimePlannedPathResidentExecutionAdapter()
        ).run_resumed_session(
            prompt_runtime.ResumedSessionRunRequest(
                prompt="already rendered prompt",
                worktree=WorktreeMount(Path(".")),
                model="gpt-5.4",
                effort="medium",
                session_plan=session_plan,
                tool_policy=runtime.ToolPolicy.FULL,
                token=cancelled_token,
            )
        )
    )

    assert result == prompt_runtime.RuntimeOutcome.cancelled(
        output="",
        invocation_progress=prompt_runtime.InvocationProgress.NOT_STARTED,
    )


def test_resumed_session_runtime_reports_started_progress_for_timed_out_outcome(
    session_store_factory: Callable[..., _SessionStore],
    resident_provider_session_adapter: _ResidentPlanningProviderSessionAdapter,
) -> None:
    service = cast(ExecutionProvider, _ExecutionService("codex"))
    session_plan = plan_resumable_session(
        ResumableSessionPlanRequest(
            worktree=Path("."),
            role=InvocationRole("implementer"),
            namespace="main",
            service=service,
            session_store=session_store_factory(),
            provider_session_adapter=resident_provider_session_adapter,
        )
    )

    result = asyncio.run(
        prompt_runtime.ResumedSessionRuntime(
            execution_adapter=_StartedTimeoutResidentExecutionAdapter()
        ).run_resumed_session(
            prompt_runtime.ResumedSessionRunRequest(
                prompt="already rendered prompt",
                worktree=WorktreeMount(Path(".")),
                model="gpt-5.4",
                effort="medium",
                session_plan=session_plan,
                tool_policy=runtime.ToolPolicy.FULL,
            )
        )
    )

    assert result == prompt_runtime.RuntimeOutcome.timed_out(
        output="",
        invocation_progress=prompt_runtime.InvocationProgress.STARTED,
        continuation=prompt_runtime.Continuation(
            selected_service="codex",
            selected_model="gpt-5.4",
            selected_effort="medium",
            tool_access=runtime.ToolAccess.workspace_backed(Path(".")),
            provider_resume_state={
                "run_kind": "resume",
                "provider_session_id": "prepared:recovered-session",
                "provider_state_dir_relpath": "state/",
                "exact_transcript_match": False,
            },
        ),
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


def test_resumable_run_request_carries_workspace_backed_tool_access() -> None:
    tool_access = runtime.ToolAccess.workspace_backed(
        Path("/repo"),
        tool_policy=runtime.ToolPolicy.PARTIAL,
    )

    request = prompt_runtime.ResumedSessionRunRequest(
        prompt="already rendered prompt",
        worktree=WorktreeMount(Path("/repo")),
        model="gpt-5.4",
        effort="medium",
        session_plan=ResumableSessionPlan(
            role=InvocationRole("reviewer"),
            worktree=Path("/repo"),
            namespace="main",
            service=cast(ExecutionProvider, _ExecutionService("codex")),
            run_kind=RunKind.FRESH,
            provider_state_dir=None,
            provider_session_id=None,
            auth_seeding_requirement=AuthSeedingRequirement.NOT_REQUIRED,
        ),
        tool_access=tool_access,
    )

    assert request.tool_access == tool_access
    assert request.tool_access.workspace == Path("/repo")


def test_resumable_run_request_accepts_explicit_no_tools_tool_access() -> None:
    request = prompt_runtime.ResumedSessionRunRequest(
        prompt="already rendered prompt",
        worktree=WorktreeMount(Path("/repo")),
        model="gpt-5.4",
        effort="medium",
        session_plan=ResumableSessionPlan(
            role=InvocationRole("reviewer"),
            worktree=Path("/repo"),
            namespace="main",
            service=cast(ExecutionProvider, _ExecutionService("codex")),
            run_kind=RunKind.FRESH,
            provider_state_dir=None,
            provider_session_id=None,
            auth_seeding_requirement=AuthSeedingRequirement.NOT_REQUIRED,
        ),
        tool_access=runtime.ToolAccess.no_tools(),
    )

    assert request.tool_access == runtime.ToolAccess.no_tools()
    assert request.tool_policy == runtime.ToolAccess.no_tools().tool_policy


def test_resumable_run_request_rejects_workspace_backed_tool_access_for_other_worktree() -> (
    None
):
    with pytest.raises(
        ValueError,
        match=re.escape(
            "ResumedSessionRunRequest workspace-backed tool access requires worktree /repo, got /other."
        ),
    ):
        prompt_runtime.ResumedSessionRunRequest(
            prompt="already rendered prompt",
            worktree=WorktreeMount(Path("/other")),
            model="gpt-5.4",
            effort="medium",
            session_plan=ResumableSessionPlan(
                role=InvocationRole("reviewer"),
                worktree=Path("/other"),
                namespace="main",
                service=cast(ExecutionProvider, _ExecutionService("codex")),
                run_kind=RunKind.FRESH,
                provider_state_dir=None,
                provider_session_id=None,
                auth_seeding_requirement=AuthSeedingRequirement.NOT_REQUIRED,
            ),
            tool_access=runtime.ToolAccess.workspace_backed(
                Path("/repo"),
                tool_policy=runtime.ToolPolicy.FULL,
            ),
        )


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


def test_resumable_runtime_request_rejects_request_level_invocation_role() -> None:
    with pytest.raises(TypeError):
        prompt_runtime.ResumedSessionRunRequest(
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
            tool_policy=runtime.ToolPolicy.FULL,
            role=InvocationRole("implementer"),
        )  # type: ignore[call-arg]


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
    assert token_counts == []


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
