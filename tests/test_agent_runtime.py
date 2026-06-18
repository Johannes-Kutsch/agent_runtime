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
    AgentCredentialFailureError,
    AgentFailedError,
    AgentRuntimeError,
    AgentTimeoutError,
    HardAgentError,
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
    ToolPolicyObservingExecutionServiceFake as _ToolPolicyObservingExecutionService,
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


class _OneShotWorkRunner:
    def __init__(
        self,
        service: _ExecutionService,
        *,
        invocation_order: list[str],
        attempts_by_service: dict[str, int],
    ) -> None:
        self._service = service
        self._invocation_order = invocation_order
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
        self._invocation_order.append(service_name)
        attempt_count = self._attempts_by_service.get(service_name, 0) + 1
        self._attempts_by_service[service_name] = attempt_count

        if service_name == "codex":
            if attempt_count > 1:
                raise AssertionError("one-shot retried the exhausted primary service")
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


class _OneShotExecutionAdapter:
    def __init__(
        self,
        *,
        invocation_order: list[str],
        attempts_by_service: dict[str, int],
    ) -> None:
        self._invocation_order = invocation_order
        self._attempts_by_service = attempts_by_service

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
                    _OneShotWorkRunner(
                        execution_service,
                        invocation_order=self._invocation_order,
                        attempts_by_service=self._attempts_by_service,
                    ),
                ),
                get_git_identity=lambda: ("Runtime Test", "runtime@example.com"),
            ),
            failure_handling=WorkFailureHandling(timeout_retries=0),
            presentation=WorkPresentationDependencies(),
        )


class _RoleAwarePreparedRunSession(_PreparedRunSession):
    def __init__(self, run_session: Any, observed_run_sessions: list[Any]) -> None:
        super().__init__()
        self.provider_state_dir_container_path = (
            f"/workspace/state/{run_session.role.value}/{run_session.session_namespace}"
        )
        observed_run_sessions.append(run_session)


class _RoleAwareOneShotWorkRunner:
    def __init__(self, observed_roles: list[InvocationRole]) -> None:
        self._observed_roles = observed_roles

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

        self._observed_roles.append(role)
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


class _RoleAwareOneShotExecutionAdapter:
    def __init__(self) -> None:
        self.observed_run_sessions: list[Any] = []
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
        return WorkInvocationDependencies(
            execution=WorkExecutionDependencies(
                container_workspace="/workspace",
                prepare_session=lambda run_session: cast(
                    PreparedRunSessionState,
                    _RoleAwarePreparedRunSession(
                        run_session,
                        self.observed_run_sessions,
                    ),
                ),
                build_session=lambda mount_path, service, provider_state_dir: _Session(
                    provider_state_dir
                ),
                build_runner=lambda session, status_display: cast(
                    WorkExecutionAdapter,
                    _RoleAwareOneShotWorkRunner(self.observed_roles),
                ),
                get_git_identity=lambda: ("Runtime Test", "runtime@example.com"),
            ),
            failure_handling=WorkFailureHandling(timeout_retries=0),
            presentation=WorkPresentationDependencies(),
        )


class _UsageLimitWithoutMappingRunner(_RoleAwareOneShotWorkRunner):
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
        del prompt, run_kind, session_uuid, on_provider_session_id
        raise UsageLimitError(reset_time=None, service_name="codex")


class _UsageLimitWithoutMappingExecutionAdapter(_RoleAwareOneShotExecutionAdapter):
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
                    PreparedRunSessionState,
                    _RoleAwarePreparedRunSession(
                        run_session,
                        self.observed_run_sessions,
                    ),
                ),
                build_session=lambda mount_path, service, provider_state_dir: _Session(
                    provider_state_dir
                ),
                build_runner=lambda session, status_display: cast(
                    WorkExecutionAdapter,
                    _UsageLimitWithoutMappingRunner(self.observed_roles),
                ),
                get_git_identity=lambda: ("Runtime Test", "runtime@example.com"),
            ),
            failure_handling=WorkFailureHandling(timeout_retries=0),
            presentation=WorkPresentationDependencies(),
        )


class _PromptOnlyOneShotWorkRunner:
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
        raise AssertionError("one-shot used tool-capable work invocation")

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
        raise AssertionError("one-shot used tool-capable work_text invocation")

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


class _PromptOnlyOneShotExecutionAdapter:
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
                    _PromptOnlyOneShotWorkRunner(),
                ),
                get_git_identity=lambda: ("Runtime Test", "runtime@example.com"),
            ),
            failure_handling=WorkFailureHandling(timeout_retries=0),
            presentation=WorkPresentationDependencies(),
        )


class _NormalizedPromptOnlyOneShotWorkRunner:
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


class _NormalizedPromptOnlyOneShotExecutionAdapter:
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
                    _NormalizedPromptOnlyOneShotWorkRunner(),
                ),
                get_git_identity=lambda: ("Runtime Test", "runtime@example.com"),
            ),
            failure_handling=WorkFailureHandling(timeout_retries=0),
            presentation=WorkPresentationDependencies(),
        )


class _ToolCapableOnlyOneShotWorkRunner:
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
        raise AssertionError("one-shot fell back to tool-capable work invocation")

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
        raise AssertionError("one-shot fell back to tool-capable work_text invocation")


class _MissingPromptOnlyOneShotExecutionAdapter:
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
                    _ToolCapableOnlyOneShotWorkRunner(),
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


class _ToolPolicyObservingResidentRunner(_ResidentSeamRunner):
    def __init__(
        self,
        session: _Session,
        service: _ExecutionService,
        observed_tool_policies: list[Any],
    ) -> None:
        super().__init__(session)
        self._service = service
        self._observed_tool_policies = observed_tool_policies

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
        self._service.build_command(
            role,
            "gpt-5.4",
            "medium",
            run_kind,
            session_uuid,
            tool_policy=tool_policy,
        )
        self._observed_tool_policies.append(tool_policy)
        return await super().work_text(
            prompt,
            role=role,
            tool_policy=tool_policy,
            run_kind=run_kind,
            session_uuid=session_uuid,
            on_provider_session_id=on_provider_session_id,
        )


class _ToolPolicyObservingResidentExecutionAdapter:
    def __init__(self) -> None:
        self.observed_tool_policies: list[Any] = []
        self.observed_service_tool_policies: list[Any] = []

    def resolve_service(self, service_name: str = "") -> ExecutionProvider:
        return _ToolPolicyObservingExecutionService(
            service_name,
            self.observed_service_tool_policies,
        )

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
                    _ToolPolicyObservingResidentRunner(
                        cast(_Session, session),
                        execution_service,
                        self.observed_tool_policies,
                    ),
                ),
                get_git_identity=lambda: ("Runtime Test", "runtime@example.com"),
            ),
            failure_handling=WorkFailureHandling(timeout_retries=0),
            presentation=WorkPresentationDependencies(),
        )


class _ToolPolicyObservingPromptRunner:
    def __init__(
        self,
        service: _ExecutionService,
        observed_tool_policies: list[Any],
    ) -> None:
        self._service = service
        self._observed_tool_policies = observed_tool_policies

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
        del prompt
        self._service.build_command(
            role,
            "gpt-5.4",
            "medium",
            run_kind,
            session_uuid,
            tool_policy=tool_policy,
        )
        self._observed_tool_policies.append(tool_policy)
        assert callable(on_provider_session_id)
        on_provider_session_id("provider-session")
        return "tool-capable-output"


class _ToolPolicyObservingPromptExecutionAdapter:
    def __init__(self) -> None:
        self.observed_tool_policies: list[Any] = []
        self.observed_service_tool_policies: list[Any] = []

    def resolve_service(self, service_name: str = "") -> ExecutionProvider:
        return _ToolPolicyObservingExecutionService(
            service_name,
            self.observed_service_tool_policies,
        )

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
                    _ToolPolicyObservingPromptRunner(
                        execution_service,
                        self.observed_tool_policies,
                    ),
                ),
                get_git_identity=lambda: ("Runtime Test", "runtime@example.com"),
            ),
            failure_handling=WorkFailureHandling(timeout_retries=0),
            presentation=WorkPresentationDependencies(),
        )


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


class _PlannedStatePathObservingResidentExecutionAdapter:
    def __init__(self) -> None:
        self.observed_provider_state_dir_container_paths: list[str | None] = []

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
            self.observed_provider_state_dir_container_paths.append(
                run_session.provider_state_dir_container_path
            )
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


def test_package_exports_runtime_surface() -> None:
    assert runtime.__all__ == [
        "AgentCredentialFailureError",
        "AgentFailedError",
        "AgentRuntimeError",
        "AgentTimeoutError",
        "HardAgentError",
        "ExecutionProvider",
        "InvocationRole",
        "ProviderSessionAdapter",
        "RuntimeConfigurationError",
        "RunKind",
        "StageSelection",
        "ToolPolicy",
        "ToolPolicyProfile",
        "TransientAgentError",
        "UsageLimitError",
        "UsageLimitScope",
    ]
    assert runtime.StageSelection.__module__.startswith("agent_runtime")
    assert not hasattr(runtime, "StageOverride")
    assert runtime.AgentRuntimeError is AgentRuntimeError
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
    assert not hasattr(prompt_runtime, "run_one_shot")
    assert not hasattr(prompt_runtime, "run_prompt")
    assert not hasattr(prompt_runtime, "run_resumable_prompt")
    assert not hasattr(prompt_runtime, "ResidentRunRequest")
    assert not hasattr(prompt_runtime, "ResidentRunResult")
    assert not hasattr(prompt_runtime, "ResidentRuntime")
    assert not hasattr(prompt_runtime, "ResidentRuntimeExecutionAdapter")
    assert not hasattr(prompt_runtime, "ResidentRuntimeMetadata")
    assert {
        "ResumableRunRequest",
        "ResumableRunResult",
        "ResumableRuntime",
        "ResumableRuntimeExecutionAdapter",
        "ResumableRuntimeMetadata",
    } <= set(prompt_runtime.__all__)


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
            lambda: prompt_runtime.ResumableRunRequest(
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
            "ResumableRunRequest requires an explicit `tool_policy` value.",
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


def test_agent_failed_error_rejects_unsafe_runtime_identity_labels_before_building_diagnostics() -> (
    None
):
    with pytest.raises(ValueError):
        AgentFailedError(
            invocation_role="implementer",
            worktree_path=Path("."),
            namespace="../escape",
        )

    with pytest.raises(ValueError):
        AgentFailedError(
            invocation_role="implementer",
            worktree_path=Path("."),
            service_name="a/b",
        )


def test_runtime_errors_expose_invocation_role_metadata() -> None:
    timeout = AgentTimeoutError(
        "timed out",
        invocation_role="reviewer",
        worktree_path=Path("worktree"),
    )
    failed = AgentFailedError(
        invocation_role="reviewer",
        worktree_path=Path("worktree"),
        namespace="main",
        service_name="codex",
    )

    assert timeout.invocation_role == "reviewer"
    assert failed.invocation_role == "reviewer"
    assert failed.session_dir == "reviewer/main/codex"


def test_usage_limit_error_exposes_usage_limit_scope_metadata() -> None:
    error = UsageLimitError(
        reset_time=None,
        usage_limit_scope=runtime.UsageLimitScope("quota-review"),
    )

    assert error.usage_limit_scope == runtime.UsageLimitScope("quota-review")


def test_one_shot_runtime_exposes_usage_limit_service_name_metadata(
    one_shot_request_factory: Callable[..., prompt_runtime.OneShotRunRequest],
    service_registry_factory: Callable[..., ServiceRegistry],
) -> None:
    runtime_instance = prompt_runtime.OneShotRuntime(
        execution_adapter=_RoleAwareOneShotExecutionAdapter(),
        service_registry=service_registry_factory("codex", unavailable={"codex"}),
    )

    with pytest.raises(UsageLimitError) as excinfo:
        asyncio.run(runtime_instance.run_one_shot(one_shot_request_factory()))

    assert excinfo.value.service_name == "codex"


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
    one_shot_request_factory: Callable[..., prompt_runtime.OneShotRunRequest],
    service_registry_factory: Callable[..., ServiceRegistry],
    stage_selection_factory: Callable[..., runtime.StageSelection],
) -> None:
    result = asyncio.run(
        prompt_runtime.OneShotRuntime(
            execution_adapter=_RoleAwareOneShotExecutionAdapter(),
            service_registry=service_registry_factory("codex"),
        ).run_one_shot(
            one_shot_request_factory(
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


def test_agent_invocation_log_records_conditional_usage_limit_scope(
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

    with invocation_log.open_work_invocation(
        log_path=log_path,
        role=InvocationRole("implementer"),
        usage_limit_scope=runtime.UsageLimitScope("repo-write"),
        run_kind=RunKind.RESUME,
        session_uuid=None,
        prompt="different scope from role",
    ) as work_invocation:
        work_invocation.record_provider_session_id("provider-session")

    headers = [
        record
        for record in (
            json.loads(line) for line in log_path.read_text().splitlines() if line
        )
        if record.get("type") == "agent_invocation"
    ]

    assert "usage_limit_scope" not in headers[0]
    assert headers[1]["invocation_role"] == "implementer"
    assert headers[1]["usage_limit_scope"] == "repo-write"
    assert headers[1]["provider_session_id"] == "provider-session"


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


def test_one_shot_run_request_override_rejects_invalid_stage_fallback() -> None:
    with pytest.raises(ValueError, match="effort"):
        prompt_runtime.OneShotRunRequest(
            prompt="already rendered prompt",
            worktree=WorktreeMount(Path(".")),
            override=runtime.StageSelection(
                service="codex",
                model="gpt-5.4",
                effort="medium",
                fallback=runtime.StageSelection(
                    service="claude",
                    model="sonnet",
                    effort="",
                ),
            ),
            role=InvocationRole("implementer"),
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


def test_one_shot_runtime_falls_back_after_usage_limit_with_fresh_service_resolution(
    stage_selection_factory: Callable[..., runtime.StageSelection],
    service_registry_factory: Callable[..., ServiceRegistry],
    one_shot_request_factory: Callable[..., prompt_runtime.OneShotRunRequest],
) -> None:
    invocation_order: list[str] = []
    attempts_by_service: dict[str, int] = {}
    registry = service_registry_factory(
        "codex",
        "claude",
        wake_times={"claude": datetime(2026, 1, 2, tzinfo=timezone.utc)},
    )

    result = asyncio.run(
        prompt_runtime.OneShotRuntime(
            execution_adapter=_OneShotExecutionAdapter(
                invocation_order=invocation_order,
                attempts_by_service=attempts_by_service,
            ),
            service_registry=registry,
        ).run_one_shot(
            one_shot_request_factory(
                stage=stage_selection_factory(
                    service="codex",
                    fallback=stage_selection_factory(
                        service="claude",
                        model="sonnet",
                        effort="high",
                    ),
                )
            )
        )
    )

    assert result == prompt_runtime.OneShotRunResult(
        output="claude:already rendered prompt",
        selected_service="claude",
        selected_model="sonnet",
        selected_effort="high",
        used_fallback=True,
        metadata=prompt_runtime.OneShotResultMetadata(
            selected_service_path=("codex", "claude"),
            runtime=prompt_runtime.OneShotRuntimeMetadata(
                provider_session_id="provider-claude",
                run_kind=RunKind.FRESH,
                session_namespace="",
            ),
        ),
    )
    assert invocation_order == ["codex", "claude"]


def test_one_shot_runtime_request_requires_explicit_invocation_role(
    stage_selection_factory: Callable[..., runtime.StageSelection],
) -> None:
    with pytest.raises(TypeError):
        prompt_runtime.OneShotRunRequest(
            prompt="already rendered prompt",
            worktree=WorktreeMount(Path(".")),
            override=stage_selection_factory(),
        )  # type: ignore[call-arg]


def test_one_shot_runtime_uses_supplied_invocation_role_across_execution_surfaces(
    service_registry_factory: Callable[..., ServiceRegistry],
    one_shot_request_factory: Callable[..., prompt_runtime.OneShotRunRequest],
    stage_selection_factory: Callable[..., runtime.StageSelection],
) -> None:
    role = InvocationRole("reviewer")
    execution_adapter = _RoleAwareOneShotExecutionAdapter()
    runtime_instance = prompt_runtime.OneShotRuntime(
        execution_adapter=execution_adapter,
        service_registry=service_registry_factory("codex"),
    )
    request = one_shot_request_factory(
        override=stage_selection_factory(),
        role=role,
        session_namespace="main",
    )

    result = asyncio.run(runtime_instance.run_one_shot(request))

    assert result.output == "reviewer:already rendered prompt"
    assert result.runtime_metadata == prompt_runtime.OneShotRuntimeMetadata(
        provider_session_id="provider-reviewer",
        run_kind=RunKind.FRESH,
        session_namespace="main",
    )
    assert request.stage == stage_selection_factory()
    assert request.override == request.stage
    assert execution_adapter.observed_roles == [role]
    assert execution_adapter.observed_run_sessions[0].role == role
    assert execution_adapter.observed_run_sessions[0].session_namespace == "main"
    cancelled_token = CancellationToken()
    cancelled = one_shot_request_factory(
        override=stage_selection_factory(),
        role=role,
        token=cancelled_token,
    )
    cancelled_token.cancel()

    with pytest.raises(UsageLimitError) as excinfo:
        asyncio.run(runtime_instance.run_one_shot(cancelled))

    assert excinfo.value.usage_limit_scope == runtime.UsageLimitScope("reviewer")


def test_one_shot_runtime_separates_usage_limit_scope_from_invocation_role(
    service_registry_factory: Callable[..., ServiceRegistry],
    one_shot_request_factory: Callable[..., prompt_runtime.OneShotRunRequest],
    stage_selection_factory: Callable[..., runtime.StageSelection],
) -> None:
    role = InvocationRole("reviewer")
    execution_adapter = _RoleAwareOneShotExecutionAdapter()
    runtime_instance = prompt_runtime.OneShotRuntime(
        execution_adapter=execution_adapter,
        service_registry=service_registry_factory("codex"),
    )

    result = asyncio.run(
        runtime_instance.run_one_shot(
            one_shot_request_factory(
                stage=stage_selection_factory(),
                role=role,
                usage_limit_scope=runtime.UsageLimitScope("quota-review"),
                session_namespace="main",
            )
        )
    )

    assert result.output == "reviewer:already rendered prompt"
    assert execution_adapter.observed_roles == [role]
    assert execution_adapter.observed_run_sessions[0].role == role

    cancelled_token = CancellationToken()
    cancelled_token.cancel()

    with pytest.raises(UsageLimitError) as excinfo:
        asyncio.run(
            runtime_instance.run_one_shot(
                one_shot_request_factory(
                    override=stage_selection_factory(),
                    role=role,
                    usage_limit_scope=runtime.UsageLimitScope("quota-review"),
                    token=cancelled_token,
                )
            )
        )

    assert excinfo.value.usage_limit_scope == runtime.UsageLimitScope("quota-review")


def test_one_shot_runtime_fills_usage_limit_scope_without_role_mapping_hook(
    service_registry_factory: Callable[..., ServiceRegistry],
    one_shot_request_factory: Callable[..., prompt_runtime.OneShotRunRequest],
    stage_selection_factory: Callable[..., runtime.StageSelection],
) -> None:
    role = InvocationRole("reviewer")
    execution_adapter = _UsageLimitWithoutMappingExecutionAdapter()
    runtime_instance = prompt_runtime.OneShotRuntime(
        execution_adapter=execution_adapter,
        service_registry=service_registry_factory("codex"),
    )

    with pytest.raises(UsageLimitError) as excinfo:
        asyncio.run(
            runtime_instance.run_one_shot(
                one_shot_request_factory(
                    stage=stage_selection_factory(),
                    role=role,
                    usage_limit_scope=runtime.UsageLimitScope("quota-review"),
                    session_namespace="main",
                )
            )
        )

    assert excinfo.value.usage_limit_scope == runtime.UsageLimitScope("quota-review")
    assert execution_adapter.observed_roles == [role]
    assert execution_adapter.observed_run_sessions[0].role == role


def test_one_shot_run_request_uses_stage_selection_vocabulary() -> None:
    stage = runtime.StageSelection(
        service="codex",
        model="gpt-5.4",
        effort="medium",
    )

    request = prompt_runtime.OneShotRunRequest(
        prompt="already rendered prompt",
        worktree=WorktreeMount(Path(".")),
        stage=stage,
        role=InvocationRole("implementer"),
    )

    assert request.stage is stage
    assert request.override is stage


def test_one_shot_run_request_accepts_plain_worktree_path() -> None:
    request = prompt_runtime.OneShotRunRequest(
        prompt="already rendered prompt",
        worktree=Path("."),
        stage=runtime.StageSelection(
            service="codex",
            model="gpt-5.4",
            effort="medium",
        ),
        role=InvocationRole("implementer"),
    )

    assert request.worktree == Path(".")
    assert request.mount_path == Path(".")


def test_one_shot_run_request_preserves_override_keyword_compatibility() -> None:
    stage = runtime.StageSelection(
        service="codex",
        model="gpt-5.4",
        effort="medium",
        fallback=runtime.StageSelection(
            service="claude",
            model="sonnet",
            effort="high",
        ),
    )

    request = prompt_runtime.OneShotRunRequest(
        prompt="already rendered prompt",
        worktree=WorktreeMount(Path(".")),
        override=stage,
        role=InvocationRole("implementer"),
    )

    assert request.stage == stage
    assert request.override == stage


def test_one_shot_run_request_uses_direct_session_namespace_field() -> None:
    request = prompt_runtime.OneShotRunRequest(
        prompt="already rendered prompt",
        worktree=Path("."),
        stage=runtime.StageSelection(
            service="codex",
            model="gpt-5.4",
            effort="medium",
        ),
        role=InvocationRole("implementer"),
        session_namespace="main",
    )

    assert request.session_namespace == "main"


def test_one_shot_run_request_does_not_expose_tool_policy() -> None:
    stage = runtime.StageSelection(
        service="codex",
        model="gpt-5.4",
        effort="medium",
    )

    with pytest.raises(TypeError):
        prompt_runtime.OneShotRunRequest(
            prompt="already rendered prompt",
            worktree=WorktreeMount(Path(".")),
            stage=stage,
            role=InvocationRole("implementer"),
            tool_policy=runtime.ToolPolicy.FULL,
        )  # type: ignore[call-arg]


def test_one_shot_runtime_uses_prompt_only_provider_invocation() -> None:
    runtime_instance = prompt_runtime.OneShotRuntime(
        execution_adapter=_PromptOnlyOneShotExecutionAdapter(),
        service_registry=ServiceRegistry(
            {
                "codex": cast(
                    ServiceSelectionProvider,
                    _Service(
                        "codex",
                        available=True,
                        wake_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
                    ),
                )
            }
        ),
    )

    result = asyncio.run(
        runtime_instance.run_one_shot(
            prompt_runtime.OneShotRunRequest(
                prompt="already rendered prompt",
                worktree=WorktreeMount(Path(".")),
                stage=runtime.StageSelection(
                    service="codex",
                    model="gpt-5.4",
                    effort="medium",
                ),
                role=InvocationRole("implementer"),
            )
        )
    )

    assert result.output == "implementer:already rendered prompt:prompt_only"
    assert result.runtime_metadata == prompt_runtime.OneShotRuntimeMetadata(
        provider_session_id="provider-prompt-only",
        run_kind=RunKind.FRESH,
        session_namespace="",
    )


def test_one_shot_runtime_requires_prompt_only_provider_invocation() -> None:
    runtime_instance = prompt_runtime.OneShotRuntime(
        execution_adapter=_MissingPromptOnlyOneShotExecutionAdapter(),
        service_registry=ServiceRegistry(
            {
                "codex": cast(
                    ServiceSelectionProvider,
                    _Service(
                        "codex",
                        available=True,
                        wake_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
                    ),
                )
            }
        ),
    )

    with pytest.raises(runtime.RuntimeConfigurationError) as excinfo:
        asyncio.run(
            runtime_instance.run_one_shot(
                prompt_runtime.OneShotRunRequest(
                    prompt="already rendered prompt",
                    worktree=WorktreeMount(Path(".")),
                    stage=runtime.StageSelection(
                        service="codex",
                        model="gpt-5.4",
                        effort="medium",
                    ),
                    role=InvocationRole("implementer"),
                )
            )
        )

    assert str(excinfo.value) == (
        "One-shot runtime requires a work runner with callable `prompt_only()`."
    )


def test_one_shot_runtime_returns_normalized_text_output() -> None:
    runtime_instance = prompt_runtime.OneShotRuntime(
        execution_adapter=_NormalizedPromptOnlyOneShotExecutionAdapter(),
        service_registry=ServiceRegistry(
            {
                "codex": cast(
                    ServiceSelectionProvider,
                    _Service(
                        "codex",
                        available=True,
                        wake_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
                    ),
                )
            }
        ),
    )

    result = asyncio.run(
        runtime_instance.run_one_shot(
            prompt_runtime.OneShotRunRequest(
                prompt="already rendered prompt",
                worktree=WorktreeMount(Path(".")),
                stage=runtime.StageSelection(
                    service="codex",
                    model="gpt-5.4",
                    effort="medium",
                ),
                role=InvocationRole("implementer"),
            )
        )
    )

    assert result.output == "normalized:already rendered prompt"


def test_one_shot_run_result_groups_runtime_metadata_under_metadata() -> None:
    runtime_instance = prompt_runtime.OneShotRuntime(
        execution_adapter=_PromptOnlyOneShotExecutionAdapter(),
        service_registry=ServiceRegistry(
            {
                "codex": cast(
                    ServiceSelectionProvider,
                    _Service(
                        "codex",
                        available=True,
                        wake_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
                    ),
                )
            }
        ),
    )

    result = asyncio.run(
        runtime_instance.run_one_shot(
            prompt_runtime.OneShotRunRequest(
                prompt="already rendered prompt",
                worktree=WorktreeMount(Path(".")),
                stage=runtime.StageSelection(
                    service="codex",
                    model="gpt-5.4",
                    effort="medium",
                ),
                role=InvocationRole("implementer"),
            )
        )
    )

    assert result.metadata == prompt_runtime.OneShotResultMetadata(
        selected_service_path=("codex",),
        runtime=prompt_runtime.OneShotRuntimeMetadata(
            provider_session_id="provider-prompt-only",
            run_kind=RunKind.FRESH,
            session_namespace="",
        ),
    )


def test_one_shot_runtime_reports_selected_service_path_without_fallback() -> None:
    runtime_instance = prompt_runtime.OneShotRuntime(
        execution_adapter=_PromptOnlyOneShotExecutionAdapter(),
        service_registry=ServiceRegistry(
            {
                "codex": cast(
                    ServiceSelectionProvider,
                    _Service(
                        "codex",
                        available=True,
                        wake_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
                    ),
                ),
                "claude": cast(
                    ServiceSelectionProvider,
                    _Service(
                        "claude",
                        available=True,
                        wake_time=datetime(2026, 1, 2, tzinfo=timezone.utc),
                    ),
                ),
            }
        ),
    )

    result = asyncio.run(
        runtime_instance.run_one_shot(
            prompt_runtime.OneShotRunRequest(
                prompt="already rendered prompt",
                worktree=WorktreeMount(Path(".")),
                stage=runtime.StageSelection(
                    service="codex",
                    model="gpt-5.4",
                    effort="medium",
                    fallback=runtime.StageSelection(
                        service="claude",
                        model="sonnet",
                        effort="high",
                    ),
                ),
                role=InvocationRole("implementer"),
            )
        )
    )

    assert result.selected_service == "codex"
    assert result.selected_model == "gpt-5.4"
    assert result.selected_effort == "medium"
    assert result.used_fallback is False
    assert result.metadata.selected_service_path == ("codex",)


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


def test_resumable_runtime_preserves_resumable_behavior_through_run_session_seam(
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
        prompt_runtime.ResumableRuntime(
            execution_adapter=_ResidentSeamExecutionAdapter()
        ).run_resumable_prompt(
            prompt_runtime.ResumableRunRequest(
                prompt="already rendered prompt",
                worktree=WorktreeMount(Path(".")),
                model="gpt-5.4",
                effort="medium",
                session_plan=session_plan,
                tool_policy=runtime.ToolPolicy.FULL,
            )
        )
    )

    assert result == prompt_runtime.ResumableRunResult(
        output="resume:prepared:recovered-session:/workspace/runtime-state/",
        runtime_metadata=prompt_runtime.ResumableRuntimeMetadata(
            service_name="codex",
            provider_session_id="prepared:recovered-session",
            run_kind=RunKind.RESUME,
            session_namespace="main",
            exact_transcript_match=False,
        ),
    )


def test_resumable_runtime_uses_invocation_role_from_session_plan(
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
        prompt_runtime.ResumableRuntime(
            execution_adapter=execution_adapter
        ).run_resumable_prompt(
            prompt_runtime.ResumableRunRequest(
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


def test_resumable_runtime_preserves_planned_relative_provider_state_path(
    execution_service_factory: Callable[..., ExecutionProvider],
    session_store_factory: Callable[..., _SessionStore],
    external_state_provider_session_adapter: _ExternalStateResidentPlanningProviderSessionAdapter,
) -> None:
    worktree = Path("/repo")
    execution_adapter = _PlannedStatePathObservingResidentExecutionAdapter()
    session_plan = plan_resumable_session(
        ResumableSessionPlanRequest(
            worktree=worktree,
            role=InvocationRole("implementer"),
            namespace="main",
            service=execution_service_factory(),
            session_store=session_store_factory(),
            provider_session_adapter=external_state_provider_session_adapter,
        )
    )

    result = asyncio.run(
        prompt_runtime.ResumableRuntime(
            execution_adapter=execution_adapter
        ).run_resumable_prompt(
            prompt_runtime.ResumableRunRequest(
                prompt="already rendered prompt",
                worktree=WorktreeMount(worktree),
                model="gpt-5.4",
                effort="medium",
                session_plan=session_plan,
                tool_policy=runtime.ToolPolicy.FULL,
            )
        )
    )

    assert execution_adapter.observed_provider_state_dir_container_paths == [
        "/workspace/runtime-state/"
    ]
    assert (
        result.output == "resume:prepared:recovered-session:/workspace/runtime-state/"
    )


@pytest.mark.parametrize("tool_policy", list(runtime.ToolPolicy))
def test_resumable_runtime_passes_explicit_tool_policy_to_tool_capable_execution(
    tool_policy: runtime.ToolPolicy,
    session_store_factory: Callable[..., _SessionStore],
    resident_provider_session_adapter: _ResidentPlanningProviderSessionAdapter,
) -> None:
    execution_adapter = _ToolPolicyObservingResidentExecutionAdapter()
    service = cast(
        ExecutionProvider,
        _ToolPolicyObservingExecutionService(
            "codex",
            execution_adapter.observed_service_tool_policies,
        ),
    )
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

    asyncio.run(
        prompt_runtime.ResumableRuntime(
            execution_adapter=execution_adapter
        ).run_resumable_prompt(
            prompt_runtime.ResumableRunRequest(
                prompt="already rendered prompt",
                worktree=WorktreeMount(Path(".")),
                model="gpt-5.4",
                effort="medium",
                session_plan=session_plan,
                tool_policy=tool_policy,
            )
        )
    )

    assert execution_adapter.observed_tool_policies == [tool_policy]
    assert execution_adapter.observed_service_tool_policies == [tool_policy]


@pytest.mark.parametrize("tool_policy", list(runtime.ToolPolicy))
def test_prompt_runtime_passes_explicit_tool_policy_to_tool_capable_execution(
    tool_policy: runtime.ToolPolicy,
    prompt_run_request_factory: Callable[..., PromptRunRequest],
    service_registry_factory: Callable[..., ServiceRegistry],
    stage_selection_factory: Callable[..., runtime.StageSelection],
) -> None:
    execution_adapter = _ToolPolicyObservingPromptExecutionAdapter()

    output = asyncio.run(
        prompt_runtime._run_prompt(
            runner=execution_adapter,
            service_registry=service_registry_factory("codex"),
            request=prompt_run_request_factory(
                stage=stage_selection_factory(),
                tool_policy=tool_policy,
            ),
        )
    )

    assert output == "tool-capable-output"
    assert execution_adapter.observed_tool_policies == [tool_policy]
    assert execution_adapter.observed_service_tool_policies == [tool_policy]


def test_resumable_runtime_request_rejects_request_level_invocation_role() -> None:
    with pytest.raises(TypeError):
        prompt_runtime.ResumableRunRequest(
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


def test_reduce_text_output_events_returns_result_and_maps_errors() -> None:
    token_counts: list[int] = []
    turns: list[str] = []
    result = reduce_text_output_events(
        [
            PromptTokens(2),
            UnsupportedTokens(3, "source"),
            AssistantTurn("hello"),
            Result("done"),
        ],
        turns.append,
        token_counts.append,
        provider="codex",
    )

    assert result == "done"
    assert turns == ["hello"]
    assert token_counts == [2]

    observation = ProviderErrorObservation(
        service_name="codex",
        raw_provider_text="bad credential",
        source_stream="stderr",
    )
    with pytest.raises(UsageLimitError):
        reduce_text_output_events(
            [UsageLimit(reset_time=None)], turns.append, provider="codex"
        )
    with pytest.raises(TransientAgentError):
        reduce_text_output_events(
            [TransientError(status_code=503, raw_message="retry")],
            turns.append,
            provider="codex",
        )
    with pytest.raises(HardAgentError):
        reduce_text_output_events(
            [
                HardError(
                    status_code=400, raw_message="bad", observations=(observation,)
                )
            ],
            turns.append,
            provider="codex",
        )
    with pytest.raises(AgentCredentialFailureError):
        reduce_text_output_events(
            [
                CredentialFailure(
                    raw_message="missing auth",
                    service_name="codex",
                    source_observations=(observation,),
                )
            ],
            turns.append,
            provider="codex",
        )


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


def test_runtime_errors_capture_context() -> None:
    timeout = AgentTimeoutError("timed out")
    usage_limit = UsageLimitError(reset_time=None)
    transient = TransientAgentError("transient", status_code=502)
    hard = HardAgentError("hard", status_code=400, service_name="codex")
    failed = AgentFailedError("implementer", Path("worktree"), service_name="codex")

    assert isinstance(timeout, AgentRuntimeError)
    assert usage_limit.service_name is None
    assert transient.status_code == 502
    assert hard.service_name == "codex"
    assert failed.session_dir == "implementer/codex"
