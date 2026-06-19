from __future__ import annotations

import asyncio
import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import pytest

import agent_runtime as runtime
import agent_runtime.contracts as contracts_runtime
import agent_runtime._runtime_compat as compat_runtime
import agent_runtime.provider_session_adapter as provider_session_adapter_runtime
import agent_runtime.runtime as prompt_runtime
import agent_runtime.session as session_runtime
from agent_runtime.contracts import AssistantTurn, ExecutionProvider, TransientError
from agent_runtime.errors import (
    AgentCancelledError,
    AgentCredentialFailureError,
    AgentFailedError,
    AgentTimeoutError,
    HardAgentError,
    TransientAgentError,
    UsageLimitError,
)
from agent_runtime.execution_contracts import (
    WorkExecutionAdapter,
    WorkExecutionDependencies,
    WorkFailureHandling,
    WorkInvocationDependencies,
    WorkPresentationDependencies,
    WorktreeMount,
)
from agent_runtime.provider_output import reduce_text_output_events
from agent_runtime.provider_session_adapter import ProviderSessionPlanningRequest
from agent_runtime.roles import InvocationRole
from agent_runtime.service_registry import ServiceRegistry
from agent_runtime.session import RunKind

from tests.runtime_boundary_fakes import (
    ExecutionServiceFake as _ExecutionService,
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


class _Session:
    def __init__(self, provider_state_dir: str | None = None) -> None:
        self.provider_state_dir = provider_state_dir


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
        tool_policy: Any = runtime.ToolPolicy.UNRESTRICTED,
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
        tool_policy: Any = runtime.ToolPolicy.UNRESTRICTED,
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
        tool_policy: Any = runtime.ToolPolicy.UNRESTRICTED,
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
        tool_policy: Any = runtime.ToolPolicy.UNRESTRICTED,
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
        tool_policy: Any = runtime.ToolPolicy.UNRESTRICTED,
        run_kind: RunKind = RunKind.FRESH,
        session_uuid: str | None = None,
        on_provider_session_id: Any = None,
    ) -> str:
        del prompt, role, tool_policy, run_kind, session_uuid, on_provider_session_id
        raise AgentCancelledError(
            invocation_progress=runtime.InvocationProgress.STARTED
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
        tool_policy: Any = runtime.ToolPolicy.UNRESTRICTED,
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


class _RetryableProviderFailureResidentRunner(_ResidentSeamRunner):
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
        tool_policy: Any = runtime.ToolPolicy.UNRESTRICTED,
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
        tool_policy: Any = runtime.ToolPolicy.UNRESTRICTED,
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
        tool_policy: Any = runtime.ToolPolicy.UNRESTRICTED,
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


class _CredentialFailureResidentRunner(_ResidentSeamRunner):
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
        tool_policy: Any = runtime.ToolPolicy.UNRESTRICTED,
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
        tool_policy: Any = runtime.ToolPolicy.UNRESTRICTED,
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
        tool_policy: Any = runtime.ToolPolicy.UNRESTRICTED,
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


def test_new_session_runtime_selects_fallback_service_before_binding_continuation(
    stage_selection_factory: Callable[..., runtime.StageSelection],
    service_registry_factory: Callable[..., ServiceRegistry],
    session_store_factory: Callable[..., _SessionStore],
) -> None:
    worktree = Path("/repo")
    tool_access = contracts_runtime.ToolAccess.workspace_backed(
        worktree,
        tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
    )

    result = asyncio.run(
        compat_runtime.NewSessionRuntime(
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
    tool_access = contracts_runtime.ToolAccess.workspace_backed(
        worktree,
        tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
    )

    result = asyncio.run(
        compat_runtime.NewSessionRuntime(
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
    tool_access = contracts_runtime.ToolAccess.workspace_backed(
        worktree,
        tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
    )

    result = asyncio.run(
        compat_runtime.NewSessionRuntime(
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
    tool_access = contracts_runtime.ToolAccess.workspace_backed(
        worktree,
        tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
    )

    result = asyncio.run(
        compat_runtime.NewSessionRuntime(
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
    tool_access = contracts_runtime.ToolAccess.workspace_backed(
        worktree,
        tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
    )
    adapter_resume_state = {
        "provider_session_id": "prepared:recovered-codex",
        "provider_state_dir_relpath": "codex-runtime-state/",
        "exact_transcript_match": False,
        "cursor": {"turn": 4},
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

            def _prepare_session(run_session: Any) -> _AdapterOwnedPreparedRunSession:
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
        compat_runtime.NewSessionRuntime(
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
    tool_access = contracts_runtime.ToolAccess.workspace_backed(
        worktree,
        tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
    )

    result = asyncio.run(
        compat_runtime.NewSessionRuntime(
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
    tool_access = contracts_runtime.ToolAccess.workspace_backed(
        worktree,
        tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
    )
    execution_adapter = _PreparedNotStartedUsageLimitNewSessionExecutionAdapter()

    result = asyncio.run(
        compat_runtime.NewSessionRuntime(
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
    tool_access = contracts_runtime.ToolAccess.workspace_backed(
        worktree,
        tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
    )

    result = asyncio.run(
        compat_runtime.NewSessionRuntime(
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
    tool_access = contracts_runtime.ToolAccess.workspace_backed(
        worktree,
        tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
    )
    execution_adapter = _PreparedNotStartedCancellationNewSessionExecutionAdapter()

    result = asyncio.run(
        compat_runtime.NewSessionRuntime(
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
    tool_access = contracts_runtime.ToolAccess.workspace_backed(
        worktree,
        tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
    )

    result = asyncio.run(
        compat_runtime.NewSessionRuntime(
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
    tool_access = contracts_runtime.ToolAccess.workspace_backed(
        worktree,
        tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
    )

    result = asyncio.run(
        compat_runtime.NewSessionRuntime(
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
    tool_access = contracts_runtime.ToolAccess.workspace_backed(
        worktree,
        tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
    )
    execution_adapter = _TimeoutResidentExecutionAdapter()

    result = asyncio.run(
        compat_runtime.NewSessionRuntime(
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
    tool_access = contracts_runtime.ToolAccess.workspace_backed(
        worktree,
        tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
    )

    result = asyncio.run(
        compat_runtime.NewSessionRuntime(
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
        tool_access=contracts_runtime.ToolAccess.workspace_backed(
            Path("/repo"),
            tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
        ),
    )

    with pytest.raises(runtime.RuntimeConfigurationError):
        asyncio.run(
            compat_runtime.NewSessionRuntime(
                execution_adapter=cast(Any, object()),
                service_registry=service_registry_factory("codex"),
            ).run_new_session(request)
        )

    with pytest.raises(AgentCredentialFailureError):
        asyncio.run(
            compat_runtime.NewSessionRuntime(
                execution_adapter=_CredentialFailureResidentExecutionAdapter(),
                service_registry=service_registry_factory("codex"),
            ).run_new_session(request)
        )

    with pytest.raises(HardAgentError):
        asyncio.run(
            compat_runtime.NewSessionRuntime(
                execution_adapter=_HardFailureResidentExecutionAdapter(),
                service_registry=service_registry_factory("codex"),
            ).run_new_session(request)
        )

    with pytest.raises(TransientAgentError):
        asyncio.run(
            compat_runtime.NewSessionRuntime(
                execution_adapter=_TransientProviderFailureResidentExecutionAdapter(),
                service_registry=service_registry_factory("codex"),
            ).run_new_session(request)
        )

    with pytest.raises(AgentFailedError):
        asyncio.run(
            compat_runtime.NewSessionRuntime(
                execution_adapter=_UnclassifiedProviderFailureResidentExecutionAdapter(),
                service_registry=service_registry_factory("codex"),
            ).run_new_session(request)
        )

    with pytest.raises(RuntimeError, match="unexpected failure"):
        asyncio.run(
            compat_runtime.NewSessionRuntime(
                execution_adapter=_UnexpectedFailureResidentExecutionAdapter(),
                service_registry=service_registry_factory("codex"),
            ).run_new_session(request)
        )


def test_runtime_client_writes_session_invocation_log_to_logs_dir_without_mixing_runtime_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stage_selection_factory: Callable[..., runtime.StageSelection],
) -> None:
    worktree = tmp_path / "worktree"
    runtime_state_dir = tmp_path / "runtime-state"
    logs_dir = tmp_path / "runtime-logs"
    worktree.mkdir()

    class _FakeProcess:
        stdout = iter(
            [
                (
                    '{"type":"assistant","message":{"content":[{"type":"text",'
                    '"text":"hello from claude"}],"usage":{"input_tokens":100}}}\n'
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

    outcome = asyncio.run(
        prompt_runtime.RuntimeClient().run_new_session(
            prompt_runtime.NewSessionRunRequest(
                prompt="already rendered prompt",
                worktree=worktree,
                runtime_state_dir=runtime_state_dir,
                stage=stage_selection_factory(
                    service="claude",
                    model="sonnet",
                    effort="medium",
                ),
                role=InvocationRole("implementer"),
                tool_access=contracts_runtime.ToolAccess.workspace_backed(worktree),
                provider_auth=prompt_runtime.ProviderAuth(
                    claude_code_oauth_token="token"
                ),
                logs_dir=logs_dir,
            )
        )
    )

    assert outcome.output == "hello from claude"
    assert outcome.usage == runtime.ProviderUsage(
        input_tokens=100,
        output_tokens=None,
        cache_read_input_tokens=None,
        cache_creation_input_tokens=None,
        cost_usd=None,
        duration_seconds=None,
    )
    assert list(worktree.rglob("*.log")) == []
    assert list(runtime_state_dir.rglob("*.log")) == []
    log_paths = sorted(logs_dir.glob("*.log"))
    assert len(log_paths) == 1
    log_text = log_paths[0].read_text()
    header = json.loads(log_text.splitlines()[0])
    assert isinstance(outcome.result, prompt_runtime.SessionRunResult)
    assert header == {
        "type": "agent_invocation",
        "invocation_role": "implementer",
        "run_kind": "fresh",
        "provider_session_id": outcome.result.runtime_metadata.provider_session_id,
        "prompt": "already rendered prompt",
    }
    assert '"type":"assistant"' in log_text
    assert '"type":"result"' in log_text


def test_runtime_client_new_opencode_session_uses_runtime_state_dir_and_relative_continuation(
    stage_selection_factory: Callable[..., runtime.StageSelection],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worktree = tmp_path / "worktree"
    runtime_state_dir = tmp_path / "runtime-state"
    worktree.mkdir()
    runtime_state_dir.mkdir()
    tool_access = contracts_runtime.ToolAccess.workspace_backed(
        worktree,
        tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
    )
    expected_state_relpath = session_runtime.provider_state_relpath(
        InvocationRole("implementer"),
        "opencode",
        "main",
    )
    expected_state_dir = runtime_state_dir / expected_state_relpath
    observed: dict[str, Any] = {}

    class _FakePopen:
        def __init__(
            self,
            command: str,
            *,
            shell: bool,
            cwd: Path,
            env: dict[str, str],
            stdout: Any,
            stderr: Any,
            text: bool,
        ) -> None:
            del stdout, stderr
            observed["command"] = command
            observed["shell"] = shell
            observed["cwd"] = cwd
            observed["env"] = env
            observed["text"] = text
            self.stdout = iter(
                [
                    json.dumps(
                        {
                            "type": "text",
                            "sessionID": "provider-session-123",
                            "part": {
                                "type": "text",
                                "text": "OpenCode answer",
                                "time": {"end": "2026-01-01T00:00:00Z"},
                            },
                        }
                    )
                    + "\n",
                    json.dumps(
                        {
                            "type": "session.status",
                            "status": {"type": "idle"},
                        }
                    )
                    + "\n",
                ]
            )
            self.stderr = iter(())

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(subprocess, "Popen", _FakePopen)

    result = asyncio.run(
        prompt_runtime.RuntimeClient().run_new_session(
            prompt_runtime.NewSessionRunRequest(
                prompt="already rendered prompt",
                worktree=worktree,
                runtime_state_dir=runtime_state_dir,
                stage=stage_selection_factory(
                    service="opencode",
                    model="glm-5",
                    effort="medium",
                ),
                role=InvocationRole("implementer"),
                session_namespace="main",
                provider_auth=prompt_runtime.ProviderAuth(opencode_api_key="test-key"),
                tool_access=tool_access,
            )
        )
    )

    assert observed["command"].startswith("opencode run --format json")
    assert "--model opencode-go/glm-5" in observed["command"]
    assert observed["env"]["OPENCODE_HOME"] == str(expected_state_dir)
    assert expected_state_dir.is_dir()
    assert isinstance(result.result, prompt_runtime.SessionRunResult)
    assert result.result.continuation == prompt_runtime.Continuation(
        selected_service="opencode",
        selected_model="glm-5",
        selected_effort="medium",
        tool_access=tool_access,
        provider_resume_state={
            "provider_session_id": "provider-session-123",
            "provider_state": {"session_id": "provider-session-123"},
            "exact_transcript_match": False,
        },
    )


def test_runtime_client_new_opencode_session_resumes_recovered_state_dir_session_id(
    stage_selection_factory: Callable[..., runtime.StageSelection],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worktree = tmp_path / "worktree"
    runtime_state_dir = tmp_path / "runtime-state"
    worktree.mkdir()
    runtime_state_dir.mkdir()
    tool_access = contracts_runtime.ToolAccess.workspace_backed(
        worktree,
        tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
    )
    provider_state_dir_relpath = session_runtime.provider_state_relpath(
        InvocationRole("implementer"),
        "opencode",
        "main",
    )
    provider_state_dir = runtime_state_dir / provider_state_dir_relpath
    provider_state_dir.mkdir(parents=True)
    (provider_state_dir / "resume.jsonl").write_text("[]", encoding="utf-8")
    (provider_state_dir / "session_id").write_text(
        "recovered-state-dir-session\n",
        encoding="utf-8",
    )
    observed: dict[str, Any] = {}

    class _FakePopen:
        def __init__(
            self,
            command: str,
            *,
            shell: bool,
            cwd: Path,
            env: dict[str, str],
            stdout: Any,
            stderr: Any,
            text: bool,
        ) -> None:
            del stdout, stderr
            observed["command"] = command
            observed["shell"] = shell
            observed["cwd"] = cwd
            observed["env"] = env
            observed["text"] = text
            self.stdout = iter(
                [
                    json.dumps(
                        {
                            "type": "text",
                            "sessionID": "continued-session-2",
                            "part": {
                                "type": "text",
                                "text": "continued answer",
                                "time": {"end": "2026-01-01T00:00:00Z"},
                            },
                        }
                    )
                    + "\n",
                    json.dumps(
                        {
                            "type": "session.status",
                            "status": {"type": "idle"},
                        }
                    )
                    + "\n",
                ]
            )
            self.stderr = iter(())

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(subprocess, "Popen", _FakePopen)

    result = asyncio.run(
        prompt_runtime.RuntimeClient().run_new_session(
            prompt_runtime.NewSessionRunRequest(
                prompt="already rendered prompt",
                worktree=worktree,
                runtime_state_dir=runtime_state_dir,
                stage=stage_selection_factory(
                    service="opencode",
                    model="glm-5",
                    effort="medium",
                ),
                role=InvocationRole("implementer"),
                session_namespace="main",
                provider_auth=prompt_runtime.ProviderAuth(opencode_api_key="test-key"),
                tool_access=tool_access,
            )
        )
    )

    assert "--session recovered-state-dir-session" in observed["command"]
    assert isinstance(result.result, prompt_runtime.SessionRunResult)
    assert result.result.runtime_metadata == prompt_runtime.SessionRuntimeMetadata(
        service_name="opencode",
        provider_session_id="continued-session-2",
        run_kind=RunKind.RESUME,
        session_namespace="main",
        exact_transcript_match=True,
    )
    assert result.result.runtime_metadata.selected_model == "glm-5"
    assert result.result.runtime_metadata.selected_effort == "medium"
    assert (
        result.result.runtime_metadata.tool_policy
        == runtime.ToolPolicy.NO_FILE_MUTATION
    )
    assert result.result.continuation == prompt_runtime.Continuation(
        selected_service="opencode",
        selected_model="glm-5",
        selected_effort="medium",
        tool_access=tool_access,
        provider_resume_state={
            "provider_session_id": "continued-session-2",
            "provider_state": {
                "session_id": "continued-session-2",
                "resume_jsonl": "[]",
            },
            "exact_transcript_match": True,
        },
    )
    assert (provider_state_dir / "session_id").read_text(encoding="utf-8").strip() == (
        "continued-session-2"
    )


def test_runtime_client_new_opencode_session_keeps_observed_session_id_on_started_usage_limit(
    stage_selection_factory: Callable[..., runtime.StageSelection],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worktree = tmp_path / "worktree"
    runtime_state_dir = tmp_path / "runtime-state"
    worktree.mkdir()
    runtime_state_dir.mkdir()
    tool_access = contracts_runtime.ToolAccess.workspace_backed(
        worktree,
        tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
    )

    class _FakePopen:
        def __init__(
            self,
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
            self.stdout = iter(
                [
                    json.dumps(
                        {
                            "type": "text",
                            "sessionID": "provider-session-123",
                            "part": {
                                "type": "text",
                                "text": "OpenCode answer",
                                "time": {"end": "2026-01-01T00:00:00Z"},
                            },
                        }
                    )
                    + "\n",
                    json.dumps(
                        {
                            "type": "error",
                            "sessionID": "provider-session-123",
                            "error": {
                                "name": "RateLimitError",
                                "data": {
                                    "message": "You have reached your OpenCode Go usage limit.",
                                    "statusCode": 429,
                                    "isRetryable": True,
                                },
                            },
                        }
                    )
                    + "\n",
                ]
            )
            self.stderr = iter(())

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(subprocess, "Popen", _FakePopen)

    result = asyncio.run(
        prompt_runtime.RuntimeClient().run_new_session(
            prompt_runtime.NewSessionRunRequest(
                prompt="already rendered prompt",
                worktree=worktree,
                runtime_state_dir=runtime_state_dir,
                stage=stage_selection_factory(
                    service="opencode",
                    model="glm-5",
                    effort="medium",
                ),
                role=InvocationRole("implementer"),
                session_namespace="main",
                provider_auth=prompt_runtime.ProviderAuth(opencode_api_key="test-key"),
                tool_access=tool_access,
            )
        )
    )

    assert result == prompt_runtime.RuntimeOutcome.usage_limited(
        output="",
        service_name="opencode",
        reset_time=None,
        usage_limit_scope=None,
        invocation_progress=runtime.InvocationProgress.STARTED,
        continuation=prompt_runtime.Continuation(
            selected_service="opencode",
            selected_model="glm-5",
            selected_effort="medium",
            tool_access=tool_access,
            provider_resume_state={
                "provider_session_id": "provider-session-123",
                "provider_state": {},
                "exact_transcript_match": False,
            },
        ),
    )


def test_runtime_client_writes_new_opencode_session_invocation_log_header_with_observed_provider_session_id(
    stage_selection_factory: Callable[..., runtime.StageSelection],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worktree = tmp_path / "worktree"
    runtime_state_dir = tmp_path / "runtime-state"
    logs_dir = tmp_path / "runtime-logs"
    worktree.mkdir()
    runtime_state_dir.mkdir()

    class _FakePopen:
        def __init__(
            self,
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
            self.stdout = iter(
                [
                    json.dumps(
                        {
                            "type": "text",
                            "sessionID": "provider-session-123",
                            "part": {
                                "type": "text",
                                "text": "OpenCode answer",
                                "time": {"end": "2026-01-01T00:00:00Z"},
                            },
                        }
                    )
                    + "\n",
                    json.dumps(
                        {
                            "type": "session.status",
                            "status": {"type": "idle"},
                        }
                    )
                    + "\n",
                ]
            )
            self.stderr = iter(())

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(subprocess, "Popen", _FakePopen)

    outcome = asyncio.run(
        prompt_runtime.RuntimeClient().run_new_session(
            prompt_runtime.NewSessionRunRequest(
                prompt="already rendered prompt",
                worktree=worktree,
                runtime_state_dir=runtime_state_dir,
                stage=stage_selection_factory(
                    service="opencode",
                    model="glm-5",
                    effort="medium",
                ),
                role=InvocationRole("implementer"),
                session_namespace="main",
                provider_auth=prompt_runtime.ProviderAuth(opencode_api_key="test-key"),
                tool_access=contracts_runtime.ToolAccess.no_tools(),
                logs_dir=logs_dir,
            )
        )
    )

    assert outcome.output == "OpenCode answer"
    assert list(worktree.rglob("*.log")) == []
    assert list(runtime_state_dir.rglob("*.log")) == []
    log_paths = sorted(logs_dir.glob("*.log"))
    assert len(log_paths) == 1
    log_text = log_paths[0].read_text(encoding="utf-8")
    header = json.loads(log_text.splitlines()[0])
    assert isinstance(outcome.result, prompt_runtime.SessionRunResult)
    assert header == {
        "type": "agent_invocation",
        "invocation_role": "implementer",
        "run_kind": "fresh",
        "provider_session_id": outcome.result.runtime_metadata.provider_session_id,
        "prompt": "already rendered prompt",
    }
    assert '"sessionID": "provider-session-123"' in log_text
