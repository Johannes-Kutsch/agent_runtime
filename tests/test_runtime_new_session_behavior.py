from __future__ import annotations

import asyncio
import json
import subprocess
import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import pytest

import agent_runtime as runtime
import agent_runtime.contracts as contracts_runtime
import agent_runtime._runtime_compat as compat_runtime
import agent_runtime._provider_session_adapter as provider_session_adapter_runtime
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
from agent_runtime._execution_contracts import (
    WorkExecutionAdapter,
    WorkExecutionDependencies,
    WorkFailureHandling,
    WorkInvocationDependencies,
    WorkPresentationDependencies,
    WorktreeMount,
)
from agent_runtime.provider_output import reduce_text_output_events
from agent_runtime._provider_session_adapter import ProviderSessionPlanningRequest
from agent_runtime.roles import InvocationRole
from agent_runtime._service_registry import ServiceRegistry
from agent_runtime.session import RunKind

from tests.runtime_boundary_fakes import (
    ExecutionServiceFake as _ExecutionService,
    SessionStoreFake as _SessionStore,
)


def _observed_command_text(command: str | tuple[str, ...]) -> str:
    return command if isinstance(command, str) else " ".join(command)


def _assert_runtime_outcome(
    actual: prompt_runtime.RuntimeOutcome,
    expected: prompt_runtime.RuntimeOutcome,
) -> None:
    assert actual == dataclasses.replace(
        expected, invocation_records=actual.invocation_records
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
            account_label="team-account-1",
            usage=runtime.ProviderUsage(input_tokens=128),
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
            lambda _turn, _raw: None,
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
            lambda _turn, _raw: None,
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
            lambda _turn, _raw: None,
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


def test_new_session_runtime_requires_selected_configured_service(
    stage_selection_factory: Callable[..., runtime.ProviderSelection],
    service_registry_factory: Callable[..., ServiceRegistry],
    session_store_factory: Callable[..., _SessionStore],
) -> None:
    worktree = Path("/repo")
    tool_access = contracts_runtime.ToolAccess.workspace_backed(
        worktree,
        tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
    )

    with pytest.raises(runtime.RuntimeConfigurationError):
        asyncio.run(
            compat_runtime.NewSessionRuntime(
                execution_adapter=_RuntimePlannedPathResidentExecutionAdapter(),
                service_registry=service_registry_factory("claude"),
                session_store=session_store_factory(),
                provider_session_adapter=_NamedExternalStateResidentPlanningProviderSessionAdapter(
                    "claude"
                ),
            ).run_new_session(
                prompt_runtime.NewSessionRunRequest(
                    prompt="already rendered prompt",
                    worktree=WorktreeMount(worktree),
                    provider_selection=stage_selection_factory(
                        service="missing",
                        fallback=stage_selection_factory(
                            service="claude",
                            model="sonnet",
                            effort="high",
                        ),
                    ),
                    role=InvocationRole("implementer"),
                    session_namespace="main",
                    tool_access=tool_access,
                )
            )
        )


def test_new_session_runtime_reports_selected_usage_limit_without_fallback(
    stage_selection_factory: Callable[..., runtime.ProviderSelection],
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
            session_store=session_store_factory(),
            provider_session_adapter=_NamedExternalStateResidentPlanningProviderSessionAdapter(
                "claude"
            ),
        ).run_new_session(
            prompt_runtime.NewSessionRunRequest(
                prompt="already rendered prompt",
                worktree=WorktreeMount(worktree),
                provider_selection=stage_selection_factory(
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
                tool_access=tool_access,
            )
        )
    )

    _assert_runtime_outcome(
        result,
        prompt_runtime.RuntimeOutcome.usage_limited(
            output="",
            service_name="codex",
            reset_time=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
            invocation_progress=runtime.InvocationProgress.NOT_STARTED,
        ),
    )
    assert result.continuation is None


def test_new_session_runtime_keeps_started_usage_limit_outcome(
    stage_selection_factory: Callable[..., runtime.ProviderSelection],
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
            session_store=session_store_factory(),
            provider_session_adapter=_NamedExternalStateResidentPlanningProviderSessionAdapter(
                "codex"
            ),
        ).run_new_session(
            prompt_runtime.NewSessionRunRequest(
                prompt="already rendered prompt",
                worktree=WorktreeMount(worktree),
                provider_selection=stage_selection_factory(
                    service="codex",
                    model="gpt-5.4",
                    effort="medium",
                ),
                role=InvocationRole("implementer"),
                session_namespace="main",
                tool_access=tool_access,
            )
        )
    )

    _assert_runtime_outcome(
        result,
        prompt_runtime.RuntimeOutcome.usage_limited(
            output="",
            service_name="codex",
            reset_time=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
            invocation_progress=runtime.InvocationProgress.STARTED,
            account_label="team-account-1",
            usage=runtime.ProviderUsage(input_tokens=128),
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
        ),
    )


def test_new_session_runtime_returns_continuation_for_started_interruption(
    stage_selection_factory: Callable[..., runtime.ProviderSelection],
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
            session_store=session_store_factory(),
            provider_session_adapter=_NamedExternalStateResidentPlanningProviderSessionAdapter(
                "codex"
            ),
        ).run_new_session(
            prompt_runtime.NewSessionRunRequest(
                prompt="already rendered prompt",
                worktree=WorktreeMount(worktree),
                provider_selection=stage_selection_factory(
                    service="codex",
                    model="gpt-5.4",
                    effort="medium",
                ),
                role=InvocationRole("implementer"),
                session_namespace="main",
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
    stage_selection_factory: Callable[..., runtime.ProviderSelection],
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
            session_store=session_store_factory(),
            provider_session_adapter=_NamedExternalStateResidentPlanningProviderSessionAdapter(
                "codex"
            ),
        ).run_new_session(
            prompt_runtime.NewSessionRunRequest(
                prompt="already rendered prompt",
                worktree=WorktreeMount(worktree),
                provider_selection=stage_selection_factory(
                    service="codex",
                    model="gpt-5.4",
                    effort="medium",
                ),
                role=InvocationRole("implementer"),
                session_namespace="main",
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


def test_new_session_runtime_keeps_not_started_usage_limit_without_continuation(
    stage_selection_factory: Callable[..., runtime.ProviderSelection],
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
            session_store=session_store_factory(),
            provider_session_adapter=_NamedExternalStateResidentPlanningProviderSessionAdapter(
                "codex"
            ),
        ).run_new_session(
            prompt_runtime.NewSessionRunRequest(
                prompt="already rendered prompt",
                worktree=WorktreeMount(worktree),
                provider_selection=stage_selection_factory(
                    service="codex",
                    model="gpt-5.4",
                    effort="medium",
                ),
                role=InvocationRole("implementer"),
                session_namespace="main",
                tool_access=tool_access,
            )
        )
    )

    _assert_runtime_outcome(
        result,
        prompt_runtime.RuntimeOutcome.usage_limited(
            output="",
            service_name="codex",
            reset_time=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
            invocation_progress=runtime.InvocationProgress.NOT_STARTED,
        ),
    )
    assert result.continuation is None


def test_new_session_runtime_does_not_create_continuation_from_session_allocation_alone(
    stage_selection_factory: Callable[..., runtime.ProviderSelection],
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
            session_store=session_store_factory(),
            provider_session_adapter=_NamedExternalStateResidentPlanningProviderSessionAdapter(
                "codex"
            ),
        ).run_new_session(
            prompt_runtime.NewSessionRunRequest(
                prompt="already rendered prompt",
                worktree=WorktreeMount(worktree),
                provider_selection=stage_selection_factory(
                    service="codex",
                    model="gpt-5.4",
                    effort="medium",
                ),
                role=InvocationRole("implementer"),
                session_namespace="main",
                tool_access=tool_access,
            )
        )
    )

    assert execution_adapter.prepare_session_calls == 1
    assert result.invocation_progress is runtime.InvocationProgress.NOT_STARTED
    assert result.continuation is None


def test_new_session_runtime_returns_continuation_for_started_cancellation(
    stage_selection_factory: Callable[..., runtime.ProviderSelection],
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
            session_store=session_store_factory(),
            provider_session_adapter=_NamedExternalStateResidentPlanningProviderSessionAdapter(
                "codex"
            ),
        ).run_new_session(
            prompt_runtime.NewSessionRunRequest(
                prompt="already rendered prompt",
                worktree=WorktreeMount(worktree),
                provider_selection=stage_selection_factory(
                    service="codex",
                    model="gpt-5.4",
                    effort="medium",
                ),
                role=InvocationRole("implementer"),
                session_namespace="main",
                tool_access=tool_access,
            )
        )
    )

    _assert_runtime_outcome(
        result,
        prompt_runtime.RuntimeOutcome.cancelled(
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
        ),
    )


def test_new_session_runtime_keeps_not_started_cancellation_without_continuation(
    stage_selection_factory: Callable[..., runtime.ProviderSelection],
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
            session_store=session_store_factory(),
            provider_session_adapter=_NamedExternalStateResidentPlanningProviderSessionAdapter(
                "codex"
            ),
        ).run_new_session(
            prompt_runtime.NewSessionRunRequest(
                prompt="already rendered prompt",
                worktree=WorktreeMount(worktree),
                provider_selection=stage_selection_factory(
                    service="codex",
                    model="gpt-5.4",
                    effort="medium",
                ),
                role=InvocationRole("implementer"),
                session_namespace="main",
                tool_access=tool_access,
            )
        )
    )

    assert execution_adapter.prepare_session_calls == 1
    _assert_runtime_outcome(
        result,
        prompt_runtime.RuntimeOutcome.cancelled(
            output="",
            invocation_progress=runtime.InvocationProgress.NOT_STARTED,
        ),
    )
    assert result.continuation is None


def test_new_session_runtime_returns_timed_out_outcome_with_continuation_after_model_activity(
    stage_selection_factory: Callable[..., runtime.ProviderSelection],
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
            session_store=session_store_factory(),
            provider_session_adapter=_NamedExternalStateResidentPlanningProviderSessionAdapter(
                "codex"
            ),
        ).run_new_session(
            prompt_runtime.NewSessionRunRequest(
                prompt="already rendered prompt",
                worktree=WorktreeMount(worktree),
                provider_selection=stage_selection_factory(
                    service="codex",
                    model="gpt-5.4",
                    effort="medium",
                ),
                role=InvocationRole("implementer"),
                session_namespace="main",
                tool_access=tool_access,
            )
        )
    )

    _assert_runtime_outcome(
        result,
        prompt_runtime.RuntimeOutcome.timed_out(
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
        ),
    )


def test_new_session_runtime_returns_retryable_provider_failure_outcome_with_continuation_after_model_activity(
    stage_selection_factory: Callable[..., runtime.ProviderSelection],
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
            session_store=session_store_factory(),
            provider_session_adapter=_NamedExternalStateResidentPlanningProviderSessionAdapter(
                "codex"
            ),
        ).run_new_session(
            prompt_runtime.NewSessionRunRequest(
                prompt="already rendered prompt",
                worktree=WorktreeMount(worktree),
                provider_selection=stage_selection_factory(
                    service="codex",
                    model="gpt-5.4",
                    effort="medium",
                ),
                role=InvocationRole("implementer"),
                session_namespace="main",
                tool_access=tool_access,
            )
        )
    )

    _assert_runtime_outcome(
        result,
        prompt_runtime.RuntimeOutcome.retryable_provider_failure(
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
        ),
    )


def test_new_session_runtime_keeps_not_started_timeout_without_continuation(
    stage_selection_factory: Callable[..., runtime.ProviderSelection],
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
            session_store=session_store_factory(),
            provider_session_adapter=_NamedExternalStateResidentPlanningProviderSessionAdapter(
                "codex"
            ),
        ).run_new_session(
            prompt_runtime.NewSessionRunRequest(
                prompt="already rendered prompt",
                worktree=WorktreeMount(worktree),
                provider_selection=stage_selection_factory(
                    service="codex",
                    model="gpt-5.4",
                    effort="medium",
                ),
                role=InvocationRole("implementer"),
                session_namespace="main",
                tool_access=tool_access,
            )
        )
    )

    _assert_runtime_outcome(
        result,
        prompt_runtime.RuntimeOutcome.timed_out(
            output="",
            invocation_progress=runtime.InvocationProgress.NOT_STARTED,
        ),
    )
    assert result.continuation is None


def test_new_session_runtime_keeps_not_started_retryable_provider_failure_without_continuation(
    stage_selection_factory: Callable[..., runtime.ProviderSelection],
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
            session_store=session_store_factory(),
            provider_session_adapter=_NamedExternalStateResidentPlanningProviderSessionAdapter(
                "codex"
            ),
        ).run_new_session(
            prompt_runtime.NewSessionRunRequest(
                prompt="already rendered prompt",
                worktree=WorktreeMount(worktree),
                provider_selection=stage_selection_factory(
                    service="codex",
                    model="gpt-5.4",
                    effort="medium",
                ),
                role=InvocationRole("implementer"),
                session_namespace="main",
                tool_access=tool_access,
            )
        )
    )

    _assert_runtime_outcome(
        result,
        prompt_runtime.RuntimeOutcome.retryable_provider_failure(
            output="",
            service_name="codex",
            invocation_progress=runtime.InvocationProgress.NOT_STARTED,
        ),
    )
    assert result.continuation is None


def _seed_codex_host_auth(monkeypatch: pytest.MonkeyPatch, home_dir: Path) -> Path:
    auth_dir = home_dir / ".codex"
    auth_dir.mkdir(parents=True)
    auth_path = auth_dir / "auth.json"
    auth_path.write_text('{"access_token":"token"}', encoding="utf-8")
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module,
        "_codex_host_auth_path",
        lambda: auth_path,
        raising=False,
    )
    return auth_path


def _seed_empty_codex_host_auth(
    monkeypatch: pytest.MonkeyPatch,
    home_dir: Path,
) -> Path:
    auth_path = home_dir / ".codex" / "auth.json"
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module,
        "_codex_host_auth_path",
        lambda: auth_path,
        raising=False,
    )
    return auth_path


def _stub_codex_prompt_path(monkeypatch: pytest.MonkeyPatch) -> None:
    prompt_path = Path("/tmp/.provider_prompt")
    original_write_text = Path.write_text
    original_unlink = Path.unlink

    def _fake_write_text(self: Path, data: str, *args: Any, **kwargs: Any) -> int:
        if self == prompt_path:
            return len(data)
        return original_write_text(self, data, *args, **kwargs)

    def _fake_unlink(self: Path, *args: Any, **kwargs: Any) -> None:
        if self == prompt_path:
            return None
        return original_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", _fake_write_text)
    monkeypatch.setattr(Path, "unlink", _fake_unlink)


def test_new_session_runtime_keeps_exceptional_failures_exceptional(
    stage_selection_factory: Callable[..., runtime.ProviderSelection],
    service_registry_factory: Callable[..., ServiceRegistry],
    session_store_factory: Callable[..., _SessionStore],
) -> None:
    request = prompt_runtime.NewSessionRunRequest(
        prompt="already rendered prompt",
        worktree=WorktreeMount(Path("/repo")),
        provider_selection=stage_selection_factory(
            service="codex",
            model="gpt-5.4",
            effort="medium",
        ),
        role=InvocationRole("implementer"),
        session_namespace="main",
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
                session_store=session_store_factory(),
                provider_session_adapter=_NamedExternalStateResidentPlanningProviderSessionAdapter(
                    "codex"
                ),
            ).run_new_session(request)
        )

    with pytest.raises(AgentCredentialFailureError):
        asyncio.run(
            compat_runtime.NewSessionRuntime(
                execution_adapter=_CredentialFailureResidentExecutionAdapter(),
                service_registry=service_registry_factory("codex"),
                session_store=session_store_factory(),
                provider_session_adapter=_NamedExternalStateResidentPlanningProviderSessionAdapter(
                    "codex"
                ),
            ).run_new_session(request)
        )

    with pytest.raises(HardAgentError):
        asyncio.run(
            compat_runtime.NewSessionRuntime(
                execution_adapter=_HardFailureResidentExecutionAdapter(),
                service_registry=service_registry_factory("codex"),
                session_store=session_store_factory(),
                provider_session_adapter=_NamedExternalStateResidentPlanningProviderSessionAdapter(
                    "codex"
                ),
            ).run_new_session(request)
        )

    with pytest.raises(TransientAgentError):
        asyncio.run(
            compat_runtime.NewSessionRuntime(
                execution_adapter=_TransientProviderFailureResidentExecutionAdapter(),
                service_registry=service_registry_factory("codex"),
                session_store=session_store_factory(),
                provider_session_adapter=_NamedExternalStateResidentPlanningProviderSessionAdapter(
                    "codex"
                ),
            ).run_new_session(request)
        )

    with pytest.raises(AgentFailedError):
        asyncio.run(
            compat_runtime.NewSessionRuntime(
                execution_adapter=_UnclassifiedProviderFailureResidentExecutionAdapter(),
                service_registry=service_registry_factory("codex"),
                session_store=session_store_factory(),
                provider_session_adapter=_NamedExternalStateResidentPlanningProviderSessionAdapter(
                    "codex"
                ),
            ).run_new_session(request)
        )

    with pytest.raises(RuntimeError, match="unexpected failure"):
        asyncio.run(
            compat_runtime.NewSessionRuntime(
                execution_adapter=_UnexpectedFailureResidentExecutionAdapter(),
                service_registry=service_registry_factory("codex"),
                session_store=session_store_factory(),
                provider_session_adapter=_NamedExternalStateResidentPlanningProviderSessionAdapter(
                    "codex"
                ),
            ).run_new_session(request)
        )


def test_runtime_client_returns_claude_invocation_record_without_mixing_runtime_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stage_selection_factory: Callable[..., runtime.ProviderSelection],
) -> None:
    worktree = tmp_path / "worktree"
    runtime_state_dir = tmp_path / "runtime-state"
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
                provider_selection=stage_selection_factory(
                    service="claude",
                    model="sonnet",
                    effort="medium",
                    auth=prompt_runtime.ProviderAuth(claude_code_oauth_token="token"),
                ),
                role=InvocationRole("implementer"),
                tool_access=contracts_runtime.ToolAccess.workspace_backed(worktree),
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
    assert isinstance(outcome.result, prompt_runtime.SessionRunResult)
    invocation_record = cast(
        prompt_runtime.InvocationRecord, outcome.invocation_records[0]
    )
    assert invocation_record.run_kind is RunKind.FRESH
    assert invocation_record.service_name == "claude"
    assert (
        invocation_record.provider_session_id
        == outcome.result.runtime_metadata.provider_session_id
    )
    assert invocation_record.prompt == "already rendered prompt"
    assert invocation_record.provider_output is not None
    assert b'"type":"assistant"' in invocation_record.provider_output
    assert (
        b'"type":"result","result":"hello from claude"'
        in invocation_record.provider_output
    )


def test_runtime_client_new_codex_session_reports_isolated_missing_host_auth(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stage_selection_factory: Callable[..., runtime.ProviderSelection],
) -> None:
    worktree = tmp_path / "worktree"
    runtime_state_dir = tmp_path / "runtime-state"
    host_home_with_login = tmp_path / "host-home-with-login"
    worktree.mkdir()
    runtime_state_dir.mkdir()
    _seed_codex_host_auth(monkeypatch, host_home_with_login)
    _seed_empty_codex_host_auth(monkeypatch, tmp_path / "isolated-home-without-login")

    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module.Path,
        "home",
        lambda: host_home_with_login,
    )

    def _unexpected_popen(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise AssertionError("subprocess should not start without isolated host auth")

    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module.subprocess,
        "Popen",
        _unexpected_popen,
    )

    with pytest.raises(AgentCredentialFailureError) as exc_info:
        asyncio.run(
            prompt_runtime.RuntimeClient().run_new_session(
                prompt_runtime.NewSessionRunRequest(
                    prompt="already rendered prompt",
                    worktree=worktree,
                    runtime_state_dir=runtime_state_dir,
                    provider_selection=stage_selection_factory(
                        service="codex",
                        model="gpt-5.4",
                        effort="medium",
                    ),
                    role=InvocationRole("implementer"),
                    session_namespace="main",
                    tool_access=contracts_runtime.ToolAccess.workspace_backed(worktree),
                )
            )
        )

    assert str(exc_info.value) == (
        "Codex authentication missing: run `codex login` on the host."
    )
    assert exc_info.value.service_name == "codex"
    assert exc_info.value.status_code == 401


def test_runtime_client_new_codex_session_uses_isolated_present_host_auth(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stage_selection_factory: Callable[..., runtime.ProviderSelection],
) -> None:
    worktree = tmp_path / "worktree"
    runtime_state_dir = tmp_path / "runtime-state"
    host_home_without_login = tmp_path / "host-home-without-login"
    worktree.mkdir()
    runtime_state_dir.mkdir()
    host_home_without_login.mkdir()
    isolated_auth_path = _seed_codex_host_auth(
        monkeypatch, tmp_path / "isolated-home-with-login"
    )
    _stub_codex_prompt_path(monkeypatch)

    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module.Path,
        "home",
        lambda: host_home_without_login,
    )

    class _FakeProcess:
        stdout = iter(
            (
                '{"type":"thread.started","thread_id":"thread-123"}\n',
                '{"type":"item.completed","item":{"type":"agent_message","text":"hello from codex"}}\n',
                '{"type":"turn.completed"}\n',
            )
        )
        stderr = iter(())

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
        del command, shell, stdout, stderr, text
        assert cwd == worktree
        assert env["CODEX_HOME"].startswith(str(runtime_state_dir))
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
                provider_selection=stage_selection_factory(
                    service="codex",
                    model="gpt-5.4",
                    effort="medium",
                ),
                role=InvocationRole("implementer"),
                session_namespace="main",
                tool_access=contracts_runtime.ToolAccess.workspace_backed(worktree),
            )
        )
    )

    provider_state_dir = runtime_state_dir / session_runtime.provider_state_relpath(
        InvocationRole("implementer"),
        "codex",
        "main",
    )
    assert provider_state_dir.joinpath("auth.json").read_text(encoding="utf-8") == (
        isolated_auth_path.read_text(encoding="utf-8")
    )
    assert outcome.output == "hello from codex"
    assert isinstance(outcome.result, prompt_runtime.SessionRunResult)
    assert outcome.result.runtime_metadata == prompt_runtime.SessionRuntimeMetadata(
        service_name="codex",
        provider_session_id="thread-123",
        run_kind=RunKind.FRESH,
        session_namespace="main",
        exact_transcript_match=False,
        selected_model="gpt-5.4",
        selected_effort="medium",
        tool_policy=runtime.ToolPolicy.UNRESTRICTED,
    )
    assert outcome.result.continuation == prompt_runtime.Continuation(
        selected_service="codex",
        selected_model="gpt-5.4",
        selected_effort="medium",
        tool_access=contracts_runtime.ToolAccess.workspace_backed(worktree),
        provider_resume_state={
            "provider_session_id": "thread-123",
            "provider_state_dir_relpath": "implementer/main/codex/",
            "run_kind": "resume",
            "exact_transcript_match": False,
        },
    )


def test_runtime_client_new_opencode_session_uses_runtime_state_dir_and_relative_continuation(
    stage_selection_factory: Callable[..., runtime.ProviderSelection],
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
    prompt_path = worktree / ".provider_prompt"
    observed: dict[str, Any] = {}

    class _Stdin:
        def write(self, data: str) -> None:
            observed["prompt"] = data

        def close(self) -> None:
            observed["stdin_closed"] = True

    class _FakePopen:
        def __init__(
            self,
            command: str | tuple[str, ...],
            *,
            shell: bool,
            cwd: Path,
            env: dict[str, str],
            stdout: Any,
            stderr: Any,
            text: bool,
            stdin: Any | None = None,
        ) -> None:
            del stdout, stderr, stdin
            observed["command"] = _observed_command_text(command)
            observed["shell"] = shell
            observed["cwd"] = cwd
            observed["env"] = env
            observed["text"] = text
            self.stdin = _Stdin()
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
                provider_selection=stage_selection_factory(
                    service="opencode",
                    model="glm-5.2",
                    effort="medium",
                    auth=prompt_runtime.ProviderAuth(opencode_api_key="test-key"),
                ),
                role=InvocationRole("implementer"),
                session_namespace="main",
                tool_access=tool_access,
            )
        )
    )

    opencode_executable = prompt_runtime._opencode_command(
        model="glm-5.2", effort="medium"
    )[0]
    assert observed["command"].startswith(f"{opencode_executable} run --format json")
    assert "--model opencode-go/glm-5.2" in observed["command"]
    assert observed["shell"] is False
    assert observed["prompt"] == "already rendered prompt"
    assert observed["stdin_closed"] is True
    assert observed["env"]["OPENCODE_HOME"] == str(expected_state_dir)
    assert expected_state_dir.is_dir()
    assert not prompt_path.exists()
    assert isinstance(result.result, prompt_runtime.SessionRunResult)
    assert result.result.continuation == prompt_runtime.Continuation(
        selected_service="opencode",
        selected_model="glm-5.2",
        selected_effort="medium",
        tool_access=tool_access,
        provider_resume_state={
            "provider_session_id": "provider-session-123",
            "provider_state": {"session_id": "provider-session-123"},
            "exact_transcript_match": False,
        },
    )


def test_runtime_client_new_opencode_session_resumes_recovered_state_dir_session_id(
    stage_selection_factory: Callable[..., runtime.ProviderSelection],
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
    prompt_path = worktree / ".provider_prompt"
    observed: dict[str, Any] = {}

    class _Stdin:
        def write(self, data: str) -> None:
            observed["prompt"] = data

        def close(self) -> None:
            observed["stdin_closed"] = True

    class _FakePopen:
        def __init__(
            self,
            command: str | tuple[str, ...],
            *,
            shell: bool,
            cwd: Path,
            env: dict[str, str],
            stdout: Any,
            stderr: Any,
            text: bool,
            stdin: Any | None = None,
        ) -> None:
            del stdout, stderr, stdin
            observed["command"] = _observed_command_text(command)
            observed["shell"] = shell
            observed["cwd"] = cwd
            observed["env"] = env
            observed["text"] = text
            self.stdin = _Stdin()
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
                provider_selection=stage_selection_factory(
                    service="opencode",
                    model="glm-5.2",
                    effort="medium",
                    auth=prompt_runtime.ProviderAuth(opencode_api_key="test-key"),
                ),
                role=InvocationRole("implementer"),
                session_namespace="main",
                tool_access=tool_access,
            )
        )
    )

    assert "--session recovered-state-dir-session" in observed["command"]
    assert observed["shell"] is False
    assert observed["prompt"] == "already rendered prompt"
    assert observed["stdin_closed"] is True
    assert not prompt_path.exists()
    assert isinstance(result.result, prompt_runtime.SessionRunResult)
    assert result.result.runtime_metadata == prompt_runtime.SessionRuntimeMetadata(
        service_name="opencode",
        provider_session_id="continued-session-2",
        run_kind=RunKind.RESUME,
        session_namespace="main",
        exact_transcript_match=True,
    )
    assert result.result.runtime_metadata.selected_model == "glm-5.2"
    assert result.result.runtime_metadata.selected_effort == "medium"
    assert (
        result.result.runtime_metadata.tool_policy
        == runtime.ToolPolicy.NO_FILE_MUTATION
    )
    assert result.result.continuation == prompt_runtime.Continuation(
        selected_service="opencode",
        selected_model="glm-5.2",
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
    stage_selection_factory: Callable[..., runtime.ProviderSelection],
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
                provider_selection=stage_selection_factory(
                    service="opencode",
                    model="glm-5.2",
                    effort="medium",
                    auth=prompt_runtime.ProviderAuth(opencode_api_key="test-key"),
                ),
                role=InvocationRole("implementer"),
                session_namespace="main",
                tool_access=tool_access,
            )
        )
    )

    _assert_runtime_outcome(
        result,
        prompt_runtime.RuntimeOutcome.usage_limited(
            output="",
            service_name="opencode",
            reset_time=None,
            invocation_progress=runtime.InvocationProgress.STARTED,
            continuation=prompt_runtime.Continuation(
                selected_service="opencode",
                selected_model="glm-5.2",
                selected_effort="medium",
                tool_access=tool_access,
                provider_resume_state={
                    "provider_session_id": "provider-session-123",
                    "provider_state": {},
                    "exact_transcript_match": False,
                },
            ),
        ),
    )


def test_runtime_client_returns_new_opencode_invocation_record_with_observed_provider_session_id(
    stage_selection_factory: Callable[..., runtime.ProviderSelection],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worktree = tmp_path / "worktree"
    runtime_state_dir = tmp_path / "runtime-state"
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
                provider_selection=stage_selection_factory(
                    service="opencode",
                    model="glm-5.2",
                    effort="medium",
                    auth=prompt_runtime.ProviderAuth(opencode_api_key="test-key"),
                ),
                role=InvocationRole("implementer"),
                session_namespace="main",
                tool_access=contracts_runtime.ToolAccess.no_tools(),
            )
        )
    )

    assert outcome.output == "OpenCode answer"
    assert list(worktree.rglob("*.log")) == []
    assert list(runtime_state_dir.rglob("*.log")) == []
    invocation_records = cast(
        tuple[prompt_runtime.InvocationRecord, ...], outcome.invocation_records
    )
    assert len(invocation_records) == 1
    invocation_record = invocation_records[0]
    assert invocation_record.run_kind is RunKind.FRESH
    assert invocation_record.service_name == "opencode"
    assert isinstance(outcome.result, prompt_runtime.SessionRunResult)
    assert (
        invocation_record.provider_session_id
        == outcome.result.runtime_metadata.provider_session_id
    )
    normalized_provider_output = (invocation_record.provider_output or b"").replace(
        b" ", b""
    )
    assert b'"type":"text","sessionID":"provider-session-123"' in (
        normalized_provider_output
    )
