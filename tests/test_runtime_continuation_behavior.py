from __future__ import annotations

import asyncio
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, cast

import pytest

import agent_runtime as runtime
import agent_runtime.contracts as contracts_runtime
import agent_runtime._runtime_compat as compat_runtime
import agent_runtime.runtime as prompt_runtime
import agent_runtime.session_planning as session_planning_runtime
from agent_runtime._portable_continuation_payload import (
    create_portable_continuation_payload,
)
from agent_runtime.contracts import (
    AssistantTurn,
    ExecutionProvider,
    ModelActivity,
    ResumabilityProvider,
    TransientError,
    UsageLimit,
)
from agent_runtime.errors import (
    AgentCancelledError,
    AgentTimeoutError,
    NoServiceAvailableError,
    RuntimeConfigurationError,
    UsageLimitError,
)
from agent_runtime.execution_contracts import (
    CancellationToken,
    WorkExecutionAdapter,
    WorkExecutionDependencies,
    WorkFailureHandling,
    WorkInvocationDependencies,
    WorkPresentationDependencies,
    WorktreeMount,
)
from agent_runtime.provider_output import reduce_text_output_events
from agent_runtime.roles import InvocationRole
from agent_runtime.session import RunKind
from agent_runtime.session_planning import (
    AuthSeedingRequirement,
    ResumableSessionPlan,
    ResumableSessionPlanRequest,
    plan_resumable_session,
)

from tests.runtime_boundary_fakes import (
    ExecutionServiceFake as _ExecutionService,
    ExternalStateResidentPlanningProviderSessionAdapterFake as _ExternalStateResidentPlanningProviderSessionAdapter,
    ResidentPlanningProviderSessionAdapterFake as _ResidentPlanningProviderSessionAdapter,
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
                    WorkExecutionAdapter,
                    _ResidentSeamRunner(cast(_Session, session)),
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


class _ToolPolicyRenderingResidentRunner(_ResidentSeamRunner):
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
        tool_policy: Any = runtime.ToolPolicy.UNRESTRICTED,
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
        tool_policy: Any = runtime.ToolPolicy.UNRESTRICTED,
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


def test_continuation_provider_resume_state_requires_json_compatible_data() -> None:
    continuation = prompt_runtime.Continuation(
        selected_service="codex",
        selected_model="gpt-5.4",
        selected_effort="medium",
        tool_access=contracts_runtime.ToolAccess.no_tools(),
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
            tool_access=contracts_runtime.ToolAccess.no_tools(),
            provider_resume_state={"provider_state_dir": Path("/repo/state")},
        )


def test_resumed_session_runtime_rejects_non_object_portable_continuation_resume_state() -> (
    None
):
    continuation = prompt_runtime.Continuation(
        selected_service="bound-service",
        selected_model="gpt-5.4",
        selected_effort="medium",
        tool_access=contracts_runtime.ToolAccess.workspace_backed(Path("/repo")),
        provider_resume_state=["resume"],
    )

    with pytest.raises(RuntimeConfigurationError) as exc_info:
        asyncio.run(
            compat_runtime.ResumedSessionRuntime(
                execution_adapter=_RuntimePlannedPathResidentExecutionAdapter()
            ).run_resumed_session(
                prompt_runtime.ResumedSessionRunRequest(
                    prompt="already rendered prompt",
                    worktree=WorktreeMount(Path("/repo")),
                    continuation=continuation,
                    role=InvocationRole("implementer"),
                )
            )
        )

    assert str(exc_info.value) == (
        "Continuation provider_resume_state must be a JSON object."
    )


def test_resumed_session_runtime_resumes_from_portable_continuation_data() -> None:
    worktree = Path("/repo")
    continuation = prompt_runtime.Continuation(
        selected_service="codex",
        selected_model="gpt-5.4",
        selected_effort="medium",
        tool_access=contracts_runtime.ToolAccess.workspace_backed(
            worktree,
            tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
        ),
        provider_resume_state={
            "run_kind": "resume",
            "provider_session_id": "recovered-session",
            "provider_state_dir_relpath": "runtime-state/",
            "exact_transcript_match": False,
        },
    )

    result = asyncio.run(
        compat_runtime.ResumedSessionRuntime(
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
        tool_access=contracts_runtime.ToolAccess.workspace_backed(
            worktree,
            tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
        ),
        provider_resume_state={
            "run_kind": "resume",
            "provider_session_id": "prepared:recovered-session",
            "provider_state_dir_relpath": "runtime-state/",
            "exact_transcript_match": False,
        },
    )
    assert (
        result.result.runtime_metadata.tool_policy
        == runtime.ToolPolicy.NO_FILE_MUTATION
    )


def test_resumed_session_runtime_resumes_from_continuation_with_minimal_request_fields() -> (
    None
):
    worktree = Path("/repo")
    continuation = prompt_runtime.Continuation(
        selected_service="codex",
        selected_model="gpt-5.4",
        selected_effort="medium",
        tool_access=contracts_runtime.ToolAccess.no_tools(),
        provider_resume_state={
            "run_kind": "resume",
            "provider_session_id": "recovered-session",
            "provider_state_dir_relpath": "runtime-state/",
            "exact_transcript_match": False,
        },
    )

    result = asyncio.run(
        compat_runtime.ResumedSessionRuntime(
            execution_adapter=_RuntimePlannedPathResidentExecutionAdapter()
        ).run_resumed_session(
            prompt_runtime.ResumedSessionRunRequest(
                prompt="already rendered prompt",
                worktree=WorktreeMount(worktree),
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
                session_namespace="",
                exact_transcript_match=False,
            ),
        ),
    )
    assert isinstance(result.result, prompt_runtime.SessionRunResult)
    assert result.result.continuation == prompt_runtime.Continuation(
        selected_service="codex",
        selected_model="gpt-5.4",
        selected_effort="medium",
        tool_access=contracts_runtime.ToolAccess.no_tools(),
        provider_resume_state={
            "run_kind": "resume",
            "provider_session_id": "prepared:recovered-session",
            "provider_state_dir_relpath": "runtime-state/",
            "exact_transcript_match": False,
        },
    )
    assert result.result.runtime_metadata.tool_policy == runtime.ToolPolicy.NONE


def test_resumed_session_runtime_reuses_continuation_tool_policy_across_roundtrip() -> (
    None
):
    worktree = Path("/repo")
    continuation = prompt_runtime.Continuation(
        selected_service="codex",
        selected_model="gpt-5.4",
        selected_effort="medium",
        tool_access=contracts_runtime.ToolAccess.workspace_backed(
            worktree,
            tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
        ),
        provider_resume_state={
            "run_kind": "resume",
            "provider_session_id": "recovered-session",
            "provider_state_dir_relpath": "runtime-state/",
            "exact_transcript_match": False,
        },
    )

    first = asyncio.run(
        compat_runtime.ResumedSessionRuntime(
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
    assert first.kind == "completed"
    assert isinstance(first.result, prompt_runtime.SessionRunResult)
    assert first.result.runtime_metadata.tool_policy == (
        runtime.ToolPolicy.NO_FILE_MUTATION
    )

    second = asyncio.run(
        compat_runtime.ResumedSessionRuntime(
            execution_adapter=_RuntimePlannedPathResidentExecutionAdapter()
        ).run_resumed_session(
            prompt_runtime.ResumedSessionRunRequest(
                prompt="already rendered prompt",
                worktree=WorktreeMount(worktree),
                role=InvocationRole("implementer"),
                session_namespace="main",
                continuation=first.result.continuation,
            )
        )
    )
    assert second.kind == "completed"
    assert isinstance(second.result, prompt_runtime.SessionRunResult)
    assert second.result.runtime_metadata.tool_policy == (
        runtime.ToolPolicy.NO_FILE_MUTATION
    )


def test_resumed_session_runtime_resumes_from_opaque_continuation_token() -> None:
    worktree = Path("/repo")
    payload = create_portable_continuation_payload(
        service_name="codex",
        model="gpt-5.4",
        effort="medium",
        tool_access=contracts_runtime.ToolAccess.no_tools(),
        provider_resume_state={
            "run_kind": "resume",
            "provider_session_id": "recovered-session",
            "provider_state_dir_relpath": "runtime-state/",
            "exact_transcript_match": False,
        },
    )
    continuation = prompt_runtime.Continuation(serialized=payload.serialized)

    result = asyncio.run(
        compat_runtime.ResumedSessionRuntime(
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

    assert isinstance(result.result, prompt_runtime.SessionRunResult)
    assert result.result.runtime_metadata.selected_model == "gpt-5.4"
    assert result.result.runtime_metadata.selected_effort == "medium"
    assert result.result.runtime_metadata.tool_policy == runtime.ToolPolicy.NONE
    assert result.result.runtime_metadata.service_name == "codex"
    assert not hasattr(result.result.continuation, "selected_service")
    assert not hasattr(result.result.continuation, "selected_model")
    assert not hasattr(result.result.continuation, "selected_effort")


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
        compat_runtime.ResumedSessionRuntime(
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
                    tool_access=contracts_runtime.ToolAccess.workspace_backed(
                        worktree,
                        tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
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
        tool_access=contracts_runtime.ToolAccess.workspace_backed(worktree),
        provider_resume_state={
            "run_kind": "resume",
            "provider_session_id": "recovered-session",
            "provider_state_dir_relpath": "runtime-state/",
            "exact_transcript_match": False,
        },
    )

    result = asyncio.run(
        compat_runtime.ResumedSessionRuntime(
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
        tool_access=contracts_runtime.ToolAccess.workspace_backed(worktree),
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
        compat_runtime.ResumedSessionRuntime(
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
                    tool_access=contracts_runtime.ToolAccess.workspace_backed(worktree),
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
        tool_access=contracts_runtime.ToolAccess.workspace_backed(worktree),
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
        compat_runtime.ResumedSessionRuntime(
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
                    tool_access=contracts_runtime.ToolAccess.workspace_backed(worktree),
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
        tool_access=contracts_runtime.ToolAccess.workspace_backed(worktree),
        provider_resume_state={
            "resume_cursor": {"session": "prepared:next-session", "turn": 9},
            "phase": "completed",
        },
    )


def test_resumed_session_runtime_from_continuation_defaults_and_overrides_model_and_effort() -> (
    None
):
    worktree = Path("/repo")
    continuation = prompt_runtime.Continuation(
        selected_service="codex",
        selected_model="gpt-5.4",
        selected_effort="medium",
        tool_access=contracts_runtime.ToolAccess.workspace_backed(worktree),
        provider_resume_state={
            "run_kind": "resume",
            "provider_session_id": "recovered-session",
            "provider_state_dir_relpath": "runtime-state/",
            "exact_transcript_match": False,
        },
    )

    defaulted_result = asyncio.run(
        compat_runtime.ResumedSessionRuntime(
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
        compat_runtime.ResumedSessionRuntime(
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
        tool_access=contracts_runtime.ToolAccess.workspace_backed(worktree),
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
        tool_access=contracts_runtime.ToolAccess.workspace_backed(worktree),
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
        compat_runtime.ResumedSessionRuntime(
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
        continuation=prompt_runtime.Continuation(
            selected_service="bound-service",
            selected_model="gpt-5.4",
            selected_effort="medium",
            tool_access=contracts_runtime.ToolAccess.workspace_backed(Path("/repo")),
            provider_resume_state={
                "run_kind": "resume",
                "provider_session_id": "prepared:recovered-session",
                "provider_state_dir_relpath": "runtime-state/",
                "exact_transcript_match": False,
            },
        ),
    )


@pytest.mark.parametrize("tool_policy", _TOOL_POLICY_CASES)
def test_resumed_session_runtime_exposes_tool_policy_effects_through_runtime_result(
    tool_policy: runtime.ToolPolicy | contracts_runtime.ToolPolicyProfile,
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
        compat_runtime.ResumedSessionRuntime(
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
        compat_runtime.ResumedSessionRuntime(
            execution_adapter=_StartedUsageLimitResidentExecutionAdapter()
        ).run_resumed_session(
            prompt_runtime.ResumedSessionRunRequest(
                prompt="already rendered prompt",
                worktree=WorktreeMount(Path(".")),
                model="gpt-5.4",
                effort="medium",
                session_plan=session_plan,
                tool_policy=runtime.ToolPolicy.UNRESTRICTED,
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
            tool_access=contracts_runtime.ToolAccess.workspace_backed(Path(".")),
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
        compat_runtime.ResumedSessionRuntime(
            execution_adapter=_ModelActivityUsageLimitResidentExecutionAdapter()
        ).run_resumed_session(
            prompt_runtime.ResumedSessionRunRequest(
                prompt="already rendered prompt",
                worktree=WorktreeMount(Path(".")),
                model="gpt-5.4",
                effort="medium",
                session_plan=session_plan,
                tool_policy=runtime.ToolPolicy.UNRESTRICTED,
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
            tool_access=contracts_runtime.ToolAccess.workspace_backed(Path(".")),
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
        compat_runtime.ResumedSessionRuntime(
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
            tool_policy: Any = runtime.ToolPolicy.UNRESTRICTED,
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
        compat_runtime.ResumedSessionRuntime(
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
        continuation=prompt_runtime.Continuation(
            selected_service="bound-service",
            selected_model="gpt-5.4",
            selected_effort="medium",
            tool_access=contracts_runtime.ToolAccess.workspace_backed(Path("/repo")),
            provider_resume_state={
                "run_kind": "resume",
                "provider_session_id": "prepared:recovered-session",
                "provider_state_dir_relpath": "runtime-state/",
                "exact_transcript_match": False,
            },
        ),
    )


def test_resumed_session_runtime_returns_timed_out_outcome_with_input_continuation_after_model_activity() -> (
    None
):
    worktree = Path("/repo")
    continuation = _bound_service_resumed_continuation(worktree)

    result = asyncio.run(
        compat_runtime.ResumedSessionRuntime(
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
        continuation=prompt_runtime.Continuation(
            selected_service="bound-service",
            selected_model="gpt-5.4",
            selected_effort="medium",
            tool_access=contracts_runtime.ToolAccess.workspace_backed(Path("/repo")),
            provider_resume_state={
                "run_kind": "resume",
                "provider_session_id": "prepared:recovered-session",
                "provider_state_dir_relpath": "runtime-state/",
                "exact_transcript_match": False,
            },
        ),
    )


def test_resumed_session_runtime_returns_retryable_provider_failure_outcome_with_input_continuation_after_model_activity() -> (
    None
):
    worktree = Path("/repo")
    continuation = _bound_service_resumed_continuation(worktree)

    result = asyncio.run(
        compat_runtime.ResumedSessionRuntime(
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
        continuation=prompt_runtime.Continuation(
            selected_service="bound-service",
            selected_model="gpt-5.4",
            selected_effort="medium",
            tool_access=contracts_runtime.ToolAccess.workspace_backed(Path("/repo")),
            provider_resume_state={
                "run_kind": "resume",
                "provider_session_id": "prepared:recovered-session",
                "provider_state_dir_relpath": "runtime-state/",
                "exact_transcript_match": False,
            },
        ),
    )


def _bound_service_resumed_continuation(
    worktree: Path,
) -> prompt_runtime.Continuation:
    return prompt_runtime.Continuation(
        selected_service="bound-service",
        selected_model="gpt-5.4",
        selected_effort="medium",
        tool_access=contracts_runtime.ToolAccess.workspace_backed(worktree),
        provider_resume_state={
            "run_kind": "resume",
            "provider_session_id": "recovered-session",
            "provider_state_dir_relpath": "runtime-state/",
            "exact_transcript_match": False,
        },
    )


def test_runtime_client_writes_resumed_session_invocation_log_to_logs_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
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

    continuation = prompt_runtime.Continuation(
        selected_service="claude",
        selected_model="sonnet",
        selected_effort="medium",
        tool_access=contracts_runtime.ToolAccess.workspace_backed(worktree),
        provider_resume_state={
            "run_kind": "resume",
            "provider_session_id": "recovered-session",
            "provider_state_dir_relpath": "implementer/main/claude",
            "exact_transcript_match": False,
        },
    )

    outcome = asyncio.run(
        prompt_runtime.RuntimeClient().run_resumed_session(
            prompt_runtime.ResumedSessionRunRequest(
                prompt="already rendered prompt",
                worktree=worktree,
                runtime_state_dir=runtime_state_dir,
                continuation=continuation,
                role=InvocationRole("implementer"),
                provider_auth=prompt_runtime.ProviderAuth(
                    claude_code_oauth_token="token"
                ),
                logs_dir=logs_dir,
            )
        )
    )

    assert outcome.output == "hello from claude"
    assert list(worktree.rglob("*.log")) == []
    assert list(runtime_state_dir.rglob("*.log")) == []
    log_paths = sorted(logs_dir.glob("*.log"))
    assert len(log_paths) == 1
    log_text = log_paths[0].read_text()
    header = json.loads(log_text.splitlines()[0])
    assert header == {
        "type": "agent_invocation",
        "invocation_role": "implementer",
        "run_kind": "fresh",
        "provider_session_id": "recovered-session",
        "prompt": "already rendered prompt",
    }
    assert '"type":"assistant"' in log_text
    assert '"type":"result"' in log_text


def test_runtime_client_writes_resumed_opencode_session_log_with_observed_provider_session_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    worktree = tmp_path / "worktree"
    runtime_state_dir = tmp_path / "runtime-state"
    logs_dir = tmp_path / "runtime-logs"
    provider_state_dir_relpath = "implementer/main/opencode/"
    provider_state_dir = runtime_state_dir / provider_state_dir_relpath
    worktree.mkdir()
    provider_state_dir.mkdir(parents=True)
    (provider_state_dir / "resume.jsonl").write_text("[]", encoding="utf-8")
    (provider_state_dir / "session_id").write_text(
        "recovered-session\n",
        encoding="utf-8",
    )

    class _FakeProcess:
        stdout = iter(
            [
                (
                    '{"type":"text","sessionID":"observed-session","part":'
                    '{"type":"text","text":"hello from opencode",'
                    '"time":{"end":"2026-01-01T00:00:00Z"}}}\n'
                ),
                '{"type":"session.status","status":{"type":"idle"}}\n',
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

    continuation = prompt_runtime.Continuation(
        selected_service="opencode",
        selected_model="glm-5",
        selected_effort="medium",
        tool_access=contracts_runtime.ToolAccess.workspace_backed(
            worktree,
            tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
        ),
        provider_resume_state={
            "provider_session_id": "recovered-session",
            "provider_state": {
                "session_id": "recovered-session",
                "resume_jsonl": "[]",
            },
            "exact_transcript_match": True,
        },
    )

    outcome = asyncio.run(
        prompt_runtime.RuntimeClient().run_resumed_session(
            prompt_runtime.ResumedSessionRunRequest(
                prompt="already rendered prompt",
                worktree=worktree,
                runtime_state_dir=runtime_state_dir,
                continuation=continuation,
                role=InvocationRole("implementer"),
                session_namespace="main",
                provider_auth=prompt_runtime.ProviderAuth(opencode_api_key="token"),
                logs_dir=logs_dir,
            )
        )
    )

    assert outcome.output == "hello from opencode"
    assert list(worktree.rglob("*.log")) == []
    assert list(runtime_state_dir.rglob("*.log")) == []
    log_paths = sorted(logs_dir.glob("*.log"))
    assert len(log_paths) == 1
    log_text = log_paths[0].read_text()
    header = json.loads(log_text.splitlines()[0])
    assert header == {
        "type": "agent_invocation",
        "invocation_role": "implementer",
        "run_kind": "resume",
        "provider_session_id": "observed-session",
        "prompt": "already rendered prompt",
    }
    assert '"sessionID":"observed-session"' in log_text
    assert (provider_state_dir / "session_id").read_text(encoding="utf-8").strip() == (
        "observed-session"
    )


def test_resumed_session_runtime_returns_usage_limited_outcome_without_continuation_before_model_activity() -> (
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
                    tool_policy: Any = runtime.ToolPolicy.UNRESTRICTED,
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
        compat_runtime.ResumedSessionRuntime(
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
        continuation=None,
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
        compat_runtime.ResumedSessionRuntime(
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


def test_resumed_session_runtime_returns_cancelled_outcome_without_continuation_before_model_activity() -> (
    None
):
    worktree = Path("/repo")
    continuation = _bound_service_resumed_continuation(worktree)
    cancelled_token = CancellationToken()
    cancelled_token.cancel()

    result = asyncio.run(
        compat_runtime.ResumedSessionRuntime(
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
        continuation=None,
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
def test_resumed_session_runtime_clears_continuation_for_not_started_interruption_outcomes(
    execution_adapter: Any,
    expected_kind: str,
    service_name: str | None,
) -> None:
    worktree = Path("/repo")
    continuation = _bound_service_resumed_continuation(worktree)

    result = asyncio.run(
        compat_runtime.ResumedSessionRuntime(
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
        continuation=None,
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
        compat_runtime.ResumedSessionRuntime(
            execution_adapter=_RuntimePlannedPathResidentExecutionAdapter()
        ).run_resumed_session(
            prompt_runtime.ResumedSessionRunRequest(
                prompt="already rendered prompt",
                worktree=WorktreeMount(Path(".")),
                model="gpt-5.4",
                effort="medium",
                session_plan=session_plan,
                tool_policy=runtime.ToolPolicy.UNRESTRICTED,
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
        compat_runtime.ResumedSessionRuntime(
            execution_adapter=execution_adapter
        ).run_resumed_session(
            prompt_runtime.ResumedSessionRunRequest(
                prompt="already rendered prompt",
                worktree=WorktreeMount(Path(".")),
                model="gpt-5.4",
                effort="medium",
                session_plan=session_plan,
                tool_policy=runtime.ToolPolicy.UNRESTRICTED,
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
        compat_runtime.ResumedSessionRuntime(
            execution_adapter=_RuntimePlannedPathResidentExecutionAdapter()
        ).run_resumed_session(
            prompt_runtime.ResumedSessionRunRequest(
                prompt="already rendered prompt",
                worktree=WorktreeMount(worktree),
                model="gpt-5.4",
                effort="medium",
                session_plan=session_plan,
                tool_policy=runtime.ToolPolicy.UNRESTRICTED,
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
    tool_access = contracts_runtime.ToolAccess.workspace_backed(
        worktree,
        tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
    )

    result = asyncio.run(
        compat_runtime.ResumedSessionRuntime(
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
        compat_runtime.ResumedSessionRuntime(
            execution_adapter=_RuntimePlannedPathResidentExecutionAdapter()
        ).run_resumed_session(
            prompt_runtime.ResumedSessionRunRequest(
                prompt="already rendered prompt",
                worktree=WorktreeMount(Path(".")),
                model="gpt-5.4",
                effort="medium",
                session_plan=session_plan,
                tool_policy=runtime.ToolPolicy.UNRESTRICTED,
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
        compat_runtime.ResumedSessionRuntime(
            execution_adapter=_StartedTimeoutResidentExecutionAdapter()
        ).run_resumed_session(
            prompt_runtime.ResumedSessionRunRequest(
                prompt="already rendered prompt",
                worktree=WorktreeMount(Path(".")),
                model="gpt-5.4",
                effort="medium",
                session_plan=session_plan,
                tool_policy=runtime.ToolPolicy.UNRESTRICTED,
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
            tool_access=contracts_runtime.ToolAccess.workspace_backed(Path(".")),
            provider_resume_state={
                "run_kind": "resume",
                "provider_session_id": "prepared:recovered-session",
                "provider_state_dir_relpath": "state/",
                "exact_transcript_match": False,
            },
        ),
    )


def test_runtime_client_resumed_opencode_session_uses_continuation_state_dir_and_session_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worktree = tmp_path / "worktree"
    runtime_state_dir = tmp_path / "runtime-state"
    worktree.mkdir()
    runtime_state_dir.mkdir()
    continuation = prompt_runtime.Continuation(
        selected_service="opencode",
        selected_model="glm-5",
        selected_effort="medium",
        tool_access=contracts_runtime.ToolAccess.workspace_backed(
            worktree,
            tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
        ),
        provider_resume_state={
            "provider_session_id": "persisted-session-1",
            "provider_state": {
                "session_id": "persisted-session-1",
                "resume_jsonl": "[]",
            },
        },
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
                            "sessionID": "persisted-session-2",
                            "part": {
                                "type": "text",
                                "text": "resumed answer",
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
        prompt_runtime.RuntimeClient().run_resumed_session(
            prompt_runtime.ResumedSessionRunRequest(
                prompt="already rendered prompt",
                worktree=worktree,
                runtime_state_dir=runtime_state_dir,
                role=InvocationRole("implementer"),
                session_namespace="main",
                continuation=continuation,
                provider_auth=prompt_runtime.ProviderAuth(opencode_api_key="test-key"),
            )
        )
    )

    assert "--session persisted-session-1" in observed["command"]
    assert observed["env"]["OPENCODE_HOME"] == str(
        runtime_state_dir / "implementer/main/opencode"
    )
    assert isinstance(result.result, prompt_runtime.SessionRunResult)
    assert result.result.runtime_metadata == prompt_runtime.SessionRuntimeMetadata(
        service_name="opencode",
        provider_session_id="persisted-session-2",
        run_kind=RunKind.RESUME,
        session_namespace="main",
        exact_transcript_match=False,
    )
    assert (
        result.result.runtime_metadata.tool_policy
        == runtime.ToolPolicy.NO_FILE_MUTATION
    )
    assert result.result.continuation == prompt_runtime.Continuation(
        selected_service="opencode",
        selected_model="glm-5",
        selected_effort="medium",
        tool_access=continuation.tool_access,
        provider_resume_state={
            "provider_session_id": "persisted-session-2",
            "provider_state": {
                "session_id": "persisted-session-2",
                "resume_jsonl": "[]",
            },
            "exact_transcript_match": False,
        },
    )


def test_runtime_client_resumed_opencode_session_restores_continuity_without_runtime_state_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worktree = tmp_path / "worktree"
    continuation = prompt_runtime.Continuation(
        selected_service="opencode",
        selected_model="glm-5",
        selected_effort="medium",
        tool_access=contracts_runtime.ToolAccess.workspace_backed(
            worktree,
            tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
        ),
        provider_resume_state={
            "provider_session_id": "persisted-session-1",
            "provider_state": {
                "session_id": "persisted-session-1",
                "resume_jsonl": "[]",
            },
            "exact_transcript_match": False,
        },
    )
    observed: dict[str, Any] = {}
    worktree.mkdir()

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
                            "sessionID": "persisted-session-2",
                            "part": {
                                "type": "text",
                                "text": "resumed answer",
                                "time": {"end": "2026-01-01T00:00:00Z"},
                            },
                        }
                    )
                    + "\n",
                    json.dumps({"type": "session.status", "status": {"type": "idle"}})
                    + "\n",
                ]
            )
            self.stderr = iter(())

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(subprocess, "Popen", _FakePopen)

    result = asyncio.run(
        prompt_runtime.RuntimeClient().run_resumed_session(
            prompt_runtime.ResumedSessionRunRequest(
                prompt="already rendered prompt",
                worktree=worktree,
                runtime_state_dir=None,
                role=InvocationRole("implementer"),
                session_namespace="main",
                continuation=continuation,
                provider_auth=prompt_runtime.ProviderAuth(opencode_api_key="test-key"),
            )
        )
    )

    assert "--session persisted-session-1" in observed["command"]
    assert "OPENCODE_HOME" in observed["env"]
    assert "--model opencode-go/glm-5" in observed["command"]
    assert isinstance(result.result, prompt_runtime.SessionRunResult)
    assert result.result.runtime_metadata == prompt_runtime.SessionRuntimeMetadata(
        service_name="opencode",
        provider_session_id="persisted-session-2",
        run_kind=RunKind.RESUME,
        session_namespace="main",
        exact_transcript_match=False,
    )
    assert result.result.continuation == prompt_runtime.Continuation(
        selected_service="opencode",
        selected_model="glm-5",
        selected_effort="medium",
        tool_access=continuation.tool_access,
        provider_resume_state={
            "provider_session_id": "persisted-session-2",
            "provider_state": {
                "session_id": "persisted-session-2",
                "resume_jsonl": "[]",
            },
            "exact_transcript_match": False,
        },
    )


def test_runtime_client_resumed_opencode_session_keeps_saved_exact_match_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worktree = tmp_path / "worktree"
    runtime_state_dir = tmp_path / "runtime-state"
    provider_state_dir_relpath = "continuations/opencode/"
    provider_state_dir = runtime_state_dir / provider_state_dir_relpath
    worktree.mkdir()
    provider_state_dir.mkdir(parents=True)
    (provider_state_dir / "resume.jsonl").write_text("[]", encoding="utf-8")
    (provider_state_dir / "session_id").write_text(
        "persisted-session-1\n",
        encoding="utf-8",
    )
    continuation = prompt_runtime.Continuation(
        selected_service="opencode",
        selected_model="glm-5",
        selected_effort="medium",
        tool_access=contracts_runtime.ToolAccess.workspace_backed(
            worktree,
            tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
        ),
        provider_resume_state={
            "provider_session_id": "persisted-session-1",
            "provider_state": {
                "session_id": "persisted-session-1",
                "resume_jsonl": "[]",
            },
            "exact_transcript_match": False,
        },
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
            del stdout, stderr, shell, cwd, text
            observed["command"] = command
            observed["env"] = env
            self.stdout = iter(
                [
                    json.dumps(
                        {
                            "type": "text",
                            "sessionID": "persisted-session-1",
                            "part": {
                                "type": "text",
                                "text": "resumed answer",
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
        prompt_runtime.RuntimeClient().run_resumed_session(
            prompt_runtime.ResumedSessionRunRequest(
                prompt="already rendered prompt",
                worktree=worktree,
                runtime_state_dir=runtime_state_dir,
                role=InvocationRole("implementer"),
                session_namespace="main",
                continuation=continuation,
                provider_auth=prompt_runtime.ProviderAuth(opencode_api_key="test-key"),
            )
        )
    )

    assert "--session persisted-session-1" in observed["command"]
    assert observed["env"]["OPENCODE_HOME"] == str(
        runtime_state_dir / "implementer/main/opencode"
    )
    assert isinstance(result.result, prompt_runtime.SessionRunResult)
    assert result.result.runtime_metadata == prompt_runtime.SessionRuntimeMetadata(
        service_name="opencode",
        provider_session_id="persisted-session-1",
        run_kind=RunKind.RESUME,
        session_namespace="main",
        exact_transcript_match=False,
    )
    assert result.result.continuation == prompt_runtime.Continuation(
        selected_service="opencode",
        selected_model="glm-5",
        selected_effort="medium",
        tool_access=continuation.tool_access,
        provider_resume_state={
            "provider_session_id": "persisted-session-1",
            "provider_state": {
                "session_id": "persisted-session-1",
                "resume_jsonl": "[]",
            },
            "exact_transcript_match": False,
        },
    )


def test_runtime_client_resumed_opencode_session_restores_only_portable_provider_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worktree = tmp_path / "worktree"
    runtime_state_dir = tmp_path / "runtime-state"
    provider_state_dir = runtime_state_dir / "implementer/main/opencode"
    worktree.mkdir()
    provider_state_dir.mkdir(parents=True)
    (provider_state_dir / "resume.jsonl").write_text(
        '[{"stale":true}]',
        encoding="utf-8",
    )
    (provider_state_dir / "session_id").write_text(
        "stale-session\n",
        encoding="utf-8",
    )
    continuation = prompt_runtime.Continuation(
        selected_service="opencode",
        selected_model="glm-5",
        selected_effort="medium",
        tool_access=contracts_runtime.ToolAccess.workspace_backed(
            worktree,
            tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
        ),
        provider_resume_state={
            "provider_session_id": "portable-session",
            "provider_state": {},
            "exact_transcript_match": False,
        },
    )
    observed: dict[str, str | None] = {}

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
            del command, shell, cwd, stdout, stderr, text
            opencode_home = Path(env["OPENCODE_HOME"])
            observed["resume_jsonl"] = (
                (opencode_home / "resume.jsonl").read_text(encoding="utf-8")
                if (opencode_home / "resume.jsonl").exists()
                else None
            )
            observed["session_id"] = (
                (opencode_home / "session_id").read_text(encoding="utf-8")
                if (opencode_home / "session_id").exists()
                else None
            )
            self.stdout = iter(
                [
                    json.dumps(
                        {
                            "type": "text",
                            "sessionID": "portable-session",
                            "part": {
                                "type": "text",
                                "text": "resumed answer",
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
        prompt_runtime.RuntimeClient().run_resumed_session(
            prompt_runtime.ResumedSessionRunRequest(
                prompt="already rendered prompt",
                worktree=worktree,
                runtime_state_dir=runtime_state_dir,
                role=InvocationRole("implementer"),
                session_namespace="main",
                continuation=continuation,
                provider_auth=prompt_runtime.ProviderAuth(opencode_api_key="test-key"),
            )
        )
    )

    assert observed["resume_jsonl"] is None
    assert observed["session_id"] is None
    assert isinstance(result.result, prompt_runtime.SessionRunResult)
    assert result.result.continuation == prompt_runtime.Continuation(
        selected_service="opencode",
        selected_model="glm-5",
        selected_effort="medium",
        tool_access=continuation.tool_access,
        provider_resume_state={
            "provider_session_id": "portable-session",
            "provider_state": {"session_id": "portable-session"},
            "exact_transcript_match": False,
        },
    )


def test_runtime_client_resumed_opencode_session_keeps_observed_session_id_on_started_usage_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worktree = tmp_path / "worktree"
    runtime_state_dir = tmp_path / "runtime-state"
    worktree.mkdir()
    runtime_state_dir.mkdir()
    continuation = prompt_runtime.Continuation(
        selected_service="opencode",
        selected_model="glm-5",
        selected_effort="medium",
        tool_access=contracts_runtime.ToolAccess.workspace_backed(
            worktree,
            tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
        ),
        provider_resume_state={
            "provider_session_id": "persisted-session-1",
            "provider_state": {
                "session_id": "persisted-session-1",
                "resume_jsonl": "[]",
            },
            "exact_transcript_match": False,
        },
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
                            "sessionID": "persisted-session-2",
                            "part": {
                                "type": "text",
                                "text": "partial answer",
                                "time": {"end": "2026-01-01T00:00:00Z"},
                            },
                        }
                    )
                    + "\n",
                    json.dumps(
                        {
                            "type": "error",
                            "sessionID": "persisted-session-2",
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
        prompt_runtime.RuntimeClient().run_resumed_session(
            prompt_runtime.ResumedSessionRunRequest(
                prompt="already rendered prompt",
                worktree=worktree,
                runtime_state_dir=runtime_state_dir,
                role=InvocationRole("implementer"),
                session_namespace="main",
                continuation=continuation,
                provider_auth=prompt_runtime.ProviderAuth(opencode_api_key="test-key"),
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
            tool_access=continuation.tool_access,
            provider_resume_state={
                "provider_session_id": "persisted-session-2",
                "provider_state": {
                    "session_id": "persisted-session-1",
                    "resume_jsonl": "[]",
                },
                "exact_transcript_match": False,
            },
        ),
    )
    assert result.continuation is not None
    assert (
        result.continuation.tool_access.tool_policy
        == runtime.ToolPolicy.NO_FILE_MUTATION
    )
