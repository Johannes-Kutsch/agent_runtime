from __future__ import annotations

import asyncio
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import pytest

import agent_runtime as runtime
import agent_runtime.runtime as prompt_runtime
from agent_runtime._import_isolation import assert_runtime_import_isolation
from agent_runtime.contracts import (
    AssistantTurn,
    CredentialFailure,
    HardError,
    PromptTokens,
    Result,
    TransientError,
    UnsupportedTokens,
    UsageLimit,
)
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
    PreparedRunSessionState,
    WorkExecutionAdapter,
    WorkInvocationDependencies,
    WorktreeMount,
)
from agent_runtime.provider_errors import ProviderErrorObservation
from agent_runtime.roles import AgentRole
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
from agent_runtime.session_planning import ResidentSessionPlan
from agent_runtime.work import reduce_text_output_events


class _Service:
    def __init__(self, name: str, *, available: bool, wake_time: datetime) -> None:
        self.name = name
        self._available = available
        self._wake_time = wake_time
        self.available_checks: list[datetime | None] = []

    def is_available(self, now: datetime | None = None) -> bool:
        self.available_checks.append(now)
        return self._available

    def next_wake_time(self) -> datetime:
        return self._wake_time

    def mark_exhausted(self, reset_time: datetime | None) -> None:
        self._available = False
        if reset_time is not None:
            self._wake_time = reset_time

    def state_dir_relpath(self, role: AgentRole, namespace: str = "") -> str | None:
        del role, namespace
        return None

    def is_resumable(self, state_dir: Path) -> bool:
        del state_dir
        return False

    def valid_models(self) -> frozenset[str]:
        return frozenset()

    def valid_efforts(self) -> frozenset[str]:
        return frozenset()


class _ExecutionService:
    def __init__(self, name: str) -> None:
        self.name = name
        self.exhausted_reset_times: list[datetime | None] = []

    def mark_exhausted(self, reset_time: datetime | None) -> None:
        self.exhausted_reset_times.append(reset_time)

    def build_command(
        self,
        role: AgentRole,
        model: str,
        effort: str,
        run_kind: RunKind,
        session_uuid: str | None,
        *,
        tool_policy: Any | None = None,
    ) -> str:
        del role, model, effort, run_kind, session_uuid, tool_policy
        return ""

    def build_env(
        self,
        state_dir_container_path: str | None = None,
        token: str | None = None,
    ) -> dict[str, str]:
        del state_dir_container_path, token
        return {}

    def run(
        self,
        lines: Iterable[str],
        on_provider_session_id: Any = None,
    ) -> Iterator[runtime.ParsedTurn]:
        del lines, on_provider_session_id
        return iter(())

    def state_dir_relpath(self, role: AgentRole, namespace: str = "") -> str | None:
        del role, namespace
        return None

    def is_resumable(self, state_dir: Path) -> bool:
        del state_dir
        return False

    def valid_models(self) -> frozenset[str]:
        return frozenset()

    def valid_efforts(self) -> frozenset[str]:
        return frozenset()


@dataclass
class _RoleSession:
    service_sessions: dict[str, str | None]
    service_metadata: dict[str, dict[str, str] | None]
    exact_transcript_service: str | None = None

    def session_uuid(self) -> str:
        return "session-uuid"

    def service_session_id(self, service_name: str) -> str | None:
        return self.service_sessions.get(service_name)

    def save_service_session_id(self, service_name: str, session_id: str) -> None:
        self.service_sessions[service_name] = session_id

    def service_session_metadata(self, service_name: str) -> dict[str, str] | None:
        return self.service_metadata.get(service_name)

    def exact_transcript_service_name(self) -> str | None:
        return self.exact_transcript_service


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
        role: AgentRole,
        prompt: str,
        *,
        run_kind: RunKind = RunKind.FRESH,
        session_uuid: str | None = None,
        on_provider_session_id: Any = None,
    ) -> dict[str, str]:
        assert role is AgentRole.IMPLEMENTER
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
                provider=service_name,
            )

        assert callable(on_provider_session_id)
        on_provider_session_id(f"provider-{service_name}")
        return {"service": service_name, "prompt": prompt}

    async def work_text(
        self,
        prompt: str,
        *,
        role: AgentRole = AgentRole.IMPLEMENTER,
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


class _OneShotExecutionAdapter:
    def __init__(
        self,
        *,
        invocation_order: list[str],
        attempts_by_service: dict[str, int],
    ) -> None:
        self._invocation_order = invocation_order
        self._attempts_by_service = attempts_by_service

    def resolve_service(self, service_name: str = "") -> runtime.ExecutionService:
        return _ExecutionService(service_name)

    def build_work_dependencies(
        self,
        *,
        name: str,
        model: str,
        effort: str,
        service: runtime.ExecutionService,
    ) -> WorkInvocationDependencies:
        del name, model, effort
        execution_service = cast(_ExecutionService, service)
        return WorkInvocationDependencies(
            container_workspace="/workspace",
            timeout_retries=0,
            stage_key_for_role=lambda role: role.value,
            prepare_session=lambda _run_session: cast(
                PreparedRunSessionState, _PreparedRunSession()
            ),
            build_session=lambda mount_path, service, provider_state_dir: _Session(),
            build_runner=lambda session, status_display: cast(
                WorkExecutionAdapter,
                _OneShotWorkRunner(
                    execution_service,
                    invocation_order=self._invocation_order,
                    attempts_by_service=self._attempts_by_service,
                ),
            ),
            get_git_identity=lambda: ("Runtime Test", "runtime@example.com"),
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
        role: AgentRole,
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
        role: AgentRole = AgentRole.IMPLEMENTER,
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
    def resolve_service(self, service_name: str = "") -> runtime.ExecutionService:
        return _ExecutionService(service_name)

    def build_work_dependencies(
        self,
        *,
        name: str,
        model: str,
        effort: str,
        service: runtime.ExecutionService,
    ) -> WorkInvocationDependencies:
        del name, model, effort, service

        def _prepare_session(run_session: Any) -> _ResidentAdapterPreparedRunSession:
            return _ResidentAdapterPreparedRunSession(
                provider_state_dir_container_path="/workspace/runtime-state/",
                run_kind=run_session.run_kind,
                provider_session_id=f"prepared:{run_session.provider_session_id}",
            )

        return WorkInvocationDependencies(
            container_workspace="/workspace",
            timeout_retries=0,
            stage_key_for_role=lambda role: role.value,
            prepare_session=cast(Any, _prepare_session),
            build_session=lambda mount_path, service, provider_state_dir: _Session(
                provider_state_dir
            ),
            build_runner=lambda session, status_display: cast(
                WorkExecutionAdapter, _ResidentSeamRunner(cast(_Session, session))
            ),
            get_git_identity=lambda: ("Runtime Test", "runtime@example.com"),
        )


def test_package_exports_runtime_surface() -> None:
    assert runtime.StageOverride.__module__.startswith("agent_runtime")
    assert runtime.AgentRuntimeError is AgentRuntimeError
    assert not hasattr(runtime, "run_prompt")
    assert not hasattr(runtime, "ServiceRegistry")


def test_import_isolation_helper_reports_forbidden_modules() -> None:
    with pytest.raises(ImportError) as excinfo:
        assert_runtime_import_isolation(
            importer="agent_runtime",
            newly_loaded_modules={"allowed.mod", "forbidden.pkg", "forbidden.pkg.sub"},
            forbidden_prefixes=("forbidden.pkg",),
        )

    assert "forbidden.pkg" in str(excinfo.value)


def test_stage_chain_resolution_prefers_first_available_configured_service() -> None:
    override = runtime.StageOverride(
        service="missing",
        model="ignored",
        effort="medium",
        fallback=runtime.StageOverride(
            service="codex",
            model="gpt-5.4",
            effort="medium",
            fallback=runtime.StageOverride(
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
    assert selection.selected_chain == runtime.StageOverride(
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


def test_service_registry_resolve_and_wake_time() -> None:
    services: dict[str, runtime.ServiceSelectionProvider] = {
        "codex": cast(
            runtime.ServiceSelectionProvider,
            _Service(
                "codex",
                available=False,
                wake_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
            ),
        ),
        "claude": cast(
            runtime.ServiceSelectionProvider,
            _Service(
                "claude",
                available=True,
                wake_time=datetime(2026, 1, 2, tzinfo=timezone.utc),
            ),
        ),
    }
    registry = ServiceRegistry(services)
    override = runtime.StageOverride(
        service="codex",
        model="gpt-5.4",
        effort="medium",
        fallback=runtime.StageOverride(
            service="claude",
            model="sonnet",
            effort="high",
        ),
    )

    resolved = registry.resolve(override, datetime(2026, 1, 1, tzinfo=timezone.utc))

    assert resolved == runtime.StageOverride(
        service="claude",
        model="sonnet",
        effort="high",
    )
    assert registry.has_available(datetime(2026, 1, 1, tzinfo=timezone.utc)) is True
    assert registry.next_wake_time(
        datetime(2026, 1, 1, tzinfo=timezone.utc)
    ) == datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_one_shot_runtime_falls_back_after_usage_limit_with_fresh_service_resolution() -> (
    None
):
    invocation_order: list[str] = []
    attempts_by_service: dict[str, int] = {}
    registry = ServiceRegistry(
        {
            "codex": cast(
                runtime.ServiceSelectionProvider,
                _Service(
                    "codex",
                    available=True,
                    wake_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
                ),
            ),
            "claude": cast(
                runtime.ServiceSelectionProvider,
                _Service(
                    "claude",
                    available=True,
                    wake_time=datetime(2026, 1, 2, tzinfo=timezone.utc),
                ),
            ),
        }
    )

    result = asyncio.run(
        prompt_runtime.OneShotRuntime(
            execution_adapter=_OneShotExecutionAdapter(
                invocation_order=invocation_order,
                attempts_by_service=attempts_by_service,
            ),
            service_registry=registry,
        ).run_one_shot(
            prompt_runtime.OneShotRunRequest(
                prompt="already rendered prompt",
                worktree=WorktreeMount(Path(".")),
                override=runtime.StageOverride(
                    service="codex",
                    model="gpt-5.4",
                    effort="medium",
                    fallback=runtime.StageOverride(
                        service="claude",
                        model="sonnet",
                        effort="high",
                    ),
                ),
            )
        )
    )

    assert result == prompt_runtime.OneShotRunResult(
        selected_service="claude",
        selected_model="sonnet",
        selected_effort="high",
        used_fallback=True,
        selected_service_path=("codex", "claude"),
        raw_output={"service": "claude", "prompt": "already rendered prompt"},
        runtime_metadata=prompt_runtime.OneShotRuntimeMetadata(
            provider_session_id="provider-claude",
            run_kind=RunKind.FRESH,
            session_namespace="",
        ),
    )
    assert invocation_order == ["codex", "claude"]


def test_resident_runtime_preserves_resumable_behavior_through_run_session_seam() -> (
    None
):
    result = asyncio.run(
        prompt_runtime.ResidentRuntime(
            execution_adapter=_ResidentSeamExecutionAdapter()
        ).run_resident_prompt(
            prompt_runtime.ResidentRunRequest(
                prompt="already rendered prompt",
                worktree=WorktreeMount(Path(".")),
                model="gpt-5.4",
                effort="medium",
                session_plan=ResidentSessionPlan(
                    role=AgentRole.IMPLEMENTER,
                    worktree=Path("."),
                    namespace="main",
                    service=cast(runtime.ExecutionProvider, _ExecutionService("codex")),
                    run_kind=RunKind.RESUME,
                    service_state_dir=Path("state"),
                    provider_state_dir_relpath="state/",
                    host_provider_state_dir=Path("state"),
                    provider_session_id="recovered-session",
                    auth_seeding_requirement=cast(Any, object()),
                ),
            )
        )
    )

    assert result == prompt_runtime.ResidentRunResult(
        output="resume:prepared:recovered-session:/workspace/runtime-state/",
        runtime_metadata=prompt_runtime.ResidentRuntimeMetadata(
            service_name="codex",
            provider_session_id="prepared:recovered-session",
            run_kind=RunKind.RESUME,
            session_namespace="main",
            exact_transcript_match=False,
        ),
    )


def test_provider_state_helpers_normalize_legacy_layout_and_build_session_id_path() -> (
    None
):
    legacy = ".runtime-session/implementer/main/codex/"

    assert (
        provider_state_relpath(
            AgentRole.IMPLEMENTER,
            "codex",
            session_root=".runtime-session",
        )
        == ".runtime-session/implementer/codex/"
    )
    assert (
        normalize_state_dir_relpath(
            AgentRole.IMPLEMENTER,
            "main",
            "codex",
            legacy,
        )
        == ".runtime-session/implementer/main/codex/"
    )
    assert provider_state_session_id_path(Path("state"), "codex") == Path(
        "state/thread_id"
    )


def test_select_resumable_provider_session_id_recovers_and_persists_state() -> None:
    state_dir = Path("state")
    role_session = _RoleSession(
        service_sessions={},
        service_metadata={},
    )

    selection = select_resumable_provider_session_id(
        role_session,
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
    assert role_session.service_session_id("codex") == "provider-session"


def test_exact_resumable_service_session_requires_matching_metadata_and_maybe_matcher() -> (
    None
):
    role_session = _RoleSession(
        service_sessions={"codex": "provider-session"},
        service_metadata={"codex": {"provider_session_id": "provider-session"}},
        exact_transcript_service="codex",
    )

    assert (
        is_exact_resumable_service_session(
            role_session,
            "codex",
            provider_session_id="provider-session",
            provider_state_dir=Path("state"),
        )
        is True
    )
    assert (
        is_exact_resumable_service_session(
            role_session,
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


def test_runtime_errors_capture_context() -> None:
    timeout = AgentTimeoutError("timed out")
    usage_limit = UsageLimitError(reset_time=None)
    transient = TransientAgentError("transient", status_code=502)
    hard = HardAgentError("hard", status_code=400, service_name="codex")
    failed = AgentFailedError("implementer", Path("worktree"), service_name="codex")

    assert isinstance(timeout, AgentRuntimeError)
    assert usage_limit.provider is None
    assert transient.status_code == 502
    assert hard.service_name == "codex"
    assert failed.session_dir == "implementer/codex"
