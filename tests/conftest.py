from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import pytest

import agent_runtime as runtime
import agent_runtime.contracts as contracts_runtime
import agent_runtime.runtime as prompt_runtime
from agent_runtime.contracts import ExecutionProvider, ServiceSelectionProvider
from agent_runtime.execution_contracts import PromptRunRequest, WorktreeMount
from agent_runtime.roles import InvocationRole
from agent_runtime.service_registry import ServiceRegistry
from agent_runtime.types import StageSelection as InternalStageSelection

from tests.runtime_boundary_fakes import (
    ExecutionServiceFake,
    ExternalStateResidentPlanningProviderSessionAdapterFake,
    ResidentPlanningProviderSessionAdapterFake,
    SelectionServiceFake,
    SessionStoreFake,
)


@pytest.fixture
def stage_selection_factory() -> Callable[..., InternalStageSelection]:
    def _factory(
        service: str = "codex",
        *,
        model: str = "gpt-5.4",
        effort: str = "medium",
        fallback: InternalStageSelection | None = None,
    ) -> InternalStageSelection:
        return InternalStageSelection(
            service=service,
            model=model,
            effort=effort,
            fallback=fallback,
        )

    return _factory


@pytest.fixture
def provider_selection_factory() -> Callable[..., runtime.ProviderSelection]:
    def _factory(
        service: str = "codex",
        *,
        model: str = "gpt-5.4",
        effort: str = "medium",
    ) -> runtime.ProviderSelection:
        return runtime.ProviderSelection(
            service=service,
            model=model,
            effort=effort,
        )

    return _factory


@pytest.fixture
def execution_service_factory() -> Callable[[str], ExecutionProvider]:
    def _factory(service_name: str = "codex") -> ExecutionProvider:
        return cast(ExecutionProvider, ExecutionServiceFake(service_name))

    return _factory


@pytest.fixture
def session_store_factory() -> Callable[..., SessionStoreFake]:
    def _factory(
        *,
        service_sessions: dict[str, str | None] | None = None,
        service_metadata: dict[str, dict[str, str] | None] | None = None,
        exact_transcript_service: str | None = None,
    ) -> SessionStoreFake:
        return SessionStoreFake(
            service_sessions={} if service_sessions is None else service_sessions,
            service_metadata={} if service_metadata is None else service_metadata,
            exact_transcript_service=exact_transcript_service,
        )

    return _factory


@pytest.fixture
def resident_provider_session_adapter() -> ResidentPlanningProviderSessionAdapterFake:
    return ResidentPlanningProviderSessionAdapterFake()


@pytest.fixture
def external_state_provider_session_adapter() -> (
    ExternalStateResidentPlanningProviderSessionAdapterFake
):
    return ExternalStateResidentPlanningProviderSessionAdapterFake()


@pytest.fixture
def service_registry_factory() -> Callable[..., ServiceRegistry]:
    def _factory(
        *service_names: str,
        unavailable: set[str] | None = None,
        wake_times: dict[str, datetime] | None = None,
    ) -> ServiceRegistry:
        unavailable_names = unavailable or set()
        per_service_wake_times = wake_times or {}
        services = {
            service_name: cast(
                ServiceSelectionProvider,
                SelectionServiceFake(
                    service_name,
                    available=service_name not in unavailable_names,
                    wake_time=per_service_wake_times.get(
                        service_name,
                        datetime(2026, 1, 1, tzinfo=timezone.utc),
                    ),
                ),
            )
            for service_name in service_names
        }
        return ServiceRegistry(services)

    return _factory


@pytest.fixture
def ephemeral_request_factory(
    provider_selection_factory: Callable[..., runtime.ProviderSelection],
) -> Callable[..., prompt_runtime.EphemeralRunRequest]:
    def _factory(
        *,
        prompt: str = "already rendered prompt",
        worktree: Path | WorktreeMount = WorktreeMount(Path(".")),
        stage: runtime.ProviderSelection | None = None,
        tool_access: contracts_runtime.ToolAccess | None = None,
        tool_policy: runtime.ToolPolicy = runtime.ToolPolicy.NONE,
        token: Any = None,
    ) -> prompt_runtime.EphemeralRunRequest:
        kwargs: dict[str, Any] = {"tool_policy": tool_policy}
        if tool_access is not None:
            kwargs["tool_access"] = tool_access
        return prompt_runtime.EphemeralRunRequest(
            prompt=prompt,
            worktree=worktree,
            provider_selection=stage or provider_selection_factory(),
            **kwargs,
            token=token,
        )

    return _factory


@pytest.fixture
def prompt_run_request_factory(
    stage_selection_factory: Callable[..., InternalStageSelection],
) -> Callable[..., PromptRunRequest]:
    def _factory(
        *,
        prompt: str = "already rendered prompt",
        worktree: WorktreeMount = WorktreeMount(Path(".")),
        stage: InternalStageSelection | None = None,
        role: InvocationRole = InvocationRole("implementer"),
        tool_policy: runtime.ToolPolicy = runtime.ToolPolicy.UNRESTRICTED,
        token: Any = None,
    ) -> PromptRunRequest:
        return PromptRunRequest(
            prompt=prompt,
            worktree=worktree,
            stage=stage or stage_selection_factory(),
            role=role,
            tool_policy=tool_policy,
            token=token,
        )

    return _factory
