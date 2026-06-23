from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import cast

from .contracts import ExecutionProvider, ResumabilityProvider
from .identity import validate_session_namespace
from ._provider_session_adapter import (
    ProviderSessionAdapter,
    ProviderSessionPlanningRequest,
)
from .session import (
    ProviderSessionStateRequest,
    RunKind,
    normalize_state_dir_relpath,
)


@dataclasses.dataclass(frozen=True)
class ProviderSessionDecision:
    run_kind: RunKind
    provider_session_id: str | None
    state_dir_relpath: str | None
    state_dir_path: Path | None
    service_state_dir: Path | None = None
    exact_transcript_match: bool = False
    use_service_state_dir_for_container: bool = False

    def container_state_dir(self) -> Path | None:
        if (
            self.use_service_state_dir_for_container
            and self.service_state_dir is not None
        ):
            return self.service_state_dir
        return self.state_dir_path

    def container_state_dir_path(
        self,
        *,
        worktree: Path,
        container_workspace: str,
    ) -> str | None:
        container_state_dir = self.container_state_dir()
        if container_state_dir is not None:
            try:
                container_relpath = container_state_dir.relative_to(worktree)
            except ValueError:
                pass
            else:
                return f"{container_workspace}/{container_relpath.as_posix()}/"
        if self.state_dir_relpath is None:
            return None
        return f"{container_workspace}/{self.state_dir_relpath}"


@dataclasses.dataclass(frozen=True)
class ProviderSessionPlanRequest:
    worktree: Path
    namespace: str
    resumability_service: ResumabilityProvider
    provider_session_adapter: ProviderSessionAdapter


@dataclasses.dataclass
class _ProviderRunStatePlan:
    service_name: str
    run_kind: RunKind
    provider_state_dir: Path | None
    provider_state_dir_relpath: str | None
    provider_session_id: str | None
    provider_session_adapter: ProviderSessionAdapter = dataclasses.field(
        repr=False,
        compare=False,
    )
    service_state_dir: Path | None = None
    exact_transcript_match: bool = False
    use_service_state_dir_for_container: bool = False

    def provider_session_decision(self) -> ProviderSessionDecision:
        return ProviderSessionDecision(
            run_kind=self.run_kind,
            provider_session_id=self.provider_session_id,
            state_dir_relpath=self.provider_state_dir_relpath,
            state_dir_path=self.provider_state_dir,
            service_state_dir=self.service_state_dir,
            exact_transcript_match=self.exact_transcript_match,
            use_service_state_dir_for_container=(
                self.use_service_state_dir_for_container
            ),
        )

    def provider_state_dir_container_path(
        self,
        *,
        worktree: Path,
        container_workspace: str,
    ) -> str | None:
        container_state_dir = self.provider_state_dir
        if (
            self.use_service_state_dir_for_container
            and self.service_state_dir is not None
        ):
            container_state_dir = self.service_state_dir
        if container_state_dir is not None:
            try:
                container_relpath = container_state_dir.relative_to(worktree)
            except ValueError:
                pass
            else:
                return f"{container_workspace}/{container_relpath.as_posix()}/"
        if self.provider_state_dir_relpath is None:
            return None
        return f"{container_workspace}/{self.provider_state_dir_relpath}"

    def prepare_provider_state_dir(self) -> None:
        self.provider_session_adapter.prepare_local_provider_run_state(
            self.provider_state_dir,
        )


@dataclasses.dataclass(frozen=True)
class ResumableSessionPlanRequest:
    worktree: Path
    namespace: str
    service: ExecutionProvider
    provider_session_adapter: ProviderSessionAdapter
    resumability_service: ResumabilityProvider | None = None

    def __post_init__(self) -> None:
        validate_session_namespace(self.namespace)


@dataclasses.dataclass(frozen=True)
class ResumableSessionPlan:
    worktree: Path
    namespace: str
    service: ExecutionProvider
    run_kind: RunKind
    provider_state_dir: Path | None
    provider_session_id: str | None
    exact_transcript_match: bool = False


def plan_resumable_session(
    request: ResumableSessionPlanRequest,
) -> ResumableSessionPlan:
    provider_run_state_plan = _plan_provider_run_state(
        ProviderSessionPlanRequest(
            worktree=request.worktree,
            namespace=request.namespace,
            resumability_service=_resumable_resumability_service(request),
            provider_session_adapter=request.provider_session_adapter,
        )
    )
    session_plan = ResumableSessionPlan(
        worktree=request.worktree,
        namespace=request.namespace,
        service=request.service,
        run_kind=provider_run_state_plan.run_kind,
        provider_state_dir=_public_provider_state_dir(provider_run_state_plan),
        provider_session_id=provider_run_state_plan.provider_session_id,
        exact_transcript_match=provider_run_state_plan.exact_transcript_match,
    )
    object.__setattr__(
        session_plan,
        "_provider_state_dir_relpath",
        provider_run_state_plan.provider_state_dir_relpath,
    )
    return session_plan


def _public_provider_state_dir(
    provider_run_state_plan: _ProviderRunStatePlan,
) -> Path | None:
    if (
        provider_run_state_plan.use_service_state_dir_for_container
        and provider_run_state_plan.service_state_dir is not None
    ):
        return provider_run_state_plan.service_state_dir
    return provider_run_state_plan.provider_state_dir


def plan_provider_session(
    request: ProviderSessionPlanRequest,
) -> ProviderSessionDecision:
    return _plan_provider_run_state(request).provider_session_decision()


def _plan_provider_run_state(
    request: ProviderSessionPlanRequest,
) -> _ProviderRunStatePlan:
    provider_session_adapter = request.provider_session_adapter
    provider_session_planning_facts = (
        provider_session_adapter.provider_session_planning_facts(
            ProviderSessionPlanningRequest(
                worktree=request.worktree,
                namespace=request.namespace,
            )
        )
    )
    state_dir_relpath = normalize_state_dir_relpath(
        "implementer",
        request.namespace,
        provider_session_adapter.service_name,
        provider_session_planning_facts.state_dir_relpath,
    )
    host_state_dir = provider_session_planning_facts.provider_state_dir
    has_resumable_provider_state = (
        provider_session_planning_facts.has_resumable_provider_state
    )
    if state_dir_relpath != provider_session_planning_facts.state_dir_relpath:
        host_state_dir = _host_state_dir(request.worktree, state_dir_relpath)
        has_resumable_provider_state = (
            host_state_dir is not None
            and request.resumability_service.is_resumable(host_state_dir)
        )
    provider_session_state = provider_session_adapter.provider_session_state(
        ProviderSessionStateRequest(
            provider_state_dir=host_state_dir,
            has_resumable_provider_state=has_resumable_provider_state,
            state_dir_relpath=state_dir_relpath,
            require_exact_transcript_match=True,
        )
    )
    selected_provider_state_dir = (
        provider_session_state.state_dir_path or host_state_dir
    )
    return _ProviderRunStatePlan(
        provider_session_adapter=provider_session_adapter,
        service_name=provider_session_adapter.service_name,
        run_kind=provider_session_state.run_kind,
        provider_session_id=provider_session_state.provider_session_id,
        provider_state_dir=selected_provider_state_dir,
        provider_state_dir_relpath=(
            provider_session_state.state_dir_relpath or state_dir_relpath
        ),
        service_state_dir=host_state_dir,
        exact_transcript_match=provider_session_state.exact_transcript_match,
        use_service_state_dir_for_container=(
            provider_session_state.use_service_state_dir_for_container
        ),
    )


def _host_state_dir(worktree: Path, state_dir_relpath: str | None) -> Path | None:
    if state_dir_relpath is None:
        return None
    return worktree / state_dir_relpath.rstrip("/")


def _resumable_resumability_service(
    request: ResumableSessionPlanRequest,
) -> ResumabilityProvider:
    resumability_service = request.resumability_service
    if resumability_service is not None:
        return resumability_service
    return cast(ResumabilityProvider, request.service)


__all__ = [
    "ProviderSessionDecision",
    "ProviderSessionPlanRequest",
    "ResumableSessionPlan",
    "ResumableSessionPlanRequest",
    "plan_provider_session",
    "plan_resumable_session",
]
