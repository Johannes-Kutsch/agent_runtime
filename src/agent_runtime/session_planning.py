from __future__ import annotations

import dataclasses
import shutil
from enum import Enum
from pathlib import Path
from typing import Protocol, cast

from .contracts import ExecutionProvider, ResumabilityProvider
from .identity import validate_session_namespace
from .errors import AgentCredentialFailureError
from .provider_session_adapter import (
    ProviderSessionAdapter,
    ProviderSessionPlanningRequest,
)
from .provider_errors import ProviderErrorObservation
from .roles import InvocationRole
from .session import (
    ProviderSessionStateRequest,
    RunKind,
    SessionStore,
    normalize_state_dir_relpath,
)
from .usage_limit_scope import UsageLimitScope


class AuthSeedingRequirement(Enum):
    REQUIRED = "required"
    NOT_REQUIRED = "not_required"


@dataclasses.dataclass(frozen=True)
class LocalAuthSeedAction:
    source: Path
    destination: Path
    missing_source_message: str | None = dataclasses.field(default=None, compare=False)
    missing_source_service_name: str | None = dataclasses.field(
        default=None,
        compare=False,
    )
    missing_source_status_code: int | None = dataclasses.field(
        default=None,
        compare=False,
    )
    missing_source_classification: str | None = dataclasses.field(
        default=None,
        compare=False,
    )
    missing_source_observations: tuple[ProviderErrorObservation, ...] = (
        dataclasses.field(default=(), compare=False)
    )

    def require_source(self) -> Path:
        if not self.source.exists():
            if (
                self.missing_source_message is None
                or self.missing_source_service_name is None
            ):
                raise FileNotFoundError(self.source)
            raise AgentCredentialFailureError(
                self.missing_source_message,
                status_code=self.missing_source_status_code,
                service_name=self.missing_source_service_name,
                classification=self.missing_source_classification,
                observations=self.missing_source_observations,
            )
        return self.source

    def apply(self) -> None:
        if self.destination.exists():
            return
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.require_source(), self.destination)


class RecoveredSessionIdPersistence(Enum):
    PERSIST = "persist"
    SKIP = "skip"


@dataclasses.dataclass(frozen=True)
class ProviderSessionDecision:
    run_kind: RunKind
    provider_session_id: str | None
    state_dir_relpath: str | None
    state_dir_path: Path | None
    recovered_session_id_persistence: RecoveredSessionIdPersistence
    service_state_dir: Path | None = None
    exact_transcript_match: bool = False
    auth_seeding_requirement: AuthSeedingRequirement = (
        AuthSeedingRequirement.NOT_REQUIRED
    )
    auth_seed_action: LocalAuthSeedAction | None = None
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


class _SuccessfulRunSessionStore(SessionStore, Protocol):
    def record_successful_provider_session_metadata(
        self,
        service_name: str,
        provider_session_id: str | None,
    ) -> None: ...


@dataclasses.dataclass(frozen=True)
class ProviderSessionPlanRequest:
    worktree: Path
    role: InvocationRole
    namespace: str
    resumability_service: ResumabilityProvider
    session_store: _SuccessfulRunSessionStore
    provider_session_adapter: ProviderSessionAdapter


@dataclasses.dataclass
class _ProviderRunStatePlan:
    session_store: _SuccessfulRunSessionStore = dataclasses.field(
        repr=False,
        compare=False,
    )
    service_name: str
    run_kind: RunKind
    provider_state_dir: Path | None
    provider_state_dir_relpath: str | None
    provider_session_id: str | None
    auth_seeding_requirement: AuthSeedingRequirement
    recovered_session_id_persistence: RecoveredSessionIdPersistence
    provider_session_adapter: ProviderSessionAdapter = dataclasses.field(
        repr=False,
        compare=False,
    )
    service_state_dir: Path | None = None
    exact_transcript_match: bool = False
    auth_seed_action: LocalAuthSeedAction | None = None
    use_service_state_dir_for_container: bool = False

    def provider_session_decision(self) -> ProviderSessionDecision:
        return ProviderSessionDecision(
            run_kind=self.run_kind,
            provider_session_id=self.provider_session_id,
            state_dir_relpath=self.provider_state_dir_relpath,
            state_dir_path=self.provider_state_dir,
            recovered_session_id_persistence=self.recovered_session_id_persistence,
            service_state_dir=self.service_state_dir,
            exact_transcript_match=self.exact_transcript_match,
            auth_seeding_requirement=self.auth_seeding_requirement,
            auth_seed_action=self.auth_seed_action,
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
            self.auth_seed_action,
        )

    def prepared_provider_session_id(self) -> str | None:
        provider_session_id = self.provider_session_id
        if provider_session_id is None:
            return None
        if (
            self.recovered_session_id_persistence
            is RecoveredSessionIdPersistence.PERSIST
        ):
            self.remember_provider_session_id(provider_session_id)
        return provider_session_id

    def remember_provider_session_id(self, provider_session_id: str) -> None:
        self.provider_session_id = provider_session_id
        self.provider_session_adapter.record_provider_session_id(
            session_store=self.session_store,
            provider_session_id=provider_session_id,
            service_state_dir=self.service_state_dir,
        )

    def record_successful_run(self, provider_session_id: str | None) -> None:
        self.session_store.record_successful_provider_session_metadata(
            self.service_name,
            provider_session_id,
        )


@dataclasses.dataclass(frozen=True)
class ResumableSessionPlanRequest:
    worktree: Path
    role: InvocationRole
    namespace: str
    service: ExecutionProvider
    session_store: _SuccessfulRunSessionStore
    provider_session_adapter: ProviderSessionAdapter
    resumability_service: ResumabilityProvider | None = None
    usage_limit_scope: UsageLimitScope | None = None

    def __post_init__(self) -> None:
        validate_session_namespace(self.namespace)


@dataclasses.dataclass(frozen=True)
class ResumableSessionPlan:
    role: InvocationRole
    worktree: Path
    namespace: str
    service: ExecutionProvider
    run_kind: RunKind
    provider_state_dir: Path | None
    provider_session_id: str | None
    auth_seeding_requirement: AuthSeedingRequirement
    auth_seed_action: LocalAuthSeedAction | None = None
    exact_transcript_match: bool = False
    usage_limit_scope: UsageLimitScope | None = None


def plan_resumable_session(
    request: ResumableSessionPlanRequest,
) -> ResumableSessionPlan:
    provider_run_state_plan = _plan_provider_run_state(
        ProviderSessionPlanRequest(
            worktree=request.worktree,
            role=request.role,
            namespace=request.namespace,
            resumability_service=_resumable_resumability_service(request),
            session_store=request.session_store,
            provider_session_adapter=request.provider_session_adapter,
        )
    )
    session_plan = ResumableSessionPlan(
        role=request.role,
        worktree=request.worktree,
        namespace=request.namespace,
        service=request.service,
        usage_limit_scope=request.usage_limit_scope,
        run_kind=provider_run_state_plan.run_kind,
        provider_state_dir=_public_provider_state_dir(provider_run_state_plan),
        provider_session_id=provider_run_state_plan.provider_session_id,
        auth_seeding_requirement=provider_run_state_plan.auth_seeding_requirement,
        auth_seed_action=provider_run_state_plan.auth_seed_action,
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
                role=request.role,
                namespace=request.namespace,
            )
        )
    )
    state_dir_relpath = normalize_state_dir_relpath(
        request.role,
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
            session_store=cast(SessionStore, request.session_store),
            provider_state_dir=host_state_dir,
            has_resumable_provider_state=has_resumable_provider_state,
            state_dir_relpath=state_dir_relpath,
            require_exact_transcript_match=True,
        )
    )
    recovered_session_id_persistence = RecoveredSessionIdPersistence.SKIP
    if provider_session_state.persist_provider_session_id:
        recovered_session_id_persistence = RecoveredSessionIdPersistence.PERSIST
    selected_provider_state_dir = (
        provider_session_state.state_dir_path or host_state_dir
    )
    auth_seeding_requirement = (
        provider_session_state.auth_seeding_requirement
        or AuthSeedingRequirement.NOT_REQUIRED
    )
    return _ProviderRunStatePlan(
        session_store=request.session_store,
        provider_session_adapter=provider_session_adapter,
        service_name=provider_session_adapter.service_name,
        run_kind=provider_session_state.run_kind,
        provider_session_id=provider_session_state.provider_session_id,
        provider_state_dir=selected_provider_state_dir,
        provider_state_dir_relpath=(
            provider_session_state.state_dir_relpath or state_dir_relpath
        ),
        auth_seeding_requirement=auth_seeding_requirement,
        recovered_session_id_persistence=recovered_session_id_persistence,
        service_state_dir=host_state_dir,
        exact_transcript_match=provider_session_state.exact_transcript_match,
        auth_seed_action=provider_session_state.auth_seed_action,
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
    "AuthSeedingRequirement",
    "LocalAuthSeedAction",
    "ProviderSessionDecision",
    "ProviderSessionPlanRequest",
    "RecoveredSessionIdPersistence",
    "ResumableSessionPlan",
    "ResumableSessionPlanRequest",
    "plan_provider_session",
    "plan_resumable_session",
]
