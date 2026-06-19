from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, cast

from . import _time as _time_module
from ._portable_continuation_payload import (
    create_portable_continuation_payload,
    read_portable_continuation_payload,
)
from ._runtime_lifecycle import (
    _DEFAULT_EPHEMERAL_ROLE,
    _DEFAULT_EPHEMERAL_SESSION_NAMESPACE,
    Continuation,
    EphemeralResultMetadata,
    EphemeralRunRequest,
    EphemeralRunResult,
    EphemeralRuntimeMetadata,
    NewSessionRunRequest,
    ResumedSessionRunRequest,
    RuntimeOutcome,
    SessionRunResult,
    SessionRuntimeMetadata,
)
from .contracts import ToolAccess
from .errors import (
    AgentCancelledError,
    AgentTimeoutError,
    NoServiceAvailableError,
    RetryableProviderFailureError,
    RuntimeConfigurationError,
    UsageLimitError,
)
from .execution_contracts import (
    CancellationToken,
    PromptRuntimeExecutionAdapter,
    RunSessionPlan,
    TextOutputAdapter,
    WorkInvocationPresentation,
    WorkInvocationRequest,
    WorktreeMount,
)
from .invocation_progress import InvocationProgress
from .roles import InvocationRole
from .service_registry import ServiceRegistry
from .session import RunKind
from .session_planning import (
    ResumableSessionPlan,
    ResumableSessionPlanRequest,
    plan_resumable_session,
)
from .types import StageSelection
from .usage_limit_scope import UsageLimitScope
from .work import invoke_work

_DEFAULT_RUNTIME_NAME = "Runtime Agent"


def _selected_service_path(
    override: StageSelection,
    *,
    selected_service: str,
) -> tuple[str, ...]:
    path: list[str] = []
    current: StageSelection | None = override
    while current is not None:
        if current.service:
            path.append(current.service)
            if current.service == selected_service:
                return tuple(path)
        current = current.fallback
    return (selected_service,)


@dataclasses.dataclass(frozen=True)
class _RuntimeIntent:
    run_session: RunSessionPlan
    model: str
    effort: str
    output_adapter: Any = dataclasses.field(repr=False)
    dependencies: Any = dataclasses.field(repr=False)
    presentation: WorkInvocationPresentation = dataclasses.field(
        default_factory=WorkInvocationPresentation
    )
    token: CancellationToken | None = None
    allow_non_typed_resume_retry: bool = False


@dataclasses.dataclass
class _EphemeralPreparedProviderRunSession:
    run_kind: RunKind = RunKind.FRESH
    provider_session_id: str | None = None

    def record_provider_session_id(self, provider_session_id: str) -> None:
        self.provider_session_id = provider_session_id

    def record_successful_run(self) -> None:
        return None


class _EphemeralPreparedRunSessionState:
    provider_state_dir_container_path: str | None = None

    def __init__(self) -> None:
        self._provider_run_session = _EphemeralPreparedProviderRunSession()

    def prepare_for_run(self) -> None:
        return None

    def initial_provider_run_session(self) -> _EphemeralPreparedProviderRunSession:
        return self._provider_run_session

    def resumable_provider_run_session(self) -> _EphemeralPreparedProviderRunSession:
        return self._provider_run_session

    def protocol_reprompt_provider_run_session(self) -> None:
        return None


@dataclasses.dataclass
class _TrackedPreparedSessionState:
    _prepared_session: Any
    latest_provider_run_session: Any | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._prepared_session, name)


def _coerce_service_registry(
    service_registry: ServiceRegistry | dict[str, Any] | None,
) -> ServiceRegistry:
    if isinstance(service_registry, ServiceRegistry):
        return service_registry
    return ServiceRegistry(service_registry or {})


def _require_execution_adapter_method(
    adapter: PromptRuntimeExecutionAdapter,
    method_name: str,
) -> Any:
    method = getattr(adapter, method_name, None)
    if callable(method):
        return method
    raise RuntimeConfigurationError(
        f"Prompt runtime requires an execution adapter with callable `{method_name}()`."
    )


def _build_run_session(
    *,
    mount_path: Any,
    role: InvocationRole,
    session_namespace: str,
    service: Any,
    container_workspace: str,
    usage_limit_scope: UsageLimitScope | None = None,
    run_kind: RunKind = RunKind.FRESH,
    provider_session_id: str | None = None,
    provider_resume_state: Any = None,
    provider_state_dir_container_path: str | None = None,
    exact_transcript_match: bool = False,
) -> RunSessionPlan:
    return RunSessionPlan(
        mount_path=mount_path,
        role=role,
        session_namespace=session_namespace,
        service=service,
        container_workspace=container_workspace,
        usage_limit_scope=usage_limit_scope,
        run_kind=run_kind,
        provider_session_id=provider_session_id,
        provider_resume_state=provider_resume_state,
        provider_state_dir_container_path=provider_state_dir_container_path,
        exact_transcript_match=exact_transcript_match,
    )


def _latest_provider_run_session(prepared_session: Any) -> Any:
    provider_run_session = getattr(
        prepared_session,
        "latest_provider_run_session",
        None,
    )
    if provider_run_session is not None:
        return provider_run_session
    return prepared_session.initial_provider_run_session()


async def _invoke_runtime_intent(intent: _RuntimeIntent) -> Any:
    return await invoke_work(
        WorkInvocationRequest(
            run_session=intent.run_session,
            model=intent.model,
            effort=intent.effort,
            output_adapter=intent.output_adapter,
            dependencies=intent.dependencies,
            presentation=intent.presentation,
            token=intent.token,
            allow_non_typed_resume_retry=intent.allow_non_typed_resume_retry,
        )
    )


async def _run_runtime_outcome(
    run_result: Any,
) -> RuntimeOutcome:
    try:
        result = await run_result
    except AgentCancelledError as exc:
        return RuntimeOutcome.cancelled(
            output="",
            invocation_progress=exc.invocation_progress,
            continuation=exc.continuation,
            usage=exc.usage,
        )
    except AgentTimeoutError as exc:
        return RuntimeOutcome.timed_out(
            output="",
            invocation_progress=exc.invocation_progress,
            continuation=exc.continuation,
            usage=exc.usage,
        )
    except NoServiceAvailableError as exc:
        return RuntimeOutcome.no_service_available(
            output="",
            reset_time=exc.reset_time,
            usage_limit_scope=exc.usage_limit_scope,
            invocation_progress=exc.invocation_progress,
            continuation=exc.continuation,
            usage=exc.usage,
        )
    except RetryableProviderFailureError as exc:
        return RuntimeOutcome.retryable_provider_failure(
            output="",
            service_name=exc.service_name,
            invocation_progress=exc.invocation_progress,
            continuation=exc.continuation,
            usage=exc.usage,
        )
    except UsageLimitError as exc:
        return RuntimeOutcome.usage_limited(
            output="",
            service_name=exc.service_name,
            reset_time=exc.reset_time,
            usage_limit_scope=exc.usage_limit_scope,
            invocation_progress=exc.invocation_progress,
            continuation=exc.continuation,
            usage=exc.usage,
        )
    return RuntimeOutcome.completed(output=result.output, result=result)


async def _run_ephemeral_outcome(
    *,
    runner: PromptRuntimeExecutionAdapter,
    service_registry: ServiceRegistry,
    request: EphemeralRunRequest,
) -> RuntimeOutcome:
    outcome = await _run_runtime_outcome(
        _run_ephemeral(
            runner=runner,
            service_registry=service_registry,
            request=request,
        )
    )
    if outcome.kind in {"usage_limited", "no_service_available"}:
        return dataclasses.replace(outcome, usage_limit_scope=None)
    return outcome


async def _run_new_session_outcome(
    *,
    runner: PromptRuntimeExecutionAdapter,
    service_registry: ServiceRegistry,
    request: NewSessionRunRequest,
) -> RuntimeOutcome:
    return await _run_runtime_outcome(
        _run_new_session(
            runner=runner,
            service_registry=service_registry,
            request=request,
        )
    )


async def _run_resumed_session_outcome(
    *,
    runner: PromptRuntimeExecutionAdapter,
    request: ResumedSessionRunRequest,
) -> RuntimeOutcome:
    return await _run_runtime_outcome(
        _run_resumed_session(
            runner=runner,
            request=request,
        )
    )


async def _run_ephemeral(
    *,
    runner: PromptRuntimeExecutionAdapter,
    service_registry: ServiceRegistry,
    request: EphemeralRunRequest,
) -> EphemeralRunResult:
    if not service_registry.has_configured_candidate(request.stage):
        raise RuntimeConfigurationError(
            "Ephemeral runtime requires at least one configured service candidate."
        )

    role = _DEFAULT_EPHEMERAL_ROLE
    resolve_service = _require_execution_adapter_method(runner, "resolve_service")
    build_work_dependencies = _require_execution_adapter_method(
        runner,
        "build_work_dependencies",
    )

    while True:
        now = _time_module.now_local()
        if request.token is not None and request.token.is_cancelled:
            raise AgentCancelledError(
                invocation_progress=InvocationProgress.NOT_STARTED,
            )
        if not service_registry.has_available_for(request.stage, now):
            next_wake_time = service_registry.next_wake_time_for(
                request.stage,
                now,
            )
            raise NoServiceAvailableError(
                reset_time=next_wake_time,
                usage_limit_scope=None,
            )

        resolved_override = service_registry.resolve(request.stage, now)
        resolved_service = resolve_service(resolved_override.service)
        dependencies = build_work_dependencies(
            name=_DEFAULT_RUNTIME_NAME,
            model=resolved_override.model,
            effort=resolved_override.effort,
            service=resolved_service,
        )
        raw_output = await _invoke_runtime_intent(
            _RuntimeIntent(
                run_session=_build_run_session(
                    mount_path=request.mount_path,
                    role=role,
                    session_namespace=_DEFAULT_EPHEMERAL_SESSION_NAMESPACE,
                    service=resolved_service,
                    container_workspace=dependencies.execution.container_workspace,
                    usage_limit_scope=None,
                ),
                model=resolved_override.model,
                effort=resolved_override.effort,
                output_adapter=TextOutputAdapter(
                    prompt=request.prompt,
                    tool_access=request.tool_access,
                    workspace=request.invocation_dir,
                ),
                dependencies=dataclasses.replace(
                    dependencies,
                    execution=dataclasses.replace(
                        dependencies.execution,
                        prepare_session=lambda _run_session: cast(
                            Any,
                            _EphemeralPreparedRunSessionState(),
                        ),
                    ),
                ),
                presentation=WorkInvocationPresentation(
                    name=_DEFAULT_RUNTIME_NAME,
                ),
                token=request.token,
            )
        )
        selected_service_path = _selected_service_path(
            request.stage,
            selected_service=resolved_service.name,
        )
        return EphemeralRunResult(
            output=raw_output if isinstance(raw_output, str) else str(raw_output),
            selected_service=resolved_service.name,
            selected_model=resolved_override.model,
            selected_effort=resolved_override.effort,
            tool_access=request.tool_access,
            used_fallback=len(selected_service_path) > 1,
            metadata=EphemeralResultMetadata(
                selected_service_path=selected_service_path,
                runtime=EphemeralRuntimeMetadata(
                    run_kind=RunKind.FRESH,
                ),
            ),
        )


async def _run_new_session(
    *,
    runner: PromptRuntimeExecutionAdapter,
    service_registry: ServiceRegistry,
    request: NewSessionRunRequest,
) -> SessionRunResult:
    if not service_registry.has_configured_candidate(request.stage):
        raise RuntimeConfigurationError(
            "New-session runtime requires at least one configured service candidate."
        )
    while True:
        now = _time_module.now_local()
        if request.token is not None and request.token.is_cancelled:
            raise AgentCancelledError(
                invocation_progress=InvocationProgress.NOT_STARTED,
            )
        if not service_registry.has_available_for(request.stage, now):
            raise NoServiceAvailableError(
                reset_time=service_registry.next_wake_time_for(request.stage, now),
                usage_limit_scope=request.usage_limit_scope
                or UsageLimitScope(request.role.value),
            )

        resolved_override = service_registry.resolve(request.stage, now)
        resolve_service = _require_execution_adapter_method(runner, "resolve_service")
        resolved_service = resolve_service(resolved_override.service)
        session_plan = plan_resumable_session(
            ResumableSessionPlanRequest(
                worktree=request.invocation_dir,
                role=request.role,
                namespace=request.session_namespace,
                service=resolved_service,
                session_store=request.session_store,
                provider_session_adapter=request.provider_session_adapter,
                usage_limit_scope=request.usage_limit_scope,
            )
        )
        try:
            return await _run_resumed_session(
                runner=runner,
                request=ResumedSessionRunRequest(
                    prompt=request.prompt,
                    invocation_dir=WorktreeMount(request.invocation_dir),
                    model=resolved_override.model,
                    effort=resolved_override.effort,
                    session_plan=session_plan,
                    tool_access=request.tool_access,
                    name=request.name,
                    status_display=request.status_display,
                    work_body=request.work_body,
                    token=request.token,
                ),
            )
        except UsageLimitError as exc:
            if exc.invocation_progress is not InvocationProgress.NOT_STARTED:
                raise
            service_registry.mark_exhausted(
                resolved_override.service,
                reset_time=exc.reset_time,
            )
            exhausted_now = _time_module.now_local()
            if not service_registry.has_available_for(request.stage, exhausted_now):
                raise NoServiceAvailableError(
                    reset_time=service_registry.next_wake_time_for(
                        request.stage,
                        exhausted_now,
                    ),
                    usage_limit_scope=exc.usage_limit_scope,
                    invocation_progress=exc.invocation_progress,
                ) from exc


async def _run_resumed_session(
    *,
    runner: PromptRuntimeExecutionAdapter,
    request: ResumedSessionRunRequest,
) -> SessionRunResult:
    resolve_service = _require_execution_adapter_method(runner, "resolve_service")
    build_work_dependencies = _require_execution_adapter_method(
        runner,
        "build_work_dependencies",
    )
    if request.continuation is not None:
        continuation = request.continuation
        service_name = continuation.selected_service
        provider_resume_state = _continuation_resume_state(continuation)
        try:
            service = resolve_service(service_name)
        except NoServiceAvailableError as exc:
            exc.continuation = continuation
            raise
        run_kind = RunKind.RESUME
        provider_session_id = cast(
            str | None,
            provider_resume_state.get("provider_session_id"),
        )
        provider_state_dir_relpath = cast(
            str | None,
            provider_resume_state.get("provider_state_dir_relpath"),
        )
        exact_transcript_match = bool(
            provider_resume_state.get("exact_transcript_match", False)
        )
        provider_state_dir = None
    else:
        plan = cast(ResumableSessionPlan, request.session_plan)
        service = plan.service
        service_name = service.name
        run_kind = plan.run_kind
        provider_session_id = plan.provider_session_id
        provider_state_dir_relpath = getattr(
            plan,
            "_provider_state_dir_relpath",
            None,
        )
        exact_transcript_match = plan.exact_transcript_match
        provider_state_dir = plan.provider_state_dir
    dependencies = build_work_dependencies(
        name=request.name,
        model=request.model,
        effort=request.effort,
        service=service,
    )
    prepared_session: Any = None

    def _prepare_session(run_session: RunSessionPlan) -> Any:
        nonlocal prepared_session
        if prepared_session is None:
            prepared_session = _TrackedPreparedSessionState(
                dependencies.execution.prepare_session(run_session)
            )
        return prepared_session

    resumable_dependencies = dataclasses.replace(
        dependencies,
        execution=dataclasses.replace(
            dependencies.execution,
            prepare_session=_prepare_session,
        ),
    )
    run_session = _build_run_session(
        mount_path=request.invocation_dir.host_path,
        role=request.role,
        session_namespace=request.session_namespace,
        service=service,
        container_workspace=dependencies.execution.container_workspace,
        usage_limit_scope=request.usage_limit_scope,
        run_kind=run_kind,
        provider_session_id=provider_session_id,
        provider_resume_state=(
            provider_resume_state if request.continuation is not None else None
        ),
        provider_state_dir_container_path=_provider_state_dir_container_path(
            worktree=request.invocation_dir.host_path,
            provider_state_dir=provider_state_dir,
            provider_state_dir_relpath=provider_state_dir_relpath,
            container_workspace=dependencies.execution.container_workspace,
        ),
        exact_transcript_match=exact_transcript_match,
    )
    try:
        output = await _invoke_runtime_intent(
            _RuntimeIntent(
                run_session=run_session,
                model=request.model,
                effort=request.effort,
                output_adapter=TextOutputAdapter(
                    prompt=request.prompt,
                    tool_access=request.tool_access,
                    workspace=request.invocation_dir.host_path,
                ),
                dependencies=resumable_dependencies,
                presentation=WorkInvocationPresentation(
                    name=request.name,
                    status_display=request.status_display,
                    work_body=request.work_body,
                ),
                token=request.token,
            )
        )
    except (
        AgentCancelledError,
        AgentTimeoutError,
        RetryableProviderFailureError,
        UsageLimitError,
    ) as exc:
        exc.continuation = (
            request.continuation
            if request.continuation is not None
            else _interruption_continuation(
                request=request,
                service_name=service_name,
                run_kind=run_kind,
                provider_state_dir_relpath=provider_state_dir_relpath,
                exact_transcript_match=exact_transcript_match,
                prepared_session=prepared_session,
                prepare_session=resumable_dependencies.execution.prepare_session,
                run_session=run_session,
                invocation_progress=exc.invocation_progress,
            )
        )
        raise
    if prepared_session is None:
        prepared_session = resumable_dependencies.execution.prepare_session(run_session)
    provider_run_session = _latest_provider_run_session(prepared_session)
    return SessionRunResult(
        output=output,
        runtime_metadata=SessionRuntimeMetadata(
            service_name=service_name,
            provider_session_id=provider_run_session.provider_session_id,
            run_kind=run_kind,
            session_namespace=request.session_namespace,
            exact_transcript_match=exact_transcript_match,
        ),
        continuation=_build_continuation(
            service_name=service_name,
            model=request.model,
            effort=request.effort,
            tool_access=request.tool_access,
            run_kind=run_kind,
            provider_session_id=provider_run_session.provider_session_id,
            provider_state_dir_relpath=provider_state_dir_relpath,
            exact_transcript_match=exact_transcript_match,
            prepared_session=prepared_session,
            provider_run_session=provider_run_session,
        ),
    )


def _provider_state_dir_container_path(
    *,
    worktree: Path,
    provider_state_dir: Path | None,
    provider_state_dir_relpath: str | None,
    container_workspace: str,
) -> str | None:
    if provider_state_dir is None:
        return (
            None
            if provider_state_dir_relpath is None
            else f"{container_workspace}/{provider_state_dir_relpath}"
        )
    try:
        container_relpath = provider_state_dir.relative_to(worktree)
    except ValueError:
        return (
            None
            if provider_state_dir_relpath is None
            else f"{container_workspace}/{provider_state_dir_relpath}"
        )
    return f"{container_workspace}/{container_relpath.as_posix()}/"


def _continuation_resume_state(continuation: Continuation) -> dict[str, Any]:
    try:
        return read_portable_continuation_payload(continuation).provider_resume_state
    except TypeError as exc:
        raise RuntimeConfigurationError(str(exc)) from exc


def _build_continuation(
    *,
    service_name: str,
    model: str,
    effort: str,
    tool_access: ToolAccess,
    run_kind: RunKind,
    provider_session_id: str | None,
    provider_state_dir_relpath: str | None,
    exact_transcript_match: bool,
    prepared_session: Any | None = None,
    provider_run_session: Any | None = None,
) -> Continuation:
    if provider_run_session is not None and hasattr(
        provider_run_session,
        "latest_provider_resume_state",
    ):
        provider_resume_state = getattr(
            provider_run_session,
            "latest_provider_resume_state",
        )
    elif provider_run_session is not None and hasattr(
        provider_run_session,
        "provider_resume_state",
    ):
        provider_resume_state = getattr(
            provider_run_session,
            "provider_resume_state",
        )
    elif prepared_session is not None and hasattr(
        prepared_session,
        "provider_resume_state",
    ):
        provider_resume_state = getattr(prepared_session, "provider_resume_state")
    else:
        provider_resume_state = {
            "run_kind": run_kind.value,
            "provider_session_id": provider_session_id,
            "provider_state_dir_relpath": provider_state_dir_relpath,
            "exact_transcript_match": exact_transcript_match,
        }
    return create_portable_continuation_payload(
        service_name=service_name,
        model=model,
        effort=effort,
        tool_access=tool_access,
        provider_resume_state=provider_resume_state,
    ).to_continuation()


def _interruption_continuation(
    *,
    request: ResumedSessionRunRequest,
    service_name: str,
    run_kind: RunKind,
    provider_state_dir_relpath: str | None,
    exact_transcript_match: bool,
    prepared_session: Any,
    prepare_session: Any,
    run_session: RunSessionPlan,
    invocation_progress: InvocationProgress,
) -> Continuation | None:
    if invocation_progress is not InvocationProgress.STARTED:
        return None
    if prepared_session is None:
        prepared_session = prepare_session(run_session)
    provider_run_session = _latest_provider_run_session(prepared_session)
    return _build_continuation(
        service_name=service_name,
        model=request.model,
        effort=request.effort,
        tool_access=request.tool_access,
        run_kind=run_kind,
        provider_session_id=provider_run_session.provider_session_id,
        provider_state_dir_relpath=provider_state_dir_relpath,
        exact_transcript_match=exact_transcript_match,
        prepared_session=prepared_session,
        provider_run_session=provider_run_session,
    )
