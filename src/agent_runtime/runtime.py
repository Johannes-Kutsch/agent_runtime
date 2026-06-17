from __future__ import annotations

import dataclasses
from typing import Any

from . import _time as _time_module
from .contracts import ToolPolicy
from .execution_contracts import (
    CancellationToken,
    PromptRunRequest as _PromptRunRequest,
    PromptRuntimeExecutionAdapter as _PromptRuntimeExecutionAdapter,
    RunSessionPlan,
    TextOutputAdapter,
    WorkInvocationPresentation,
    WorkInvocationRequest,
    WorktreeMount,
)
from .errors import RuntimeConfigurationError, UsageLimitError
from .roles import InvocationRole
from .service_registry import ServiceRegistry
from .session import RunKind
from .session_planning import ResidentSessionPlan
from .stage_priority_chain import iter_stage_chain
from .types import StageSelection
from .usage_limit_scope import UsageLimitScope
from .work import invoke_work

__all__ = [
    "OneShotRunRequest",
    "OneShotRunResult",
    "OneShotRuntime",
    "OneShotRuntimeExecutionAdapter",
    "OneShotRuntimeMetadata",
    "ResidentRunRequest",
    "ResidentRunResult",
    "ResidentRuntime",
    "ResidentRuntimeExecutionAdapter",
    "ResidentRuntimeMetadata",
    "ToolPolicy",
    "WorktreeMount",
]

OneShotRunRequest = _PromptRunRequest
OneShotRuntimeExecutionAdapter = _PromptRuntimeExecutionAdapter
ResidentRuntimeExecutionAdapter = _PromptRuntimeExecutionAdapter


@dataclasses.dataclass(frozen=True)
class OneShotRuntimeMetadata:
    provider_session_id: str | None
    run_kind: RunKind
    session_namespace: str


@dataclasses.dataclass(frozen=True)
class OneShotRunResult:
    selected_service: str
    selected_model: str
    selected_effort: str
    used_fallback: bool
    selected_service_path: tuple[str, ...]
    raw_output: Any
    runtime_metadata: OneShotRuntimeMetadata


@dataclasses.dataclass(frozen=True)
class ResidentRuntimeMetadata:
    service_name: str
    provider_session_id: str | None
    run_kind: RunKind
    session_namespace: str
    exact_transcript_match: bool


@dataclasses.dataclass(frozen=True)
class ResidentRunResult:
    output: str
    runtime_metadata: ResidentRuntimeMetadata


@dataclasses.dataclass(frozen=True)
class ResidentRunRequest:
    prompt: str
    worktree: WorktreeMount
    model: str
    effort: str
    session_plan: ResidentSessionPlan
    tool_policy: ToolPolicy = ToolPolicy.FULL
    name: str = "Runtime Agent"
    status_display: Any = None
    work_body: str = ""
    token: CancellationToken | None = None

    @property
    def mount_path(self) -> Any:
        return self.worktree.host_path


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


def _selected_service_path(
    override: StageSelection,
    *,
    selected_service: str,
) -> tuple[str, ...]:
    path: list[str] = []
    for node in iter_stage_chain(override):
        if not node.service:
            continue
        path.append(node.service)
        if node.service == selected_service:
            return tuple(path)
    return (selected_service,)


def _require_execution_adapter_method(
    adapter: _PromptRuntimeExecutionAdapter,
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
        provider_state_dir_container_path=provider_state_dir_container_path,
        exact_transcript_match=exact_transcript_match,
    )


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


class _OneShotOutputAdapter:
    def __init__(self, *, prompt: str, session_namespace: str) -> None:
        self._prompt = prompt
        self._session_namespace = session_namespace
        self.runtime_metadata = OneShotRuntimeMetadata(
            provider_session_id=None,
            run_kind=RunKind.FRESH,
            session_namespace=session_namespace,
        )

    async def build_prompt(
        self,
        *,
        run_kind: RunKind,
        container_exec: Any,
    ) -> str:
        del run_kind, container_exec
        return self._prompt

    async def invoke(
        self,
        *,
        runner: Any,
        role: InvocationRole,
        prompt: str,
        run_kind: RunKind,
        session_uuid: str | None,
        on_provider_session_id: Any,
    ) -> Any:
        provider_session_id: str | None = None

        def _record_provider_session_id(value: str) -> None:
            nonlocal provider_session_id
            provider_session_id = value
            on_provider_session_id(value)

        raw_output = await runner.work(
            role,
            prompt,
            run_kind=run_kind,
            session_uuid=session_uuid,
            on_provider_session_id=_record_provider_session_id,
        )
        self.runtime_metadata = OneShotRuntimeMetadata(
            provider_session_id=provider_session_id or session_uuid,
            run_kind=run_kind,
            session_namespace=self._session_namespace,
        )
        return raw_output

    def is_successful_result(self, result: Any) -> bool:
        del result
        return True

    def protocol_reprompt_message(self) -> str | None:
        return None

    def protocol_error_result(self) -> Any | None:
        return None

    def non_typed_failure_result(self) -> Any | None:
        return None

    def protocol_error_types(self) -> tuple[type[BaseException], ...]:
        return ()

    def finalize_result(
        self,
        result: Any,
        *,
        role: InvocationRole,
        mount_path: Any,
        session_namespace: str,
        service_name: str,
    ) -> Any:
        del role, mount_path, session_namespace, service_name
        return result


class OneShotRuntime:
    def __init__(
        self,
        *,
        execution_adapter: OneShotRuntimeExecutionAdapter,
        service_registry: ServiceRegistry | dict[str, Any] | None = None,
    ) -> None:
        registry = (
            service_registry
            if isinstance(service_registry, ServiceRegistry)
            else ServiceRegistry(service_registry or {})
        )
        self._service_registry = registry
        self._execution_adapter = execution_adapter

    async def run_one_shot(self, request: OneShotRunRequest) -> OneShotRunResult:
        return await _run_one_shot(
            runner=self._execution_adapter,
            service_registry=self._service_registry,
            request=request,
        )


class ResidentRuntime:
    def __init__(
        self,
        *,
        execution_adapter: ResidentRuntimeExecutionAdapter,
    ) -> None:
        self._execution_adapter = execution_adapter

    async def run_resident_prompt(
        self,
        request: ResidentRunRequest,
    ) -> ResidentRunResult:
        return await _run_resident_prompt(
            runner=self._execution_adapter,
            request=request,
        )


async def _run_prompt(
    *,
    runner: _PromptRuntimeExecutionAdapter,
    service_registry: ServiceRegistry,
    request: _PromptRunRequest,
) -> str:
    resolved_override = service_registry.resolve(
        request.override,
        _time_module.now_local(),
    )
    role = request.role
    resolve_service = _require_execution_adapter_method(runner, "resolve_service")
    build_work_dependencies = _require_execution_adapter_method(
        runner,
        "build_work_dependencies",
    )
    resolved_service = resolve_service(resolved_override.service)
    dependencies = build_work_dependencies(
        name=request.name,
        model=resolved_override.model,
        effort=resolved_override.effort,
        service=resolved_service,
    )
    return await _invoke_runtime_intent(
        _RuntimeIntent(
            run_session=_build_run_session(
                mount_path=request.mount_path,
                role=role,
                session_namespace=request.session_namespace,
                service=resolved_service,
                container_workspace=dependencies.execution.container_workspace,
                usage_limit_scope=request.usage_limit_scope,
            ),
            model=resolved_override.model,
            effort=resolved_override.effort,
            output_adapter=TextOutputAdapter(
                prompt=request.prompt, tool_policy=request.tool_policy
            ),
            dependencies=dependencies,
            presentation=WorkInvocationPresentation(
                name=request.name,
                status_display=request.status_display,
                work_body=request.work_body,
            ),
            token=request.token,
        )
    )


async def _run_one_shot(
    *,
    runner: OneShotRuntimeExecutionAdapter,
    service_registry: ServiceRegistry,
    request: OneShotRunRequest,
) -> OneShotRunResult:
    if not service_registry.has_configured_candidate(request.override):
        raise RuntimeConfigurationError(
            "One-shot runtime requires at least one configured service candidate."
        )

    role = request.role
    resolve_service = _require_execution_adapter_method(runner, "resolve_service")
    build_work_dependencies = _require_execution_adapter_method(
        runner,
        "build_work_dependencies",
    )

    while True:
        now = _time_module.now_local()
        if request.token is not None and request.token.is_cancelled:
            raise UsageLimitError(
                reset_time=None,
                usage_limit_scope=request.usage_limit_scope
                or UsageLimitScope(role.value),
            )
        if not service_registry.has_available_for(request.override, now):
            resolved_override = service_registry.resolve(request.override, now)
            selected_service_name = resolved_override.service
            next_wake_time = service_registry.next_wake_time_for(
                request.override,
                now,
            )
            raise UsageLimitError(
                reset_time=next_wake_time,
                provider=selected_service_name,
                usage_limit_scope=request.usage_limit_scope
                or UsageLimitScope(role.value),
            )

        resolved_override = service_registry.resolve(
            request.override,
            now,
        )
        resolved_service = resolve_service(resolved_override.service)
        dependencies = build_work_dependencies(
            name=request.name,
            model=resolved_override.model,
            effort=resolved_override.effort,
            service=resolved_service,
        )
        output_adapter = _OneShotOutputAdapter(
            prompt=request.prompt,
            session_namespace=request.session_namespace,
        )
        attempt_token = (
            CancellationToken() if request.token is not None else request.token
        )
        try:
            raw_output = await _invoke_runtime_intent(
                _RuntimeIntent(
                    run_session=_build_run_session(
                        mount_path=request.mount_path,
                        role=role,
                        session_namespace=request.session_namespace,
                        service=resolved_service,
                        container_workspace=dependencies.execution.container_workspace,
                        usage_limit_scope=request.usage_limit_scope,
                    ),
                    model=resolved_override.model,
                    effort=resolved_override.effort,
                    output_adapter=output_adapter,
                    dependencies=dependencies,
                    presentation=WorkInvocationPresentation(
                        name=request.name,
                        status_display=request.status_display,
                        work_body=request.work_body,
                    ),
                    token=attempt_token,
                )
            )
        except Exception as exc:
            if isinstance(exc, UsageLimitError):
                service_registry.mark_exhausted(
                    resolved_override.service,
                    reset_time=exc.reset_time,
                )
                continue
            raise

        selected_service_path = _selected_service_path(
            request.override,
            selected_service=resolved_service.name,
        )
        return OneShotRunResult(
            selected_service=resolved_service.name,
            selected_model=resolved_override.model,
            selected_effort=resolved_override.effort,
            used_fallback=len(selected_service_path) > 1,
            selected_service_path=selected_service_path,
            raw_output=raw_output,
            runtime_metadata=output_adapter.runtime_metadata,
        )


async def _run_resident_prompt(
    *,
    runner: ResidentRuntimeExecutionAdapter,
    request: ResidentRunRequest,
) -> ResidentRunResult:
    build_work_dependencies = _require_execution_adapter_method(
        runner,
        "build_work_dependencies",
    )
    plan = request.session_plan
    dependencies = build_work_dependencies(
        name=request.name,
        model=request.model,
        effort=request.effort,
        service=plan.service,
    )
    prepared_session: Any = None

    def _prepare_session(run_session: RunSessionPlan) -> Any:
        nonlocal prepared_session
        if prepared_session is None:
            prepared_session = dependencies.execution.prepare_session(run_session)
        return prepared_session

    resident_dependencies = dataclasses.replace(
        dependencies,
        execution=dataclasses.replace(
            dependencies.execution,
            prepare_session=_prepare_session,
        ),
    )
    run_session = _build_run_session(
        mount_path=plan.worktree,
        role=plan.role,
        session_namespace=plan.namespace,
        service=plan.service,
        container_workspace=dependencies.execution.container_workspace,
        usage_limit_scope=plan.usage_limit_scope,
        run_kind=plan.run_kind,
        provider_session_id=plan.provider_session_id,
        provider_state_dir_container_path=plan.provider_state_dir_container_path(
            dependencies.execution.container_workspace
        ),
        exact_transcript_match=plan.exact_transcript_match,
    )
    output = await _invoke_runtime_intent(
        _RuntimeIntent(
            run_session=run_session,
            model=request.model,
            effort=request.effort,
            output_adapter=TextOutputAdapter(
                prompt=request.prompt,
                tool_policy=request.tool_policy,
            ),
            dependencies=resident_dependencies,
            presentation=WorkInvocationPresentation(
                name=request.name,
                status_display=request.status_display,
                work_body=request.work_body,
            ),
            token=request.token,
        )
    )
    if prepared_session is None:
        prepared_session = resident_dependencies.execution.prepare_session(run_session)
    provider_run_session = prepared_session.initial_provider_run_session()
    return ResidentRunResult(
        output=output,
        runtime_metadata=ResidentRuntimeMetadata(
            service_name=plan.service.name,
            provider_session_id=provider_run_session.provider_session_id,
            run_kind=plan.run_kind,
            session_namespace=plan.namespace,
            exact_transcript_match=plan.exact_transcript_match,
        ),
    )
