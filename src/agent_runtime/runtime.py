from __future__ import annotations

import dataclasses
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from . import _time as _time_module
from .contracts import ToolAccess, ToolPolicy, ToolPolicyProfile
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
from .errors import (
    AgentCancelledError,
    AgentTimeoutError,
    RuntimeConfigurationError,
    UsageLimitError,
)
from .identity import validate_session_namespace
from .invocation_progress import InvocationProgress
from .roles import InvocationRole
from .service_registry import ServiceRegistry
from .session import RunKind
from .session_planning import ResumableSessionPlan
from .stage_priority_chain import iter_stage_chain
from .types import StageSelection, validate_stage_selection
from .usage_limit_scope import UsageLimitScope
from .work import invoke_work

__all__ = [
    "OneShotRunRequest",
    "OneShotRunResult",
    "OneShotResultMetadata",
    "OneShotRuntime",
    "OneShotRuntimeExecutionAdapter",
    "OneShotRuntimeMetadata",
    "InvocationProgress",
    "RuntimeOutcome",
    "ResumableRunRequest",
    "ResumableRunResult",
    "ResumableRuntime",
    "ResumableRuntimeExecutionAdapter",
    "ResumableRuntimeMetadata",
    "ToolAccess",
    "ToolPolicy",
    "ToolPolicyProfile",
    "WorktreeMount",
]

OneShotRuntimeExecutionAdapter = _PromptRuntimeExecutionAdapter
ResumableRuntimeExecutionAdapter = _PromptRuntimeExecutionAdapter
_MISSING_TOOL_POLICY = object()

_DEFAULT_ONE_SHOT_NAME = "Runtime Agent"


@dataclasses.dataclass(frozen=True)
class RuntimeOutcome:
    kind: str
    output: str
    result: OneShotRunResult | ResumableRunResult | None = None
    service_name: str | None = None
    reset_time: datetime | None = None
    usage_limit_scope: UsageLimitScope | None = None
    invocation_progress: InvocationProgress | None = None

    @classmethod
    def completed(
        cls,
        *,
        output: str,
        result: OneShotRunResult | ResumableRunResult,
    ) -> RuntimeOutcome:
        return cls(kind="completed", output=output, result=result)

    @classmethod
    def usage_limited(
        cls,
        *,
        output: str,
        service_name: str | None,
        reset_time: datetime | None,
        usage_limit_scope: UsageLimitScope | None,
        invocation_progress: InvocationProgress,
    ) -> RuntimeOutcome:
        return cls(
            kind="usage_limited",
            output=output,
            service_name=service_name,
            reset_time=reset_time,
            usage_limit_scope=usage_limit_scope,
            invocation_progress=invocation_progress,
        )

    @classmethod
    def cancelled(
        cls,
        *,
        output: str,
        invocation_progress: InvocationProgress,
    ) -> RuntimeOutcome:
        return cls(
            kind="cancelled",
            output=output,
            invocation_progress=invocation_progress,
        )

    @classmethod
    def timed_out(
        cls,
        *,
        output: str,
        invocation_progress: InvocationProgress,
    ) -> RuntimeOutcome:
        return cls(
            kind="timed_out",
            output=output,
            invocation_progress=invocation_progress,
        )

    @property
    def runtime_metadata(self) -> OneShotRuntimeMetadata | ResumableRuntimeMetadata:
        result = self.result
        if result is None:
            raise AttributeError("Only completed outcomes carry runtime metadata.")
        if isinstance(result, OneShotRunResult):
            return result.runtime_metadata
        return result.runtime_metadata

    @property
    def metadata(self) -> OneShotResultMetadata:
        result = self.result
        if not isinstance(result, OneShotRunResult):
            raise AttributeError("Completed outcome does not carry one-shot metadata.")
        return result.metadata

    @property
    def selected_service_path(self) -> tuple[str, ...]:
        result = self.result
        if not isinstance(result, OneShotRunResult):
            raise AttributeError(
                "Completed outcome does not carry one-shot selection metadata."
            )
        return result.selected_service_path

    @property
    def selected_service(self) -> str:
        result = self.result
        if not isinstance(result, OneShotRunResult):
            raise AttributeError(
                "Completed outcome does not carry one-shot selection metadata."
            )
        return result.selected_service

    @property
    def selected_model(self) -> str:
        result = self.result
        if not isinstance(result, OneShotRunResult):
            raise AttributeError(
                "Completed outcome does not carry one-shot selection metadata."
            )
        return result.selected_model

    @property
    def selected_effort(self) -> str:
        result = self.result
        if not isinstance(result, OneShotRunResult):
            raise AttributeError(
                "Completed outcome does not carry one-shot selection metadata."
            )
        return result.selected_effort

    @property
    def used_fallback(self) -> bool:
        result = self.result
        if not isinstance(result, OneShotRunResult):
            raise AttributeError(
                "Completed outcome does not carry one-shot selection metadata."
            )
        return result.used_fallback

    @property
    def raw_output(self) -> str:
        return self.output


@dataclasses.dataclass(frozen=True, init=False)
class OneShotRunRequest:
    prompt: str
    worktree: Path
    stage: StageSelection
    role: InvocationRole
    usage_limit_scope: UsageLimitScope | None = None
    session_namespace: str = ""
    token: CancellationToken | None = None

    def __init__(
        self,
        prompt: str,
        worktree: Path | WorktreeMount,
        stage: StageSelection | None = None,
        role: InvocationRole | None = None,
        usage_limit_scope: UsageLimitScope | None = None,
        session_namespace: str = "",
        token: CancellationToken | None = None,
        *,
        override: StageSelection | None = None,
    ) -> None:
        if stage is None:
            stage = override
        elif override is not None and override != stage:
            raise TypeError(
                "OneShotRunRequest received conflicting `stage` and `override` values."
            )
        if stage is None:
            raise TypeError("OneShotRunRequest requires a `stage` value.")
        if role is None:
            raise TypeError("OneShotRunRequest requires a `role` value.")
        validate_stage_selection(stage)

        validate_session_namespace(session_namespace)

        object.__setattr__(self, "prompt", prompt)
        object.__setattr__(
            self,
            "worktree",
            worktree.host_path if isinstance(worktree, WorktreeMount) else worktree,
        )
        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "usage_limit_scope", usage_limit_scope)
        object.__setattr__(self, "session_namespace", session_namespace)
        object.__setattr__(self, "token", token)

    @property
    def mount_path(self) -> Path:
        return self.worktree

    @property
    def override(self) -> StageSelection:
        return self.stage


@dataclasses.dataclass(frozen=True)
class OneShotRuntimeMetadata:
    provider_session_id: str | None
    run_kind: RunKind
    session_namespace: str


@dataclasses.dataclass(frozen=True)
class OneShotResultMetadata:
    selected_service_path: tuple[str, ...]
    runtime: OneShotRuntimeMetadata


@dataclasses.dataclass(frozen=True)
class OneShotRunResult:
    output: str
    selected_service: str
    selected_model: str
    selected_effort: str
    used_fallback: bool
    metadata: OneShotResultMetadata

    @property
    def selected_service_path(self) -> tuple[str, ...]:
        return self.metadata.selected_service_path

    @property
    def runtime_metadata(self) -> OneShotRuntimeMetadata:
        return self.metadata.runtime

    @property
    def raw_output(self) -> str:
        return self.output


@dataclasses.dataclass(frozen=True)
class ResumableRuntimeMetadata:
    service_name: str
    provider_session_id: str | None
    run_kind: RunKind
    session_namespace: str
    exact_transcript_match: bool


@dataclasses.dataclass(frozen=True)
class ResumableRunResult:
    output: str
    runtime_metadata: ResumableRuntimeMetadata


@dataclasses.dataclass(frozen=True, init=False)
class ResumableRunRequest:
    prompt: str
    worktree: WorktreeMount
    model: str
    effort: str
    session_plan: ResumableSessionPlan
    tool_access: ToolAccess
    name: str = "Runtime Agent"
    status_display: Any = None
    work_body: str = ""
    token: CancellationToken | None = None

    def __init__(
        self,
        prompt: str,
        worktree: WorktreeMount,
        model: str,
        effort: str,
        session_plan: ResumableSessionPlan,
        tool_policy: ToolPolicy | object = _MISSING_TOOL_POLICY,
        tool_access: ToolAccess | object = _MISSING_TOOL_POLICY,
        name: str = "Runtime Agent",
        status_display: Any = None,
        work_body: str = "",
        token: CancellationToken | None = None,
    ) -> None:
        if (
            isinstance(tool_access, ToolAccess)
            and tool_policy is not _MISSING_TOOL_POLICY
        ):
            raise TypeError(
                "ResumableRunRequest received conflicting `tool_access` and `tool_policy` values."
            )
        if isinstance(tool_access, ToolAccess):
            resolved_tool_access = tool_access
        elif tool_policy is not _MISSING_TOOL_POLICY:
            resolved_tool_access = ToolAccess.workspace_backed(
                worktree.host_path,
                tool_policy=cast(ToolPolicy | ToolPolicyProfile, tool_policy),
            )
        else:
            raise TypeError(
                "ResumableRunRequest requires an explicit `tool_policy` value."
            )
        resolved_tool_access.require_workspace(
            worktree.host_path,
            context="ResumableRunRequest",
        )

        object.__setattr__(self, "prompt", prompt)
        object.__setattr__(self, "worktree", worktree)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "effort", effort)
        object.__setattr__(self, "session_plan", session_plan)
        object.__setattr__(self, "tool_access", resolved_tool_access)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "status_display", status_display)
        object.__setattr__(self, "work_body", work_body)
        object.__setattr__(self, "token", token)

    @property
    def mount_path(self) -> Any:
        return self.worktree.host_path

    @property
    def tool_policy(self) -> ToolPolicy | ToolPolicyProfile:
        return self.tool_access.tool_policy


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

        prompt_only = getattr(runner, "prompt_only", None)
        if not callable(prompt_only):
            raise RuntimeConfigurationError(
                "One-shot runtime requires a work runner with callable `prompt_only()`."
            )

        raw_output = await prompt_only(
            prompt,
            role=role,
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

    async def run_one_shot(self, request: OneShotRunRequest) -> RuntimeOutcome:
        try:
            result = await _run_one_shot(
                runner=self._execution_adapter,
                service_registry=self._service_registry,
                request=request,
            )
        except AgentCancelledError as exc:
            return RuntimeOutcome.cancelled(
                output="",
                invocation_progress=exc.invocation_progress,
            )
        except AgentTimeoutError as exc:
            return RuntimeOutcome.timed_out(
                output="",
                invocation_progress=exc.invocation_progress,
            )
        except UsageLimitError as exc:
            return RuntimeOutcome.usage_limited(
                output="",
                service_name=exc.service_name,
                reset_time=exc.reset_time,
                usage_limit_scope=exc.usage_limit_scope,
                invocation_progress=exc.invocation_progress,
            )
        return RuntimeOutcome.completed(output=result.output, result=result)


class ResumableRuntime:
    def __init__(
        self,
        *,
        execution_adapter: ResumableRuntimeExecutionAdapter,
    ) -> None:
        self._execution_adapter = execution_adapter

    async def run_resumable_prompt(
        self,
        request: ResumableRunRequest,
    ) -> RuntimeOutcome:
        try:
            result = await _run_resumable_prompt(
                runner=self._execution_adapter,
                request=request,
            )
        except AgentCancelledError as exc:
            return RuntimeOutcome.cancelled(
                output="",
                invocation_progress=exc.invocation_progress,
            )
        except AgentTimeoutError as exc:
            return RuntimeOutcome.timed_out(
                output="",
                invocation_progress=exc.invocation_progress,
            )
        except UsageLimitError as exc:
            return RuntimeOutcome.usage_limited(
                output="",
                service_name=exc.service_name,
                reset_time=exc.reset_time,
                usage_limit_scope=exc.usage_limit_scope,
                invocation_progress=exc.invocation_progress,
            )
        return RuntimeOutcome.completed(output=result.output, result=result)


async def _run_prompt(
    *,
    runner: _PromptRuntimeExecutionAdapter,
    service_registry: ServiceRegistry,
    request: _PromptRunRequest,
) -> str:
    resolved_override = service_registry.resolve(
        request.stage,
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
                prompt=request.prompt,
                tool_access=request.tool_access,
                workspace=request.worktree.host_path,
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
    if not service_registry.has_configured_candidate(request.stage):
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
            raise AgentCancelledError(
                invocation_progress=InvocationProgress.NOT_STARTED,
            )
        if not service_registry.has_available_for(request.stage, now):
            resolved_override = service_registry.resolve(request.stage, now)
            selected_service_name = resolved_override.service
            next_wake_time = service_registry.next_wake_time_for(
                request.stage,
                now,
            )
            raise UsageLimitError(
                reset_time=next_wake_time,
                service_name=selected_service_name,
                usage_limit_scope=request.usage_limit_scope
                or UsageLimitScope(role.value),
            )

        resolved_override = service_registry.resolve(
            request.stage,
            now,
        )
        resolved_service = resolve_service(resolved_override.service)
        dependencies = build_work_dependencies(
            name=_DEFAULT_ONE_SHOT_NAME,
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
                        name=_DEFAULT_ONE_SHOT_NAME,
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
                if not service_registry.has_available_for(
                    request.stage,
                    _time_module.now_local(),
                ):
                    raise
                continue
            raise

        selected_service_path = _selected_service_path(
            request.stage,
            selected_service=resolved_service.name,
        )
        return OneShotRunResult(
            output=raw_output if isinstance(raw_output, str) else str(raw_output),
            selected_service=resolved_service.name,
            selected_model=resolved_override.model,
            selected_effort=resolved_override.effort,
            used_fallback=len(selected_service_path) > 1,
            metadata=OneShotResultMetadata(
                selected_service_path=selected_service_path,
                runtime=output_adapter.runtime_metadata,
            ),
        )


async def _run_resumable_prompt(
    *,
    runner: ResumableRuntimeExecutionAdapter,
    request: ResumableRunRequest,
) -> ResumableRunResult:
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

    resumable_dependencies = dataclasses.replace(
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
        provider_state_dir_container_path=_provider_state_dir_container_path(
            worktree=plan.worktree,
            provider_state_dir=plan.provider_state_dir,
            provider_state_dir_relpath=getattr(
                plan, "_provider_state_dir_relpath", None
            ),
            container_workspace=dependencies.execution.container_workspace,
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
                tool_access=request.tool_access,
                workspace=request.worktree.host_path,
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
    if prepared_session is None:
        prepared_session = resumable_dependencies.execution.prepare_session(run_session)
    provider_run_session = prepared_session.initial_provider_run_session()
    return ResumableRunResult(
        output=output,
        runtime_metadata=ResumableRuntimeMetadata(
            service_name=plan.service.name,
            provider_session_id=provider_run_session.provider_session_id,
            run_kind=plan.run_kind,
            session_namespace=plan.namespace,
            exact_transcript_match=plan.exact_transcript_match,
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
