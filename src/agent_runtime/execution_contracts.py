from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import Any, Generic, Protocol, TypeVar

from .contracts import ExecutionService, ToolPolicy
from .errors import AgentTimeoutError, UsageLimitError
from .roles import AgentRole
from .session import RunKind
from .types import StageOverride

WorkResultT = TypeVar("WorkResultT")


@dataclasses.dataclass(frozen=True)
class WorktreeMount:
    host_path: Path


@dataclasses.dataclass(frozen=True)
class PromptRunSession:
    namespace: str = ""
    plan: Any = None


class PromptRuntimeExecutionAdapter(Protocol):
    def resolve_service(self, service_name: str = "") -> ExecutionService: ...

    def build_work_dependencies(
        self,
        *,
        name: str,
        model: str,
        effort: str,
        service: ExecutionService,
    ) -> WorkInvocationDependencies: ...


@dataclasses.dataclass(frozen=True)
class PromptRunRequest:
    prompt: str
    worktree: WorktreeMount
    override: StageOverride
    tool_policy: ToolPolicy = ToolPolicy.FULL
    name: str = "Runtime Agent"
    status_display: Any = None
    work_body: str = ""
    token: CancellationToken | None = None
    session: PromptRunSession = dataclasses.field(default_factory=PromptRunSession)

    @property
    def mount_path(self) -> Path:
        return self.worktree.host_path

    @property
    def session_namespace(self) -> str:
        return self.session.namespace

    @property
    def run_session_plan(self) -> Any:
        return self.session.plan


@dataclasses.dataclass(frozen=True)
class RunSessionPlan:
    mount_path: Path
    role: AgentRole
    session_namespace: str
    service: ExecutionService
    container_workspace: str
    run_session_plan: Any = None


@dataclasses.dataclass(frozen=True)
class WorkModelDisplayMetadata:
    service: str
    model: str
    effort: str


class WorkStatusDisplay(Protocol):
    def register(
        self,
        caller: str,
        kind: str,
        startup_message: str = "started",
        work_body: str = "",
        initial_phase: str = "Setup",
        color_key: int | None = None,
        model_display: WorkModelDisplayMetadata | None = None,
    ) -> None: ...

    def update_phase(self, name: str, phase: str) -> None: ...

    def reset_idle_timer(self, name: str) -> None: ...

    def update_tokens(self, name: str, current_tokens: int) -> None: ...

    def remove(
        self,
        caller: str,
        shutdown_message: str = "finished",
        shutdown_style: str = "success",
    ) -> None: ...

    def print(self, caller: str, message: object, style: str | None = None) -> None: ...


class WorkStatusRow(Protocol):
    def close(
        self,
        shutdown_message: str = "finished",
        *,
        shutdown_style: str = "success",
    ) -> None: ...


class _PlainStatusDisplay:
    def __init__(self) -> None:
        self._last_caller: str | None = None
        self._last_kind: str | None = None
        self._kinds: dict[str, str] = {}

    def _blank_before(self, caller: str) -> bool:
        if caller == "":
            return True
        if caller == self._last_caller:
            return False
        kinds = {self._last_kind, self._kinds.get(caller)}
        if "agent" in kinds and kinds <= {"phase", "agent"}:
            return False
        return True

    def register(
        self,
        caller: str,
        kind: str,
        startup_message: str = "started",
        work_body: str = "",
        initial_phase: str = "Setup",
        color_key: int | None = None,
        model_display: WorkModelDisplayMetadata | None = None,
    ) -> None:
        del work_body, initial_phase, color_key, model_display
        if caller != "":
            self._kinds[caller] = kind
        self.print(caller, startup_message)

    def update_phase(self, name: str, phase: str) -> None:
        del name, phase

    def reset_idle_timer(self, name: str) -> None:
        del name

    def update_tokens(self, name: str, current_tokens: int) -> None:
        del name, current_tokens

    def remove(
        self,
        caller: str,
        shutdown_message: str = "finished",
        shutdown_style: str = "success",
    ) -> None:
        del shutdown_style
        self.print(caller, shutdown_message)
        self._kinds.pop(caller, None)

    def print(self, caller: str, message: object, style: str | None = None) -> None:
        del style
        lines = str(message).split("\n")
        if self._blank_before(caller):
            print()
        self._last_caller = caller
        self._last_kind = self._kinds.get(caller)
        for line in lines:
            if caller:
                print(f"[{caller}] {line}")
            else:
                print(line)


class _StatusRowHandle:
    def __init__(self, status_display: WorkStatusDisplay, caller: str) -> None:
        self._status_display = status_display
        self._caller = caller
        self._closed = False

    def close(
        self,
        shutdown_message: str = "finished",
        *,
        shutdown_style: str = "success",
    ) -> None:
        if self._closed:
            return
        self._status_display.remove(
            self._caller,
            shutdown_message,
            shutdown_style,
        )
        self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed


class _DefaultStatusRow:
    def __init__(
        self,
        status_display: WorkStatusDisplay,
        caller: str,
        *,
        kind: str,
        must_close: bool,
        color_key: int | None = None,
        work_body: str = "",
        initial_phase: str = "Setup",
        startup_message: str = "started",
        model_display: WorkModelDisplayMetadata | None = None,
    ) -> None:
        self._status_display = status_display
        self._caller = caller
        self._must_close = must_close
        self._kind = kind
        self._color_key = color_key
        self._work_body = work_body
        self._initial_phase = initial_phase
        self._startup_message = startup_message
        self._model_display = model_display
        self._row = _StatusRowHandle(status_display, caller)

    async def __aenter__(self) -> WorkStatusRow:
        self._status_display.register(
            self._caller,
            self._kind,
            startup_message=self._startup_message,
            work_body=self._work_body,
            initial_phase=self._initial_phase,
            color_key=self._color_key,
            model_display=self._model_display,
        )
        return self._row

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        del tb
        if self._row.closed:
            return False
        if exc is None:
            if self._must_close:
                self._row.close("failed", shutdown_style="error")
            else:
                self._row.close()
            return False
        if isinstance(exc, UsageLimitError):
            self._row.close("usage limit reached", shutdown_style="interrupted")
            return False
        if isinstance(exc, AgentTimeoutError):
            self._row.close("timed out", shutdown_style="interrupted")
            return False
        self._row.close("failed", shutdown_style="error")
        return False


def _default_status_display_factory() -> WorkStatusDisplay:
    return _PlainStatusDisplay()


def _default_status_row_factory(
    status_display: WorkStatusDisplay,
    caller: str,
    *,
    kind: str,
    must_close: bool,
    color_key: int | None = None,
    work_body: str = "",
    initial_phase: str = "Setup",
    startup_message: str = "started",
    model_display: WorkModelDisplayMetadata | None = None,
) -> AbstractAsyncContextManager[WorkStatusRow]:
    return _DefaultStatusRow(
        status_display,
        caller,
        kind=kind,
        must_close=must_close,
        color_key=color_key,
        work_body=work_body,
        initial_phase=initial_phase,
        startup_message=startup_message,
        model_display=model_display,
    )


class PreparedProviderRunSession(Protocol):
    run_kind: RunKind
    provider_session_id: str | None

    def record_provider_session_id(self, provider_session_id: str) -> None: ...

    def record_successful_run(self) -> None: ...


class PreparedRunSessionState(Protocol):
    provider_state_dir_container_path: str | None

    def prepare_for_run(self) -> None: ...

    def initial_provider_run_session(self) -> PreparedProviderRunSession: ...

    def resumable_provider_run_session(self) -> PreparedProviderRunSession: ...

    def protocol_reprompt_provider_run_session(
        self,
    ) -> PreparedProviderRunSession | None: ...


PreparedSession = PreparedRunSessionState
PrepareSessionAdapter = Callable[[RunSessionPlan], PreparedRunSessionState]
StatusRowFactory = Callable[..., AbstractAsyncContextManager[Any]]
SetupFailureTranslator = Callable[[AgentRole, BaseException], BaseException | None]
ProviderAccountExhaustionHandler = Callable[[ExecutionService, Any], None]
StatusDisplayFactory = Callable[[], WorkStatusDisplay]


def _default_provider_account_exhaustion_handler(
    service: ExecutionService,
    error: Any,
) -> None:
    service.mark_exhausted(error.reset_time)


@dataclasses.dataclass
class CancellationToken:
    _event: asyncio.Event = dataclasses.field(
        default_factory=asyncio.Event,
        init=False,
        repr=False,
    )

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> None:
        self._event.set()


class WorkExecutionAdapter(Protocol):
    async def setup(
        self, git_name: str, git_email: str, work_body: str = ""
    ) -> None: ...

    async def work(
        self,
        role: AgentRole,
        prompt: str,
        *,
        run_kind: RunKind = RunKind.FRESH,
        session_uuid: str | None = None,
        on_provider_session_id: Callable[[str], None] | None = None,
    ) -> Any: ...

    async def work_text(
        self,
        prompt: str,
        *,
        role: AgentRole = AgentRole.IMPLEMENTER,
        tool_policy: Any = ToolPolicy.FULL,
        run_kind: RunKind = RunKind.FRESH,
        session_uuid: str | None = None,
        on_provider_session_id: Callable[[str], None] | None = None,
    ) -> str: ...


class WorkOutputAdapter(Protocol[WorkResultT]):
    async def build_prompt(
        self,
        *,
        run_kind: RunKind,
        container_exec: Callable[[str], Awaitable[str]],
    ) -> str: ...

    async def invoke(
        self,
        *,
        runner: WorkExecutionAdapter,
        role: AgentRole,
        prompt: str,
        run_kind: RunKind,
        session_uuid: str | None,
        on_provider_session_id: Callable[[str], None],
    ) -> WorkResultT: ...

    def is_successful_result(self, result: WorkResultT) -> bool: ...

    def protocol_reprompt_message(self) -> str | None: ...

    def protocol_error_result(self) -> WorkResultT | None: ...

    def protocol_error_types(self) -> tuple[type[BaseException], ...]: ...

    def non_typed_failure_result(self) -> WorkResultT | None: ...

    def finalize_result(
        self,
        result: WorkResultT,
        *,
        role: AgentRole,
        mount_path: Path,
        session_namespace: str,
        service_name: str,
    ) -> WorkResultT: ...


@dataclasses.dataclass(frozen=True)
class WorkInvocationDependencies:
    container_workspace: str
    timeout_retries: int
    stage_key_for_role: Callable[[AgentRole], str | None]
    prepare_session: PrepareSessionAdapter
    build_session: Callable[[Path, ExecutionService, str | None], Any]
    build_runner: Callable[[Any, Any], WorkExecutionAdapter]
    get_git_identity: Callable[[], tuple[str, str]]
    status_display_factory: StatusDisplayFactory = _default_status_display_factory
    status_row_factory: StatusRowFactory = _default_status_row_factory
    translate_setup_failure: SetupFailureTranslator | None = None
    build_model_display_metadata: Callable[[str, str, str], Any | None] | None = None
    handle_provider_account_exhaustion: ProviderAccountExhaustionHandler = (
        _default_provider_account_exhaustion_handler
    )
    transient_status_message: Callable[[Any], str] | None = None


@dataclasses.dataclass(frozen=True)
class WorkInvocationRequest(Generic[WorkResultT]):
    name: str
    mount_path: Path
    role: AgentRole
    service: ExecutionService
    model: str
    effort: str
    output_adapter: WorkOutputAdapter[WorkResultT] = dataclasses.field(repr=False)
    dependencies: WorkInvocationDependencies = dataclasses.field(repr=False)
    status_display: Any = None
    token: CancellationToken | None = None
    work_body: str = ""
    session_namespace: str = ""
    run_session_plan: Any = None
    run_session: RunSessionPlan | None = None
    color_key: int | None = None
    allow_non_typed_resume_retry: bool = False

    def __post_init__(self) -> None:
        run_session = self.run_session
        if run_session is None:
            run_session = RunSessionPlan(
                mount_path=self.mount_path,
                role=self.role,
                session_namespace=self.session_namespace,
                service=self.service,
                container_workspace=self.dependencies.container_workspace,
                run_session_plan=self.run_session_plan,
            )
        object.__setattr__(self, "run_session", run_session)
        object.__setattr__(self, "mount_path", run_session.mount_path)
        object.__setattr__(self, "role", run_session.role)
        object.__setattr__(self, "service", run_session.service)
        object.__setattr__(self, "session_namespace", run_session.session_namespace)
        object.__setattr__(self, "run_session_plan", run_session.run_session_plan)


@dataclasses.dataclass(frozen=True)
class TextOutputAdapter:
    prompt: str
    tool_policy: Any = ToolPolicy.FULL

    async def build_prompt(
        self,
        *,
        run_kind: RunKind,
        container_exec: Callable[[str], Awaitable[str]],
    ) -> str:
        del run_kind, container_exec
        return self.prompt

    async def invoke(
        self,
        *,
        runner: WorkExecutionAdapter,
        role: AgentRole,
        prompt: str,
        run_kind: RunKind,
        session_uuid: str | None,
        on_provider_session_id: Callable[[str], None],
    ) -> str:
        return await runner.work_text(
            prompt,
            role=role,
            tool_policy=self.tool_policy,
            run_kind=run_kind,
            session_uuid=session_uuid,
            on_provider_session_id=on_provider_session_id,
        )

    def is_successful_result(self, result: str) -> bool:
        return True

    def protocol_reprompt_message(self) -> str | None:
        return None

    def protocol_error_result(self) -> str | None:
        return None

    def non_typed_failure_result(self) -> str | None:
        return None

    def protocol_error_types(self) -> tuple[type[BaseException], ...]:
        return ()

    def finalize_result(
        self,
        result: str,
        *,
        role: AgentRole,
        mount_path: Path,
        session_namespace: str,
        service_name: str,
    ) -> str:
        del role, mount_path, session_namespace, service_name
        return result


__all__ = [
    "CancellationToken",
    "PreparedProviderRunSession",
    "PreparedRunSessionState",
    "PreparedSession",
    "PrepareSessionAdapter",
    "PromptRunRequest",
    "PromptRunSession",
    "PromptRuntimeExecutionAdapter",
    "ProviderAccountExhaustionHandler",
    "RunSessionPlan",
    "SetupFailureTranslator",
    "StatusDisplayFactory",
    "StatusRowFactory",
    "TextOutputAdapter",
    "WorkExecutionAdapter",
    "WorkInvocationDependencies",
    "WorkInvocationRequest",
    "WorkModelDisplayMetadata",
    "WorkOutputAdapter",
    "WorkResultT",
    "WorkStatusDisplay",
    "WorkStatusRow",
    "WorktreeMount",
]
