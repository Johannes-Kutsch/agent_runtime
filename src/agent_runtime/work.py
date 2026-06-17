from __future__ import annotations

import asyncio
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import Any

from .contracts import ExecutionProvider
from .execution_contracts import (
    CancellationToken,
    PreparedProviderRunSession,
    PreparedRunSessionState,
    PreparedSession,
    PrepareSessionAdapter,
    ProviderAccountExhaustionHandler,
    RunSessionPlan,
    SetupFailureTranslator,
    TextOutputAdapter,
    WorkExecutionAdapter,
    WorkExecutionDependencies,
    WorkFailureHandling,
    WorkInvocationDependencies,
    WorkInvocationPresentation,
    WorkInvocationRequest,
    WorkModelDisplayMetadata,
    WorkOutputAdapter,
    WorkPresentationDependencies,
    WorkResultT,
    WorkStatusDisplay,
    WorkStatusRow,
)
from .errors import (
    AgentCredentialFailureError,
    AgentTimeoutError,
    HardAgentError,
    TransientAgentError,
    UsageLimitError,
)
from .roles import InvocationRole
from .session import RunKind
from .usage_limit_scope import UsageLimitScope


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


def _default_provider_account_exhaustion_handler(
    service: ExecutionProvider,
    error: UsageLimitError,
) -> None:
    service.mark_exhausted(error.reset_time)


def _ensure_timeout_context(
    error: AgentTimeoutError,
    *,
    role: InvocationRole,
    mount_path: Path,
) -> AgentTimeoutError:
    if not error.role_value:
        error.role_value = role.value
        error.worktree_path = mount_path
    return error


async def invoke_work(request: WorkInvocationRequest[WorkResultT]) -> WorkResultT:
    status_display = request.status_display
    if status_display is None:
        status_display = request.dependencies.presentation.status_display_factory()

    token = request.token if request.token is not None else CancellationToken()
    if token.is_cancelled:
        raise UsageLimitError(
            reset_time=None,
            usage_limit_scope=(
                request.run_session.usage_limit_scope
                or UsageLimitScope(request.role.value)
            ),
        )

    run_session = request.run_session
    prepared_session = request.dependencies.execution.prepare_session(run_session)
    non_typed_retry_done = False
    initial_attempt = True

    async with request.dependencies.presentation.status_row_factory(
        status_display,
        request.name,
        kind="agent",
        must_close=False,
        work_body=request.work_body,
        color_key=request.color_key,
        model_display=_build_model_display_metadata(request),
    ) as row:
        session = request.dependencies.execution.build_session(
            request.mount_path,
            request.service,
            prepared_session.provider_state_dir_container_path,
        )
        runner = request.dependencies.execution.build_runner(session, status_display)
        try:
            git_name, git_email = request.dependencies.execution.get_git_identity()
            try:
                await runner.setup(git_name, git_email, request.work_body)
            except Exception as exc:
                translator = (
                    request.dependencies.failure_handling.translate_setup_failure
                )
                if translator is not None:
                    translated = translator(request.role, exc)
                    if translated is not None:
                        raise translated from exc
                raise

            prepared_session.prepare_for_run()
            loop = asyncio.get_running_loop()

            async def container_exec(cmd: str) -> str:
                return await loop.run_in_executor(None, session.exec_simple, cmd)

            retries_left = request.dependencies.failure_handling.timeout_retries
            while True:
                provider_run_session = (
                    prepared_session.initial_provider_run_session()
                    if initial_attempt
                    else prepared_session.resumable_provider_run_session()
                )
                try:
                    prompt = await request.output_adapter.build_prompt(
                        run_kind=provider_run_session.run_kind,
                        container_exec=container_exec,
                    )
                    result, successful_run_session = await _invoke_work_attempt(
                        request=request,
                        row=row,
                        prepared_session=prepared_session,
                        runner=runner,
                        prompt=prompt,
                        provider_run_session=provider_run_session,
                    )
                    if request.output_adapter.is_successful_result(result):
                        successful_run_session.record_successful_run()
                    else:
                        row.close("failed", shutdown_style="error")
                    return request.output_adapter.finalize_result(
                        result,
                        role=request.role,
                        mount_path=request.mount_path,
                        session_namespace=request.session_namespace,
                        service_name=request.service.name,
                    )
                except AgentTimeoutError as err:
                    _ensure_timeout_context(
                        err,
                        role=request.role,
                        mount_path=request.mount_path,
                    )
                    if retries_left <= 0:
                        raise
                    restart_num = (
                        request.dependencies.failure_handling.timeout_retries
                        - retries_left
                        + 1
                    )
                    status_display.print(
                        request.name,
                        "Timeout — restarting"
                        " "
                        f"(attempt {restart_num}/"
                        f"{request.dependencies.failure_handling.timeout_retries})",
                    )
                    retries_left -= 1
                    initial_attempt = False
                except UsageLimitError as err:
                    if err.usage_limit_scope is None:
                        err.usage_limit_scope = (
                            request.run_session.usage_limit_scope
                            or UsageLimitScope(request.role.value)
                        )
                        err.stage_key = err.usage_limit_scope.value
                    request.dependencies.failure_handling.handle_provider_account_exhaustion(
                        request.service,
                        err,
                    )
                    token.cancel()
                    raise
                except TransientAgentError as err:
                    token.cancel()
                    transient_status_message = (
                        request.dependencies.failure_handling.transient_status_message
                    )
                    if transient_status_message is not None:
                        status_display.print(
                            request.name,
                            transient_status_message(err),
                        )
                    raise
                except AgentCredentialFailureError as err:
                    token.cancel()
                    err.caller = request.name
                    if not err.service_name:
                        err.service_name = request.service.name
                    raise
                except HardAgentError as err:
                    token.cancel()
                    err.caller = request.name
                    err.service_name = request.service.name
                    raise
                except Exception:
                    if (
                        not request.allow_non_typed_resume_retry
                        or provider_run_session.run_kind != RunKind.RESUME
                    ):
                        raise
                    failure_result = request.output_adapter.non_typed_failure_result()
                    if failure_result is None:
                        raise
                    if non_typed_retry_done:
                        row.close("failed", shutdown_style="error")
                        return request.output_adapter.finalize_result(
                            failure_result,
                            role=request.role,
                            mount_path=request.mount_path,
                            session_namespace=request.session_namespace,
                            service_name=request.service.name,
                        )
                    non_typed_retry_done = True
        finally:
            try:
                session.__exit__(None, None, None)
            except Exception:
                pass


async def _invoke_work_attempt(
    *,
    request: WorkInvocationRequest[WorkResultT],
    row: Any,
    prepared_session: Any,
    runner: WorkExecutionAdapter,
    prompt: str,
    provider_run_session: Any,
) -> tuple[WorkResultT, Any]:
    reprompt_message = request.output_adapter.protocol_reprompt_message()
    protocol_error_result = request.output_adapter.protocol_error_result()
    protocol_error_types = request.output_adapter.protocol_error_types()
    max_attempts = (
        3 if reprompt_message is not None and protocol_error_result is not None else 1
    )
    work_prompt = prompt
    work_run_session = provider_run_session
    for _ in range(max_attempts):
        try:
            result = await request.output_adapter.invoke(
                runner=runner,
                role=request.role,
                prompt=work_prompt,
                run_kind=work_run_session.run_kind,
                session_uuid=work_run_session.provider_session_id,
                on_provider_session_id=work_run_session.record_provider_session_id,
            )
            return result, work_run_session
        except Exception as exc:
            if not protocol_error_types or not isinstance(exc, protocol_error_types):
                raise
            if reprompt_message is None or protocol_error_result is None:
                raise
            next_run_session = prepared_session.protocol_reprompt_provider_run_session()
            if next_run_session is None:
                row.close("failed", shutdown_style="error")
                return protocol_error_result, work_run_session
            work_prompt = reprompt_message
            work_run_session = next_run_session
    row.close("failed", shutdown_style="error")
    assert protocol_error_result is not None
    return protocol_error_result, work_run_session


def _build_model_display_metadata(
    request: WorkInvocationRequest[Any],
) -> WorkModelDisplayMetadata | None:
    build_model_display_metadata = (
        request.dependencies.presentation.build_model_display_metadata
    )
    if build_model_display_metadata is None:
        return WorkModelDisplayMetadata(
            service=request.service.name,
            model=request.model,
            effort=request.effort,
        )
    return build_model_display_metadata(
        request.service.name,
        request.model,
        request.effort,
    )


__all__ = [
    "CancellationToken",
    "PreparedProviderRunSession",
    "PreparedRunSessionState",
    "PreparedSession",
    "PrepareSessionAdapter",
    "RunSessionPlan",
    "ProviderAccountExhaustionHandler",
    "SetupFailureTranslator",
    "TextOutputAdapter",
    "WorkModelDisplayMetadata",
    "WorkExecutionAdapter",
    "WorkExecutionDependencies",
    "WorkFailureHandling",
    "WorkStatusDisplay",
    "WorkStatusRow",
    "WorkInvocationDependencies",
    "WorkInvocationPresentation",
    "WorkInvocationRequest",
    "WorkOutputAdapter",
    "WorkPresentationDependencies",
    "invoke_work",
]
