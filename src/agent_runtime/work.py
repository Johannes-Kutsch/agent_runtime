from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import Any

from .contracts import ExecutionService, ParsedTurn
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
    WorkInvocationDependencies,
    WorkInvocationRequest,
    WorkModelDisplayMetadata,
    WorkOutputAdapter,
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
from .roles import AgentRole
from .session import RunKind


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
    service: ExecutionService,
    error: UsageLimitError,
) -> None:
    service.mark_exhausted(error.reset_time)


def reduce_text_output_events(
    events: Iterable[ParsedTurn],
    on_turn: Callable[[str], None],
    on_tokens: Callable[[int], None] | None = None,
    *,
    provider: str,
) -> str:
    from .contracts import (
        AssistantTurn,
        CredentialFailure,
        HardError,
        PromptTokens,
        Result,
        TransientError,
        UnsupportedTokens,
        UsageLimit,
    )

    result_text: str | None = None
    collected_turns: list[str] = []
    for event in events:
        if isinstance(event, UsageLimit):
            raise UsageLimitError(
                reset_time=event.reset_time,
                raw_message=event.raw_message,
                provider=provider,
                is_permanent=event.is_permanent,
            )
        if isinstance(event, TransientError):
            raise TransientAgentError(
                message=event.raw_message,
                status_code=event.status_code,
            )
        if isinstance(event, HardError):
            raise HardAgentError(
                message=event.raw_message,
                status_code=event.status_code,
                service_name=provider,
                classification=event.classification,
                observations=event.observations,
            )
        if isinstance(event, CredentialFailure):
            raise AgentCredentialFailureError(
                message=event.raw_message,
                status_code=event.status_code,
                service_name=event.service_name,
                classification=event.classification,
                observations=event.source_observations,
            )
        if isinstance(event, PromptTokens):
            if on_tokens is not None:
                on_tokens(event.count)
            continue
        if isinstance(event, UnsupportedTokens):
            continue
        if isinstance(event, AssistantTurn):
            on_turn(event.text)
            collected_turns.append(event.text)
            continue
        if isinstance(event, Result):
            result_text = event.text
            break
    if result_text is not None:
        return result_text
    return "\n".join(collected_turns)


def _ensure_timeout_context(
    error: AgentTimeoutError,
    *,
    role: AgentRole,
    mount_path: Path,
) -> AgentTimeoutError:
    if not error.role_value:
        error.role_value = role.value
        error.worktree_path = mount_path
    return error


async def invoke_work(request: WorkInvocationRequest[WorkResultT]) -> WorkResultT:
    status_display = request.status_display
    if status_display is None:
        status_display = request.dependencies.status_display_factory()

    token = request.token if request.token is not None else CancellationToken()
    if token.is_cancelled:
        raise UsageLimitError(
            reset_time=None,
            stage_key=request.dependencies.stage_key_for_role(request.role),
        )

    run_session = request.run_session
    assert run_session is not None
    prepared_session = request.dependencies.prepare_session(run_session)
    non_typed_retry_done = False
    initial_attempt = True

    async with request.dependencies.status_row_factory(
        status_display,
        request.name,
        kind="agent",
        must_close=False,
        work_body=request.work_body,
        color_key=request.color_key,
        model_display=_build_model_display_metadata(request),
    ) as row:
        session = request.dependencies.build_session(
            request.mount_path,
            request.service,
            prepared_session.provider_state_dir_container_path,
        )
        runner = request.dependencies.build_runner(session, status_display)
        try:
            git_name, git_email = request.dependencies.get_git_identity()
            try:
                await runner.setup(git_name, git_email, request.work_body)
            except Exception as exc:
                if request.dependencies.translate_setup_failure is not None:
                    translated = request.dependencies.translate_setup_failure(
                        request.role, exc
                    )
                    if translated is not None:
                        raise translated from exc
                raise

            prepared_session.prepare_for_run()
            loop = asyncio.get_running_loop()

            async def container_exec(cmd: str) -> str:
                return await loop.run_in_executor(None, session.exec_simple, cmd)

            retries_left = request.dependencies.timeout_retries
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
                        request.dependencies.timeout_retries - retries_left + 1
                    )
                    status_display.print(
                        request.name,
                        "Timeout — restarting"
                        f" (attempt {restart_num}/{request.dependencies.timeout_retries})",
                    )
                    retries_left -= 1
                    initial_attempt = False
                except UsageLimitError as err:
                    if err.stage_key is None:
                        err.stage_key = request.dependencies.stage_key_for_role(
                            request.role
                        )
                    request.dependencies.handle_provider_account_exhaustion(
                        request.service,
                        err,
                    )
                    token.cancel()
                    raise
                except TransientAgentError as err:
                    token.cancel()
                    if request.dependencies.transient_status_message is not None:
                        status_display.print(
                            request.name,
                            request.dependencies.transient_status_message(err),
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
    if request.dependencies.build_model_display_metadata is None:
        return WorkModelDisplayMetadata(
            service=request.service.name,
            model=request.model,
            effort=request.effort,
        )
    return request.dependencies.build_model_display_metadata(
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
    "WorkStatusDisplay",
    "WorkStatusRow",
    "WorkInvocationDependencies",
    "WorkInvocationRequest",
    "WorkOutputAdapter",
    "invoke_work",
]
