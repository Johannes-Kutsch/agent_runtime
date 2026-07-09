from __future__ import annotations

import logging
import tempfile
import subprocess as _subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable, cast

from . import _builtin_provider_rendering as _builtin_provider_rendering_module
from ._builtin_provider_stream_interpretation import (
    BuiltInProviderStreamInterpretation,
    resolve_built_in_provider_session_id,
    classify_built_in_provider_invocation_progress,
    emit_built_in_provider_live_output_event,
    opencode_lifecycle_built_in_provider_stream_interpretation,
    opencode_built_in_provider_stream_interpretation,
    reduce_codex_stream,
    reduce_claude_stream,
    reduce_opencode_stream,
)
from ._provider_invocation import (
    InvocationFailureKind,
    ProductionProviderInvocationAdapter,
    ProviderInvocationAdapter,
    ProviderInvocationFailure,
    ProviderInvocationPrompt,
    ProviderInvocationRequest,
    ProviderInvocationResult,
    ProviderOutputReductionHooks,
)
from ._runtime_lifecycle import (
    AgentEvent,
    CancellationToken,
    EphemeralRunRequest,
    ProviderAuth,
    ProviderUsage,
    ResumedSessionRunRequest,
    RunResult,
    RuntimeOutcome,
    NewSessionRunRequest,
)
from .types import ResolvedProvider
from .errors import (
    AgentCancelledError,
    AgentCredentialFailureError,
    AgentTimeoutError,
    ProviderUnavailableError,
    ProviderUnavailableReason,
    RuntimeConfigurationError,
    UsageLimitError,
)
from .invocation_progress import InvocationProgress
from .session import RunKind
from .types import ProviderSelection
from ._session_backed_provider_lifecycle_policy import policy_for_service

_log = logging.getLogger(__name__)
subprocess = _subprocess
_CLAUDE_VALID_MODELS = _builtin_provider_rendering_module._CLAUDE_VALID_MODELS
_CLAUDE_VALID_EFFORTS = _builtin_provider_rendering_module._CLAUDE_VALID_EFFORTS
_CODEX_VALID_MODELS = _builtin_provider_rendering_module._CODEX_VALID_MODELS
_CODEX_VALID_EFFORTS = _builtin_provider_rendering_module._CODEX_VALID_EFFORTS
_OPENCODE_GO_PROVIDER_ID = _builtin_provider_rendering_module._OPENCODE_GO_PROVIDER_ID
_OPENCODE_GO_BASE_URL = _builtin_provider_rendering_module._OPENCODE_GO_BASE_URL
_OPENCODE_SESSION_ID_FILENAME = "session_id"
_BUILTIN_PROVIDER_PROMPT_FILENAME = ".provider_prompt"
_OPENCODE_GO_MODELS = _builtin_provider_rendering_module._OPENCODE_GO_MODELS
_OPENCODE_VALID_EFFORTS = _builtin_provider_rendering_module._OPENCODE_VALID_EFFORTS
_SUPPORTED_BUILTIN_SERVICES = frozenset({"claude", "codex", "opencode"})
_PORTABLE_CONTINUATION_PROVIDERS = frozenset({"claude", "codex", "opencode"})
_SERVICE_NOT_AVAILABLE_DETAIL = (
    "No configured service candidates are currently available."
)


def _builtin_provider_prompt_path(invocation_dir: Path) -> Path:
    return invocation_dir / _BUILTIN_PROVIDER_PROMPT_FILENAME


def _builtin_provider_temp_prompt_path() -> Path:
    return _builtin_provider_prompt_path(Path("/tmp"))


def supported_builtin_provider_selection(
    provider_selection: ProviderSelection,
) -> ProviderSelection | None:
    if provider_selection.service in _SUPPORTED_BUILTIN_SERVICES:
        return provider_selection
    return None


class _ObservedOutputReducer:
    __slots__ = ("reduce_output", "consume_stdout_lines")

    def __init__(
        self,
        reduce_output: Callable[[list[str]], tuple[str, ProviderUsage | None]],
        consume_stdout_lines: Callable[[list[str]], None],
    ) -> None:
        self.reduce_output = reduce_output
        self.consume_stdout_lines = consume_stdout_lines

    def __call__(self, lines: list[str]) -> tuple[str, ProviderUsage | None]:
        return self.reduce_output(lines)


class _SessionTimeoutState:
    def __init__(
        self,
        *,
        tracking_interpretation: BuiltInProviderStreamInterpretation,
        fallback_provider_session_id: str | None,
    ) -> None:
        self._tracking_interpretation = tracking_interpretation
        self._fallback_provider_session_id = fallback_provider_session_id
        self._observed_lines: list[str] = []
        self.usage: ProviderUsage | None = None
        self.provider_session_id: str | None = fallback_provider_session_id
        self.invocation_progress = InvocationProgress.NOT_STARTED

    def record(self, lines: list[str]) -> None:
        self._observed_lines.extend(lines)
        try:
            _, self.usage = self._tracking_interpretation.reduce_output(
                self._observed_lines
            )
        except (UsageLimitError, ProviderUnavailableError) as exc:
            if exc.usage is not None:
                self.usage = exc.usage
        self.provider_session_id = resolve_built_in_provider_session_id(
            self._tracking_interpretation,
            self._observed_lines,
            fallback_provider_session_id=self._fallback_provider_session_id,
        )
        self.invocation_progress = classify_built_in_provider_invocation_progress(
            self._tracking_interpretation,
            self._observed_lines,
            provider_session_id=self.provider_session_id,
        )

    def apply_to_timeout(self, exc: AgentTimeoutError) -> None:
        if exc.usage is None:
            exc.usage = self.usage
        exc.invocation_progress = self.invocation_progress
        setattr(exc, "provider_session_id", self.provider_session_id)

    def apply_to_cancellation(self, exc: AgentCancelledError) -> None:
        if exc.usage is None:
            exc.usage = self.usage
        exc.invocation_progress = self.invocation_progress
        setattr(exc, "provider_session_id", self.provider_session_id)


def _observe_output_lines(
    *,
    lines: list[str],
    on_live_output: Callable[[AgentEvent], None] | None,
    stream_interpretation: BuiltInProviderStreamInterpretation,
) -> None:
    if on_live_output is None:
        return
    for line in lines:
        emit_built_in_provider_live_output_event(
            stream_interpretation.build_agent_event(line),
            on_live_output,
        )


def _observe_output_reducer(
    stream_interpretation: BuiltInProviderStreamInterpretation,
    on_live_output: Callable[[AgentEvent], None] | None,
) -> Callable[[list[str]], tuple[str, ProviderUsage | None]]:
    if on_live_output is None:
        return stream_interpretation.reduce_output

    return _ObservedOutputReducer(
        reduce_output=stream_interpretation.reduce_output,
        consume_stdout_lines=(
            lambda lines: _observe_output_lines(
                lines=lines,
                on_live_output=on_live_output,
                stream_interpretation=stream_interpretation,
            )
        ),
    )


def _with_observed_output(
    stream_interpretation: BuiltInProviderStreamInterpretation,
    on_live_output: Callable[[AgentEvent], None] | None,
) -> BuiltInProviderStreamInterpretation:
    if on_live_output is None:
        return stream_interpretation
    return BuiltInProviderStreamInterpretation(
        reduce_output=_observe_output_reducer(stream_interpretation, on_live_output),
        build_agent_event=stream_interpretation.build_agent_event,
        classify_invocation_progress=(
            stream_interpretation.classify_invocation_progress
        ),
        extract_provider_session_id=stream_interpretation.extract_provider_session_id,
    )


def _with_reduce_output(
    stream_interpretation: BuiltInProviderStreamInterpretation,
    reduce_output: Callable[[list[str]], tuple[str, ProviderUsage | None]],
) -> BuiltInProviderStreamInterpretation:
    return BuiltInProviderStreamInterpretation(
        reduce_output=reduce_output,
        build_agent_event=stream_interpretation.build_agent_event,
        classify_invocation_progress=(
            stream_interpretation.classify_invocation_progress
        ),
        extract_provider_session_id=stream_interpretation.extract_provider_session_id,
    )


def _with_session_timeout_state(
    stream_interpretation: BuiltInProviderStreamInterpretation,
    *,
    tracking_interpretation: BuiltInProviderStreamInterpretation,
    fallback_provider_session_id: str | None,
) -> tuple[BuiltInProviderStreamInterpretation, _SessionTimeoutState]:
    timeout_state = _SessionTimeoutState(
        tracking_interpretation=tracking_interpretation,
        fallback_provider_session_id=fallback_provider_session_id,
    )
    consume_stdout_lines = getattr(
        stream_interpretation.reduce_output, "consume_stdout_lines", None
    )

    def _consume_with_timeout_state(lines: list[str]) -> None:
        timeout_state.record(lines)
        if callable(consume_stdout_lines):
            try:
                consume_stdout_lines(lines)
            except AgentTimeoutError as exc:
                timeout_state.apply_to_timeout(exc)
                raise

    return (
        _with_reduce_output(
            stream_interpretation,
            _ObservedOutputReducer(
                reduce_output=stream_interpretation.reduce_output,
                consume_stdout_lines=_consume_with_timeout_state,
            ),
        ),
        timeout_state,
    )


def _validate_codex_auth() -> None:
    _builtin_provider_rendering_module._require_codex_auth()


def _codex_host_auth_path() -> Path:
    return _builtin_provider_rendering_module._codex_host_auth_path()


def _missing_codex_auth_error() -> AgentCredentialFailureError:
    return _builtin_provider_rendering_module._missing_codex_auth_error()


def _opencode_stream_interpretation(
    *,
    on_live_output: Callable[[AgentEvent], None] | None = None,
    on_provider_session_id: Callable[[str], None] | None = None,
    fallback_provider_session_id: str | None = None,
    reduce_output: Callable[[list[str]], tuple[str, ProviderUsage | None]]
    | None = None,
    extract_provider_session_id: Callable[[list[str]], str | None] | None = None,
) -> BuiltInProviderStreamInterpretation:
    if (
        on_live_output is not None
        or on_provider_session_id is not None
        or fallback_provider_session_id is not None
    ):
        return opencode_lifecycle_built_in_provider_stream_interpretation(
            on_live_output=on_live_output,
            on_provider_session_id=on_provider_session_id,
            fallback_provider_session_id=fallback_provider_session_id,
            reduce_output=reduce_output,
        )
    return opencode_built_in_provider_stream_interpretation(
        reduce_output=reduce_output,
        extract_provider_session_id=extract_provider_session_id,
    )


def _provider_invocation_error_from_failure(
    service_name: str,
    failure: ProviderInvocationFailure,
) -> UsageLimitError | ProviderUnavailableError:
    stream_interpretation = policy_for_service(service_name).stream_interpretation()
    invocation_progress = (
        InvocationProgress.STARTED
        if classify_built_in_provider_invocation_progress(
            stream_interpretation,
            list(failure.stdout_lines),
            provider_session_id=failure.provider_session_id,
        )
        is InvocationProgress.STARTED
        else InvocationProgress.NOT_STARTED
    )
    if failure.kind is InvocationFailureKind.USAGE_LIMITED:
        error: UsageLimitError | ProviderUnavailableError = UsageLimitError(
            reset_time=cast(datetime | None, failure.reset_time),
            raw_message=(failure.detail if failure.reset_time is None else None),
            service_name=service_name,
            invocation_progress=invocation_progress,
            usage=failure.usage,
        )
    else:
        error = ProviderUnavailableError(
            failure.detail,
            reason=(
                failure.provider_unavailable_reason
                or (
                    ProviderUnavailableReason.SERVICE_NOT_AVAILABLE
                    if failure.detail == _SERVICE_NOT_AVAILABLE_DETAIL
                    else ProviderUnavailableReason.TRANSIENT_API_ERROR
                )
            ),
            service_name=service_name,
            invocation_progress=invocation_progress,
            usage=failure.usage,
        )
    setattr(error, "provider_session_id", failure.provider_session_id)
    return error


def _select_builtin_stage(stage: ProviderSelection) -> ProviderSelection:
    candidate = supported_builtin_provider_selection(stage)
    if candidate is not None:
        return candidate
    raise RuntimeConfigurationError(
        "RuntimeClient requires at least one supported built-in service candidate."
    )


def _new_provider_session_id() -> str:
    return str(uuid.uuid4())


def _default_provider_invocation_adapter() -> ProviderInvocationAdapter:
    return ProductionProviderInvocationAdapter()


def _invoke_provider(
    *,
    provider_invocation_adapter: ProviderInvocationAdapter,
    command: str,
    command_argv: tuple[str, ...],
    prefer_argv: bool,
    worktree: Path,
    environment: dict[str, str],
    prompt_content: str,
    prompt_path: Path | None,
    cleanup_prompt_path: bool,
    run_kind: RunKind,
    provider_session_id: str | None,
    stream_interpretation: BuiltInProviderStreamInterpretation,
    timeout_seconds: int = 300,
) -> ProviderInvocationResult | ProviderInvocationFailure:
    return provider_invocation_adapter.execute(
        ProviderInvocationRequest(
            command=command,
            argv=command_argv,
            prefer_argv=prefer_argv,
            worktree=worktree,
            environment=environment,
            prompt=ProviderInvocationPrompt(
                content=prompt_content,
                path=prompt_path,
                cleanup_path=cleanup_prompt_path,
            ),
            run_kind=run_kind,
            provider_session_id=provider_session_id,
            output_hooks=ProviderOutputReductionHooks(
                reduce_output=stream_interpretation.reduce_output,
                extract_provider_session_id=(
                    stream_interpretation.extract_provider_session_id
                ),
            ),
            timeout_seconds=timeout_seconds,
        )
    )


def _provider_invocation_request_from_rendered_invocation(
    *,
    rendered: _builtin_provider_rendering_module.BuiltInProviderRenderedInvocation,
    invocation_dir: Path,
    prompt: str,
    run_kind: RunKind,
    provider_session_id: str | None,
    stream_interpretation: BuiltInProviderStreamInterpretation,
    normalize_prompt_file_command_for_argv: bool = False,
    timeout_seconds: int = 300,
    token: CancellationToken | None = None,
) -> ProviderInvocationRequest:
    command = rendered.legacy_command_text or ""
    if (
        normalize_prompt_file_command_for_argv
        and rendered.prefer_argv
        and rendered.prompt_transport_preference
        is _builtin_provider_rendering_module.PromptTransportPreference.PROMPT_FILE
    ):
        command = ""
    return ProviderInvocationRequest(
        command=command,
        argv=rendered.canonical_argv,
        prefer_argv=rendered.prefer_argv,
        worktree=invocation_dir,
        environment=dict(rendered.environment),
        prompt=ProviderInvocationPrompt(
            content=prompt,
            path=rendered.prompt_path,
            cleanup_path=(
                rendered.prompt_cleanup_choice
                is _builtin_provider_rendering_module.PromptCleanupChoice.DELETE_AFTER_INVOCATION
            ),
        ),
        run_kind=run_kind,
        provider_session_id=provider_session_id,
        output_hooks=ProviderOutputReductionHooks(
            reduce_output=stream_interpretation.reduce_output,
            extract_provider_session_id=(
                stream_interpretation.extract_provider_session_id
            ),
        ),
        timeout_seconds=timeout_seconds,
        token=token,
    )


def _execute_rendered_provider_invocation(
    *,
    provider_invocation_adapter: ProviderInvocationAdapter,
    rendered: _builtin_provider_rendering_module.BuiltInProviderRenderedInvocation,
    invocation_dir: Path,
    argv_transform: (
        Callable[[tuple[str, ...], Path, dict[str, str]], tuple[str, ...]] | None
    ) = None,
    prompt: str,
    run_kind: RunKind,
    provider_session_id: str | None,
    stream_interpretation: BuiltInProviderStreamInterpretation,
    normalize_prompt_file_command_for_argv: bool = False,
    timeout_seconds: int = 300,
    token: CancellationToken | None = None,
) -> ProviderInvocationResult | ProviderInvocationFailure:
    request = _provider_invocation_request_from_rendered_invocation(
        rendered=rendered,
        invocation_dir=invocation_dir,
        prompt=prompt,
        run_kind=run_kind,
        provider_session_id=provider_session_id,
        stream_interpretation=stream_interpretation,
        normalize_prompt_file_command_for_argv=(normalize_prompt_file_command_for_argv),
        timeout_seconds=timeout_seconds,
        token=token,
    )
    if argv_transform is None:
        return provider_invocation_adapter.execute(request)
    return provider_invocation_adapter.execute(request, argv_transform=argv_transform)


def _ephemeral_provider_state_dir(
    service_name: str,
    invocation_dir: Path,
) -> tuple[Path, Callable[[], None]] | tuple[None, Callable[[], None]]:
    if service_name == "opencode":
        return invocation_dir, lambda: None
    temp_dir = tempfile.TemporaryDirectory(prefix="ephemeral-provider-state-")
    return Path(temp_dir.name), temp_dir.cleanup


def _ephemeral_render_invocation_dir(
    service_name: str,
    invocation_dir: Path,
) -> Path:
    if service_name == "opencode":
        return Path("/tmp")
    return invocation_dir


def _render_ephemeral_provider_invocation(
    request: EphemeralRunRequest,
    stage: ProviderSelection,
    provider_state_dir: Path | None,
) -> _builtin_provider_rendering_module.BuiltInProviderRenderedInvocation:
    return _builtin_provider_rendering_module.render_built_in_provider_invocation(
        _builtin_provider_rendering_module.BuiltInProviderRenderRequest(
            provider_selection=(
                _builtin_provider_rendering_module.BuiltInProviderSelectionFacts(
                    service=stage.service,
                    model=stage.model,
                    effort=stage.effort,
                )
            ),
            run_kind=RunKind.FRESH,
            tool_access=request.tool_access,
            auth=_selection_auth(stage),
            invocation_dir=_ephemeral_render_invocation_dir(
                stage.service,
                request.invocation_dir,
            ),
            provider_state_dir=provider_state_dir,
        ),
        argv_transform=request.argv_transform,
    )


def _run_builtin_ephemeral(
    request: EphemeralRunRequest,
    *,
    provider_invocation_adapter: ProviderInvocationAdapter | None = None,
    select_builtin_stage: Callable[
        [ProviderSelection], ProviderSelection
    ] = _select_builtin_stage,
    reduce_claude_stream: Callable[
        [list[str], Callable[[AgentEvent], None] | None],
        tuple[str, ProviderUsage | None],
    ] = reduce_claude_stream,
    reduce_codex_stream: Callable[
        [list[str], Callable[[AgentEvent], None] | None],
        tuple[str, ProviderUsage | None],
    ] = reduce_codex_stream,
    reduce_opencode_stream: Callable[
        [list[str], Callable[[AgentEvent], None] | None],
        tuple[str, ProviderUsage | None],
    ] = reduce_opencode_stream,
) -> RunResult:
    if request.token is not None and request.token.is_cancelled:
        raise AgentCancelledError()
    invocation_adapter = (
        _default_provider_invocation_adapter()
        if provider_invocation_adapter is None
        else provider_invocation_adapter
    )
    selected_stage = select_builtin_stage(request.provider_selection)
    provider_state_dir, cleanup_provider_state_dir = _ephemeral_provider_state_dir(
        selected_stage.service,
        request.invocation_dir,
    )
    try:
        rendered = _render_ephemeral_provider_invocation(
            request,
            selected_stage,
            provider_state_dir=provider_state_dir,
        )
        stream_interpretation: BuiltInProviderStreamInterpretation
        if selected_stage.service == "codex":
            from . import (
                _session_backed_provider_state_resolution as _provider_state_resolution,
            )

            _provider_state_resolution._seed_codex_auth(
                cast(Path, provider_state_dir),
                _codex_host_auth_path(),
            )
            stream_interpretation = _with_observed_output(
                _with_reduce_output(
                    policy_for_service("codex").stream_interpretation(),
                    lambda lines: reduce_codex_stream(lines, None),
                ),
                request.on_live_output,
            )
        elif selected_stage.service == "opencode":
            stream_interpretation = _opencode_stream_interpretation(
                on_live_output=request.on_live_output,
                reduce_output=lambda lines: reduce_opencode_stream(lines, None),
            )
        else:
            stream_interpretation = _with_observed_output(
                _with_reduce_output(
                    policy_for_service("claude").stream_interpretation(),
                    lambda lines: reduce_claude_stream(lines, None),
                ),
                request.on_live_output,
            )
        invocation_result = _execute_rendered_provider_invocation(
            provider_invocation_adapter=invocation_adapter,
            rendered=rendered,
            invocation_dir=request.invocation_dir,
            argv_transform=request.argv_transform,
            prompt=request.prompt,
            run_kind=RunKind.FRESH,
            provider_session_id=rendered.provider_session_id,
            stream_interpretation=stream_interpretation,
            normalize_prompt_file_command_for_argv=True,
            timeout_seconds=request.timeout_seconds,
            token=request.token,
        )
    finally:
        cleanup_provider_state_dir()
    if isinstance(invocation_result, ProviderInvocationFailure):
        raise _provider_invocation_error_from_failure(
            selected_stage.service,
            invocation_result,
        )
    result_text = invocation_result.output
    usage = invocation_result.usage
    return RunResult(
        output=result_text,
        usage=usage,
        continuation=None,
        selected=ResolvedProvider(
            service=selected_stage.service,
            model=selected_stage.model,
            effort=selected_stage.effort,
        ),
    )


def _new_session_runtime_state_dir(
    request: NewSessionRunRequest,
    *,
    context: str,
) -> tuple[Path, Callable[[], None], bool]:
    runtime_state_dir = request.session_store
    if runtime_state_dir is not None:
        return runtime_state_dir, lambda: None, True
    temp_dir = tempfile.TemporaryDirectory(prefix=f"{context}-provider-state-")
    return Path(temp_dir.name), temp_dir.cleanup, False


def _selection_auth(selection: ProviderSelection) -> ProviderAuth | None:
    return selection.auth


def _require_portable_continuation_support(service_name: str) -> None:
    if service_name not in _PORTABLE_CONTINUATION_PROVIDERS:
        raise RuntimeConfigurationError(
            f"Portable continuation support is required for session-backed "
            f"execution with {service_name!r}."
        )


def _run_builtin_new_session(
    request: NewSessionRunRequest,
    *,
    provider_invocation_adapter: ProviderInvocationAdapter | None = None,
) -> RuntimeOutcome:
    from . import _session_backed_provider_execution as _module

    return _module._run_builtin_new_session(
        request,
        provider_invocation_adapter=provider_invocation_adapter,
    )


def _run_builtin_resumed_session(
    request: ResumedSessionRunRequest,
    *,
    provider_invocation_adapter: ProviderInvocationAdapter | None = None,
) -> RuntimeOutcome:
    from . import _session_backed_provider_execution as _module

    return _module._run_builtin_resumed_session(
        request,
        provider_invocation_adapter=provider_invocation_adapter,
    )
