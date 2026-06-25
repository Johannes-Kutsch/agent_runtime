from __future__ import annotations

import json
import logging
import os
import shlex
import tempfile
import subprocess as _subprocess
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, cast

from . import _time as _time_module
from . import _builtin_provider_rendering as _builtin_provider_rendering_module
from ._builtin_provider_stream_interpretation import (
    BuiltInProviderStreamInterpretation,
    classify_built_in_provider_invocation_progress,
    claude_built_in_provider_stream_interpretation,
    codex_built_in_provider_stream_interpretation,
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
    EphemeralRunRequest,
    ProviderAuth,
    ProviderUsage,
    ResumedSessionRunRequest,
    RunResult,
    RuntimeOutcome,
    NewSessionRunRequest,
)
from .types import ResolvedProvider
from .contracts import (
    ToolAccess,
    ToolPolicy,
    ToolPolicyProfile,
)
from .errors import (
    AgentCredentialFailureError,
    ProviderUnavailableError,
    ProviderUnavailableReason,
    RuntimeConfigurationError,
    UsageLimitError,
)
from .invocation_progress import InvocationProgress
from .session import RunKind, provider_state_relpath
from .types import ProviderSelection

_log = logging.getLogger(__name__)
subprocess = _subprocess
_CLAUDE_VALID_MODELS = _builtin_provider_rendering_module._CLAUDE_VALID_MODELS
_CLAUDE_VALID_EFFORTS = _builtin_provider_rendering_module._CLAUDE_VALID_EFFORTS
_CODEX_VALID_MODELS = frozenset(
    {
        "gpt-5.5",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.3-codex",
        "gpt-5.3-codex-spark",
        "gpt-5.2",
    }
)
_CODEX_VALID_EFFORTS = frozenset({"low", "medium", "high", "xhigh"})
_OPENCODE_GO_PROVIDER_ID = "opencode-go"
_OPENCODE_GO_BASE_URL = "https://opencode.ai/zen/go/v1"
_OPENCODE_SESSION_ID_FILENAME = "session_id"
_BUILTIN_PROVIDER_PROMPT_FILENAME = ".provider_prompt"
_OPENCODE_GO_MODELS = frozenset(
    {
        "deepseek-v4-flash",
        "deepseek-v4-pro",
        "glm-5.1",
        "glm-5.2",
        "kimi-k2.6",
        "kimi-k2.7-code",
        "mimo-v2.5",
        "mimo-v2.5-pro",
        "minimax-m2.7",
        "minimax-m3",
        "qwen3.6-plus",
        "qwen3.7-max",
        "qwen3.7-plus",
    }
)
_OPENCODE_VALID_EFFORTS = frozenset({"medium"})
_SUPPORTED_BUILTIN_SERVICES = frozenset({"claude", "codex", "opencode"})
_PORTABLE_CONTINUATION_PROVIDERS = frozenset({"claude", "codex", "opencode"})
_WAKE_TIME_BUFFER = timedelta(minutes=2)
_SERVICE_NOT_AVAILABLE_DETAIL = (
    "No configured service candidates are currently available."
)


def _builtin_provider_prompt_path(invocation_dir: Path) -> Path:
    return invocation_dir / _BUILTIN_PROVIDER_PROMPT_FILENAME


def _builtin_provider_temp_prompt_path() -> Path:
    return _builtin_provider_prompt_path(Path("/tmp"))


def compute_wake_time(
    reset_time: datetime | None,
    now: datetime,
) -> tuple[datetime, bool]:
    if reset_time is not None:
        return reset_time + _WAKE_TIME_BUFFER, False
    next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    return next_hour + _WAKE_TIME_BUFFER, True


class BuiltInAvailabilityState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._exhausted_until_by_service: dict[str, datetime] = {}

    def _is_available_locked(self, service_name: str, now: datetime) -> bool:
        exhausted_until = self._exhausted_until_by_service.get(service_name)
        if exhausted_until is None:
            return True
        if exhausted_until <= now:
            self._exhausted_until_by_service.pop(service_name, None)
            return True
        return False

    def first_available_stage(
        self,
        stage: ProviderSelection,
        *,
        now: datetime,
    ) -> ProviderSelection | None:
        with self._lock:
            if stage.service not in _SUPPORTED_BUILTIN_SERVICES:
                return None
            if self._is_available_locked(stage.service, now):
                return stage
        return None

    def has_available_stage(self, stage: ProviderSelection, *, now: datetime) -> bool:
        return self.first_available_stage(stage, now=now) is not None

    def next_wake_time(
        self, stage: ProviderSelection, *, now: datetime
    ) -> datetime | None:
        with self._lock:
            if stage.service not in _SUPPORTED_BUILTIN_SERVICES:
                return None
            exhausted_until = self._exhausted_until_by_service.get(stage.service)
            if exhausted_until is None:
                return None
            if exhausted_until <= now:
                self._exhausted_until_by_service.pop(stage.service, None)
                return None
            return exhausted_until

    def mark_exhausted(
        self,
        service_name: str,
        *,
        reset_time: datetime | None,
        now: datetime,
    ) -> None:
        wake, _ = compute_wake_time(reset_time, now)
        if wake.tzinfo is None:
            wake = wake.replace(tzinfo=timezone.utc)
        with self._lock:
            current = self._exhausted_until_by_service.get(service_name)
            if current is None or wake > current:
                self._exhausted_until_by_service[service_name] = wake


def supported_builtin_provider_selection(
    provider_selection: ProviderSelection,
) -> ProviderSelection | None:
    if provider_selection.service in _SUPPORTED_BUILTIN_SERVICES:
        return provider_selection
    return None


def _validate_claude_stage(stage: ProviderSelection) -> None:
    _builtin_provider_rendering_module._validate_claude_selection(
        _builtin_provider_rendering_module.BuiltInProviderSelectionFacts(
            service=stage.service,
            model=stage.model,
            effort=stage.effort,
        )
    )


def _validate_codex_stage(stage: ProviderSelection) -> None:
    if stage.model not in _CODEX_VALID_MODELS:
        raise RuntimeConfigurationError(f"Unsupported Codex model {stage.model!r}.")
    if stage.effort not in _CODEX_VALID_EFFORTS:
        raise RuntimeConfigurationError(f"Unsupported Codex effort {stage.effort!r}.")


def _validate_opencode_stage(stage: ProviderSelection) -> None:
    if stage.model not in _OPENCODE_GO_MODELS:
        raise RuntimeConfigurationError(f"Unsupported OpenCode model {stage.model!r}.")
    if stage.effort not in _OPENCODE_VALID_EFFORTS:
        raise RuntimeConfigurationError(
            f"Unsupported OpenCode effort {stage.effort!r}."
        )


def _claude_command(
    *,
    model: str,
    effort: str,
    tool_access: ToolAccess,
    run_kind: RunKind = RunKind.FRESH,
    session_uuid: str | None = None,
) -> tuple[str, ...]:
    return _builtin_provider_rendering_module.render_built_in_provider_invocation(
        _builtin_provider_rendering_module.BuiltInProviderRenderRequest(
            provider_selection=(
                _builtin_provider_rendering_module.BuiltInProviderSelectionFacts(
                    service="claude",
                    model=model,
                    effort=effort,
                )
            ),
            run_kind=run_kind,
            tool_access=tool_access,
            auth=ProviderAuth(claude_code_oauth_token="token"),
            invocation_dir=Path("/tmp"),
            provider_session_id=session_uuid,
        )
    ).canonical_argv


def _claude_legacy_command_text(
    *,
    model: str,
    effort: str,
    tool_access: ToolAccess,
    prompt_path: Path,
    run_kind: RunKind = RunKind.FRESH,
    session_uuid: str | None = None,
) -> str:
    rendered = _builtin_provider_rendering_module.render_built_in_provider_invocation(
        _builtin_provider_rendering_module.BuiltInProviderRenderRequest(
            provider_selection=(
                _builtin_provider_rendering_module.BuiltInProviderSelectionFacts(
                    service="claude",
                    model=model,
                    effort=effort,
                )
            ),
            run_kind=run_kind,
            tool_access=tool_access,
            auth=ProviderAuth(claude_code_oauth_token="token"),
            invocation_dir=prompt_path.parent,
            provider_session_id=session_uuid,
        )
    )
    assert rendered.legacy_command_text is not None
    command_prefix, separator, _ = rendered.legacy_command_text.rpartition(" < ")
    if not separator:
        return rendered.legacy_command_text
    return f"{command_prefix} < {shlex.quote(str(prompt_path))}"


def _claude_env(
    *,
    auth: ProviderAuth | None,
    state_dir_container_path: str | None = None,
) -> dict[str, str]:
    return _builtin_provider_rendering_module._claude_environment(
        auth,
        (None if state_dir_container_path is None else Path(state_dir_container_path)),
    )


def _codex_command(
    *,
    model: str,
    effort: str,
    tool_access: ToolAccess,
    run_kind: RunKind = RunKind.FRESH,
    session_uuid: str | None = None,
    os_name: str | None = None,
) -> tuple[str, ...]:
    tool_policy = tool_access.tool_policy
    executable = "codex.cmd" if (os_name or os.name) == "nt" else "codex"
    if run_kind == RunKind.RESUME and session_uuid:
        parts = [executable, "exec", "resume", session_uuid]
    else:
        parts = [executable, "exec"]
    if model:
        parts.extend(["-m", model])
    if effort:
        parts.extend(["-c", f"model_reasoning_effort={effort}"])
    parts.extend(["-c", "approval_policy=never"])
    if tool_policy is ToolPolicy.UNRESTRICTED:
        parts.extend(["--sandbox", "danger-full-access"])
    elif tool_policy is ToolPolicy.NONE:
        parts.extend(["--sandbox", "read-only"])
    elif tool_policy is ToolPolicy.INSPECT_ONLY:
        parts.extend(["--sandbox", "read-only"])
    elif tool_policy is ToolPolicy.NO_FILE_MUTATION:
        parts.extend(["--sandbox", "read-only"])
    else:
        parts.extend(["--sandbox", "danger-full-access"])
    parts.append("--json")
    return tuple(parts)


def _codex_env(
    *,
    state_dir_container_path: str | None = None,
) -> dict[str, str]:
    env: dict[str, str] = {"TZ": "UTC"}
    if state_dir_container_path:
        env["CODEX_HOME"] = state_dir_container_path
    return env


def _opencode_go_model_ref(model: str) -> str:
    if "/" in model:
        return model
    return f"{_OPENCODE_GO_PROVIDER_ID}/{model}"


def _opencode_tool_policy_permission(
    tool_policy: ToolPolicy | ToolPolicyProfile,
) -> dict[str, str] | str | None:
    profile = (
        tool_policy.profile if isinstance(tool_policy, ToolPolicy) else tool_policy
    )
    if profile == ToolPolicy.NONE.profile:
        return "deny"
    if profile == ToolPolicy.INSPECT_ONLY.profile:
        return {"edit": "deny", "bash": "deny"}
    if profile == ToolPolicy.NO_FILE_MUTATION.profile:
        return {"edit": "deny"}
    return None


def _opencode_go_config_content(
    *,
    tool_policy: ToolPolicy | ToolPolicyProfile | None = None,
) -> str:
    config: dict[str, Any] = {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            _OPENCODE_GO_PROVIDER_ID: {
                "npm": "@ai-sdk/openai-compatible",
                "name": "OpenCode Go",
                "options": {
                    "baseURL": _OPENCODE_GO_BASE_URL,
                    "apiKey": "{env:OPENCODE_GO_API_KEY}",
                },
                "models": {
                    model: {"name": model} for model in sorted(_OPENCODE_GO_MODELS)
                },
            }
        },
    }
    if tool_policy is not None:
        permission = _opencode_tool_policy_permission(tool_policy)
        if permission is not None:
            config["permission"] = permission
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def _opencode_command(
    *,
    model: str,
    effort: str,
    run_kind: RunKind = RunKind.FRESH,
    session_uuid: str | None = None,
    os_name: str | None = None,
) -> tuple[str, ...]:
    del effort
    executable = "opencode.cmd" if (os_name or os.name) == "nt" else "opencode"
    parts = [executable, "run", "--format", "json"]
    if run_kind == RunKind.RESUME and session_uuid:
        parts.extend(["--session", session_uuid])
    if model:
        parts.extend(["--model", _opencode_go_model_ref(model)])
    return tuple(parts)


def _windows_process_base_env(
    *,
    os_name: str | None = None,
    environ: dict[str, str] | None = None,
) -> dict[str, str]:
    if (os_name or os.name) != "nt":
        return {}
    source_env = os.environ if environ is None else environ
    return {
        key: source_env[key]
        for key in ("PATH", "PATHEXT", "SystemRoot", "ComSpec", "WINDIR")
        if key in source_env and source_env[key]
    }


def _legacy_command_text(
    command_argv: tuple[str, ...],
    prompt_path: Path,
    *,
    opencode_prompt_substitution: bool = False,
) -> str:
    command = " ".join(shlex.quote(part) for part in command_argv)
    if opencode_prompt_substitution:
        return f'{command} "$(cat {shlex.quote(str(prompt_path))})"'
    return f"{command} < {shlex.quote(str(prompt_path))}"


def _opencode_env(
    *,
    auth: ProviderAuth | None,
    state_dir_container_path: str | None = None,
    tool_policy: ToolPolicy | ToolPolicyProfile | None = None,
    os_name: str | None = None,
    environ: dict[str, str] | None = None,
) -> dict[str, str]:
    env: dict[str, str] = {
        **_windows_process_base_env(os_name=os_name, environ=environ),
        "TZ": "UTC",
    }
    if state_dir_container_path:
        env["OPENCODE_HOME"] = state_dir_container_path
    api_key = None if auth is None else auth.opencode_api_key
    if api_key:
        env["OPENCODE_GO_API_KEY"] = api_key
        env["OPENCODE_CONFIG_CONTENT"] = _opencode_go_config_content(
            tool_policy=tool_policy
        )
    return env


class _IdleTimeoutWatchdog:
    def __init__(self, timeout_seconds: int) -> None:
        self.timeout_seconds = timeout_seconds
        self._lock = threading.Lock()
        self._start_time: datetime | None = None
        self._last_event_time: datetime | None = None
        self._stop_event = threading.Event()
        self._timeout_occurred = False

    def reset_timer(self) -> None:
        with self._lock:
            self._last_event_time = _time_module.now_local()

    def start_monitoring(self) -> None:
        with self._lock:
            self._start_time = _time_module.now_local()
            self._last_event_time = self._start_time
        self._stop_event.clear()
        thread = threading.Thread(target=self._monitor_loop, daemon=True)
        thread.start()

    def stop_monitoring(self) -> None:
        self._stop_event.set()

    def _monitor_loop(self) -> None:
        while not self._stop_event.is_set():
            with self._lock:
                if self._last_event_time is not None:
                    elapsed = (
                        _time_module.now_local() - self._last_event_time
                    ).total_seconds()
                    if elapsed > self.timeout_seconds:
                        self._timeout_occurred = True
                        return
            self._stop_event.wait(timeout=0.1)

    def check_timeout(self) -> None:
        with self._lock:
            if self._timeout_occurred:
                from .errors import AgentTimeoutError

                raise AgentTimeoutError(
                    "Idle timeout: no Agent Event within configured window"
                )


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


def _wrap_on_live_output_with_timeout(
    on_live_output: Callable[[AgentEvent], None] | None,
    timeout_seconds: int,
) -> tuple[Callable[[AgentEvent], None] | None, _IdleTimeoutWatchdog | None]:
    if timeout_seconds <= 0:
        return on_live_output, None

    watchdog = _IdleTimeoutWatchdog(timeout_seconds)
    watchdog.start_monitoring()

    def wrapper(event: AgentEvent) -> None:
        watchdog.reset_timer()
        if on_live_output is not None:
            on_live_output(event)
        watchdog.check_timeout()

    return wrapper, watchdog


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


def _validate_codex_auth() -> None:
    auth_path = _codex_host_auth_path()
    if auth_path.exists():
        return
    raise _missing_codex_auth_error()


def _codex_host_auth_path() -> Path:
    return Path.home() / ".codex" / "auth.json"


def _missing_codex_auth_error() -> AgentCredentialFailureError:
    message = "Codex authentication missing: run `codex login` on the host."
    return AgentCredentialFailureError(
        message=message,
        service_name="codex",
    )


def _claude_stream_interpretation() -> BuiltInProviderStreamInterpretation:
    return claude_built_in_provider_stream_interpretation()


def _codex_stream_interpretation() -> BuiltInProviderStreamInterpretation:
    return codex_built_in_provider_stream_interpretation()


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


def _stream_interpretation_for_service(
    service_name: str,
) -> BuiltInProviderStreamInterpretation:
    if service_name == "claude":
        return _claude_stream_interpretation()
    if service_name == "codex":
        return _codex_stream_interpretation()
    if service_name == "opencode":
        return _opencode_stream_interpretation()
    raise RuntimeConfigurationError(
        "RuntimeClient session-backed execution is only implemented for Claude, Codex, and OpenCode."
    )


def _provider_invocation_error_from_failure(
    service_name: str,
    failure: ProviderInvocationFailure,
) -> UsageLimitError | ProviderUnavailableError:
    stream_interpretation = _stream_interpretation_for_service(service_name)
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
                ProviderUnavailableReason.SERVICE_NOT_AVAILABLE
                if failure.detail == _SERVICE_NOT_AVAILABLE_DETAIL
                else ProviderUnavailableReason.TRANSIENT_API_ERROR
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


def _opencode_provider_state_dir_relpath(
    *,
    role: Any,
    session_namespace: str,
) -> str:
    return cast(str, provider_state_relpath(role, "opencode", session_namespace))


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
            log_context=None,
            provider_session_id=provider_session_id,
            output_hooks=ProviderOutputReductionHooks(
                reduce_output=stream_interpretation.reduce_output,
                extract_provider_session_id=(
                    stream_interpretation.extract_provider_session_id
                ),
            ),
        )
    )


def _invoke_claude_session_provider(
    *,
    provider_invocation_adapter: ProviderInvocationAdapter,
    invocation_dir: Path,
    prompt: str,
    model: str,
    effort: str,
    tool_access: ToolAccess,
    auth: ProviderAuth | None,
    provider_state_dir: Path | None,
    run_kind: RunKind,
    provider_session_id: str,
    on_live_output: Callable[[AgentEvent], None] | None = None,
) -> ProviderInvocationResult | ProviderInvocationFailure:
    rendered = _builtin_provider_rendering_module.render_built_in_provider_invocation(
        _builtin_provider_rendering_module.BuiltInProviderRenderRequest(
            provider_selection=(
                _builtin_provider_rendering_module.BuiltInProviderSelectionFacts(
                    service="claude",
                    model=model,
                    effort=effort,
                )
            ),
            run_kind=run_kind,
            tool_access=tool_access,
            auth=auth,
            invocation_dir=invocation_dir,
            provider_state_dir=provider_state_dir,
            provider_session_id=provider_session_id,
        )
    )
    stream_interpretation = _with_observed_output(
        _claude_stream_interpretation(),
        on_live_output,
    )
    return _invoke_provider(
        provider_invocation_adapter=provider_invocation_adapter,
        command=rendered.legacy_command_text or "",
        command_argv=rendered.canonical_argv,
        prefer_argv=rendered.prefer_argv,
        worktree=invocation_dir,
        environment=dict(rendered.environment),
        prompt_content=prompt,
        prompt_path=rendered.prompt_path,
        cleanup_prompt_path=(
            rendered.prompt_cleanup_choice
            is _builtin_provider_rendering_module.PromptCleanupChoice.DELETE_AFTER_INVOCATION
        ),
        run_kind=run_kind,
        provider_session_id=provider_session_id,
        stream_interpretation=stream_interpretation,
    )


def _invoke_claude_new_session_provider(
    *,
    provider_invocation_adapter: ProviderInvocationAdapter,
    request: NewSessionRunRequest,
    stage: ProviderSelection,
    provider_state_dir: Path,
    run_kind: RunKind,
    provider_session_id: str,
    on_live_output: Callable[[AgentEvent], None] | None = None,
) -> ProviderInvocationResult | ProviderInvocationFailure:
    return _invoke_claude_session_provider(
        provider_invocation_adapter=provider_invocation_adapter,
        invocation_dir=request.invocation_dir,
        prompt=request.prompt,
        model=stage.model,
        effort=stage.effort,
        tool_access=request.tool_access,
        auth=_selection_auth(stage),
        provider_state_dir=provider_state_dir,
        run_kind=run_kind,
        provider_session_id=provider_session_id,
        on_live_output=on_live_output,
    )


def _invoke_codex_new_session_provider(
    *,
    provider_invocation_adapter: ProviderInvocationAdapter,
    request: NewSessionRunRequest,
    stage: ProviderSelection,
    provider_state_dir: Path,
    on_live_output: Callable[[AgentEvent], None] | None = None,
) -> ProviderInvocationResult | ProviderInvocationFailure:
    stream_interpretation = _with_observed_output(
        _codex_stream_interpretation(),
        on_live_output,
    )
    command_argv = _codex_command(
        model=stage.model,
        effort=stage.effort,
        tool_access=request.tool_access,
        run_kind=RunKind.FRESH,
        session_uuid=None,
    )
    return _invoke_provider(
        provider_invocation_adapter=provider_invocation_adapter,
        command="",
        command_argv=command_argv,
        prefer_argv=True,
        worktree=request.invocation_dir,
        environment=_codex_env(
            state_dir_container_path=str(provider_state_dir),
        ),
        prompt_content=request.prompt,
        prompt_path=_builtin_provider_temp_prompt_path(),
        cleanup_prompt_path=True,
        run_kind=RunKind.FRESH,
        provider_session_id=None,
        stream_interpretation=stream_interpretation,
    )


def _invoke_codex_resumed_session_provider(
    *,
    provider_invocation_adapter: ProviderInvocationAdapter,
    request: ResumedSessionRunRequest,
    provider_state_dir: Path | None,
    provider_session_id: str,
    on_live_output: Callable[[AgentEvent], None] | None = None,
) -> ProviderInvocationResult | ProviderInvocationFailure:
    stream_interpretation = _with_observed_output(
        _codex_stream_interpretation(),
        on_live_output,
    )
    command_argv = _codex_command(
        model=request.model,
        effort=request.effort,
        tool_access=request.tool_access,
        run_kind=RunKind.RESUME,
        session_uuid=provider_session_id,
    )
    return _invoke_provider(
        provider_invocation_adapter=provider_invocation_adapter,
        command="",
        command_argv=command_argv,
        prefer_argv=True,
        worktree=request.invocation_dir,
        environment=_codex_env(
            state_dir_container_path=(
                str(provider_state_dir) if provider_state_dir is not None else None
            ),
        ),
        prompt_content=request.prompt,
        prompt_path=_builtin_provider_temp_prompt_path(),
        cleanup_prompt_path=True,
        run_kind=RunKind.RESUME,
        provider_session_id=provider_session_id,
        stream_interpretation=stream_interpretation,
    )


def _invoke_opencode_new_session_provider(
    *,
    provider_invocation_adapter: ProviderInvocationAdapter,
    request: NewSessionRunRequest,
    stage: ProviderSelection,
    provider_state_dir: Path,
    run_kind: RunKind,
    provider_session_id: str,
    on_live_output: Callable[[AgentEvent], None] | None = None,
) -> ProviderInvocationResult | ProviderInvocationFailure:
    stream_interpretation = _opencode_stream_interpretation(
        on_live_output=on_live_output,
        fallback_provider_session_id=provider_session_id,
    )

    command_argv = _opencode_command(
        model=stage.model,
        effort=stage.effort,
        run_kind=run_kind,
        session_uuid=provider_session_id,
    )
    invocation_result = _invoke_provider(
        provider_invocation_adapter=provider_invocation_adapter,
        command="",
        command_argv=command_argv,
        prefer_argv=True,
        worktree=request.invocation_dir,
        environment=_opencode_env(
            auth=_selection_auth(stage),
            state_dir_container_path=str(provider_state_dir),
            tool_policy=request.tool_access.tool_policy,
        ),
        prompt_content=request.prompt,
        prompt_path=_builtin_provider_prompt_path(request.invocation_dir),
        cleanup_prompt_path=True,
        run_kind=run_kind,
        provider_session_id=provider_session_id,
        stream_interpretation=stream_interpretation,
    )
    return invocation_result


def _run_builtin_ephemeral(
    request: EphemeralRunRequest,
    *,
    provider_invocation_adapter: ProviderInvocationAdapter | None = None,
    select_builtin_stage: Callable[
        [ProviderSelection], ProviderSelection
    ] = _select_builtin_stage,
    validate_claude_stage: Callable[[ProviderSelection], None] = _validate_claude_stage,
    validate_codex_stage: Callable[[ProviderSelection], None] = _validate_codex_stage,
    validate_opencode_stage: Callable[
        [ProviderSelection], None
    ] = _validate_opencode_stage,
    claude_command: Callable[..., tuple[str, ...]] = _claude_command,
    claude_env: Callable[..., dict[str, str]] = _claude_env,
    reduce_claude_stream: Callable[
        [list[str], Callable[[AgentEvent], None] | None],
        tuple[str, ProviderUsage | None],
    ] = reduce_claude_stream,
    codex_command: Callable[..., tuple[str, ...]] = _codex_command,
    codex_env: Callable[..., dict[str, str]] = _codex_env,
    reduce_codex_stream: Callable[
        [list[str], Callable[[AgentEvent], None] | None],
        tuple[str, ProviderUsage | None],
    ] = reduce_codex_stream,
    opencode_command: Callable[..., tuple[str, ...]] = _opencode_command,
    opencode_env: Callable[..., dict[str, str]] = _opencode_env,
    reduce_opencode_stream: Callable[
        [list[str], Callable[[AgentEvent], None] | None],
        tuple[str, ProviderUsage | None],
    ] = reduce_opencode_stream,
    validate_codex_auth: Callable[[], None] = _validate_codex_auth,
) -> RunResult:
    invocation_adapter = (
        _default_provider_invocation_adapter()
        if provider_invocation_adapter is None
        else provider_invocation_adapter
    )

    wrapped_on_live_output, timeout_watchdog = _wrap_on_live_output_with_timeout(
        request.on_live_output,
        request.timeout_seconds,
    )

    try:
        selected_stage = select_builtin_stage(request.provider_selection)
        selected_stage_auth = _selection_auth(selected_stage)
        if selected_stage.service == "codex":
            validate_codex_stage(selected_stage)
            validate_codex_auth()
            prompt_path = _builtin_provider_temp_prompt_path()
        elif selected_stage.service == "opencode":
            validate_opencode_stage(selected_stage)
            if selected_stage_auth is None or not selected_stage_auth.opencode_api_key:
                message = "Missing OpenCode API key."
                raise AgentCredentialFailureError(
                    message=message,
                    service_name="opencode",
                    classification="operator_actionable_agent_credential_failure",
                )
            prompt_path = _builtin_provider_temp_prompt_path()
        else:
            validate_claude_stage(selected_stage)
            if (
                selected_stage_auth is None
                or not selected_stage_auth.claude_code_oauth_token
            ):
                raise AgentCredentialFailureError(
                    message="Missing Claude Code OAuth token.",
                    service_name="claude",
                )
            prompt_path = _builtin_provider_prompt_path(request.invocation_dir)
        if selected_stage.service == "codex":
            command_argv = codex_command(
                model=selected_stage.model,
                effort=selected_stage.effort,
                tool_access=request.tool_access,
            )
            invocation_result = _invoke_provider(
                provider_invocation_adapter=invocation_adapter,
                command="",
                command_argv=command_argv,
                prefer_argv=True,
                worktree=request.invocation_dir,
                environment=codex_env(),
                prompt_content=request.prompt,
                prompt_path=prompt_path,
                cleanup_prompt_path=True,
                run_kind=RunKind.FRESH,
                provider_session_id=None,
                stream_interpretation=_with_observed_output(
                    _with_reduce_output(
                        _codex_stream_interpretation(),
                        lambda lines: reduce_codex_stream(lines, None),
                    ),
                    wrapped_on_live_output,
                ),
            )
        elif selected_stage.service == "opencode":
            command_argv = opencode_command(
                model=selected_stage.model,
                effort=selected_stage.effort,
                run_kind=RunKind.FRESH,
                session_uuid=None,
            )
            invocation_result = _invoke_provider(
                provider_invocation_adapter=invocation_adapter,
                command="",
                command_argv=command_argv,
                prefer_argv=True,
                worktree=request.invocation_dir,
                environment=opencode_env(
                    auth=selected_stage_auth,
                    state_dir_container_path=str(request.invocation_dir),
                    tool_policy=request.tool_access.tool_policy,
                ),
                prompt_content=request.prompt,
                prompt_path=prompt_path,
                cleanup_prompt_path=True,
                run_kind=RunKind.FRESH,
                provider_session_id=None,
                stream_interpretation=_opencode_stream_interpretation(
                    on_live_output=wrapped_on_live_output,
                    reduce_output=lambda lines: reduce_opencode_stream(lines, None),
                ),
            )
        else:
            command_argv = claude_command(
                model=selected_stage.model,
                effort=selected_stage.effort,
                tool_access=request.tool_access,
                run_kind=RunKind.FRESH,
            )
            invocation_result = _invoke_provider(
                provider_invocation_adapter=invocation_adapter,
                command=_claude_legacy_command_text(
                    model=selected_stage.model,
                    effort=selected_stage.effort,
                    tool_access=request.tool_access,
                    prompt_path=prompt_path,
                    run_kind=RunKind.FRESH,
                ),
                command_argv=command_argv,
                prefer_argv=True,
                worktree=request.invocation_dir,
                environment=claude_env(auth=selected_stage_auth),
                prompt_content=request.prompt,
                prompt_path=prompt_path,
                cleanup_prompt_path=True,
                run_kind=RunKind.FRESH,
                provider_session_id=None,
                stream_interpretation=_with_observed_output(
                    _with_reduce_output(
                        _claude_stream_interpretation(),
                        lambda lines: reduce_claude_stream(lines, None),
                    ),
                    wrapped_on_live_output,
                ),
            )
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
    finally:
        if timeout_watchdog is not None:
            timeout_watchdog.stop_monitoring()


def _new_session_runtime_state_dir(
    request: NewSessionRunRequest,
    *,
    context: str,
) -> tuple[Path, Callable[[], None], bool]:
    runtime_state_dir = request._runtime_state_dir
    if runtime_state_dir is not None:
        return runtime_state_dir, lambda: None, True
    temp_dir = tempfile.TemporaryDirectory(prefix=f"{context}-provider-state-")
    return Path(temp_dir.name), temp_dir.cleanup, False


def _require_opencode_auth(auth: ProviderAuth | None) -> None:
    if auth is not None and auth.opencode_api_key:
        return
    message = "Missing OpenCode API key."
    raise AgentCredentialFailureError(
        message=message,
        service_name="opencode",
        classification="operator_actionable_agent_credential_failure",
    )


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
