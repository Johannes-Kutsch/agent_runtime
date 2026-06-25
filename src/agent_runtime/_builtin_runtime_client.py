from __future__ import annotations

import json
import logging
import os
import re
import shlex
import shutil
import tempfile
import subprocess as _subprocess
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, cast

from . import _time as _time_module
from ._builtin_provider_stream_interpretation import (
    BuiltInProviderStreamInterpretation,
    build_claude_agent_event as _live_output_event_for_claude_line,
    build_codex_agent_event as _live_output_event_for_codex_line,
    build_opencode_agent_event as _live_output_event_for_opencode_line,
    claude_built_in_provider_stream_interpretation,
    classify_built_in_provider_invocation_progress,
    codex_built_in_provider_stream_interpretation,
    emit_built_in_provider_live_output_event,
    extract_codex_provider_session_id as _extract_codex_provider_session_id,
    is_claude_subscription_access_denial,
    observe_opencode_output as _observe_output_opencode,
    opencode_built_in_provider_stream_interpretation,
    parse_claude_event_with_dependencies,
    parse_claude_reset_time,
    parse_opencode_reset_time,
    reduce_codex_stream as _reduce_codex_stream,
    reduce_claude_stream_with_dependencies,
    reduce_opencode_stream as _reduce_opencode_stream,
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
from ._portable_continuation_payload import (
    create_portable_continuation_payload,
    read_portable_continuation_payload,
)
from ._runtime_lifecycle import (
    Completed,
    Continuation,
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
_CLAUDE_VALID_MODELS = frozenset({"haiku", "sonnet", "opus"})
_CLAUDE_VALID_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})
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
_CLAUDE_RESET_PATTERN = re.compile(
    r"resets\s+"
    r"(?:(?P<month>[A-Za-z]+)\s+(?P<day>\d{1,2}),\s+)?"
    r"(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?(?P<ampm>am|pm)\s+\(UTC\)",
    re.IGNORECASE,
)
_CLAUDE_MONTHS = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sept": 9,
    "sep": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}

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
    if stage.model not in _CLAUDE_VALID_MODELS:
        raise RuntimeConfigurationError(f"Unsupported Claude model {stage.model!r}.")
    if stage.effort not in _CLAUDE_VALID_EFFORTS:
        raise RuntimeConfigurationError(f"Unsupported Claude effort {stage.effort!r}.")


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
    profile = _claude_tool_policy_profile(tool_access)
    flags = [
        "--verbose",
        "--dangerously-skip-permissions",
        "--output-format",
        "stream-json",
        "-p",
        "-",
        "--disable-slash-commands",
        "--exclude-dynamic-system-prompt-sections",
    ]
    if profile.allowed_tools is not None:
        flags.extend(["--tools", " ".join(profile.allowed_tools)])
    if profile.disallowed_tools:
        flags.extend(["--disallowedTools", " ".join(profile.disallowed_tools)])
    if profile.strict_mcp_config:
        flags.extend(
            [
                "--strict-mcp-config",
                "--mcp-config",
                '{"mcpServers":{}}',
            ]
        )
    if model:
        flags.extend(["--model", model])
    if effort:
        flags.extend(["--effort", effort])
    if session_uuid:
        if run_kind == RunKind.RESUME:
            flags.extend(["--resume", session_uuid])
        else:
            flags.extend(["--session-id", session_uuid])
    return ("claude", *flags)


def _claude_tool_policy_profile(tool_access: ToolAccess) -> ToolPolicyProfile:
    if isinstance(tool_access.tool_policy, ToolPolicy):
        if tool_access.tool_policy is ToolPolicy.NONE:
            return ToolPolicyProfile(disallowed_tools=("all",))
        if tool_access.tool_policy is ToolPolicy.INSPECT_ONLY:
            return ToolPolicyProfile(allowed_tools=("Read", "Glob"))
        if tool_access.tool_policy is ToolPolicy.NO_FILE_MUTATION:
            return ToolPolicyProfile(disallowed_tools=("Edit", "Write", "NotebookEdit"))
        return ToolPolicyProfile()
    return tool_access.tool_policy


def _claude_legacy_command_text(
    *,
    model: str,
    effort: str,
    tool_access: ToolAccess,
    prompt_path: Path,
    run_kind: RunKind = RunKind.FRESH,
    session_uuid: str | None = None,
) -> str:
    profile = _claude_tool_policy_profile(tool_access)
    flags = (
        "--verbose --dangerously-skip-permissions --output-format stream-json -p -"
        " --disable-slash-commands --exclude-dynamic-system-prompt-sections"
    )
    if profile.allowed_tools is not None:
        flags += f" --tools {shlex.quote(' '.join(profile.allowed_tools))}"
    if profile.disallowed_tools:
        flags += f' --disallowedTools "{" ".join(profile.disallowed_tools)}"'
    if profile.strict_mcp_config:
        flags += " --strict-mcp-config --mcp-config '{\"mcpServers\":{}}'"
    if model:
        flags += f" --model {model}"
    if effort:
        flags += f" --effort {effort}"
    if session_uuid:
        if run_kind == RunKind.RESUME:
            flags += f" --resume {shlex.quote(session_uuid)}"
        else:
            flags += f" --session-id {shlex.quote(session_uuid)}"
    return f"claude {flags} < {shlex.quote(str(prompt_path))}"


def _claude_env(
    *,
    auth: ProviderAuth | None,
    state_dir_container_path: str | None = None,
) -> dict[str, str]:
    env: dict[str, str] = {}
    token = None if auth is None else auth.claude_code_oauth_token
    if token:
        env["CLAUDE_CODE_OAUTH_TOKEN"] = token
    if state_dir_container_path:
        env["CLAUDE_CONFIG_DIR"] = state_dir_container_path
    return env


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


def _is_claude_subscription_access_denial(event: dict[str, Any]) -> bool:
    return is_claude_subscription_access_denial(event)


def _parse_claude_reset_time(retry_text: object) -> datetime | None:
    return parse_claude_reset_time(retry_text)


def _parse_opencode_reset_time(retry_text: object) -> datetime | None:
    return parse_opencode_reset_time(retry_text)


def _parse_claude_event_with_dependencies(
    line: str,
    *,
    parse_claude_reset_time: Callable[[object], datetime | None],
    is_claude_subscription_access_denial: Callable[[dict[str, Any]], bool],
) -> list[Any]:
    return parse_claude_event_with_dependencies(
        line,
        parse_claude_reset_time=parse_claude_reset_time,
        is_claude_subscription_access_denial=is_claude_subscription_access_denial,
    )


def _reduce_claude_stream_with_dependencies(
    lines: list[str],
    *,
    parse_claude_event: Callable[[str], list[Any]],
    on_live_output: Callable[[AgentEvent], None] | None = None,
) -> tuple[str, ProviderUsage | None]:
    return reduce_claude_stream_with_dependencies(
        lines,
        parse_claude_event=parse_claude_event,
        on_live_output=on_live_output,
    )


def _reduce_claude_stream_with_runtime_overrides(
    lines: list[str],
    on_live_output: Callable[[AgentEvent], None] | None = None,
) -> tuple[str, ProviderUsage | None]:
    return _reduce_claude_stream_with_dependencies(
        lines,
        parse_claude_event=lambda line: _parse_claude_event_with_dependencies(
            line,
            parse_claude_reset_time=_parse_claude_reset_time,
            is_claude_subscription_access_denial=(
                _is_claude_subscription_access_denial
            ),
        ),
        on_live_output=on_live_output,
    )


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


def _raw_event_payload(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _render_tool_call_display_message(tool_name: str, payload: str) -> str:
    if payload:
        return f"{tool_name}({payload})"
    return tool_name


def _message_event(line: str, text: str) -> AgentEvent:
    return AgentEvent(
        type="agent_message",
        display_message=text,
        raw_provider_output=line,
    )


def _tool_call_event(line: str, tool_name: str, payload: str) -> AgentEvent:
    return AgentEvent(
        type="agent_tool_call",
        display_message=_render_tool_call_display_message(tool_name, payload),
        raw_provider_output=line,
    )


def _other_event(line: str, descriptor: str) -> AgentEvent:
    return AgentEvent(
        type="other",
        display_message=descriptor,
        raw_provider_output=line,
    )


def _live_output_event_for_provider_line(service_name: str, line: str) -> AgentEvent:
    if service_name == "claude":
        return _live_output_event_for_claude_line(line)
    if service_name == "codex":
        return _live_output_event_for_codex_line(line)
    if service_name == "opencode":
        return _live_output_event_for_opencode_line(line)
    return AgentEvent(
        type="other",
        display_message="other",
        raw_provider_output=line,
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


def _with_observed_opencode_output(
    stream_interpretation: BuiltInProviderStreamInterpretation,
    on_live_output: Callable[[AgentEvent], None] | None,
) -> BuiltInProviderStreamInterpretation:
    if on_live_output is None:
        return stream_interpretation
    return BuiltInProviderStreamInterpretation(
        reduce_output=_ObservedOutputReducer(
            reduce_output=stream_interpretation.reduce_output,
            consume_stdout_lines=_observe_output_opencode(
                stream_interpretation=stream_interpretation,
                on_live_output=on_live_output,
            ),
        ),
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
    reduce_output: Callable[[list[str]], tuple[str, ProviderUsage | None]]
    | None = None,
    extract_provider_session_id: Callable[[list[str]], str | None] | None = None,
) -> BuiltInProviderStreamInterpretation:
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


def _provider_session_id_from_stdout_lines(
    stream_interpretation: BuiltInProviderStreamInterpretation,
    stdout_lines: tuple[str, ...],
) -> str | None:
    if stream_interpretation.extract_provider_session_id is None:
        return None
    return stream_interpretation.extract_provider_session_id(list(stdout_lines))


def _provider_invocation_failure_started(
    stream_interpretation: BuiltInProviderStreamInterpretation,
    failure: ProviderInvocationFailure,
) -> bool:
    return (
        classify_built_in_provider_invocation_progress(
            stream_interpretation,
            list(failure.stdout_lines),
            provider_session_id=failure.provider_session_id,
        )
        is InvocationProgress.STARTED
    )


def _provider_invocation_error_from_failure(
    service_name: str,
    failure: ProviderInvocationFailure,
) -> UsageLimitError | ProviderUnavailableError:
    stream_interpretation = _stream_interpretation_for_service(service_name)
    invocation_progress = (
        InvocationProgress.STARTED
        if _provider_invocation_failure_started(stream_interpretation, failure)
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


def _provider_session_id_from_failure(
    service_name: str,
    failure: ProviderInvocationFailure,
    *,
    fallback_provider_session_id: str | None = None,
) -> str | None:
    stream_interpretation = _stream_interpretation_for_service(service_name)
    fallback_session_id = _provider_session_id_from_stdout_lines(
        stream_interpretation,
        failure.stdout_lines,
    )
    if fallback_session_id is not None:
        return fallback_session_id
    if failure.provider_session_id is not None:
        return failure.provider_session_id
    return fallback_provider_session_id


def _select_builtin_stage(stage: ProviderSelection) -> ProviderSelection:
    candidate = supported_builtin_provider_selection(stage)
    if candidate is not None:
        return candidate
    raise RuntimeConfigurationError(
        "RuntimeClient requires at least one supported built-in service candidate."
    )


def _new_provider_session_id() -> str:
    return str(uuid.uuid4())


def _codex_provider_state_dir_relpath(
    *,
    role: Any,
    session_namespace: str,
) -> str:
    return cast(str, provider_state_relpath(role, "codex", session_namespace))


def _codex_is_resumable(state_dir: Path) -> bool:
    sessions_dir = state_dir / "sessions"
    if not sessions_dir.is_dir():
        return False
    return any(sessions_dir.rglob("rollout-*.jsonl"))


def _codex_prepare_runtime_state(
    runtime_state_dir: Path,
    *,
    role: Any,
    session_namespace: str,
) -> tuple[str, Path]:
    provider_state_dir_relpath = _codex_provider_state_dir_relpath(
        role=role,
        session_namespace=session_namespace,
    )
    provider_state_dir = runtime_state_dir / provider_state_dir_relpath
    provider_state_dir.mkdir(parents=True, exist_ok=True)
    return provider_state_dir_relpath, provider_state_dir


def _read_codex_rollout_thread_ids(rollout_path: Path) -> set[str]:
    thread_ids: set[str] = set()
    if not rollout_path.is_file():
        return thread_ids
    try:
        for line in rollout_path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict) or event.get("type") != "thread.started":
                continue
            thread_id = event.get("thread_id")
            if isinstance(thread_id, str):
                stripped = thread_id.strip()
                if stripped:
                    thread_ids.add(stripped)
    except (OSError, UnicodeDecodeError):
        return set()
    return thread_ids


def _recover_codex_rollout_thread_id(state_dir: Path | None) -> str | None:
    if state_dir is None:
        return None
    sessions_dir = state_dir / "sessions"
    if not sessions_dir.is_dir():
        return None
    thread_ids: set[str] = set()
    for rollout_path in sessions_dir.rglob("rollout-*.jsonl"):
        thread_ids.update(_read_codex_rollout_thread_ids(rollout_path))
        if len(thread_ids) > 1:
            return None
    if len(thread_ids) != 1:
        return None
    return next(iter(thread_ids))


def _resolve_recoverable_codex_session_id(
    *,
    provider_state_dir: Path,
    provider_session_id: str | None,
) -> str:
    recovered_thread_id = _recover_codex_rollout_thread_id(provider_state_dir)
    if not _codex_is_resumable(provider_state_dir) or recovered_thread_id is None:
        raise RuntimeConfigurationError(
            "Codex continuation is not recoverable from provider state."
        )
    if provider_session_id:
        return provider_session_id
    return recovered_thread_id


def _codex_seed_auth(provider_state_dir: Path) -> None:
    provider_auth_path = provider_state_dir / "auth.json"
    if provider_auth_path.exists():
        return
    host_auth_path = _codex_host_auth_path()
    if not host_auth_path.exists():
        raise _missing_codex_auth_error()
    shutil.copyfile(host_auth_path, provider_auth_path)


def _build_codex_continuation(
    *,
    model: str,
    effort: str,
    tool_access: ToolAccess,
    provider_session_id: str,
    provider_state_dir_relpath: str | None = None,
) -> Continuation:
    provider_resume_state: dict[str, Any] = {
        "run_kind": RunKind.RESUME.value,
        "provider_session_id": provider_session_id,
        "exact_transcript_match": False,
    }
    if provider_state_dir_relpath is not None:
        provider_resume_state["provider_state_dir_relpath"] = provider_state_dir_relpath
    return create_portable_continuation_payload(
        service_name="codex",
        model=model,
        effort=effort,
        tool_access=tool_access,
        provider_resume_state=provider_resume_state,
    ).to_continuation()


def _claude_provider_state_dir_relpath(
    *,
    role: Any,
    session_namespace: str,
) -> str:
    return cast(str, provider_state_relpath(role, "claude", session_namespace))


def _opencode_provider_state_dir_relpath(
    *,
    role: Any,
    session_namespace: str,
) -> str:
    return cast(str, provider_state_relpath(role, "opencode", session_namespace))


def _opencode_prepare_runtime_state(
    runtime_state_dir: Path,
    *,
    role: Any,
    session_namespace: str,
) -> tuple[str, Path]:
    provider_state_dir_relpath = _opencode_provider_state_dir_relpath(
        role=role,
        session_namespace=session_namespace,
    )
    provider_state_dir = runtime_state_dir / provider_state_dir_relpath
    provider_state_dir.mkdir(parents=True, exist_ok=True)
    return provider_state_dir_relpath, provider_state_dir


def _load_opencode_state_dir_session_id(state_dir: Path | None) -> str | None:
    if state_dir is None:
        return None
    path = state_dir / _OPENCODE_SESSION_ID_FILENAME
    if not path.is_file():
        return None
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return value or None


def _opencode_is_resumable(state_dir: Path) -> bool:
    return (state_dir / "resume.jsonl").is_file() or (
        state_dir / _OPENCODE_SESSION_ID_FILENAME
    ).is_file()


def _opencode_provider_state_from_runtime_dir(state_dir: Path | None) -> dict[str, Any]:
    if state_dir is None:
        return {}
    provider_state: dict[str, Any] = {}
    session_id = _load_opencode_state_dir_session_id(state_dir)
    if session_id is not None:
        provider_state["session_id"] = session_id
    resume_jsonl_path = state_dir / "resume.jsonl"
    if resume_jsonl_path.is_file():
        try:
            provider_state["resume_jsonl"] = resume_jsonl_path.read_text(
                encoding="utf-8"
            )
        except OSError:
            pass
    return provider_state


def _seed_opencode_provider_state_dir(
    state_dir: Path,
    provider_state: dict[str, Any] | None,
) -> None:
    for state_filename in (_OPENCODE_SESSION_ID_FILENAME, "resume.jsonl"):
        state_path = state_dir / state_filename
        if state_path.exists():
            state_path.unlink()
    if not isinstance(provider_state, dict):
        return
    provider_session_id = provider_state.get("session_id")
    if isinstance(provider_session_id, str) and provider_session_id:
        _persist_opencode_session_id(state_dir, provider_session_id)
    resume_jsonl = provider_state.get("resume_jsonl")
    if isinstance(resume_jsonl, str) and resume_jsonl:
        (state_dir / "resume.jsonl").write_text(
            resume_jsonl,
            encoding="utf-8",
        )


def _restore_opencode_state_dir(
    request: ResumedSessionRunRequest,
    continuation_provider_state: dict[str, Any] | None,
) -> tuple[Path, Callable[[], None]]:
    if request._runtime_state_dir is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="opencode-provider-state-")

        def cleanup() -> None:
            temp_dir.cleanup()

        state_dir = Path(temp_dir.name)
        _seed_opencode_provider_state_dir(state_dir, continuation_provider_state)
        return state_dir, cleanup
    provider_state_dir_relpath = _opencode_provider_state_dir_relpath(
        role="implementer",
        session_namespace=request._session_namespace,
    )
    state_dir = request._runtime_state_dir / provider_state_dir_relpath
    state_dir.mkdir(parents=True, exist_ok=True)
    _seed_opencode_provider_state_dir(state_dir, continuation_provider_state)
    return state_dir, lambda: None


def _persist_opencode_session_id(state_dir: Path, provider_session_id: str) -> None:
    (state_dir / _OPENCODE_SESSION_ID_FILENAME).write_text(
        f"{provider_session_id}\n",
        encoding="utf-8",
    )


def _opencode_exact_transcript_match(
    *,
    saved_exact_transcript_match: bool,
    provider_session_id: str | None,
    state_dir_session_id: str | None,
) -> bool:
    return (
        saved_exact_transcript_match
        and provider_session_id is not None
        and state_dir_session_id == provider_session_id
    )


def _claude_is_resumable(state_dir: Path) -> bool:
    return state_dir.is_dir() and any(path.is_file() for path in state_dir.rglob("*"))


def _claude_prepare_runtime_state(
    runtime_state_dir: Path,
    *,
    role: Any,
    session_namespace: str,
) -> tuple[str, Path]:
    provider_state_dir_relpath = _claude_provider_state_dir_relpath(
        role=role,
        session_namespace=session_namespace,
    )
    provider_state_dir = runtime_state_dir / provider_state_dir_relpath
    provider_state_dir.mkdir(parents=True, exist_ok=True)
    return provider_state_dir_relpath, provider_state_dir


def _claude_run_kind_for_state_dir(state_dir: Path) -> RunKind:
    if _claude_is_resumable(state_dir):
        return RunKind.RESUME
    return RunKind.FRESH


def _build_claude_continuation(
    *,
    model: str,
    effort: str,
    tool_access: ToolAccess,
    provider_session_id: str,
) -> Continuation:
    return create_portable_continuation_payload(
        service_name="claude",
        model=model,
        effort=effort,
        tool_access=tool_access,
        provider_resume_state={
            "run_kind": RunKind.RESUME.value,
            "provider_session_id": provider_session_id,
            "exact_transcript_match": False,
        },
    ).to_continuation()


def _build_opencode_continuation(
    *,
    model: str,
    effort: str,
    tool_access: ToolAccess,
    provider_session_id: str,
    provider_state: dict[str, Any] | None = None,
    provider_state_dir: Path | None = None,
    exact_transcript_match: bool | None = None,
) -> Continuation:
    if provider_state is None and provider_state_dir is not None:
        provider_state = _opencode_provider_state_from_runtime_dir(provider_state_dir)
    if provider_state is None:
        provider_state = {}
    provider_resume_state: dict[str, Any] = {
        "provider_session_id": provider_session_id,
        "provider_state": provider_state,
    }
    if exact_transcript_match is not None:
        provider_resume_state["exact_transcript_match"] = exact_transcript_match
    return create_portable_continuation_payload(
        service_name="opencode",
        model=model,
        effort=effort,
        tool_access=tool_access,
        provider_resume_state=provider_resume_state,
    ).to_continuation()


def _completed_outcome(
    *,
    output: str,
    usage: ProviderUsage | None,
    continuation: Continuation | None,
    service: str,
    model: str,
    effort: str,
) -> RuntimeOutcome:
    return RuntimeOutcome(
        kind=Completed(),
        result=RunResult(
            output=output,
            usage=usage,
            continuation=continuation,
            selected=ResolvedProvider(service=service, model=model, effort=effort),
        ),
    )


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
    stream_interpretation = _with_observed_output(
        _claude_stream_interpretation(),
        on_live_output,
    )
    return _invoke_provider(
        provider_invocation_adapter=provider_invocation_adapter,
        command=_claude_legacy_command_text(
            model=stage.model,
            effort=stage.effort,
            tool_access=request.tool_access,
            prompt_path=_builtin_provider_prompt_path(request.invocation_dir),
            run_kind=run_kind,
            session_uuid=provider_session_id,
        ),
        command_argv=_claude_command(
            model=stage.model,
            effort=stage.effort,
            tool_access=request.tool_access,
            run_kind=run_kind,
            session_uuid=provider_session_id,
        ),
        prefer_argv=True,
        worktree=request.invocation_dir,
        environment=_claude_env(
            auth=_selection_auth(stage),
            state_dir_container_path=str(provider_state_dir),
        ),
        prompt_content=request.prompt,
        prompt_path=_builtin_provider_prompt_path(request.invocation_dir),
        cleanup_prompt_path=True,
        run_kind=run_kind,
        provider_session_id=provider_session_id,
        stream_interpretation=stream_interpretation,
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


def _active_codex_provider_session_id_from_result(
    invocation_result: ProviderInvocationResult,
    *,
    fallback_provider_session_id: str | None,
) -> str | None:
    return (
        _extract_codex_provider_session_id(list(invocation_result.stdout_lines))
        or invocation_result.provider_session_id
        or fallback_provider_session_id
    )


def _active_codex_provider_session_id_from_failure(
    failure: ProviderInvocationFailure,
    *,
    fallback_provider_session_id: str | None,
) -> str | None:
    return (
        _extract_codex_provider_session_id(list(failure.stdout_lines))
        or failure.provider_session_id
        or fallback_provider_session_id
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
) -> tuple[ProviderInvocationResult | ProviderInvocationFailure, str]:
    observed_provider_session_id = provider_session_id

    def _record_opencode_session_id(session_id: str) -> None:
        nonlocal observed_provider_session_id
        observed_provider_session_id = session_id

    def _reduce_opencode_session_output(
        lines: list[str],
    ) -> tuple[str, ProviderUsage | None]:
        return _reduce_opencode_stream(
            lines,
            None,
            on_provider_session_id=_record_opencode_session_id,
        )

    stream_interpretation = _with_observed_opencode_output(
        _opencode_stream_interpretation(
            reduce_output=_reduce_opencode_session_output,
            extract_provider_session_id=lambda _lines: observed_provider_session_id,
        ),
        on_live_output,
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
    return invocation_result, (
        invocation_result.provider_session_id
        or observed_provider_session_id
        or provider_session_id
    )


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
    ] = _reduce_claude_stream_with_runtime_overrides,
    codex_command: Callable[..., tuple[str, ...]] = _codex_command,
    codex_env: Callable[..., dict[str, str]] = _codex_env,
    reduce_codex_stream: Callable[
        [list[str], Callable[[AgentEvent], None] | None],
        tuple[str, ProviderUsage | None],
    ] = _reduce_codex_stream,
    opencode_command: Callable[..., tuple[str, ...]] = _opencode_command,
    opencode_env: Callable[..., dict[str, str]] = _opencode_env,
    reduce_opencode_stream: Callable[
        [list[str], Callable[[AgentEvent], None] | None],
        tuple[str, ProviderUsage | None],
    ] = _reduce_opencode_stream,
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
                    BuiltInProviderStreamInterpretation(
                        reduce_output=lambda lines: reduce_codex_stream(lines, None),
                        build_agent_event=_live_output_event_for_codex_line,
                        classify_invocation_progress=(
                            _codex_stream_interpretation().classify_invocation_progress
                        ),
                        extract_provider_session_id=_extract_codex_provider_session_id,
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
                stream_interpretation=_with_observed_opencode_output(
                    _opencode_stream_interpretation(
                        reduce_output=lambda lines: reduce_opencode_stream(lines, None),
                    ),
                    wrapped_on_live_output,
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


def _require_claude_auth(auth: ProviderAuth | None) -> None:
    if auth is not None and auth.claude_code_oauth_token:
        return
    raise AgentCredentialFailureError(
        message="Missing Claude Code OAuth token.",
        service_name="claude",
    )


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
    invocation_adapter = (
        _default_provider_invocation_adapter()
        if provider_invocation_adapter is None
        else provider_invocation_adapter
    )
    runtime_state_dir, cleanup_runtime_state_dir, is_caller_managed_runtime_state = (
        _new_session_runtime_state_dir(
            request,
            context="new-session",
        )
    )
    _on_live_output, timeout_watchdog = _wrap_on_live_output_with_timeout(
        request.on_live_output,
        request.timeout_seconds,
    )
    try:
        if supported_builtin_provider_selection(request.provider_selection) is None:
            raise RuntimeConfigurationError(
                "RuntimeClient requires at least one supported built-in service candidate."
            )
        selected_stage = _select_builtin_stage(request.provider_selection)
        selected_stage_auth = _selection_auth(selected_stage)
        _require_portable_continuation_support(selected_stage.service)

        def _portable_codex_state_dir_relpath(
            provider_state_dir_relpath: str | None,
        ) -> str | None:
            if is_caller_managed_runtime_state:
                return provider_state_dir_relpath
            return None

        if selected_stage.service == "codex":
            _validate_codex_stage(selected_stage)
            provider_state_dir_relpath, provider_state_dir = (
                _codex_prepare_runtime_state(
                    runtime_state_dir,
                    role="implementer",
                    session_namespace=request._session_namespace,
                )
            )
            _codex_seed_auth(provider_state_dir)
            recovered_thread_id = _recover_codex_rollout_thread_id(provider_state_dir)
            if (
                _codex_is_resumable(provider_state_dir)
                and recovered_thread_id is not None
            ):
                return _run_builtin_resumed_session(
                    ResumedSessionRunRequest(
                        prompt=request.prompt,
                        invocation_dir=cast(Any, request.invocation_dir),
                        _runtime_state_dir=runtime_state_dir,
                        continuation=_build_codex_continuation(
                            model=selected_stage.model,
                            effort=selected_stage.effort,
                            tool_access=request.tool_access,
                            provider_session_id=recovered_thread_id,
                            provider_state_dir_relpath=_portable_codex_state_dir_relpath(
                                provider_state_dir_relpath
                            ),
                        ),
                        provider_auth=selected_stage_auth,
                        on_live_output=_on_live_output,
                        timeout_seconds=0,
                        _session_namespace=request._session_namespace,
                    ),
                    provider_invocation_adapter=invocation_adapter,
                )
            provider_session_id: str | None = None
            invocation_result = _invoke_codex_new_session_provider(
                provider_invocation_adapter=invocation_adapter,
                request=request,
                stage=selected_stage,
                provider_state_dir=provider_state_dir,
                on_live_output=_on_live_output,
            )
            if isinstance(invocation_result, ProviderInvocationFailure):
                provider_session_id = _active_codex_provider_session_id_from_failure(
                    invocation_result,
                    fallback_provider_session_id=provider_session_id,
                )
                failure_error = _provider_invocation_error_from_failure(
                    "codex",
                    invocation_result,
                )
                if provider_session_id is not None:
                    failure_error.invocation_progress = InvocationProgress.STARTED
                failure_error.continuation = (
                    _build_codex_continuation(
                        model=selected_stage.model,
                        effort=selected_stage.effort,
                        tool_access=request.tool_access,
                        provider_session_id=provider_session_id,
                        provider_state_dir_relpath=_portable_codex_state_dir_relpath(
                            provider_state_dir_relpath
                        ),
                    )
                    if failure_error.invocation_progress is InvocationProgress.STARTED
                    and provider_session_id is not None
                    else None
                )
                raise failure_error
            else:
                provider_session_id = _active_codex_provider_session_id_from_result(
                    invocation_result,
                    fallback_provider_session_id=provider_session_id,
                )
                result_text = invocation_result.output
                usage = invocation_result.usage
            return _completed_outcome(
                output=result_text,
                usage=usage,
                continuation=(
                    _build_codex_continuation(
                        model=selected_stage.model,
                        effort=selected_stage.effort,
                        tool_access=request.tool_access,
                        provider_session_id=provider_session_id,
                        provider_state_dir_relpath=(
                            _portable_codex_state_dir_relpath(
                                provider_state_dir_relpath
                            )
                        ),
                    )
                    if provider_session_id is not None
                    else None
                ),
                service="codex",
                model=selected_stage.model,
                effort=selected_stage.effort,
            )
        elif selected_stage.service == "claude":
            provider_state_dir_relpath, provider_state_dir = (
                _claude_prepare_runtime_state(
                    runtime_state_dir,
                    role="implementer",
                    session_namespace=request._session_namespace,
                )
            )
            if _claude_is_resumable(provider_state_dir):
                return _run_builtin_resumed_session(
                    ResumedSessionRunRequest(
                        prompt=request.prompt,
                        invocation_dir=cast(Any, request.invocation_dir),
                        _runtime_state_dir=runtime_state_dir,
                        on_live_output=_on_live_output,
                        timeout_seconds=0,
                        continuation=_build_claude_continuation(
                            model=selected_stage.model,
                            effort=selected_stage.effort,
                            tool_access=request.tool_access,
                            provider_session_id=_new_provider_session_id(),
                        ),
                        provider_auth=selected_stage_auth,
                        _session_namespace=request._session_namespace,
                    ),
                    provider_invocation_adapter=invocation_adapter,
                )
            _validate_claude_stage(selected_stage)
            _require_claude_auth(selected_stage_auth)
            provider_session_id = _new_provider_session_id()
            run_kind = RunKind.FRESH
            exact_transcript_match = False
        elif selected_stage.service == "opencode":
            provider_state_dir_relpath, provider_state_dir = (
                _opencode_prepare_runtime_state(
                    runtime_state_dir,
                    role="implementer",
                    session_namespace=request._session_namespace,
                )
            )
            _validate_opencode_stage(selected_stage)
            _require_opencode_auth(selected_stage_auth)
            recovered_state_dir_session_id = _load_opencode_state_dir_session_id(
                provider_state_dir
            )
            if (
                _opencode_is_resumable(provider_state_dir)
                and recovered_state_dir_session_id
            ):
                provider_session_id = recovered_state_dir_session_id
                run_kind = RunKind.RESUME
                exact_transcript_match = True
            else:
                provider_session_id = _new_provider_session_id()
                run_kind = RunKind.FRESH
                exact_transcript_match = False
        else:
            raise RuntimeConfigurationError(
                "RuntimeClient session-backed execution is only implemented for Claude, Codex, and OpenCode."
            )
        if selected_stage.service == "claude":
            assert provider_session_id is not None
            invocation_result = _invoke_claude_new_session_provider(
                provider_invocation_adapter=invocation_adapter,
                request=request,
                stage=selected_stage,
                provider_state_dir=provider_state_dir,
                run_kind=run_kind,
                provider_session_id=provider_session_id,
                on_live_output=_on_live_output,
            )
        else:
            assert provider_session_id is not None
            invocation_result, provider_session_id = (
                _invoke_opencode_new_session_provider(
                    provider_invocation_adapter=invocation_adapter,
                    request=request,
                    stage=selected_stage,
                    provider_state_dir=provider_state_dir,
                    run_kind=run_kind,
                    provider_session_id=provider_session_id,
                    on_live_output=_on_live_output,
                )
            )
            _persist_opencode_session_id(provider_state_dir, provider_session_id)
        if isinstance(invocation_result, ProviderInvocationFailure):
            provider_session_id = _provider_session_id_from_failure(
                selected_stage.service,
                invocation_result,
                fallback_provider_session_id=provider_session_id,
            )
            failure_error = _provider_invocation_error_from_failure(
                selected_stage.service,
                invocation_result,
            )
            failure_error.continuation = None
            if (
                failure_error.invocation_progress is InvocationProgress.STARTED
                and provider_session_id is not None
            ):
                failure_error.continuation = (
                    _build_claude_continuation(
                        model=selected_stage.model,
                        effort=selected_stage.effort,
                        tool_access=request.tool_access,
                        provider_session_id=provider_session_id,
                    )
                    if selected_stage.service == "claude"
                    else _build_opencode_continuation(
                        model=selected_stage.model,
                        effort=selected_stage.effort,
                        tool_access=request.tool_access,
                        provider_session_id=provider_session_id,
                        provider_state_dir=provider_state_dir,
                        exact_transcript_match=exact_transcript_match,
                    )
                )
            raise failure_error
        if selected_stage.service == "claude":
            provider_session_id = (
                invocation_result.provider_session_id or provider_session_id
            )
        assert provider_session_id is not None
        result_text = invocation_result.output
        usage = invocation_result.usage
        return _completed_outcome(
            output=result_text,
            usage=usage,
            continuation=(
                _build_claude_continuation(
                    model=selected_stage.model,
                    effort=selected_stage.effort,
                    tool_access=request.tool_access,
                    provider_session_id=provider_session_id,
                )
                if selected_stage.service == "claude"
                else _build_opencode_continuation(
                    model=selected_stage.model,
                    effort=selected_stage.effort,
                    tool_access=request.tool_access,
                    provider_session_id=provider_session_id,
                    provider_state_dir=provider_state_dir,
                    exact_transcript_match=exact_transcript_match,
                )
            ),
            service=selected_stage.service,
            model=selected_stage.model,
            effort=selected_stage.effort,
        )
    finally:
        cleanup_runtime_state_dir()
        if timeout_watchdog is not None:
            timeout_watchdog.stop_monitoring()


def _run_builtin_resumed_session(
    request: ResumedSessionRunRequest,
    *,
    provider_invocation_adapter: ProviderInvocationAdapter | None = None,
) -> RuntimeOutcome:
    invocation_adapter = (
        _default_provider_invocation_adapter()
        if provider_invocation_adapter is None
        else provider_invocation_adapter
    )
    _on_live_output, timeout_watchdog = _wrap_on_live_output_with_timeout(
        request.on_live_output,
        request.timeout_seconds,
    )
    runtime_state_dir = request._runtime_state_dir
    continuation = request.continuation
    if continuation is None:
        raise RuntimeConfigurationError(
            "RuntimeClient resumed-session execution requires a continuation."
        )
    try:
        continuation_payload = read_portable_continuation_payload(continuation)
    except TypeError as exc:
        raise RuntimeConfigurationError(str(exc)) from exc
    continuation_service = continuation_payload.service_name
    _require_portable_continuation_support(continuation_service)
    provider_resume_state = continuation_payload.provider_resume_state
    provider_session_id: str | None
    provider_state_dir_relpath: str | None = None
    provider_state_dir: Path | None = None

    def _no_cleanup() -> None:
        return None

    if continuation_service == "codex":
        _validate_codex_stage(
            ProviderSelection(
                service="codex",
                model=request.model,
                effort=request.effort,
            )
        )
        provider_state_dir_relpath = cast(
            str | None,
            provider_resume_state.get("provider_state_dir_relpath"),
        )
        provider_session_id = cast(
            str | None,
            provider_resume_state.get("provider_session_id"),
        )
        if provider_session_id is not None:
            provider_session_id = provider_session_id.strip() or None
        if runtime_state_dir is not None and provider_state_dir_relpath:
            provider_state_dir = runtime_state_dir / provider_state_dir_relpath
            provider_state_dir.mkdir(parents=True, exist_ok=True)
            _codex_seed_auth(provider_state_dir)
            provider_session_id = _resolve_recoverable_codex_session_id(
                provider_state_dir=provider_state_dir,
                provider_session_id=provider_session_id,
            )
        elif provider_session_id is None:
            raise RuntimeConfigurationError(
                "Codex continuation is missing `provider_session_id`."
            )
        run_kind = RunKind.RESUME
        active_provider_session_id: str | None = provider_session_id
        invocation_result = _invoke_codex_resumed_session_provider(
            provider_invocation_adapter=invocation_adapter,
            provider_session_id=provider_session_id,
            request=request,
            provider_state_dir=provider_state_dir,
            on_live_output=_on_live_output,
        )
        if isinstance(invocation_result, ProviderInvocationFailure):
            active_provider_session_id = _active_codex_provider_session_id_from_failure(
                invocation_result,
                fallback_provider_session_id=active_provider_session_id,
            )
            failure_error = _provider_invocation_error_from_failure(
                "codex",
                invocation_result,
            )
            if active_provider_session_id is not None:
                failure_error.invocation_progress = InvocationProgress.STARTED
            failure_error.continuation = (
                _build_codex_continuation(
                    model=request.model,
                    effort=request.effort,
                    tool_access=request.tool_access,
                    provider_session_id=active_provider_session_id,
                    provider_state_dir_relpath=provider_state_dir_relpath,
                )
                if failure_error.invocation_progress is InvocationProgress.STARTED
                and active_provider_session_id is not None
                else None
            )
            raise failure_error
        else:
            active_provider_session_id = _active_codex_provider_session_id_from_result(
                invocation_result,
                fallback_provider_session_id=provider_session_id,
            )
            result_text = invocation_result.output
            usage = invocation_result.usage
        return _completed_outcome(
            output=result_text,
            usage=usage,
            continuation=(
                _build_codex_continuation(
                    model=request.model,
                    effort=request.effort,
                    tool_access=request.tool_access,
                    provider_session_id=active_provider_session_id,
                    provider_state_dir_relpath=provider_state_dir_relpath,
                )
                if active_provider_session_id is not None
                else None
            ),
            service="codex",
            model=request.model,
            effort=request.effort,
        )
    if continuation_service not in {"claude", "opencode"}:
        raise RuntimeConfigurationError(
            "RuntimeClient session-backed execution is only implemented for Claude, Codex, and OpenCode."
        )
    provider_session_id = cast(
        str | None,
        provider_resume_state.get("provider_session_id"),
    )
    if continuation_service == "claude":
        _require_claude_auth(request.provider_auth)
        provider_state_dir_relpath = cast(
            str | None,
            provider_resume_state.get("provider_state_dir_relpath"),
        )
        if provider_state_dir_relpath and request._runtime_state_dir is not None:
            provider_state_dir = request._runtime_state_dir / provider_state_dir_relpath
            provider_state_dir.mkdir(parents=True, exist_ok=True)
        state_dir_session_id = None
        run_kind = (
            _claude_run_kind_for_state_dir(provider_state_dir)
            if provider_state_dir is not None
            else RunKind.RESUME
        )
        exact_transcript_match = False
        if not provider_session_id:
            provider_session_id = _new_provider_session_id()
        cleanup_opencode_state_dir = _no_cleanup
    else:
        _require_opencode_auth(request.provider_auth)
        continuation_provider_state = provider_resume_state.get("provider_state")
        if not isinstance(continuation_provider_state, dict):
            continuation_provider_state = None
        provider_state_dir, cleanup_opencode_state_dir = _restore_opencode_state_dir(
            request=request,
            continuation_provider_state=continuation_provider_state,
        )
        state_dir_session_id = _load_opencode_state_dir_session_id(provider_state_dir)
        saved_exact_transcript_match = bool(
            provider_resume_state.get("exact_transcript_match", False)
        )
        if provider_session_id is None:
            provider_session_id = state_dir_session_id
        if provider_session_id is None:
            provider_session_id = _new_provider_session_id()
        exact_transcript_match = _opencode_exact_transcript_match(
            saved_exact_transcript_match=saved_exact_transcript_match,
            provider_session_id=provider_session_id,
            state_dir_session_id=state_dir_session_id,
        )
        run_kind = RunKind.RESUME
    prompt_path = _builtin_provider_prompt_path(request.invocation_dir)

    def _reduce_opencode_session_output(
        lines: list[str],
    ) -> tuple[str, ProviderUsage | None]:
        return _reduce_opencode_stream(lines)

    if continuation_service == "claude":
        command_argv = _claude_command(
            model=request.model,
            effort=request.effort,
            tool_access=request.tool_access,
            run_kind=run_kind,
            session_uuid=provider_session_id,
        )
        command = _claude_legacy_command_text(
            model=request.model,
            effort=request.effort,
            tool_access=request.tool_access,
            prompt_path=prompt_path,
            run_kind=run_kind,
            session_uuid=provider_session_id,
        )
        environment = _claude_env(
            auth=request.provider_auth,
            state_dir_container_path=(
                str(provider_state_dir) if provider_state_dir is not None else None
            ),
        )

        stream_interpretation = _with_observed_output(
            _claude_stream_interpretation(),
            _on_live_output,
        )
    else:
        command_argv = _opencode_command(
            model=request.model,
            effort=request.effort,
            run_kind=run_kind,
            session_uuid=provider_session_id,
        )
        command = _legacy_command_text(
            command_argv,
            prompt_path,
            opencode_prompt_substitution=True,
        )
        environment = _opencode_env(
            auth=request.provider_auth,
            state_dir_container_path=str(provider_state_dir),
            tool_policy=request.tool_access.tool_policy,
        )
        stream_interpretation = _with_observed_opencode_output(
            _opencode_stream_interpretation(
                reduce_output=_reduce_opencode_session_output,
            ),
            _on_live_output,
        )
    invocation_result = _invoke_provider(
        provider_invocation_adapter=invocation_adapter,
        command="" if continuation_service == "opencode" else command,
        command_argv=command_argv,
        prefer_argv=(continuation_service in {"claude", "opencode"}),
        worktree=request.invocation_dir,
        environment=environment,
        prompt_content=request.prompt,
        prompt_path=prompt_path,
        cleanup_prompt_path=True,
        run_kind=run_kind,
        provider_session_id=provider_session_id,
        stream_interpretation=stream_interpretation,
    )
    if isinstance(invocation_result, ProviderInvocationFailure):
        provider_session_id = _provider_session_id_from_failure(
            continuation_service,
            invocation_result,
            fallback_provider_session_id=provider_session_id,
        )
        if continuation_service == "opencode":
            exact_transcript_match = _opencode_exact_transcript_match(
                saved_exact_transcript_match=saved_exact_transcript_match,
                provider_session_id=provider_session_id,
                state_dir_session_id=state_dir_session_id,
            )
        failure_error = _provider_invocation_error_from_failure(
            continuation_service,
            invocation_result,
        )
        if (
            failure_error.invocation_progress is InvocationProgress.STARTED
            and provider_session_id is not None
        ):
            if continuation_service == "claude":
                failure_error.continuation = _build_claude_continuation(
                    model=request.model,
                    effort=request.effort,
                    tool_access=request.tool_access,
                    provider_session_id=provider_session_id,
                )
            else:
                failure_error.continuation = _build_opencode_continuation(
                    model=request.model,
                    effort=request.effort,
                    tool_access=request.tool_access,
                    provider_session_id=provider_session_id,
                    provider_state_dir=provider_state_dir,
                    exact_transcript_match=exact_transcript_match,
                )
        else:
            failure_error.continuation = None
        cleanup_opencode_state_dir()
        raise failure_error
    if continuation_service == "opencode":
        provider_session_id = invocation_result.provider_session_id
        assert provider_session_id is not None
        assert provider_state_dir is not None
        exact_transcript_match = _opencode_exact_transcript_match(
            saved_exact_transcript_match=saved_exact_transcript_match,
            provider_session_id=provider_session_id,
            state_dir_session_id=state_dir_session_id,
        )
        _persist_opencode_session_id(provider_state_dir, provider_session_id)
    if continuation_service == "claude":
        provider_session_id = (
            invocation_result.provider_session_id or provider_session_id
        )
    assert provider_session_id is not None
    result_text = invocation_result.output
    usage = invocation_result.usage
    if continuation_service == "claude":
        result_continuation = _build_claude_continuation(
            model=request.model,
            effort=request.effort,
            tool_access=request.tool_access,
            provider_session_id=provider_session_id,
        )
    else:
        result_continuation = _build_opencode_continuation(
            model=request.model,
            effort=request.effort,
            tool_access=request.tool_access,
            provider_session_id=provider_session_id,
            provider_state_dir=provider_state_dir,
            exact_transcript_match=exact_transcript_match,
        )
    cleanup_opencode_state_dir()
    return _completed_outcome(
        output=result_text,
        usage=usage,
        continuation=result_continuation,
        service=continuation_service,
        model=request.model,
        effort=request.effort,
    )
