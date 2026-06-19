from __future__ import annotations

import json
import logging
import re
import shlex
import shutil
import subprocess as _subprocess
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, cast

from . import _time as _time_module
from .agent_log import AgentInvocationLog, WorkInvocationLog
from ._provider_invocation import (
    ProductionProviderInvocationAdapter,
    ProviderInvocationAdapter,
    ProviderInvocationLogContext,
    ProviderInvocationPrompt,
    ProviderInvocationRequest,
    ProviderInvocationResult,
    ProviderOutputReductionHooks,
    provider_invocation_failure_provider_session_id,
    provider_invocation_failure_stdout_lines,
)
from ._portable_continuation_payload import (
    create_portable_continuation_payload,
    read_portable_continuation_payload,
)
from ._runtime_lifecycle import (
    _DEFAULT_EPHEMERAL_ROLE,
    Continuation,
    EphemeralResultMetadata,
    EphemeralRunRequest,
    EphemeralRunResult,
    EphemeralRuntimeMetadata,
    ProviderAuth,
    ProviderUsage,
    ResumedSessionRunRequest,
    RuntimeOutcome,
    SessionRunResult,
    SessionRuntimeMetadata,
    NewSessionRunRequest,
)
from .contracts import (
    AssistantTurn,
    CredentialFailure,
    HardError,
    PromptTokens,
    Result,
    ToolAccess,
    ToolPolicy,
    TransientError,
    UsageLimit,
)
from .errors import (
    AgentCredentialFailureError,
    RetryableProviderFailureError,
    RuntimeConfigurationError,
    UsageLimitError,
)
from .invocation_progress import InvocationProgress
from .provider_errors import ProviderErrorObservation
from .provider_output import reduce_text_output_events
from .session import RunKind, provider_state_relpath
from .stage_priority_chain import iter_stage_chain
from .types import StageSelection

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
_OPENCODE_GO_MODELS = frozenset(
    {
        "deepseek-v4-flash",
        "deepseek-v4-pro",
        "glm-5",
        "glm-5.1",
        "hy3-preview",
        "kimi-k2.5",
        "kimi-k2.6",
        "mimo-v2-omni",
        "mimo-v2-pro",
        "mimo-v2.5",
        "mimo-v2.5-pro",
        "minimax-m2.5",
        "minimax-m2.7",
        "qwen3.5-plus",
        "qwen3.6-plus",
        "qwen3.7-max",
    }
)
_OPENCODE_VALID_EFFORTS = frozenset({"medium"})
_CLAUDE_SUBSCRIPTION_ACCESS_DENIAL_PHRASE = (
    "disabled Claude subscription access for Claude Code"
)
_CODEX_USAGE_LIMIT_SUBSTRING = "You've hit your usage limit"
_CODEX_RESET_PATTERN = re.compile(
    r"(?:(?P<month>[A-Za-z]+)\s+(?P<day>\d{1,2}),\s+)?"
    r"(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?(?P<ampm>am|pm)\s+\(UTC\)",
    re.IGNORECASE,
)
_CODEX_GENERIC_AUTH_RE = re.compile(
    r"\b(?:401|unauthorized|invalid_grant|invalid token|missing bearer|basic authentication)\b",
    re.IGNORECASE,
)
_CODEX_HTTP_STATUS_RE = re.compile(r"\bstatus\s+(?P<status>\d{3})\b", re.IGNORECASE)
_CLAUDE_RESET_PATTERN = re.compile(
    r"resets\s+"
    r"(?:(?P<month>[A-Za-z]+)\s+(?P<day>\d{1,2}),\s+)?"
    r"(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?(?P<ampm>am|pm)\s+\(UTC\)",
    re.IGNORECASE,
)
_OPENCODE_RESET_PATTERN = re.compile(
    r"Try again at\s+"
    r"(?P<month>[A-Za-z]+)\s+"
    r"(?P<day>\d{1,2})(?:st|nd|rd|th)?,\s+"
    r"(?P<year>\d{4})\s+"
    r"(?P<hour>\d{1,2}):(?P<minute>\d{2})\s+"
    r"(?P<ampm>AM|PM)\.",
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
_WAKE_TIME_BUFFER = timedelta(minutes=2)


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
        stage: StageSelection,
        *,
        now: datetime,
    ) -> StageSelection | None:
        with self._lock:
            for candidate in iter_stage_chain(stage):
                if candidate.service not in _SUPPORTED_BUILTIN_SERVICES:
                    continue
                if self._is_available_locked(candidate.service, now):
                    return candidate
        return None

    def has_available_stage(self, stage: StageSelection, *, now: datetime) -> bool:
        return self.first_available_stage(stage, now=now) is not None

    def next_wake_time(
        self, stage: StageSelection, *, now: datetime
    ) -> datetime | None:
        with self._lock:
            wake_times = []
            for candidate in iter_stage_chain(stage):
                if candidate.service not in _SUPPORTED_BUILTIN_SERVICES:
                    continue
                exhausted_until = self._exhausted_until_by_service.get(
                    candidate.service
                )
                if exhausted_until is None:
                    continue
                if exhausted_until <= now:
                    self._exhausted_until_by_service.pop(candidate.service, None)
                    continue
                wake_times.append(exhausted_until)
            if not wake_times:
                return None
            return min(wake_times)

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


def supported_builtin_stage(stage: StageSelection) -> StageSelection | None:
    for candidate in iter_stage_chain(stage):
        if candidate.service in _SUPPORTED_BUILTIN_SERVICES:
            return candidate
    return None


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


def _validate_claude_stage(stage: StageSelection) -> None:
    if stage.model not in _CLAUDE_VALID_MODELS:
        raise RuntimeConfigurationError(f"Unsupported Claude model {stage.model!r}.")
    if stage.effort not in _CLAUDE_VALID_EFFORTS:
        raise RuntimeConfigurationError(f"Unsupported Claude effort {stage.effort!r}.")


def _validate_codex_stage(stage: StageSelection) -> None:
    if stage.model not in _CODEX_VALID_MODELS:
        raise RuntimeConfigurationError(f"Unsupported Codex model {stage.model!r}.")
    if stage.effort not in _CODEX_VALID_EFFORTS:
        raise RuntimeConfigurationError(f"Unsupported Codex effort {stage.effort!r}.")


def _validate_opencode_stage(stage: StageSelection) -> None:
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
    prompt_path: Path,
    run_kind: RunKind = RunKind.FRESH,
    session_uuid: str | None = None,
) -> str:
    profile = (
        tool_access.tool_policy.profile
        if isinstance(tool_access.tool_policy, ToolPolicy)
        else tool_access.tool_policy
    )
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
) -> str:
    tool_policy = tool_access.tool_policy
    if run_kind == RunKind.RESUME and session_uuid:
        parts = ["codex exec resume", session_uuid]
    else:
        parts = ["codex exec"]
    if model:
        parts.append(f"-m {model}")
    if effort:
        parts.append(f"-c model_reasoning_effort={effort}")
    parts.append("-c approval_policy=never")
    if tool_policy is ToolPolicy.PARTIAL:
        parts.append("--dangerously-bypass-approvals-and-sandbox")
    else:
        parts.append("--sandbox danger-full-access")
    parts.append("--json")
    parts.append("< /tmp/.pycastle_prompt")
    return " ".join(parts)


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


def _opencode_go_config_content() -> str:
    return json.dumps(
        {
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
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _opencode_command(
    *,
    model: str,
    effort: str,
    run_kind: RunKind = RunKind.FRESH,
    session_uuid: str | None = None,
) -> str:
    del effort
    parts = ["opencode run", "--format json"]
    if run_kind == RunKind.RESUME and session_uuid:
        parts.append(f"--session {session_uuid}")
    if model:
        parts.append(f"--model {_opencode_go_model_ref(model)}")
    parts.append('"$(cat /tmp/.pycastle_prompt)"')
    return " ".join(parts)


def _opencode_env(
    *,
    auth: ProviderAuth | None,
    state_dir_container_path: str | None = None,
) -> dict[str, str]:
    env: dict[str, str] = {"TZ": "UTC"}
    if state_dir_container_path:
        env["OPENCODE_HOME"] = state_dir_container_path
    api_key = None if auth is None else auth.opencode_api_key
    if api_key:
        env["OPENCODE_GO_API_KEY"] = api_key
        env["OPENCODE_CONFIG_CONTENT"] = _opencode_go_config_content()
    return env


def _is_claude_subscription_access_denial(event: dict[str, Any]) -> bool:
    result = event.get("result")
    return (
        event.get("is_error") is True
        and event.get("api_error_status") == 403
        and isinstance(result, str)
        and _CLAUDE_SUBSCRIPTION_ACCESS_DENIAL_PHRASE.lower() in result.lower()
    )


def _provider_error_observation(
    *,
    raw_provider_text: str,
    source_stream: str,
    status_code: int | None = None,
    provider_code: str | None = None,
) -> ProviderErrorObservation:
    return ProviderErrorObservation(
        service_name="codex",
        raw_provider_text=raw_provider_text,
        source_stream=source_stream,
        status_code=status_code,
        provider_code=provider_code,
    )


def _classify_codex_error_message(
    message: str,
    *,
    source_stream: str,
) -> CredentialFailure | HardError | TransientError | None:
    lowered_message = message.lower()
    if "refresh_token_reused" in message:
        return CredentialFailure(
            raw_message=message,
            service_name="codex",
            status_code=401,
            classification="codex_auth_lineage_exhausted",
            source_observations=(
                _provider_error_observation(
                    raw_provider_text=message,
                    source_stream=source_stream,
                    status_code=401,
                    provider_code="refresh_token_reused",
                ),
            ),
        )
    if (
        "access token could not be refreshed" in lowered_message
        and "refresh token was already used" in lowered_message
    ):
        return CredentialFailure(
            raw_message=message,
            service_name="codex",
            status_code=401,
            classification="codex_auth_lineage_exhausted",
            source_observations=(
                _provider_error_observation(
                    raw_provider_text=message,
                    source_stream=source_stream,
                    status_code=401,
                ),
            ),
        )
    if _CODEX_GENERIC_AUTH_RE.search(message):
        return HardError(
            status_code=401,
            raw_message=message,
            observations=(
                _provider_error_observation(
                    raw_provider_text=message,
                    source_stream=source_stream,
                    status_code=401,
                ),
            ),
        )
    match = _CODEX_HTTP_STATUS_RE.search(message)
    if match is None:
        return None
    status = int(match.group("status"))
    observation = _provider_error_observation(
        raw_provider_text=message,
        source_stream=source_stream,
        status_code=status,
    )
    if status >= 500:
        return TransientError(
            status_code=status,
            raw_message=message,
            observations=(observation,),
        )
    if 400 <= status < 500:
        return HardError(
            status_code=status,
            raw_message=message,
            observations=(observation,),
        )
    return None


def _extract_codex_usage_limit(message: str) -> UsageLimit | None:
    if _CODEX_USAGE_LIMIT_SUBSTRING not in message:
        return None
    reset_time = _parse_codex_reset_time(message)
    return UsageLimit(
        reset_time=reset_time,
        raw_message=None if reset_time is not None else message,
    )


def _parse_claude_event(line: str) -> list[Any]:
    return _parse_claude_event_with_dependencies(
        line,
        parse_claude_reset_time=_parse_claude_reset_time,
        is_claude_subscription_access_denial=_is_claude_subscription_access_denial,
    )


def _parse_claude_event_with_dependencies(
    line: str,
    *,
    parse_claude_reset_time: Callable[[object], datetime | None],
    is_claude_subscription_access_denial: Callable[[dict[str, Any]], bool],
) -> list[Any]:
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return []
    if not isinstance(event, dict):
        return []
    if event.get("api_error_status") == 429:
        reset_time = parse_claude_reset_time(event.get("result"))
        return [
            UsageLimit(
                reset_time=reset_time,
                raw_message=None if reset_time is not None else line,
            )
        ]
    if is_claude_subscription_access_denial(event):
        denial_message = event.get("result")
        return [
            CredentialFailure(
                raw_message=line,
                service_name="claude",
                source_observations=(
                    ProviderErrorObservation(
                        service_name="claude",
                        raw_provider_text=(
                            denial_message if isinstance(denial_message, str) else line
                        ),
                        source_stream="json_event.result",
                        status_code=403,
                    ),
                ),
                status_code=403,
            )
        ]
    if event.get("is_error") and event.get("type") == "result":
        status = event.get("api_error_status")
        if status is None or (isinstance(status, int) and status >= 500):
            return [
                TransientError(
                    status_code=status if isinstance(status, int) else None,
                    raw_message=line,
                )
            ]
        if isinstance(status, int) and 400 <= status < 500:
            return [HardError(status_code=status, raw_message=line)]
        return []
    if event.get("type") == "assistant":
        message = event.get("message") or {}
        content = message.get("content") or []
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        usage = message.get("usage") or {}
        total_tokens = (
            int(usage.get("input_tokens") or 0)
            + int(usage.get("cache_creation_input_tokens") or 0)
            + int(usage.get("cache_read_input_tokens") or 0)
        )
        parsed_events: list[Any] = []
        if total_tokens > 0:
            parsed_events.append(PromptTokens(count=total_tokens))
        if parts:
            parsed_events.append(AssistantTurn(text="\n\n".join(parts)))
        return parsed_events
    if (
        event.get("type") == "result"
        and event.get("is_error") is not True
        and isinstance(event.get("result"), str)
    ):
        return [Result(text=cast(str, event["result"]))]
    return []


def _reduce_claude_stream(lines: list[str]) -> tuple[str, ProviderUsage | None]:
    return _reduce_claude_stream_with_dependencies(
        lines,
        parse_claude_event=_parse_claude_event,
    )


def _merge_provider_usage(
    current: ProviderUsage | None,
    observed: ProviderUsage | None,
) -> ProviderUsage | None:
    if observed is None:
        return current
    if current is None:
        return observed
    return ProviderUsage(
        input_tokens=(
            observed.input_tokens
            if observed.input_tokens is not None
            else current.input_tokens
        ),
        output_tokens=(
            observed.output_tokens
            if observed.output_tokens is not None
            else current.output_tokens
        ),
        cache_read_input_tokens=(
            observed.cache_read_input_tokens
            if observed.cache_read_input_tokens is not None
            else current.cache_read_input_tokens
        ),
        cache_creation_input_tokens=(
            observed.cache_creation_input_tokens
            if observed.cache_creation_input_tokens is not None
            else current.cache_creation_input_tokens
        ),
        cost_usd=observed.cost_usd
        if observed.cost_usd is not None
        else current.cost_usd,
        duration_seconds=(
            observed.duration_seconds
            if observed.duration_seconds is not None
            else current.duration_seconds
        ),
    )


def _reduce_claude_stream_with_dependencies(
    lines: list[str],
    *,
    parse_claude_event: Callable[[str], list[Any]],
) -> tuple[str, ProviderUsage | None]:
    usage: ProviderUsage | None = None
    parsed_events: list[Any] = []
    for line in lines:
        usage = _merge_provider_usage(usage, _parse_claude_usage(line))
        parsed_events.extend(parse_claude_event(line))
    try:
        output = reduce_text_output_events(
            parsed_events,
            lambda _turn: None,
            provider="claude",
        )
    except (RetryableProviderFailureError, UsageLimitError) as exc:
        if exc.usage is None:
            exc.usage = usage
        raise
    return output, usage


def _parse_claude_usage(line: str) -> ProviderUsage | None:
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(event, dict) or event.get("type") != "assistant":
        return None
    message = event.get("message")
    if not isinstance(message, dict):
        return None
    usage = message.get("usage")
    if not isinstance(usage, dict):
        return None
    input_tokens = usage.get("input_tokens")
    cache_creation_input_tokens = usage.get("cache_creation_input_tokens")
    cache_read_input_tokens = usage.get("cache_read_input_tokens")
    if not any(
        value is not None
        for value in (
            input_tokens,
            cache_creation_input_tokens,
            cache_read_input_tokens,
        )
    ):
        return None
    return ProviderUsage(
        input_tokens=int(input_tokens) if input_tokens is not None else None,
        cache_creation_input_tokens=(
            int(cache_creation_input_tokens)
            if cache_creation_input_tokens is not None
            else None
        ),
        cache_read_input_tokens=(
            int(cache_read_input_tokens)
            if cache_read_input_tokens is not None
            else None
        ),
    )


def _parse_codex_event(line: str) -> list[Any]:
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return []
    if not isinstance(event, dict):
        return []
    event_type = event.get("type")
    if event_type == "item.completed":
        item = event.get("item") or {}
        if item.get("type") != "agent_message":
            return []
        content = item.get("text")
        if content is None:
            content = item.get("content") or ""
        return [AssistantTurn(text=content)] if content else []
    if event_type == "turn.failed":
        error = event.get("error") or {}
        message = error.get("message") or ""
        limit = _extract_codex_usage_limit(message)
        if limit is not None:
            return [limit]
        classified = _classify_codex_error_message(
            message,
            source_stream="json_event.turn_failed",
        )
        if classified is not None:
            _log.warning("codex turn.failed: %s", message)
            return [classified]
        return []
    if event_type == "error":
        message = event.get("message") or ""
        limit = _extract_codex_usage_limit(message)
        if limit is not None:
            return [limit]
        classified = _classify_codex_error_message(
            message,
            source_stream="json_event.error",
        )
        if classified is not None:
            _log.warning("codex error: %s", message)
            return [classified]
        return []
    return []


def _reduce_codex_stream(lines: list[str]) -> tuple[str, ProviderUsage | None]:
    usage: ProviderUsage | None = None
    parsed_events: list[Any] = []
    for line in lines:
        usage = _merge_provider_usage(usage, _parse_codex_usage(line))
        parsed_events.extend(_parse_codex_event(line))
    try:
        output = reduce_text_output_events(
            parsed_events,
            lambda _turn: None,
            provider="codex",
        )
    except (RetryableProviderFailureError, UsageLimitError) as exc:
        if exc.usage is None:
            exc.usage = usage
        raise
    return output, usage


def _parse_codex_usage(line: str) -> ProviderUsage | None:
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(event, dict) or event.get("type") != "turn.completed":
        return None
    usage = event.get("usage")
    if not isinstance(usage, dict):
        return None
    input_tokens = usage.get("input_tokens")
    cached_tokens = usage.get("cached_tokens")
    output_tokens = usage.get("output_tokens")
    if not any(
        value is not None for value in (input_tokens, cached_tokens, output_tokens)
    ):
        return None
    return ProviderUsage(
        input_tokens=int(input_tokens) if input_tokens is not None else None,
        output_tokens=int(output_tokens) if output_tokens is not None else None,
        cache_read_input_tokens=(
            int(cached_tokens) if cached_tokens is not None else None
        ),
    )


def _parse_opencode_reset_time(retry_text: object) -> datetime | None:
    if not isinstance(retry_text, str):
        return None
    match = _OPENCODE_RESET_PATTERN.search(retry_text)
    if match is None:
        return None
    month = _CLAUDE_MONTHS.get(match.group("month").lower())
    if month is None:
        return None
    hour = int(match.group("hour"))
    if not 1 <= hour <= 12:
        return None
    if match.group("ampm").lower() == "pm" and hour != 12:
        hour += 12
    elif match.group("ampm").lower() == "am" and hour == 12:
        hour = 0
    minute = int(match.group("minute"))
    if not 0 <= minute <= 59:
        return None
    return datetime(
        int(match.group("year")),
        month,
        int(match.group("day")),
        hour,
        minute,
        tzinfo=timezone.utc,
    ).astimezone()


def _parse_claude_reset_time(retry_text: object) -> datetime | None:
    if not isinstance(retry_text, str):
        return None
    match = _CLAUDE_RESET_PATTERN.search(retry_text)
    if match is None:
        return None
    hour = int(match.group("hour"))
    if not 1 <= hour <= 12:
        return None
    ampm = match.group("ampm").lower()
    if ampm == "pm" and hour != 12:
        hour += 12
    elif ampm == "am" and hour == 12:
        hour = 0
    minute = int(match.group("minute") or 0)
    if not 0 <= minute <= 59:
        return None
    now_local = _time_module.now_local()
    utc_now = now_local.astimezone(timezone.utc)
    month_text = match.group("month")
    day_text = match.group("day")
    if month_text is not None or day_text is not None:
        if month_text is None or day_text is None:
            return None
        month = _CLAUDE_MONTHS.get(month_text.lower())
        if month is None:
            return None
        utc_dt = datetime(
            utc_now.year,
            month,
            int(day_text),
            hour,
            minute,
            tzinfo=timezone.utc,
        )
        local_dt = utc_dt.astimezone(now_local.tzinfo)
        if local_dt < now_local - timedelta(days=31):
            return datetime(
                utc_dt.year + 1,
                month,
                int(day_text),
                hour,
                minute,
                tzinfo=timezone.utc,
            ).astimezone(now_local.tzinfo)
        return local_dt
    utc_dt = datetime.combine(
        utc_now.date(),
        datetime.min.time(),
        tzinfo=timezone.utc,
    ).replace(hour=hour, minute=minute)
    if utc_dt < utc_now - timedelta(minutes=2):
        utc_dt += timedelta(days=1)
    return utc_dt.astimezone(now_local.tzinfo)


def _parse_codex_reset_time(retry_text: object) -> datetime | None:
    if not isinstance(retry_text, str):
        return None
    match = _CODEX_RESET_PATTERN.search(retry_text)
    if match is None:
        return None
    hour = int(match.group("hour"))
    if not 1 <= hour <= 12:
        return None
    ampm = match.group("ampm").lower()
    if ampm == "pm" and hour != 12:
        hour += 12
    elif ampm == "am" and hour == 12:
        hour = 0
    minute = int(match.group("minute") or 0)
    if not 0 <= minute <= 59:
        return None
    now_local = _time_module.now_local()
    utc_now = now_local.astimezone(timezone.utc)
    month_text = match.group("month")
    day_text = match.group("day")
    if month_text is not None or day_text is not None:
        if month_text is None or day_text is None:
            return None
        month = _CLAUDE_MONTHS.get(month_text.lower())
        if month is None:
            return None
        utc_dt = datetime(
            utc_now.year,
            month,
            int(day_text),
            hour,
            minute,
            tzinfo=timezone.utc,
        )
        local_dt = utc_dt.astimezone(now_local.tzinfo)
        if local_dt < now_local - timedelta(days=31):
            return datetime(
                utc_dt.year + 1,
                month,
                int(day_text),
                hour,
                minute,
                tzinfo=timezone.utc,
            ).astimezone(now_local.tzinfo)
        return local_dt
    utc_dt = datetime.combine(
        utc_now.date(),
        datetime.min.time(),
        tzinfo=timezone.utc,
    ).replace(hour=hour, minute=minute)
    if utc_dt < utc_now - timedelta(minutes=2):
        utc_dt += timedelta(days=1)
    return utc_dt.astimezone(now_local.tzinfo)


def _validate_codex_auth() -> None:
    auth_path = Path.home() / ".codex" / "auth.json"
    if auth_path.exists():
        return
    raise _missing_codex_auth_error()


def _missing_codex_auth_error() -> AgentCredentialFailureError:
    message = "Codex authentication missing: run `codex login` on the host."
    return AgentCredentialFailureError(
        message=message,
        service_name="codex",
        status_code=401,
        observations=(
            ProviderErrorObservation(
                service_name="codex",
                raw_provider_text=message,
                source_stream="pre-dispatch host check",
                status_code=401,
            ),
        ),
    )


def _opencode_error_data(event: dict[str, Any]) -> dict[str, Any] | None:
    error = event.get("error")
    if not isinstance(error, dict):
        return None
    data = error.get("data")
    if not isinstance(data, dict):
        return None
    return data


def _extract_opencode_usage_limit(event: dict[str, Any]) -> UsageLimit | None:
    data = _opencode_error_data(event)
    if data is None or data.get("statusCode") != 429:
        return None
    message = data.get("message")
    if not isinstance(message, str):
        return UsageLimit(reset_time=None, raw_message=None)
    reset_time = _parse_opencode_reset_time(message)
    return UsageLimit(
        reset_time=reset_time,
        raw_message=None if reset_time is not None else message,
    )


def _extract_opencode_credential_failure(
    event: dict[str, Any],
) -> CredentialFailure | None:
    data = _opencode_error_data(event)
    if data is None:
        return None
    status = data.get("statusCode")
    message = data.get("message")
    error = event.get("error")
    error_name = error.get("name") if isinstance(error, dict) else None
    if (
        status == 401
        and isinstance(message, str)
        and message.lower() == "invalid api key"
        and error_name == "AuthenticationError"
    ):
        return CredentialFailure(
            raw_message=message,
            service_name="opencode",
            classification="operator_actionable_agent_credential_failure",
            source_observations=(
                ProviderErrorObservation(
                    service_name="opencode",
                    raw_provider_text=message,
                    source_stream="json_event.error",
                    status_code=401,
                    error_name="AuthenticationError",
                ),
            ),
            status_code=401,
        )
    return None


def _extract_opencode_error(
    event: dict[str, Any],
) -> HardError | TransientError | None:
    data = _opencode_error_data(event)
    if data is None:
        return None
    message = data.get("message")
    if not isinstance(message, str) or not message:
        return None
    status = data.get("statusCode")
    if isinstance(status, int):
        if status >= 500:
            return TransientError(status_code=status, raw_message=message)
        if 400 <= status < 500:
            return HardError(status_code=status, raw_message=message)
    if status is None and message.lower().startswith("model not found:"):
        return HardError(status_code=400, raw_message=message)
    if status is None:
        return TransientError(status_code=None, raw_message=message)
    return None


def _parse_opencode_events(
    lines: list[str],
    *,
    on_provider_session_id: Callable[[str], None] | None = None,
) -> list[Any]:
    parsed_events: list[Any] = []
    assistant_turns: list[str] = []
    seen_session_id: str | None = None
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        session_id = event.get("sessionID")
        if (
            isinstance(session_id, str)
            and session_id
            and session_id != seen_session_id
            and on_provider_session_id is not None
        ):
            seen_session_id = session_id
            on_provider_session_id(session_id)
        if event.get("type") == "text":
            part = event.get("part")
            if not isinstance(part, dict):
                continue
            if part.get("type") != "text":
                continue
            time = part.get("time")
            if not isinstance(time, dict) or time.get("end") is None:
                continue
            text = part.get("text")
            if not isinstance(text, str):
                continue
            stripped = text.strip()
            if not stripped:
                continue
            assistant_turns.append(stripped)
            parsed_events.append(AssistantTurn(text=stripped))
            continue
        if event.get("type") == "session.status":
            status = event.get("status")
            if (
                isinstance(status, dict)
                and status.get("type") == "idle"
                and assistant_turns
            ):
                parsed_events.append(Result(text="\n\n".join(assistant_turns)))
                return parsed_events
            continue
        if event.get("type") == "error":
            limit = _extract_opencode_usage_limit(event)
            if limit is not None:
                parsed_events.append(limit)
            else:
                classified: CredentialFailure | HardError | TransientError | None = (
                    _extract_opencode_credential_failure(event)
                )
                if classified is None:
                    classified = _extract_opencode_error(event)
                if classified is not None:
                    parsed_events.append(classified)
            return parsed_events
    return parsed_events


def _extract_opencode_provider_session_id(lines: list[str]) -> str | None:
    provider_session_id: str | None = None

    def _record_provider_session_id(session_id: str) -> None:
        nonlocal provider_session_id
        provider_session_id = session_id

    _parse_opencode_events(
        lines,
        on_provider_session_id=_record_provider_session_id,
    )
    return provider_session_id


def _reduce_opencode_stream(lines: list[str]) -> str:
    return reduce_text_output_events(
        _parse_opencode_events(lines),
        lambda _turn: None,
        provider="opencode",
    )


def _reduce_logged_opencode_stream(
    lines: list[str],
    *,
    work_invocation_log: WorkInvocationLog,
) -> str:
    return reduce_text_output_events(
        _parse_opencode_events(
            lines,
            on_provider_session_id=work_invocation_log.record_provider_session_id,
        ),
        lambda _turn: None,
        provider="opencode",
    )


def _select_builtin_stage(stage: StageSelection) -> StageSelection:
    candidate = supported_builtin_stage(stage)
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
    host_auth_path = Path.home() / ".codex" / "auth.json"
    if not host_auth_path.exists():
        raise _missing_codex_auth_error()
    shutil.copyfile(host_auth_path, provider_auth_path)


def _extract_codex_provider_session_id(lines: list[str]) -> str | None:
    thread_ids: set[str] = set()
    for line in lines:
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
        if len(thread_ids) > 1:
            return None
    if len(thread_ids) != 1:
        return None
    return next(iter(thread_ids))


def _build_codex_continuation(
    *,
    model: str,
    effort: str,
    tool_access: ToolAccess,
    provider_session_id: str,
    provider_state_dir_relpath: str,
) -> Continuation:
    return create_portable_continuation_payload(
        service_name="codex",
        model=model,
        effort=effort,
        tool_access=tool_access,
        provider_resume_state={
            "run_kind": RunKind.RESUME.value,
            "provider_session_id": provider_session_id,
            "provider_state_dir_relpath": provider_state_dir_relpath,
            "exact_transcript_match": False,
        },
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
    provider_state_dir_relpath: str,
) -> Continuation:
    return create_portable_continuation_payload(
        service_name="claude",
        model=model,
        effort=effort,
        tool_access=tool_access,
        provider_resume_state={
            "run_kind": RunKind.RESUME.value,
            "provider_session_id": provider_session_id,
            "provider_state_dir_relpath": provider_state_dir_relpath,
            "exact_transcript_match": False,
        },
    ).to_continuation()


def _build_opencode_continuation(
    *,
    model: str,
    effort: str,
    tool_access: ToolAccess,
    provider_session_id: str,
    provider_state_dir_relpath: str,
    exact_transcript_match: bool | None = None,
) -> Continuation:
    provider_resume_state: dict[str, Any] = {
        "provider_session_id": provider_session_id,
        "provider_state_dir_relpath": provider_state_dir_relpath,
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


def _start_invocation_log(
    *,
    logs_dir: Path | None,
    role: Any,
) -> Any:
    if logs_dir is None:
        return None
    return AgentInvocationLog().start_logical_session(
        log_name=role.value,
        logs_dir=logs_dir,
    )


def _provider_invocation_log_context(
    *,
    invocation_log: Any,
    role: Any,
    usage_limit_scope: Any,
) -> ProviderInvocationLogContext | None:
    if invocation_log is None:
        return None
    return ProviderInvocationLogContext(
        invocation_log=invocation_log,
        role=role,
        usage_limit_scope=usage_limit_scope,
    )


def _reduce_logged_opencode_stream_with_usage(
    lines: list[str],
    *,
    work_invocation_log: WorkInvocationLog,
) -> tuple[str, ProviderUsage | None]:
    return (
        _reduce_logged_opencode_stream(
            lines,
            work_invocation_log=work_invocation_log,
        ),
        None,
    )


def _default_provider_invocation_adapter() -> ProviderInvocationAdapter:
    return ProductionProviderInvocationAdapter()


def _invoke_provider(
    *,
    provider_invocation_adapter: ProviderInvocationAdapter,
    command: str,
    worktree: Path,
    environment: dict[str, str],
    prompt_content: str,
    prompt_path: Path | None,
    cleanup_prompt_path: bool,
    run_kind: RunKind,
    role: Any,
    usage_limit_scope: Any,
    provider_session_id: str | None,
    reduce_output: Callable[[list[str]], tuple[str, ProviderUsage | None]],
    invocation_log: Any,
    reduce_logged_output: Callable[
        [list[str], WorkInvocationLog],
        tuple[str, ProviderUsage | None],
    ]
    | None = None,
    extract_provider_session_id: Callable[[list[str]], str | None] | None = None,
) -> ProviderInvocationResult:
    return provider_invocation_adapter.execute(
        ProviderInvocationRequest(
            command=command,
            worktree=worktree,
            environment=environment,
            prompt=ProviderInvocationPrompt(
                content=prompt_content,
                path=prompt_path,
                cleanup_path=cleanup_prompt_path,
            ),
            run_kind=run_kind,
            role=role,
            usage_limit_scope=usage_limit_scope,
            log_context=_provider_invocation_log_context(
                invocation_log=invocation_log,
                role=role,
                usage_limit_scope=usage_limit_scope,
            ),
            provider_session_id=provider_session_id,
            output_hooks=ProviderOutputReductionHooks(
                reduce_output=reduce_output,
                reduce_logged_output=reduce_logged_output,
                extract_provider_session_id=extract_provider_session_id,
            ),
        )
    )


def _invoke_claude_new_session_provider(
    *,
    provider_invocation_adapter: ProviderInvocationAdapter,
    request: NewSessionRunRequest,
    stage: StageSelection,
    provider_state_dir: Path,
    run_kind: RunKind,
    provider_session_id: str,
) -> ProviderInvocationResult:
    return _invoke_provider(
        provider_invocation_adapter=provider_invocation_adapter,
        command=_claude_command(
            model=stage.model,
            effort=stage.effort,
            tool_access=request.tool_access,
            prompt_path=request.worktree / ".pycastle_prompt",
            run_kind=run_kind,
            session_uuid=provider_session_id,
        ),
        worktree=request.worktree,
        environment=_claude_env(
            auth=request.provider_auth,
            state_dir_container_path=str(provider_state_dir),
        ),
        prompt_content=request.prompt,
        prompt_path=request.worktree / ".pycastle_prompt",
        cleanup_prompt_path=True,
        run_kind=run_kind,
        role=request.role,
        usage_limit_scope=request.usage_limit_scope,
        provider_session_id=provider_session_id,
        reduce_output=_reduce_claude_stream,
        invocation_log=_start_invocation_log(
            logs_dir=request.logs_dir,
            role=request.role,
        ),
    )


def _invoke_codex_new_session_provider(
    *,
    provider_invocation_adapter: ProviderInvocationAdapter,
    request: NewSessionRunRequest,
    stage: StageSelection,
    provider_state_dir: Path,
) -> ProviderInvocationResult:
    return _invoke_provider(
        provider_invocation_adapter=provider_invocation_adapter,
        command=_codex_command(
            model=stage.model,
            effort=stage.effort,
            tool_access=request.tool_access,
            run_kind=RunKind.FRESH,
            session_uuid=None,
        ),
        worktree=request.worktree,
        environment=_codex_env(
            state_dir_container_path=str(provider_state_dir),
        ),
        prompt_content=request.prompt,
        prompt_path=Path("/tmp/.pycastle_prompt"),
        cleanup_prompt_path=True,
        run_kind=RunKind.FRESH,
        role=request.role,
        usage_limit_scope=request.usage_limit_scope,
        provider_session_id=None,
        reduce_output=_reduce_codex_stream,
        invocation_log=_start_invocation_log(
            logs_dir=request.logs_dir,
            role=request.role,
        ),
    )


def _invoke_codex_resumed_session_provider(
    *,
    provider_invocation_adapter: ProviderInvocationAdapter,
    request: ResumedSessionRunRequest,
    provider_state_dir: Path,
    provider_session_id: str,
) -> ProviderInvocationResult:
    return _invoke_provider(
        provider_invocation_adapter=provider_invocation_adapter,
        command=_codex_command(
            model=request.model,
            effort=request.effort,
            tool_access=request.tool_access,
            run_kind=RunKind.RESUME,
            session_uuid=provider_session_id,
        ),
        worktree=request.worktree.host_path,
        environment=_codex_env(
            state_dir_container_path=str(provider_state_dir),
        ),
        prompt_content=request.prompt,
        prompt_path=Path("/tmp/.pycastle_prompt"),
        cleanup_prompt_path=True,
        run_kind=RunKind.RESUME,
        role=request.role,
        usage_limit_scope=request.usage_limit_scope,
        provider_session_id=provider_session_id,
        reduce_output=_reduce_codex_stream,
        invocation_log=None,
        extract_provider_session_id=_extract_codex_provider_session_id,
    )


def _active_codex_provider_session_id_from_result(
    invocation_result: ProviderInvocationResult,
    *,
    fallback_provider_session_id: str | None,
) -> str | None:
    return (
        _extract_codex_provider_session_id(list(invocation_result.stdout_lines))
        or fallback_provider_session_id
    )


def _active_codex_provider_session_id_from_failure(
    error: UsageLimitError | RetryableProviderFailureError,
    *,
    fallback_provider_session_id: str | None,
) -> str | None:
    return (
        _extract_codex_provider_session_id(
            list(provider_invocation_failure_stdout_lines(error))
        )
        or provider_invocation_failure_provider_session_id(error)
        or cast(
            str | None,
            getattr(error, "provider_session_id", fallback_provider_session_id),
        )
    )


def _invoke_opencode_new_session_provider(
    *,
    provider_invocation_adapter: ProviderInvocationAdapter,
    request: NewSessionRunRequest,
    stage: StageSelection,
    provider_state_dir: Path,
    run_kind: RunKind,
    provider_session_id: str,
) -> tuple[ProviderInvocationResult, str]:
    invocation_log = _start_invocation_log(
        logs_dir=request.logs_dir,
        role=request.role,
    )
    observed_provider_session_id = provider_session_id

    def _record_opencode_session_id(session_id: str) -> None:
        nonlocal observed_provider_session_id
        observed_provider_session_id = session_id

    def _reduce_opencode_session_output(
        lines: list[str],
    ) -> tuple[str, ProviderUsage | None]:
        return (
            reduce_text_output_events(
                _parse_opencode_events(
                    lines,
                    on_provider_session_id=_record_opencode_session_id,
                ),
                lambda _turn: None,
                provider="opencode",
            ),
            None,
        )

    def _reduce_logged_opencode_session_output(
        lines: list[str],
        work_invocation_log: WorkInvocationLog,
    ) -> tuple[str, ProviderUsage | None]:
        def _record_session_id(session_id: str) -> None:
            _record_opencode_session_id(session_id)
            work_invocation_log.record_provider_session_id(session_id)

        return (
            reduce_text_output_events(
                _parse_opencode_events(
                    lines,
                    on_provider_session_id=_record_session_id,
                ),
                lambda _turn: None,
                provider="opencode",
            ),
            None,
        )

    invocation_result = _invoke_provider(
        provider_invocation_adapter=provider_invocation_adapter,
        command=_opencode_command(
            model=stage.model,
            effort=stage.effort,
            run_kind=run_kind,
            session_uuid=provider_session_id,
        ),
        worktree=request.worktree,
        environment=_opencode_env(
            auth=request.provider_auth,
            state_dir_container_path=str(provider_state_dir),
        ),
        prompt_content=request.prompt,
        prompt_path=request.worktree / ".pycastle_prompt",
        cleanup_prompt_path=True,
        run_kind=run_kind,
        role=request.role,
        usage_limit_scope=request.usage_limit_scope,
        provider_session_id=provider_session_id,
        reduce_output=_reduce_opencode_session_output,
        invocation_log=invocation_log,
        reduce_logged_output=_reduce_logged_opencode_session_output,
        extract_provider_session_id=lambda _lines: observed_provider_session_id,
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
        [StageSelection], StageSelection
    ] = _select_builtin_stage,
    validate_claude_stage: Callable[[StageSelection], None] = _validate_claude_stage,
    validate_codex_stage: Callable[[StageSelection], None] = _validate_codex_stage,
    validate_opencode_stage: Callable[
        [StageSelection], None
    ] = _validate_opencode_stage,
    claude_command: Callable[..., str] = _claude_command,
    claude_env: Callable[..., dict[str, str]] = _claude_env,
    reduce_claude_stream: Callable[
        [list[str]], tuple[str, ProviderUsage | None]
    ] = _reduce_claude_stream,
    codex_command: Callable[..., str] = _codex_command,
    codex_env: Callable[..., dict[str, str]] = _codex_env,
    reduce_codex_stream: Callable[
        [list[str]], tuple[str, ProviderUsage | None]
    ] = _reduce_codex_stream,
    opencode_command: Callable[..., str] = _opencode_command,
    opencode_env: Callable[..., dict[str, str]] = _opencode_env,
    reduce_opencode_stream: Callable[[list[str]], str] = _reduce_opencode_stream,
    validate_codex_auth: Callable[[], None] = _validate_codex_auth,
    selected_service_path: Callable[..., tuple[str, ...]] = _selected_service_path,
) -> EphemeralRunResult:
    invocation_adapter = (
        _default_provider_invocation_adapter()
        if provider_invocation_adapter is None
        else provider_invocation_adapter
    )
    selected_stage = select_builtin_stage(request.stage)
    if selected_stage.service == "codex":
        validate_codex_stage(selected_stage)
        validate_codex_auth()
        prompt_path = Path("/tmp/.pycastle_prompt")
    elif selected_stage.service == "opencode":
        validate_opencode_stage(selected_stage)
        if request.auth is None or not request.auth.opencode_api_key:
            message = "Missing OpenCode API key."
            raise AgentCredentialFailureError(
                message=message,
                service_name="opencode",
                classification="operator_actionable_agent_credential_failure",
                observations=(
                    ProviderErrorObservation(
                        service_name="opencode",
                        raw_provider_text=message,
                        source_stream="pre-dispatch auth check",
                        status_code=401,
                    ),
                ),
                status_code=401,
            )
        prompt_path = Path("/tmp/.pycastle_prompt")
    else:
        validate_claude_stage(selected_stage)
        if request.auth is None or not request.auth.claude_code_oauth_token:
            raise AgentCredentialFailureError(
                message="Missing Claude Code OAuth token.",
                service_name="claude",
                observations=(),
            )
        prompt_path = request.worktree / ".pycastle_prompt"
    invocation_log = _start_invocation_log(
        logs_dir=None,
        role=_DEFAULT_EPHEMERAL_ROLE,
    )
    if selected_stage.service == "codex":
        invocation_result = _invoke_provider(
            provider_invocation_adapter=invocation_adapter,
            command=codex_command(
                model=selected_stage.model,
                effort=selected_stage.effort,
                tool_access=request.tool_access,
            ),
            worktree=request.worktree,
            environment=codex_env(),
            prompt_content=request.prompt,
            prompt_path=prompt_path,
            cleanup_prompt_path=True,
            run_kind=RunKind.FRESH,
            role=_DEFAULT_EPHEMERAL_ROLE,
            usage_limit_scope=None,
            provider_session_id=None,
            reduce_output=reduce_codex_stream,
            invocation_log=invocation_log,
        )
    elif selected_stage.service == "opencode":
        invocation_result = _invoke_provider(
            provider_invocation_adapter=invocation_adapter,
            command=opencode_command(
                model=selected_stage.model,
                effort=selected_stage.effort,
                run_kind=RunKind.FRESH,
                session_uuid=None,
            ),
            worktree=request.worktree,
            environment=opencode_env(
                auth=request.auth,
                state_dir_container_path=str(request.worktree),
            ),
            prompt_content=request.prompt,
            prompt_path=prompt_path,
            cleanup_prompt_path=True,
            run_kind=RunKind.FRESH,
            role=_DEFAULT_EPHEMERAL_ROLE,
            usage_limit_scope=None,
            provider_session_id=None,
            reduce_output=lambda lines: (reduce_opencode_stream(lines), None),
            invocation_log=invocation_log,
            reduce_logged_output=lambda lines, work_invocation_log: (
                _reduce_logged_opencode_stream_with_usage(
                    lines,
                    work_invocation_log=work_invocation_log,
                )
            ),
        )
    else:
        invocation_result = _invoke_provider(
            provider_invocation_adapter=invocation_adapter,
            command=claude_command(
                model=selected_stage.model,
                effort=selected_stage.effort,
                tool_access=request.tool_access,
                prompt_path=prompt_path,
            ),
            worktree=request.worktree,
            environment=claude_env(auth=request.auth),
            prompt_content=request.prompt,
            prompt_path=prompt_path,
            cleanup_prompt_path=True,
            run_kind=RunKind.FRESH,
            role=_DEFAULT_EPHEMERAL_ROLE,
            usage_limit_scope=None,
            provider_session_id=None,
            reduce_output=reduce_claude_stream,
            invocation_log=invocation_log,
        )
    result_text = invocation_result.output
    usage = invocation_result.usage
    service_path = selected_service_path(
        request.stage,
        selected_service=selected_stage.service,
    )
    return EphemeralRunResult(
        output=result_text,
        selected_service=selected_stage.service,
        selected_model=selected_stage.model,
        selected_effort=selected_stage.effort,
        tool_access=request.tool_access,
        used_fallback=len(service_path) > 1,
        metadata=EphemeralResultMetadata(
            selected_service_path=service_path,
            runtime=EphemeralRuntimeMetadata(
                run_kind=RunKind.FRESH,
            ),
        ),
        usage=usage,
    )


def _require_runtime_state_dir(runtime_state_dir: Path | None, *, context: str) -> Path:
    if runtime_state_dir is None:
        raise TypeError(f"{context} requires a `runtime_state_dir` value.")
    return runtime_state_dir


def _require_claude_auth(auth: ProviderAuth | None) -> None:
    if auth is not None and auth.claude_code_oauth_token:
        return
    raise AgentCredentialFailureError(
        message="Missing Claude Code OAuth token.",
        service_name="claude",
        observations=(),
    )


def _require_opencode_auth(auth: ProviderAuth | None) -> None:
    if auth is not None and auth.opencode_api_key:
        return
    message = "Missing OpenCode API key."
    raise AgentCredentialFailureError(
        message=message,
        service_name="opencode",
        classification="operator_actionable_agent_credential_failure",
        observations=(
            ProviderErrorObservation(
                service_name="opencode",
                raw_provider_text=message,
                source_stream="pre-dispatch auth check",
                status_code=401,
            ),
        ),
        status_code=401,
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
    runtime_state_dir = _require_runtime_state_dir(
        request.runtime_state_dir,
        context="NewSessionRunRequest",
    )
    if supported_builtin_stage(request.stage) is None:
        raise RuntimeConfigurationError(
            "RuntimeClient requires at least one supported built-in service candidate."
        )
    selected_stage = _select_builtin_stage(request.stage)
    if selected_stage.service == "codex":
        _validate_codex_stage(selected_stage)
        provider_state_dir_relpath, provider_state_dir = _codex_prepare_runtime_state(
            runtime_state_dir,
            role=request.role,
            session_namespace=request.session_namespace,
        )
        _codex_seed_auth(provider_state_dir)
        recovered_thread_id = _recover_codex_rollout_thread_id(provider_state_dir)
        if _codex_is_resumable(provider_state_dir) and recovered_thread_id is not None:
            return _run_builtin_resumed_session(
                ResumedSessionRunRequest(
                    prompt=request.prompt,
                    worktree=cast(Any, request.worktree),
                    runtime_state_dir=runtime_state_dir,
                    continuation=_build_codex_continuation(
                        model=selected_stage.model,
                        effort=selected_stage.effort,
                        tool_access=request.tool_access,
                        provider_session_id=recovered_thread_id,
                        provider_state_dir_relpath=provider_state_dir_relpath,
                    ),
                    role=request.role,
                    provider_auth=request.provider_auth,
                    usage_limit_scope=request.usage_limit_scope,
                    session_namespace=request.session_namespace,
                ),
                provider_invocation_adapter=invocation_adapter,
            )
        provider_session_id: str | None = None
        try:
            invocation_result = _invoke_codex_new_session_provider(
                provider_invocation_adapter=invocation_adapter,
                request=request,
                stage=selected_stage,
                provider_state_dir=provider_state_dir,
            )
            provider_session_id = _active_codex_provider_session_id_from_result(
                invocation_result,
                fallback_provider_session_id=provider_session_id,
            )
            result_text = invocation_result.output
            usage = invocation_result.usage
        except (UsageLimitError, RetryableProviderFailureError) as exc:
            provider_session_id = _active_codex_provider_session_id_from_failure(
                exc,
                fallback_provider_session_id=provider_session_id,
            )
            exc.continuation = (
                _build_codex_continuation(
                    model=selected_stage.model,
                    effort=selected_stage.effort,
                    tool_access=request.tool_access,
                    provider_session_id=provider_session_id,
                    provider_state_dir_relpath=provider_state_dir_relpath,
                )
                if exc.invocation_progress is InvocationProgress.STARTED
                and provider_session_id is not None
                else None
            )
            raise
        return RuntimeOutcome.completed(
            output=result_text,
            result=SessionRunResult(
                output=result_text,
                runtime_metadata=SessionRuntimeMetadata(
                    service_name="codex",
                    provider_session_id=provider_session_id,
                    run_kind=RunKind.FRESH,
                    session_namespace=request.session_namespace,
                    exact_transcript_match=False,
                ),
                continuation=(
                    _build_codex_continuation(
                        model=selected_stage.model,
                        effort=selected_stage.effort,
                        tool_access=request.tool_access,
                        provider_session_id=provider_session_id,
                        provider_state_dir_relpath=provider_state_dir_relpath,
                    )
                    if provider_session_id is not None
                    else None
                ),
            ),
            usage=usage,
        )
    if selected_stage.service == "claude":
        provider_state_dir_relpath, provider_state_dir = _claude_prepare_runtime_state(
            runtime_state_dir,
            role=request.role,
            session_namespace=request.session_namespace,
        )
        if _claude_is_resumable(provider_state_dir):
            return _run_builtin_resumed_session(
                ResumedSessionRunRequest(
                    prompt=request.prompt,
                    worktree=cast(Any, request.worktree),
                    runtime_state_dir=runtime_state_dir,
                    continuation=_build_claude_continuation(
                        model=selected_stage.model,
                        effort=selected_stage.effort,
                        tool_access=request.tool_access,
                        provider_session_id=_new_provider_session_id(),
                        provider_state_dir_relpath=provider_state_dir_relpath,
                    ),
                    role=request.role,
                    provider_auth=request.provider_auth,
                    usage_limit_scope=request.usage_limit_scope,
                    session_namespace=request.session_namespace,
                    logs_dir=request.logs_dir,
                ),
                provider_invocation_adapter=invocation_adapter,
            )
        _validate_claude_stage(selected_stage)
        _require_claude_auth(request.provider_auth)
        provider_session_id = _new_provider_session_id()
        run_kind = RunKind.FRESH
        exact_transcript_match = False
    elif selected_stage.service == "opencode":
        provider_state_dir_relpath, provider_state_dir = (
            _opencode_prepare_runtime_state(
                runtime_state_dir,
                role=request.role,
                session_namespace=request.session_namespace,
            )
        )
        _validate_opencode_stage(selected_stage)
        _require_opencode_auth(request.provider_auth)
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
    try:
        if selected_stage.service == "claude":
            invocation_result = _invoke_claude_new_session_provider(
                provider_invocation_adapter=invocation_adapter,
                request=request,
                stage=selected_stage,
                provider_state_dir=provider_state_dir,
                run_kind=run_kind,
                provider_session_id=provider_session_id,
            )
        else:
            invocation_result, provider_session_id = (
                _invoke_opencode_new_session_provider(
                    provider_invocation_adapter=invocation_adapter,
                    request=request,
                    stage=selected_stage,
                    provider_state_dir=provider_state_dir,
                    run_kind=run_kind,
                    provider_session_id=provider_session_id,
                )
            )
            _persist_opencode_session_id(provider_state_dir, provider_session_id)
    except (UsageLimitError, RetryableProviderFailureError) as exc:
        observed_failure_provider_session_id = (
            provider_invocation_failure_provider_session_id(exc)
        )
        if observed_failure_provider_session_id is not None:
            provider_session_id = observed_failure_provider_session_id
        exc.continuation = None
        if (
            exc.invocation_progress is InvocationProgress.STARTED
            and provider_session_id is not None
        ):
            exc.continuation = (
                _build_claude_continuation(
                    model=selected_stage.model,
                    effort=selected_stage.effort,
                    tool_access=request.tool_access,
                    provider_session_id=provider_session_id,
                    provider_state_dir_relpath=provider_state_dir_relpath,
                )
                if selected_stage.service == "claude"
                else _build_opencode_continuation(
                    model=selected_stage.model,
                    effort=selected_stage.effort,
                    tool_access=request.tool_access,
                    provider_session_id=provider_session_id,
                    provider_state_dir_relpath=provider_state_dir_relpath,
                    exact_transcript_match=exact_transcript_match,
                )
            )
        raise
    if selected_stage.service == "claude":
        provider_session_id = (
            invocation_result.provider_session_id or provider_session_id
        )
    assert provider_session_id is not None
    result_text = invocation_result.output
    usage = invocation_result.usage
    return RuntimeOutcome.completed(
        output=result_text,
        result=SessionRunResult(
            output=result_text,
            runtime_metadata=SessionRuntimeMetadata(
                service_name=selected_stage.service,
                provider_session_id=provider_session_id,
                run_kind=run_kind,
                session_namespace=request.session_namespace,
                exact_transcript_match=exact_transcript_match,
            ),
            continuation=(
                _build_claude_continuation(
                    model=selected_stage.model,
                    effort=selected_stage.effort,
                    tool_access=request.tool_access,
                    provider_session_id=provider_session_id,
                    provider_state_dir_relpath=provider_state_dir_relpath,
                )
                if selected_stage.service == "claude"
                else _build_opencode_continuation(
                    model=selected_stage.model,
                    effort=selected_stage.effort,
                    tool_access=request.tool_access,
                    provider_session_id=provider_session_id,
                    provider_state_dir_relpath=provider_state_dir_relpath,
                    exact_transcript_match=exact_transcript_match,
                )
            ),
        ),
        usage=usage,
    )


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
    runtime_state_dir = _require_runtime_state_dir(
        request.runtime_state_dir,
        context="ResumedSessionRunRequest",
    )
    continuation = request.continuation
    if continuation is None:
        raise RuntimeConfigurationError(
            "RuntimeClient resumed-session execution requires a continuation."
        )
    provider_session_id: str | None
    if continuation.selected_service == "codex":
        _validate_codex_stage(
            StageSelection(
                service="codex",
                model=request.model,
                effort=request.effort,
            )
        )
        try:
            provider_resume_state = read_portable_continuation_payload(
                continuation
            ).provider_resume_state
        except TypeError as exc:
            raise RuntimeConfigurationError(str(exc)) from exc
        provider_state_dir_relpath = cast(
            str | None,
            provider_resume_state.get("provider_state_dir_relpath"),
        )
        if not provider_state_dir_relpath:
            raise RuntimeConfigurationError(
                "Codex continuation is missing `provider_state_dir_relpath`."
            )
        provider_state_dir = runtime_state_dir / provider_state_dir_relpath
        provider_state_dir.mkdir(parents=True, exist_ok=True)
        _codex_seed_auth(provider_state_dir)
        provider_session_id = _resolve_recoverable_codex_session_id(
            provider_state_dir=provider_state_dir,
            provider_session_id=cast(
                str | None,
                provider_resume_state.get("provider_session_id"),
            ),
        )
        run_kind = RunKind.RESUME
        active_provider_session_id: str | None = provider_session_id
        try:
            invocation_result = _invoke_codex_resumed_session_provider(
                provider_invocation_adapter=invocation_adapter,
                provider_session_id=provider_session_id,
                request=request,
                provider_state_dir=provider_state_dir,
            )
            active_provider_session_id = _active_codex_provider_session_id_from_result(
                invocation_result,
                fallback_provider_session_id=provider_session_id,
            )
            result_text = invocation_result.output
            usage = invocation_result.usage
        except (UsageLimitError, RetryableProviderFailureError) as exc:
            active_provider_session_id = _active_codex_provider_session_id_from_failure(
                exc,
                fallback_provider_session_id=active_provider_session_id,
            )
            exc.continuation = (
                _build_codex_continuation(
                    model=request.model,
                    effort=request.effort,
                    tool_access=request.tool_access,
                    provider_session_id=active_provider_session_id,
                    provider_state_dir_relpath=provider_state_dir_relpath,
                )
                if exc.invocation_progress is InvocationProgress.STARTED
                and active_provider_session_id is not None
                else None
            )
            raise
        return RuntimeOutcome.completed(
            output=result_text,
            result=SessionRunResult(
                output=result_text,
                runtime_metadata=SessionRuntimeMetadata(
                    service_name="codex",
                    provider_session_id=active_provider_session_id,
                    run_kind=run_kind,
                    session_namespace=request.session_namespace,
                    exact_transcript_match=False,
                ),
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
            ),
            usage=usage,
        )
    if continuation.selected_service not in {"claude", "opencode"}:
        raise RuntimeConfigurationError(
            "RuntimeClient session-backed execution is only implemented for Claude, Codex, and OpenCode."
        )
    if continuation.selected_service == "claude":
        _require_claude_auth(request.provider_auth)
    else:
        _require_opencode_auth(request.provider_auth)
    try:
        provider_resume_state = read_portable_continuation_payload(
            continuation
        ).provider_resume_state
    except TypeError as exc:
        raise RuntimeConfigurationError(str(exc)) from exc
    provider_state_dir_relpath = cast(
        str | None,
        provider_resume_state.get("provider_state_dir_relpath"),
    )
    if not provider_state_dir_relpath:
        raise RuntimeConfigurationError(
            f"{continuation.selected_service.capitalize()} continuation is missing `provider_state_dir_relpath`."
        )
    provider_session_id = cast(
        str | None,
        provider_resume_state.get("provider_session_id"),
    )
    provider_state_dir = runtime_state_dir / provider_state_dir_relpath
    provider_state_dir.mkdir(parents=True, exist_ok=True)
    state_dir_session_id = (
        _load_opencode_state_dir_session_id(provider_state_dir)
        if continuation.selected_service == "opencode"
        else None
    )
    if continuation.selected_service == "opencode":
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
    else:
        if not provider_session_id:
            provider_session_id = _new_provider_session_id()
        exact_transcript_match = False
        run_kind = _claude_run_kind_for_state_dir(provider_state_dir)
    prompt_path = request.worktree.host_path / ".pycastle_prompt"
    invocation_log = _start_invocation_log(
        logs_dir=request.logs_dir,
        role=request.role,
    )

    def _reduce_opencode_session_output(
        lines: list[str],
    ) -> tuple[str, ProviderUsage | None]:
        return (
            reduce_text_output_events(
                _parse_opencode_events(lines),
                lambda _turn: None,
                provider="opencode",
            ),
            None,
        )

    def _reduce_logged_opencode_session_output(
        lines: list[str],
        work_invocation_log: WorkInvocationLog,
    ) -> tuple[str, ProviderUsage | None]:
        return (
            reduce_text_output_events(
                _parse_opencode_events(
                    lines,
                    on_provider_session_id=work_invocation_log.record_provider_session_id,
                ),
                lambda _turn: None,
                provider="opencode",
            ),
            None,
        )

    try:
        if continuation.selected_service == "claude":
            command = _claude_command(
                model=request.model,
                effort=request.effort,
                tool_access=request.tool_access,
                prompt_path=prompt_path,
                run_kind=run_kind,
                session_uuid=provider_session_id,
            )
            environment = _claude_env(
                auth=request.provider_auth,
                state_dir_container_path=str(provider_state_dir),
            )
            reduce_output = _reduce_claude_stream
            reduce_logged_output = None
            extract_provider_session_id = None
        else:
            command = _opencode_command(
                model=request.model,
                effort=request.effort,
                run_kind=run_kind,
                session_uuid=provider_session_id,
            )
            environment = _opencode_env(
                auth=request.provider_auth,
                state_dir_container_path=str(provider_state_dir),
            )
            reduce_output = _reduce_opencode_session_output
            reduce_logged_output = _reduce_logged_opencode_session_output
            extract_provider_session_id = _extract_opencode_provider_session_id
        invocation_result = _invoke_provider(
            provider_invocation_adapter=invocation_adapter,
            command=command,
            worktree=request.worktree.host_path,
            environment=environment,
            prompt_content=request.prompt,
            prompt_path=prompt_path,
            cleanup_prompt_path=True,
            run_kind=run_kind,
            role=request.role,
            usage_limit_scope=request.usage_limit_scope,
            provider_session_id=provider_session_id,
            reduce_output=reduce_output,
            invocation_log=invocation_log,
            reduce_logged_output=reduce_logged_output,
            extract_provider_session_id=extract_provider_session_id,
        )
        if continuation.selected_service == "opencode":
            provider_session_id = invocation_result.provider_session_id
            assert provider_session_id is not None
            exact_transcript_match = _opencode_exact_transcript_match(
                saved_exact_transcript_match=saved_exact_transcript_match,
                provider_session_id=provider_session_id,
                state_dir_session_id=state_dir_session_id,
            )
            _persist_opencode_session_id(provider_state_dir, provider_session_id)
    except (UsageLimitError, RetryableProviderFailureError) as exc:
        observed_failure_provider_session_id = (
            provider_invocation_failure_provider_session_id(exc)
        )
        if observed_failure_provider_session_id is not None:
            provider_session_id = observed_failure_provider_session_id
        if continuation.selected_service == "opencode":
            provider_session_id = observed_failure_provider_session_id
            exact_transcript_match = _opencode_exact_transcript_match(
                saved_exact_transcript_match=saved_exact_transcript_match,
                provider_session_id=provider_session_id,
                state_dir_session_id=state_dir_session_id,
            )
        exc.continuation = (
            (
                _build_claude_continuation(
                    model=request.model,
                    effort=request.effort,
                    tool_access=request.tool_access,
                    provider_session_id=provider_session_id,
                    provider_state_dir_relpath=provider_state_dir_relpath,
                )
                if continuation.selected_service == "claude"
                else _build_opencode_continuation(
                    model=request.model,
                    effort=request.effort,
                    tool_access=request.tool_access,
                    provider_session_id=provider_session_id,
                    provider_state_dir_relpath=provider_state_dir_relpath,
                    exact_transcript_match=exact_transcript_match,
                )
            )
            if exc.invocation_progress is InvocationProgress.STARTED
            and provider_session_id is not None
            else None
        )
        raise
    if continuation.selected_service == "claude":
        provider_session_id = (
            invocation_result.provider_session_id or provider_session_id
        )
    assert provider_session_id is not None
    result_text = invocation_result.output
    usage = invocation_result.usage
    return RuntimeOutcome.completed(
        output=result_text,
        result=SessionRunResult(
            output=result_text,
            runtime_metadata=SessionRuntimeMetadata(
                service_name=continuation.selected_service,
                provider_session_id=provider_session_id,
                run_kind=run_kind,
                session_namespace=request.session_namespace,
                exact_transcript_match=exact_transcript_match,
            ),
            continuation=(
                _build_claude_continuation(
                    model=request.model,
                    effort=request.effort,
                    tool_access=request.tool_access,
                    provider_session_id=provider_session_id,
                    provider_state_dir_relpath=provider_state_dir_relpath,
                )
                if continuation.selected_service == "claude"
                else _build_opencode_continuation(
                    model=request.model,
                    effort=request.effort,
                    tool_access=request.tool_access,
                    provider_session_id=provider_session_id,
                    provider_state_dir_relpath=provider_state_dir_relpath,
                    exact_transcript_match=exact_transcript_match,
                )
            ),
        ),
        usage=usage,
    )
