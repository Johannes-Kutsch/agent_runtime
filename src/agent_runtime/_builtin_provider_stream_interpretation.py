from __future__ import annotations

import dataclasses
import json
import logging
import re
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any, cast

from . import _time as _time_module
from .contracts import (
    AssistantTurn,
    CredentialFailure,
    HardError,
    PromptTokens,
    Result,
    TransientError,
    UsageLimit,
)
from .errors import ProviderUnavailableError, UsageLimitError
from ._live_runtime_output_exceptions import (
    is_live_runtime_output_exception,
    is_live_runtime_output_timeout_wrapper,
    mark_live_runtime_output_exception,
)
from .provider_output import reduce_text_output_events
from ._runtime_lifecycle import AgentEvent, ProviderUsage
from .invocation_progress import InvocationProgress

_log = logging.getLogger(__name__)

BuiltInProviderOutputReducer = Callable[[list[str]], tuple[str, ProviderUsage | None]]
BuiltInProviderAgentEventBuilder = Callable[[str], AgentEvent]
BuiltInProviderSessionIdExtractor = Callable[[list[str]], str | None]
BuiltInProviderInvocationProgressClassifier = Callable[[list[str]], InvocationProgress]


@dataclasses.dataclass(frozen=True, slots=True)
class BuiltInProviderStreamInterpretation:
    reduce_output: BuiltInProviderOutputReducer
    build_agent_event: BuiltInProviderAgentEventBuilder
    classify_invocation_progress: BuiltInProviderInvocationProgressClassifier
    extract_provider_session_id: BuiltInProviderSessionIdExtractor | None = None


class _ObservedOutputReducer:
    __slots__ = ("reduce_output", "consume_stdout_lines")

    def __init__(
        self,
        reduce_output: BuiltInProviderOutputReducer,
        consume_stdout_lines: Callable[[list[str]], None],
    ) -> None:
        self.reduce_output = reduce_output
        self.consume_stdout_lines = consume_stdout_lines

    def __call__(self, lines: list[str]) -> tuple[str, ProviderUsage | None]:
        return self.reduce_output(lines)


def emit_built_in_provider_live_output_event(
    event: AgentEvent,
    on_live_output: Callable[[AgentEvent], None] | None,
) -> None:
    if on_live_output is None:
        return
    try:
        on_live_output(event)
    except Exception as exc:
        if not is_live_runtime_output_timeout_wrapper(
            on_live_output
        ) and not is_live_runtime_output_exception(exc):
            mark_live_runtime_output_exception(exc)
        raise


def is_built_in_provider_live_output_exception(exc: BaseException) -> bool:
    return is_live_runtime_output_exception(exc)


def classify_built_in_provider_invocation_progress(
    interpretation: BuiltInProviderStreamInterpretation,
    lines: list[str],
    *,
    provider_session_id: str | None = None,
) -> InvocationProgress:
    if provider_session_id is not None:
        return InvocationProgress.STARTED
    if interpretation.extract_provider_session_id is not None:
        observed_provider_session_id = interpretation.extract_provider_session_id(lines)
        if observed_provider_session_id is not None:
            return InvocationProgress.STARTED
    return interpretation.classify_invocation_progress(lines)


def built_in_provider_invocation_started(
    interpretation: BuiltInProviderStreamInterpretation,
    lines: list[str],
    *,
    provider_session_id: str | None = None,
) -> bool:
    return (
        classify_built_in_provider_invocation_progress(
            interpretation,
            lines,
            provider_session_id=provider_session_id,
        )
        is InvocationProgress.STARTED
    )


def resolve_built_in_provider_session_id(
    interpretation: BuiltInProviderStreamInterpretation,
    lines: list[str],
    *,
    provider_session_id: str | None = None,
    fallback_provider_session_id: str | None = None,
) -> str | None:
    if interpretation.extract_provider_session_id is not None:
        observed_provider_session_id = interpretation.extract_provider_session_id(lines)
        if observed_provider_session_id is not None:
            return observed_provider_session_id
    if provider_session_id is not None:
        return provider_session_id
    return fallback_provider_session_id


_CLAUDE_MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}
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
_CODEX_HTTP_STATUS_RE = re.compile(
    r"\bstatus\s+(?P<status>\d{3})\b",
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


def is_claude_subscription_access_denial(event: dict[str, Any]) -> bool:
    result = event.get("result")
    return (
        event.get("is_error") is True
        and event.get("api_error_status") == 403
        and isinstance(result, str)
        and _CLAUDE_SUBSCRIPTION_ACCESS_DENIAL_PHRASE.lower() in result.lower()
    )


def parse_claude_reset_time(retry_text: object) -> datetime | None:
    if not isinstance(retry_text, str):
        return None
    import re

    claude_reset_pattern = re.compile(
        r"resets?\s+"
        r"(?:(?P<month>[A-Za-z]+)\s+(?P<day>\d{1,2}),\s+)?"
        r"(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?(?P<ampm>am|pm)\s+\(UTC\)",
        re.IGNORECASE,
    )
    match = claude_reset_pattern.search(retry_text)
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


def parse_claude_event(line: str) -> list[Any]:
    return parse_claude_event_with_dependencies(
        line,
        parse_claude_reset_time=parse_claude_reset_time,
        is_claude_subscription_access_denial=is_claude_subscription_access_denial,
    )


def parse_claude_event_with_dependencies(
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
        return [
            CredentialFailure(
                raw_message=line,
                service_name="claude",
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


def reduce_claude_stream(
    lines: list[str],
    on_live_output: Callable[[AgentEvent], None] | None = None,
) -> tuple[str, ProviderUsage | None]:
    return reduce_claude_stream_with_dependencies(
        lines,
        parse_claude_event=parse_claude_event,
        on_live_output=on_live_output,
    )


def reduce_claude_stream_with_dependencies(
    lines: list[str],
    *,
    parse_claude_event: Callable[[str], list[Any]],
    on_live_output: Callable[[AgentEvent], None] | None = None,
) -> tuple[str, ProviderUsage | None]:
    usage: ProviderUsage | None = None
    parsed_events: list[Any] = []
    for line in lines:
        usage = _merge_provider_usage(usage, parse_claude_usage(line))
        parsed_events.extend(parse_claude_event(line))
    if on_live_output is not None:
        for line in lines:
            emit_built_in_provider_live_output_event(
                build_claude_agent_event(line),
                on_live_output,
            )
    try:
        output = reduce_text_output_events(
            parsed_events,
            lambda _turn, _raw: None,
            provider="claude",
        )
    except (ProviderUnavailableError, UsageLimitError) as exc:
        if is_built_in_provider_live_output_exception(exc):
            raise
        if exc.usage is None:
            exc.usage = usage
        raise
    return output, usage


def parse_claude_usage(line: str) -> ProviderUsage | None:
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


def build_claude_agent_event(line: str) -> AgentEvent:
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return _other_event(line, "unparsed")
    if not isinstance(event, dict):
        return _other_event(line, "non_object")
    if event.get("type") == "assistant":
        message = event.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, list):
                text_parts: list[str] = []
                tool_blocks: list[dict[str, object]] = []
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    block_type = block.get("type")
                    if block_type == "text":
                        text = block.get("text")
                        if isinstance(text, str) and text.strip():
                            text_parts.append(text.strip())
                    elif block_type == "tool_use":
                        tool_blocks.append(cast(dict[str, object], block))
                if text_parts:
                    return _message_event(line, "\n\n".join(text_parts))
                if tool_blocks:
                    first_tool = tool_blocks[0]
                    tool_name = first_tool.get("name")
                    if not isinstance(tool_name, str) or not tool_name:
                        tool_name = "tool_use"
                    payload_value: object = (
                        first_tool.get("input")
                        if len(tool_blocks) == 1 and first_tool.get("input") is not None
                        else tool_blocks
                    )
                    return _tool_call_event(
                        line, tool_name, _raw_event_payload(payload_value)
                    )
    event_type = event.get("type")
    descriptor = event_type if isinstance(event_type, str) and event_type else "other"
    return _other_event(line, descriptor)


def _classify_codex_error_message(
    message: str,
) -> CredentialFailure | HardError | TransientError | None:
    lowered_message = message.lower()
    if "refresh_token_reused" in message:
        return CredentialFailure(
            raw_message=message,
            service_name="codex",
            status_code=401,
            classification="codex_auth_lineage_exhausted",
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
        )
    if _CODEX_GENERIC_AUTH_RE.search(message):
        return HardError(status_code=401, raw_message=message)
    match = _CODEX_HTTP_STATUS_RE.search(message)
    if match is None:
        return None
    status = int(match.group("status"))
    if status >= 500:
        return TransientError(status_code=status, raw_message=message)
    if 400 <= status < 500:
        return HardError(status_code=status, raw_message=message)
    return None


def parse_codex_reset_time(retry_text: object) -> datetime | None:
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


def _extract_codex_usage_limit(message: str) -> UsageLimit | None:
    if _CODEX_USAGE_LIMIT_SUBSTRING not in message:
        return None
    reset_time = parse_codex_reset_time(message)
    return UsageLimit(
        reset_time=reset_time,
        raw_message=None if reset_time is not None else message,
    )


def _codex_tool_payload(item: dict[str, object]) -> str:
    for key in ("arguments", "input", "payload"):
        value = item.get(key)
        if value is not None:
            return _raw_event_payload(value)
    return _raw_event_payload(item)


def build_codex_agent_event(line: str) -> AgentEvent:
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return _other_event(line, "unparsed")
    if not isinstance(event, dict):
        return _other_event(line, "non_object")
    event_type = event.get("type")
    if event_type in {"item.completed", "item.started"}:
        item = event.get("item")
        if isinstance(item, dict):
            item_type = item.get("type")
            if item_type == "agent_message":
                content = item.get("text")
                if content is None:
                    content = item.get("content") or ""
                if isinstance(content, str):
                    return _message_event(line, content)
            if isinstance(item_type, str):
                tool_name = item.get("name")
                if not isinstance(tool_name, str) or not tool_name:
                    tool_name = item_type
                return _tool_call_event(line, tool_name, _codex_tool_payload(item))
    descriptor = event_type if isinstance(event_type, str) and event_type else "other"
    return _other_event(line, descriptor)


def parse_codex_event(line: str) -> list[Any]:
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
        classified = _classify_codex_error_message(message)
        if classified is not None:
            _log.warning("codex turn.failed: %s", message)
            return [classified]
        return []
    if event_type == "error":
        message = event.get("message") or ""
        limit = _extract_codex_usage_limit(message)
        if limit is not None:
            return [limit]
        classified = _classify_codex_error_message(message)
        if classified is not None:
            _log.warning("codex error: %s", message)
            return [classified]
        return []
    return []


def parse_codex_usage(line: str) -> ProviderUsage | None:
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


def reduce_codex_stream(
    lines: list[str],
    on_live_output: Callable[[AgentEvent], None] | None = None,
) -> tuple[str, ProviderUsage | None]:
    usage: ProviderUsage | None = None
    parsed_events: list[Any] = []
    for line in lines:
        usage = _merge_provider_usage(usage, parse_codex_usage(line))
        parsed_events.extend(parse_codex_event(line))
    if on_live_output is not None:
        for line in lines:
            emit_built_in_provider_live_output_event(
                build_codex_agent_event(line),
                on_live_output,
            )
    try:
        output = reduce_text_output_events(
            parsed_events,
            lambda _turn, _raw: None,
            provider="codex",
        )
    except (ProviderUnavailableError, UsageLimitError) as exc:
        if is_built_in_provider_live_output_exception(exc):
            raise
        if exc.usage is None:
            exc.usage = usage
        raise
    return output, usage


def extract_codex_provider_session_id(lines: list[str]) -> str | None:
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


def codex_built_in_provider_stream_interpretation() -> (
    BuiltInProviderStreamInterpretation
):
    def _classify_progress(lines: list[str]) -> InvocationProgress:
        parsed_events = [event for line in lines for event in parse_codex_event(line)]
        if any(isinstance(event, (AssistantTurn, Result)) for event in parsed_events):
            return InvocationProgress.STARTED
        return InvocationProgress.NOT_STARTED

    return BuiltInProviderStreamInterpretation(
        reduce_output=reduce_codex_stream,
        build_agent_event=build_codex_agent_event,
        classify_invocation_progress=_classify_progress,
        extract_provider_session_id=extract_codex_provider_session_id,
    )


def parse_opencode_reset_time(retry_text: object) -> datetime | None:
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
    reset_time = parse_opencode_reset_time(message)
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


def parse_opencode_event(line: str) -> list[Any]:
    return parse_opencode_events([line])


def parse_opencode_events(
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


def build_opencode_agent_event(line: str) -> AgentEvent:
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return _other_event(line, "unparsed")
    if not isinstance(event, dict):
        return _other_event(line, "non_object")
    if event.get("type") == "text":
        part = event.get("part")
        if isinstance(part, dict):
            part_type = part.get("type")
            if part_type == "text":
                time = part.get("time")
                if isinstance(time, dict) and time.get("end") is not None:
                    text = part.get("text")
                    if isinstance(text, str):
                        stripped = text.strip()
                        if stripped:
                            return _message_event(line, stripped)
            if part_type == "tool":
                tool_name = part.get("name")
                if not isinstance(tool_name, str) or not tool_name:
                    tool_name = "tool"
                payload_value = (
                    part.get("input")
                    if part.get("input") is not None
                    else part.get("text", part)
                )
                return _tool_call_event(
                    line, tool_name, _raw_event_payload(payload_value)
                )
    if event.get("type") == "session.status":
        status = event.get("status")
        descriptor = "session.status"
        if isinstance(status, dict):
            status_type = status.get("type")
            if isinstance(status_type, str):
                descriptor = status_type
        return _other_event(line, descriptor)
    event_type = event.get("type")
    descriptor = event_type if isinstance(event_type, str) and event_type else "other"
    return _other_event(line, descriptor)


def observe_opencode_output(
    *,
    stream_interpretation: BuiltInProviderStreamInterpretation,
    on_live_output: Callable[[AgentEvent], None],
    on_provider_session_id: Callable[[str], None] | None = None,
) -> Callable[[list[str]], None]:
    seen_session_id: str | None = None
    is_complete = False

    def _observe_output_lines(lines: list[str]) -> None:
        nonlocal seen_session_id, is_complete
        if is_complete:
            return
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
            emit_built_in_provider_live_output_event(
                stream_interpretation.build_agent_event(line),
                on_live_output,
            )
            if event.get("type") == "session.status":
                status = event.get("status")
                if isinstance(status, dict) and status.get("type") == "idle":
                    is_complete = True
                    return
                continue
            if event.get("type") == "error":
                is_complete = True
                return

    return _observe_output_lines


def opencode_lifecycle_built_in_provider_stream_interpretation(
    *,
    on_live_output: Callable[[AgentEvent], None] | None = None,
    on_provider_session_id: Callable[[str], None] | None = None,
    fallback_provider_session_id: str | None = None,
    reduce_output: BuiltInProviderOutputReducer | None = None,
) -> BuiltInProviderStreamInterpretation:
    observed_provider_session_id: str | None = None

    def _record_provider_session_id(session_id: str) -> None:
        nonlocal observed_provider_session_id
        observed_provider_session_id = session_id
        if on_provider_session_id is not None:
            on_provider_session_id(session_id)

    def _reduce_output(lines: list[str]) -> tuple[str, ProviderUsage | None]:
        reducer = reduce_output or (
            lambda output_lines: reduce_opencode_stream(
                output_lines,
                on_provider_session_id=_record_provider_session_id,
            )
        )
        return reducer(lines)

    def _extract_provider_session_id(lines: list[str]) -> str | None:
        if observed_provider_session_id is not None:
            return observed_provider_session_id
        extracted_provider_session_id = extract_opencode_provider_session_id(lines)
        if extracted_provider_session_id is not None:
            return extracted_provider_session_id
        return fallback_provider_session_id

    interpretation = opencode_built_in_provider_stream_interpretation(
        reduce_output=_reduce_output,
        extract_provider_session_id=_extract_provider_session_id,
    )
    if on_live_output is None:
        return interpretation
    return dataclasses.replace(
        interpretation,
        reduce_output=_ObservedOutputReducer(
            reduce_output=interpretation.reduce_output,
            consume_stdout_lines=observe_opencode_output(
                stream_interpretation=interpretation,
                on_live_output=on_live_output,
                on_provider_session_id=_record_provider_session_id,
            ),
        ),
    )


def extract_opencode_provider_session_id(lines: list[str]) -> str | None:
    provider_session_id: str | None = None

    def _record_provider_session_id(session_id: str) -> None:
        nonlocal provider_session_id
        provider_session_id = session_id

    parse_opencode_events(
        lines,
        on_provider_session_id=_record_provider_session_id,
    )
    return provider_session_id


def reduce_opencode_stream(
    lines: list[str],
    on_live_output: Callable[[AgentEvent], None] | None = None,
    *,
    on_provider_session_id: Callable[[str], None] | None = None,
) -> tuple[str, ProviderUsage | None]:
    observed_provider_session_id: str | None = None

    def _record_provider_session_id(session_id: str) -> None:
        nonlocal observed_provider_session_id
        observed_provider_session_id = session_id
        if on_provider_session_id is not None:
            on_provider_session_id(session_id)

    if on_live_output is not None:
        for line in lines:
            emit_built_in_provider_live_output_event(
                build_opencode_agent_event(line),
                on_live_output,
            )
    try:
        output = reduce_text_output_events(
            parse_opencode_events(
                lines,
                on_provider_session_id=_record_provider_session_id,
            ),
            lambda _turn, _raw: None,
            provider="opencode",
        )
    except (ProviderUnavailableError, UsageLimitError) as exc:
        if observed_provider_session_id is not None:
            exc.invocation_progress = InvocationProgress.STARTED
        raise
    return output, None


def opencode_built_in_provider_stream_interpretation(
    *,
    reduce_output: BuiltInProviderOutputReducer | None = None,
    extract_provider_session_id: BuiltInProviderSessionIdExtractor | None = None,
) -> BuiltInProviderStreamInterpretation:
    def _classify_progress(lines: list[str]) -> InvocationProgress:
        parsed_events = parse_opencode_events(lines)
        if any(isinstance(event, (AssistantTurn, Result)) for event in parsed_events):
            return InvocationProgress.STARTED
        return InvocationProgress.NOT_STARTED

    return BuiltInProviderStreamInterpretation(
        reduce_output=reduce_output or (lambda lines: reduce_opencode_stream(lines)),
        build_agent_event=build_opencode_agent_event,
        classify_invocation_progress=_classify_progress,
        extract_provider_session_id=(
            extract_provider_session_id or extract_opencode_provider_session_id
        ),
    )


def claude_built_in_provider_stream_interpretation() -> (
    BuiltInProviderStreamInterpretation
):
    def _classify_progress(lines: list[str]) -> InvocationProgress:
        parsed_events = [event for line in lines for event in parse_claude_event(line)]
        if any(isinstance(event, (AssistantTurn, Result)) for event in parsed_events):
            return InvocationProgress.STARTED
        return InvocationProgress.NOT_STARTED

    return BuiltInProviderStreamInterpretation(
        reduce_output=reduce_claude_stream,
        build_agent_event=build_claude_agent_event,
        classify_invocation_progress=_classify_progress,
    )
