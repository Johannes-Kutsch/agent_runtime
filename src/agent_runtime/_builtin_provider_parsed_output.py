from __future__ import annotations

import json
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
    SessionGone,
    TransientError,
    UsageLimit,
)
from ._runtime_lifecycle import ProviderUsage

_MONTH_ABBREVIATIONS = {
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
_CLAUDE_SESSION_NOT_FOUND_PHRASE = "no conversation found with session id"


def _extract_claude_session_gone_message(event: dict[str, Any]) -> str | None:
    errors = event.get("errors")
    if not isinstance(errors, list):
        return None
    for error in errors:
        if not isinstance(error, dict):
            continue
        message = error.get("message")
        if (
            isinstance(message, str)
            and _CLAUDE_SESSION_NOT_FOUND_PHRASE in message.lower()
        ):
            return message
    return None


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
        month = _MONTH_ABBREVIATIONS.get(month_text.lower())
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
        session_gone_message = _extract_claude_session_gone_message(event)
        if session_gone_message is not None:
            return [SessionGone(raw_message=session_gone_message)]
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
