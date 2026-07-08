from __future__ import annotations

import dataclasses
import json
import logging
import re
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from ._builtin_provider_agent_event_building import (
    build_claude_agent_event,
    build_codex_agent_event,
    build_opencode_agent_event,
)
from ._builtin_provider_parsed_output import (
    _MONTH_ABBREVIATIONS,
    classify_codex_invocation_progress,
    extract_codex_provider_session_id,
    parse_claude_event,
    parse_claude_usage,
    parse_codex_event,
    parse_codex_usage,
)
from .contracts import (
    AssistantTurn,
    CredentialFailure,
    HardError,
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


def reduce_claude_stream(
    lines: list[str],
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


def codex_built_in_provider_stream_interpretation() -> (
    BuiltInProviderStreamInterpretation
):
    return BuiltInProviderStreamInterpretation(
        reduce_output=reduce_codex_stream,
        build_agent_event=build_codex_agent_event,
        classify_invocation_progress=classify_codex_invocation_progress,
        extract_provider_session_id=extract_codex_provider_session_id,
    )


def parse_opencode_reset_time(retry_text: object) -> datetime | None:
    if not isinstance(retry_text, str):
        return None
    match = _OPENCODE_RESET_PATTERN.search(retry_text)
    if match is None:
        return None
    month = _MONTH_ABBREVIATIONS.get(match.group("month").lower())
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
