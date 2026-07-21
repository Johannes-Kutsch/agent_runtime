from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable
from typing import Any

from ._builtin_provider_agent_event_building import (
    build_claude_agent_event,
    build_codex_agent_event,
    build_opencode_agent_event,
)
from ._builtin_provider_parsed_output import (
    classify_codex_invocation_progress,
    classify_opencode_invocation_progress,
    classify_opencode_output_line,
    extract_codex_provider_session_id,
    extract_opencode_provider_session_id,
    parse_claude_event,
    parse_claude_usage,
    parse_codex_event,
    parse_codex_usage,
    parse_opencode_events,
)
from .contracts import (
    AssistantTurn,
    Result,
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


# ---------------------------------------------------------------------------
# Shared skeleton
# ---------------------------------------------------------------------------

_ParseLines = Callable[[list[str]], tuple[list[Any], ProviderUsage | None]]
_Repair = Callable[
    [ProviderUnavailableError | UsageLimitError, ProviderUsage | None], None
]
_InvocationContext = tuple[_ParseLines, _Repair]


@dataclasses.dataclass(frozen=True, slots=True)
class _ProviderStreamBundle:
    provider: str
    build_agent_event: Callable[[str], AgentEvent]
    # Returns fresh (parse_lines, repair) sharing per-invocation mutable state.
    begin: Callable[[], _InvocationContext]


def _reduce_provider_stream(
    lines: list[str],
    on_live_output: Callable[[AgentEvent], None] | None,
    bundle: _ProviderStreamBundle,
) -> tuple[str, ProviderUsage | None]:
    parse_lines, repair = bundle.begin()
    parsed_events, usage = parse_lines(lines)
    if on_live_output is not None:
        for line in lines:
            emit_built_in_provider_live_output_event(
                bundle.build_agent_event(line), on_live_output
            )
    try:
        output = reduce_text_output_events(
            parsed_events,
            lambda _turn, _raw: None,
            provider=bundle.provider,
        )
    except (ProviderUnavailableError, UsageLimitError) as exc:
        repair(exc, usage)
        raise
    return output, usage


# ---------------------------------------------------------------------------
# Per-service bundles
# ---------------------------------------------------------------------------


def _usage_repair(
    exc: ProviderUnavailableError | UsageLimitError,
    usage: ProviderUsage | None,
) -> None:
    if not is_built_in_provider_live_output_exception(exc) and exc.usage is None:
        exc.usage = usage


def _begin_claude() -> _InvocationContext:
    def _parse(lines: list[str]) -> tuple[list[Any], ProviderUsage | None]:
        usage: ProviderUsage | None = None
        events: list[Any] = []
        for line in lines:
            usage = _merge_provider_usage(usage, parse_claude_usage(line))
            events.extend(parse_claude_event(line))
        return events, usage

    return _parse, _usage_repair


_CLAUDE_BUNDLE = _ProviderStreamBundle(
    provider="claude",
    build_agent_event=build_claude_agent_event,
    begin=_begin_claude,
)


def _begin_codex() -> _InvocationContext:
    def _parse(lines: list[str]) -> tuple[list[Any], ProviderUsage | None]:
        usage: ProviderUsage | None = None
        events: list[Any] = []
        for line in lines:
            usage = _merge_provider_usage(usage, parse_codex_usage(line))
            events.extend(parse_codex_event(line))
        return events, usage

    return _parse, _usage_repair


_CODEX_BUNDLE = _ProviderStreamBundle(
    provider="codex",
    build_agent_event=build_codex_agent_event,
    begin=_begin_codex,
)


def _make_opencode_begin(
    on_provider_session_id: Callable[[str], None] | None = None,
) -> Callable[[], _InvocationContext]:
    def _begin() -> _InvocationContext:
        observed: list[str | None] = [None]

        def _record(session_id: str) -> None:
            observed[0] = session_id
            if on_provider_session_id is not None:
                on_provider_session_id(session_id)

        def _parse(lines: list[str]) -> tuple[list[Any], ProviderUsage | None]:
            return (
                parse_opencode_events(lines, on_provider_session_id=_record),
                None,
            )

        def _repair(
            exc: ProviderUnavailableError | UsageLimitError,
            _usage: ProviderUsage | None,
        ) -> None:
            if observed[0] is not None:
                exc.invocation_progress = InvocationProgress.STARTED

        return _parse, _repair

    return _begin


_OPENCODE_BUNDLE = _ProviderStreamBundle(
    provider="opencode",
    build_agent_event=build_opencode_agent_event,
    begin=_make_opencode_begin(),
)


# ---------------------------------------------------------------------------
# Private per-service reducer wrappers
# ---------------------------------------------------------------------------


def _reduce_claude_stream(
    lines: list[str],
    on_live_output: Callable[[AgentEvent], None] | None = None,
) -> tuple[str, ProviderUsage | None]:
    return _reduce_provider_stream(lines, on_live_output, _CLAUDE_BUNDLE)


def _reduce_codex_stream(
    lines: list[str],
    on_live_output: Callable[[AgentEvent], None] | None = None,
) -> tuple[str, ProviderUsage | None]:
    return _reduce_provider_stream(lines, on_live_output, _CODEX_BUNDLE)


# ---------------------------------------------------------------------------
# Public constructor entrypoints
# ---------------------------------------------------------------------------


def codex_built_in_provider_stream_interpretation() -> (
    BuiltInProviderStreamInterpretation
):
    return BuiltInProviderStreamInterpretation(
        reduce_output=_reduce_codex_stream,
        build_agent_event=build_codex_agent_event,
        classify_invocation_progress=classify_codex_invocation_progress,
        extract_provider_session_id=extract_codex_provider_session_id,
    )


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
            session_id, is_terminal, is_json_object = classify_opencode_output_line(
                line
            )
            if not is_json_object:
                continue
            if (
                session_id is not None
                and session_id != seen_session_id
                and on_provider_session_id is not None
            ):
                seen_session_id = session_id
                on_provider_session_id(session_id)
            emit_built_in_provider_live_output_event(
                stream_interpretation.build_agent_event(line),
                on_live_output,
            )
            if is_terminal:
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

    _lifecycle_bundle = _ProviderStreamBundle(
        provider="opencode",
        build_agent_event=build_opencode_agent_event,
        begin=_make_opencode_begin(on_provider_session_id=_record_provider_session_id),
    )

    def _reduce_output(lines: list[str]) -> tuple[str, ProviderUsage | None]:
        reducer = reduce_output or (
            lambda output_lines: _reduce_provider_stream(
                output_lines, None, _lifecycle_bundle
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


def opencode_built_in_provider_stream_interpretation(
    *,
    reduce_output: BuiltInProviderOutputReducer | None = None,
    extract_provider_session_id: BuiltInProviderSessionIdExtractor | None = None,
) -> BuiltInProviderStreamInterpretation:
    return BuiltInProviderStreamInterpretation(
        reduce_output=reduce_output
        or (lambda lines: _reduce_provider_stream(lines, None, _OPENCODE_BUNDLE)),
        build_agent_event=build_opencode_agent_event,
        classify_invocation_progress=classify_opencode_invocation_progress,
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
        reduce_output=_reduce_claude_stream,
        build_agent_event=build_claude_agent_event,
        classify_invocation_progress=_classify_progress,
    )
