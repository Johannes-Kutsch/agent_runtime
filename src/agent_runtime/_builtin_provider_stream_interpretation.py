from __future__ import annotations

import dataclasses
from collections.abc import Callable

from ._runtime_lifecycle import AgentEvent, ProviderUsage
from .invocation_progress import InvocationProgress

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


def emit_built_in_provider_live_output_event(
    event: AgentEvent,
    on_live_output: Callable[[AgentEvent], None] | None,
) -> None:
    if on_live_output is None:
        return
    try:
        on_live_output(event)
    except Exception as exc:
        setattr(exc, "_is_live_output_exception", True)
        raise


def is_built_in_provider_live_output_exception(exc: BaseException) -> bool:
    return bool(getattr(exc, "_is_live_output_exception", False))


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
