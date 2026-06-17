from __future__ import annotations

from collections.abc import Callable, Iterable

from .contracts import (
    AssistantTurn,
    CredentialFailure,
    HardError,
    ParsedTurn,
    PromptTokens,
    Result,
    TransientError,
    UnsupportedTokens,
    UsageLimit,
)
from .errors import (
    AgentCredentialFailureError,
    HardAgentError,
    TransientAgentError,
    UsageLimitError,
)


def reduce_text_output_events(
    events: Iterable[ParsedTurn],
    on_turn: Callable[[str], None],
    on_tokens: Callable[[int], None] | None = None,
    *,
    provider: str,
) -> str:
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


__all__ = ["reduce_text_output_events"]
