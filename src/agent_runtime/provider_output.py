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
    RetryableProviderFailureError,
    TransientAgentError,
    UsageLimitError,
)
from .invocation_progress import InvocationProgress


def reduce_text_output_events(
    events: Iterable[ParsedTurn],
    on_turn: Callable[[str], None],
    on_tokens: Callable[[int], None] | None = None,
    *,
    provider: str,
) -> str:
    result_text: str | None = None
    collected_turns: list[str] = []
    invocation_progress = InvocationProgress.NOT_STARTED
    for event in events:
        if isinstance(event, UsageLimit):
            raise UsageLimitError(
                reset_time=event.reset_time,
                raw_message=event.raw_message,
                service_name=provider,
                is_permanent=event.is_permanent,
                invocation_progress=invocation_progress,
            )
        if isinstance(event, TransientError):
            if event.classification == "retryable":
                raise RetryableProviderFailureError(
                    message=event.raw_message,
                    status_code=event.status_code,
                    service_name=provider,
                    classification=event.classification,
                    observations=event.observations,
                    invocation_progress=invocation_progress,
                )
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
            invocation_progress = InvocationProgress.STARTED
            continue
        if isinstance(event, Result):
            result_text = event.text
            invocation_progress = InvocationProgress.STARTED
            break
    if result_text is not None:
        return result_text
    return "\n".join(collected_turns)


__all__ = ["reduce_text_output_events"]
