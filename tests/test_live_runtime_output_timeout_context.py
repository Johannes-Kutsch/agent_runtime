from __future__ import annotations

import agent_runtime._live_runtime_output_timeout_context as timeout_context_module
import agent_runtime.runtime as prompt_runtime


def _event(message: str) -> prompt_runtime.AgentEvent:
    return prompt_runtime.AgentEvent(
        type="other",
        display_message=message,
        raw_provider_output=message,
    )


def test_timeout_context_keeps_live_runtime_output_live_only() -> None:
    observed_events: list[prompt_runtime.AgentEvent] = []
    context = timeout_context_module._LiveRuntimeOutputTimeoutContext(
        on_live_output=observed_events.append,
        timeout_seconds=0,
    )

    wrapped_on_live_output = context.wrapped_on_live_output
    assert wrapped_on_live_output is not None

    first_event = _event("first")
    second_event = _event("second")
    wrapped_on_live_output(first_event)
    wrapped_on_live_output(second_event)

    assert observed_events == [first_event, second_event]
    assert not hasattr(context, "__dict__")
