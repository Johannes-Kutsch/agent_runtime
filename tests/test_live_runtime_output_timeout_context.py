from __future__ import annotations

import gc
import weakref
from typing import Callable

import pytest

import agent_runtime._live_runtime_output_timeout_context as timeout_context_module
import agent_runtime.runtime as prompt_runtime


def _event(message: str) -> prompt_runtime.AgentEvent:
    return prompt_runtime.AgentEvent(
        type="other",
        display_message=message,
        raw_provider_output=message,
    )


class _WatchdogProbe:
    def __init__(self, timeout_seconds: int) -> None:
        self.timeout_seconds = timeout_seconds
        self.events: list[str] = []
        self.timer_refreshed = False
        self.timeout_checked = False

    def start_monitoring(self) -> None:
        self.events.append("start_monitoring")

    def reset_timer(self) -> None:
        self.timer_refreshed = True
        self.events.append("reset_timer")

    def check_timeout(self) -> None:
        self.timeout_checked = True
        self.events.append("check_timeout")

    def stop_monitoring(self) -> None:
        self.events.append("stop_monitoring")


@pytest.fixture
def watchdog_probe_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> list[_WatchdogProbe]:
    probes: list[_WatchdogProbe] = []

    def create_probe(timeout_seconds: int) -> _WatchdogProbe:
        probe = _WatchdogProbe(timeout_seconds)
        probes.append(probe)
        return probe

    monkeypatch.setattr(
        timeout_context_module,
        "_IdleTimeoutWatchdog",
        create_probe,
    )
    return probes


@pytest.mark.parametrize("timeout_seconds", [0, -1])
def test_live_runtime_output_timeout_context_disables_idle_timeout_for_non_positive_timeout_values(
    timeout_seconds: int,
) -> None:
    observed_events: list[prompt_runtime.AgentEvent] = []
    observer = observed_events.append

    wrapped_on_live_output, timeout_context = (
        timeout_context_module._wrap_on_live_output_with_timeout(
            observer,
            timeout_seconds,
        )
    )

    assert wrapped_on_live_output is observer
    assert timeout_context is None
    assert wrapped_on_live_output is not None

    first_event = _event("first")
    second_event = _event("second")
    wrapped_on_live_output(first_event)
    wrapped_on_live_output(second_event)

    assert observed_events == [first_event, second_event]


def test_live_runtime_output_timeout_context_refreshes_idle_timeout_before_notifying_consumer_and_checks_after_return(
    watchdog_probe_factory: list[_WatchdogProbe],
) -> None:
    observed_events: list[prompt_runtime.AgentEvent] = []

    def observer(event: prompt_runtime.AgentEvent) -> None:
        probe = watchdog_probe_factory[0]
        assert probe.timer_refreshed is True
        assert probe.timeout_checked is False
        observed_events.append(event)
        probe.events.append("observer")

    wrapped_on_live_output, timeout_context = (
        timeout_context_module._wrap_on_live_output_with_timeout(observer, 7)
    )

    assert timeout_context is not None
    assert wrapped_on_live_output is not None
    wrapped_on_live_output(_event("heartbeat"))

    assert observed_events == [_event("heartbeat")]
    assert watchdog_probe_factory[0].events == [
        "start_monitoring",
        "reset_timer",
        "observer",
        "check_timeout",
    ]


def test_live_runtime_output_timeout_context_propagates_consumer_observer_exceptions_unchanged(
    watchdog_probe_factory: list[_WatchdogProbe],
) -> None:
    observer_failure = RuntimeError("observer failed")

    def observer(_: prompt_runtime.AgentEvent) -> None:
        raise observer_failure

    wrapped_on_live_output, timeout_context = (
        timeout_context_module._wrap_on_live_output_with_timeout(observer, 7)
    )

    assert timeout_context is not None
    assert wrapped_on_live_output is not None

    with pytest.raises(RuntimeError, match="observer failed") as excinfo:
        wrapped_on_live_output(_event("heartbeat"))

    assert excinfo.value is observer_failure
    assert watchdog_probe_factory[0].events == [
        "start_monitoring",
        "reset_timer",
    ]


def test_live_runtime_output_timeout_context_keeps_live_runtime_output_live_only(
    watchdog_probe_factory: list[_WatchdogProbe],
) -> None:
    observed_messages: list[str] = []

    wrapped_on_live_output, timeout_context = (
        timeout_context_module._wrap_on_live_output_with_timeout(
            lambda event: observed_messages.append(event.display_message),
            7,
        )
    )

    assert timeout_context is not None
    assert wrapped_on_live_output is not None

    event = _event("heartbeat")
    event_reference = weakref.ref(event)
    wrapped_on_live_output(event)
    del event
    gc.collect()

    assert observed_messages == ["heartbeat"]
    assert event_reference() is None


def test_live_runtime_output_timeout_context_stops_idle_timeout_monitoring_after_one_invocation(
    watchdog_probe_factory: list[_WatchdogProbe],
) -> None:
    observed_events: list[prompt_runtime.AgentEvent] = []

    result = timeout_context_module._run_with_live_runtime_output_timeout_context(
        observed_events.append,
        7,
        lambda on_live_output: (
            on_live_output(_event("heartbeat")) if on_live_output is not None else None
        ),
    )

    assert result is None
    assert observed_events == [_event("heartbeat")]
    assert watchdog_probe_factory[0].events == [
        "start_monitoring",
        "reset_timer",
        "check_timeout",
        "stop_monitoring",
    ]


def test_live_runtime_output_timeout_context_stops_idle_timeout_monitoring_when_one_invocation_raises(
    watchdog_probe_factory: list[_WatchdogProbe],
) -> None:
    invocation_failure = RuntimeError("invocation failed")

    def run_once(
        on_live_output: Callable[[prompt_runtime.AgentEvent], None] | None,
    ) -> None:
        if on_live_output is not None:
            on_live_output(_event("heartbeat"))
        raise invocation_failure

    with pytest.raises(RuntimeError, match="invocation failed") as excinfo:
        timeout_context_module._run_with_live_runtime_output_timeout_context(
            None,
            7,
            run_once,
        )

    assert excinfo.value is invocation_failure
    assert watchdog_probe_factory[0].events == [
        "start_monitoring",
        "reset_timer",
        "check_timeout",
        "stop_monitoring",
    ]


@pytest.mark.parametrize("timeout_seconds", [0, -1])
def test_live_runtime_output_timeout_context_leaves_one_invocation_unwrapped_when_idle_timeout_is_disabled(
    timeout_seconds: int,
) -> None:
    observed_events: list[prompt_runtime.AgentEvent] = []
    observer = observed_events.append

    def run_once(
        on_live_output: Callable[[prompt_runtime.AgentEvent], None] | None,
    ) -> str:
        assert on_live_output is observer
        on_live_output(_event("heartbeat"))
        return "completed"

    result = timeout_context_module._run_with_live_runtime_output_timeout_context(
        observer,
        timeout_seconds,
        run_once,
    )

    assert result == "completed"
    assert observed_events == [_event("heartbeat")]
