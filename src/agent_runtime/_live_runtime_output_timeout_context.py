from __future__ import annotations

import threading
from datetime import datetime
from typing import Callable

from . import _time as _time_module
from ._runtime_lifecycle import AgentEvent
from .errors import AgentTimeoutError


class _IdleTimeoutWatchdog:
    def __init__(self, timeout_seconds: int) -> None:
        self.timeout_seconds = timeout_seconds
        self._lock = threading.Lock()
        self._start_time: datetime | None = None
        self._last_event_time: datetime | None = None
        self._stop_event = threading.Event()
        self._timeout_occurred = False

    def reset_timer(self) -> None:
        with self._lock:
            self._last_event_time = _time_module.now_local()

    def start_monitoring(self) -> None:
        with self._lock:
            self._start_time = _time_module.now_local()
            self._last_event_time = self._start_time
        self._stop_event.clear()
        thread = threading.Thread(target=self._monitor_loop, daemon=True)
        thread.start()

    def stop_monitoring(self) -> None:
        self._stop_event.set()

    def _monitor_loop(self) -> None:
        while not self._stop_event.is_set():
            with self._lock:
                if self._last_event_time is not None:
                    elapsed = (
                        _time_module.now_local() - self._last_event_time
                    ).total_seconds()
                    if elapsed > self.timeout_seconds:
                        self._timeout_occurred = True
                        return
            self._stop_event.wait(timeout=0.1)

    def check_timeout(self) -> None:
        with self._lock:
            if self._timeout_occurred:
                raise AgentTimeoutError(
                    "Idle timeout: no Agent Event within configured window"
                )


class _LiveRuntimeOutputTimeoutContext:
    def __init__(
        self,
        on_live_output: Callable[[AgentEvent], None] | None,
        timeout_seconds: int,
    ) -> None:
        self._on_live_output = on_live_output
        self._watchdog = (
            None if timeout_seconds <= 0 else _IdleTimeoutWatchdog(timeout_seconds)
        )
        if self._watchdog is not None:
            self._watchdog.start_monitoring()

    @property
    def wrapped_on_live_output(self) -> Callable[[AgentEvent], None] | None:
        if self._watchdog is None:
            return self._on_live_output
        watchdog = self._watchdog

        def wrapper(event: AgentEvent) -> None:
            watchdog.reset_timer()
            if self._on_live_output is not None:
                self._on_live_output(event)
            watchdog.check_timeout()

        return wrapper

    def stop_monitoring(self) -> None:
        if self._watchdog is not None:
            self._watchdog.stop_monitoring()


def _wrap_on_live_output_with_timeout(
    on_live_output: Callable[[AgentEvent], None] | None,
    timeout_seconds: int,
) -> tuple[
    Callable[[AgentEvent], None] | None,
    _LiveRuntimeOutputTimeoutContext | None,
]:
    if timeout_seconds <= 0:
        return on_live_output, None
    context = _LiveRuntimeOutputTimeoutContext(on_live_output, timeout_seconds)
    return context.wrapped_on_live_output, context
