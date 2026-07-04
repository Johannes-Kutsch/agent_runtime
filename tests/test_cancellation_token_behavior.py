from __future__ import annotations

import threading

from agent_runtime._runtime_lifecycle import CancellationToken


def test_cancellation_token_is_not_cancelled_before_cancel() -> None:
    token = CancellationToken()
    assert token.is_cancelled is False


def test_cancellation_token_is_cancelled_after_cancel() -> None:
    token = CancellationToken()
    token.cancel()
    assert token.is_cancelled is True


def test_cancellation_token_cancel_is_idempotent() -> None:
    token = CancellationToken()
    token.cancel()
    token.cancel()
    assert token.is_cancelled is True


def test_cancellation_token_cancel_from_different_thread_is_visible_on_caller_thread() -> (
    None
):
    token = CancellationToken()
    ready = threading.Event()

    def _cancel() -> None:
        ready.wait()
        token.cancel()

    t = threading.Thread(target=_cancel)
    t.start()
    ready.set()
    t.join()

    assert token.is_cancelled is True


def test_cancellation_token_each_instance_is_independent() -> None:
    a = CancellationToken()
    b = CancellationToken()
    a.cancel()
    assert a.is_cancelled is True
    assert b.is_cancelled is False
