from __future__ import annotations

from collections.abc import Callable
from typing import Any

_LIVE_RUNTIME_OUTPUT_EXCEPTION = "_is_live_output_exception"
_LIVE_RUNTIME_OUTPUT_TIMEOUT_WRAPPER = "_is_live_output_timeout_wrapper"


def mark_live_runtime_output_exception(exc: BaseException) -> BaseException:
    setattr(exc, _LIVE_RUNTIME_OUTPUT_EXCEPTION, True)
    return exc


def is_live_runtime_output_exception(exc: BaseException) -> bool:
    return bool(getattr(exc, _LIVE_RUNTIME_OUTPUT_EXCEPTION, False))


def mark_live_runtime_output_timeout_wrapper(
    on_live_output: Callable[..., Any],
) -> Callable[..., Any]:
    setattr(on_live_output, _LIVE_RUNTIME_OUTPUT_TIMEOUT_WRAPPER, True)
    return on_live_output


def is_live_runtime_output_timeout_wrapper(
    on_live_output: Callable[..., Any],
) -> bool:
    return bool(getattr(on_live_output, _LIVE_RUNTIME_OUTPUT_TIMEOUT_WRAPPER, False))
