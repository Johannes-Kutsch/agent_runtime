from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

import agent_runtime as runtime
from agent_runtime._built_in_provider_lifecycle_policy import policy_for_service
from agent_runtime._provider_invocation import (
    InvocationFailureKind,
    ProviderInvocationFailure,
)
from agent_runtime._provider_invocation_failure_error import (
    provider_invocation_error_from_failure,
)
from agent_runtime.errors import (
    ProviderUnavailableError,
    ProviderUnavailableReason,
    UsageLimitError,
)
from agent_runtime.invocation_progress import InvocationProgress


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SERVICES = ["claude", "codex", "opencode"]

_RESET_TIME = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
_USAGE = runtime.ProviderUsage(input_tokens=10, output_tokens=5)


def _usage_limited_failure(
    *,
    reset_time: datetime | None = None,
    detail: str = "limit",
    stdout_lines: tuple[str, ...] = (),
    provider_session_id: str | None = None,
    usage: runtime.ProviderUsage | None = None,
) -> ProviderInvocationFailure:
    return ProviderInvocationFailure(
        kind=InvocationFailureKind.USAGE_LIMITED,
        detail=detail,
        stdout_lines=stdout_lines,
        reset_time=reset_time,
        provider_session_id=provider_session_id,
        usage=usage,
    )


def _unavailable_failure(
    *,
    detail: str = "something went wrong",
    provider_unavailable_reason: ProviderUnavailableReason | None = None,
    stdout_lines: tuple[str, ...] = (),
    provider_session_id: str | None = None,
    usage: runtime.ProviderUsage | None = None,
) -> ProviderInvocationFailure:
    return ProviderInvocationFailure(
        kind=InvocationFailureKind.PROVIDER_UNAVAILABLE,
        detail=detail,
        stdout_lines=stdout_lines,
        provider_unavailable_reason=provider_unavailable_reason,
        provider_session_id=provider_session_id,
        usage=usage,
    )


# ---------------------------------------------------------------------------
# USAGE_LIMITED: reset_time present → raw_message is None
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("service", _SERVICES)
def test_usage_limited_with_reset_time_sets_raw_message_to_none(service: str) -> None:
    policy = policy_for_service(service)
    failure = _usage_limited_failure(reset_time=_RESET_TIME, detail="ignored detail")

    error = provider_invocation_error_from_failure(policy, failure, service)

    assert isinstance(error, UsageLimitError)
    assert error.reset_time == _RESET_TIME
    assert error.raw_message is None


# ---------------------------------------------------------------------------
# USAGE_LIMITED: reset_time absent → raw_message equals detail
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("service", _SERVICES)
def test_usage_limited_without_reset_time_sets_raw_message_to_detail(
    service: str,
) -> None:
    policy = policy_for_service(service)
    failure = _usage_limited_failure(reset_time=None, detail="rate limited message")

    error = provider_invocation_error_from_failure(policy, failure, service)

    assert isinstance(error, UsageLimitError)
    assert error.reset_time is None
    assert error.raw_message == "rate limited message"


# ---------------------------------------------------------------------------
# PROVIDER_UNAVAILABLE: explicit provider_unavailable_reason
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("service", _SERVICES)
def test_provider_unavailable_with_explicit_reason_preserves_reason(
    service: str,
) -> None:
    policy = policy_for_service(service)
    failure = _unavailable_failure(
        provider_unavailable_reason=ProviderUnavailableReason.SERVICE_NOT_AVAILABLE,
        detail="any detail",
    )

    error = provider_invocation_error_from_failure(policy, failure, service)

    assert isinstance(error, ProviderUnavailableError)
    assert error.reason is ProviderUnavailableReason.SERVICE_NOT_AVAILABLE


# ---------------------------------------------------------------------------
# PROVIDER_UNAVAILABLE: sentinel detail → SERVICE_NOT_AVAILABLE
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("service", _SERVICES)
def test_provider_unavailable_with_sentinel_detail_yields_service_not_available(
    service: str,
) -> None:
    from agent_runtime._provider_invocation_failure_error import (
        _SERVICE_NOT_AVAILABLE_DETAIL,
    )

    policy = policy_for_service(service)
    failure = _unavailable_failure(
        detail=_SERVICE_NOT_AVAILABLE_DETAIL,
        provider_unavailable_reason=None,
    )

    error = provider_invocation_error_from_failure(policy, failure, service)

    assert isinstance(error, ProviderUnavailableError)
    assert error.reason is ProviderUnavailableReason.SERVICE_NOT_AVAILABLE


# ---------------------------------------------------------------------------
# PROVIDER_UNAVAILABLE: arbitrary detail → TRANSIENT_API_ERROR
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("service", _SERVICES)
def test_provider_unavailable_with_arbitrary_detail_yields_transient_api_error(
    service: str,
) -> None:
    policy = policy_for_service(service)
    failure = _unavailable_failure(
        detail="some transient problem",
        provider_unavailable_reason=None,
    )

    error = provider_invocation_error_from_failure(policy, failure, service)

    assert isinstance(error, ProviderUnavailableError)
    assert error.reason is ProviderUnavailableReason.TRANSIENT_API_ERROR


# ---------------------------------------------------------------------------
# service_name preservation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("service", _SERVICES)
def test_service_name_is_preserved_on_returned_error(service: str) -> None:
    policy = policy_for_service(service)
    failure = _usage_limited_failure()

    error = provider_invocation_error_from_failure(policy, failure, service)

    assert error.service_name == service


# ---------------------------------------------------------------------------
# usage preservation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("service", _SERVICES)
def test_usage_is_preserved_on_usage_limited_error(service: str) -> None:
    policy = policy_for_service(service)
    failure = _usage_limited_failure(usage=_USAGE)

    error = provider_invocation_error_from_failure(policy, failure, service)

    assert error.usage == _USAGE


@pytest.mark.parametrize("service", _SERVICES)
def test_usage_is_preserved_on_provider_unavailable_error(service: str) -> None:
    policy = policy_for_service(service)
    failure = _unavailable_failure(usage=_USAGE)

    error = provider_invocation_error_from_failure(policy, failure, service)

    assert error.usage == _USAGE


# ---------------------------------------------------------------------------
# invocation_progress: started vs not-started
# ---------------------------------------------------------------------------


def test_claude_invocation_progress_is_started_when_stdout_has_assistant_output() -> (
    None
):
    policy = policy_for_service("claude")
    started_line = json.dumps(
        {
            "type": "assistant",
            "message": {
                "content": [{"type": "text", "text": "hello"}],
                "usage": {},
            },
        }
    )
    failure = _usage_limited_failure(
        stdout_lines=(started_line + "\n",),
        provider_session_id=None,
    )

    error = provider_invocation_error_from_failure(policy, failure, "claude")

    assert error.invocation_progress is InvocationProgress.STARTED


def test_claude_invocation_progress_is_not_started_when_stdout_is_empty() -> None:
    policy = policy_for_service("claude")
    failure = _usage_limited_failure(stdout_lines=(), provider_session_id=None)

    error = provider_invocation_error_from_failure(policy, failure, "claude")

    assert error.invocation_progress is InvocationProgress.NOT_STARTED


def test_codex_invocation_progress_is_started_when_stdout_has_thread_started() -> None:
    policy = policy_for_service("codex")
    thread_line = (
        json.dumps({"type": "thread.started", "thread_id": "t-started"}) + "\n"
    )
    failure = _usage_limited_failure(
        stdout_lines=(thread_line,),
        provider_session_id=None,
    )

    error = provider_invocation_error_from_failure(policy, failure, "codex")

    assert error.invocation_progress is InvocationProgress.STARTED


def test_codex_invocation_progress_is_not_started_when_stdout_is_empty() -> None:
    policy = policy_for_service("codex")
    failure = _usage_limited_failure(stdout_lines=(), provider_session_id=None)

    error = provider_invocation_error_from_failure(policy, failure, "codex")

    assert error.invocation_progress is InvocationProgress.NOT_STARTED


def test_opencode_invocation_progress_is_started_when_stdout_has_session_id() -> None:
    policy = policy_for_service("opencode")
    text_line = (
        json.dumps(
            {
                "type": "text",
                "sessionID": "oc-session-1",
                "timestamp": 1,
                "part": {"type": "text", "text": "hello", "time": {"start": 1}},
            }
        )
        + "\n"
    )
    failure = _usage_limited_failure(
        stdout_lines=(text_line,),
        provider_session_id=None,
    )

    error = provider_invocation_error_from_failure(policy, failure, "opencode")

    assert error.invocation_progress is InvocationProgress.STARTED


def test_opencode_invocation_progress_is_not_started_when_stdout_is_empty() -> None:
    policy = policy_for_service("opencode")
    failure = _usage_limited_failure(stdout_lines=(), provider_session_id=None)

    error = provider_invocation_error_from_failure(policy, failure, "opencode")

    assert error.invocation_progress is InvocationProgress.NOT_STARTED


# ---------------------------------------------------------------------------
# provider_session_id: from failure's own provider_session_id
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("service", _SERVICES)
def test_provider_session_id_attribute_is_set_from_failure_provider_session_id(
    service: str,
) -> None:
    policy = policy_for_service(service)
    failure = _usage_limited_failure(
        provider_session_id="direct-session-id",
        stdout_lines=(),
    )

    error = provider_invocation_error_from_failure(policy, failure, service)

    assert getattr(error, "provider_session_id") == "direct-session-id"


# ---------------------------------------------------------------------------
# provider_session_id: from stream interpretation (Codex and OpenCode)
# ---------------------------------------------------------------------------


def test_codex_provider_session_id_attribute_is_resolved_from_thread_started_stream() -> (
    None
):
    policy = policy_for_service("codex")
    thread_line = (
        json.dumps({"type": "thread.started", "thread_id": "stream-thread-99"}) + "\n"
    )
    failure = _usage_limited_failure(
        stdout_lines=(thread_line,),
        provider_session_id=None,
    )

    error = provider_invocation_error_from_failure(policy, failure, "codex")

    assert getattr(error, "provider_session_id") == "stream-thread-99"


def test_opencode_provider_session_id_attribute_is_resolved_from_stream_session_id() -> (
    None
):
    policy = policy_for_service("opencode")
    event_line = (
        json.dumps(
            {
                "type": "error",
                "sessionID": "oc-stream-session-7",
                "timestamp": 1,
                "error": {"name": "RateLimitError", "data": {}},
            }
        )
        + "\n"
    )
    failure = _usage_limited_failure(
        stdout_lines=(event_line,),
        provider_session_id=None,
    )

    error = provider_invocation_error_from_failure(policy, failure, "opencode")

    assert getattr(error, "provider_session_id") == "oc-stream-session-7"
