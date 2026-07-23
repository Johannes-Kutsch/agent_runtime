from __future__ import annotations

from datetime import datetime
from typing import cast

from ._built_in_provider_lifecycle_policy import BuiltInProviderLifecyclePolicy
from ._builtin_provider_stream_interpretation import (
    classify_built_in_provider_invocation_progress,
    resolve_built_in_provider_session_id,
)
from ._provider_invocation import (
    InvocationFailureKind,
    ProviderInvocationFailure,
)
from .errors import (
    ProviderUnavailableError,
    ProviderUnavailableReason,
    UsageLimitError,
)
from .invocation_progress import InvocationProgress

_SERVICE_NOT_AVAILABLE_DETAIL = (
    "No configured service candidates are currently available."
)


def provider_invocation_error_from_failure(
    policy: BuiltInProviderLifecyclePolicy,
    failure: ProviderInvocationFailure,
    service_name: str,
) -> UsageLimitError | ProviderUnavailableError:
    stream_interpretation = policy.stream_interpretation()
    stdout_lines = list(failure.stdout_lines)
    invocation_progress = (
        InvocationProgress.STARTED
        if classify_built_in_provider_invocation_progress(
            stream_interpretation,
            stdout_lines,
            provider_session_id=failure.provider_session_id,
        )
        is InvocationProgress.STARTED
        else InvocationProgress.NOT_STARTED
    )
    provider_session_id = resolve_built_in_provider_session_id(
        stream_interpretation,
        stdout_lines,
        provider_session_id=failure.provider_session_id,
    )
    if failure.kind is InvocationFailureKind.USAGE_LIMITED:
        error: UsageLimitError | ProviderUnavailableError = UsageLimitError(
            reset_time=cast(datetime | None, failure.reset_time),
            raw_message=(failure.detail if failure.reset_time is None else None),
            service_name=service_name,
            invocation_progress=invocation_progress,
            usage=failure.usage,
        )
    else:
        error = ProviderUnavailableError(
            failure.detail,
            reason=(
                failure.provider_unavailable_reason
                or (
                    ProviderUnavailableReason.SERVICE_NOT_AVAILABLE
                    if failure.detail == _SERVICE_NOT_AVAILABLE_DETAIL
                    else ProviderUnavailableReason.TRANSIENT_API_ERROR
                )
            ),
            service_name=service_name,
            invocation_progress=invocation_progress,
            usage=failure.usage,
        )
    setattr(error, "provider_session_id", provider_session_id)
    return error
