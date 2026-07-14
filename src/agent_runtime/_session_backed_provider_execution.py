from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, TypeVar, cast

from . import _builtin_runtime_client as _builtin_runtime_client_module
from . import _session_backed_provider_state_resolution as _provider_state_resolution
from . import _built_in_provider_lifecycle_policy as _lifecycle_policy_module
from ._built_in_provider_session_invocation_dispatch import (
    dispatch_built_in_provider_session_invocation,
)
from ._builtin_provider_stream_interpretation import (
    BuiltInProviderStreamInterpretation,
    classify_built_in_provider_invocation_progress,
)
from ._provider_invocation import (
    InvocationFailureKind,
    ProviderInvocationAdapter,
    ProviderInvocationFailure,
    ProviderInvocationResult,
)
from ._runtime_lifecycle import (
    Continuation,
    NewSessionRunRequest,
    ProviderUsage,
    ResumedSessionRunRequest,
    RunResult,
)
from .errors import (
    AgentCancelledError,
    AgentTimeoutError,
    ProviderUnavailableError,
    ProviderUnavailableReason,
    RuntimeConfigurationError,
    UsageLimitError,
)
from .invocation_progress import InvocationProgress
from .types import ProviderSelection, ResolvedProvider


def _resolve_active_provider_session_id(
    *,
    stream_interpretation: BuiltInProviderStreamInterpretation,
    invocation_result: ProviderInvocationResult | ProviderInvocationFailure,
    prepared_or_continuation_provider_session_id: str | None,
) -> str | None:
    if stream_interpretation.extract_provider_session_id is not None:
        observed_provider_session_id = (
            stream_interpretation.extract_provider_session_id(
                list(invocation_result.stdout_lines)
            )
        )
        if observed_provider_session_id is not None:
            return observed_provider_session_id
    if invocation_result.provider_session_id is not None:
        return invocation_result.provider_session_id
    return prepared_or_continuation_provider_session_id


def _interruption_continuation(
    *,
    provider_work_started: bool,
    provider_session_id: str | None,
    build_continuation: Callable[[str], Continuation],
) -> Continuation | None:
    if not provider_work_started or provider_session_id is None:
        return None
    return build_continuation(provider_session_id)


def _completed_result(
    *,
    output: str,
    usage: ProviderUsage | None,
    continuation: Continuation | None,
    service: str,
    model: str,
    effort: str,
) -> RunResult:
    return RunResult(
        output=output,
        usage=usage,
        continuation=continuation,
        selected=ResolvedProvider(service=service, model=model, effort=effort),
    )


def _augment_timeout_interruption(
    *,
    error: AgentTimeoutError,
    provider_session_id: str | None,
    build_continuation: Callable[[str], Continuation],
    fallback_continuation: Continuation | None = None,
) -> None:
    timeout_provider_session_id = cast(
        str | None,
        getattr(error, "provider_session_id", provider_session_id),
    )
    if error.invocation_progress is InvocationProgress.STARTED:
        error.continuation = _interruption_continuation(
            provider_work_started=True,
            provider_session_id=timeout_provider_session_id,
            build_continuation=build_continuation,
        )
        return
    if fallback_continuation is not None:
        error.continuation = fallback_continuation


def _augment_cancellation_interruption(
    *,
    error: AgentCancelledError,
    provider_session_id: str | None,
    build_continuation: Callable[[str], Continuation],
    fallback_continuation: Continuation | None = None,
) -> None:
    cancel_provider_session_id = cast(
        str | None,
        getattr(error, "provider_session_id", provider_session_id),
    )
    if error.invocation_progress is InvocationProgress.STARTED:
        error.continuation = _interruption_continuation(
            provider_work_started=True,
            provider_session_id=cancel_provider_session_id,
            build_continuation=build_continuation,
        )
        return
    if fallback_continuation is not None:
        error.continuation = fallback_continuation


_InvocationResultT = TypeVar("_InvocationResultT")


def _invoke_with_interruption_continuations(
    *,
    invoke: Callable[[], _InvocationResultT],
    provider_session_id: str | None,
    build_continuation: Callable[[str], Continuation],
    fallback_continuation: Continuation | None = None,
) -> _InvocationResultT:
    try:
        return invoke()
    except AgentTimeoutError as exc:
        _augment_timeout_interruption(
            error=exc,
            provider_session_id=provider_session_id,
            build_continuation=build_continuation,
            fallback_continuation=fallback_continuation,
        )
        raise
    except AgentCancelledError as exc:
        _augment_cancellation_interruption(
            error=exc,
            provider_session_id=provider_session_id,
            build_continuation=build_continuation,
            fallback_continuation=fallback_continuation,
        )
        raise


def _provider_invocation_error_from_failure(
    service_name: str,
    policy: _lifecycle_policy_module.BuiltInProviderLifecyclePolicy,
    failure: ProviderInvocationFailure,
) -> UsageLimitError | ProviderUnavailableError:
    stream_interpretation = policy.stream_interpretation()
    invocation_progress = (
        InvocationProgress.STARTED
        if classify_built_in_provider_invocation_progress(
            stream_interpretation,
            list(failure.stdout_lines),
            provider_session_id=failure.provider_session_id,
        )
        is InvocationProgress.STARTED
        else InvocationProgress.NOT_STARTED
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
                    if failure.detail
                    == _builtin_runtime_client_module._SERVICE_NOT_AVAILABLE_DETAIL
                    else ProviderUnavailableReason.TRANSIENT_API_ERROR
                )
            ),
            service_name=service_name,
            invocation_progress=invocation_progress,
            usage=failure.usage,
        )
    setattr(error, "provider_session_id", failure.provider_session_id)
    return error


def _run_builtin_new_session(
    request: NewSessionRunRequest,
    *,
    provider_invocation_adapter: ProviderInvocationAdapter | None = None,
    on_live_output: Callable[[Any], None] | None = None,
):
    if request.token is not None and request.token.is_cancelled:
        raise AgentCancelledError()
    if request.session_store is None:
        raise RuntimeConfigurationError(
            "RuntimeClient Start Session Run requires a `session_store`."
        )
    invocation_adapter = (
        _builtin_runtime_client_module._default_provider_invocation_adapter()
        if provider_invocation_adapter is None
        else provider_invocation_adapter
    )
    runtime_state_dir, cleanup_runtime_state_dir, is_caller_managed_runtime_state = (
        _builtin_runtime_client_module._new_session_runtime_state_dir(
            request,
            context="new-session",
        )
    )
    try:
        if (
            _builtin_runtime_client_module.supported_builtin_provider_selection(
                request.provider_selection
            )
            is None
        ):
            raise RuntimeConfigurationError(
                "RuntimeClient requires at least one supported built-in service candidate."
            )
        selected_stage = _builtin_runtime_client_module._select_builtin_stage(
            request.provider_selection
        )
        selected_stage_auth = _builtin_runtime_client_module._selection_auth(
            selected_stage
        )
        _builtin_runtime_client_module._require_portable_continuation_support(
            selected_stage.service
        )
        policy = _lifecycle_policy_module.policy_for_service(selected_stage.service)
        policy.validate_stage(selected_stage)
        policy.require_auth(selected_stage_auth)
        outcome = policy.resolve_new_session_facts(
            runtime_state_dir,
            is_caller_managed_runtime_state,
            selected_stage.model,
            selected_stage.effort,
        )
        if isinstance(outcome, _lifecycle_policy_module.NewSessionRedirect):
            return _run_builtin_resumed_session(
                _builtin_runtime_client_module.ResumedSessionRunRequest(
                    prompt=request.prompt,
                    invocation_dir=request.invocation_dir,
                    session_store=runtime_state_dir,
                    continuation=_provider_state_resolution.build_session_backed_continuation(
                        outcome.continuation_input_facts,
                        tool_access=request.tool_access,
                    ),
                    provider_auth=selected_stage_auth,
                    on_live_output=on_live_output,
                    timeout_seconds=0,
                    argv_transform=request.argv_transform,
                    token=request.token,
                ),
                provider_invocation_adapter=invocation_adapter,
                on_live_output=on_live_output,
            )
        provider_state_dir = outcome.provider_state_dir
        continuation_input_facts = outcome.continuation_input_facts
        provider_session_id: str | None = (
            continuation_input_facts.provider_session_id.value
            if continuation_input_facts.provider_session_id is not None
            else None
        )
        run_kind = continuation_input_facts.run_kind
        invocation_result = _invoke_with_interruption_continuations(
            invoke=lambda: dispatch_built_in_provider_session_invocation(
                service_name=selected_stage.service,
                run_kind=run_kind,
                invocation_dir=request.invocation_dir,
                prompt=request.prompt,
                model=selected_stage.model,
                effort=selected_stage.effort,
                tool_access=request.tool_access,
                auth=selected_stage_auth,
                provider_state_dir=provider_state_dir,
                provider_session_id=provider_session_id,
                argv_transform=request.argv_transform,
                on_live_output=on_live_output,
                timeout_seconds=request.timeout_seconds,
                token=request.token,
                provider_invocation_adapter=invocation_adapter,
            ),
            provider_session_id=provider_session_id,
            build_continuation=lambda active_provider_session_id: (
                _provider_state_resolution.build_session_backed_continuation(
                    continuation_input_facts,
                    tool_access=request.tool_access,
                    provider_session_id=active_provider_session_id,
                )
            ),
        )
        stream_interpretation = policy.stream_interpretation()
        if isinstance(invocation_result, ProviderInvocationFailure):
            provider_session_id = _resolve_active_provider_session_id(
                stream_interpretation=stream_interpretation,
                invocation_result=invocation_result,
                prepared_or_continuation_provider_session_id=provider_session_id,
            )
            failure_error = _provider_invocation_error_from_failure(
                selected_stage.service, policy, invocation_result
            )
            failure_error.continuation = _interruption_continuation(
                provider_work_started=(
                    failure_error.invocation_progress is InvocationProgress.STARTED
                ),
                provider_session_id=provider_session_id,
                build_continuation=lambda active_provider_session_id: (
                    _provider_state_resolution.build_session_backed_continuation(
                        policy.refresh_active_session_facts(
                            continuation_input_facts,
                            active_provider_session_id,
                        ),
                        tool_access=request.tool_access,
                        provider_session_id=active_provider_session_id,
                    )
                ),
            )
            raise failure_error
        provider_session_id = _resolve_active_provider_session_id(
            stream_interpretation=stream_interpretation,
            invocation_result=invocation_result,
            prepared_or_continuation_provider_session_id=provider_session_id,
        )
        continuation_input_facts = policy.refresh_active_session_facts(
            continuation_input_facts,
            provider_session_id,
        )
        return _completed_result(
            output=invocation_result.output,
            usage=invocation_result.usage,
            continuation=(
                _provider_state_resolution.build_session_backed_continuation(
                    continuation_input_facts,
                    tool_access=request.tool_access,
                    provider_session_id=provider_session_id,
                )
                if provider_session_id is not None
                else None
            ),
            service=selected_stage.service,
            model=selected_stage.model,
            effort=selected_stage.effort,
        )
    finally:
        cleanup_runtime_state_dir()


def _run_builtin_resumed_session(
    request: ResumedSessionRunRequest,
    *,
    provider_invocation_adapter: ProviderInvocationAdapter | None = None,
    on_live_output: Callable[[Any], None] | None = None,
):
    if request.token is not None and request.token.is_cancelled:
        raise AgentCancelledError()
    if request.session_store is None:
        raise RuntimeConfigurationError(
            "RuntimeClient Resume Session Run requires a `session_store`."
        )
    invocation_adapter = (
        _builtin_runtime_client_module._default_provider_invocation_adapter()
        if provider_invocation_adapter is None
        else provider_invocation_adapter
    )
    runtime_state_dir = request.session_store
    continuation = request.continuation
    if continuation is None:
        raise RuntimeConfigurationError(
            "RuntimeClient resumed-session execution requires a continuation."
        )
    try:
        continuation_facts = continuation.session_backed_facts
    except TypeError as exc:
        raise RuntimeConfigurationError(str(exc)) from exc
    continuation_service = continuation_facts.selected.service
    _builtin_runtime_client_module._require_portable_continuation_support(
        continuation_service
    )
    policy = _lifecycle_policy_module.policy_for_service(continuation_service)
    policy.validate_stage(
        ProviderSelection(
            service=continuation_service,
            model=request.model,
            effort=request.effort,
        )
    )
    policy.require_auth(request.provider_auth)
    resumed_outcome = policy.resolve_resumed_session_facts(
        _lifecycle_policy_module.ResumedSessionFactsInput(
            runtime_state_dir=runtime_state_dir,
            provider_state_dir_relpath=continuation_facts.provider_state_dir_relpath,
            provider_session_id=continuation_facts.provider_session_id,
            exact_transcript_match=continuation_facts.exact_transcript_match,
            model=request.model,
            effort=request.effort,
            continuation=request.continuation,
        )
    )
    provider_state_dir = resumed_outcome.provider_state_dir
    continuation_input_facts = resumed_outcome.continuation_input_facts
    provider_session_id: str | None = cast(
        str,
        cast(
            _provider_state_resolution.PreparedOrRecoveredProviderSessionId,
            continuation_input_facts.provider_session_id,
        ).value,
    )
    run_kind = continuation_input_facts.run_kind
    invocation_result = _invoke_with_interruption_continuations(
        invoke=lambda: dispatch_built_in_provider_session_invocation(
            service_name=continuation_service,
            run_kind=run_kind,
            invocation_dir=request.invocation_dir,
            prompt=request.prompt,
            model=request.model,
            effort=request.effort,
            tool_access=request.tool_access,
            auth=request.provider_auth,
            provider_state_dir=provider_state_dir,
            provider_session_id=cast(str, provider_session_id),
            argv_transform=request.argv_transform,
            on_live_output=on_live_output,
            timeout_seconds=request.timeout_seconds,
            token=request.token,
            provider_invocation_adapter=invocation_adapter,
        ),
        provider_session_id=provider_session_id,
        build_continuation=lambda resumed_provider_session_id: (
            _provider_state_resolution.build_session_backed_continuation(
                continuation_input_facts,
                tool_access=request.tool_access,
                provider_session_id=resumed_provider_session_id,
            )
        ),
        fallback_continuation=request.continuation,
    )
    stream_interpretation = policy.stream_interpretation()
    if isinstance(invocation_result, ProviderInvocationFailure):
        provider_session_id = _resolve_active_provider_session_id(
            stream_interpretation=stream_interpretation,
            invocation_result=invocation_result,
            prepared_or_continuation_provider_session_id=provider_session_id,
        )
        continuation_input_facts = policy.refresh_active_session_facts(
            continuation_input_facts,
            provider_session_id,
        )
        failure_error = _provider_invocation_error_from_failure(
            continuation_service, policy, invocation_result
        )
        failure_error.continuation = _interruption_continuation(
            provider_work_started=(
                failure_error.invocation_progress is InvocationProgress.STARTED
            ),
            provider_session_id=provider_session_id,
            build_continuation=lambda active_provider_session_id: (
                _provider_state_resolution.build_session_backed_continuation(
                    continuation_input_facts,
                    tool_access=request.tool_access,
                    provider_session_id=active_provider_session_id,
                )
            ),
        )
        raise failure_error
    provider_session_id = _resolve_active_provider_session_id(
        stream_interpretation=stream_interpretation,
        invocation_result=invocation_result,
        prepared_or_continuation_provider_session_id=provider_session_id,
    )
    continuation_input_facts = policy.refresh_active_session_facts(
        continuation_input_facts,
        provider_session_id,
    )
    assert provider_session_id is not None
    return _completed_result(
        output=invocation_result.output,
        usage=invocation_result.usage,
        continuation=_provider_state_resolution.build_session_backed_continuation(
            continuation_input_facts,
            tool_access=request.tool_access,
            provider_session_id=provider_session_id,
        ),
        service=continuation_service,
        model=request.model,
        effort=request.effort,
    )
