from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, TypeVar, cast

from . import _builtin_runtime_client as _builtin_runtime_client_module
from . import _session_backed_provider_state_resolution as _provider_state_resolution
from . import _session_backed_provider_lifecycle_policy as _lifecycle_policy_module
from ._built_in_provider_session_invocation_dispatch import (
    dispatch_built_in_provider_session_invocation,
)
from ._builtin_provider_stream_interpretation import BuiltInProviderStreamInterpretation
from ._provider_invocation import (
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
from .errors import RuntimeConfigurationError
from .errors import AgentCancelledError, AgentTimeoutError
from .invocation_progress import InvocationProgress
from .session import RunKind
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

        def _portable_claude_state_dir_relpath(
            provider_state_dir_relpath: str | None,
        ) -> str | None:
            if is_caller_managed_runtime_state:
                return provider_state_dir_relpath
            return None

        continuation_input_facts: (
            _provider_state_resolution.ContinuationInputFacts | None
        ) = None

        if selected_stage.service == "codex":
            _lifecycle_policy_module.policy_for_service(
                selected_stage.service
            ).validate_stage(selected_stage)
            host_auth_path = _builtin_runtime_client_module._codex_host_auth_path()
            if not host_auth_path.exists():
                raise _builtin_runtime_client_module._missing_codex_auth_error()
            codex_resolution = (
                _provider_state_resolution.resolve_codex_new_session_facts(
                    runtime_state_dir=runtime_state_dir,
                    caller_owned_session_store=is_caller_managed_runtime_state,
                    model=selected_stage.model,
                    effort=selected_stage.effort,
                    host_auth_path=host_auth_path,
                )
            )
            provider_state_dir = codex_resolution.provider_state_dir
            continuation_input_facts = codex_resolution.continuation_input_facts
            if continuation_input_facts.run_kind is RunKind.RESUME:
                return _run_builtin_resumed_session(
                    _builtin_runtime_client_module.ResumedSessionRunRequest(
                        prompt=request.prompt,
                        invocation_dir=request.invocation_dir,
                        session_store=runtime_state_dir,
                        continuation=_provider_state_resolution.build_session_backed_continuation(
                            continuation_input_facts,
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
            provider_session_id: str | None = None
            codex_continuation_input_facts = continuation_input_facts
            invocation_result = _invoke_with_interruption_continuations(
                invoke=lambda: dispatch_built_in_provider_session_invocation(
                    service_name="codex",
                    run_kind=RunKind.FRESH,
                    invocation_dir=request.invocation_dir,
                    prompt=request.prompt,
                    model=selected_stage.model,
                    effort=selected_stage.effort,
                    tool_access=request.tool_access,
                    auth=None,
                    provider_state_dir=provider_state_dir,
                    provider_session_id=None,
                    argv_transform=request.argv_transform,
                    on_live_output=on_live_output,
                    timeout_seconds=request.timeout_seconds,
                    token=request.token,
                    provider_invocation_adapter=invocation_adapter,
                ),
                provider_session_id=provider_session_id,
                build_continuation=lambda active_provider_session_id: (
                    _provider_state_resolution.build_session_backed_continuation(
                        codex_continuation_input_facts,
                        tool_access=request.tool_access,
                        provider_session_id=active_provider_session_id,
                    )
                ),
            )
            if isinstance(invocation_result, ProviderInvocationFailure):
                provider_session_id = _resolve_active_provider_session_id(
                    stream_interpretation=(
                        _lifecycle_policy_module.policy_for_service(
                            "codex"
                        ).stream_interpretation()
                    ),
                    invocation_result=invocation_result,
                    prepared_or_continuation_provider_session_id=provider_session_id,
                )
                failure_error = _builtin_runtime_client_module._provider_invocation_error_from_failure(
                    "codex",
                    invocation_result,
                )
                failure_error.continuation = _interruption_continuation(
                    provider_work_started=(
                        failure_error.invocation_progress is InvocationProgress.STARTED
                    ),
                    provider_session_id=provider_session_id,
                    build_continuation=lambda active_provider_session_id: (
                        _provider_state_resolution.build_session_backed_continuation(
                            codex_continuation_input_facts,
                            tool_access=request.tool_access,
                            provider_session_id=active_provider_session_id,
                        )
                    ),
                )
                raise failure_error
            else:
                provider_session_id = _resolve_active_provider_session_id(
                    stream_interpretation=(
                        _lifecycle_policy_module.policy_for_service(
                            "codex"
                        ).stream_interpretation()
                    ),
                    invocation_result=invocation_result,
                    prepared_or_continuation_provider_session_id=None,
                )
                result_text = invocation_result.output
                usage = invocation_result.usage
            return _completed_result(
                output=result_text,
                usage=usage,
                continuation=(
                    _provider_state_resolution.build_session_backed_continuation(
                        codex_continuation_input_facts,
                        tool_access=request.tool_access,
                        provider_session_id=provider_session_id,
                    )
                    if provider_session_id is not None
                    else None
                ),
                service="codex",
                model=selected_stage.model,
                effort=selected_stage.effort,
            )
        elif selected_stage.service == "claude":
            claude_resolution = (
                _provider_state_resolution.resolve_claude_new_session_facts(
                    runtime_state_dir=runtime_state_dir,
                    caller_owned_session_store=is_caller_managed_runtime_state,
                    model=selected_stage.model,
                    effort=selected_stage.effort,
                )
            )
            provider_state_dir = claude_resolution.provider_state_dir
            continuation_input_facts = claude_resolution.continuation_input_facts
            if continuation_input_facts.run_kind is RunKind.RESUME:
                return _run_builtin_resumed_session(
                    _builtin_runtime_client_module.ResumedSessionRunRequest(
                        prompt=request.prompt,
                        invocation_dir=request.invocation_dir,
                        session_store=runtime_state_dir,
                        on_live_output=on_live_output,
                        timeout_seconds=0,
                        argv_transform=request.argv_transform,
                        continuation=_provider_state_resolution.build_session_backed_continuation(
                            continuation_input_facts,
                            tool_access=request.tool_access,
                        ),
                        provider_auth=selected_stage_auth,
                        token=request.token,
                    ),
                    provider_invocation_adapter=invocation_adapter,
                    on_live_output=on_live_output,
                )
            _lifecycle_policy_module.policy_for_service(
                selected_stage.service
            ).validate_stage(selected_stage)
            _lifecycle_policy_module.policy_for_service(
                selected_stage.service
            ).require_auth(selected_stage_auth)
        elif selected_stage.service == "opencode":
            opencode_resolution = (
                _provider_state_resolution.resolve_opencode_new_session_facts(
                    runtime_state_dir=runtime_state_dir,
                    caller_owned_session_store=is_caller_managed_runtime_state,
                    model=selected_stage.model,
                    effort=selected_stage.effort,
                )
            )
            provider_state_dir = opencode_resolution.provider_state_dir
            _lifecycle_policy_module.policy_for_service(
                selected_stage.service
            ).validate_stage(selected_stage)
            _lifecycle_policy_module.policy_for_service(
                selected_stage.service
            ).require_auth(selected_stage_auth)
            continuation_input_facts = opencode_resolution.continuation_input_facts
            provider_session_id = cast(
                str,
                cast(
                    _provider_state_resolution.PreparedOrRecoveredProviderSessionId,
                    continuation_input_facts.provider_session_id,
                ).value,
            )
            run_kind = continuation_input_facts.run_kind
        else:
            raise RuntimeConfigurationError(
                "RuntimeClient session-backed execution is only implemented for Claude, Codex, and OpenCode."
            )
        assert continuation_input_facts is not None
        active_continuation_input_facts = continuation_input_facts
        if selected_stage.service == "claude":
            provider_session_id = cast(
                str,
                cast(
                    _provider_state_resolution.PreparedOrRecoveredProviderSessionId,
                    active_continuation_input_facts.provider_session_id,
                ).value,
            )
            run_kind = active_continuation_input_facts.run_kind
            invocation_result = _invoke_with_interruption_continuations(
                invoke=lambda: dispatch_built_in_provider_session_invocation(
                    service_name="claude",
                    run_kind=run_kind,
                    invocation_dir=request.invocation_dir,
                    prompt=request.prompt,
                    model=selected_stage.model,
                    effort=selected_stage.effort,
                    tool_access=request.tool_access,
                    auth=selected_stage_auth,
                    provider_state_dir=provider_state_dir,
                    provider_session_id=cast(str, provider_session_id),
                    argv_transform=request.argv_transform,
                    on_live_output=on_live_output,
                    timeout_seconds=request.timeout_seconds,
                    token=request.token,
                    provider_invocation_adapter=invocation_adapter,
                ),
                provider_session_id=provider_session_id,
                build_continuation=lambda active_provider_session_id: (
                    _provider_state_resolution.build_session_backed_continuation(
                        active_continuation_input_facts,
                        tool_access=request.tool_access,
                        provider_session_id=active_provider_session_id,
                    )
                ),
            )
        else:
            assert provider_session_id is not None
            invocation_result = _invoke_with_interruption_continuations(
                invoke=lambda: dispatch_built_in_provider_session_invocation(
                    service_name="opencode",
                    run_kind=run_kind,
                    invocation_dir=request.invocation_dir,
                    prompt=request.prompt,
                    model=selected_stage.model,
                    effort=selected_stage.effort,
                    tool_access=request.tool_access,
                    auth=selected_stage_auth,
                    provider_state_dir=provider_state_dir,
                    provider_session_id=cast(str, provider_session_id),
                    argv_transform=request.argv_transform,
                    on_live_output=on_live_output,
                    timeout_seconds=request.timeout_seconds,
                    token=request.token,
                    provider_invocation_adapter=invocation_adapter,
                ),
                provider_session_id=provider_session_id,
                build_continuation=lambda active_provider_session_id: (
                    _provider_state_resolution.build_session_backed_continuation(
                        active_continuation_input_facts,
                        tool_access=request.tool_access,
                        provider_session_id=active_provider_session_id,
                    )
                ),
            )
        stream_interpretation = _lifecycle_policy_module.policy_for_service(
            selected_stage.service
        ).stream_interpretation()
        if isinstance(invocation_result, ProviderInvocationFailure):
            provider_session_id = _resolve_active_provider_session_id(
                stream_interpretation=stream_interpretation,
                invocation_result=invocation_result,
                prepared_or_continuation_provider_session_id=provider_session_id,
            )
            failure_error = (
                _builtin_runtime_client_module._provider_invocation_error_from_failure(
                    selected_stage.service,
                    invocation_result,
                )
            )
            failure_error.continuation = _interruption_continuation(
                provider_work_started=(
                    failure_error.invocation_progress is InvocationProgress.STARTED
                ),
                provider_session_id=provider_session_id,
                build_continuation=lambda active_provider_session_id: (
                    _provider_state_resolution.build_session_backed_continuation(
                        _provider_state_resolution.resolve_opencode_active_session_facts(
                            active_continuation_input_facts,
                            provider_session_id=active_provider_session_id,
                        ),
                        tool_access=request.tool_access,
                    )
                    if selected_stage.service == "opencode"
                    else _provider_state_resolution.build_session_backed_continuation(
                        active_continuation_input_facts,
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
        if selected_stage.service == "opencode":
            active_continuation_input_facts = (
                _provider_state_resolution.resolve_opencode_active_session_facts(
                    active_continuation_input_facts,
                    provider_session_id=provider_session_id,
                )
            )
            provider_session_id = cast(
                str,
                cast(
                    _provider_state_resolution.PreparedOrRecoveredProviderSessionId,
                    active_continuation_input_facts.provider_session_id,
                ).value,
            )
        assert provider_session_id is not None
        result_text = invocation_result.output
        usage = invocation_result.usage
        return _completed_result(
            output=result_text,
            usage=usage,
            continuation=_provider_state_resolution.build_session_backed_continuation(
                active_continuation_input_facts,
                tool_access=request.tool_access,
                provider_session_id=provider_session_id,
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
    provider_session_id: str | None
    provider_state_dir_relpath: str | None = None
    provider_state_dir: Path | None = None

    def _no_cleanup() -> None:
        return None

    if continuation_service == "codex":
        _lifecycle_policy_module.policy_for_service(
            continuation_service
        ).validate_stage(
            ProviderSelection(
                service="codex",
                model=request.model,
                effort=request.effort,
            )
        )
        provider_state_dir_relpath = continuation_facts.provider_state_dir_relpath
        provider_session_id = continuation_facts.provider_session_id
        host_auth_path = _builtin_runtime_client_module._codex_host_auth_path()
        if provider_state_dir_relpath is not None and not host_auth_path.exists():
            raise _builtin_runtime_client_module._missing_codex_auth_error()
        codex_resolution = (
            _provider_state_resolution.resolve_codex_resumed_session_facts(
                runtime_state_dir=runtime_state_dir,
                provider_state_dir_relpath=provider_state_dir_relpath,
                model=request.model,
                effort=request.effort,
                provider_session_id=provider_session_id,
                host_auth_path=host_auth_path,
            )
        )
        provider_state_dir = codex_resolution.provider_state_dir
        continuation_input_facts = codex_resolution.continuation_input_facts
        provider_session_id = cast(
            str,
            cast(
                _provider_state_resolution.PreparedOrRecoveredProviderSessionId,
                continuation_input_facts.provider_session_id,
            ).value,
        )
        active_provider_session_id: str | None = provider_session_id
        invocation_result = _invoke_with_interruption_continuations(
            invoke=lambda: dispatch_built_in_provider_session_invocation(
                service_name="codex",
                run_kind=RunKind.RESUME,
                invocation_dir=request.invocation_dir,
                prompt=request.prompt,
                model=request.model,
                effort=request.effort,
                tool_access=request.tool_access,
                auth=None,
                provider_state_dir=provider_state_dir,
                provider_session_id=cast(str, provider_session_id),
                argv_transform=request.argv_transform,
                on_live_output=on_live_output,
                timeout_seconds=request.timeout_seconds,
                token=request.token,
                provider_invocation_adapter=invocation_adapter,
            ),
            provider_session_id=active_provider_session_id,
            build_continuation=lambda resumed_provider_session_id: (
                _provider_state_resolution.build_session_backed_continuation(
                    continuation_input_facts,
                    tool_access=request.tool_access,
                    provider_session_id=resumed_provider_session_id,
                )
            ),
            fallback_continuation=request.continuation,
        )
        if isinstance(invocation_result, ProviderInvocationFailure):
            active_provider_session_id = _resolve_active_provider_session_id(
                stream_interpretation=(
                    _lifecycle_policy_module.policy_for_service(
                        "codex"
                    ).stream_interpretation()
                ),
                invocation_result=invocation_result,
                prepared_or_continuation_provider_session_id=(
                    active_provider_session_id
                ),
            )
            failure_error = (
                _builtin_runtime_client_module._provider_invocation_error_from_failure(
                    "codex",
                    invocation_result,
                )
            )
            failure_error.continuation = _interruption_continuation(
                provider_work_started=(
                    failure_error.invocation_progress is InvocationProgress.STARTED
                ),
                provider_session_id=active_provider_session_id,
                build_continuation=lambda resumed_provider_session_id: (
                    _provider_state_resolution.build_session_backed_continuation(
                        continuation_input_facts,
                        tool_access=request.tool_access,
                        provider_session_id=resumed_provider_session_id,
                    )
                ),
            )
            raise failure_error
        else:
            active_provider_session_id = _resolve_active_provider_session_id(
                stream_interpretation=(
                    _lifecycle_policy_module.policy_for_service(
                        "codex"
                    ).stream_interpretation()
                ),
                invocation_result=invocation_result,
                prepared_or_continuation_provider_session_id=provider_session_id,
            )
            result_text = invocation_result.output
            usage = invocation_result.usage
        return _completed_result(
            output=result_text,
            usage=usage,
            continuation=(
                _provider_state_resolution.build_session_backed_continuation(
                    continuation_input_facts,
                    tool_access=request.tool_access,
                    provider_session_id=active_provider_session_id,
                )
                if active_provider_session_id is not None
                else None
            ),
            service="codex",
            model=request.model,
            effort=request.effort,
        )

    if continuation_service not in {"claude", "opencode"}:
        raise RuntimeConfigurationError(
            "RuntimeClient session-backed execution is only implemented for Claude, Codex, and OpenCode."
        )
    provider_session_id = cast(
        str | None,
        continuation_facts.provider_session_id,
    )
    if continuation_service == "claude":
        _lifecycle_policy_module.policy_for_service(continuation_service).require_auth(
            request.provider_auth
        )
        provider_state_dir_relpath = continuation_facts.provider_state_dir_relpath
        claude_resolution = (
            _provider_state_resolution.resolve_claude_resumed_session_facts(
                runtime_state_dir=runtime_state_dir,
                provider_state_dir_relpath=provider_state_dir_relpath,
                model=request.model,
                effort=request.effort,
                provider_session_id=provider_session_id,
            )
        )
        provider_state_dir = claude_resolution.provider_state_dir
        continuation_input_facts = claude_resolution.continuation_input_facts
        provider_session_id = cast(
            str,
            cast(
                _provider_state_resolution.PreparedOrRecoveredProviderSessionId,
                continuation_input_facts.provider_session_id,
            ).value,
        )
        run_kind = continuation_input_facts.run_kind
        cleanup_opencode_state_dir = _no_cleanup
    else:
        _lifecycle_policy_module.policy_for_service(continuation_service).require_auth(
            request.provider_auth
        )
        opencode_resolution = (
            _provider_state_resolution.resolve_opencode_resumed_session_facts(
                runtime_state_dir=cast(Path, request.session_store),
                continuation=cast(Continuation, request.continuation),
                model=request.model,
                effort=request.effort,
            )
        )
        provider_state_dir = opencode_resolution.provider_state_dir
        continuation_input_facts = opencode_resolution.continuation_input_facts
        provider_session_id = cast(
            str,
            cast(
                _provider_state_resolution.PreparedOrRecoveredProviderSessionId,
                continuation_input_facts.provider_session_id,
            ).value,
        )
        run_kind = continuation_input_facts.run_kind
        cleanup_opencode_state_dir = _no_cleanup
    if continuation_service == "claude":
        invocation_result = _invoke_with_interruption_continuations(
            invoke=lambda: dispatch_built_in_provider_session_invocation(
                service_name="claude",
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
    else:
        invocation_result = _invoke_with_interruption_continuations(
            invoke=lambda: dispatch_built_in_provider_session_invocation(
                service_name="opencode",
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
    active_provider_session_interpretation = (
        _lifecycle_policy_module.policy_for_service(
            continuation_service
        ).stream_interpretation()
    )
    if isinstance(invocation_result, ProviderInvocationFailure):
        provider_session_id = _resolve_active_provider_session_id(
            stream_interpretation=active_provider_session_interpretation,
            invocation_result=invocation_result,
            prepared_or_continuation_provider_session_id=provider_session_id,
        )
        if continuation_service == "opencode":
            continuation_input_facts = (
                _provider_state_resolution.resolve_opencode_active_session_facts(
                    continuation_input_facts,
                    provider_session_id=provider_session_id,
                )
            )
            active_provider_session = cast(
                _provider_state_resolution.PreparedOrRecoveredProviderSessionId | None,
                continuation_input_facts.provider_session_id,
            )
            provider_session_id = (
                active_provider_session.value
                if active_provider_session is not None
                else None
            )
        failure_error = (
            _builtin_runtime_client_module._provider_invocation_error_from_failure(
                continuation_service,
                invocation_result,
            )
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
        cleanup_opencode_state_dir()
        raise failure_error
    provider_session_id = _resolve_active_provider_session_id(
        stream_interpretation=active_provider_session_interpretation,
        invocation_result=invocation_result,
        prepared_or_continuation_provider_session_id=provider_session_id,
    )
    if continuation_service == "opencode":
        continuation_input_facts = (
            _provider_state_resolution.resolve_opencode_active_session_facts(
                continuation_input_facts,
                provider_session_id=provider_session_id,
            )
        )
        provider_session_id = cast(
            str,
            cast(
                _provider_state_resolution.PreparedOrRecoveredProviderSessionId,
                continuation_input_facts.provider_session_id,
            ).value,
        )
    assert provider_session_id is not None
    result_text = invocation_result.output
    usage = invocation_result.usage
    result_continuation = _provider_state_resolution.build_session_backed_continuation(
        continuation_input_facts,
        tool_access=request.tool_access,
        provider_session_id=provider_session_id,
    )
    cleanup_opencode_state_dir()
    return _completed_result(
        output=result_text,
        usage=usage,
        continuation=result_continuation,
        service=continuation_service,
        model=request.model,
        effort=request.effort,
    )
