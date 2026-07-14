from __future__ import annotations

from pathlib import Path
from typing import Callable

from . import _builtin_provider_rendering as _builtin_provider_rendering_module
from ._builtin_runtime_client import (
    _execute_rendered_provider_invocation,
    _opencode_stream_interpretation,
    _with_observed_output,
    _with_session_timeout_state,
)
from ._built_in_provider_lifecycle_policy import (
    policy_for_service as _policy_for_service,
)
from ._provider_invocation import (
    ProviderInvocationAdapter,
    ProviderInvocationFailure,
    ProviderInvocationResult,
)
from ._runtime_lifecycle import (
    AgentEvent,
    CancellationToken,
    ProviderAuth,
)
from .contracts import ToolAccess
from .errors import AgentCancelledError, AgentTimeoutError
from .session import RunKind


def dispatch_built_in_provider_session_invocation(
    *,
    service_name: str,
    run_kind: RunKind,
    invocation_dir: Path,
    prompt: str,
    model: str,
    effort: str,
    tool_access: ToolAccess,
    auth: ProviderAuth | None,
    provider_state_dir: Path | None,
    provider_session_id: str | None,
    argv_transform: (
        Callable[[tuple[str, ...], Path, dict[str, str]], tuple[str, ...]] | None
    ) = None,
    on_live_output: Callable[[AgentEvent], None] | None = None,
    timeout_seconds: int = 300,
    token: CancellationToken | None = None,
    provider_invocation_adapter: ProviderInvocationAdapter,
) -> ProviderInvocationResult | ProviderInvocationFailure:
    if service_name == "claude":
        return _dispatch_claude(
            provider_invocation_adapter=provider_invocation_adapter,
            invocation_dir=invocation_dir,
            prompt=prompt,
            model=model,
            effort=effort,
            tool_access=tool_access,
            auth=auth,
            provider_state_dir=provider_state_dir,
            run_kind=run_kind,
            provider_session_id=provider_session_id,
            argv_transform=argv_transform,
            on_live_output=on_live_output,
            timeout_seconds=timeout_seconds,
            token=token,
        )
    if service_name == "codex":
        return _dispatch_codex(
            provider_invocation_adapter=provider_invocation_adapter,
            invocation_dir=invocation_dir,
            prompt=prompt,
            model=model,
            effort=effort,
            tool_access=tool_access,
            provider_state_dir=provider_state_dir,
            run_kind=run_kind,
            provider_session_id=provider_session_id,
            argv_transform=argv_transform,
            on_live_output=on_live_output,
            timeout_seconds=timeout_seconds,
            token=token,
        )
    if service_name == "opencode":
        return _dispatch_opencode(
            provider_invocation_adapter=provider_invocation_adapter,
            invocation_dir=invocation_dir,
            prompt=prompt,
            model=model,
            effort=effort,
            tool_access=tool_access,
            auth=auth,
            provider_state_dir=provider_state_dir,
            run_kind=run_kind,
            provider_session_id=provider_session_id,
            argv_transform=argv_transform,
            on_live_output=on_live_output,
            timeout_seconds=timeout_seconds,
            token=token,
        )
    raise ValueError(
        f"dispatch_built_in_provider_session_invocation: unknown service {service_name!r}"
    )


def _dispatch_claude(
    *,
    provider_invocation_adapter: ProviderInvocationAdapter,
    invocation_dir: Path,
    prompt: str,
    model: str,
    effort: str,
    tool_access: ToolAccess,
    auth: ProviderAuth | None,
    provider_state_dir: Path | None,
    run_kind: RunKind,
    provider_session_id: str | None,
    argv_transform: (
        Callable[[tuple[str, ...], Path, dict[str, str]], tuple[str, ...]] | None
    ) = None,
    on_live_output: Callable[[AgentEvent], None] | None = None,
    timeout_seconds: int = 300,
    token: CancellationToken | None = None,
) -> ProviderInvocationResult | ProviderInvocationFailure:
    rendered = _builtin_provider_rendering_module.render_built_in_provider_invocation(
        _builtin_provider_rendering_module.BuiltInProviderRenderRequest(
            provider_selection=(
                _builtin_provider_rendering_module.BuiltInProviderSelectionFacts(
                    service="claude",
                    model=model,
                    effort=effort,
                )
            ),
            run_kind=run_kind,
            tool_access=tool_access,
            auth=auth,
            invocation_dir=invocation_dir,
            provider_state_dir=provider_state_dir,
            provider_session_id=provider_session_id,
        )
    )
    stream_interpretation = _with_observed_output(
        _policy_for_service("claude").stream_interpretation(),
        on_live_output,
    )
    stream_interpretation, timeout_state = _with_session_timeout_state(
        stream_interpretation,
        tracking_interpretation=_policy_for_service("claude").stream_interpretation(),
        fallback_provider_session_id=provider_session_id,
    )
    try:
        return _execute_rendered_provider_invocation(
            provider_invocation_adapter=provider_invocation_adapter,
            rendered=rendered,
            invocation_dir=invocation_dir,
            argv_transform=argv_transform,
            prompt=prompt,
            run_kind=run_kind,
            provider_session_id=provider_session_id,
            stream_interpretation=stream_interpretation,
            timeout_seconds=timeout_seconds,
            token=token,
        )
    except AgentTimeoutError as exc:
        timeout_state.apply_to_timeout(exc)
        raise
    except AgentCancelledError as exc:
        timeout_state.apply_to_cancellation(exc)
        raise


def _dispatch_codex(
    *,
    provider_invocation_adapter: ProviderInvocationAdapter,
    invocation_dir: Path,
    prompt: str,
    model: str,
    effort: str,
    tool_access: ToolAccess,
    provider_state_dir: Path | None,
    run_kind: RunKind,
    provider_session_id: str | None,
    argv_transform: (
        Callable[[tuple[str, ...], Path, dict[str, str]], tuple[str, ...]] | None
    ) = None,
    on_live_output: Callable[[AgentEvent], None] | None = None,
    timeout_seconds: int = 300,
    token: CancellationToken | None = None,
) -> ProviderInvocationResult | ProviderInvocationFailure:
    stream_interpretation = _with_observed_output(
        _policy_for_service("codex").stream_interpretation(),
        on_live_output,
    )
    stream_interpretation, timeout_state = _with_session_timeout_state(
        stream_interpretation,
        tracking_interpretation=_policy_for_service("codex").stream_interpretation(),
        fallback_provider_session_id=provider_session_id,
    )
    rendered = _builtin_provider_rendering_module._render_codex_invocation(
        _builtin_provider_rendering_module.BuiltInProviderRenderRequest(
            provider_selection=(
                _builtin_provider_rendering_module.BuiltInProviderSelectionFacts(
                    service="codex",
                    model=model,
                    effort=effort,
                )
            ),
            run_kind=run_kind,
            tool_access=tool_access,
            auth=None,
            invocation_dir=invocation_dir,
            provider_state_dir=provider_state_dir,
            provider_session_id=provider_session_id,
        ),
        validate_auth=False,
        argv_transform=argv_transform,
    )
    try:
        return _execute_rendered_provider_invocation(
            provider_invocation_adapter=provider_invocation_adapter,
            rendered=rendered,
            invocation_dir=invocation_dir,
            argv_transform=argv_transform,
            prompt=prompt,
            run_kind=run_kind,
            provider_session_id=provider_session_id,
            stream_interpretation=stream_interpretation,
            timeout_seconds=timeout_seconds,
            token=token,
        )
    except AgentTimeoutError as exc:
        timeout_state.apply_to_timeout(exc)
        raise
    except AgentCancelledError as exc:
        timeout_state.apply_to_cancellation(exc)
        raise


def _dispatch_opencode(
    *,
    provider_invocation_adapter: ProviderInvocationAdapter,
    invocation_dir: Path,
    prompt: str,
    model: str,
    effort: str,
    tool_access: ToolAccess,
    auth: ProviderAuth | None,
    provider_state_dir: Path | None,
    run_kind: RunKind,
    provider_session_id: str | None,
    argv_transform: (
        Callable[[tuple[str, ...], Path, dict[str, str]], tuple[str, ...]] | None
    ) = None,
    on_live_output: Callable[[AgentEvent], None] | None = None,
    timeout_seconds: int = 300,
    token: CancellationToken | None = None,
) -> ProviderInvocationResult | ProviderInvocationFailure:
    stream_interpretation = _opencode_stream_interpretation(
        on_live_output=on_live_output,
        fallback_provider_session_id=provider_session_id,
    )
    stream_interpretation, timeout_state = _with_session_timeout_state(
        stream_interpretation,
        tracking_interpretation=_opencode_stream_interpretation(
            fallback_provider_session_id=provider_session_id,
        ),
        fallback_provider_session_id=provider_session_id,
    )
    rendered = _builtin_provider_rendering_module.render_built_in_provider_invocation(
        _builtin_provider_rendering_module.BuiltInProviderRenderRequest(
            provider_selection=(
                _builtin_provider_rendering_module.BuiltInProviderSelectionFacts(
                    service="opencode",
                    model=model,
                    effort=effort,
                )
            ),
            run_kind=run_kind,
            tool_access=tool_access,
            auth=auth,
            invocation_dir=invocation_dir,
            provider_state_dir=provider_state_dir,
            provider_session_id=provider_session_id,
        )
    )
    try:
        return _execute_rendered_provider_invocation(
            provider_invocation_adapter=provider_invocation_adapter,
            rendered=rendered,
            invocation_dir=invocation_dir,
            argv_transform=argv_transform,
            prompt=prompt,
            run_kind=run_kind,
            provider_session_id=rendered.provider_session_id,
            stream_interpretation=stream_interpretation,
            timeout_seconds=timeout_seconds,
            token=token,
        )
    except AgentTimeoutError as exc:
        timeout_state.apply_to_timeout(exc)
        raise
    except AgentCancelledError as exc:
        timeout_state.apply_to_cancellation(exc)
        raise
