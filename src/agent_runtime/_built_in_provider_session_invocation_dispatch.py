from __future__ import annotations

from pathlib import Path
from typing import Callable

from . import _builtin_provider_rendering as _builtin_provider_rendering_module
from ._builtin_runtime_client import (
    _execute_rendered_provider_invocation,
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


def _make_render_request(
    *,
    service: str,
    model: str,
    effort: str,
    run_kind: RunKind,
    tool_access: ToolAccess,
    auth: ProviderAuth | None,
    invocation_dir: Path,
    provider_state_dir: Path | None,
    provider_session_id: str | None,
) -> _builtin_provider_rendering_module.BuiltInProviderRenderRequest:
    return _builtin_provider_rendering_module.BuiltInProviderRenderRequest(
        provider_selection=_builtin_provider_rendering_module.BuiltInProviderSelectionFacts(
            service=service,
            model=model,
            effort=effort,
        ),
        run_kind=run_kind,
        tool_access=tool_access,
        auth=auth,
        invocation_dir=invocation_dir,
        provider_state_dir=provider_state_dir,
        provider_session_id=provider_session_id,
    )


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
    policy = _policy_for_service(service_name)
    stream_interpretation, timeout_state = policy.build_session_dispatch_interpretation(
        on_live_output=on_live_output,
        fallback_provider_session_id=provider_session_id,
        on_provider_session_id=None,
    )
    rendered = policy.render_invocation(
        _make_render_request(
            service=service_name,
            model=model,
            effort=effort,
            run_kind=run_kind,
            tool_access=tool_access,
            auth=policy.select_auth(auth),
            invocation_dir=invocation_dir,
            provider_state_dir=provider_state_dir,
            provider_session_id=provider_session_id,
        ),
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
            provider_session_id=policy.execute_provider_session_id(
                rendered, provider_session_id
            ),
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
