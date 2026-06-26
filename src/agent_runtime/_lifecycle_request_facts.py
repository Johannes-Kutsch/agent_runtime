from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from ._request_normalization import (
    normalize_continuation_request,
    normalize_provider_selection_request,
)
from .contracts import ToolAccess
from .errors import RuntimeConfigurationError
from .types import ProviderSelection


@dataclasses.dataclass(frozen=True)
class _LifecycleRequestFacts:
    invocation_dir: Path
    tool_access: ToolAccess
    session_namespace: str


@dataclasses.dataclass(frozen=True)
class _ProviderSelectionLifecycleRequestFacts(_LifecycleRequestFacts):
    provider_selection: ProviderSelection


@dataclasses.dataclass(frozen=True)
class _ResumedLifecycleRequestFacts(_LifecycleRequestFacts):
    model: str
    effort: str


def _provider_selection_request_facts(
    *,
    provider_selection: ProviderSelection | None,
    worktree: Path,
    tool_access: Any,
    tool_policy: Any,
    missing_sentinel: object,
    session_namespace: str,
    context: str,
    missing_message: str,
    workspace_name: str = "worktree",
) -> _ProviderSelectionLifecycleRequestFacts:
    normalized_request = normalize_provider_selection_request(
        provider_selection=provider_selection,
        worktree=worktree,
        tool_access=tool_access,
        tool_policy=tool_policy,
        missing_sentinel=missing_sentinel,
        session_namespace=session_namespace,
        context=context,
        missing_message=missing_message,
        workspace_name=workspace_name,
    )
    return _ProviderSelectionLifecycleRequestFacts(
        invocation_dir=normalized_request.worktree.path,
        provider_selection=normalized_request.provider_selection,
        tool_access=normalized_request.tool_access,
        session_namespace=normalized_request.session_namespace,
    )


def _resumed_session_request_facts(
    *,
    continuation: Any,
    worktree: Path,
    session_namespace: str,
    context: str,
    workspace_name: str = "worktree",
) -> _ResumedLifecycleRequestFacts:
    from ._portable_continuation_payload import read_portable_continuation_payload

    try:
        continuation_payload = read_portable_continuation_payload(continuation)
    except TypeError as exc:
        raise RuntimeConfigurationError(str(exc)) from exc
    normalized_request = normalize_continuation_request(
        worktree=worktree,
        tool_access=continuation_payload.tool_access,
        session_namespace=session_namespace,
        context=context,
        workspace_name=workspace_name,
    )
    return _ResumedLifecycleRequestFacts(
        invocation_dir=normalized_request.worktree.path,
        tool_access=normalized_request.tool_access,
        session_namespace=normalized_request.session_namespace,
        model=continuation_payload.model,
        effort=continuation_payload.effort,
    )
