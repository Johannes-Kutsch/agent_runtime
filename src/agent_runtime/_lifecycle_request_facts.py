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
class _EphemeralRunRequestFacts(_ProviderSelectionLifecycleRequestFacts):
    argv_transform: Any


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


def _ephemeral_run_request_facts(
    *,
    invocation_dir: Path | None,
    compatibility_kwargs: dict[str, Any],
    provider_selection: ProviderSelection | None,
    tool_access: Any,
    tool_policy: Any,
    missing_sentinel: object,
    session_namespace: str,
    context: str,
    missing_message: str,
    public_invocation_dir_name: str,
) -> _EphemeralRunRequestFacts:
    argv_transform = compatibility_kwargs.pop("argv_transform", None)
    legacy_worktree = compatibility_kwargs.pop("worktree", None)
    if compatibility_kwargs:
        unexpected_argument = next(iter(compatibility_kwargs))
        raise TypeError(
            f"{context} got an unexpected keyword argument '{unexpected_argument}'."
        )
    if invocation_dir is not None and legacy_worktree is not None:
        raise TypeError(
            f"{context} received conflicting `{public_invocation_dir_name}` and `worktree` values."
        )
    resolved_invocation_dir = (
        invocation_dir if invocation_dir is not None else legacy_worktree
    )
    if resolved_invocation_dir is None:
        raise TypeError(f"{context} requires an `{public_invocation_dir_name}` value.")
    normalized_request = _provider_selection_request_facts(
        provider_selection=provider_selection,
        worktree=resolved_invocation_dir,
        tool_access=tool_access,
        tool_policy=tool_policy,
        missing_sentinel=missing_sentinel,
        session_namespace=session_namespace,
        context=context,
        missing_message=missing_message,
        workspace_name=public_invocation_dir_name,
    )
    return _EphemeralRunRequestFacts(
        invocation_dir=normalized_request.invocation_dir,
        provider_selection=normalized_request.provider_selection,
        tool_access=normalized_request.tool_access,
        session_namespace=normalized_request.session_namespace,
        argv_transform=argv_transform,
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
