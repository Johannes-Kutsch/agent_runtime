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


@dataclasses.dataclass(frozen=True)
class _ProviderSelectionLifecycleRequestFacts(_LifecycleRequestFacts):
    provider_selection: ProviderSelection


@dataclasses.dataclass(frozen=True)
class _EphemeralRunRequestFacts(_ProviderSelectionLifecycleRequestFacts):
    argv_transform: Any


@dataclasses.dataclass(frozen=True)
class _NewSessionRunRequestFacts(_ProviderSelectionLifecycleRequestFacts):
    session_store: Path | None
    argv_transform: Any


@dataclasses.dataclass(frozen=True)
class _ResumedLifecycleRequestFacts(_LifecycleRequestFacts):
    model: str
    effort: str
    session_store: Path | None
    argv_transform: Any


def _resolve_invocation_dir(
    *,
    invocation_dir: Path | None,
    compatibility_kwargs: dict[str, Any],
    context: str,
    public_invocation_dir_name: str,
) -> Path:
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
    return resolved_invocation_dir


def _provider_selection_request_facts(
    *,
    provider_selection: ProviderSelection | None,
    invocation_dir: Path,
    tool_access: Any,
    tool_policy: Any,
    missing_sentinel: object,
    context: str,
    missing_message: str,
    workspace_name: str = "invocation_dir",
) -> _ProviderSelectionLifecycleRequestFacts:
    normalized_request = normalize_provider_selection_request(
        provider_selection=provider_selection,
        invocation_dir=invocation_dir,
        tool_access=tool_access,
        tool_policy=tool_policy,
        missing_sentinel=missing_sentinel,
        context=context,
        missing_message=missing_message,
        workspace_name=workspace_name,
    )
    return _ProviderSelectionLifecycleRequestFacts(
        invocation_dir=normalized_request.invocation_dir,
        provider_selection=normalized_request.provider_selection,
        tool_access=normalized_request.tool_access,
    )


def _ephemeral_run_request_facts(
    *,
    invocation_dir: Path | None,
    compatibility_kwargs: dict[str, Any],
    provider_selection: ProviderSelection | None,
    tool_access: Any,
    tool_policy: Any,
    missing_sentinel: object,
    context: str,
    missing_message: str,
    public_invocation_dir_name: str,
) -> _EphemeralRunRequestFacts:
    argv_transform = compatibility_kwargs.pop("argv_transform", None)
    resolved_invocation_dir = _resolve_invocation_dir(
        invocation_dir=invocation_dir,
        compatibility_kwargs=compatibility_kwargs,
        context=context,
        public_invocation_dir_name=public_invocation_dir_name,
    )
    normalized_request = _provider_selection_request_facts(
        provider_selection=provider_selection,
        invocation_dir=resolved_invocation_dir,
        tool_access=tool_access,
        tool_policy=tool_policy,
        missing_sentinel=missing_sentinel,
        context=context,
        missing_message=missing_message,
        workspace_name=public_invocation_dir_name,
    )
    return _EphemeralRunRequestFacts(
        invocation_dir=normalized_request.invocation_dir,
        provider_selection=normalized_request.provider_selection,
        tool_access=normalized_request.tool_access,
        argv_transform=argv_transform,
    )


def _new_session_run_request_facts(
    *,
    invocation_dir: Path | None,
    compatibility_kwargs: dict[str, Any],
    provider_selection: ProviderSelection | None,
    tool_access: Any,
    tool_policy: Any,
    session_store: Path | None,
    missing_sentinel: object,
    context: str,
    missing_message: str,
    public_invocation_dir_name: str,
) -> _NewSessionRunRequestFacts:
    argv_transform = compatibility_kwargs.pop("argv_transform", None)
    compatibility_runtime_state_dir = compatibility_kwargs.pop(
        "runtime_state_dir",
        session_store,
    )
    if session_store is not None and compatibility_runtime_state_dir != session_store:
        raise TypeError(
            f"{context} received conflicting `runtime_state_dir` and `session_store` values."
        )
    resolved_invocation_dir = _resolve_invocation_dir(
        invocation_dir=invocation_dir,
        compatibility_kwargs=compatibility_kwargs,
        context=context,
        public_invocation_dir_name=public_invocation_dir_name,
    )
    normalized_request = _provider_selection_request_facts(
        provider_selection=provider_selection,
        invocation_dir=resolved_invocation_dir,
        tool_access=tool_access,
        tool_policy=tool_policy,
        missing_sentinel=missing_sentinel,
        context=context,
        missing_message=missing_message,
        workspace_name=public_invocation_dir_name,
    )
    return _NewSessionRunRequestFacts(
        invocation_dir=normalized_request.invocation_dir,
        provider_selection=normalized_request.provider_selection,
        tool_access=normalized_request.tool_access,
        session_store=compatibility_runtime_state_dir,
        argv_transform=argv_transform,
    )


def _resumed_session_run_request_facts(
    *,
    invocation_dir: Path | None,
    compatibility_kwargs: dict[str, Any],
    continuation: Any,
    tool_access: Any,
    session_store: Path | None,
    context: str,
    public_invocation_dir_name: str,
) -> _ResumedLifecycleRequestFacts:
    argv_transform = compatibility_kwargs.pop("argv_transform", None)
    compatibility_runtime_state_dir = compatibility_kwargs.pop(
        "runtime_state_dir",
        session_store,
    )
    if session_store is not None and compatibility_runtime_state_dir != session_store:
        raise TypeError(
            f"{context} received conflicting `runtime_state_dir` and `session_store` values."
        )
    resolved_invocation_dir = _resolve_invocation_dir(
        invocation_dir=invocation_dir,
        compatibility_kwargs=compatibility_kwargs,
        context=context,
        public_invocation_dir_name=public_invocation_dir_name,
    )
    if continuation is None:
        raise TypeError(f"{context} requires a `continuation` value.")
    if isinstance(tool_access, ToolAccess):
        raise TypeError(
            f"{context} derives fixed tool access from `continuation` and does not accept `tool_access` or `tool_policy` overrides."
        )
    try:
        continuation_resume_facts = continuation.resume_facts
    except TypeError as exc:
        raise RuntimeConfigurationError(str(exc)) from exc
    normalized_request = normalize_continuation_request(
        invocation_dir=resolved_invocation_dir,
        tool_access=continuation_resume_facts.tool_access,
        context=context,
        workspace_name=public_invocation_dir_name,
    )
    return _ResumedLifecycleRequestFacts(
        invocation_dir=normalized_request.invocation_dir,
        tool_access=normalized_request.tool_access,
        model=continuation_resume_facts.selected.model,
        effort=continuation_resume_facts.selected.effort,
        session_store=compatibility_runtime_state_dir,
        argv_transform=argv_transform,
    )
