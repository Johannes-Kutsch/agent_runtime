from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from .identity import validate_session_namespace
from .types import (
    ProviderSelection,
    validate_provider_selection,
)

if TYPE_CHECKING:
    from .contracts import ToolAccess, ToolPolicy, ToolPolicyProfile


@dataclasses.dataclass(frozen=True)
class NormalizedWorktree:
    path: Path


@dataclasses.dataclass(frozen=True)
class NormalizedProviderSelectionRequest:
    provider_selection: ProviderSelection
    worktree: NormalizedWorktree
    tool_access: ToolAccess
    session_namespace: str


@dataclasses.dataclass(frozen=True)
class NormalizedResumedRequest:
    worktree: NormalizedWorktree
    tool_access: ToolAccess
    session_namespace: str


NormalizedStageRequest = NormalizedProviderSelectionRequest


def normalize_provider_selection(
    provider_selection: ProviderSelection | None,
    *,
    context: str,
    validate: bool = True,
) -> ProviderSelection:
    if provider_selection is None:
        raise TypeError(f"{context} requires a `provider_selection` value.")
    if validate:
        validate_provider_selection(provider_selection)
    return provider_selection


def normalize_session_namespace(session_namespace: str) -> str:
    validate_session_namespace(session_namespace)
    return session_namespace


def normalize_worktree(worktree: Path) -> NormalizedWorktree:
    return NormalizedWorktree(path=worktree)


def normalize_tool_access(
    *,
    tool_access: Any,
    tool_policy: Any,
    missing_sentinel: object,
    workspace: Path,
    context: str,
    missing_message: str,
    workspace_name: str = "worktree",
) -> ToolAccess:
    from .contracts import ToolAccess, ToolPolicy, ToolPolicyProfile

    if isinstance(tool_access, ToolAccess) and tool_policy is not missing_sentinel:
        raise TypeError(
            f"{context} received conflicting `tool_access` and `tool_policy` values."
        )
    if isinstance(tool_access, ToolAccess):
        resolved_tool_access = tool_access
    elif tool_policy is not missing_sentinel:
        resolved_tool_policy = cast(ToolPolicy | ToolPolicyProfile, tool_policy)
        if resolved_tool_policy is ToolPolicy.NONE:
            resolved_tool_access = ToolAccess.no_tools()
        else:
            resolved_tool_access = ToolAccess.workspace_backed(
                workspace,
                tool_policy=resolved_tool_policy,
            )
    else:
        raise TypeError(missing_message)
    return normalize_resolved_tool_access(
        tool_access=resolved_tool_access,
        workspace=workspace,
        context=context,
        workspace_name=workspace_name,
    )


def normalize_resolved_tool_access(
    *,
    tool_access: "ToolAccess",
    workspace: Path | None,
    context: str,
    workspace_name: str = "worktree",
) -> "ToolAccess":
    tool_access.require_workspace(
        workspace,
        context=context,
        workspace_name=workspace_name,
    )
    return tool_access


def normalize_provider_selection_request(
    *,
    provider_selection: ProviderSelection | None,
    worktree: Path,
    tool_access: Any,
    tool_policy: Any,
    missing_sentinel: object,
    session_namespace: str,
    context: str,
    missing_message: str,
    validate_stage: bool = True,
    workspace_name: str = "worktree",
) -> NormalizedProviderSelectionRequest:
    normalized_worktree = normalize_worktree(worktree)
    return NormalizedProviderSelectionRequest(
        provider_selection=normalize_provider_selection(
            provider_selection,
            context=context,
            validate=validate_stage,
        ),
        worktree=normalized_worktree,
        tool_access=normalize_tool_access(
            tool_access=tool_access,
            tool_policy=tool_policy,
            missing_sentinel=missing_sentinel,
            workspace=normalized_worktree.path,
            context=context,
            missing_message=missing_message,
            workspace_name=workspace_name,
        ),
        session_namespace=normalize_session_namespace(session_namespace),
    )


def normalize_continuation_request(
    *,
    worktree: Path,
    tool_access: "ToolAccess",
    session_namespace: str,
    context: str,
    workspace_name: str = "worktree",
) -> NormalizedResumedRequest:
    normalized_worktree = normalize_worktree(worktree)
    return NormalizedResumedRequest(
        worktree=normalized_worktree,
        tool_access=normalize_resolved_tool_access(
            tool_access=tool_access,
            workspace=normalized_worktree.path,
            context=context,
            workspace_name=workspace_name,
        ),
        session_namespace=normalize_session_namespace(session_namespace),
    )


def normalize_tool_policy(
    *,
    tool_access: Any,
    tool_policy: Any,
    missing_sentinel: object,
    workspace: Path | None,
    context: str,
    missing_message: str,
) -> "ToolPolicy | ToolPolicyProfile":
    from .contracts import ToolAccess, ToolPolicy, ToolPolicyProfile

    if isinstance(tool_access, ToolAccess):
        tool_access.require_workspace(workspace, context=context)
        return tool_access.tool_policy
    if tool_policy is missing_sentinel:
        raise TypeError(missing_message)
    return cast(ToolPolicy | ToolPolicyProfile, tool_policy)
