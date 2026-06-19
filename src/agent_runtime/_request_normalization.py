from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from .roles import InvocationRole
from .types import StageSelection, validate_stage_selection

if TYPE_CHECKING:
    from .contracts import ToolAccess, ToolPolicy, ToolPolicyProfile
    from .execution_contracts import WorktreeMount


def normalize_stage_selection(
    stage: StageSelection | None,
    *,
    override: StageSelection | None,
    context: str,
    validate: bool = True,
) -> StageSelection:
    if stage is None:
        stage = override
    elif override is not None and override != stage:
        raise TypeError(
            f"{context} received conflicting `stage` and `override` values."
        )
    if stage is None:
        raise TypeError(f"{context} requires a `stage` value.")
    if validate:
        validate_stage_selection(stage)
    return stage


def require_invocation_role(
    role: InvocationRole | None,
    *,
    context: str,
    message: str | None = None,
) -> InvocationRole:
    if role is None:
        raise TypeError(message or f"{context} requires a `role` value.")
    return role


def normalize_worktree_path(worktree: Path | WorktreeMount) -> Path:
    from .execution_contracts import WorktreeMount

    if isinstance(worktree, WorktreeMount):
        return worktree.host_path
    return worktree


def normalize_worktree_mount(worktree: Path | WorktreeMount) -> WorktreeMount:
    from .execution_contracts import WorktreeMount

    if isinstance(worktree, WorktreeMount):
        return worktree
    return WorktreeMount(worktree)


def normalize_tool_access(
    *,
    tool_access: Any,
    tool_policy: Any,
    missing_sentinel: object,
    workspace: Path,
    context: str,
    missing_message: str,
) -> ToolAccess:
    from .contracts import ToolAccess, ToolPolicy, ToolPolicyProfile

    if isinstance(tool_access, ToolAccess) and tool_policy is not missing_sentinel:
        raise TypeError(
            f"{context} received conflicting `tool_access` and `tool_policy` values."
        )
    if isinstance(tool_access, ToolAccess):
        resolved_tool_access = tool_access
    elif tool_policy is not missing_sentinel:
        resolved_tool_access = ToolAccess.workspace_backed(
            workspace,
            tool_policy=cast(ToolPolicy | ToolPolicyProfile, tool_policy),
        )
    else:
        raise TypeError(missing_message)
    resolved_tool_access.require_workspace(workspace, context=context)
    return resolved_tool_access


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
