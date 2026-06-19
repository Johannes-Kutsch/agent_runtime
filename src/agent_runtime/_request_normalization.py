from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from .identity import validate_session_namespace
from .roles import InvocationRole
from .types import StageSelection, validate_stage_selection

if TYPE_CHECKING:
    from .contracts import ToolAccess, ToolPolicy, ToolPolicyProfile
    from .execution_contracts import WorktreeMount


@dataclasses.dataclass(frozen=True)
class NormalizedWorktree:
    path: Path
    mount: WorktreeMount


@dataclasses.dataclass(frozen=True)
class NormalizedStageRequest:
    stage: StageSelection
    role: InvocationRole
    worktree: NormalizedWorktree
    tool_access: ToolAccess
    session_namespace: str


@dataclasses.dataclass(frozen=True)
class NormalizedResumedRequest:
    role: InvocationRole
    worktree: NormalizedWorktree
    tool_access: ToolAccess
    session_namespace: str


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


def normalize_session_namespace(session_namespace: str) -> str:
    validate_session_namespace(session_namespace)
    return session_namespace


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


def normalize_worktree(worktree: Path | WorktreeMount) -> NormalizedWorktree:
    return NormalizedWorktree(
        path=normalize_worktree_path(worktree),
        mount=normalize_worktree_mount(worktree),
    )


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
    )


def normalize_resolved_tool_access(
    *,
    tool_access: "ToolAccess",
    workspace: Path | None,
    context: str,
) -> "ToolAccess":
    tool_access.require_workspace(workspace, context=context)
    return tool_access


def normalize_stage_request(
    *,
    stage: StageSelection | None,
    override: StageSelection | None,
    role: InvocationRole | None,
    worktree: Path | WorktreeMount,
    tool_access: Any,
    tool_policy: Any,
    missing_sentinel: object,
    session_namespace: str,
    context: str,
    missing_message: str,
    validate_stage: bool = True,
) -> NormalizedStageRequest:
    normalized_worktree = normalize_worktree(worktree)
    return NormalizedStageRequest(
        stage=normalize_stage_selection(
            stage,
            override=override,
            context=context,
            validate=validate_stage,
        ),
        role=require_invocation_role(role, context=context),
        worktree=normalized_worktree,
        tool_access=normalize_tool_access(
            tool_access=tool_access,
            tool_policy=tool_policy,
            missing_sentinel=missing_sentinel,
            workspace=normalized_worktree.path,
            context=context,
            missing_message=missing_message,
        ),
        session_namespace=normalize_session_namespace(session_namespace),
    )


def normalize_continuation_request(
    *,
    role: InvocationRole | None,
    worktree: Path | WorktreeMount,
    tool_access: "ToolAccess",
    session_namespace: str,
    context: str,
    role_message: str,
) -> NormalizedResumedRequest:
    normalized_worktree = normalize_worktree(worktree)
    return NormalizedResumedRequest(
        role=require_invocation_role(role, context=context, message=role_message),
        worktree=normalized_worktree,
        tool_access=normalize_resolved_tool_access(
            tool_access=tool_access,
            workspace=normalized_worktree.path,
            context=context,
        ),
        session_namespace=normalize_session_namespace(session_namespace),
    )


def normalize_session_plan_request(
    *,
    role: InvocationRole,
    worktree: Path | WorktreeMount,
    tool_access: Any,
    tool_policy: Any,
    missing_sentinel: object,
    session_namespace: str,
    context: str,
    missing_message: str,
) -> NormalizedResumedRequest:
    normalized_worktree = normalize_worktree(worktree)
    return NormalizedResumedRequest(
        role=role,
        worktree=normalized_worktree,
        tool_access=normalize_tool_access(
            tool_access=tool_access,
            tool_policy=tool_policy,
            missing_sentinel=missing_sentinel,
            workspace=normalized_worktree.path,
            context=context,
            missing_message=missing_message,
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
