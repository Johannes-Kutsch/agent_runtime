from __future__ import annotations

from pathlib import Path

from .execution_contracts import WorktreeMount
from .roles import InvocationRole
from .types import StageSelection, validate_stage_selection


def normalize_stage_selection(
    stage: StageSelection | None,
    *,
    override: StageSelection | None,
    context: str,
) -> StageSelection:
    if stage is None:
        stage = override
    elif override is not None and override != stage:
        raise TypeError(
            f"{context} received conflicting `stage` and `override` values."
        )
    if stage is None:
        raise TypeError(f"{context} requires a `stage` value.")
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
    if isinstance(worktree, WorktreeMount):
        return worktree.host_path
    return worktree


def normalize_worktree_mount(worktree: Path | WorktreeMount) -> WorktreeMount:
    if isinstance(worktree, WorktreeMount):
        return worktree
    return WorktreeMount(worktree)
