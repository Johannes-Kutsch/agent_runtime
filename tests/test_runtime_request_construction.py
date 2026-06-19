from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

import agent_runtime as runtime
import agent_runtime.execution_contracts as execution_contracts_runtime
import agent_runtime.runtime as prompt_runtime
from agent_runtime.contracts import ExecutionProvider
from agent_runtime.execution_contracts import WorktreeMount
from agent_runtime.roles import InvocationRole
from agent_runtime.session import RunKind
from agent_runtime.session_planning import (
    AuthSeedingRequirement,
    ResumableSessionPlan,
)

from tests.runtime_boundary_fakes import ExecutionServiceFake as _ExecutionService


def test_ephemeral_run_request_uses_override_stage_selection_when_stage_missing(
    stage_selection_factory: Callable[..., runtime.StageSelection],
) -> None:
    override_stage = stage_selection_factory(
        service="claude",
        model="sonnet",
        effort="high",
    )

    request = prompt_runtime.EphemeralRunRequest(
        prompt="already rendered prompt",
        worktree=Path("/repo"),
        override=override_stage,
        role=InvocationRole("implementer"),
        tool_access=runtime.ToolAccess.no_tools(),
    )

    assert request.stage == override_stage
    assert request.override == override_stage


def test_ephemeral_run_request_rejects_conflicting_stage_selection_and_override(
    stage_selection_factory: Callable[..., runtime.StageSelection],
) -> None:
    with pytest.raises(
        TypeError,
        match=re.escape(
            "EphemeralRunRequest received conflicting `stage` and `override` values."
        ),
    ):
        prompt_runtime.EphemeralRunRequest(
            prompt="already rendered prompt",
            worktree=Path("/repo"),
            stage=stage_selection_factory(service="codex"),
            override=stage_selection_factory(service="claude"),
            role=InvocationRole("implementer"),
            tool_access=runtime.ToolAccess.no_tools(),
        )


def test_new_session_run_request_requires_invocation_role(
    stage_selection_factory: Callable[..., runtime.StageSelection],
) -> None:
    with pytest.raises(
        TypeError,
        match=re.escape("NewSessionRunRequest requires a `role` value."),
    ):
        prompt_runtime.NewSessionRunRequest(
            prompt="already rendered prompt",
            worktree=Path("/repo"),
            stage=stage_selection_factory(),
            tool_access=runtime.ToolAccess.no_tools(),
        )


def test_prompt_run_request_uses_compatibility_tool_policy_for_workspace_backed_tool_access(
    stage_selection_factory: Callable[..., runtime.StageSelection],
) -> None:
    request = execution_contracts_runtime.PromptRunRequest(
        prompt="already rendered prompt",
        worktree=WorktreeMount(Path("/repo")),
        stage=stage_selection_factory(service="codex"),
        role=InvocationRole("implementer"),
        tool_policy=runtime.ToolPolicy.PARTIAL,
    )

    assert request.tool_access == runtime.ToolAccess.workspace_backed(
        Path("/repo"),
        tool_policy=runtime.ToolPolicy.PARTIAL,
    )
    assert request.tool_policy is runtime.ToolPolicy.PARTIAL


def test_ephemeral_run_request_uses_compatibility_tool_policy_for_workspace_backed_tool_access(
    stage_selection_factory: Callable[..., runtime.StageSelection],
) -> None:
    request = prompt_runtime.EphemeralRunRequest(
        prompt="already rendered prompt",
        worktree=Path("/repo"),
        stage=stage_selection_factory(service="codex"),
        role=InvocationRole("implementer"),
        tool_policy=runtime.ToolPolicy.FULL,
    )

    assert request.tool_access == runtime.ToolAccess.workspace_backed(Path("/repo"))
    assert request.tool_policy is runtime.ToolPolicy.FULL


def test_new_session_run_request_uses_compatibility_tool_policy_for_workspace_backed_tool_access(
    stage_selection_factory: Callable[..., runtime.StageSelection],
) -> None:
    request = prompt_runtime.NewSessionRunRequest(
        prompt="already rendered prompt",
        worktree=Path("/repo"),
        stage=stage_selection_factory(service="codex"),
        role=InvocationRole("implementer"),
        tool_policy=runtime.ToolPolicy.PARTIAL,
    )

    assert request.tool_access == runtime.ToolAccess.workspace_backed(
        Path("/repo"),
        tool_policy=runtime.ToolPolicy.PARTIAL,
    )
    assert request.tool_policy is runtime.ToolPolicy.PARTIAL


def test_resumed_session_run_request_uses_compatibility_tool_policy_for_workspace_backed_tool_access() -> (
    None
):
    request = prompt_runtime.ResumedSessionRunRequest(
        prompt="already rendered prompt",
        worktree=Path("/repo"),
        model="gpt-5.4",
        effort="medium",
        session_plan=ResumableSessionPlan(
            role=InvocationRole("reviewer"),
            worktree=Path("/repo"),
            namespace="main",
            service=cast(ExecutionProvider, _ExecutionService("codex")),
            run_kind=RunKind.FRESH,
            provider_state_dir=None,
            provider_session_id=None,
            auth_seeding_requirement=AuthSeedingRequirement.NOT_REQUIRED,
        ),
        tool_policy=runtime.ToolPolicy.PARTIAL,
    )

    assert request.tool_access == runtime.ToolAccess.workspace_backed(
        Path("/repo"),
        tool_policy=runtime.ToolPolicy.PARTIAL,
    )
    assert request.tool_policy is runtime.ToolPolicy.PARTIAL


def test_prompt_run_request_rejects_workspace_backed_tool_access_for_other_worktree(
    stage_selection_factory: Callable[..., runtime.StageSelection],
) -> None:
    with pytest.raises(
        ValueError,
        match=re.escape(
            "PromptRunRequest workspace-backed tool access requires worktree /other, got /repo."
        ),
    ):
        execution_contracts_runtime.PromptRunRequest(
            prompt="already rendered prompt",
            worktree=WorktreeMount(Path("/repo")),
            stage=stage_selection_factory(service="codex"),
            role=InvocationRole("implementer"),
            tool_access=runtime.ToolAccess.workspace_backed(Path("/other")),
        )


@pytest.mark.parametrize(
    ("request_factory", "expected_message"),
    [
        (
            lambda stage_selection_factory: prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                worktree=Path("/repo"),
                stage=stage_selection_factory(service="codex"),
                role=InvocationRole("implementer"),
            ),
            "EphemeralRunRequest requires an explicit `tool_access` value.",
        ),
        (
            lambda stage_selection_factory: prompt_runtime.NewSessionRunRequest(
                prompt="already rendered prompt",
                worktree=Path("/repo"),
                stage=stage_selection_factory(service="codex"),
                role=InvocationRole("implementer"),
            ),
            "NewSessionRunRequest requires an explicit `tool_access` value.",
        ),
    ],
)
def test_lifecycle_request_construction_requires_explicit_tool_access(
    stage_selection_factory: Callable[..., runtime.StageSelection],
    request_factory: Callable[
        [Callable[..., runtime.StageSelection]],
        object,
    ],
    expected_message: str,
) -> None:
    with pytest.raises(TypeError, match=re.escape(expected_message)):
        request_factory(stage_selection_factory)


def test_resumed_session_run_request_coerces_path_worktree_to_worktree_mount() -> None:
    request = prompt_runtime.ResumedSessionRunRequest(
        prompt="already rendered prompt",
        worktree=Path("/repo"),
        model="gpt-5.4",
        effort="medium",
        session_plan=ResumableSessionPlan(
            role=InvocationRole("reviewer"),
            worktree=Path("/repo"),
            namespace="main",
            service=cast(ExecutionProvider, _ExecutionService("codex")),
            run_kind=RunKind.FRESH,
            provider_state_dir=None,
            provider_session_id=None,
            auth_seeding_requirement=AuthSeedingRequirement.NOT_REQUIRED,
        ),
        tool_access=runtime.ToolAccess.no_tools(),
    )

    assert request.worktree == WorktreeMount(Path("/repo"))
    assert request.mount_path == Path("/repo")


def test_runtime_public_surface_keeps_request_normalization_module_private() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "_request_normalization" not in runtime.__all__
    assert "_request_normalization" not in prompt_runtime.__all__
    assert "_request_normalization" not in execution_contracts_runtime.__all__
    assert "_request_normalization" not in readme
