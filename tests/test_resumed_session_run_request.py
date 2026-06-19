from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, cast

import pytest

import agent_runtime as runtime
import agent_runtime.runtime as prompt_runtime
from agent_runtime.contracts import ExecutionProvider
from agent_runtime.execution_contracts import WorktreeMount
from agent_runtime.roles import InvocationRole
from agent_runtime.session import RunKind
from agent_runtime.session_planning import (
    AuthSeedingRequirement,
    ResumableSessionPlan,
    ResumableSessionPlanRequest,
    plan_resumable_session,
)

from tests.runtime_boundary_fakes import (
    ExecutionServiceFake as _ExecutionService,
    ResidentPlanningProviderSessionAdapterFake as _ResidentPlanningProviderSessionAdapter,
    SessionStoreFake as _SessionStore,
)


def test_resumed_session_run_request_from_continuation_rejects_tool_access_override() -> (
    None
):
    with pytest.raises(
        TypeError,
        match=re.escape(
            "ResumedSessionRunRequest derives fixed tool access from `continuation` and does not accept `tool_access` or `tool_policy` overrides."
        ),
    ):
        prompt_runtime.ResumedSessionRunRequest(
            prompt="already rendered prompt",
            worktree=WorktreeMount(Path("/repo")),
            role=InvocationRole("implementer"),
            session_namespace="main",
            continuation=prompt_runtime.Continuation(
                selected_service="codex",
                selected_model="gpt-5.4",
                selected_effort="medium",
                tool_access=runtime.ToolAccess.no_tools(),
                provider_resume_state={"run_kind": "resume"},
            ),
            tool_access=runtime.ToolAccess.workspace_backed(Path("/repo")),
        )


def test_resumed_session_run_request_from_continuation_rejects_tool_policy_override_before_validating_session_namespace() -> (
    None
):
    with pytest.raises(
        TypeError,
        match=re.escape(
            "ResumedSessionRunRequest derives fixed tool access from `continuation` and does not accept `tool_access` or `tool_policy` overrides."
        ),
    ):
        prompt_runtime.ResumedSessionRunRequest(
            prompt="already rendered prompt",
            worktree=WorktreeMount(Path("/repo")),
            role=InvocationRole("implementer"),
            session_namespace="../escape",
            continuation=prompt_runtime.Continuation(
                selected_service="codex",
                selected_model="gpt-5.4",
                selected_effort="medium",
                tool_access=runtime.ToolAccess.no_tools(),
                provider_resume_state={"run_kind": "resume"},
            ),
            tool_policy=runtime.ToolPolicy.FULL,
        )


def test_resumed_session_run_request_from_continuation_rejects_tool_policy_override_before_validating_workspace_backed_tool_access() -> (
    None
):
    with pytest.raises(
        TypeError,
        match=re.escape(
            "ResumedSessionRunRequest derives fixed tool access from `continuation` and does not accept `tool_access` or `tool_policy` overrides."
        ),
    ):
        prompt_runtime.ResumedSessionRunRequest(
            prompt="already rendered prompt",
            worktree=WorktreeMount(Path("/other")),
            role=InvocationRole("implementer"),
            continuation=prompt_runtime.Continuation(
                selected_service="codex",
                selected_model="gpt-5.4",
                selected_effort="medium",
                tool_access=runtime.ToolAccess.workspace_backed(
                    Path("/repo"),
                    tool_policy=runtime.ToolPolicy.PARTIAL,
                ),
                provider_resume_state={"run_kind": "resume"},
            ),
            tool_policy=runtime.ToolPolicy.FULL,
        )


def test_resumed_session_run_request_from_continuation_requires_role() -> None:
    with pytest.raises(
        TypeError,
        match=re.escape(
            "ResumedSessionRunRequest requires a `role` value when constructed from a continuation."
        ),
    ):
        prompt_runtime.ResumedSessionRunRequest(
            prompt="already rendered prompt",
            worktree=WorktreeMount(Path("/repo")),
            session_namespace="main",
            continuation=prompt_runtime.Continuation(
                selected_service="codex",
                selected_model="gpt-5.4",
                selected_effort="medium",
                tool_access=runtime.ToolAccess.no_tools(),
                provider_resume_state={"run_kind": "resume"},
            ),
        )


@pytest.mark.parametrize("label", ["", "../escape"])
def test_resumed_session_run_request_from_continuation_preserves_empty_session_namespace_and_rejects_unsafe_non_empty_values(
    label: str,
) -> None:
    continuation = prompt_runtime.Continuation(
        selected_service="codex",
        selected_model="gpt-5.4",
        selected_effort="medium",
        tool_access=runtime.ToolAccess.no_tools(),
        provider_resume_state={"run_kind": "resume"},
    )

    if label == "":
        request = prompt_runtime.ResumedSessionRunRequest(
            prompt="already rendered prompt",
            worktree=WorktreeMount(Path("/repo")),
            role=InvocationRole("implementer"),
            session_namespace=label,
            continuation=continuation,
        )

        assert request.session_namespace == ""
        return

    with pytest.raises(ValueError):
        prompt_runtime.ResumedSessionRunRequest(
            prompt="already rendered prompt",
            worktree=WorktreeMount(Path("/repo")),
            role=InvocationRole("implementer"),
            session_namespace=label,
            continuation=continuation,
        )


def test_resumed_session_run_request_rejects_conflicting_continuation_and_session_plan(
    session_store_factory: Callable[..., _SessionStore],
    resident_provider_session_adapter: _ResidentPlanningProviderSessionAdapter,
) -> None:
    service = cast(ExecutionProvider, _ExecutionService("codex"))
    session_plan = plan_resumable_session(
        ResumableSessionPlanRequest(
            worktree=Path("/repo"),
            role=InvocationRole("implementer"),
            namespace="main",
            service=service,
            session_store=session_store_factory(),
            provider_session_adapter=resident_provider_session_adapter,
        )
    )

    with pytest.raises(
        TypeError,
        match=re.escape(
            "ResumedSessionRunRequest received conflicting `session_plan` and `continuation` values."
        ),
    ):
        prompt_runtime.ResumedSessionRunRequest(
            prompt="already rendered prompt",
            worktree=WorktreeMount(Path("/repo")),
            model="gpt-5.4",
            effort="medium",
            session_plan=session_plan,
            continuation=prompt_runtime.Continuation(
                selected_service="codex",
                selected_model="gpt-5.4",
                selected_effort="medium",
                tool_access=runtime.ToolAccess.no_tools(),
                provider_resume_state={"run_kind": "resume"},
            ),
            role=InvocationRole("implementer"),
            tool_policy=runtime.ToolPolicy.FULL,
        )


def test_resumed_session_run_request_rejects_conflicting_tool_access_and_tool_policy() -> (
    None
):
    with pytest.raises(
        TypeError,
        match=re.escape(
            "ResumedSessionRunRequest received conflicting `tool_access` and `tool_policy` values."
        ),
    ):
        prompt_runtime.ResumedSessionRunRequest(
            prompt="already rendered prompt",
            worktree=WorktreeMount(Path("/repo")),
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
            tool_policy=runtime.ToolPolicy.FULL,
        )


def test_resumed_session_run_request_carries_workspace_backed_tool_access() -> None:
    tool_access = runtime.ToolAccess.workspace_backed(
        Path("/repo"),
        tool_policy=runtime.ToolPolicy.PARTIAL,
    )

    request = prompt_runtime.ResumedSessionRunRequest(
        prompt="already rendered prompt",
        worktree=WorktreeMount(Path("/repo")),
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
        tool_access=tool_access,
    )

    assert request.tool_access == tool_access
    assert request.tool_access.workspace == Path("/repo")


def test_resumed_session_run_request_accepts_explicit_no_tools_tool_access() -> None:
    request = prompt_runtime.ResumedSessionRunRequest(
        prompt="already rendered prompt",
        worktree=WorktreeMount(Path("/repo")),
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

    assert request.tool_access == runtime.ToolAccess.no_tools()
    assert request.tool_policy == runtime.ToolAccess.no_tools().tool_policy


def test_resumed_session_run_request_rejects_workspace_backed_tool_access_for_other_worktree() -> (
    None
):
    with pytest.raises(
        ValueError,
        match=re.escape(
            "ResumedSessionRunRequest workspace-backed tool access requires worktree /repo, got /other."
        ),
    ):
        prompt_runtime.ResumedSessionRunRequest(
            prompt="already rendered prompt",
            worktree=WorktreeMount(Path("/other")),
            model="gpt-5.4",
            effort="medium",
            session_plan=ResumableSessionPlan(
                role=InvocationRole("reviewer"),
                worktree=Path("/other"),
                namespace="main",
                service=cast(ExecutionProvider, _ExecutionService("codex")),
                run_kind=RunKind.FRESH,
                provider_state_dir=None,
                provider_session_id=None,
                auth_seeding_requirement=AuthSeedingRequirement.NOT_REQUIRED,
            ),
            tool_access=runtime.ToolAccess.workspace_backed(
                Path("/repo"),
                tool_policy=runtime.ToolPolicy.FULL,
            ),
        )


def test_resumed_session_run_request_from_continuation_rejects_workspace_backed_tool_access_for_other_worktree() -> (
    None
):
    with pytest.raises(
        ValueError,
        match=re.escape(
            "ResumedSessionRunRequest workspace-backed tool access requires worktree /repo, got /other."
        ),
    ):
        prompt_runtime.ResumedSessionRunRequest(
            prompt="already rendered prompt",
            worktree=WorktreeMount(Path("/other")),
            role=InvocationRole("implementer"),
            session_namespace="main",
            continuation=prompt_runtime.Continuation(
                selected_service="codex",
                selected_model="gpt-5.4",
                selected_effort="medium",
                tool_access=runtime.ToolAccess.workspace_backed(
                    Path("/repo"),
                    tool_policy=runtime.ToolPolicy.PARTIAL,
                ),
                provider_resume_state={"run_kind": "resume"},
            ),
        )


def test_resumed_session_run_request_rejects_request_level_invocation_role() -> None:
    with pytest.raises(TypeError):
        prompt_runtime.ResumedSessionRunRequest(
            prompt="already rendered prompt",
            worktree=WorktreeMount(Path(".")),
            model="gpt-5.4",
            effort="medium",
            session_plan=ResumableSessionPlan(
                role=InvocationRole("reviewer"),
                worktree=Path("."),
                namespace="main",
                service=cast(ExecutionProvider, _ExecutionService("codex")),
                run_kind=RunKind.FRESH,
                provider_state_dir=None,
                provider_session_id=None,
                auth_seeding_requirement=AuthSeedingRequirement.NOT_REQUIRED,
            ),
            tool_policy=runtime.ToolPolicy.FULL,
            role=InvocationRole("implementer"),
        )  # type: ignore[call-arg]
