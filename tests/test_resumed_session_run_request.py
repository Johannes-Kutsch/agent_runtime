from __future__ import annotations

import re
from pathlib import Path
from typing import cast

import pytest

import agent_runtime as runtime
import agent_runtime.contracts as contracts_runtime
import agent_runtime.runtime as prompt_runtime
from agent_runtime.contracts import ExecutionProvider
from agent_runtime.session import RunKind
from agent_runtime.session_planning import (
    ResumableSessionPlan,
)

from tests.runtime_boundary_fakes import ExecutionServiceFake as _ExecutionService


def _session_plan(*, worktree: Path = Path("/repo")) -> ResumableSessionPlan:
    return ResumableSessionPlan(
        worktree=worktree,
        namespace="main",
        service=cast(ExecutionProvider, _ExecutionService("codex")),
        run_kind=RunKind.FRESH,
        provider_state_dir=None,
        provider_session_id=None,
    )


def _continuation(
    *,
    tool_access: contracts_runtime.ToolAccess | None = None,
) -> prompt_runtime.Continuation:
    return prompt_runtime.Continuation(
        selected_service="codex",
        selected_model="gpt-5.4",
        selected_effort="medium",
        tool_access=tool_access or contracts_runtime.ToolAccess.no_tools(),
        provider_resume_state={"run_kind": "resume"},
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
            invocation_dir=Path("/repo"),
            continuation=_continuation(),
            tool_access=contracts_runtime.ToolAccess.workspace_backed(Path("/repo")),
        )


def test_resumed_session_run_request_from_continuation_accepts_minimal_fields() -> None:
    request = prompt_runtime.ResumedSessionRunRequest(
        prompt="already rendered prompt",
        invocation_dir=Path("/repo"),
        continuation=_continuation(),
    )

    assert request.model == "gpt-5.4"
    assert request.effort == "medium"
    assert request.provider_auth is None
    assert request.token is None
    assert request.tool_access == contracts_runtime.ToolAccess.no_tools()
    assert request.invocation_dir == Path("/repo")


def test_resumed_session_run_request_from_continuation_rejects_model_override() -> None:
    with pytest.raises(
        TypeError,
        match=re.escape(
            "ResumedSessionRunRequest got an unexpected keyword argument 'model'"
        ),
    ):
        prompt_runtime.ResumedSessionRunRequest(
            prompt="already rendered prompt",
            invocation_dir=Path("/repo"),
            continuation=_continuation(),
            model="gpt-5.5",
        )


def test_resumed_session_run_request_from_continuation_rejects_effort_override() -> (
    None
):
    with pytest.raises(
        TypeError,
        match=re.escape(
            "ResumedSessionRunRequest got an unexpected keyword argument 'effort'"
        ),
    ):
        prompt_runtime.ResumedSessionRunRequest(
            prompt="already rendered prompt",
            invocation_dir=Path("/repo"),
            continuation=_continuation(),
            effort="high",
        )


@pytest.mark.parametrize("label", ["", "../escape"])
def test_resumed_session_run_request_from_continuation_preserves_empty_internal_session_namespace_and_rejects_unsafe_values(
    label: str,
) -> None:
    if label == "":
        request = prompt_runtime.ResumedSessionRunRequest(
            prompt="already rendered prompt",
            invocation_dir=Path("/repo"),
            _session_namespace=label,
            continuation=_continuation(),
        )

        assert request._session_namespace == ""
        return

    with pytest.raises(ValueError):
        prompt_runtime.ResumedSessionRunRequest(
            prompt="already rendered prompt",
            invocation_dir=Path("/repo"),
            _session_namespace=label,
            continuation=_continuation(),
        )


def test_resumed_session_run_request_from_opaque_continuation_defaults_model_effort_and_tool_access() -> (
    None
):
    request = prompt_runtime.ResumedSessionRunRequest(
        prompt="already rendered prompt",
        invocation_dir=Path("/repo"),
        continuation=_continuation(
            tool_access=contracts_runtime.ToolAccess.workspace_backed(
                Path("/repo"),
                tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
            )
        ),
    )

    assert request.model == "gpt-5.4"
    assert request.effort == "medium"
    assert request.tool_access == contracts_runtime.ToolAccess.workspace_backed(
        Path("/repo"),
        tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
    )
    assert request.tool_policy == runtime.ToolPolicy.NO_FILE_MUTATION


def test_resumed_session_run_request_rejects_conflicting_continuation_and_session_plan() -> (
    None
):
    with pytest.raises(
        TypeError,
        match=re.escape(
            "ResumedSessionRunRequest got an unexpected keyword argument 'session_plan'"
        ),
    ):
        prompt_runtime.ResumedSessionRunRequest(
            prompt="already rendered prompt",
            invocation_dir=Path("/repo"),
            continuation=_continuation(),
            session_plan=_session_plan(),
        )


def test_resumed_session_run_request_rejects_conflicting_tool_access_and_tool_policy() -> (
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
            invocation_dir=Path("/repo"),
            continuation=_continuation(),
            tool_access=contracts_runtime.ToolAccess.no_tools(),
        )


def test_resumed_session_run_request_carries_workspace_backed_tool_access() -> None:
    tool_access = contracts_runtime.ToolAccess.workspace_backed(
        Path("/repo"),
        tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
    )

    request = prompt_runtime.ResumedSessionRunRequest(
        prompt="already rendered prompt",
        invocation_dir=Path("/repo"),
        continuation=_continuation(tool_access=tool_access),
    )

    assert request.tool_access == tool_access
    assert request.tool_access.workspace == Path("/repo")


def test_resumed_session_run_request_rejects_explicit_no_tools_tool_access() -> None:
    with pytest.raises(
        TypeError,
        match=re.escape(
            "ResumedSessionRunRequest derives fixed tool access from `continuation` and does not accept `tool_access` or `tool_policy` overrides."
        ),
    ):
        prompt_runtime.ResumedSessionRunRequest(
            prompt="already rendered prompt",
            invocation_dir=Path("/repo"),
            continuation=_continuation(),
            tool_access=contracts_runtime.ToolAccess.no_tools(),
        )


def test_resumed_session_run_request_rejects_workspace_backed_tool_access_for_other_invocation_dir() -> (
    None
):
    with pytest.raises(
        ValueError,
        match=re.escape(
            "ResumedSessionRunRequest workspace-backed tool access requires invocation_dir /repo, got /other."
        ),
    ):
        prompt_runtime.ResumedSessionRunRequest(
            prompt="already rendered prompt",
            invocation_dir=Path("/other"),
            continuation=_continuation(
                tool_access=contracts_runtime.ToolAccess.workspace_backed(
                    Path("/repo"),
                    tool_policy=runtime.ToolPolicy.UNRESTRICTED,
                )
            ),
        )


def test_resumed_session_run_request_from_continuation_rejects_tool_policy_override() -> (
    None
):
    with pytest.raises(
        TypeError,
        match=re.escape(
            "ResumedSessionRunRequest got an unexpected keyword argument 'tool_policy'"
        ),
    ):
        prompt_runtime.ResumedSessionRunRequest(
            prompt="already rendered prompt",
            invocation_dir=Path("/repo"),
            continuation=_continuation(),
            tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
        )


def test_resumed_session_run_request_from_continuation_rejects_workspace_backed_tool_access_for_other_invocation_dir() -> (
    None
):
    with pytest.raises(
        ValueError,
        match=re.escape(
            "ResumedSessionRunRequest workspace-backed tool access requires invocation_dir /repo, got /other."
        ),
    ):
        prompt_runtime.ResumedSessionRunRequest(
            prompt="already rendered prompt",
            invocation_dir=Path("/other"),
            continuation=_continuation(
                tool_access=contracts_runtime.ToolAccess.workspace_backed(
                    Path("/repo"),
                    tool_policy=runtime.ToolPolicy.UNRESTRICTED,
                )
            ),
        )


def test_resumed_session_run_request_rejects_missing_continuation() -> None:
    with pytest.raises(
        TypeError,
        match=re.escape("ResumedSessionRunRequest requires a `continuation` value."),
    ):
        prompt_runtime.ResumedSessionRunRequest(
            prompt="already rendered prompt",
            invocation_dir=Path("/repo"),
        )
