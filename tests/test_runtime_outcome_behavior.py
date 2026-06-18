from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Callable, cast

import pytest

import agent_runtime as runtime
import agent_runtime._runtime_compat as compat_runtime
import agent_runtime.runtime as prompt_runtime
from agent_runtime.contracts import ExecutionProvider
from agent_runtime.execution_contracts import (
    PreparedRunSessionState,
    WorkExecutionAdapter,
    WorkExecutionDependencies,
    WorkFailureHandling,
    WorkInvocationDependencies,
    WorkPresentationDependencies,
)
from agent_runtime.roles import InvocationRole
from agent_runtime.service_registry import ServiceRegistry
from agent_runtime.session import RunKind

from tests.runtime_boundary_fakes import ExecutionServiceFake as _ExecutionService


class _PreparedRunSession:
    provider_state_dir_container_path: str | None = None

    def prepare_for_run(self) -> None:
        return None

    def initial_provider_run_session(self) -> None:
        return None

    def resumable_provider_run_session(self) -> None:
        return None

    def protocol_reprompt_provider_run_session(self) -> None:
        return None


class _Session:
    provider_state_dir: str | None = None


def _tool_policy_effect_text(tool_policy: Any) -> str:
    profile = (
        tool_policy.profile
        if isinstance(tool_policy, runtime.ToolPolicy)
        else tool_policy
    )
    allowed_tools = profile.allowed_tools or ()
    disallowed_tools = profile.disallowed_tools or ()
    allowed = ",".join(allowed_tools) or "all"
    disallowed = ",".join(disallowed_tools) or "none"
    return f"allowed={allowed};disallowed={disallowed}"


class _ToolPolicyRenderingPromptRunner:
    async def setup(self, git_name: str, git_email: str, work_body: str = "") -> None:
        del git_name, git_email, work_body

    async def work_text(
        self,
        prompt: str,
        *,
        role: InvocationRole = InvocationRole("implementer"),
        tool_policy: Any,
        run_kind: RunKind = RunKind.FRESH,
        session_uuid: str | None = None,
        on_provider_session_id: Any = None,
    ) -> str:
        del prompt, role, run_kind, session_uuid
        assert callable(on_provider_session_id)
        on_provider_session_id("provider-session")
        return _tool_policy_effect_text(tool_policy)


class _ToolPolicyRenderingEphemeralExecutionAdapter:
    def resolve_service(self, service_name: str = "") -> ExecutionProvider:
        return _ExecutionService(service_name)

    def build_work_dependencies(
        self,
        *,
        name: str,
        model: str,
        effort: str,
        service: ExecutionProvider,
    ) -> WorkInvocationDependencies:
        del name, model, effort, service
        return WorkInvocationDependencies(
            execution=WorkExecutionDependencies(
                container_workspace="/workspace",
                prepare_session=lambda _run_session: cast(
                    PreparedRunSessionState, _PreparedRunSession()
                ),
                build_session=lambda mount_path, service, provider_state_dir: (
                    _Session()
                ),
                build_runner=lambda session, status_display: cast(
                    WorkExecutionAdapter,
                    _ToolPolicyRenderingPromptRunner(),
                ),
                get_git_identity=lambda: ("Runtime Test", "runtime@example.com"),
            ),
            failure_handling=WorkFailureHandling(timeout_retries=0),
            presentation=WorkPresentationDependencies(),
        )


def test_ephemeral_runtime_returns_completed_outcome_with_selected_runtime_metadata_and_tool_access(
    stage_selection_factory: Callable[..., runtime.StageSelection],
    service_registry_factory: Callable[..., ServiceRegistry],
) -> None:
    tool_access = runtime.ToolAccess.workspace_backed(
        Path("/repo"),
        tool_policy=runtime.ToolPolicy.PARTIAL,
    )

    result = asyncio.run(
        compat_runtime.EphemeralRuntime(
            execution_adapter=_ToolPolicyRenderingEphemeralExecutionAdapter(),
            service_registry=service_registry_factory("claude"),
        ).run_ephemeral(
            prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                worktree=Path("/repo"),
                stage=stage_selection_factory(
                    service="claude",
                    model="gpt-5",
                    effort="medium",
                ),
                role=InvocationRole("implementer"),
                tool_access=tool_access,
            )
        )
    )

    assert result == prompt_runtime.RuntimeOutcome.completed(
        output=_tool_policy_effect_text(runtime.ToolPolicy.PARTIAL),
        result=prompt_runtime.EphemeralRunResult(
            output=_tool_policy_effect_text(runtime.ToolPolicy.PARTIAL),
            selected_service="claude",
            selected_model="gpt-5",
            selected_effort="medium",
            tool_access=tool_access,
            used_fallback=False,
            metadata=prompt_runtime.EphemeralResultMetadata(
                selected_service_path=("claude",),
                runtime=prompt_runtime.EphemeralRuntimeMetadata(
                    run_kind=RunKind.FRESH,
                    session_namespace="",
                ),
            ),
        ),
    )
    assert result.tool_access == tool_access


def test_completed_runtime_outcome_only_exposes_ephemeral_selection_metadata_for_ephemeral_results() -> (
    None
):
    tool_access = runtime.ToolAccess.no_tools()
    result = prompt_runtime.EphemeralRunResult(
        output="done",
        selected_service="claude",
        selected_model="gpt-5",
        selected_effort="medium",
        tool_access=tool_access,
        used_fallback=True,
        metadata=prompt_runtime.EphemeralResultMetadata(
            selected_service_path=("codex", "claude"),
            runtime=prompt_runtime.EphemeralRuntimeMetadata(
                run_kind=RunKind.FRESH,
                session_namespace="review",
            ),
        ),
    )
    outcome = prompt_runtime.RuntimeOutcome.completed(output="done", result=result)

    assert outcome.runtime_metadata == result.runtime_metadata
    assert outcome.metadata == result.metadata
    assert outcome.selected_service_path == ("codex", "claude")
    assert outcome.selected_service == "claude"
    assert outcome.selected_model == "gpt-5"
    assert outcome.selected_effort == "medium"
    assert outcome.used_fallback is True
    assert outcome.tool_access == tool_access


def test_completed_runtime_outcome_rejects_ephemeral_selection_metadata_for_session_results() -> (
    None
):
    continuation = prompt_runtime.Continuation(
        selected_service="claude",
        selected_model="gpt-5",
        selected_effort="medium",
        tool_access=runtime.ToolAccess.workspace_backed(Path("/repo")),
        provider_resume_state={"provider_session_id": "session-123"},
    )
    result = prompt_runtime.SessionRunResult(
        output="done",
        continuation=continuation,
        runtime_metadata=prompt_runtime.SessionRuntimeMetadata(
            provider_session_id="session-123",
            run_kind=RunKind.RESUME,
            session_namespace="review",
            service_name="claude",
            exact_transcript_match=False,
        ),
    )
    outcome = prompt_runtime.RuntimeOutcome.completed(output="done", result=result)

    assert outcome.runtime_metadata == result.runtime_metadata

    with pytest.raises(AttributeError, match="ephemeral metadata"):
        _ = outcome.metadata
    with pytest.raises(AttributeError, match="selection metadata"):
        _ = outcome.selected_service_path
    with pytest.raises(AttributeError, match="selection metadata"):
        _ = outcome.selected_service
    with pytest.raises(AttributeError, match="selection metadata"):
        _ = outcome.selected_model
    with pytest.raises(AttributeError, match="selection metadata"):
        _ = outcome.selected_effort
    with pytest.raises(AttributeError, match="selection metadata"):
        _ = outcome.used_fallback
    with pytest.raises(AttributeError, match="tool access"):
        _ = outcome.tool_access
