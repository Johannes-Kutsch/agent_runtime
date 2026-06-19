from __future__ import annotations

import inspect
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest

import agent_runtime as runtime
import agent_runtime.contracts as contracts_runtime
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


def test_ephemeral_run_request_only_accepts_minimal_ephemeral_fields(
    stage_selection_factory: Callable[..., runtime.StageSelection],
) -> None:
    request = prompt_runtime.EphemeralRunRequest(
        prompt="already rendered prompt",
        invocation_dir=Path("/repo"),
        stage=stage_selection_factory(service="codex"),
        tool_access=contracts_runtime.ToolAccess.no_tools(),
        auth=runtime.ProviderAuth(opencode_api_key="go-key"),
        token=execution_contracts_runtime.CancellationToken(),
    )

    assert request.auth == runtime.ProviderAuth(opencode_api_key="go-key")
    assert request.token is not None
    for field_name in ("role", "logs_dir", "usage_limit_scope", "session_namespace"):
        with pytest.raises(AttributeError, match=field_name):
            getattr(request, field_name)
    assert tuple(inspect.signature(prompt_runtime.EphemeralRunRequest).parameters) == (
        "prompt",
        "invocation_dir",
        "stage",
        "tool_policy",
        "token",
        "auth",
        "override",
    )


def test_resumed_session_run_request_has_minimal_public_signature() -> None:
    assert tuple(
        inspect.signature(prompt_runtime.ResumedSessionRunRequest).parameters
    ) == (
        "prompt",
        "invocation_dir",
        "continuation",
        "provider_auth",
        "model",
        "effort",
        "token",
    )


def test_lifecycle_request_signatures_no_longer_show_tool_access() -> None:
    assert (
        "tool_access"
        not in inspect.signature(
            prompt_runtime.EphemeralRunRequest,
        ).parameters
    )
    assert (
        "tool_access"
        not in inspect.signature(
            prompt_runtime.NewSessionRunRequest,
        ).parameters
    )
    assert (
        "tool_access"
        not in inspect.signature(
            prompt_runtime.ResumedSessionRunRequest,
        ).parameters
    )


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
        invocation_dir=Path("/repo"),
        override=override_stage,
        tool_access=contracts_runtime.ToolAccess.no_tools(),
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
            invocation_dir=Path("/repo"),
            stage=stage_selection_factory(service="codex"),
            override=stage_selection_factory(service="claude"),
            tool_access=contracts_runtime.ToolAccess.no_tools(),
        )


def test_new_session_run_request_defaults_to_implementer_without_caller_managed_inputs(
    stage_selection_factory: Callable[..., runtime.StageSelection],
    tmp_path: Path,
) -> None:
    request = prompt_runtime.NewSessionRunRequest(
        prompt="already rendered prompt",
        invocation_dir=tmp_path,
        stage=stage_selection_factory(service="codex"),
        tool_access=contracts_runtime.ToolAccess.no_tools(),
    )

    assert request.role == InvocationRole("implementer")
    assert request.runtime_state_dir is None
    assert not hasattr(request, "usage_limit_scope")
    assert request.session_namespace == ""
    assert not hasattr(request, "logs_dir")


@pytest.mark.parametrize(
    ("request_factory", "request_name"),
    [
        (
            lambda stage_selection_factory, tmp_path, unexpected_name, unexpected_value: (
                prompt_runtime.EphemeralRunRequest(
                    prompt="already rendered prompt",
                    invocation_dir=tmp_path,
                    stage=stage_selection_factory(service="codex"),
                    tool_policy=runtime.ToolPolicy.NONE,
                    **{unexpected_name: unexpected_value},
                )
            ),
            "EphemeralRunRequest",
        ),
        (
            lambda stage_selection_factory, tmp_path, unexpected_name, unexpected_value: (
                prompt_runtime.NewSessionRunRequest(
                    prompt="already rendered prompt",
                    invocation_dir=tmp_path,
                    stage=stage_selection_factory(service="codex"),
                    tool_policy=runtime.ToolPolicy.NONE,
                    **{unexpected_name: unexpected_value},
                )
            ),
            "NewSessionRunRequest",
        ),
        (
            lambda stage_selection_factory, tmp_path, unexpected_name, unexpected_value: (
                prompt_runtime.ResumedSessionRunRequest(
                    prompt="already rendered prompt",
                    invocation_dir=WorktreeMount(tmp_path),
                    continuation=prompt_runtime.Continuation(
                        selected_service="codex",
                        selected_model="gpt-5.4",
                        selected_effort="medium",
                        tool_access=contracts_runtime.ToolAccess.no_tools(),
                        provider_resume_state={"run_kind": "resume"},
                    ),
                    **{unexpected_name: unexpected_value},
                )
            ),
            "ResumedSessionRunRequest",
        ),
    ],
)
@pytest.mark.parametrize(
    ("unexpected_name", "unexpected_value"),
    [
        ("logs_dir", Path("/tmp/runtime-logs")),
        ("log_name", "implementer"),
    ],
)
def test_ordinary_runtime_requests_reject_runtime_managed_log_inputs(
    stage_selection_factory: Callable[..., runtime.StageSelection],
    tmp_path: Path,
    request_factory: Callable[..., object],
    request_name: str,
    unexpected_name: str,
    unexpected_value: object,
) -> None:
    with pytest.raises(
        TypeError,
        match=(
            f"{request_name} got an unexpected keyword argument '{unexpected_name}'."
        ),
    ):
        request_factory(
            stage_selection_factory,
            tmp_path,
            unexpected_name,
            unexpected_value,
        )


def test_new_session_run_request_rejects_caller_provided_usage_limit_scope() -> None:
    with pytest.raises(
        TypeError,
        match="got an unexpected keyword argument 'usage_limit_scope'",
    ):
        cast(Any, prompt_runtime.NewSessionRunRequest)(
            prompt="already rendered prompt",
            invocation_dir=Path("/repo"),
            stage=runtime.StageSelection(
                service="codex",
                model="gpt-5.4",
                effort="high",
            ),
            tool_access=contracts_runtime.ToolAccess.no_tools(),
            usage_limit_scope=runtime.UsageLimitScope("review"),
        )


def test_ephemeral_run_request_rejects_caller_provided_usage_limit_scope() -> None:
    with pytest.raises(
        TypeError,
        match="got an unexpected keyword argument 'usage_limit_scope'",
    ):
        cast(Any, prompt_runtime.EphemeralRunRequest)(
            prompt="already rendered prompt",
            invocation_dir=Path("/repo"),
            stage=runtime.StageSelection(
                service="codex",
                model="gpt-5.4",
                effort="high",
            ),
            tool_access=contracts_runtime.ToolAccess.no_tools(),
            usage_limit_scope=runtime.UsageLimitScope("review"),
        )


def test_resumed_session_run_request_rejects_caller_provided_usage_limit_scope() -> (
    None
):
    with pytest.raises(
        TypeError,
        match="got an unexpected keyword argument 'usage_limit_scope'",
    ):
        cast(Any, prompt_runtime.ResumedSessionRunRequest)(
            prompt="already rendered prompt",
            invocation_dir=Path("/repo"),
            continuation=prompt_runtime.Continuation(
                selected_service="codex",
                selected_model="gpt-5.4",
                selected_effort="medium",
                tool_access=contracts_runtime.ToolAccess.no_tools(),
                provider_resume_state={"run_kind": "resume"},
            ),
            usage_limit_scope=runtime.UsageLimitScope("review"),
        )


def test_prompt_run_request_rejects_caller_provided_usage_limit_scope() -> None:
    with pytest.raises(
        TypeError,
        match="got an unexpected keyword argument 'usage_limit_scope'",
    ):
        cast(Any, execution_contracts_runtime.PromptRunRequest)(
            prompt="already rendered prompt",
            worktree=WorktreeMount(Path("/repo")),
            stage=runtime.StageSelection(
                service="codex",
                model="gpt-5.4",
                effort="high",
            ),
            role=InvocationRole("implementer"),
            tool_access=contracts_runtime.ToolAccess.no_tools(),
            usage_limit_scope=runtime.UsageLimitScope("review"),
        )


def test_prompt_run_request_uses_compatibility_tool_policy_for_workspace_backed_tool_access(
    stage_selection_factory: Callable[..., runtime.StageSelection],
) -> None:
    request = execution_contracts_runtime.PromptRunRequest(
        prompt="already rendered prompt",
        worktree=WorktreeMount(Path("/repo")),
        stage=stage_selection_factory(service="codex"),
        role=InvocationRole("implementer"),
        tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
    )

    assert request.tool_access == contracts_runtime.ToolAccess.workspace_backed(
        Path("/repo"),
        tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
    )
    assert request.tool_policy is runtime.ToolPolicy.NO_FILE_MUTATION


def test_ephemeral_run_request_uses_compatibility_tool_policy_for_workspace_backed_tool_access(
    stage_selection_factory: Callable[..., runtime.StageSelection],
) -> None:
    request = prompt_runtime.EphemeralRunRequest(
        prompt="already rendered prompt",
        invocation_dir=Path("/repo"),
        stage=stage_selection_factory(service="codex"),
        tool_policy=runtime.ToolPolicy.UNRESTRICTED,
    )

    assert request.tool_access == contracts_runtime.ToolAccess.workspace_backed(
        Path("/repo")
    )
    assert request.tool_policy is runtime.ToolPolicy.UNRESTRICTED


def test_ephemeral_run_request_uses_none_tool_policy_for_explicit_no_tools_access(
    stage_selection_factory: Callable[..., runtime.StageSelection],
) -> None:
    request = prompt_runtime.EphemeralRunRequest(
        prompt="already rendered prompt",
        invocation_dir=Path("/repo"),
        stage=stage_selection_factory(service="codex"),
        tool_policy=runtime.ToolPolicy.NONE,
    )

    assert request.tool_access == contracts_runtime.ToolAccess.no_tools()
    assert request.tool_policy is runtime.ToolPolicy.NONE


def test_new_session_run_request_uses_compatibility_tool_policy_for_workspace_backed_tool_access(
    stage_selection_factory: Callable[..., runtime.StageSelection],
) -> None:
    request = prompt_runtime.NewSessionRunRequest(
        prompt="already rendered prompt",
        invocation_dir=Path("/repo"),
        stage=stage_selection_factory(service="codex"),
        role=InvocationRole("implementer"),
        tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
    )

    assert request.tool_access == contracts_runtime.ToolAccess.workspace_backed(
        Path("/repo"),
        tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
    )
    assert request.tool_policy is runtime.ToolPolicy.NO_FILE_MUTATION


def test_new_session_request_non_none_tool_policy_uses_invocation_dir_as_tool_workspace(
    stage_selection_factory: Callable[..., runtime.StageSelection],
    tmp_path: Path,
) -> None:
    request = prompt_runtime.NewSessionRunRequest(
        prompt="already rendered prompt",
        invocation_dir=tmp_path,
        stage=stage_selection_factory(service="codex"),
        role=InvocationRole("implementer"),
        tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
    )

    assert request.tool_access == contracts_runtime.ToolAccess.workspace_backed(
        tmp_path,
        tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
    )


@pytest.mark.parametrize(
    ("request_factory", "expected_message"),
    [
        (
            lambda stage_selection_factory, session_namespace: (
                prompt_runtime.NewSessionRunRequest(
                    prompt="already rendered prompt",
                    invocation_dir=Path("/repo"),
                    stage=stage_selection_factory(service="codex"),
                    role=InvocationRole("implementer"),
                    tool_access=contracts_runtime.ToolAccess.no_tools(),
                    session_namespace=session_namespace,
                )
            ),
            "Session namespace",
        ),
    ],
)
def test_lifecycle_request_construction_preserves_empty_session_namespace_and_rejects_unsafe_non_empty_values(
    stage_selection_factory: Callable[..., runtime.StageSelection],
    request_factory: Callable[
        [Callable[..., runtime.StageSelection], str],
        prompt_runtime.NewSessionRunRequest,
    ],
    expected_message: str,
) -> None:
    request = request_factory(stage_selection_factory, "")

    assert request.session_namespace == ""

    with pytest.raises(ValueError, match=re.escape(expected_message)):
        request_factory(stage_selection_factory, "../escape")


def test_resumed_session_run_request_uses_compatibility_tool_policy_for_workspace_backed_tool_access() -> (
    None
):
    request = prompt_runtime.ResumedSessionRunRequest(
        prompt="already rendered prompt",
        invocation_dir=Path("/repo"),
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
        tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
    )

    assert request.tool_access == contracts_runtime.ToolAccess.workspace_backed(
        Path("/repo"),
        tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
    )
    assert request.tool_policy is runtime.ToolPolicy.NO_FILE_MUTATION


def test_resumed_session_run_request_from_session_plan_keeps_namespace_from_session_plan() -> (
    None
):
    request = prompt_runtime.ResumedSessionRunRequest(
        prompt="already rendered prompt",
        invocation_dir=Path("/repo"),
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
        tool_access=contracts_runtime.ToolAccess.no_tools(),
        session_namespace="../escape",
    )

    assert request.session_namespace == "main"


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
            tool_access=contracts_runtime.ToolAccess.workspace_backed(Path("/other")),
        )


@pytest.mark.parametrize(
    ("request_factory", "expected_message"),
    [
        (
            lambda stage_selection_factory: prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                invocation_dir=Path("/repo"),
                stage=stage_selection_factory(service="codex"),
                tool_access=contracts_runtime.ToolAccess.workspace_backed(
                    Path("/other"),
                    tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
                ),
            ),
            "EphemeralRunRequest workspace-backed tool access requires invocation_dir /other, got /repo.",
        ),
        (
            lambda stage_selection_factory: prompt_runtime.NewSessionRunRequest(
                prompt="already rendered prompt",
                invocation_dir=Path("/repo"),
                stage=stage_selection_factory(service="codex"),
                role=InvocationRole("implementer"),
                tool_access=contracts_runtime.ToolAccess.workspace_backed(
                    Path("/other"),
                    tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
                ),
            ),
            "NewSessionRunRequest workspace-backed tool access requires invocation_dir /other, got /repo.",
        ),
    ],
)
def test_lifecycle_request_construction_rejects_workspace_backed_tool_access_for_other_worktree(
    stage_selection_factory: Callable[..., runtime.StageSelection],
    request_factory: Callable[[Callable[..., runtime.StageSelection]], object],
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=re.escape(expected_message)):
        request_factory(stage_selection_factory)


@pytest.mark.parametrize(
    ("request_factory", "expected_message"),
    [
        (
            lambda stage_selection_factory: prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                invocation_dir=Path("/repo"),
                stage=stage_selection_factory(service="codex"),
            ),
            "EphemeralRunRequest requires an explicit `tool_policy` value.",
        ),
        (
            lambda stage_selection_factory: prompt_runtime.NewSessionRunRequest(
                prompt="already rendered prompt",
                invocation_dir=Path("/repo"),
                stage=stage_selection_factory(service="codex"),
                role=InvocationRole("implementer"),
            ),
            "NewSessionRunRequest requires an explicit `tool_policy` value.",
        ),
    ],
)
def test_lifecycle_request_construction_requires_explicit_tool_policy(
    stage_selection_factory: Callable[..., runtime.StageSelection],
    request_factory: Callable[
        [Callable[..., runtime.StageSelection]],
        object,
    ],
    expected_message: str,
) -> None:
    with pytest.raises(TypeError, match=re.escape(expected_message)):
        request_factory(stage_selection_factory)


def test_ephemeral_request_none_tool_policy_prohibits_provider_tools_and_requires_invocation_dir(
    stage_selection_factory: Callable[..., runtime.StageSelection],
    tmp_path: Path,
) -> None:
    request = prompt_runtime.EphemeralRunRequest(
        prompt="already rendered prompt",
        invocation_dir=tmp_path,
        stage=stage_selection_factory(service="codex"),
        tool_policy=runtime.ToolPolicy.NONE,
    )

    assert request.invocation_dir == tmp_path
    assert request.tool_access == contracts_runtime.ToolAccess.no_tools()
    assert request.tool_policy is runtime.ToolPolicy.NONE


def test_new_session_request_none_tool_policy_prohibits_provider_tools_and_requires_invocation_dir(
    stage_selection_factory: Callable[..., runtime.StageSelection],
    tmp_path: Path,
) -> None:
    request = prompt_runtime.NewSessionRunRequest(
        prompt="already rendered prompt",
        invocation_dir=tmp_path,
        stage=stage_selection_factory(service="codex"),
        role=InvocationRole("implementer"),
        tool_policy=runtime.ToolPolicy.NONE,
    )

    assert request.invocation_dir == tmp_path
    assert request.tool_access == contracts_runtime.ToolAccess.no_tools()
    assert request.tool_policy is runtime.ToolPolicy.NONE


def test_resumed_session_run_request_coerces_path_invocation_dir_to_worktree_mount() -> (
    None
):
    request = prompt_runtime.ResumedSessionRunRequest(
        prompt="already rendered prompt",
        invocation_dir=Path("/repo"),
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
        tool_access=contracts_runtime.ToolAccess.no_tools(),
    )

    assert request.invocation_dir == WorktreeMount(Path("/repo"))
    assert request.mount_path == Path("/repo")


@pytest.mark.parametrize(
    ("request_factory", "expected_invocation_dir"),
    [
        (
            lambda stage_selection_factory: prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                worktree=Path("/repo"),
                stage=stage_selection_factory(service="codex"),
                tool_access=contracts_runtime.ToolAccess.no_tools(),
            ),
            Path("/repo"),
        ),
        (
            lambda stage_selection_factory: prompt_runtime.NewSessionRunRequest(
                prompt="already rendered prompt",
                worktree=Path("/repo"),
                stage=stage_selection_factory(service="codex"),
                role=InvocationRole("implementer"),
                tool_access=contracts_runtime.ToolAccess.no_tools(),
            ),
            Path("/repo"),
        ),
        (
            lambda _stage_selection_factory: prompt_runtime.ResumedSessionRunRequest(
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
                tool_access=contracts_runtime.ToolAccess.no_tools(),
            ),
            WorktreeMount(Path("/repo")),
        ),
    ],
)
def test_lifecycle_request_construction_keeps_legacy_worktree_kwarg_outside_public_surface(
    stage_selection_factory: Callable[..., runtime.StageSelection],
    request_factory: Callable[[Callable[..., runtime.StageSelection]], object],
    expected_invocation_dir: Path | WorktreeMount,
) -> None:
    request = request_factory(stage_selection_factory)

    assert getattr(request, "invocation_dir") == expected_invocation_dir


@pytest.mark.parametrize(
    "request_factory",
    [
        lambda stage_selection_factory: prompt_runtime.EphemeralRunRequest(
            prompt="already rendered prompt",
            invocation_dir=Path("/repo"),
            worktree=Path("/other"),
            stage=stage_selection_factory(service="codex"),
            tool_access=contracts_runtime.ToolAccess.no_tools(),
        ),
        lambda stage_selection_factory: prompt_runtime.NewSessionRunRequest(
            prompt="already rendered prompt",
            invocation_dir=Path("/repo"),
            worktree=Path("/other"),
            stage=stage_selection_factory(service="codex"),
            role=InvocationRole("implementer"),
            tool_access=contracts_runtime.ToolAccess.no_tools(),
        ),
        lambda _stage_selection_factory: prompt_runtime.ResumedSessionRunRequest(
            prompt="already rendered prompt",
            invocation_dir=Path("/repo"),
            worktree=Path("/other"),
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
            tool_access=contracts_runtime.ToolAccess.no_tools(),
        ),
    ],
)
def test_lifecycle_request_construction_rejects_conflicting_invocation_dir_and_legacy_worktree(
    stage_selection_factory: Callable[..., runtime.StageSelection],
    request_factory: Callable[[Callable[..., runtime.StageSelection]], object],
) -> None:
    with pytest.raises(
        TypeError,
        match=re.escape("received conflicting `invocation_dir` and `worktree` values."),
    ):
        request_factory(stage_selection_factory)


def test_runtime_public_surface_keeps_request_normalization_module_private() -> None:
    assert "_request_normalization" not in runtime.__all__
    assert "_request_normalization" not in prompt_runtime.__all__
    assert "_request_normalization" not in execution_contracts_runtime.__all__
