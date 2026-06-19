from __future__ import annotations

import importlib
import inspect
from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from typing import Any, cast

import pytest

import agent_runtime as runtime
import agent_runtime.contracts as contracts_runtime
import agent_runtime._provider_invocation as provider_invocation_runtime
import agent_runtime._runtime_compat as compat_runtime
import agent_runtime.provider_session_adapter as provider_session_adapter_runtime
import agent_runtime.runtime as prompt_runtime
import agent_runtime.session as session_runtime
import agent_runtime.session_planning as session_planning_runtime
from agent_runtime.errors import AgentRuntimeError
from agent_runtime.provider_usage import ProviderUsage
from agent_runtime.roles import InvocationRole
from agent_runtime.session import RunKind
from agent_runtime.usage_limit_scope import UsageLimitScope


def test_package_exports_runtime_surface() -> None:
    assert runtime.__all__ == [
        "AgentCredentialFailureError",
        "AgentFailedError",
        "AgentRuntimeError",
        "AgentTimeoutError",
        "Continuation",
        "HardAgentError",
        "InvocationRole",
        "InvocationProgress",
        "ProviderAuth",
        "ProviderUsage",
        "RuntimeClient",
        "RuntimeConfigurationError",
        "RuntimeOutcome",
        "RunKind",
        "StageSelection",
        "ToolPolicy",
        "TransientAgentError",
        "UsageLimitError",
        "UsageLimitScope",
    ]
    assert runtime.StageSelection.__module__.startswith("agent_runtime")
    assert not hasattr(runtime, "StageOverride")
    assert runtime.AgentRuntimeError is AgentRuntimeError
    assert runtime.RuntimeOutcome is prompt_runtime.RuntimeOutcome
    assert "ToolAccess" not in runtime.__all__
    assert "ToolPolicyProfile" not in runtime.__all__
    assert not hasattr(runtime, "ToolAccess")
    assert not hasattr(runtime, "ToolPolicyProfile")
    assert hasattr(contracts_runtime, "ToolAccess")
    assert hasattr(contracts_runtime, "ToolPolicyProfile")
    assert not hasattr(runtime, "assert_runtime_import_isolation")
    assert not hasattr(runtime, "run_prompt")
    assert not hasattr(runtime, "ExecutionProvider")
    assert not hasattr(runtime, "ServiceRegistry")
    assert not hasattr(runtime, "ProviderSessionAdapter")
    assert not hasattr(runtime, "ProviderSessionPreferences")
    assert not hasattr(runtime, "ProviderSessionPreferencesRequest")
    assert not hasattr(runtime, "ProviderSessionState")
    assert not hasattr(runtime, "ProviderSessionStateRequest")
    assert not hasattr(prompt_runtime, "PromptRuntime")
    assert not hasattr(prompt_runtime, "PromptRunRequest")
    assert not hasattr(prompt_runtime, "PromptRuntimeExecutionAdapter")
    assert not hasattr(prompt_runtime, "run_ephemeral")
    assert not hasattr(prompt_runtime, "run_prompt")
    assert not hasattr(prompt_runtime, "run_resumable_prompt")
    assert not hasattr(prompt_runtime, "ResidentRunRequest")
    assert not hasattr(prompt_runtime, "ResidentRunResult")
    assert not hasattr(prompt_runtime, "ResidentRuntime")
    assert not hasattr(prompt_runtime, "ResidentRuntimeExecutionAdapter")
    assert not hasattr(prompt_runtime, "ResidentRuntimeMetadata")
    assert {
        "Continuation",
        "EphemeralRunRequest",
        "NewSessionRunRequest",
        "ProviderAuth",
        "ProviderUsage",
        "ResumedSessionRunRequest",
        "RuntimeClient",
        "RuntimeOutcome",
        "SessionRunResult",
        "SessionRuntimeMetadata",
        "WorktreeMount",
    } <= set(prompt_runtime.__all__)
    assert "ToolAccess" not in prompt_runtime.__all__
    assert "ToolPolicyProfile" not in prompt_runtime.__all__
    assert not hasattr(prompt_runtime, "ToolAccess")
    assert not hasattr(prompt_runtime, "ToolPolicyProfile")
    assert {
        "EphemeralRunRequest",
        "EphemeralRunResult",
        "EphemeralResultMetadata",
        "EphemeralRuntimeMetadata",
        "Continuation",
        "ProviderUsage",
        "ResumedSessionRunRequest",
        "SessionRunResult",
        "SessionRuntimeMetadata",
    } <= set(prompt_runtime.__all__)
    assert "EphemeralRuntime" not in prompt_runtime.__all__
    assert "NewSessionRuntime" not in prompt_runtime.__all__
    assert "ResumedSessionRuntime" not in prompt_runtime.__all__
    assert "EphemeralRuntimeExecutionAdapter" not in prompt_runtime.__all__
    assert "NewSessionRuntimeExecutionAdapter" not in prompt_runtime.__all__
    assert "ResumedSessionRuntimeExecutionAdapter" not in prompt_runtime.__all__
    assert not hasattr(prompt_runtime, "ResumableRunResult")
    assert not hasattr(prompt_runtime, "ResumableRuntimeMetadata")
    assert "ResumableRunRequest" not in prompt_runtime.__all__
    assert not hasattr(prompt_runtime, "ResumableRunRequest")
    assert "ResumableRuntime" not in prompt_runtime.__all__
    assert "ResumableRuntimeExecutionAdapter" not in prompt_runtime.__all__
    assert "OneShotRunRequest" not in prompt_runtime.__all__
    assert "OneShotRunResult" not in prompt_runtime.__all__
    assert "OneShotResultMetadata" not in prompt_runtime.__all__
    assert "OneShotRuntime" not in prompt_runtime.__all__
    assert "OneShotRuntimeExecutionAdapter" not in prompt_runtime.__all__
    assert "OneShotRuntimeMetadata" not in prompt_runtime.__all__
    assert not hasattr(runtime, "ProviderInvocationRequest")
    assert not hasattr(runtime, "ProviderInvocationResult")
    assert not hasattr(runtime, "ProviderInvocationAdapter")
    assert not hasattr(prompt_runtime, "ProviderInvocationRequest")
    assert not hasattr(prompt_runtime, "ProviderInvocationResult")
    assert not hasattr(prompt_runtime, "ProviderInvocationAdapter")


def test_built_in_provider_invocation_seam_stays_private_to_runtime_public_surface() -> (
    None
):
    with pytest.raises(ImportError):
        exec("from agent_runtime import ProviderInvocationRequest", {}, {})
    with pytest.raises(ImportError):
        exec("from agent_runtime.runtime import ProviderInvocationRequest", {}, {})
    with pytest.raises(ImportError):
        exec("from agent_runtime.runtime import ProviderInvocationResult", {}, {})
    with pytest.raises(ImportError):
        exec("from agent_runtime.runtime import ProviderInvocationAdapter", {}, {})


@pytest.mark.parametrize(
    ("module_name", "removed_name"),
    [
        ("agent_runtime", "ToolAccess"),
        ("agent_runtime", "ToolPolicyProfile"),
        ("agent_runtime.runtime", "ToolAccess"),
        ("agent_runtime.runtime", "ToolPolicyProfile"),
    ],
)
def test_removed_tool_policy_compatibility_names_fail_on_ordinary_runtime_surface(
    module_name: str,
    removed_name: str,
) -> None:
    with pytest.raises(ImportError):
        exec(f"from {module_name} import {removed_name}", {}, {})

    imported_module = importlib.import_module(module_name)
    with pytest.raises(AttributeError, match="Runtime Public Surface"):
        getattr(imported_module, removed_name)

    compatibility_imports: dict[str, object] = {}
    exec(
        f"from agent_runtime.contracts import {removed_name}",
        {},
        compatibility_imports,
    )
    assert compatibility_imports[removed_name] is getattr(
        contracts_runtime,
        removed_name,
    )


def test_runtime_client_constructor_stays_on_public_default_surface() -> None:
    signature = inspect.signature(runtime.RuntimeClient)
    assert list(signature.parameters) == []
    unexpected_kwargs: dict[str, object] = {"_provider_invocation_adapter": None}
    with pytest.raises(TypeError):
        runtime.RuntimeClient(**unexpected_kwargs)


def test_built_in_provider_invocation_seam_uses_frozen_contract_values() -> None:
    def reduce_output(lines: list[str]) -> tuple[str, None]:
        return "".join(lines), None

    hooks = provider_invocation_runtime.ProviderOutputReductionHooks(
        reduce_output=reduce_output
    )
    prompt = provider_invocation_runtime.ProviderInvocationPrompt(
        content="prompt body",
        path=Path("/tmp/prompt.txt"),
        cleanup_path=True,
    )
    request = provider_invocation_runtime.ProviderInvocationRequest(
        command="provider --run",
        worktree=Path("/tmp/worktree"),
        environment={"PATH": "/usr/bin"},
        prompt=prompt,
        run_kind=RunKind.FRESH,
        role=InvocationRole("review"),
        usage_limit_scope=UsageLimitScope("review"),
        log_context=None,
        provider_session_id="session-123",
        output_hooks=hooks,
    )
    result = provider_invocation_runtime.ProviderInvocationResult(
        output="ok",
        usage=ProviderUsage(output_tokens=3),
        stdout_lines=("line 1", "line 2"),
        provider_session_id="session-123",
    )

    assert [field.name for field in fields(prompt)] == [
        "content",
        "path",
        "cleanup_path",
    ]
    assert [field.name for field in fields(hooks)] == [
        "reduce_output",
        "reduce_logged_output",
        "extract_provider_session_id",
    ]
    assert [field.name for field in fields(request)] == [
        "command",
        "worktree",
        "environment",
        "prompt",
        "run_kind",
        "role",
        "usage_limit_scope",
        "log_context",
        "provider_session_id",
        "output_hooks",
    ]
    assert [field.name for field in fields(result)] == [
        "output",
        "usage",
        "stdout_lines",
        "provider_session_id",
    ]
    assert request.prompt.cleanup_path is True
    assert request.output_hooks.reduce_output(["a", "b"]) == ("ab", None)
    assert result.stdout_lines == ("line 1", "line 2")
    with pytest.raises(FrozenInstanceError):
        setattr(request, "command", "changed")
    with pytest.raises(FrozenInstanceError):
        setattr(result, "output", "changed")


def test_runtime_star_import_uses_lifecycle_surface_while_removed_legacy_aliases_fail_direct_import() -> (
    None
):
    exported_names: dict[str, object] = {}

    exec("from agent_runtime.runtime import *", {}, exported_names)

    assert "EphemeralRunRequest" in exported_names
    assert "RuntimeClient" in exported_names
    assert "ResumedSessionRunRequest" in exported_names
    assert "EphemeralRuntime" not in exported_names
    assert "NewSessionRuntime" not in exported_names
    assert "ResumedSessionRuntime" not in exported_names
    assert "EphemeralRuntimeExecutionAdapter" not in exported_names
    assert "NewSessionRuntimeExecutionAdapter" not in exported_names
    assert "ResumedSessionRuntimeExecutionAdapter" not in exported_names
    assert "ResumableRuntime" not in exported_names
    assert "ResumableRunRequest" not in exported_names
    assert "OneShotRuntime" not in exported_names
    assert "OneShotRunRequest" not in exported_names
    with pytest.raises(ImportError):
        exec("from agent_runtime.runtime import EphemeralRuntime", {}, {})
    with pytest.raises(ImportError):
        exec("from agent_runtime.runtime import NewSessionRuntime", {}, {})
    with pytest.raises(ImportError):
        exec("from agent_runtime.runtime import ResumedSessionRuntime", {}, {})
    with pytest.raises(ImportError):
        exec("from agent_runtime.runtime import ResumableRunRequest", {}, {})
    with pytest.raises(ImportError):
        exec("from agent_runtime.runtime import OneShotRuntime", {}, {})
    with pytest.raises(ImportError):
        exec("from agent_runtime.runtime import OneShotRunRequest", {}, {})


@pytest.mark.parametrize(
    "removed_name",
    [
        "OneShotRunRequest",
        "OneShotRunResult",
        "OneShotResultMetadata",
        "OneShotRuntime",
        "OneShotRuntimeExecutionAdapter",
        "OneShotRuntimeMetadata",
    ],
)
def test_runtime_direct_import_rejects_removed_legacy_names(
    removed_name: str,
) -> None:
    with pytest.raises(ImportError):
        exec(f"from agent_runtime.runtime import {removed_name}", {}, {})

    with pytest.raises(AttributeError):
        getattr(prompt_runtime, removed_name)


def test_runtime_direct_import_rejects_removed_resumable_completed_result_names() -> (
    None
):
    with pytest.raises(AttributeError):
        getattr(prompt_runtime, "ResumableRuntime")
    with pytest.raises(AttributeError):
        getattr(prompt_runtime, "ResumableRuntimeExecutionAdapter")
    with pytest.raises(ImportError):
        exec("from agent_runtime.runtime import ResumableRuntime", {}, {})
    with pytest.raises(ImportError):
        exec(
            "from agent_runtime.runtime import ResumableRuntimeExecutionAdapter", {}, {}
        )
    with pytest.raises(ImportError):
        exec("from agent_runtime.runtime import ResumableRunResult", {}, {})
    with pytest.raises(ImportError):
        exec("from agent_runtime.runtime import ResumableRuntimeMetadata", {}, {})


def test_types_module_exposes_stage_selection_as_the_only_stage_chain_value() -> None:
    types_module = importlib.import_module("agent_runtime.types")

    assert types_module.StageSelection.__module__.startswith("agent_runtime")
    assert not hasattr(types_module, "StageOverride")
    with pytest.raises(ImportError, match="StageOverride"):
        exec("from agent_runtime.types import StageOverride", {})


def test_runtime_surface_exposes_resumed_session_lifecycle_names() -> None:
    assert {
        "NewSessionRunRequest",
        "ResumedSessionRunRequest",
        "RuntimeClient",
    } <= set(prompt_runtime.__all__)
    assert hasattr(prompt_runtime, "ResumedSessionRunRequest")
    assert hasattr(prompt_runtime, "RuntimeClient")
    assert prompt_runtime.ResumedSessionRunRequest.__name__ == (
        "ResumedSessionRunRequest"
    )


def test_runtime_lifecycle_request_values_expose_invocation_dir_without_public_worktree_alias(
    stage_selection_factory,
) -> None:
    ephemeral_request = prompt_runtime.EphemeralRunRequest(
        prompt="already rendered prompt",
        invocation_dir=Path("/tmp/worktree"),
        stage=stage_selection_factory(service="codex"),
        tool_access=contracts_runtime.ToolAccess.no_tools(),
    )
    new_session_request = prompt_runtime.NewSessionRunRequest(
        prompt="already rendered prompt",
        invocation_dir=Path("/tmp/worktree"),
        stage=stage_selection_factory(service="codex"),
        role=InvocationRole("implementer"),
        tool_access=contracts_runtime.ToolAccess.no_tools(),
    )
    resumed_session_request = prompt_runtime.ResumedSessionRunRequest(
        prompt="already rendered prompt",
        invocation_dir=Path("/tmp/worktree"),
        model="gpt-5.4",
        effort="medium",
        session_plan=session_planning_runtime.ResumableSessionPlan(
            role=InvocationRole("implementer"),
            worktree=Path("/tmp/worktree"),
            namespace="main",
            service=cast(Any, object()),
            run_kind=RunKind.FRESH,
            provider_state_dir=None,
            provider_session_id=None,
            auth_seeding_requirement=(
                session_planning_runtime.AuthSeedingRequirement.NOT_REQUIRED
            ),
        ),
        tool_access=contracts_runtime.ToolAccess.no_tools(),
    )

    assert ephemeral_request.invocation_dir == Path("/tmp/worktree")
    assert new_session_request.invocation_dir == Path("/tmp/worktree")
    assert resumed_session_request.invocation_dir.host_path == Path("/tmp/worktree")
    for request in (
        ephemeral_request,
        new_session_request,
        resumed_session_request,
    ):
        with pytest.raises(AttributeError):
            getattr(request, "worktree")


def test_runtime_lifecycle_values_keep_runtime_module_names_after_extraction() -> None:
    for exported_name in (
        "Continuation",
        "EphemeralResultMetadata",
        "EphemeralRunRequest",
        "EphemeralRunResult",
        "EphemeralRuntimeMetadata",
        "InvocationRecord",
        "NewSessionRunRequest",
        "ProviderAuth",
        "ResumedSessionRunRequest",
        "RuntimeOutcome",
        "SessionRunResult",
        "SessionRuntimeMetadata",
    ):
        assert getattr(prompt_runtime, exported_name).__module__ == (
            "agent_runtime.runtime"
        )


def test_runtime_package_root_exports_keep_runtime_lifecycle_identity() -> None:
    assert runtime.Continuation is prompt_runtime.Continuation
    assert runtime.ProviderAuth is prompt_runtime.ProviderAuth
    assert runtime.RuntimeOutcome is prompt_runtime.RuntimeOutcome


def test_internal_runtime_compatibility_module_keeps_resume_wrapper_private() -> None:
    runtime_instance = compat_runtime.ResumedSessionRuntime(
        execution_adapter=cast(
            compat_runtime.ResumedSessionRuntimeExecutionAdapter,
            object(),
        )
    )

    assert hasattr(runtime_instance, "run_resumed_session")
    assert not hasattr(runtime_instance, "run_resumable_prompt")


def test_contracts_expose_execution_provider_as_canonical_public_protocol_name() -> (
    None
):
    contracts = importlib.import_module("agent_runtime.contracts")

    assert "ExecutionProvider" in contracts.__all__
    assert "ResumableExecutionProvider" in contracts.__all__
    assert not hasattr(contracts, "ExecutionService")
    assert not hasattr(contracts, "ResidentExecutionProvider")
    with pytest.raises(AttributeError):
        getattr(runtime, "ExecutionProvider")
    with pytest.raises(ImportError):
        exec("from agent_runtime import ExecutionProvider", {}, {})


def test_session_planning_surface_uses_resumable_vocabulary() -> None:
    assert not hasattr(session_planning_runtime, "ResidentSessionPlan")
    assert not hasattr(session_planning_runtime, "ResidentSessionPlanRequest")
    assert not hasattr(session_planning_runtime, "plan_resident_session")
    assert {
        "ResumableSessionPlan",
        "ResumableSessionPlanRequest",
        "plan_resumable_session",
    } <= set(session_planning_runtime.__all__)


def test_provider_session_planning_surface_exposes_immutable_decision_only() -> None:
    assert session_planning_runtime.ProviderSessionPlanRequest.__name__ == (
        "ProviderSessionPlanRequest"
    )
    assert not hasattr(session_planning_runtime, "ProviderRunStatePlan")
    assert not hasattr(session_planning_runtime, "plan_provider_run_state")
    assert not hasattr(
        session_planning_runtime,
        "record_observed_provider_session_id",
    )
    assert not hasattr(
        session_planning_runtime,
        "record_successful_provider_session_metadata",
    )
    assert {
        "ProviderSessionDecision",
        "ProviderSessionPlanRequest",
        "plan_provider_session",
    } <= set(session_planning_runtime.__all__)


def test_provider_session_dtos_remain_on_focused_session_seam() -> None:
    assert session_runtime.ProviderSessionState.__module__ == "agent_runtime.session"
    assert (
        session_runtime.ProviderSessionStateRequest.__module__
        == "agent_runtime.session"
    )


def test_provider_session_seams_consolidate_public_session_store_vocabulary() -> None:
    assert "SessionStore" in session_runtime.__all__
    assert not hasattr(session_runtime, "ServiceResumeIdentityStore")
    assert not hasattr(
        importlib.import_module("agent_runtime.contracts"),
        "ProviderSessionRecordingStore",
    )


def test_provider_session_adapter_public_seam_stays_narrow() -> None:
    assert provider_session_adapter_runtime.__all__ == [
        "ProviderSessionAdapter",
        "ProviderSessionPlanningFacts",
        "ProviderSessionPlanningRequest",
    ]
    adapter_members = provider_session_adapter_runtime.ProviderSessionAdapter.__dict__

    assert "provider_session_planning_facts" in adapter_members
    assert "provider_session_state" in adapter_members
    assert "prepare_local_provider_run_state" in adapter_members
    assert "record_provider_session_id" in adapter_members
    assert "provider_session_preferences" not in adapter_members
    assert "recover_provider_session_id" not in adapter_members
    assert "is_exact_resumable_provider_session" not in adapter_members
    assert not hasattr(provider_session_adapter_runtime, "ProviderSessionService")


def test_provider_session_public_dtos_expose_only_runtime_planning_fields() -> None:
    assert [
        field.name for field in fields(session_runtime.ProviderSessionStateRequest)
    ] == [
        "session_store",
        "provider_state_dir",
        "has_resumable_provider_state",
        "state_dir_relpath",
        "require_exact_transcript_match",
    ]
    assert [field.name for field in fields(session_runtime.ProviderSessionState)] == [
        "run_kind",
        "provider_session_id",
        "state_dir_relpath",
        "state_dir_path",
        "exact_transcript_match",
        "persist_provider_session_id",
        "auth_seeding_requirement",
        "auth_seed_action",
        "use_service_state_dir_for_container",
    ]


def test_package_surface_exposes_invocation_role_value_object() -> None:
    role = runtime.InvocationRole("implementer")

    assert role.value == "implementer"


def test_package_surface_exposes_usage_limit_scope_value_object() -> None:
    usage_limit_scope = runtime.UsageLimitScope("quota-review")

    assert usage_limit_scope.value == "quota-review"


def test_tool_policy_inspect_only_resolves_to_provider_neutral_profile() -> None:
    profile = runtime.ToolPolicy.INSPECT_ONLY.profile

    assert profile.allowed_tools == ("Read", "Glob")
    assert profile.disallowed_tools == ()
    assert profile.strict_mcp_config is True


def test_tool_policy_none_resolves_to_closed_no_tools_profile() -> None:
    profile = runtime.ToolPolicy.NONE.profile

    assert profile.allowed_tools == ("none",)
    assert profile.disallowed_tools == ("all",)
    assert profile.strict_mcp_config is True


def test_runtime_surface_exposes_tool_policy_profiles_for_no_file_mutation_and_unrestricted() -> (
    None
):
    partial = runtime.ToolPolicy.NO_FILE_MUTATION.profile
    full = runtime.ToolPolicy.UNRESTRICTED.profile

    assert isinstance(partial, contracts_runtime.ToolPolicyProfile)
    assert partial.allowed_tools is None
    assert partial.disallowed_tools == ("Edit", "Write", "NotebookEdit")
    assert partial.strict_mcp_config is True
    assert isinstance(full, contracts_runtime.ToolPolicyProfile)
    assert full.allowed_tools is None
    assert full.disallowed_tools == ()
    assert full.strict_mcp_config is True


def test_tool_policy_profiles_stay_provider_neutral() -> None:
    for policy in runtime.ToolPolicy:
        profile = policy.profile
        rendered_values = (profile.allowed_tools or ()) + profile.disallowed_tools

        assert profile.strict_mcp_config is True
        assert all(not value.startswith("-") for value in rendered_values)
        assert all(
            provider not in value.lower()
            for value in rendered_values
            for provider in ("claude", "codex", "opencode")
        )
