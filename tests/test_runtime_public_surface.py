from __future__ import annotations

import importlib
from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from typing import Callable, cast

import pytest

import agent_runtime as runtime
import agent_runtime.provider_session_adapter as provider_session_adapter_runtime
import agent_runtime.runtime as prompt_runtime
import agent_runtime.session as session_runtime
import agent_runtime.session_planning as session_planning_runtime
from agent_runtime.contracts import ExecutionProvider, ResumabilityProvider
from agent_runtime.roles import InvocationRole
from agent_runtime.session import RunKind
from agent_runtime.session_planning import (
    ResumableSessionPlanRequest,
    plan_resumable_session,
)
from agent_runtime.errors import AgentRuntimeError

from tests.runtime_boundary_fakes import (
    ResidentPlanningProviderSessionAdapterFake as _ResidentPlanningProviderSessionAdapter,
    SessionStoreFake as _SessionStore,
)


def test_package_exports_runtime_surface() -> None:
    assert runtime.__all__ == [
        "AgentCredentialFailureError",
        "AgentFailedError",
        "AgentRuntimeError",
        "AgentTimeoutError",
        "Continuation",
        "HardAgentError",
        "ExecutionProvider",
        "InvocationRole",
        "InvocationProgress",
        "ProviderAuth",
        "ProviderSessionAdapter",
        "RuntimeClient",
        "RuntimeConfigurationError",
        "RuntimeOutcome",
        "RunKind",
        "StageSelection",
        "ToolAccess",
        "ToolPolicy",
        "ToolPolicyProfile",
        "TransientAgentError",
        "UsageLimitError",
        "UsageLimitScope",
    ]
    assert runtime.StageSelection.__module__.startswith("agent_runtime")
    assert not hasattr(runtime, "StageOverride")
    assert runtime.AgentRuntimeError is AgentRuntimeError
    assert runtime.RuntimeOutcome is prompt_runtime.RuntimeOutcome
    assert not hasattr(runtime, "assert_runtime_import_isolation")
    assert not hasattr(runtime, "run_prompt")
    assert not hasattr(runtime, "ServiceRegistry")
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
        "EphemeralRunRequest",
        "EphemeralRunResult",
        "EphemeralResultMetadata",
        "EphemeralRuntime",
        "EphemeralRuntimeExecutionAdapter",
        "EphemeralRuntimeMetadata",
        "Continuation",
        "ResumedSessionRunRequest",
        "SessionRunResult",
        "ResumedSessionRuntime",
        "ResumedSessionRuntimeExecutionAdapter",
        "SessionRuntimeMetadata",
    } <= set(prompt_runtime.__all__)
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


def test_runtime_star_import_uses_lifecycle_surface_while_removed_legacy_aliases_fail_direct_import() -> (
    None
):
    exported_names: dict[str, object] = {}

    exec("from agent_runtime.runtime import *", {}, exported_names)

    assert "EphemeralRuntime" in exported_names
    assert "EphemeralRunRequest" in exported_names
    assert "ResumedSessionRuntime" in exported_names
    assert "ResumedSessionRunRequest" in exported_names
    assert "ResumableRuntime" not in exported_names
    assert "ResumableRunRequest" not in exported_names
    assert "OneShotRuntime" not in exported_names
    assert "OneShotRunRequest" not in exported_names
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
        "NewSessionRuntime",
        "ResumedSessionRunRequest",
        "ResumedSessionRuntime",
    } <= set(prompt_runtime.__all__)
    assert hasattr(prompt_runtime, "ResumedSessionRunRequest")
    assert hasattr(prompt_runtime, "ResumedSessionRuntime")
    assert prompt_runtime.ResumedSessionRunRequest.__name__ == (
        "ResumedSessionRunRequest"
    )


def test_runtime_lifecycle_values_keep_runtime_module_names_after_extraction() -> None:
    for exported_name in (
        "Continuation",
        "EphemeralResultMetadata",
        "EphemeralRunRequest",
        "EphemeralRunResult",
        "EphemeralRuntimeMetadata",
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


def test_resumed_session_runtime_exposes_only_lifecycle_resume_method() -> None:
    runtime_instance = prompt_runtime.ResumedSessionRuntime(
        execution_adapter=cast(
            prompt_runtime.ResumedSessionRuntimeExecutionAdapter,
            object(),
        )
    )

    assert hasattr(runtime_instance, "run_resumed_session")
    assert not hasattr(runtime_instance, "run_resumable_prompt")


def test_readme_guides_consumers_to_lifecycle_session_entrypoints() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "ResumedSessionRuntime" in readme
    assert "ResumedSessionRunRequest" in readme
    assert "run_resumed_session" in readme
    assert "run_resumable_prompt" not in readme


def test_contracts_expose_execution_provider_as_canonical_public_protocol_name() -> (
    None
):
    contracts = importlib.import_module("agent_runtime.contracts")

    assert "ExecutionProvider" in contracts.__all__
    assert "ResumableExecutionProvider" in contracts.__all__
    assert not hasattr(contracts, "ExecutionService")
    assert not hasattr(contracts, "ResidentExecutionProvider")
    assert runtime.ExecutionProvider is contracts.ExecutionProvider


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


def test_provider_session_planning_returns_immutable_decision_value(
    execution_service_factory: Callable[..., ExecutionProvider],
    session_store_factory: Callable[..., _SessionStore],
    resident_provider_session_adapter: _ResidentPlanningProviderSessionAdapter,
) -> None:
    provider_session_decision = session_planning_runtime.plan_provider_session(
        session_planning_runtime.ProviderSessionPlanRequest(
            worktree=Path("."),
            role=InvocationRole("implementer"),
            namespace="main",
            resumability_service=cast(
                ResumabilityProvider, execution_service_factory()
            ),
            session_store=session_store_factory(),
            provider_session_adapter=resident_provider_session_adapter,
        )
    )

    assert (
        provider_session_decision
        == session_planning_runtime.ProviderSessionDecision(
            run_kind=RunKind.RESUME,
            provider_session_id="recovered-session",
            state_dir_relpath="state/",
            state_dir_path=Path("state"),
            recovered_session_id_persistence=(
                session_planning_runtime.RecoveredSessionIdPersistence.SKIP
            ),
            service_state_dir=Path("state"),
            exact_transcript_match=False,
            auth_seeding_requirement=(
                session_planning_runtime.AuthSeedingRequirement.NOT_REQUIRED
            ),
            auth_seed_action=None,
        )
    )
    with pytest.raises(FrozenInstanceError):
        setattr(provider_session_decision, "provider_session_id", "other")


def test_resumable_session_plan_exposes_public_value_fields_only(
    execution_service_factory: Callable[..., ExecutionProvider],
    session_store_factory: Callable[..., _SessionStore],
    resident_provider_session_adapter: _ResidentPlanningProviderSessionAdapter,
) -> None:
    service = execution_service_factory()

    session_plan = plan_resumable_session(
        ResumableSessionPlanRequest(
            worktree=Path("."),
            role=InvocationRole("implementer"),
            namespace="main",
            service=service,
            session_store=session_store_factory(),
            provider_session_adapter=resident_provider_session_adapter,
        )
    )

    assert session_plan.role == InvocationRole("implementer")
    assert session_plan.worktree == Path(".")
    assert session_plan.namespace == "main"
    assert session_plan.service is service
    assert session_plan.run_kind is RunKind.RESUME
    assert session_plan.provider_state_dir == Path("state")
    assert session_plan.provider_session_id == "recovered-session"
    assert (
        session_plan.auth_seeding_requirement
        is session_planning_runtime.AuthSeedingRequirement.NOT_REQUIRED
    )
    assert session_plan.auth_seed_action is None
    assert session_plan.exact_transcript_match is False
    assert session_plan.usage_limit_scope is None
    with pytest.raises(FrozenInstanceError):
        setattr(session_plan, "provider_state_dir", Path("other-state"))


def test_resumable_session_plan_hides_container_state_selection_metadata(
    execution_service_factory: Callable[..., ExecutionProvider],
    session_store_factory: Callable[..., _SessionStore],
    resident_provider_session_adapter: _ResidentPlanningProviderSessionAdapter,
) -> None:
    service = execution_service_factory()

    session_plan = plan_resumable_session(
        ResumableSessionPlanRequest(
            worktree=Path("."),
            role=InvocationRole("implementer"),
            namespace="main",
            service=service,
            session_store=session_store_factory(),
            provider_session_adapter=resident_provider_session_adapter,
        )
    )

    field_names = {field.name for field in fields(session_plan)}

    assert "service_state_dir" not in field_names
    assert "use_service_state_dir_for_container" not in field_names


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


def test_tool_policy_restricted_resolves_to_provider_neutral_profile() -> None:
    profile = runtime.ToolPolicy.RESTRICTED.profile

    assert profile.allowed_tools == ("Read", "Glob")
    assert profile.disallowed_tools == ()
    assert profile.strict_mcp_config is True


def test_runtime_surface_exposes_tool_policy_profiles_for_partial_and_full() -> None:
    partial = runtime.ToolPolicy.PARTIAL.profile
    full = runtime.ToolPolicy.FULL.profile

    assert isinstance(partial, prompt_runtime.ToolPolicyProfile)
    assert partial.allowed_tools is None
    assert partial.disallowed_tools == ("Edit", "Write", "NotebookEdit")
    assert partial.strict_mcp_config is True
    assert isinstance(full, prompt_runtime.ToolPolicyProfile)
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
