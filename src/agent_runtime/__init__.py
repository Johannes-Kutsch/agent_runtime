from __future__ import annotations

from typing import TYPE_CHECKING

from ._import_isolation import assert_runtime_import_isolation

from .agent_log import AgentInvocationLog, LogicalAgentInvocationLog, WorkInvocationLog
from .contracts import (
    AssistantTurn,
    ExecutionService,
    CredentialFailure,
    ExecutionProvider,
    HardError,
    ParsedTurn,
    PromptTokens,
    ProviderSessionRecordingStore,
    ProviderStatePreparationAction,
    ResumabilityProvider,
    ResidentExecutionProvider,
    Result,
    ServiceSelectionProvider,
    SessionPlanningProvider,
    ToolPolicy,
    TransientError,
    UnsupportedTokens,
    UsageLimit,
)
from .execution_contracts import (
    RunSessionPlan,
    TextOutputAdapter,
    WorkInvocationDependencies,
    WorkInvocationRequest,
)
from .errors import (
    AgentCredentialFailureError,
    AgentFailedError,
    AgentRuntimeError,
    AgentTimeoutError,
    HardAgentError,
    RuntimeConfigurationError,
    TransientAgentError,
    UsageLimitError,
)
from .provider_errors import ProviderErrorObservation
from .provider_session_adapter import ProviderSessionAdapter
from .roles import AgentRole
from .session import (
    ProviderSessionPreferences,
    ProviderSessionPreferencesRequest,
    ProviderSessionState,
    ProviderSessionStateRequest,
    RunKind,
)
from .types import StageOverride

if TYPE_CHECKING:
    from agent_runtime.session_planning import (
        ProviderRunStatePlan,
        ProviderRunStatePlanRequest,
        ResidentSessionPlan,
        ResidentSessionPlanRequest,
        plan_provider_run_state,
        plan_resident_session,
    )
    from agent_runtime.service_registry import ServiceRegistry
    from agent_runtime.stage_priority_chain import (
        ChainEntry,
        ConfiguredCandidateChain,
        ConfiguredCandidateSelection,
    )
    from agent_runtime.usage_limit_decision import (
        ContinueNow,
        SleepUntil,
        Stop,
        UsageLimitContinuationDecision,
        UsageLimitOutcome,
    )
    from agent_runtime.work import (
        CancellationToken,
        invoke_work,
    )

__all__ = [
    "AgentCredentialFailureError",
    "AgentInvocationLog",
    "AgentFailedError",
    "AgentRuntimeError",
    "AgentRole",
    "AgentTimeoutError",
    "AssistantTurn",
    "ExecutionService",
    "ExecutionProvider",
    "CancellationToken",
    "ChainEntry",
    "ConfiguredCandidateChain",
    "ConfiguredCandidateSelection",
    "CredentialFailure",
    "HardError",
    "HardAgentError",
    "LogicalAgentInvocationLog",
    "ParsedTurn",
    "ProviderErrorObservation",
    "ProviderSessionAdapter",
    "ProviderSessionPreferences",
    "ProviderSessionPreferencesRequest",
    "ProviderSessionRecordingStore",
    "ProviderSessionState",
    "ProviderSessionStateRequest",
    "ProviderStatePreparationAction",
    "PromptTokens",
    "ResumabilityProvider",
    "ResidentExecutionProvider",
    "PromptRunRequest",
    "PromptRunSession",
    "PromptRuntime",
    "PromptRuntimeExecutionAdapter",
    "Result",
    "RunSessionPlan",
    "RuntimeConfigurationError",
    "RunKind",
    "ProviderRunStatePlan",
    "ProviderRunStatePlanRequest",
    "ResidentSessionPlan",
    "ResidentSessionPlanRequest",
    "ServiceRegistry",
    "ServiceSelectionProvider",
    "SleepUntil",
    "Stop",
    "StageOverride",
    "TextOutputAdapter",
    "OneShotRunRequest",
    "OneShotRunResult",
    "OneShotRuntime",
    "OneShotRuntimeExecutionAdapter",
    "OneShotRuntimeMetadata",
    "ResidentRunRequest",
    "ResidentRunResult",
    "ResidentRuntime",
    "ResidentRuntimeExecutionAdapter",
    "ResidentRuntimeMetadata",
    "chain_entries",
    "ContinueNow",
    "configured_candidate_chain",
    "invoke_work",
    "iter_stage_chain",
    "plan_provider_run_state",
    "referenced_service_names",
    "render_chain_label",
    "select_configured_candidate_chain",
    "plan_resident_session",
    "UsageLimitContinuationDecision",
    "UsageLimitOutcome",
    "ToolPolicy",
    "TransientError",
    "TransientAgentError",
    "UnsupportedTokens",
    "UsageLimit",
    "UsageLimitError",
    "SessionPlanningProvider",
    "validation_labels",
    "decide_usage_limit_continuation",
    "WorkInvocationDependencies",
    "WorkInvocationRequest",
    "WorkInvocationLog",
    "WorktreeMount",
    "run_one_shot",
    "run_prompt",
    "run_resident_prompt",
]


def __getattr__(name: str):
    if name in {
        "OneShotRunRequest",
        "OneShotRunResult",
        "OneShotRuntime",
        "OneShotRuntimeExecutionAdapter",
        "OneShotRuntimeMetadata",
        "ResidentRunRequest",
        "ResidentRunResult",
        "ResidentRuntime",
        "ResidentRuntimeExecutionAdapter",
        "ResidentRuntimeMetadata",
        "PromptRunRequest",
        "PromptRunSession",
        "PromptRuntimeExecutionAdapter",
        "PromptRuntime",
        "WorktreeMount",
        "run_one_shot",
        "run_prompt",
        "run_resident_prompt",
    }:
        if name in {
            "OneShotRunRequest",
            "OneShotRunResult",
            "OneShotRuntime",
            "OneShotRuntimeExecutionAdapter",
            "OneShotRuntimeMetadata",
            "ResidentRunRequest",
            "ResidentRunResult",
            "ResidentRuntime",
            "ResidentRuntimeExecutionAdapter",
            "ResidentRuntimeMetadata",
            "PromptRuntime",
            "run_one_shot",
            "run_prompt",
            "run_resident_prompt",
        }:
            from agent_runtime import runtime

            return getattr(runtime, name)
        from agent_runtime import execution_contracts

        return getattr(execution_contracts, name)
    if name == "ServiceRegistry":
        from agent_runtime.service_registry import ServiceRegistry

        return ServiceRegistry
    if name in {
        "CancellationToken",
        "PreparedProviderRunSession",
        "PreparedSession",
        "PrepareSessionAdapter",
        "RunSessionPlan",
        "SetupFailureTranslator",
        "StatusDisplayFactory",
        "StatusRowFactory",
        "TextOutputAdapter",
        "WorkExecutionAdapter",
        "WorkInvocationDependencies",
        "WorkInvocationRequest",
        "WorkModelDisplayMetadata",
        "WorkOutputAdapter",
        "WorkStatusDisplay",
        "WorkStatusRow",
        "invoke_work",
    }:
        if name == "invoke_work" or name == "CancellationToken":
            from agent_runtime import work

            return getattr(work, name)
        from agent_runtime import execution_contracts

        return getattr(execution_contracts, name)
    if name in {
        "ProviderRunStatePlan",
        "ProviderRunStatePlanRequest",
        "ResidentSessionPlan",
        "ResidentSessionPlanRequest",
        "plan_provider_run_state",
        "plan_resident_session",
    }:
        from agent_runtime import session_planning

        return getattr(session_planning, name)
    if name in {
        "ContinueNow",
        "SleepUntil",
        "Stop",
        "UsageLimitContinuationDecision",
        "UsageLimitOutcome",
        "decide_usage_limit_continuation",
    }:
        from agent_runtime import usage_limit_decision

        return getattr(usage_limit_decision, name)
    if name in {
        "ChainEntry",
        "ConfiguredCandidateChain",
        "ConfiguredCandidateSelection",
        "chain_entries",
        "configured_candidate_chain",
        "iter_stage_chain",
        "referenced_service_names",
        "render_chain_label",
        "select_configured_candidate_chain",
        "validation_labels",
    }:
        from agent_runtime import stage_priority_chain

        return getattr(stage_priority_chain, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


assert_runtime_import_isolation(importer=__name__)
