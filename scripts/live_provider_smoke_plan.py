from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping, Sequence, Any
from uuid import uuid4

from agent_runtime.errors import RuntimeConfigurationError
from agent_runtime.runtime import ToolPolicy


class LiveSmokeProviderSelectionStatus(str, Enum):
    RUNNABLE = "runnable"
    SKIPPED = "skipped"
    CONFIG_ERROR = "config_error"


class LiveSmokeCaseStatus(str, Enum):
    PASSED = "passed"
    SKIPPED = "skipped"
    CONFIG_ERROR = "config_error"
    USAGE_LIMITED = "usage_limited"
    NO_SERVICE_AVAILABLE = "no_service_available"
    FAILED = "failed"
    ERROR = "error"


@dataclass(frozen=True)
class LiveSmokeCaseResult:
    service: str
    mode: str
    policy: str | None
    status: LiveSmokeCaseStatus
    diagnostic: str | None = None
    required: bool = True


SUPPORTED_PROVIDERS: tuple[str, ...] = ("claude", "codex", "opencode")
LIVE_SMOKE_CLAUDE_MODEL_ENV = "LIVE_SMOKE_CLAUDE_MODEL"
LIVE_SMOKE_CLAUDE_EFFORT_ENV = "LIVE_SMOKE_CLAUDE_EFFORT"
LIVE_SMOKE_CODEX_MODEL_ENV = "LIVE_SMOKE_CODEX_MODEL"
LIVE_SMOKE_CODEX_EFFORT_ENV = "LIVE_SMOKE_CODEX_EFFORT"
LIVE_SMOKE_OPENCODE_MODEL_ENV = "LIVE_SMOKE_OPENCODE_MODEL"
LIVE_SMOKE_OPENCODE_EFFORT_ENV = "LIVE_SMOKE_OPENCODE_EFFORT"
LIVE_SMOKE_DEFAULTS_VERIFIED_ON = "2026-06-20"
LIVE_SMOKE_DEFAULTS: dict[str, tuple[str, str]] = {
    "claude": ("haiku", "low"),
    "codex": ("gpt-5.4-mini", "low"),
    "opencode": ("deepseek-v4-flash", "medium"),
}

_PROVIDER_MODEL_ENV_BY_SERVICE = {
    "claude": LIVE_SMOKE_CLAUDE_MODEL_ENV,
    "codex": LIVE_SMOKE_CODEX_MODEL_ENV,
    "opencode": LIVE_SMOKE_OPENCODE_MODEL_ENV,
}
_PROVIDER_EFFORT_ENV_BY_SERVICE = {
    "claude": LIVE_SMOKE_CLAUDE_EFFORT_ENV,
    "codex": LIVE_SMOKE_CODEX_EFFORT_ENV,
    "opencode": LIVE_SMOKE_OPENCODE_EFFORT_ENV,
}

_PROVIDER_CLAUDE_TOKEN_ENV = "CLAUDE_CODE_OAUTH_TOKEN"
_PROVIDER_CODEX_HOME_AUTH_PATH = Path.home() / ".codex" / "auth.json"
_PROVIDER_OPENCODE_ENV = "OPENCODE_GO_API_KEY"

_RUN_ID_SAFE_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


@dataclass(frozen=True)
class ProviderSelection:
    providers: tuple[str, ...]
    include_all: bool


@dataclass(frozen=True)
class ProviderPlan:
    service: str
    model: str
    effort: str
    status: LiveSmokeProviderSelectionStatus
    reason: str | None = None


@dataclass(frozen=True)
class PlannedCase:
    service: str
    mode: str
    policy: str | None
    model: str
    effort: str


@dataclass(frozen=True)
class DryRunPlannedCase:
    service: str
    mode: str
    policy: str | None
    model: str
    effort: str
    artifact_path: Path


@dataclass(frozen=True)
class DryRunPlan:
    run_id: str
    cases: tuple[DryRunPlannedCase, ...]
    provider_plans: tuple[ProviderPlan, ...]
    artifact_root: Path


def classify_live_smoke_case_result(
    *,
    case: PlannedCase,
    runtime_outcome: Any | None = None,
    runtime_exception: BaseException | None = None,
    artifact_error: str | None = None,
    required_output_non_empty: bool = True,
    required_continuation_text: str | None = None,
    required_continuation: bool = False,
    required: bool = True,
) -> LiveSmokeCaseResult:
    if artifact_error is not None:
        return LiveSmokeCaseResult(
            service=case.service,
            mode=case.mode,
            policy=case.policy,
            status=LiveSmokeCaseStatus.ERROR,
            diagnostic=artifact_error,
            required=required,
        )
    if runtime_exception is not None:
        return LiveSmokeCaseResult(
            service=case.service,
            mode=case.mode,
            policy=case.policy,
            status=LiveSmokeCaseStatus.ERROR,
            diagnostic=str(runtime_exception),
            required=required,
        )

    if runtime_outcome is None:
        return LiveSmokeCaseResult(
            service=case.service,
            mode=case.mode,
            policy=case.policy,
            status=LiveSmokeCaseStatus.ERROR,
            diagnostic="no runtime outcome",
            required=required,
        )

    outcome_kind = getattr(runtime_outcome, "kind", None)
    if outcome_kind == "completed":
        metadata_mismatch = _completed_outcome_metadata_mismatch(
            case=case,
            runtime_outcome=runtime_outcome,
            required_output_non_empty=required_output_non_empty,
        )
        if metadata_mismatch is not None:
            return LiveSmokeCaseResult(
                service=case.service,
                mode=case.mode,
                policy=case.policy,
                status=LiveSmokeCaseStatus.FAILED,
                diagnostic=metadata_mismatch,
                required=required,
            )
        if required_output_non_empty and not str(
            getattr(runtime_outcome, "output", "")
        ):
            return LiveSmokeCaseResult(
                service=case.service,
                mode=case.mode,
                policy=case.policy,
                status=LiveSmokeCaseStatus.FAILED,
                diagnostic="completed outcome missing required output",
                required=required,
            )
        if (
            required_continuation
            and getattr(runtime_outcome, "continuation", None) is None
        ):
            return LiveSmokeCaseResult(
                service=case.service,
                mode=case.mode,
                policy=case.policy,
                status=LiveSmokeCaseStatus.FAILED,
                diagnostic="completed outcome missing required continuation",
                required=required,
            )
        if required_continuation_text is not None:
            continuation = getattr(runtime_outcome, "continuation", None)
            continuation_text = (
                continuation.serialized if continuation is not None else None
            )
            if continuation_text is None:
                continuation_text = getattr(runtime_outcome, "result", None)
            if required_continuation_text not in str(continuation_text or ""):
                return LiveSmokeCaseResult(
                    service=case.service,
                    mode=case.mode,
                    policy=case.policy,
                    status=LiveSmokeCaseStatus.FAILED,
                    diagnostic="completed outcome missing required continuation evidence",
                    required=required,
                )
        return LiveSmokeCaseResult(
            service=case.service,
            mode=case.mode,
            policy=case.policy,
            status=LiveSmokeCaseStatus.PASSED,
            required=required,
        )

    if outcome_kind == "usage_limited":
        return LiveSmokeCaseResult(
            service=case.service,
            mode=case.mode,
            policy=case.policy,
            status=LiveSmokeCaseStatus.USAGE_LIMITED,
            required=required,
            diagnostic=getattr(runtime_outcome, "output", None),
        )

    if outcome_kind == "no_service_available":
        return LiveSmokeCaseResult(
            service=case.service,
            mode=case.mode,
            policy=case.policy,
            status=LiveSmokeCaseStatus.NO_SERVICE_AVAILABLE,
            required=required,
            diagnostic=getattr(runtime_outcome, "output", None),
        )

    if outcome_kind == "retryable_provider_failure":
        return LiveSmokeCaseResult(
            service=case.service,
            mode=case.mode,
            policy=case.policy,
            status=LiveSmokeCaseStatus.FAILED,
            required=required,
            diagnostic=getattr(runtime_outcome, "output", None),
        )

    if outcome_kind == "timed_out":
        return LiveSmokeCaseResult(
            service=case.service,
            mode=case.mode,
            policy=case.policy,
            status=LiveSmokeCaseStatus.FAILED,
            required=required,
            diagnostic=getattr(runtime_outcome, "output", None),
        )

    return LiveSmokeCaseResult(
        service=case.service,
        mode=case.mode,
        policy=case.policy,
        status=LiveSmokeCaseStatus.FAILED,
        required=required,
        diagnostic=getattr(runtime_outcome, "output", None),
    )


def _completed_outcome_metadata_mismatch(
    *,
    case: PlannedCase,
    runtime_outcome: Any,
    required_output_non_empty: bool,
) -> str | None:
    del required_output_non_empty
    selected = _extract_completed_outcome_metadata(
        case=case, runtime_outcome=runtime_outcome
    )
    expected: dict[str, Any] = {
        "service": case.service,
        "model": case.model,
        "effort": case.effort,
    }
    if case.policy is not None:
        expected["tool_policy"] = _tool_policy_from_case_policy(case.policy)

    mismatches = []
    for field, expected_value in expected.items():
        actual_value = selected.get(field)
        if actual_value is None:
            continue
        if actual_value != expected_value:
            mismatches.append(
                f"{field}: expected {expected_value!r}, got {actual_value!r}"
            )

    if mismatches:
        return "completed outcome metadata mismatch: " + ", ".join(mismatches)
    return None


def _extract_completed_outcome_metadata(
    *,
    case: PlannedCase,
    runtime_outcome: Any,
) -> dict[str, Any]:
    result = getattr(runtime_outcome, "result", None)
    if result is None:
        return {}
    if case.mode == "ephemeral":
        tool_access = getattr(result, "tool_access", None)
        tool_policy = _normalize_tool_policy(getattr(tool_access, "tool_policy", None))
        return {
            "service": getattr(result, "selected_service", None),
            "model": getattr(result, "selected_model", None),
            "effort": getattr(result, "selected_effort", None),
            "tool_policy": tool_policy,
        }
    if case.mode in {"new_session", "resumed_session"}:
        runtime_metadata = getattr(result, "runtime_metadata", None)
        return {
            "service": getattr(runtime_metadata, "service_name", None),
            "model": getattr(runtime_metadata, "selected_model", None),
            "effort": getattr(runtime_metadata, "selected_effort", None),
            "tool_policy": _normalize_tool_policy(
                getattr(runtime_metadata, "tool_policy", None)
            ),
        }
    return {}


def _tool_policy_from_case_policy(policy: str) -> ToolPolicy | str:
    return _normalize_tool_policy(policy) or policy


def _normalize_tool_policy(value: Any) -> ToolPolicy | str | None:
    if isinstance(value, ToolPolicy):
        return value
    if not isinstance(value, str):
        return value if value is not None else None
    normalized = value.strip()
    if not normalized:
        return None
    try:
        return ToolPolicy[normalized]
    except KeyError:
        try:
            return ToolPolicy[normalized.upper()]
        except KeyError:
            return normalized


def classify_live_smoke_preflight_case_result(
    *,
    case: PlannedCase,
    provider_plan: ProviderPlan,
    required: bool = True,
    dependent_skip: bool = False,
) -> LiveSmokeCaseResult:
    if provider_plan.status == LiveSmokeProviderSelectionStatus.CONFIG_ERROR:
        status = LiveSmokeCaseStatus.CONFIG_ERROR
    elif provider_plan.status == LiveSmokeProviderSelectionStatus.SKIPPED:
        status = LiveSmokeCaseStatus.SKIPPED
    else:
        raise ValueError(
            "Only non-runnable provider plans can be classified as preflight cases"
        )

    return LiveSmokeCaseResult(
        service=case.service,
        mode=case.mode,
        policy=case.policy,
        status=status,
        required=required if not dependent_skip else False,
        diagnostic=provider_plan.reason,
    )


def live_smoke_case_results_payload(
    case_results: Sequence[LiveSmokeCaseResult],
) -> dict[str, object]:
    cases = [
        {
            "service": result.service,
            "mode": result.mode,
            "policy": result.policy,
            "status": result.status.value,
            "diagnostic": result.diagnostic,
            "required": result.required,
        }
        for result in case_results
    ]
    return {
        "cases": cases,
        "status_counts": {
            status.value: sum(1 for result in case_results if result.status is status)
            for status in LiveSmokeCaseStatus
        },
    }


def compute_live_smoke_exit_status(
    case_results: Sequence[LiveSmokeCaseResult],
) -> int:
    for result in case_results:
        if result.status is LiveSmokeCaseStatus.PASSED:
            continue
        if result.status is LiveSmokeCaseStatus.SKIPPED and not result.required:
            continue
        return 1
    return 0


@dataclass(frozen=True)
class ProviderRuntimeConfiguration:
    service: str
    configured: bool


def build_dry_run_plan(
    provider_selection: str | Sequence[str],
    *,
    lifecycle_modes: tuple[str, ...],
    tool_policies: tuple[str, ...] = (),
    run_id: str | None = None,
    model_overrides: Mapping[str, str] | None = None,
    effort_overrides: Mapping[str, str] | None = None,
    artifact_root: Path | str = Path("."),
    env: Mapping[str, str] | None = None,
    claude_code_oauth_token: str | None = None,
    opencode_api_key: str | None = None,
    codex_auth_present: bool | None = None,
) -> DryRunPlan:
    selected = parse_provider_selection(provider_selection)
    resolved_run_id = resolve_run_id(run_id)
    provider_plans = plan_selected_providers(
        selected,
        model_overrides=model_overrides,
        effort_overrides=effort_overrides,
        env=env,
        claude_code_oauth_token=claude_code_oauth_token,
        opencode_api_key=opencode_api_key,
        codex_auth_present=codex_auth_present,
    )
    planned_cases = plan_smoke_cases(
        provider_plans,
        lifecycle_modes=lifecycle_modes,
        tool_policies=tool_policies,
    )
    artifact_root_path = Path(artifact_root)
    dry_run_cases = tuple(
        _build_dry_run_case(resolved_run_id, artifact_root_path, case)
        for case in planned_cases
    )
    return DryRunPlan(
        run_id=resolved_run_id,
        cases=dry_run_cases,
        provider_plans=provider_plans,
        artifact_root=artifact_root_path,
    )


def _build_dry_run_case(
    run_id: str, artifact_root: Path, planned_case: PlannedCase
) -> DryRunPlannedCase:
    artifact_path = (
        artifact_root
        / run_id
        / planned_case.service
        / planned_case.mode
        / (planned_case.policy if planned_case.policy else "default")
    )
    return DryRunPlannedCase(
        service=planned_case.service,
        mode=planned_case.mode,
        policy=planned_case.policy,
        model=planned_case.model,
        effort=planned_case.effort,
        artifact_path=artifact_path,
    )


def list_supported_providers(
    *,
    env: Mapping[str, str] | None = None,
    claude_code_oauth_token: str | None = None,
    opencode_api_key: str | None = None,
    codex_auth_present: bool | None = None,
) -> tuple[ProviderRuntimeConfiguration, ...]:
    return tuple(
        ProviderRuntimeConfiguration(
            service=provider,
            configured=_provider_has_runtime_config(
                provider,
                env=env,
                claude_code_oauth_token=claude_code_oauth_token,
                opencode_api_key=opencode_api_key,
                codex_auth_present=codex_auth_present,
            ),
        )
        for provider in SUPPORTED_PROVIDERS
    )


def dry_run_plan_to_json(dry_run_plan: DryRunPlan) -> str:
    payload = {
        "run_id": dry_run_plan.run_id,
        "artifact_root": str(dry_run_plan.artifact_root),
        "providers": [
            {
                "service": provider.service,
                "status": provider.status,
                "model": provider.model,
                "effort": provider.effort,
                "reason": provider.reason,
            }
            for provider in dry_run_plan.provider_plans
        ],
        "cases": [
            {
                "service": case.service,
                "mode": case.mode,
                "policy": case.policy,
                "model": case.model,
                "effort": case.effort,
                "artifact_path": str(case.artifact_path),
            }
            for case in dry_run_plan.cases
        ],
    }
    return json.dumps(payload, sort_keys=True)


def parse_provider_selection(selection: str | Sequence[str]) -> ProviderSelection:
    requested = _parse_selection_values(selection)
    if any(provider == "all" for provider in requested):
        if len(requested) != 1:
            raise RuntimeConfigurationError(
                "provider selection cannot combine 'all' with explicit providers"
            )
        return ProviderSelection(providers=SUPPORTED_PROVIDERS, include_all=True)
    _validate_supported_services(requested)
    unique: list[str] = []
    for provider in requested:
        if provider not in unique:
            unique.append(provider)
    if not unique:
        raise RuntimeConfigurationError("provider selection is empty")
    return ProviderSelection(providers=tuple(unique), include_all=False)


def _parse_selection_values(selection: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(selection, str):
        selected = tuple(part.strip() for part in selection.split(","))
    else:
        selected = tuple(part.strip() for part in selection)
    if not selected:
        raise RuntimeConfigurationError("provider selection is empty")
    return tuple(part for part in selected if part)


def _validate_supported_services(providers: tuple[str, ...]) -> None:
    unknown = [
        provider for provider in providers if provider not in SUPPORTED_PROVIDERS
    ]
    if unknown:
        raise RuntimeConfigurationError(
            f"Unsupported provider names: {', '.join(unknown)}"
        )


def resolve_model_and_effort(
    provider: str,
    *,
    cli_model: str | None,
    cli_effort: str | None,
    env: Mapping[str, str] | None = None,
) -> tuple[str, str]:
    if provider not in SUPPORTED_PROVIDERS:
        raise RuntimeConfigurationError(f"Unsupported provider name: {provider!r}")
    env_map = _resolve_env_map(env)
    model = (
        cli_model
        if cli_model
        else env_map.get(_PROVIDER_MODEL_ENV_BY_SERVICE[provider], "")
    )
    effort = (
        cli_effort
        if cli_effort
        else env_map.get(_PROVIDER_EFFORT_ENV_BY_SERVICE[provider], "")
    )
    if not model and not cli_model:
        model = LIVE_SMOKE_DEFAULTS[provider][0]
    if not effort and not cli_effort:
        effort = LIVE_SMOKE_DEFAULTS[provider][1]
    return model, effort


def _provider_has_runtime_config(
    provider: str,
    *,
    env: Mapping[str, str] | None = None,
    claude_code_oauth_token: str | None = None,
    opencode_api_key: str | None = None,
    codex_auth_present: bool | None = None,
) -> bool:
    env_map = _resolve_env_map(env)
    if provider == "claude":
        return bool(
            (claude_code_oauth_token or "").strip()
            or env_map.get(_PROVIDER_CLAUDE_TOKEN_ENV)
        )
    if provider == "opencode":
        return bool(
            (opencode_api_key or "").strip() or env_map.get(_PROVIDER_OPENCODE_ENV)
        )
    if provider == "codex":
        if codex_auth_present is not None:
            return bool(codex_auth_present)
        return _PROVIDER_CODEX_HOME_AUTH_PATH.exists()
    raise RuntimeConfigurationError(f"Unsupported provider name: {provider!r}")


def _resolve_env_map(env: Mapping[str, str] | None) -> Mapping[str, str]:
    return os.environ if env is None else env


def plan_selected_providers(
    provider_selection: ProviderSelection,
    *,
    model_overrides: Mapping[str, str] | None = None,
    effort_overrides: Mapping[str, str] | None = None,
    env: Mapping[str, str] | None = None,
    claude_code_oauth_token: str | None = None,
    opencode_api_key: str | None = None,
    codex_auth_present: bool | None = None,
) -> tuple[ProviderPlan, ...]:
    model_by_provider: dict[str, str] = dict(model_overrides or {})
    effort_by_provider: dict[str, str] = dict(effort_overrides or {})
    return tuple(
        _plan_provider(
            provider,
            model=model_by_provider.get(provider),
            effort=effort_by_provider.get(provider),
            env=env,
            claude_code_oauth_token=claude_code_oauth_token,
            opencode_api_key=opencode_api_key,
            codex_auth_present=codex_auth_present,
            include_all=provider_selection.include_all,
        )
        for provider in provider_selection.providers
    )


def _plan_provider(
    provider: str,
    *,
    model: str | None,
    effort: str | None,
    env: Mapping[str, str] | None,
    claude_code_oauth_token: str | None,
    opencode_api_key: str | None,
    codex_auth_present: bool | None,
    include_all: bool,
) -> ProviderPlan:
    resolved_model, resolved_effort = resolve_model_and_effort(
        provider,
        cli_model=model,
        cli_effort=effort,
        env=env,
    )
    if not resolved_model or not resolved_effort:
        reason = "missing model or effort"
        status = (
            LiveSmokeProviderSelectionStatus.SKIPPED
            if include_all
            else LiveSmokeProviderSelectionStatus.CONFIG_ERROR
        )
    elif not _provider_has_runtime_config(
        provider,
        env=env,
        claude_code_oauth_token=claude_code_oauth_token,
        opencode_api_key=opencode_api_key,
        codex_auth_present=codex_auth_present,
    ):
        reason = "provider not configured"
        status = (
            LiveSmokeProviderSelectionStatus.SKIPPED
            if include_all
            else LiveSmokeProviderSelectionStatus.CONFIG_ERROR
        )
    else:
        status = LiveSmokeProviderSelectionStatus.RUNNABLE
        reason = None
    return ProviderPlan(
        service=provider,
        model=resolved_model,
        effort=resolved_effort,
        status=status,
        reason=reason,
    )


def resolve_run_id(run_id: str | None = None) -> str:
    if not run_id:
        return uuid4().hex
    if run_id in {".", ".."}:
        raise RuntimeConfigurationError("run id must not be '.' or '..'")
    if not _RUN_ID_SAFE_PATTERN.fullmatch(run_id):
        raise RuntimeConfigurationError(
            "run id must be path-safe and may only contain letters, numbers, '.', '_' and '-'."
        )
    return run_id


def plan_smoke_cases(
    providers: Sequence[ProviderPlan],
    *,
    lifecycle_modes: tuple[str, ...],
    tool_policies: tuple[str, ...] = (),
) -> tuple[PlannedCase, ...]:
    cases: list[PlannedCase] = []
    for provider in providers:
        if provider.status is not LiveSmokeProviderSelectionStatus.RUNNABLE:
            continue
        for mode in lifecycle_modes:
            if tool_policies:
                for policy in tool_policies:
                    cases.append(
                        PlannedCase(
                            service=provider.service,
                            mode=mode,
                            policy=policy,
                            model=provider.model,
                            effort=provider.effort,
                        )
                    )
            else:
                cases.append(
                    PlannedCase(
                        service=provider.service,
                        mode=mode,
                        policy=None,
                        model=provider.model,
                        effort=provider.effort,
                    )
                )
    return tuple(cases)


def all_selected_provider_statuses_have_error(plans: Sequence[ProviderPlan]) -> bool:
    return all(
        plan.status is not LiveSmokeProviderSelectionStatus.RUNNABLE for plan in plans
    )


__all__ = [
    "LiveSmokeProviderSelectionStatus",
    "LiveSmokeCaseStatus",
    "LiveSmokeCaseResult",
    "ProviderPlan",
    "ProviderSelection",
    "PlannedCase",
    "DryRunPlannedCase",
    "DryRunPlan",
    "ProviderRuntimeConfiguration",
    "SUPPORTED_PROVIDERS",
    "LIVE_SMOKE_CLAUDE_MODEL_ENV",
    "LIVE_SMOKE_CLAUDE_EFFORT_ENV",
    "LIVE_SMOKE_CODEX_MODEL_ENV",
    "LIVE_SMOKE_CODEX_EFFORT_ENV",
    "LIVE_SMOKE_OPENCODE_MODEL_ENV",
    "LIVE_SMOKE_OPENCODE_EFFORT_ENV",
    "parse_provider_selection",
    "plan_selected_providers",
    "build_dry_run_plan",
    "list_supported_providers",
    "dry_run_plan_to_json",
    "plan_smoke_cases",
    "resolve_model_and_effort",
    "resolve_run_id",
    "all_selected_provider_statuses_have_error",
    "classify_live_smoke_case_result",
    "classify_live_smoke_preflight_case_result",
    "live_smoke_case_results_payload",
    "compute_live_smoke_exit_status",
]
