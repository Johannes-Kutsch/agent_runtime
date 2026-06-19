from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping, Sequence
from uuid import uuid4

from agent_runtime.errors import RuntimeConfigurationError


class LiveSmokeProviderSelectionStatus(str, Enum):
    RUNNABLE = "runnable"
    SKIPPED = "skipped"
    CONFIG_ERROR = "config_error"


SUPPORTED_PROVIDERS: tuple[str, ...] = ("claude", "codex", "opencode")
LIVE_SMOKE_CLAUDE_MODEL_ENV = "LIVE_SMOKE_CLAUDE_MODEL"
LIVE_SMOKE_CLAUDE_EFFORT_ENV = "LIVE_SMOKE_CLAUDE_EFFORT"
LIVE_SMOKE_CODEX_MODEL_ENV = "LIVE_SMOKE_CODEX_MODEL"
LIVE_SMOKE_CODEX_EFFORT_ENV = "LIVE_SMOKE_CODEX_EFFORT"
LIVE_SMOKE_OPENCODE_MODEL_ENV = "LIVE_SMOKE_OPENCODE_MODEL"
LIVE_SMOKE_OPENCODE_EFFORT_ENV = "LIVE_SMOKE_OPENCODE_EFFORT"

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
]
