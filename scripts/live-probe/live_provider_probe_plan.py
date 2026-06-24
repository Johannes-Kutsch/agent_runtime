"""Planning for the Live Provider Probe (manual-debug-only tooling).

This module resolves which providers can be probed and what cases to run.
It is deliberately small: classification of run quality lives in the
deterministic pytest suite, not here. The probe only proves a real provider
invocation reaches a classified runtime outcome without an unexpected
exception, so the only "verdict" this module produces is the runtime's own
outcome category. See ADR 0013.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from agent_runtime.errors import ProviderUnavailableReason, RuntimeConfigurationError
from agent_runtime.runtime import (
    ProviderAuth,
    ProviderSelection as RuntimeProviderSelection,
    ToolPolicy,
)


SUPPORTED_PROVIDERS: tuple[str, ...] = ("claude", "codex", "opencode")

# Live Probe Defaults: cheapest runtime-supported provider/model/effort tuples.
# Overridable per-provider on the CLI; no shell-environment configuration.
LIVE_PROBE_DEFAULTS_VERIFIED_ON = "2026-06-20"
LIVE_PROBE_DEFAULTS: dict[str, tuple[str, str]] = {
    "claude": ("haiku", "low"),
    "codex": ("gpt-5.4-mini", "low"),
    "opencode": ("deepseek-v4-flash", "medium"),
}

_PROVIDER_CLAUDE_TOKEN_ENV = "CLAUDE_CODE_OAUTH_TOKEN"
_PROVIDER_OPENCODE_ENV = "OPENCODE_GO_API_KEY"
_PROVIDER_CODEX_HOME_AUTH_PATH = Path.home() / ".codex" / "auth.json"


class ProviderConfigStatus(str, Enum):
    """Whether a selected provider can be probed."""

    RUNNABLE = "runnable"
    # all-configured selection, provider unconfigured: skip silently, no wipe.
    SKIPPED = "skipped"
    # explicitly named, provider unconfigured: surface (red), no wipe.
    CONFIG_ERROR = "config_error"


# Runtime outcome kind class name -> probe verdict category. Anything not in
# this map (and any unexpected exception) is "error". Only "success" is a
# completed run; every other category prints as "run not completed".
SUCCESS_CATEGORY = "success"
_OUTCOME_CATEGORY_BY_KIND: dict[str, str] = {
    "Completed": SUCCESS_CATEGORY,
    "UsageLimited": "usage_limited",
    "TimedOut": "timed_out",
    "Cancelled": "cancelled",
}


def outcome_category(runtime_outcome: Any) -> str:
    """Map a ``RuntimeOutcome`` to its probe verdict category."""

    kind = getattr(runtime_outcome, "kind", None)
    if type(kind).__name__ == "ProviderUnavailable":
        reason = getattr(kind, "reason", None)
        if reason is ProviderUnavailableReason.SERVICE_NOT_AVAILABLE:
            return "no_service_available"
        if reason is ProviderUnavailableReason.TRANSIENT_API_ERROR:
            return "retryable_failure"
    return _OUTCOME_CATEGORY_BY_KIND.get(type(kind).__name__, "error")


@dataclass(frozen=True)
class ProviderSelection:
    providers: tuple[str, ...]
    include_all: bool


@dataclass(frozen=True)
class ProviderPlan:
    service: str
    model: str
    effort: str
    provider_selection: RuntimeProviderSelection
    status: ProviderConfigStatus
    reason: str | None = None


@dataclass(frozen=True)
class ProbeCase:
    service: str
    mode: str  # "ephemeral" | "new_session" | "resumed_session"
    tool_policy: str  # ToolPolicy member name
    model: str
    effort: str
    provider_selection: RuntimeProviderSelection

    @property
    def label(self) -> str:
        return f"{self.mode}_{self.tool_policy}"


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
    nonempty = tuple(part for part in selected if part)
    if not nonempty:
        raise RuntimeConfigurationError("provider selection is empty")
    return nonempty


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
) -> tuple[str, str]:
    if provider not in SUPPORTED_PROVIDERS:
        raise RuntimeConfigurationError(f"Unsupported provider name: {provider!r}")
    model = cli_model or LIVE_PROBE_DEFAULTS[provider][0]
    effort = cli_effort or LIVE_PROBE_DEFAULTS[provider][1]
    return model, effort


def detect_codex_auth_present() -> bool:
    return _PROVIDER_CODEX_HOME_AUTH_PATH.exists()


def _resolve_env_map(env: Mapping[str, str] | None) -> Mapping[str, str]:
    return {} if env is None else env


def _resolve_explicit_or_env_value(
    *, explicit_value: str | None, env_value: str
) -> str:
    if explicit_value is not None:
        return explicit_value
    return env_value


def _provider_config_error_reason(
    provider: str,
    *,
    env: Mapping[str, str] | None,
    claude_code_oauth_token: str | None,
    opencode_api_key: str | None,
    codex_auth_present: bool | None,
) -> str | None:
    env_map = _resolve_env_map(env)
    if provider == "claude":
        token = _resolve_explicit_or_env_value(
            explicit_value=claude_code_oauth_token,
            env_value=env_map.get(_PROVIDER_CLAUDE_TOKEN_ENV, ""),
        )
        if token.strip():
            return None
        return f"missing {_PROVIDER_CLAUDE_TOKEN_ENV}"
    if provider == "opencode":
        api_key = _resolve_explicit_or_env_value(
            explicit_value=opencode_api_key,
            env_value=env_map.get(_PROVIDER_OPENCODE_ENV, ""),
        )
        if api_key.strip():
            return None
        return f"missing {_PROVIDER_OPENCODE_ENV}"
    if provider == "codex":
        present = (
            codex_auth_present
            if codex_auth_present is not None
            else detect_codex_auth_present()
        )
        if present:
            return None
        return "provider not configured"
    raise RuntimeConfigurationError(f"Unsupported provider name: {provider!r}")


def _provider_has_runtime_config(
    provider: str,
    *,
    env: Mapping[str, str] | None = None,
    claude_code_oauth_token: str | None = None,
    opencode_api_key: str | None = None,
    codex_auth_present: bool | None = None,
) -> bool:
    return (
        _provider_config_error_reason(
            provider,
            env=env,
            claude_code_oauth_token=claude_code_oauth_token,
            opencode_api_key=opencode_api_key,
            codex_auth_present=codex_auth_present,
        )
        is None
    )


def _resolve_provider_auth(
    provider: str,
    *,
    env: Mapping[str, str] | None,
    claude_code_oauth_token: str | None,
    opencode_api_key: str | None,
) -> ProviderAuth:
    env_map = _resolve_env_map(env)
    if provider == "claude":
        token = claude_code_oauth_token
        if token is None:
            token = env_map.get(_PROVIDER_CLAUDE_TOKEN_ENV)
        return ProviderAuth(claude_code_oauth_token=token)
    if provider == "opencode":
        api_key = opencode_api_key
        if api_key is None:
            api_key = env_map.get(_PROVIDER_OPENCODE_ENV)
        return ProviderAuth(opencode_api_key=api_key)
    return ProviderAuth()


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
    model_by_provider = dict(model_overrides or {})
    effort_by_provider = dict(effort_overrides or {})
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
    )
    auth = _resolve_provider_auth(
        provider,
        env=env,
        claude_code_oauth_token=claude_code_oauth_token,
        opencode_api_key=opencode_api_key,
    )
    config_error = _provider_config_error_reason(
        provider,
        env=env,
        claude_code_oauth_token=claude_code_oauth_token,
        opencode_api_key=opencode_api_key,
        codex_auth_present=codex_auth_present,
    )
    if config_error is not None:
        status = (
            ProviderConfigStatus.SKIPPED
            if include_all
            else ProviderConfigStatus.CONFIG_ERROR
        )
        reason: str | None = config_error
    else:
        status = ProviderConfigStatus.RUNNABLE
        reason = None
    return ProviderPlan(
        service=provider,
        model=resolved_model,
        effort=resolved_effort,
        provider_selection=RuntimeProviderSelection(
            service=provider,
            model=resolved_model,
            effort=resolved_effort,
            auth=auth,
        ),
        status=status,
        reason=reason,
    )


def probe_cases_for_provider(provider_plan: ProviderPlan) -> tuple[ProbeCase, ...]:
    """The six-case Live Probe Case Matrix for one provider.

    Three entry paths at ``UNRESTRICTED`` followed by ephemeral under each
    remaining ``ToolPolicy``. ``new_session`` precedes ``resumed_session`` so
    the resume case can reuse the new session's continuation.
    """

    def _case(mode: str, policy: str) -> ProbeCase:
        return ProbeCase(
            service=provider_plan.service,
            mode=mode,
            tool_policy=policy,
            model=provider_plan.model,
            effort=provider_plan.effort,
            provider_selection=provider_plan.provider_selection,
        )

    cases: list[ProbeCase] = [
        _case("ephemeral", ToolPolicy.UNRESTRICTED.name),
        _case("new_session", ToolPolicy.UNRESTRICTED.name),
        _case("resumed_session", ToolPolicy.UNRESTRICTED.name),
    ]
    for policy in ToolPolicy:
        if policy is ToolPolicy.UNRESTRICTED:
            continue
        cases.append(_case("ephemeral", policy.name))
    return tuple(cases)


@dataclass(frozen=True)
class ProviderRuntimeConfiguration:
    service: str
    configured: bool


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


__all__ = [
    "SUPPORTED_PROVIDERS",
    "LIVE_PROBE_DEFAULTS",
    "LIVE_PROBE_DEFAULTS_VERIFIED_ON",
    "SUCCESS_CATEGORY",
    "ProviderConfigStatus",
    "ProviderSelection",
    "ProviderPlan",
    "ProbeCase",
    "ProviderRuntimeConfiguration",
    "outcome_category",
    "parse_provider_selection",
    "resolve_model_and_effort",
    "detect_codex_auth_present",
    "plan_selected_providers",
    "probe_cases_for_provider",
    "list_supported_providers",
]
