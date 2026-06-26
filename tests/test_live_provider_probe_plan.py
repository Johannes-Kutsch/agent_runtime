"""Deterministic tests for the Live Provider Probe planning module.

The Live Provider Probe itself is manual-debug-only operator tooling (ADR
0013). These tests never touch live providers or real credentials: they load
``scripts/live-probe/live_provider_probe_plan.py`` directly and exercise only
the pure planning surface (selection, defaults, the case matrix, and the
outcome-category mapping) with injected/faked auth state.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "live-probe"
    / "live_provider_probe_plan.py"
)


@pytest.fixture
def plan() -> Any:
    spec = importlib.util.spec_from_file_location(
        "live_provider_probe_plan", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # type: ignore[arg-type]
    spec.loader.exec_module(module)
    return module


def _runtime_outcome(kind_name: str, **result_kwargs: Any) -> Any:
    from agent_runtime.errors import ProviderUnavailableReason
    from agent_runtime import runtime as pr

    kinds: dict[
        str,
        pr.Completed
        | pr.UsageLimited
        | pr.ProviderUnavailable
        | pr.Cancelled
        | pr.TimedOut,
    ] = {
        "completed": pr.Completed(),
        "usage_limited": pr.UsageLimited(None),
        "no_service_available": pr.ProviderUnavailable(
            reason=ProviderUnavailableReason.SERVICE_NOT_AVAILABLE,
            detail="service unavailable",
        ),
        "timed_out": pr.TimedOut(),
        "retryable_provider_failure": pr.ProviderUnavailable(
            reason=ProviderUnavailableReason.TRANSIENT_API_ERROR,
            detail="transient api error",
        ),
        "cancelled": pr.Cancelled(),
    }
    result = pr.RunResult(
        output=result_kwargs.get("output", ""),
        usage=result_kwargs.get("usage"),
        continuation=result_kwargs.get("continuation"),
        selected=result_kwargs.get(
            "selected",
            pr.ResolvedProvider(service="claude", model="haiku", effort="low"),
        ),
    )
    return pr.RuntimeOutcome(kind=kinds[kind_name], result=result)


def test_env_example_contains_only_placeholder_keys() -> None:
    example = (
        Path(__file__).resolve().parents[1] / "scripts" / "live-probe" / ".env.example"
    )
    assert example.exists()

    values: dict[str, str] = {}
    for line in example.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        assert "=" in stripped
        key, value = (part.strip() for part in stripped.split("=", 1))
        values[key] = value

    assert set(values) == {"CLAUDE_CODE_OAUTH_TOKEN", "OPENCODE_GO_API_KEY"}
    assert values["CLAUDE_CODE_OAUTH_TOKEN"] == "<replace-with-claude-code-oauth-token>"
    assert values["OPENCODE_GO_API_KEY"] == "<replace-with-opencode-go-api-key>"


def test_provider_selection_accepts_supported_services_multiple_and_all(
    plan: Any,
) -> None:
    from agent_runtime.errors import RuntimeConfigurationError

    selection = plan.parse_provider_selection(("claude", "codex", "opencode"))
    assert selection.providers == ("claude", "codex", "opencode")
    assert selection.include_all is False

    comma = plan.parse_provider_selection("claude,opencode")
    assert comma.providers == ("claude", "opencode")

    all_selection = plan.parse_provider_selection("all")
    assert all_selection.include_all is True
    assert all_selection.providers == plan.SUPPORTED_PROVIDERS

    with pytest.raises(RuntimeConfigurationError):
        plan.parse_provider_selection("gibberish")
    with pytest.raises(RuntimeConfigurationError):
        plan.parse_provider_selection(("all", "claude"))


def test_explicit_unconfigured_surfaces_config_error_while_all_skips(plan: Any) -> None:
    explicit = plan.plan_selected_providers(
        plan.parse_provider_selection("claude"),
        env={},
    )
    assert explicit[0].status is plan.ProviderConfigStatus.CONFIG_ERROR
    assert explicit[0].reason == "missing CLAUDE_CODE_OAUTH_TOKEN"

    all_plans = plan.plan_selected_providers(
        plan.parse_provider_selection("all"),
        env={},
        codex_auth_present=False,
    )
    assert all(
        provider.status is plan.ProviderConfigStatus.SKIPPED for provider in all_plans
    )


def test_configured_providers_resolve_cost_first_defaults(plan: Any) -> None:
    planned = plan.plan_selected_providers(
        plan.parse_provider_selection(("claude", "codex", "opencode")),
        env={},
        claude_code_oauth_token="token",
        opencode_api_key="api-key",
        codex_auth_present=True,
    )
    by_service = {p.service: p for p in planned}
    assert all(p.status is plan.ProviderConfigStatus.RUNNABLE for p in planned)
    assert (by_service["claude"].model, by_service["claude"].effort) == ("haiku", "low")
    assert (by_service["codex"].model, by_service["codex"].effort) == (
        "gpt-5.4-mini",
        "low",
    )
    assert (by_service["opencode"].model, by_service["opencode"].effort) == (
        "deepseek-v4-flash",
        "medium",
    )


def test_cli_overrides_take_precedence_over_defaults(plan: Any) -> None:
    planned = plan.plan_selected_providers(
        plan.parse_provider_selection("claude"),
        model_overrides={"claude": "sonnet"},
        effort_overrides={"claude": "high"},
        env={},
        claude_code_oauth_token="token",
    )
    assert planned[0].model == "sonnet"
    assert planned[0].effort == "high"
    assert planned[0].status is plan.ProviderConfigStatus.RUNNABLE


def test_provider_plan_exposes_public_provider_selection_with_auth(plan: Any) -> None:
    from agent_runtime import runtime as pr

    planned = plan.plan_selected_providers(
        plan.parse_provider_selection(("claude", "opencode")),
        env={},
        claude_code_oauth_token="claude-token",
        opencode_api_key="opencode-key",
    )
    by_service = {p.service: p for p in planned}
    assert by_service["claude"].provider_selection == pr.ProviderSelection(
        service="claude",
        model="haiku",
        effort="low",
        auth=pr.ProviderAuth(claude_code_oauth_token="claude-token"),
    )
    assert by_service["opencode"].provider_selection == pr.ProviderSelection(
        service="opencode",
        model="deepseek-v4-flash",
        effort="medium",
        auth=pr.ProviderAuth(opencode_api_key="opencode-key"),
    )


def test_codex_auth_injection_controls_runnable_status(plan: Any) -> None:
    selection = plan.parse_provider_selection("codex")
    runnable = plan.plan_selected_providers(selection, env={}, codex_auth_present=True)
    missing = plan.plan_selected_providers(selection, env={}, codex_auth_present=False)
    assert runnable[0].status is plan.ProviderConfigStatus.RUNNABLE
    assert missing[0].status is plan.ProviderConfigStatus.CONFIG_ERROR
    assert missing[0].reason == "provider not configured"


def test_probe_case_matrix_is_five_coupled_cases(plan: Any) -> None:
    provider_plan = plan.plan_selected_providers(
        plan.parse_provider_selection("claude"),
        env={},
        claude_code_oauth_token="token",
    )[0]
    cases = plan.probe_cases_for_provider(provider_plan)

    labels = [case.label for case in cases]
    assert labels == [
        "ephemeral_UNRESTRICTED",
        "new_session_UNRESTRICTED",
        "resumed_session_UNRESTRICTED",
        "ephemeral_NONE",
        "ephemeral_NO_FILE_MUTATION",
    ]
    # No duplicate ephemeral_UNRESTRICTED.
    assert len(labels) == len(set(labels))
    # resumed_session always follows new_session.
    assert labels.index("resumed_session_UNRESTRICTED") == (
        labels.index("new_session_UNRESTRICTED") + 1
    )
    assert all(case.service == "claude" for case in cases)
    assert all(
        case.provider_selection is provider_plan.provider_selection for case in cases
    )


def test_outcome_category_maps_runtime_kinds(plan: Any) -> None:
    assert plan.outcome_category(_runtime_outcome("completed")) == "success"
    assert plan.outcome_category(_runtime_outcome("usage_limited")) == "usage_limited"
    assert (
        plan.outcome_category(_runtime_outcome("no_service_available"))
        == "no_service_available"
    )
    assert plan.outcome_category(_runtime_outcome("timed_out")) == "timed_out"
    assert (
        plan.outcome_category(_runtime_outcome("retryable_provider_failure"))
        == "retryable_failure"
    )
    assert plan.outcome_category(_runtime_outcome("cancelled")) == "cancelled"
    assert plan.SUCCESS_CATEGORY == "success"


def test_outcome_category_falls_back_to_error_for_unknown_kind(plan: Any) -> None:
    from types import SimpleNamespace

    assert plan.outcome_category(SimpleNamespace(kind=object())) == "error"
    assert plan.outcome_category(SimpleNamespace(kind=None)) == "error"


def test_live_probe_defaults_are_runtime_supported_and_dated(plan: Any) -> None:
    from agent_runtime import _builtin_runtime_client as rc

    assert plan.LIVE_PROBE_DEFAULTS_VERIFIED_ON == "2026-06-20"
    assert plan.LIVE_PROBE_DEFAULTS["claude"][0] in rc._CLAUDE_VALID_MODELS
    assert plan.LIVE_PROBE_DEFAULTS["claude"][1] in rc._CLAUDE_VALID_EFFORTS
    assert plan.LIVE_PROBE_DEFAULTS["codex"][0] in rc._CODEX_VALID_MODELS
    assert plan.LIVE_PROBE_DEFAULTS["codex"][1] in rc._CODEX_VALID_EFFORTS
    assert plan.LIVE_PROBE_DEFAULTS["opencode"][0] in rc._OPENCODE_GO_MODELS
    assert plan.LIVE_PROBE_DEFAULTS["opencode"][1] in rc._OPENCODE_VALID_EFFORTS
