from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "live_provider_smoke_plan.py"
)


@pytest.fixture
def planning_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "live_provider_smoke_plan",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # type: ignore[arg-type]
    spec.loader.exec_module(module)
    return module


def test_live_smoke_provider_selection_accepts_supported_services_multiple_and_all(
    planning_module: Any,
) -> None:
    module = planning_module

    from agent_runtime.errors import RuntimeConfigurationError

    selection = module.parse_provider_selection(("claude", "codex", "opencode"))
    assert selection.providers == ("claude", "codex", "opencode")
    assert selection.include_all is False

    comma_selection = module.parse_provider_selection("claude,opencode")
    assert comma_selection.providers == ("claude", "opencode")
    assert comma_selection.include_all is False

    all_selection = module.parse_provider_selection("all")
    assert all_selection.include_all is True
    assert all_selection.providers == module.SUPPORTED_PROVIDERS

    with pytest.raises(RuntimeConfigurationError):
        module.parse_provider_selection("gibberish")


def test_live_smoke_explicit_provider_missing_config_reports_config_error_and_all_skips_unconfigured(
    planning_module: Any,
) -> None:
    module = planning_module

    explicit = module.parse_provider_selection("claude")
    explicit_plans = module.plan_selected_providers(
        explicit,
        model_overrides={"claude": "sonnet"},
        effort_overrides={"claude": "medium"},
    )
    assert (
        explicit_plans[0].status is module.LiveSmokeProviderSelectionStatus.CONFIG_ERROR
    )

    all_selection = module.parse_provider_selection("all")
    all_plans = module.plan_selected_providers(
        all_selection,
        model_overrides={
            "claude": "sonnet",
            "codex": "codex-mini",
            "opencode": "deepseek-v4-flash",
        },
        effort_overrides={
            "claude": "medium",
            "codex": "high",
            "opencode": "medium",
        },
    )
    assert all(
        plan.status is module.LiveSmokeProviderSelectionStatus.SKIPPED
        for plan in all_plans
    )
    assert module.all_selected_provider_statuses_have_error(all_plans)


def test_live_smoke_model_and_effort_resolve_from_cli_and_env_without_defaults(
    planning_module: Any,
) -> None:
    module = planning_module

    env = {
        module.LIVE_SMOKE_CLAUDE_MODEL_ENV: "env-claude-model",
        module.LIVE_SMOKE_CLAUDE_EFFORT_ENV: "high",
    }
    model, effort = module.resolve_model_and_effort(
        "claude",
        cli_model="cli-claude-model",
        cli_effort="low",
        env=env,
    )
    assert model == "cli-claude-model"
    assert effort == "low"

    parsed = module.parse_provider_selection("claude")
    without_defaults = module.plan_selected_providers(
        parsed,
        claude_code_oauth_token="token",
        env={},
    )
    assert without_defaults[0].model == ""
    assert without_defaults[0].effort == ""
    assert (
        without_defaults[0].status
        is module.LiveSmokeProviderSelectionStatus.CONFIG_ERROR
    )


def test_live_smoke_run_id_is_generated_when_missing_and_path_safe_when_supplied(
    planning_module: Any,
) -> None:
    module = planning_module

    generated_default = module.resolve_run_id()
    generated_empty = module.resolve_run_id("")
    explicit = module.resolve_run_id("smoke-run_2026.06.19")

    assert generated_default and isinstance(generated_default, str)
    assert generated_empty and isinstance(generated_empty, str)
    assert explicit == "smoke-run_2026.06.19"
    assert generated_default != generated_empty

    from agent_runtime.errors import RuntimeConfigurationError

    with pytest.raises(RuntimeConfigurationError):
        module.resolve_run_id("../unsafe-run-id")
    with pytest.raises(RuntimeConfigurationError):
        module.resolve_run_id("unsafe/run-id")


def test_live_smoke_planning_emits_provider_mode_policy_cases_without_side_effects(
    planning_module: Any,
) -> None:
    module = planning_module

    providers = module.plan_selected_providers(
        module.parse_provider_selection(("claude", "opencode")),
        model_overrides={"claude": "sonnet", "opencode": "deepseek-v4-flash"},
        effort_overrides={"claude": "medium", "opencode": "medium"},
        claude_code_oauth_token="token",
        opencode_api_key="api-key",
    )
    cases = module.plan_smoke_cases(
        providers,
        lifecycle_modes=("ephemeral", "new_session"),
        tool_policies=("NONE", "UNRESTRICTED"),
    )

    assert len(cases) == 8
    assert all(isinstance(case, module.PlannedCase) for case in cases)
    first = cases[0]
    assert first.service == "claude"
    assert first.mode in {"ephemeral", "new_session"}
    assert first.policy in {"NONE", "UNRESTRICTED"}
    assert first.model == "sonnet"
    assert first.effort == "medium"
    assert any(case.service == "opencode" for case in cases)
