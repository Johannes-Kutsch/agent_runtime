from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, cast

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


def test_live_smoke_env_example_file_contains_only_placeholder_keys() -> None:
    example = (
        Path(__file__).resolve().parents[1] / "scripts" / "live-smoke" / ".env.example"
    )

    assert example.exists()

    values = {}
    for line in example.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        assert "=" in stripped
        key, value = (part.strip() for part in stripped.split("=", 1))
        values[key] = value

    assert set(values) == {
        "CLAUDE_CODE_OAUTH_TOKEN",
        "OPENCODE_GO_API_KEY",
    }
    assert values["CLAUDE_CODE_OAUTH_TOKEN"] == "<replace-with-claude-code-oauth-token>"
    assert values["OPENCODE_GO_API_KEY"] == "<replace-with-opencode-go-api-key>"


def _planned_case(
    module: Any,
    *,
    service: str,
    mode: str,
    policy: str | None,
    model: str,
    effort: str,
    auth: Any | None = None,
) -> Any:
    from agent_runtime import runtime as prompt_runtime

    return module.PlannedCase(
        service=service,
        mode=mode,
        policy=policy,
        model=model,
        effort=effort,
        provider_selection=prompt_runtime.ProviderSelection(
            service=service,
            model=model,
            effort=effort,
            auth=auth or prompt_runtime.ProviderAuth(),
        ),
    )


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
    assert without_defaults[0].model == "haiku"
    assert without_defaults[0].effort == "low"
    assert (
        without_defaults[0].status is module.LiveSmokeProviderSelectionStatus.RUNNABLE
    )


def test_live_smoke_cli_model_and_effort_overrides_always_apply(
    planning_module: Any,
) -> None:
    module = planning_module

    cases = (
        (
            "claude",
            {"model": "cli-claude-model", "effort": "low"},
            {"claude_code_oauth_token": "token"},
            {
                module.LIVE_SMOKE_CLAUDE_MODEL_ENV: "env-claude-model",
                module.LIVE_SMOKE_CLAUDE_EFFORT_ENV: "high",
            },
        ),
        (
            "codex",
            {"model": "codex-mini", "effort": "high"},
            {"codex_auth_present": True},
            {
                module.LIVE_SMOKE_CODEX_MODEL_ENV: "env-codex-model",
                module.LIVE_SMOKE_CODEX_EFFORT_ENV: "low",
            },
        ),
        (
            "opencode",
            {"model": "deepseek-opencode", "effort": "low"},
            {"opencode_api_key": "api-key"},
            {
                module.LIVE_SMOKE_OPENCODE_MODEL_ENV: "env-opencode-model",
                module.LIVE_SMOKE_OPENCODE_EFFORT_ENV: "medium",
            },
        ),
    )

    for service, overrides, auth_kwargs, env in cases:
        parsed = module.parse_provider_selection(service)
        planned = module.plan_selected_providers(
            parsed,
            model_overrides={service: overrides["model"]},
            effort_overrides={service: overrides["effort"]},
            env=env,
            **auth_kwargs,
        )
        assert planned[0].status is module.LiveSmokeProviderSelectionStatus.RUNNABLE
        assert planned[0].model == overrides["model"]
        assert planned[0].effort == overrides["effort"]


@pytest.mark.parametrize(
    ("service", "auth_kwargs", "expected_model", "expected_effort"),
    (
        ("claude", {"claude_code_oauth_token": "token"}, "haiku", "low"),
        ("codex", {"codex_auth_present": True}, "gpt-5.4-mini", "low"),
        (
            "opencode",
            {"opencode_api_key": "opencode-key"},
            "deepseek-v4-flash",
            "medium",
        ),
    ),
)
def test_live_smoke_credentialed_provider_without_overrides_resolves_defaults(
    planning_module: Any,
    service: str,
    auth_kwargs: dict[str, object],
    expected_model: str,
    expected_effort: str,
) -> None:
    module = planning_module

    parsed = module.parse_provider_selection(service)
    planned = module.plan_selected_providers(
        parsed,
        env={},
        **auth_kwargs,
    )

    assert planned[0].status is module.LiveSmokeProviderSelectionStatus.RUNNABLE
    assert planned[0].model == expected_model
    assert planned[0].effort == expected_effort


def test_live_smoke_codex_planning_requires_injected_auth_state(
    planning_module: Any,
    tmp_path: Path,
) -> None:
    module = planning_module

    auth_file = tmp_path / "auth.json"
    auth_file.write_text("{}", encoding="utf-8")
    selected = module.parse_provider_selection("codex")

    original_auth_path = module._PROVIDER_CODEX_HOME_AUTH_PATH
    module._PROVIDER_CODEX_HOME_AUTH_PATH = auth_file
    try:
        default_planned = module.plan_selected_providers(
            selected,
            env={},
        )
        injected_planned = module.plan_selected_providers(
            selected,
            env={},
            codex_auth_present=True,
        )
        missing_planned = module.plan_selected_providers(
            selected,
            env={},
            codex_auth_present=False,
        )
    finally:
        module._PROVIDER_CODEX_HOME_AUTH_PATH = original_auth_path

    assert (
        default_planned[0].status
        is module.LiveSmokeProviderSelectionStatus.CONFIG_ERROR
    )
    assert default_planned[0].reason == "provider not configured"
    assert (
        injected_planned[0].status is module.LiveSmokeProviderSelectionStatus.RUNNABLE
    )
    assert (
        missing_planned[0].status
        is module.LiveSmokeProviderSelectionStatus.CONFIG_ERROR
    )


def test_live_smoke_provider_listing_requires_injected_codex_auth_state(
    planning_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = planning_module

    auth_file = tmp_path / "auth.json"
    auth_file.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(module, "_PROVIDER_CODEX_HOME_AUTH_PATH", auth_file)

    default_listing = module.list_supported_providers(env={})
    injected_listing = module.list_supported_providers(
        env={},
        codex_auth_present=True,
    )

    default_codex = next(
        provider for provider in default_listing if provider.service == "codex"
    )
    injected_codex = next(
        provider for provider in injected_listing if provider.service == "codex"
    )

    assert default_codex.configured is False
    assert injected_codex.configured is True


def test_live_smoke_detect_codex_auth_present_reads_auth_path(
    planning_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = planning_module

    auth_file = tmp_path / "auth.json"
    auth_file.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(module, "_PROVIDER_CODEX_HOME_AUTH_PATH", auth_file)

    assert module.detect_codex_auth_present() is True


def test_live_smoke_explicit_provider_missing_opencode_credentials_is_not_runnable(
    planning_module: Any,
) -> None:
    module = planning_module

    parsed = module.parse_provider_selection("opencode")
    planned = module.plan_selected_providers(
        parsed,
        env={"OPENCODE_GO_API_KEY": "   "},
    )

    assert planned[0].status is module.LiveSmokeProviderSelectionStatus.CONFIG_ERROR
    assert planned[0].reason == "missing OPENCODE_GO_API_KEY"


@pytest.mark.parametrize(
    ("provider", "env_key"),
    (
        ("claude", "CLAUDE_CODE_OAUTH_TOKEN"),
        ("opencode", "OPENCODE_GO_API_KEY"),
    ),
)
def test_live_smoke_missing_whitespace_credentials_name_missing_key(
    planning_module: Any,
    provider: str,
    env_key: str,
) -> None:
    module = planning_module

    parsed = module.parse_provider_selection(provider)
    planned = module.plan_selected_providers(
        parsed,
        env={env_key: "  \t  "},
    )

    assert planned[0].status is module.LiveSmokeProviderSelectionStatus.CONFIG_ERROR
    assert planned[0].reason == f"missing {env_key}"


@pytest.mark.parametrize(
    ("provider", "env_key", "auth_kwargs"),
    (
        (
            "claude",
            "CLAUDE_CODE_OAUTH_TOKEN",
            {"claude_code_oauth_token": "   "},
        ),
        (
            "opencode",
            "OPENCODE_GO_API_KEY",
            {"opencode_api_key": ""},
        ),
    ),
)
def test_live_smoke_explicit_blank_credential_does_not_fall_back_to_env(
    planning_module: Any,
    provider: str,
    env_key: str,
    auth_kwargs: dict[str, str],
) -> None:
    module = planning_module

    parsed = module.parse_provider_selection(provider)
    planned = module.plan_selected_providers(
        parsed,
        env={env_key: "credential-from-env"},
        **auth_kwargs,
    )

    assert planned[0].status is module.LiveSmokeProviderSelectionStatus.CONFIG_ERROR
    assert planned[0].reason == f"missing {env_key}"


def test_live_smoke_explicit_provider_missing_claude_credentials_is_not_runnable(
    planning_module: Any,
) -> None:
    module = planning_module

    parsed = module.parse_provider_selection("claude")
    planned = module.plan_selected_providers(
        parsed,
        env={"CLAUDE_CODE_OAUTH_TOKEN": "   "},
    )

    assert planned[0].status is module.LiveSmokeProviderSelectionStatus.CONFIG_ERROR
    assert planned[0].reason == "missing CLAUDE_CODE_OAUTH_TOKEN"


def test_live_smoke_all_selection_skips_whitespace_only_claude_credentials(
    planning_module: Any,
) -> None:
    module = planning_module

    parsed = module.parse_provider_selection("all")
    planned = module.plan_selected_providers(
        parsed,
        env={"CLAUDE_CODE_OAUTH_TOKEN": "   "},
    )

    assert all(
        provider.status is module.LiveSmokeProviderSelectionStatus.SKIPPED
        for provider in planned
    )
    assert module.all_selected_provider_statuses_have_error(planned)


def test_live_smoke_model_and_effort_fill_missing_field_from_live_smoke_defaults(
    planning_module: Any,
) -> None:
    module = planning_module

    model_from_cli, effort_from_default = module.resolve_model_and_effort(
        "codex",
        cli_model="gpt-5.4",
        cli_effort=None,
        env={},
    )
    model_from_default, effort_from_env = module.resolve_model_and_effort(
        "opencode",
        cli_model=None,
        cli_effort=None,
        env={module.LIVE_SMOKE_OPENCODE_EFFORT_ENV: "high"},
    )

    assert model_from_cli == "gpt-5.4"
    assert effort_from_default == "low"
    assert model_from_default == "deepseek-v4-flash"
    assert effort_from_env == "medium"


def test_live_smoke_planning_uses_explicit_env_mapping_for_resolution_and_config(
    planning_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = planning_module

    monkeypatch.setenv(module.LIVE_SMOKE_CLAUDE_MODEL_ENV, "outside-model")
    monkeypatch.setenv(module.LIVE_SMOKE_CLAUDE_EFFORT_ENV, "outside-effort")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "outside-token")

    selected = module.parse_provider_selection("claude")

    isolated = module.plan_selected_providers(
        selected,
        env={},
    )
    assert isolated[0].model == "haiku"
    assert isolated[0].effort == "low"
    assert isolated[0].status is module.LiveSmokeProviderSelectionStatus.CONFIG_ERROR

    planned = module.plan_selected_providers(
        selected,
        env={
            module.LIVE_SMOKE_CLAUDE_MODEL_ENV: "env-model",
            module.LIVE_SMOKE_CLAUDE_EFFORT_ENV: "env-effort",
            "CLAUDE_CODE_OAUTH_TOKEN": "env-token",
        },
    )
    assert planned[0].model == "haiku"
    assert planned[0].effort == "low"
    assert planned[0].status is module.LiveSmokeProviderSelectionStatus.RUNNABLE


def test_live_smoke_model_and_effort_ignore_shell_environment_when_no_cli_overrides(
    planning_module: Any,
) -> None:
    module = planning_module

    parsed = module.parse_provider_selection(("claude", "codex", "opencode"))
    planned = module.plan_selected_providers(
        parsed,
        env={
            module.LIVE_SMOKE_CLAUDE_MODEL_ENV: "env-claude-model",
            module.LIVE_SMOKE_CLAUDE_EFFORT_ENV: "high",
            module.LIVE_SMOKE_CODEX_MODEL_ENV: "env-codex-model",
            module.LIVE_SMOKE_CODEX_EFFORT_ENV: "medium",
            module.LIVE_SMOKE_OPENCODE_MODEL_ENV: "env-opencode-model",
            module.LIVE_SMOKE_OPENCODE_EFFORT_ENV: "high",
            "CLAUDE_CODE_OAUTH_TOKEN": "env-claude-token",
            module._PROVIDER_OPENCODE_ENV: "env-opencode-key",
        },
        claude_code_oauth_token="token",
        opencode_api_key="api-key",
        codex_auth_present=True,
    )

    expected_defaults = module.LIVE_SMOKE_DEFAULTS
    assert len(planned) == 3
    assert planned[0].model == expected_defaults["claude"][0]
    assert planned[0].effort == expected_defaults["claude"][1]
    assert planned[1].model == expected_defaults["codex"][0]
    assert planned[1].effort == expected_defaults["codex"][1]
    assert planned[2].model == expected_defaults["opencode"][0]
    assert planned[2].effort == expected_defaults["opencode"][1]
    assert all(
        plan.status is module.LiveSmokeProviderSelectionStatus.RUNNABLE
        for plan in planned
    )


def test_live_smoke_dry_run_defaults_propagate_to_provider_plans_and_cases(
    planning_module: Any,
    tmp_path: Path,
) -> None:
    module = planning_module

    summary = module.build_dry_run_plan(
        provider_selection=("claude", "codex", "opencode"),
        lifecycle_modes=("ephemeral",),
        run_id="defaults-echo",
        claude_code_oauth_token="token",
        opencode_api_key="opencode-key",
        codex_auth_present=True,
        artifact_root=tmp_path / "live-smoke-artifacts",
    )

    expected = {
        "claude": ("haiku", "low"),
        "codex": ("gpt-5.4-mini", "low"),
        "opencode": ("deepseek-v4-flash", "medium"),
    }
    plans_by_service = {plan.service: plan for plan in summary.provider_plans}
    cases_by_service = {case.service: case for case in summary.cases}

    assert set(plans_by_service) == set(expected)
    for service, (model, effort) in expected.items():
        provider_plan = plans_by_service[service]
        assert provider_plan.status is module.LiveSmokeProviderSelectionStatus.RUNNABLE
        assert provider_plan.model == model
        assert provider_plan.effort == effort
        case = cases_by_service[service]
        assert case.model == provider_plan.model
        assert case.effort == provider_plan.effort


def test_live_smoke_full_matrix_for_single_provider_matches_all_provider_subset(
    planning_module: Any,
) -> None:
    module = planning_module
    from agent_runtime import runtime as prompt_runtime

    full_tool_policies = tuple(policy.name for policy in prompt_runtime.ToolPolicy)
    lifecycle_modes = ("ephemeral", "new_session", "resumed_session")

    explicit = module.build_dry_run_plan(
        provider_selection="claude",
        lifecycle_modes=lifecycle_modes,
        tool_policies=full_tool_policies,
        run_id="claude-full",
        claude_code_oauth_token="token",
        opencode_api_key="opencode-key",
        codex_auth_present=True,
    )

    all_configured = module.build_dry_run_plan(
        provider_selection="all",
        lifecycle_modes=lifecycle_modes,
        tool_policies=full_tool_policies,
        run_id="all-full",
        claude_code_oauth_token="token",
        opencode_api_key="opencode-key",
        codex_auth_present=True,
    )

    explicit_cases = explicit.cases
    all_claude_cases = tuple(
        case for case in all_configured.cases if case.service == "claude"
    )
    assert len(explicit_cases) == len(all_claude_cases)
    assert explicit.provider_plans == all_configured.provider_plans[:1]
    assert all(
        explicit_case.service == "claude"
        and all_case.service == "claude"
        and explicit_case.mode == all_case.mode
        and explicit_case.policy == all_case.policy
        and explicit_case.model == all_case.model
        and explicit_case.effort == all_case.effort
        for explicit_case, all_case in zip(explicit_cases, all_claude_cases)
    )


def test_live_smoke_provider_plan_exposes_public_provider_selection_with_auth(
    planning_module: Any,
) -> None:
    module = planning_module
    from agent_runtime import runtime as prompt_runtime

    selected = module.parse_provider_selection(("claude", "opencode"))
    planned = module.plan_selected_providers(
        selected,
        env={},
        claude_code_oauth_token="claude-token",
        opencode_api_key="opencode-key",
    )

    claude_plan = next(plan for plan in planned if plan.service == "claude")
    opencode_plan = next(plan for plan in planned if plan.service == "opencode")

    assert claude_plan.provider_selection == prompt_runtime.ProviderSelection(
        service="claude",
        model="haiku",
        effort="low",
        auth=prompt_runtime.ProviderAuth(claude_code_oauth_token="claude-token"),
    )
    assert opencode_plan.provider_selection == prompt_runtime.ProviderSelection(
        service="opencode",
        model="deepseek-v4-flash",
        effort="medium",
        auth=prompt_runtime.ProviderAuth(opencode_api_key="opencode-key"),
    )


def test_live_smoke_dry_run_cases_reuse_planned_public_provider_selection(
    planning_module: Any,
    tmp_path: Path,
) -> None:
    module = planning_module

    summary = module.build_dry_run_plan(
        provider_selection=("claude",),
        lifecycle_modes=("ephemeral", "new_session"),
        tool_policies=("NONE",),
        run_id="planned-selection-flow",
        claude_code_oauth_token="claude-token",
        artifact_root=tmp_path / "live-smoke-artifacts",
    )

    planned_selection = summary.provider_plans[0].provider_selection

    assert all(case.provider_selection == planned_selection for case in summary.cases)


def test_live_smoke_defaults_are_documented_with_verification_date(
    planning_module: Any,
) -> None:
    module = planning_module

    from agent_runtime import _builtin_runtime_client as runtime_client

    assert module.LIVE_SMOKE_DEFAULTS == {
        "claude": ("haiku", "low"),
        "codex": ("gpt-5.4-mini", "low"),
        "opencode": ("deepseek-v4-flash", "medium"),
    }
    assert module.LIVE_SMOKE_DEFAULTS_VERIFIED_ON == "2026-06-20"
    assert (
        module.LIVE_SMOKE_DEFAULTS["claude"][0] in runtime_client._CLAUDE_VALID_MODELS
    )
    assert (
        module.LIVE_SMOKE_DEFAULTS["claude"][1] in runtime_client._CLAUDE_VALID_EFFORTS
    )
    assert module.LIVE_SMOKE_DEFAULTS["codex"][0] in runtime_client._CODEX_VALID_MODELS
    assert module.LIVE_SMOKE_DEFAULTS["codex"][1] in runtime_client._CODEX_VALID_EFFORTS
    assert (
        module.LIVE_SMOKE_DEFAULTS["opencode"][0] in runtime_client._OPENCODE_GO_MODELS
    )
    assert (
        module.LIVE_SMOKE_DEFAULTS["opencode"][1]
        in runtime_client._OPENCODE_VALID_EFFORTS
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


def test_live_smoke_dry_run_reports_planned_cases_and_artifact_paths(
    planning_module: Any,
    tmp_path: Path,
) -> None:
    module = planning_module

    summary = module.build_dry_run_plan(
        provider_selection=("claude", "opencode"),
        lifecycle_modes=("ephemeral", "new_session"),
        tool_policies=("NONE", "UNRESTRICTED"),
        run_id="smoke-run-2026.06.19",
        model_overrides={"claude": "sonnet", "opencode": "deepseek-v4-flash"},
        effort_overrides={"claude": "medium", "opencode": "high"},
        claude_code_oauth_token="token",
        opencode_api_key="api-key",
        artifact_root=tmp_path / "live-smoke-artifacts",
    )

    assert summary.run_id == "smoke-run-2026.06.19"
    assert len(summary.cases) == 8
    assert len(summary.provider_plans) == 2
    first_case = summary.cases[0]
    assert isinstance(first_case, module.DryRunPlannedCase)
    assert first_case.service in {"claude", "opencode"}
    assert first_case.mode in {"ephemeral", "new_session"}
    assert first_case.policy in {"NONE", "UNRESTRICTED"}
    assert first_case.model in {"sonnet", "deepseek-v4-flash"}
    assert first_case.effort in {"medium", "high"}
    assert "smoke-run-2026.06.19" in str(first_case.artifact_path)
    assert str(first_case.artifact_path).startswith(
        str((tmp_path / "live-smoke-artifacts" / "smoke-run-2026.06.19"))
    )


def test_live_smoke_dry_run_validation_matches_preflight_rules(
    planning_module: Any,
) -> None:
    module = planning_module

    from agent_runtime.errors import RuntimeConfigurationError

    with pytest.raises(RuntimeConfigurationError):
        module.build_dry_run_plan(
            provider_selection="bad_provider",
            lifecycle_modes=("ephemeral",),
        )

    with pytest.raises(RuntimeConfigurationError):
        module.build_dry_run_plan(
            provider_selection=("claude",),
            lifecycle_modes=("ephemeral",),
            run_id="../unsafe-run-id",
            claude_code_oauth_token="token",
        )

    explicit_missing = module.build_dry_run_plan(
        provider_selection=("codex",),
        lifecycle_modes=("ephemeral",),
        run_id="explicit-missing",
    )
    assert (
        explicit_missing.provider_plans[0].status
        is module.LiveSmokeProviderSelectionStatus.CONFIG_ERROR
    )

    all_missing = module.build_dry_run_plan(
        provider_selection="all",
        lifecycle_modes=("ephemeral",),
        run_id="all-missing",
    )
    assert all(
        plan.status is module.LiveSmokeProviderSelectionStatus.SKIPPED
        for plan in all_missing.provider_plans
    )


def test_live_smoke_dry_run_planning_writes_no_artifacts(
    planning_module: Any,
    tmp_path: Path,
) -> None:
    module = planning_module

    summary = module.build_dry_run_plan(
        provider_selection=("claude",),
        lifecycle_modes=("ephemeral", "new_session"),
        tool_policies=("NONE",),
        run_id="no-side-effects",
        model_overrides={"claude": "sonnet"},
        effort_overrides={"claude": "medium"},
        claude_code_oauth_token="token",
        artifact_root=tmp_path / "live-smoke-artifacts",
    )

    assert not any(case.artifact_path.exists() for case in summary.cases)
    assert not (tmp_path / "live-smoke-artifacts").exists()


def test_live_smoke_dry_run_to_json_is_machine_readable(
    planning_module: Any,
    tmp_path: Path,
) -> None:
    module = planning_module

    summary = module.build_dry_run_plan(
        provider_selection=("codex",),
        lifecycle_modes=("ephemeral",),
        run_id="json-readability",
        model_overrides={"codex": "codex-mini"},
        effort_overrides={"codex": "high"},
        codex_auth_present=True,
        artifact_root=tmp_path / "artifacts",
    )
    payload = json.loads(module.dry_run_plan_to_json(summary))

    assert payload["run_id"] == "json-readability"
    assert len(payload["cases"]) == 1
    assert payload["cases"][0]["service"] in {"codex"}
    assert payload["cases"][0]["policy"] is None
    assert payload["cases"][0]["artifact_path"].endswith(
        "json-readability/codex/ephemeral/default"
    )
    assert payload["providers"][0]["status"] == "runnable"


def test_live_smoke_dry_run_json_artifact_paths_use_forward_slashes_portably(
    planning_module: Any,
) -> None:
    module = planning_module

    artifact_root = r"C:\temp\live-smoke-artifacts"
    summary = module.build_dry_run_plan(
        provider_selection=("codex",),
        lifecycle_modes=("ephemeral",),
        run_id="portable-json",
        model_overrides={"codex": "codex-mini"},
        effort_overrides={"codex": "high"},
        codex_auth_present=True,
        artifact_root=artifact_root,
    )

    assert summary.artifact_root == Path(artifact_root)
    assert summary.cases[0].artifact_path == (
        Path(artifact_root) / "portable-json" / "codex" / "ephemeral" / "default"
    )

    payload = json.loads(module.dry_run_plan_to_json(summary))

    assert payload["artifact_root"] == "C:/temp/live-smoke-artifacts"
    assert (
        payload["cases"][0]["artifact_path"]
        == "C:/temp/live-smoke-artifacts/portable-json/codex/ephemeral/default"
    )


def test_live_smoke_dry_run_to_json_exposes_resolved_live_smoke_defaults(
    planning_module: Any,
    tmp_path: Path,
) -> None:
    module = planning_module

    summary = module.build_dry_run_plan(
        provider_selection=("claude",),
        lifecycle_modes=("ephemeral",),
        run_id="json-defaults",
        claude_code_oauth_token="token",
        artifact_root=tmp_path / "artifacts",
    )
    payload = json.loads(module.dry_run_plan_to_json(summary))

    assert payload["providers"][0]["model"] == "haiku"
    assert payload["providers"][0]["effort"] == "low"
    assert payload["cases"][0]["model"] == "haiku"
    assert payload["cases"][0]["effort"] == "low"


def test_live_smoke_provider_listing_reports_configuration_without_secrets(
    planning_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = planning_module

    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "very-secret-claude-token")
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "very-secret-opencode-key")

    listing = module.list_supported_providers(
        env={"CODex": "ignored"},
        claude_code_oauth_token="cli-claude-token",
        opencode_api_key="cli-opencode-key",
        codex_auth_present=False,
    )

    assert {entry.service for entry in listing} == set(module.SUPPORTED_PROVIDERS)
    status_by_service = {entry.service: entry.configured for entry in listing}
    assert status_by_service["claude"] is True
    assert status_by_service["opencode"] is True
    assert status_by_service["codex"] is False

    for entry in listing:
        text = repr(entry)
        assert "very-secret-claude-token" not in text
        assert "very-secret-opencode-key" not in text
        assert "cli-claude-token" not in text
        assert "cli-opencode-key" not in text


def test_live_smoke_aggregate_exit_status_rejects_required_failures_only(
    planning_module: Any,
) -> None:
    module = planning_module

    passing = module.LiveSmokeCaseResult(
        service="claude",
        mode="ephemeral",
        policy=None,
        status=module.LiveSmokeCaseStatus.PASSED,
        required=True,
    )
    allowed_skip = module.LiveSmokeCaseResult(
        service="codex",
        mode="ephemeral",
        policy=None,
        status=module.LiveSmokeCaseStatus.SKIPPED,
        required=False,
    )
    dependent_skip = module.LiveSmokeCaseResult(
        service="opencode",
        mode="ephemeral",
        policy="NONE",
        status=module.LiveSmokeCaseStatus.SKIPPED,
        required=False,
    )
    failed = module.LiveSmokeCaseResult(
        service="claude",
        mode="new_session",
        policy=None,
        status=module.LiveSmokeCaseStatus.FAILED,
        required=True,
    )

    clean_exit = module.compute_live_smoke_exit_status(
        (passing, allowed_skip, dependent_skip)
    )
    mixed_exit = module.compute_live_smoke_exit_status((passing, allowed_skip, failed))

    assert clean_exit == 0
    assert mixed_exit == 1


def test_live_smoke_case_status_classifies_completed_runtime_outcomes_as_passed(
    planning_module: Any,
) -> None:
    module = planning_module

    from agent_runtime import runtime as prompt_runtime

    case = _planned_case(
        module,
        service="claude",
        mode="ephemeral",
        policy=None,
        model="sonnet",
        effort="medium",
    )
    outcome = prompt_runtime.RuntimeOutcome(
        kind="completed", output="smoke output text"
    )

    result = module.classify_live_smoke_case_result(
        case=case,
        runtime_outcome=outcome,
        required_output_non_empty=True,
    )

    assert module.LiveSmokeCaseStatus.PASSED == result.status
    assert result.status.value in {
        "passed",
        "skipped",
        "config_error",
        "usage_limited",
        "no_service_available",
        "failed",
        "error",
    }
    assert result.diagnostic is None


def test_live_smoke_case_status_distinguishes_expected_outcomes_from_failures(
    planning_module: Any,
) -> None:
    module = planning_module

    from agent_runtime import runtime as prompt_runtime

    lifecycle_case = _planned_case(
        module,
        service="claude",
        mode="new_session",
        policy=None,
        model="sonnet",
        effort="medium",
    )
    passed = module.classify_live_smoke_case_result(
        case=lifecycle_case,
        runtime_outcome=prompt_runtime.RuntimeOutcome(
            kind="completed",
            output="first session response",
            continuation=prompt_runtime.Continuation(serialized="resume-token-abc"),
        ),
        required_continuation_text="resume-token-abc",
    )
    limited = module.classify_live_smoke_case_result(
        case=lifecycle_case,
        runtime_outcome=prompt_runtime.RuntimeOutcome(
            kind="usage_limited",
            output="quota exhausted",
            service_name="claude",
            reset_time=None,
            invocation_progress=prompt_runtime.InvocationProgress.NOT_STARTED,
        ),
    )
    unavailable = module.classify_live_smoke_case_result(
        case=lifecycle_case,
        runtime_outcome=prompt_runtime.RuntimeOutcome.no_service_available(
            output="temporary outage",
            reset_time=None,
            invocation_progress=prompt_runtime.InvocationProgress.NOT_STARTED,
        ),
    )
    provider_failed = module.classify_live_smoke_case_result(
        case=lifecycle_case,
        runtime_outcome=prompt_runtime.RuntimeOutcome(
            kind="retryable_provider_failure",
            output="provider side failure",
            service_name="claude",
            invocation_progress=prompt_runtime.InvocationProgress.STARTED,
        ),
    )
    timed_out = module.classify_live_smoke_case_result(
        case=lifecycle_case,
        runtime_outcome=prompt_runtime.RuntimeOutcome(
            kind="timed_out",
            output="execution timed out",
            invocation_progress=prompt_runtime.InvocationProgress.STARTED,
        ),
    )

    assert passed.status == module.LiveSmokeCaseStatus.PASSED
    assert limited.status == module.LiveSmokeCaseStatus.USAGE_LIMITED
    assert unavailable.status == module.LiveSmokeCaseStatus.NO_SERVICE_AVAILABLE
    assert provider_failed.status == module.LiveSmokeCaseStatus.FAILED
    assert timed_out.status == module.LiveSmokeCaseStatus.FAILED


def test_live_smoke_case_status_checks_completed_result_metadata(
    planning_module: Any,
) -> None:
    module = planning_module

    from types import SimpleNamespace

    from agent_runtime import runtime as prompt_runtime

    case = _planned_case(
        module,
        service="claude",
        mode="ephemeral",
        policy="UNRESTRICTED",
        model="sonnet",
        effort="medium",
    )
    wrong_result = SimpleNamespace(
        selected_service="codex",
        selected_model="other-model",
        selected_effort="low",
        tool_access=SimpleNamespace(tool_policy="inspect_only"),
    )

    mismatch = module.classify_live_smoke_case_result(
        case=case,
        runtime_outcome=prompt_runtime.RuntimeOutcome(
            kind="completed",
            output="smoke output",
            result=cast(Any, wrong_result),
        ),
    )

    assert mismatch.status == module.LiveSmokeCaseStatus.FAILED
    assert "metadata mismatch" in mismatch.diagnostic


def test_live_smoke_config_error_and_all_mode_skips_classify_distinctly(
    planning_module: Any,
) -> None:
    module = planning_module

    explicit_plan = module.plan_selected_providers(
        module.parse_provider_selection("claude"),
        model_overrides={"claude": "sonnet"},
        effort_overrides={"claude": "medium"},
    )
    explicit_case = _planned_case(
        module,
        service="claude",
        mode="ephemeral",
        policy=None,
        model="sonnet",
        effort="medium",
    )
    explicit_result = module.classify_live_smoke_preflight_case_result(
        case=explicit_case,
        provider_plan=explicit_plan[0],
        required=True,
    )
    assert explicit_result.status == module.LiveSmokeCaseStatus.CONFIG_ERROR

    all_selection = module.parse_provider_selection("all")
    all_plans = module.plan_selected_providers(
        all_selection,
        model_overrides={
            "claude": "sonnet",
            "codex": "codex-mini",
            "opencode": "deepseek",
        },
        effort_overrides={
            "claude": "medium",
            "codex": "high",
            "opencode": "medium",
        },
    )
    all_results = tuple(
        module.classify_live_smoke_preflight_case_result(
            case=_planned_case(
                module,
                service=provider_plan.service,
                mode="ephemeral",
                policy=None,
                model=provider_plan.model,
                effort=provider_plan.effort,
                auth=provider_plan.provider_selection.auth,
            ),
            provider_plan=provider_plan,
            required=False,
            dependent_skip=provider_plan.service == "opencode",
        )
        for provider_plan in all_plans
    )
    assert all(
        result.status == module.LiveSmokeCaseStatus.SKIPPED for result in all_results
    )
    assert all(result.required is False for result in all_results)


def test_live_smoke_non_passing_runtime_and_artifact_failures_emit_diagnostics(
    planning_module: Any,
) -> None:
    module = planning_module

    from agent_runtime import runtime as prompt_runtime

    case = _planned_case(
        module,
        service="claude",
        mode="new_session",
        policy="NONE",
        model="sonnet",
        effort="medium",
    )
    failure_results = (
        module.classify_live_smoke_case_result(
            case=case,
            runtime_outcome=prompt_runtime.RuntimeOutcome(
                kind="retryable_provider_failure",
                output="provider could not recover",
                service_name="claude",
                invocation_progress=prompt_runtime.InvocationProgress.STARTED,
            ),
        ),
        module.classify_live_smoke_case_result(
            case=case,
            runtime_outcome=prompt_runtime.RuntimeOutcome(
                kind="timed_out",
                output="provider timed out",
                invocation_progress=prompt_runtime.InvocationProgress.STARTED,
            ),
        ),
        module.classify_live_smoke_case_result(
            case=case,
            runtime_exception=RuntimeError("runtime exception"),
        ),
        module.classify_live_smoke_case_result(
            case=case,
            artifact_error="artifact write failure",
        ),
    )

    payload = module.live_smoke_case_results_payload(failure_results)
    statuses = {entry["status"] for entry in payload["cases"]}

    assert statuses == {
        "failed",
        "error",
    }
    assert any(
        "provider could not recover" in str(entry["diagnostic"])
        for entry in payload["cases"]
    )
    assert any(
        "runtime exception" in str(entry["diagnostic"]) for entry in payload["cases"]
    )
    assert any(
        "artifact write failure" in str(entry["diagnostic"])
        for entry in payload["cases"]
    )
