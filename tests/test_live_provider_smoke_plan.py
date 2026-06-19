from __future__ import annotations

import importlib.util
import json
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
    assert isolated[0].model == ""
    assert isolated[0].effort == ""
    assert isolated[0].status is module.LiveSmokeProviderSelectionStatus.CONFIG_ERROR

    planned = module.plan_selected_providers(
        selected,
        env={
            module.LIVE_SMOKE_CLAUDE_MODEL_ENV: "env-model",
            module.LIVE_SMOKE_CLAUDE_EFFORT_ENV: "env-effort",
            "CLAUDE_CODE_OAUTH_TOKEN": "env-token",
        },
    )
    assert planned[0].model == "env-model"
    assert planned[0].effort == "env-effort"
    assert planned[0].status is module.LiveSmokeProviderSelectionStatus.RUNNABLE


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

    case = module.PlannedCase(
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

    lifecycle_case = module.PlannedCase(
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


def test_live_smoke_config_error_and_all_mode_skips_classify_distinctly(
    planning_module: Any,
) -> None:
    module = planning_module

    explicit_plan = module.plan_selected_providers(
        module.parse_provider_selection("claude"),
        model_overrides={"claude": "sonnet"},
        effort_overrides={"claude": "medium"},
    )
    explicit_case = module.PlannedCase(
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
            case=module.PlannedCase(
                service=provider_plan.service,
                mode="ephemeral",
                policy=None,
                model=provider_plan.model,
                effort=provider_plan.effort,
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

    case = module.PlannedCase(
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
