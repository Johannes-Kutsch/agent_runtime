from __future__ import annotations

import importlib.util
import asyncio
import io
import json
import subprocess
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from types import SimpleNamespace

import pytest
from agent_runtime.session import RunKind

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "live_provider_smoke.py"


@pytest.fixture
def smoke_module() -> object:
    spec = importlib.util.spec_from_file_location(
        "live_provider_smoke",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # type: ignore[arg-type]
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class _FakeRunOutcome:
    kind: str
    output: str


def test_live_smoke_default_artifact_root_is_repo_local_with_override_and_cleanup(
    smoke_module: object, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module: Any = smoke_module

    monkeypatch.chdir(tmp_path)

    run_plan_artifacts: list[Path] = []

    def _fake_case_runner(*, artifact_dir: Path, **_: object) -> _FakeRunOutcome:
        run_plan_artifacts.append(artifact_dir)
        return _FakeRunOutcome(kind="completed", output="ok")

    default_result = module.run_live_smoke(
        provider_selection=("codex",),
        lifecycle_modes=("ephemeral",),
        model_overrides={"codex": "codex-mini"},
        effort_overrides={"codex": "high"},
        codex_auth_present=True,
        run_id="default-artifact-run",
        case_runner=_fake_case_runner,
    )
    assert isinstance(default_result, module.LiveSmokeRunResult)
    assert (
        default_result.artifact_root
        == tmp_path / module.DEFAULT_LIVE_SMOKE_ARTIFACT_ROOT
    )
    assert default_result.run_id == "default-artifact-run"
    assert (
        default_result.summary_path.parent
        == default_result.artifact_root / "default-artifact-run"
    )
    assert default_result.summary_written is True
    assert default_result.passed is True
    assert run_plan_artifacts
    assert default_result.artifact_root.exists()

    override_root = tmp_path / "custom-smoke-artifacts"
    override_result = module.run_live_smoke(
        provider_selection=("codex",),
        lifecycle_modes=("ephemeral",),
        model_overrides={"codex": "codex-mini"},
        effort_overrides={"codex": "high"},
        codex_auth_present=True,
        run_id="override-artifact-run",
        artifact_root=override_root,
        case_runner=_fake_case_runner,
    )
    assert override_result.artifact_root == override_root.resolve()
    assert (
        override_result.summary_path.parent == override_root / "override-artifact-run"
    )
    assert override_result.artifact_root.exists()

    stale_root = tmp_path / "stale-smoke-artifacts"
    stale_root.mkdir()
    (stale_root / "old.txt").write_text("stale", encoding="utf-8")
    cleaned_result = module.run_live_smoke(
        provider_selection=("codex",),
        lifecycle_modes=("ephemeral",),
        model_overrides={"codex": "codex-mini"},
        effort_overrides={"codex": "high"},
        codex_auth_present=True,
        run_id="cleaned-run",
        artifact_root=stale_root,
        cleanup_artifact_root=True,
        case_runner=_fake_case_runner,
    )
    assert not (stale_root / "old.txt").exists()
    assert cleaned_result.summary_path.parent == stale_root / "cleaned-run"


def test_live_smoke_creates_case_artifact_dir_before_running_case(
    smoke_module: object,
    tmp_path: Path,
) -> None:
    module: Any = smoke_module
    observed_dirs: list[Path] = []

    def _fake_case_runner(*, artifact_dir: Path, **_: object) -> _FakeRunOutcome:
        assert artifact_dir.is_dir()
        observed_dirs.append(artifact_dir)
        return _FakeRunOutcome(kind="completed", output="ok")

    result = module.run_live_smoke(
        provider_selection=("opencode",),
        lifecycle_modes=("ephemeral",),
        model_overrides={"opencode": "deepseek-v4-flash"},
        effort_overrides={"opencode": "medium"},
        opencode_api_key="api-key",
        run_id="precreated-case-dir",
        artifact_root=tmp_path / "smoke-artifacts",
        case_runner=_fake_case_runner,
    )

    assert result.passed is True
    assert observed_dirs == [
        tmp_path
        / "smoke-artifacts"
        / "precreated-case-dir"
        / "opencode"
        / "ephemeral"
        / "default"
    ]


def test_live_smoke_required_summary_write_failure_marks_run_non_passing(
    smoke_module: object,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module: Any = smoke_module

    def _fake_case_runner(*, artifact_dir: Path, **_: object) -> _FakeRunOutcome:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        return _FakeRunOutcome(kind="completed", output="ok")

    module.DEFAULT_LIVE_SMOKE_ARTIFACT_ROOT = str(tmp_path / "summary-failure-root")

    def _write_summary_never(*_: Any, **__: Any) -> bool:
        return False

    monkeypatch.setattr(module, "_write_required_summary", _write_summary_never)

    result = module.run_live_smoke(
        provider_selection=("codex",),
        lifecycle_modes=("ephemeral",),
        model_overrides={"codex": "codex-mini"},
        effort_overrides={"codex": "high"},
        codex_auth_present=True,
        run_id="summary-write-fails",
        case_runner=_fake_case_runner,
    )

    assert result.passed is False
    assert result.summary_written is False


def test_live_smoke_optional_diagnostics_failures_are_warnings_not_run_failures(
    smoke_module: object, tmp_path: Path
) -> None:
    module: Any = smoke_module

    def _fake_case_runner(*, artifact_dir: Path, **_: object) -> _FakeRunOutcome:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        return _FakeRunOutcome(kind="completed", output="ok")

    def _raise_optional(*_: Any, **__: Any) -> None:
        raise OSError("optional artifact write failed")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(module, "_write_optional_case_artifacts", _raise_optional)
    try:
        result = module.run_live_smoke(
            provider_selection=("codex",),
            lifecycle_modes=("ephemeral",),
            model_overrides={"codex": "codex-mini"},
            effort_overrides={"codex": "high"},
            codex_auth_present=True,
            run_id="optional-warning-run",
            artifact_root=tmp_path / "optional-artifacts",
            case_runner=_fake_case_runner,
        )
    finally:
        monkeypatch.undo()

    assert result.summary_written is True
    assert result.passed is True
    assert any("optional case artifact" in warning for warning in result.warnings)
    summary_payload = module.json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert any(
        "optional case artifact" in warning for warning in summary_payload["warnings"]
    )


def test_live_smoke_repeated_planned_cases_get_isolated_invocation_directories(
    smoke_module: object, tmp_path: Path
) -> None:
    module: Any = smoke_module

    invocation_directories: list[Path] = []

    def _fake_case_runner(*, artifact_dir: Path, **_: object) -> _FakeRunOutcome:
        invocation_directories.append(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        return _FakeRunOutcome(kind="completed", output="ok")

    result = module.run_live_smoke(
        provider_selection=("codex",),
        lifecycle_modes=("ephemeral", "ephemeral"),
        model_overrides={"codex": "codex-mini"},
        effort_overrides={"codex": "high"},
        codex_auth_present=True,
        run_id="repeated-cases-run",
        artifact_root=tmp_path / "duplicate-cases",
        case_runner=_fake_case_runner,
    )

    assert result.passed is True
    assert len(invocation_directories) == 2
    assert invocation_directories[0] != invocation_directories[1]
    assert all(path.exists() for path in invocation_directories)
    assert {case.artifact_path for case in result.cases} == {
        str(path) for path in invocation_directories
    }


def test_live_smoke_summary_persists_late_optional_warning_details(
    smoke_module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module: Any = smoke_module

    def _fake_case_runner(*, artifact_dir: Path, **_: object) -> _FakeRunOutcome:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        return _FakeRunOutcome(kind="completed", output="ok")

    def _raise_marker_write(*_: Any, **__: Any) -> None:
        raise OSError("marker write failed")

    monkeypatch.setattr(module, "_write_artifacts_marker", _raise_marker_write)

    result = module.run_live_smoke(
        provider_selection=("codex",),
        lifecycle_modes=("ephemeral",),
        model_overrides={"codex": "codex-mini"},
        effort_overrides={"codex": "high"},
        codex_auth_present=True,
        run_id="late-warning-run",
        artifact_root=tmp_path / "late-warning-artifacts",
        case_runner=_fake_case_runner,
    )

    assert result.summary_written is True
    assert any("artifacts marker" in warning for warning in result.warnings)

    summary_payload = module.json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert any("artifacts marker" in warning for warning in summary_payload["warnings"])


def test_live_smoke_artifacts_capture_required_diagnostics(
    smoke_module: object, tmp_path: Path
) -> None:
    module: Any = smoke_module

    @dataclass(frozen=True)
    class _FakeLiveTurn:
        text: str
        service_name: str

    @dataclass(frozen=True)
    class _FakeInvocationRecord:
        run_kind: str
        service_name: str
        provider_session_id: str
        prompt: str
        provider_output: bytes

    @dataclass(frozen=True)
    class _FakeCaseOutcome:
        kind: str
        output: str
        live_turns: tuple[_FakeLiveTurn, ...]
        invocation_records: tuple[_FakeInvocationRecord, ...]

    def _fake_case_runner(*, artifact_dir: Path, **_: object) -> _FakeCaseOutcome:
        return _FakeCaseOutcome(
            kind="completed",
            output="provider output value",
            live_turns=(_FakeLiveTurn(text="provider says ok", service_name="codex"),),
            invocation_records=(
                _FakeInvocationRecord(
                    run_kind="fresh",
                    service_name="codex",
                    provider_session_id="session-id",
                    prompt="prompt text",
                    provider_output=b"binary output",
                ),
            ),
        )

    result = module.run_live_smoke(
        provider_selection=("codex",),
        lifecycle_modes=("ephemeral",),
        model_overrides={"codex": "codex-mini"},
        effort_overrides={"codex": "high"},
        codex_auth_present=True,
        run_id="diagnostic-run",
        artifact_root=tmp_path / "diagnostics",
        case_runner=_fake_case_runner,
    )

    assert result.passed is True
    assert result.summary_written is True

    summary_payload = result.summary_path.read_text(encoding="utf-8")
    summary_json = module.json.loads(summary_payload)
    assert summary_json["run_id"] == "diagnostic-run"
    assert summary_json["cases"][0]["service"] == "codex"
    assert summary_json["cases"][0]["model"] == "codex-mini"
    assert summary_json["cases"][0]["effort"] == "high"
    assert summary_json["cases"][0]["policy"] is None
    assert "provider_plans" in summary_json
    assert summary_json["provider_plans"][0]["service"] == "codex"
    assert summary_json["provider_plans"][0]["model"] == "codex-mini"

    case_dir = (
        tmp_path / "diagnostics" / "diagnostic-run" / "codex" / "ephemeral" / "default"
    )
    assert (case_dir / "prompt.txt").exists()
    assert (case_dir / "final_output.txt").read_text(
        encoding="utf-8"
    ) == "provider output value"
    assert (case_dir / "live_turns.json").exists()
    assert (case_dir / "invocation_records.json").exists()
    assert (case_dir / "outcome.json").exists()
    assert (case_dir / "timings.json").exists()

    live_turns = module.json.loads(
        (case_dir / "live_turns.json").read_text(encoding="utf-8")
    )
    assert live_turns == [{"text": "provider says ok", "service_name": "codex"}]
    invocation_records = module.json.loads(
        (case_dir / "invocation_records.json").read_text(encoding="utf-8")
    )
    assert invocation_records[0]["provider_session_id"] == "session-id"
    config_summary = tmp_path / "diagnostics" / "diagnostic-run" / "config_summary.json"
    assert config_summary.exists()


def test_live_smoke_real_run_preserves_resolved_defaults_in_diagnostics_and_reruns(
    smoke_module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module: Any = smoke_module

    default_model, default_effort = module.live_provider_smoke_plan.LIVE_SMOKE_DEFAULTS[
        "codex"
    ]

    def _fake_case_runner(*, artifact_dir: Path, **_: object) -> _FakeRunOutcome:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        return _FakeRunOutcome(
            kind="retryable_provider_failure",
            output="provider runtime returned failed",
        )

    run_result = module.run_live_smoke(
        provider_selection=("codex",),
        lifecycle_modes=("ephemeral",),
        codex_auth_present=True,
        run_id="defaults-preserved-run",
        artifact_root=tmp_path / "defaults-preserved",
        case_runner=_fake_case_runner,
    )

    assert run_result.passed is False
    assert run_result.cases[0].model == default_model
    assert run_result.cases[0].effort == default_effort

    summary_payload = module.json.loads(run_result.summary_path.read_text("utf-8"))
    assert summary_payload["cases"][0]["model"] == default_model
    assert summary_payload["cases"][0]["effort"] == default_effort
    assert summary_payload["provider_plans"][0]["model"] == default_model
    assert summary_payload["provider_plans"][0]["effort"] == default_effort

    case_dir = (
        tmp_path
        / "defaults-preserved"
        / "defaults-preserved-run"
        / "codex"
        / "ephemeral"
        / "default"
    )
    outcome_payload = module.json.loads(
        (case_dir / "outcome.json").read_text(encoding="utf-8")
    )
    assert outcome_payload["service"] == "codex"
    assert outcome_payload["model"] == default_model
    assert outcome_payload["effort"] == default_effort
    assert outcome_payload["mode"] == "ephemeral"
    assert outcome_payload["policy"] is None

    monkeypatch.setattr(module, "run_live_smoke", lambda **_: run_result)
    output = io.StringIO()
    with redirect_stdout(output):
        exit_code = module.main(
            [
                "--provider",
                "codex",
                "--mode",
                "ephemeral",
                "--json",
                "--run-id",
                "defaults-preserved-run",
                "--artifact-root",
                str(tmp_path / "defaults-preserved"),
            ]
        )

    payload = module.json.loads(output.getvalue())
    assert exit_code == 1
    assert payload["failed_case_runs"] == [
        {
            "provider": "codex",
            "mode": "ephemeral",
            "policy": None,
            "status": "failed",
            "command": module._build_case_rerun_command(
                run_result.cases[0],
                run_id="defaults-preserved-run",
            ),
        }
    ]


def test_build_case_rerun_command_uses_windows_command_format(
    smoke_module: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module: Any = smoke_module
    windows_script_path = (
        r"D:\Bootcamp\Projects\agent_runtime\pycastle\.worktrees\host-check-e92b06a"
        r"\scripts\live_provider_smoke.py"
    )
    case = module.LiveSmokeRunCaseResult(
        service="codex",
        mode="ephemeral",
        policy=None,
        model="gpt-5.4-mini",
        effort="low",
        artifact_path="unused",
        status="failed",
        required=True,
        provider_output="",
        diagnostic="failed",
        traceback=None,
        duration_seconds=0.1,
    )

    monkeypatch.setattr(module, "__file__", windows_script_path)
    monkeypatch.setattr(module, "os", SimpleNamespace(name="nt"))
    expected_command = subprocess.list2cmdline(
        [
            "python",
            windows_script_path,
            "--provider",
            "codex",
            "--mode",
            "ephemeral",
            "--model",
            "codex=gpt-5.4-mini",
            "--effort",
            "codex=low",
            "--run-id",
            "windows-rerun-run",
        ]
    )
    command = module._build_case_rerun_command(case, run_id="windows-rerun-run")

    assert command == expected_command
    assert "'" not in expected_command


def test_live_smoke_artifacts_do_not_capture_credentials_or_raw_env(
    smoke_module: object, tmp_path: Path
) -> None:
    module: Any = smoke_module

    def _fake_case_runner(*, artifact_dir: Path, **_: object) -> _FakeRunOutcome:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        return _FakeRunOutcome(kind="completed", output="provider output value")

    env = {
        "CLAUDE_CODE_OAUTH_TOKEN": "super-secret-claude-token",
        "OPENCODE_GO_API_KEY": "super-secret-opencode-key",
        "HOME": "/home/agent/real-home-should-not-leak",
    }

    result = module.run_live_smoke(
        provider_selection=("codex",),
        lifecycle_modes=("ephemeral",),
        model_overrides={"codex": "codex-mini"},
        effort_overrides={"codex": "high"},
        codex_auth_present=True,
        run_id="sensitive-run",
        artifact_root=tmp_path / "sensitive",
        env=env,
        case_runner=_fake_case_runner,
    )

    assert result.passed is True
    summary_payload = result.summary_path.read_text(encoding="utf-8")
    assert "super-secret-claude-token" not in summary_payload
    assert "super-secret-opencode-key" not in summary_payload

    case_dir = (
        tmp_path / "sensitive" / "sensitive-run" / "codex" / "ephemeral" / "default"
    )
    case_output_payload = (case_dir / "final_output.txt").read_text(encoding="utf-8")
    assert case_output_payload == "provider output value"

    config_summary = tmp_path / "sensitive" / "sensitive-run" / "config_summary.json"
    config_payload = config_summary.read_text(encoding="utf-8")
    assert "super-secret-claude-token" not in config_payload
    assert "super-secret-opencode-key" not in config_payload


def test_live_smoke_preserves_provider_output_without_redaction_and_marks_sensitive(
    smoke_module: object, tmp_path: Path
) -> None:
    module: Any = smoke_module

    secret_output = "provider emitted model-output with 4f1e2d-secret"

    def _fake_case_runner(*, artifact_dir: Path, **_: object) -> _FakeRunOutcome:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        return _FakeRunOutcome(kind="completed", output=secret_output)

    run_result = module.run_live_smoke(
        provider_selection=("codex",),
        lifecycle_modes=("ephemeral",),
        model_overrides={"codex": "codex-mini"},
        effort_overrides={"codex": "high"},
        codex_auth_present=True,
        run_id="sensitive-output-run",
        artifact_root=tmp_path / "sensitive-output",
        case_runner=_fake_case_runner,
    )
    assert run_result.summary_written is True

    final_output = (
        tmp_path
        / "sensitive-output"
        / "sensitive-output-run"
        / "codex"
        / "ephemeral"
        / "default"
        / "final_output.txt"
    ).read_text(encoding="utf-8")
    assert final_output == secret_output
    assert secret_output in final_output

    marker = tmp_path / "sensitive-output" / "sensitive-output-run" / "ARTIFACTS.md"
    assert marker.exists()
    assert "potentially sensitive" in marker.read_text(encoding="utf-8").lower()


def test_live_smoke_ephemeral_runs_through_public_runtime_request_values(
    smoke_module: object,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module: Any = smoke_module
    from agent_runtime import runtime as prompt_runtime

    captured_request: dict[str, prompt_runtime.EphemeralRunRequest | None] = {
        "request": None
    }

    class _FakeRuntimeClient:
        def run_ephemeral(
            self,
            request: prompt_runtime.EphemeralRunRequest,
        ) -> object:
            captured_request["request"] = request
            if request.on_live_output is not None:
                request.on_live_output(
                    prompt_runtime.AgentMessageTurn(
                        text="provider says ok", service_name="claude"
                    )
                )
            return SimpleNamespace(
                kind="completed",
                output="provider output value",
                result=SimpleNamespace(
                    selected_service="claude",
                    selected_model="sonnet",
                    selected_effort="medium",
                    tool_access=SimpleNamespace(
                        tool_policy=prompt_runtime.ToolPolicy.UNRESTRICTED
                    ),
                ),
                live_turns=(),
                invocation_records=(),
            )

    def _fake_client() -> _FakeRuntimeClient:
        return _FakeRuntimeClient()

    monkeypatch.setattr(module, "RuntimeClient", _fake_client)

    run_result = module.run_live_smoke(
        provider_selection=("claude",),
        lifecycle_modes=("ephemeral",),
        model_overrides={"claude": "sonnet"},
        effort_overrides={"claude": "medium"},
        claude_code_oauth_token="token",
        run_id="public-runtime-run",
        artifact_root=tmp_path / "public-runtime-smoke",
    )

    assert run_result.passed is True
    assert run_result.cases[0].status == "passed"
    request = captured_request["request"]
    assert request is not None
    assert request.provider_selection.service == "claude"
    assert request.provider_selection.model == "sonnet"
    assert request.provider_selection.effort == "medium"
    assert request.provider_selection.auth == prompt_runtime.ProviderAuth(
        claude_code_oauth_token="token"
    )

    case_dir = (
        tmp_path
        / "public-runtime-smoke"
        / "public-runtime-run"
        / "claude"
        / "ephemeral"
        / "default"
    )
    live_turns = module.json.loads(
        (case_dir / "live_turns.json").read_text(encoding="utf-8")
    )
    assert live_turns == [{"text": "provider says ok", "service_name": "claude"}]


def test_live_smoke_runner_uses_planned_provider_selection_for_lifecycle_and_policy_requests(
    smoke_module: object,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module: Any = smoke_module
    from agent_runtime import runtime as prompt_runtime

    planned_selection = prompt_runtime.ProviderSelection(
        service="claude",
        model="planned-model",
        effort="planned-effort",
        auth=prompt_runtime.ProviderAuth(claude_code_oauth_token="planned-token"),
    )
    dry_run_plan = module.live_provider_smoke_plan.DryRunPlan(
        run_id="planned-selection-run",
        cases=(
            SimpleNamespace(
                service="claude",
                mode="new_session",
                policy=None,
                model="planned-model",
                effort="planned-effort",
                provider_selection=planned_selection,
            ),
            SimpleNamespace(
                service="claude",
                mode="ephemeral",
                policy="NONE",
                model="planned-model",
                effort="planned-effort",
                provider_selection=planned_selection,
            ),
        ),
        provider_plans=(
            SimpleNamespace(
                service="claude",
                status="runnable",
                model="planned-model",
                effort="planned-effort",
                provider_selection=planned_selection,
            ),
        ),
        artifact_root=tmp_path / "planned-selection-artifacts",
    )

    captured_requests: list[
        prompt_runtime.NewSessionRunRequest | prompt_runtime.EphemeralRunRequest
    ] = []

    class _FakeRuntimeClient:
        def run_new_session(
            self,
            request: prompt_runtime.NewSessionRunRequest,
        ) -> object:
            captured_requests.append(request)
            continuation = prompt_runtime.Continuation(
                selected_service="claude",
                selected_model="planned-model",
                selected_effort="planned-effort",
                tool_access=request.tool_access,
                provider_resume_state={"provider_session_id": "session-123"},
            )
            return prompt_runtime.RuntimeOutcome(
                kind="completed",
                output="new session output",
                continuation=continuation,
                result=prompt_runtime.SessionRunResult(
                    output="new session output",
                    runtime_metadata=prompt_runtime.SessionRuntimeMetadata(
                        service_name="claude",
                        provider_session_id="session-123",
                        run_kind=RunKind.FRESH,
                        session_namespace="",
                        exact_transcript_match=False,
                        selected_model="planned-model",
                        selected_effort="planned-effort",
                        tool_policy=prompt_runtime.ToolPolicy.UNRESTRICTED,
                    ),
                ),
                invocation_records=(),
            )

        def run_ephemeral(
            self,
            request: prompt_runtime.EphemeralRunRequest,
        ) -> object:
            captured_requests.append(request)
            return SimpleNamespace(
                kind="completed",
                output="ephemeral output",
                result=SimpleNamespace(
                    selected_service="claude",
                    selected_model="planned-model",
                    selected_effort="planned-effort",
                    tool_access=SimpleNamespace(tool_policy=request.tool_policy),
                ),
                invocation_records=(),
            )

    monkeypatch.setattr(
        module.live_provider_smoke_plan,
        "build_dry_run_plan",
        lambda *args, **kwargs: dry_run_plan,
    )
    monkeypatch.setattr(module, "RuntimeClient", lambda: _FakeRuntimeClient())

    run_result = module.run_live_smoke(
        provider_selection=("claude",),
        lifecycle_modes=("new_session",),
        tool_policies=("NONE",),
        claude_code_oauth_token="cli-token-that-should-not-win",
        run_id="planned-selection-run",
        artifact_root=tmp_path / "planned-selection-artifacts",
    )

    assert run_result.passed is True
    assert [request.provider_selection for request in captured_requests] == [
        planned_selection,
        planned_selection,
    ]


def test_live_smoke_timeout_outcome_is_classified_as_failed_and_retained(
    smoke_module: object,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module: Any = smoke_module
    from agent_runtime import runtime as prompt_runtime

    class _FakeRuntimeClient:
        def run_ephemeral(
            self,
            request: prompt_runtime.EphemeralRunRequest,
        ) -> object:
            return prompt_runtime.RuntimeOutcome.timed_out(
                output="provider timeout",
                invocation_progress=prompt_runtime.InvocationProgress.STARTED,
            )

    def _fake_client() -> _FakeRuntimeClient:
        return _FakeRuntimeClient()

    monkeypatch.setattr(module, "RuntimeClient", _fake_client)

    run_result = module.run_live_smoke(
        provider_selection=("claude",),
        lifecycle_modes=("ephemeral",),
        model_overrides={"claude": "sonnet"},
        effort_overrides={"claude": "medium"},
        claude_code_oauth_token="token",
        run_id="timeout-runtime-run",
        artifact_root=tmp_path / "timeout-runtime-smoke",
    )

    assert run_result.passed is False
    assert run_result.cases[0].status == "failed"
    assert run_result.cases[0].diagnostic == "provider timeout"


def test_live_smoke_public_runner_rejects_non_ephemeral_cases(
    smoke_module: object,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module: Any = smoke_module
    from agent_runtime import runtime as prompt_runtime

    captured_requests: list[Any] = []
    new_session_continuation: list[Any] = []

    class _FakeRuntimeClient:
        def run_new_session(
            self,
            request: prompt_runtime.NewSessionRunRequest,
        ) -> object:
            captured_requests.append(("new_session", request))
            continuation = prompt_runtime.Continuation(
                selected_service="claude",
                selected_model="sonnet",
                selected_effort="medium",
                tool_access=request.tool_access,
                provider_resume_state={"provider_session_id": "thread-123"},
            )
            new_session_continuation.append(continuation)
            return SimpleNamespace(
                kind="completed",
                output="start response with sentinel: session-token-2026.06.19",
                continuation=continuation,
                result=prompt_runtime.SessionRunResult(
                    output="start response with sentinel: session-token-2026.06.19",
                    runtime_metadata=prompt_runtime.SessionRuntimeMetadata(
                        service_name="claude",
                        provider_session_id="thread-123",
                        run_kind=RunKind.FRESH,
                        session_namespace="",
                        exact_transcript_match=False,
                        selected_model="sonnet",
                        selected_effort="medium",
                        tool_policy=prompt_runtime.ToolPolicy.UNRESTRICTED,
                    ),
                ),
                invocation_records=(
                    prompt_runtime.InvocationRecord(
                        run_kind=RunKind.FRESH,
                        service_name="claude",
                        provider_session_id="thread-123",
                        prompt="start response",
                        provider_output=b"start response bytes",
                    ),
                ),
            )

        def run_resumed_session(
            self,
            request: prompt_runtime.ResumedSessionRunRequest,
        ) -> object:
            captured_requests.append(("resumed_session", request))
            continuation = new_session_continuation[0]
            return SimpleNamespace(
                kind="completed",
                output="provider output: provider-session-receives session-token-2026.06.19 here",
                continuation=continuation,
                result=prompt_runtime.SessionRunResult(
                    output="provider output: provider-session-receives session-token-2026.06.19 here",
                    runtime_metadata=prompt_runtime.SessionRuntimeMetadata(
                        service_name="claude",
                        provider_session_id="thread-123",
                        run_kind=RunKind.RESUME,
                        session_namespace="",
                        exact_transcript_match=False,
                        selected_model="sonnet",
                        selected_effort="medium",
                        tool_policy=cast(
                            prompt_runtime.ToolPolicy,
                            getattr(
                                request,
                                "tool_policy",
                                prompt_runtime.ToolPolicy.UNRESTRICTED,
                            ),
                        ),
                    ),
                ),
                invocation_records=(
                    prompt_runtime.InvocationRecord(
                        run_kind=RunKind.RESUME,
                        service_name="claude",
                        provider_session_id="thread-123",
                        prompt="resume response",
                        provider_output=b"resume output bytes",
                    ),
                ),
            )

    def _fake_client() -> _FakeRuntimeClient:
        return _FakeRuntimeClient()

    monkeypatch.setattr(module, "RuntimeClient", _fake_client)

    run_result = module.run_live_smoke(
        provider_selection=("claude",),
        lifecycle_modes=("new_session", "resumed_session"),
        model_overrides={"claude": "sonnet"},
        effort_overrides={"claude": "medium"},
        claude_code_oauth_token="token",
        run_id="session-lifecycle-run",
        artifact_root=tmp_path / "session-lifecycle-smoke",
    )

    assert run_result.passed is True
    assert len(run_result.cases) == 2
    assert all(case.status == "passed" for case in run_result.cases)
    new_session_request = captured_requests[0][1]
    assert len(captured_requests) == 2
    assert captured_requests[0][0] == "new_session"
    assert captured_requests[1][0] == "resumed_session"
    assert new_session_request.provider_selection.auth == prompt_runtime.ProviderAuth(
        claude_code_oauth_token="token"
    )
    resumed_request = captured_requests[1][1]
    assert resumed_request.continuation == new_session_continuation[0]
    assert resumed_request.provider_auth == prompt_runtime.ProviderAuth(
        claude_code_oauth_token="token"
    )
    assert "session-token-2026.06.19" in run_result.cases[1].provider_output


def test_live_smoke_tool_policy_matrix_is_ephemeral_while_lifecycle_runs_all_modes(
    smoke_module: object,
    tmp_path: Path,
) -> None:
    module: Any = smoke_module
    from agent_runtime import runtime as prompt_runtime

    captured_calls: list[tuple[str, Any]] = []

    class _FakeRuntimeClient:
        def run_ephemeral(
            self,
            request: prompt_runtime.EphemeralRunRequest,
        ) -> object:
            captured_calls.append(("ephemeral", request.tool_policy))
            return SimpleNamespace(
                kind="completed",
                output="provider output value",
                result=SimpleNamespace(
                    selected_service=request.provider_selection.service,
                    selected_model=request.provider_selection.model,
                    selected_effort=request.provider_selection.effort,
                    tool_access=SimpleNamespace(
                        tool_policy=request.tool_policy,
                    ),
                ),
                invocation_records=(),
            )

        def run_new_session(
            self,
            request: prompt_runtime.NewSessionRunRequest,
        ) -> object:
            captured_calls.append(("new_session", request.tool_access.tool_policy))
            continuation = prompt_runtime.Continuation(
                selected_service="claude",
                selected_model="sonnet",
                selected_effort="medium",
                tool_access=request.tool_access,
                provider_resume_state={"provider_session_id": "session-123"},
            )
            return SimpleNamespace(
                kind="completed",
                output="start response",
                continuation=continuation,
                result=prompt_runtime.SessionRunResult(
                    output="start response",
                    runtime_metadata=prompt_runtime.SessionRuntimeMetadata(
                        service_name="claude",
                        provider_session_id="session-123",
                        run_kind=RunKind.FRESH,
                        session_namespace="",
                        exact_transcript_match=False,
                        selected_model="sonnet",
                        selected_effort="medium",
                        tool_policy=prompt_runtime.ToolPolicy.UNRESTRICTED,
                    ),
                    continuation=continuation,
                ),
                invocation_records=(),
            )

    def _fake_client() -> _FakeRuntimeClient:
        return _FakeRuntimeClient()

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(module, "RuntimeClient", _fake_client)
    try:
        run_result = module.run_live_smoke(
            provider_selection=("claude",),
            lifecycle_modes=("ephemeral", "new_session"),
            model_overrides={"claude": "sonnet"},
            effort_overrides={"claude": "medium"},
            tool_policies=(
                "NONE",
                "INSPECT_ONLY",
                "NO_FILE_MUTATION",
                "UNRESTRICTED",
            ),
            claude_code_oauth_token="token",
            run_id="tool-policy-matrix-run",
            artifact_root=tmp_path / "tool-policy-matrix",
        )
    finally:
        monkeypatch.undo()

    lifecycle_cases = [case for case in run_result.cases if case.policy is None]
    policy_cases = [case for case in run_result.cases if case.policy is not None]

    assert run_result.passed is True
    assert len(run_result.cases) == 6
    assert len(lifecycle_cases) == 2
    assert len(policy_cases) == 4
    assert all(case.mode == "ephemeral" for case in policy_cases)
    assert {case.mode for case in lifecycle_cases} == {"ephemeral", "new_session"}
    assert {str(case.policy) for case in policy_cases} == {
        "NONE",
        "INSPECT_ONLY",
        "NO_FILE_MUTATION",
        "UNRESTRICTED",
    }

    calls_by_mode = {
        "ephemeral": sum(1 for call, _ in captured_calls if call == "ephemeral"),
        "new_session": sum(1 for call, _ in captured_calls if call == "new_session"),
    }
    assert calls_by_mode["ephemeral"] == 5
    assert calls_by_mode["new_session"] == 1


def test_live_smoke_combined_mode_continues_after_lifecycle_provider_failure(
    smoke_module: object,
    tmp_path: Path,
) -> None:
    module: Any = smoke_module
    from agent_runtime import runtime as prompt_runtime
    from agent_runtime.contracts import ToolAccess
    from agent_runtime.session import RunKind

    calls: list[tuple[str, str, str | None]] = []

    codex_continuation = prompt_runtime.Continuation(
        selected_service="codex",
        selected_model="sonnet",
        selected_effort="medium",
        tool_access=ToolAccess.no_tools(),
        provider_resume_state={"provider_session_id": "codex-session"},
    )

    class _FakeRuntimeClient:
        def run_ephemeral(
            self,
            request: prompt_runtime.EphemeralRunRequest,
        ) -> object:
            calls.append(
                (
                    "ephemeral",
                    request.provider_selection.service,
                    str(request.tool_policy),
                )
            )
            return SimpleNamespace(
                kind="completed",
                output="provider output value",
                result=SimpleNamespace(
                    selected_service=request.provider_selection.service,
                    selected_model=request.provider_selection.model,
                    selected_effort=request.provider_selection.effort,
                    tool_access=SimpleNamespace(
                        tool_policy=request.tool_policy,
                    ),
                ),
                invocation_records=(),
            )

        def run_new_session(
            self,
            request: prompt_runtime.NewSessionRunRequest,
        ) -> object:
            calls.append(("new_session", request.provider_selection.service, None))
            if request.provider_selection.service == "claude":
                return prompt_runtime.RuntimeOutcome(
                    kind="usage_limited",
                    output="claude usage limit reached",
                    service_name="claude",
                    reset_time=None,
                    invocation_progress=prompt_runtime.InvocationProgress.NOT_STARTED,
                )

            return prompt_runtime.RuntimeOutcome(
                kind="completed",
                output="codex new session token codex-token",
                continuation=codex_continuation,
                result=prompt_runtime.SessionRunResult(
                    output="codex new session token codex-token",
                    runtime_metadata=prompt_runtime.SessionRuntimeMetadata(
                        service_name="codex",
                        provider_session_id="codex-session",
                        run_kind=RunKind.FRESH,
                        session_namespace="",
                        exact_transcript_match=False,
                        selected_model="sonnet",
                        selected_effort="medium",
                        tool_policy=prompt_runtime.ToolPolicy.UNRESTRICTED,
                    ),
                ),
                invocation_records=(),
            )

        def run_resumed_session(
            self,
            request: prompt_runtime.ResumedSessionRunRequest,
        ) -> object:
            continuation = cast(prompt_runtime.Continuation, request.continuation)
            selected_service = continuation.serialized_payload.service_name
            calls.append(("resumed_session", selected_service, None))
            return prompt_runtime.RuntimeOutcome(
                kind="completed",
                output="codex resumed token codex-token",
                continuation=cast(prompt_runtime.Continuation, request.continuation),
                result=prompt_runtime.SessionRunResult(
                    output="codex resumed token codex-token",
                    runtime_metadata=prompt_runtime.SessionRuntimeMetadata(
                        service_name=selected_service,
                        provider_session_id="codex-session",
                        run_kind=RunKind.RESUME,
                        session_namespace="",
                        exact_transcript_match=False,
                        selected_model="sonnet",
                        selected_effort="medium",
                        tool_policy=prompt_runtime.ToolPolicy.UNRESTRICTED,
                    ),
                ),
                invocation_records=(),
            )

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(module, "RuntimeClient", lambda: _FakeRuntimeClient())
    try:
        run_result = module.run_live_smoke(
            provider_selection=("claude", "codex"),
            lifecycle_modes=("new_session", "resumed_session"),
            model_overrides={"claude": "sonnet", "codex": "sonnet"},
            effort_overrides={"claude": "medium", "codex": "medium"},
            tool_policies=("UNRESTRICTED",),
            claude_code_oauth_token="token",
            opencode_api_key="api-key",
            codex_auth_present=True,
            run_id="combined-continue-run",
            artifact_root=tmp_path / "combined-continue",
        )
    finally:
        monkeypatch.undo()

    assert run_result.passed is False
    assert run_result.cases[0].service == "claude"
    assert run_result.cases[0].mode == "new_session"
    assert run_result.cases[0].status == "usage_limited"
    assert run_result.cases[1].service == "claude"
    assert run_result.cases[1].mode == "resumed_session"
    assert run_result.cases[1].status == "error"
    assert any(
        case.service == "claude"
        and case.mode == "ephemeral"
        and case.policy == "UNRESTRICTED"
        and case.status == "skipped"
        and case.required is False
        for case in run_result.cases
    )
    assert any(
        case.service == "codex"
        and case.mode == "new_session"
        and case.status == "passed"
        for case in run_result.cases
    )
    assert any(
        case.service == "codex"
        and case.mode == "resumed_session"
        and case.status == "passed"
        for case in run_result.cases
    )
    assert any(
        case.service == "codex"
        and case.mode == "ephemeral"
        and case.policy == "UNRESTRICTED"
        and case.status == "passed"
        for case in run_result.cases
    )
    assert any(call == ("new_session", "claude", None) for call in calls)
    assert any(call[0] == "new_session" and call[1] == "codex" for call in calls)
    assert any(call[0] == "resumed_session" and call[1] == "codex" for call in calls)


def test_public_smoke_case_resolves_async_session_runtime_methods(
    smoke_module: object,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module: Any = smoke_module
    from agent_runtime import runtime as prompt_runtime
    from agent_runtime.contracts import ToolAccess
    from agent_runtime.session import RunKind

    continuation = prompt_runtime.Continuation(
        selected_service="opencode",
        selected_model="deepseek-v4-flash",
        selected_effort="medium",
        tool_access=ToolAccess.no_tools(),
        provider_resume_state={"provider_session_id": "opencode-session"},
    )

    class _FakeRuntimeClient:
        async def run_new_session(
            self,
            request: prompt_runtime.NewSessionRunRequest,
        ) -> object:
            await asyncio.sleep(0)
            return prompt_runtime.RuntimeOutcome.completed(
                output="new session output",
                result=prompt_runtime.SessionRunResult(
                    output="new session output",
                    runtime_metadata=prompt_runtime.SessionRuntimeMetadata(
                        service_name=request.provider_selection.service,
                        provider_session_id="opencode-session",
                        run_kind=RunKind.FRESH,
                        session_namespace="",
                        exact_transcript_match=False,
                        selected_model=request.provider_selection.model,
                        selected_effort=request.provider_selection.effort,
                        tool_policy=request.tool_policy,
                    ),
                    continuation=continuation,
                ),
            )

        async def run_resumed_session(
            self,
            request: prompt_runtime.ResumedSessionRunRequest,
        ) -> object:
            await asyncio.sleep(0)
            return prompt_runtime.RuntimeOutcome.completed(
                output="resumed output",
                result=prompt_runtime.SessionRunResult(
                    output="resumed output",
                    runtime_metadata=prompt_runtime.SessionRuntimeMetadata(
                        service_name="opencode",
                        provider_session_id="opencode-session",
                        run_kind=RunKind.RESUME,
                        session_namespace="",
                        exact_transcript_match=False,
                        selected_model="deepseek-v4-flash",
                        selected_effort="medium",
                        tool_policy=prompt_runtime.ToolPolicy.UNRESTRICTED,
                    ),
                    continuation=continuation,
                ),
            )

    monkeypatch.setattr(module, "RuntimeClient", lambda: _FakeRuntimeClient())
    selection = prompt_runtime.ProviderSelection(
        service="opencode",
        model="deepseek-v4-flash",
        effort="medium",
        auth=prompt_runtime.ProviderAuth(opencode_api_key="api-key"),
    )

    new_session_outcome = module._run_public_smoke_case(
        case=SimpleNamespace(
            service="opencode",
            mode="new_session",
            policy=None,
            provider_selection=selection,
        ),
        artifact_dir=tmp_path / "new-session",
        prompt="prompt",
        env={},
        claude_code_oauth_token=None,
        opencode_api_key="api-key",
        codex_auth_present=False,
    )
    resumed_outcome = module._run_public_smoke_case(
        case=SimpleNamespace(
            service="opencode",
            mode="resumed_session",
            policy=None,
            provider_selection=selection,
        ),
        artifact_dir=tmp_path / "resumed-session",
        prompt="prompt",
        env={},
        claude_code_oauth_token=None,
        opencode_api_key="api-key",
        codex_auth_present=False,
        continuation=continuation,
    )

    assert new_session_outcome.kind == "completed"
    assert resumed_outcome.kind == "completed"


def test_live_smoke_single_tool_policy_request_reuses_ephemeral_only_path(
    smoke_module: object,
    tmp_path: Path,
) -> None:
    module: Any = smoke_module
    from agent_runtime import runtime as prompt_runtime

    captured_calls: list[Any] = []

    class _FakeRuntimeClient:
        def run_ephemeral(
            self,
            request: prompt_runtime.EphemeralRunRequest,
        ) -> object:
            captured_calls.append(request.tool_policy)
            return SimpleNamespace(
                kind="completed",
                output="provider output value",
                result=SimpleNamespace(
                    selected_service=request.provider_selection.service,
                    selected_model=request.provider_selection.model,
                    selected_effort=request.provider_selection.effort,
                    tool_access=SimpleNamespace(
                        tool_policy=request.tool_policy,
                    ),
                ),
                invocation_records=(),
            )

    def _fake_client() -> _FakeRuntimeClient:
        return _FakeRuntimeClient()

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(module, "RuntimeClient", _fake_client)
    try:
        run_result = module.run_live_smoke(
            provider_selection=("claude",),
            lifecycle_modes=("ephemeral",),
            model_overrides={"claude": "sonnet"},
            effort_overrides={"claude": "medium"},
            tool_policies=("NO_FILE_MUTATION",),
            claude_code_oauth_token="token",
            run_id="tool-policy-rerun-run",
            artifact_root=tmp_path / "tool-policy-rerun",
        )
    finally:
        monkeypatch.undo()

    assert run_result.passed is True
    assert len(run_result.cases) == 1
    assert run_result.cases[0].policy == "NO_FILE_MUTATION"
    assert run_result.cases[0].mode == "ephemeral"
    assert captured_calls == [prompt_runtime.ToolPolicy.NO_FILE_MUTATION]


def test_live_smoke_tool_policy_timeout_is_classified_as_failed_and_retained_for_matrix_case(
    smoke_module: object,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module: Any = smoke_module
    from agent_runtime import runtime as prompt_runtime

    class _FakeRuntimeClient:
        def run_ephemeral(
            self,
            request: prompt_runtime.EphemeralRunRequest,
        ) -> object:
            return prompt_runtime.RuntimeOutcome.timed_out(
                output="provider timeout",
                invocation_progress=prompt_runtime.InvocationProgress.STARTED,
            )

    monkeypatch.setattr(module, "RuntimeClient", lambda: _FakeRuntimeClient())

    run_result = module.run_live_smoke(
        provider_selection=("claude",),
        lifecycle_modes=("ephemeral",),
        model_overrides={"claude": "sonnet"},
        effort_overrides={"claude": "medium"},
        tool_policies=("NONE",),
        claude_code_oauth_token="token",
        run_id="tool-policy-timeout-run",
        artifact_root=tmp_path / "tool-policy-timeout",
    )

    assert run_result.passed is False
    assert run_result.cases[0].status == "failed"
    assert run_result.cases[0].diagnostic == "provider timeout"

    case_dir = (
        tmp_path
        / "tool-policy-timeout"
        / "tool-policy-timeout-run"
        / "claude"
        / "ephemeral"
        / "NONE"
    )
    assert (case_dir / "final_output.txt").exists()
    assert (case_dir / "final_output.txt").read_text(
        encoding="utf-8"
    ) == "provider timeout"
    assert (
        "provider timeout"
        in (
            module.json.loads(
                (
                    tmp_path
                    / "tool-policy-timeout"
                    / "tool-policy-timeout-run"
                    / "summary.json"
                ).read_text(encoding="utf-8")
            )["cases"][0]["diagnostic"]
        )
    )


def test_live_smoke_tool_policy_exception_is_classified_and_warned(
    smoke_module: object,
    tmp_path: Path,
) -> None:
    module: Any = smoke_module
    from agent_runtime import runtime as prompt_runtime

    class _FakeRuntimeClient:
        def run_ephemeral(
            self,
            request: prompt_runtime.EphemeralRunRequest,
        ) -> object:
            raise RuntimeError("provider runtime client crashed")

    def _fake_client() -> _FakeRuntimeClient:
        return _FakeRuntimeClient()

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(module, "RuntimeClient", _fake_client)
    try:
        run_result = module.run_live_smoke(
            provider_selection=("claude",),
            lifecycle_modes=("ephemeral",),
            model_overrides={"claude": "sonnet"},
            effort_overrides={"claude": "medium"},
            tool_policies=("UNRESTRICTED",),
            claude_code_oauth_token="token",
            run_id="tool-policy-exception-run",
            artifact_root=tmp_path / "tool-policy-exception",
        )
    finally:
        monkeypatch.undo()

    assert run_result.passed is False
    assert run_result.cases[0].status == "error"
    assert run_result.cases[0].diagnostic == "provider runtime client crashed"
    assert any(
        "runner exception for claude/ephemeral" in warning
        for warning in run_result.warnings
    )
    summary_payload = module.json.loads(
        (
            tmp_path
            / "tool-policy-exception"
            / "tool-policy-exception-run"
            / "summary.json"
        ).read_text(encoding="utf-8")
    )
    assert summary_payload["cases"][0]["status"] == "error"
    assert (
        summary_payload["cases"][0]["diagnostic"] == "provider runtime client crashed"
    )
    assert summary_payload["cases"][0]["required"] is True


def test_live_smoke_explicit_provider_config_error_does_not_pass_empty_run(
    smoke_module: object,
    tmp_path: Path,
) -> None:
    module: Any = smoke_module

    run_result = module.run_live_smoke(
        provider_selection=("claude",),
        lifecycle_modes=("ephemeral",),
        run_id="missing-config-run",
        artifact_root=tmp_path / "missing-config",
        env={},
    )

    assert run_result.passed is False
    assert run_result.summary_written is True
    assert run_result.cases == ()
    assert "no runnable smoke cases planned" in run_result.warnings

    summary_payload = module.json.loads(run_result.summary_path.read_text("utf-8"))
    assert summary_payload["case_count"] == 0
    assert summary_payload["provider_plans"][0]["service"] == "claude"
    assert summary_payload["provider_plans"][0]["status"] == "config_error"


def test_live_smoke_all_selection_with_no_configured_providers_does_not_pass_empty_run(
    smoke_module: object,
    tmp_path: Path,
) -> None:
    module: Any = smoke_module

    run_result = module.run_live_smoke(
        provider_selection="all",
        lifecycle_modes=("ephemeral",),
        run_id="all-missing-config-run",
        artifact_root=tmp_path / "all-missing-config",
        env={},
    )

    assert run_result.passed is False
    assert run_result.summary_written is True
    assert run_result.cases == ()
    assert "no runnable smoke cases planned" in run_result.warnings

    summary_payload = module.json.loads(run_result.summary_path.read_text("utf-8"))
    assert summary_payload["case_count"] == 0
    assert {plan["status"] for plan in summary_payload["provider_plans"]} == {"skipped"}


def test_live_smoke_cli_help_is_invokable(smoke_module: object) -> None:
    module: Any = smoke_module

    output = io.StringIO()
    with redirect_stdout(output), pytest.raises(SystemExit) as excinfo:
        module.main(["--help"])

    parser_help = output.getvalue()

    assert excinfo.value.code == 0
    assert parser_help.strip()
    assert "--dry-run" not in parser_help
    assert "--list-providers" not in parser_help


def test_live_smoke_cli_help_no_longer_mentions_environment_sources_for_model_or_effort(
    smoke_module: object,
) -> None:
    module: Any = smoke_module

    output = io.StringIO()
    with redirect_stdout(output), pytest.raises(SystemExit):
        module.main(["--help"])

    parser_help = output.getvalue()
    assert "provider-specific environment variable" not in parser_help
    assert (
        module.live_provider_smoke_plan.LIVE_SMOKE_CLAUDE_MODEL_ENV not in parser_help
    )
    assert (
        module.live_provider_smoke_plan.LIVE_SMOKE_CLAUDE_EFFORT_ENV not in parser_help
    )
    assert module.live_provider_smoke_plan.LIVE_SMOKE_CODEX_MODEL_ENV not in parser_help
    assert (
        module.live_provider_smoke_plan.LIVE_SMOKE_CODEX_EFFORT_ENV not in parser_help
    )
    assert (
        module.live_provider_smoke_plan.LIVE_SMOKE_OPENCODE_MODEL_ENV not in parser_help
    )
    assert (
        module.live_provider_smoke_plan.LIVE_SMOKE_OPENCODE_EFFORT_ENV
        not in parser_help
    )


def test_live_smoke_direct_help_invocation_succeeds_and_skips_default_artifacts(
    tmp_path: Path,
) -> None:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0
    assert proc.stdout.strip()
    assert not (tmp_path / "live-smoke-artifacts").exists()


def test_live_smoke_direct_list_providers_invocation_is_rejected(
    tmp_path: Path,
) -> None:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--list-providers"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert proc.returncode != 0
    assert "unrecognized arguments: --list-providers" in proc.stderr
    assert not (tmp_path / "live-smoke-artifacts").exists()


def test_live_smoke_direct_dry_run_invocation_is_rejected(
    tmp_path: Path,
) -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--dry-run",
            "--provider",
            "claude",
            "--mode",
            "new_session",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert proc.returncode != 0
    assert "unrecognized arguments: --dry-run" in proc.stderr
    assert not (tmp_path / "live-smoke-artifacts").exists()


def test_live_smoke_explicit_provider_config_error_reports_missing_setup_on_run(
    tmp_path: Path,
) -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--provider",
            "claude",
            "--mode",
            "ephemeral",
            "--json",
            "--run-id",
            "missing-setup-run",
            "--artifact-root",
            str(tmp_path / "missing-setup-artifacts"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    payload = json.loads(proc.stdout)

    assert proc.returncode == 1
    assert payload["run_id"] == "missing-setup-run"
    assert payload["provider_plans"][0]["status"] == "config_error"
    assert payload["cases"] == []
    assert any("provider not configured" in warning for warning in payload["warnings"])
def test_live_smoke_cli_default_console_output_reports_provider_mode_and_artifacts(
    smoke_module: object,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module: Any = smoke_module

    result = module.LiveSmokeRunResult(
        run_id="cli-default-run",
        artifact_root=tmp_path / "artifacts",
        summary_path=tmp_path / "artifacts" / "cli-default-run" / "summary.json",
        summary_written=True,
        passed=True,
        cases=(
            module.LiveSmokeRunCaseResult(
                service="codex",
                mode="ephemeral",
                policy=None,
                model="codex-mini",
                effort="high",
                artifact_path=str(
                    tmp_path
                    / "artifacts"
                    / "cli-default-run"
                    / "codex"
                    / "ephemeral"
                    / "default"
                ),
                status="passed",
                required=True,
                provider_output="provider output",
                diagnostic=None,
                traceback=None,
                duration_seconds=0.2,
            ),
        ),
    )

    received: dict[str, Any] = {}

    def _fake_runner(*args: Any, **kwargs: Any) -> module.LiveSmokeRunResult:
        assert kwargs["provider_selection"] == ("codex",)
        assert kwargs["lifecycle_modes"] == ("ephemeral",)
        received["provider_selection"] = kwargs["provider_selection"]
        received["lifecycle_modes"] = kwargs["lifecycle_modes"]
        received["run_id"] = kwargs["run_id"]
        received["artifact_root"] = kwargs["artifact_root"]
        return result

    monkeypatch.setattr(module, "run_live_smoke", _fake_runner)

    output = io.StringIO()
    with redirect_stdout(output):
        exit_code = module.main(
            [
                "--provider",
                "codex",
                "--mode",
                "ephemeral",
                "--run-id",
                "cli-default-run",
                "--artifact-root",
                str(tmp_path / "artifacts"),
            ]
        )

    stdout = output.getvalue()

    assert exit_code == 0
    assert received["provider_selection"] == ("codex",)
    assert received["lifecycle_modes"] == ("ephemeral",)
    assert received["artifact_root"] == tmp_path / "artifacts"
    assert str(tmp_path / "artifacts") in stdout
    assert "codex/ephemeral" in stdout
    assert "passed" in stdout.lower()
    assert "final status: passed" in stdout.lower()


def test_live_smoke_cli_verbose_output_exposes_rerun_context_without_credentials(
    smoke_module: object,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module: Any = smoke_module

    result = module.LiveSmokeRunResult(
        run_id="cli-verbose-run",
        artifact_root=tmp_path / "artifacts",
        summary_path=tmp_path / "artifacts" / "cli-verbose-run" / "summary.json",
        summary_written=True,
        passed=False,
        cases=(
            module.LiveSmokeRunCaseResult(
                service="claude",
                mode="new_session",
                policy="NONE",
                model="sonnet",
                effort="medium",
                artifact_path=str(
                    tmp_path
                    / "artifacts"
                    / "cli-verbose-run"
                    / "claude"
                    / "new_session"
                    / "NONE"
                ),
                status="failed",
                required=True,
                provider_output="provider stream: token=super-secret-token",
                diagnostic="usage_limited output contained retry guidance",
                traceback="Traceback: failure details",
                duration_seconds=3.4,
            ),
        ),
        warnings=("optional case artifact write warning",),
    )

    monkeypatch.setattr(module, "run_live_smoke", lambda **kwargs: result)

    output = io.StringIO()
    with redirect_stdout(output):
        exit_code = module.main(
            [
                "--provider",
                "claude",
                "--mode",
                "new_session",
                "--policy",
                "NONE",
                "--verbose",
                "--run-id",
                "cli-verbose-run",
            ]
        )

    stdout = output.getvalue()

    assert exit_code == 1
    assert "claude/new_session/NONE" in stdout
    assert "usage_limited output contained retry guidance" in stdout
    assert "optional case artifact write warning" in stdout
    assert "provider stream" not in stdout
    assert "super-secret-token" not in stdout
    assert "to rerun failed cases" in stdout.lower()


def test_live_smoke_cli_json_output_includes_rerun_targets_for_failed_cases(
    smoke_module: object,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module: Any = smoke_module

    result = module.LiveSmokeRunResult(
        run_id="cli-json-run",
        artifact_root=tmp_path / "artifacts",
        summary_path=tmp_path / "artifacts" / "cli-json-run" / "summary.json",
        summary_written=True,
        passed=False,
        cases=(
            module.LiveSmokeRunCaseResult(
                service="opencode",
                mode="ephemeral",
                policy="UNRESTRICTED",
                model="deepseek",
                effort="medium",
                artifact_path=str(
                    tmp_path
                    / "artifacts"
                    / "cli-json-run"
                    / "opencode"
                    / "ephemeral"
                    / "UNRESTRICTED"
                ),
                status="failed",
                required=True,
                provider_output="provider output",
                diagnostic="provider runtime returned failed",
                traceback=None,
                duration_seconds=1.2,
            ),
        ),
    )

    def _fake_runner(
        *args: Any, case_runner: Any = None, **_: Any
    ) -> module.LiveSmokeRunResult:
        assert not args
        assert case_runner is None
        return result

    monkeypatch.setattr(module, "run_live_smoke", _fake_runner)

    output = io.StringIO()
    with redirect_stdout(output):
        exit_code = module.main(
            [
                "--provider",
                "opencode",
                "--mode",
                "ephemeral",
                "--policy",
                "UNRESTRICTED",
                "--json",
                "--run-id",
                "cli-json-run",
                "--artifact-root",
                str(tmp_path / "artifacts"),
            ]
        )

    stdout = output.getvalue()
    payload = module.json.loads(stdout)

    assert exit_code == 1
    assert payload["run_id"] == "cli-json-run"
    assert payload["artifact_root"] == module._portable_json_path(
        tmp_path / "artifacts"
    )
    assert payload["provider_plans"][0]["status"] == "runnable"
    assert payload["cases"][0]["service"] == "opencode"
    assert payload["cases"][0]["status"] == "failed"
    assert any(
        item["provider"] == "opencode"
        and item["mode"] == "ephemeral"
        and item["policy"] == "UNRESTRICTED"
        for item in payload["failed_case_runs"]
    )
    assert any(
        "--provider opencode" in item["command"] for item in payload["failed_case_runs"]
    )


def test_live_smoke_real_run_json_artifact_paths_use_forward_slashes_portably(
    smoke_module: object,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module: Any = smoke_module
    monkeypatch.chdir(tmp_path)

    def _fake_case_runner(*, artifact_dir: Path, **_: object) -> _FakeRunOutcome:
        assert artifact_dir == (
            tmp_path
            / r"portable\artifacts"
            / "portable-json-run"
            / "codex"
            / "ephemeral"
            / "default"
        )
        return _FakeRunOutcome(kind="completed", output="ok")

    result = module.run_live_smoke(
        provider_selection=("codex",),
        lifecycle_modes=("ephemeral",),
        model_overrides={"codex": "codex-mini"},
        effort_overrides={"codex": "high"},
        codex_auth_present=True,
        run_id="portable-json-run",
        artifact_root=r"portable\artifacts",
        case_runner=_fake_case_runner,
    )

    assert result.artifact_root == (tmp_path / r"portable\artifacts").resolve()
    assert result.cases[0].artifact_path == str(
        tmp_path
        / r"portable\artifacts"
        / "portable-json-run"
        / "codex"
        / "ephemeral"
        / "default"
    )
    assert Path(result.cases[0].artifact_path).exists()

    summary_payload = module.json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary_payload["artifact_root"] == module._portable_json_path(
        (tmp_path / "portable" / "artifacts").resolve()
    )
    assert summary_payload["cases"][0]["artifact_path"] == module._portable_json_path(
        (
            tmp_path
            / "portable"
            / "artifacts"
            / "portable-json-run"
            / "codex"
            / "ephemeral"
            / "default"
        ).resolve()
    )

    monkeypatch.setattr(module, "run_live_smoke", lambda **_: result)
    output = io.StringIO()
    with redirect_stdout(output):
        exit_code = module.main(
            [
                "--provider",
                "codex",
                "--mode",
                "ephemeral",
                "--json",
                "--run-id",
                "portable-json-run",
                "--artifact-root",
                r"portable\artifacts",
            ]
        )

    payload = module.json.loads(output.getvalue())
    assert exit_code == 0
    assert payload["artifact_root"] == module._portable_json_path(
        (tmp_path / "portable" / "artifacts").resolve()
    )
    assert payload["cases"][0]["artifact_path"] == module._portable_json_path(
        (
            tmp_path
            / "portable"
            / "artifacts"
            / "portable-json-run"
            / "codex"
            / "ephemeral"
            / "default"
        ).resolve()
    )


def test_live_smoke_cli_warning_output_does_not_flip_pass_status(
    smoke_module: object,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module: Any = smoke_module

    result = module.LiveSmokeRunResult(
        run_id="cli-warning-run",
        artifact_root=tmp_path / "artifacts",
        summary_path=tmp_path / "artifacts" / "cli-warning-run" / "summary.json",
        summary_written=True,
        passed=True,
        cases=(
            module.LiveSmokeRunCaseResult(
                service="codex",
                mode="ephemeral",
                policy=None,
                model="codex-mini",
                effort="high",
                artifact_path=str(
                    tmp_path
                    / "artifacts"
                    / "cli-warning-run"
                    / "codex"
                    / "ephemeral"
                    / "default"
                ),
                status="passed",
                required=True,
                provider_output="",
                diagnostic=None,
                traceback=None,
                duration_seconds=0.1,
            ),
        ),
        warnings=("optional case artifact write failed: transient fs issue",),
    )

    monkeypatch.setattr(module, "run_live_smoke", lambda **_: result)

    output = io.StringIO()
    with redirect_stdout(output):
        exit_code = module.main(
            [
                "--provider",
                "codex",
                "--mode",
                "ephemeral",
                "--run-id",
                "cli-warning-run",
                "--verbose",
            ]
        )

    stdout = output.getvalue()
    assert exit_code == 0
    assert "optional case artifact write failed" in stdout
    assert "final status: passed" in stdout.lower()


def test_main_with_explicit_empty_argv_ignores_process_argv_and_uses_defaults(
    smoke_module: object,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module: Any = smoke_module

    result = module.LiveSmokeRunResult(
        run_id="explicit-empty-argv",
        artifact_root=tmp_path / "artifacts",
        summary_path=tmp_path / "artifacts" / "explicit-empty-argv" / "summary.json",
        summary_written=True,
        passed=True,
        cases=(),
    )

    captured: dict[str, Any] = {}

    def _fake_runner(**kwargs: Any) -> module.LiveSmokeRunResult:
        captured.update(kwargs)
        return result

    monkeypatch.setattr(module, "run_live_smoke", _fake_runner)
    monkeypatch.setattr(sys, "argv", ["live_provider_smoke.py", "--help"])

    exit_code = module.main([])

    assert exit_code == 0
    assert captured["provider_selection"] == ("all",)
    assert captured["lifecycle_modes"] == (
        "ephemeral",
        "new_session",
        "resumed_session",
    )
    assert captured["tool_policies"] == ()


def test_live_smoke_default_case_timeout_is_full_matrix_friendly(
    smoke_module: Any,
) -> None:
    module = smoke_module

    assert module._DEFAULT_CASE_TIMEOUT_SECONDS >= 180
