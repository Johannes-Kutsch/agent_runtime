from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

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
