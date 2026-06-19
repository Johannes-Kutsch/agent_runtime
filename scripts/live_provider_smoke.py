from __future__ import annotations

import json
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping
from traceback import format_exc
from importlib import util as importlib_util

try:
    import live_provider_smoke_plan
except ModuleNotFoundError:
    _plan_spec = importlib_util.spec_from_file_location(
        "live_provider_smoke_plan",
        str(Path(__file__).resolve().parent / "live_provider_smoke_plan.py"),
    )
    assert _plan_spec is not None
    assert _plan_spec.loader is not None
    live_provider_smoke_plan = ModuleType("live_provider_smoke_plan")
    sys.modules["live_provider_smoke_plan"] = live_provider_smoke_plan
    _plan_spec.loader.exec_module(live_provider_smoke_plan)


DEFAULT_LIVE_SMOKE_ARTIFACT_ROOT = "live-smoke-artifacts"
LIVE_SMOKE_SUMMARY_FILENAME = "summary.json"


@dataclass(frozen=True)
class LiveSmokeRunCaseResult:
    service: str
    mode: str
    policy: str | None
    model: str
    effort: str
    artifact_path: str
    status: str
    provider_output: str
    traceback: str | None
    duration_seconds: float


@dataclass(frozen=True)
class LiveSmokeRunResult:
    run_id: str
    artifact_root: Path
    summary_path: Path
    summary_written: bool
    passed: bool
    cases: tuple[LiveSmokeRunCaseResult, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class _StubOutcome:
    kind: str = "completed"
    output: str = ""
    live_turns: tuple[Any, ...] = ()
    invocation_records: tuple[Any, ...] = ()


def _resolve_live_smoke_artifact_root(
    artifact_root: Path | str | None,
) -> Path:
    return Path(artifact_root or DEFAULT_LIVE_SMOKE_ARTIFACT_ROOT).resolve()


def _build_summary_payload(
    run_result: LiveSmokeRunResult,
    run_duration_seconds: float,
    provider_plans: Any,
) -> dict[str, Any]:
    return {
        "run_id": run_result.run_id,
        "artifact_root": str(run_result.artifact_root),
        "summary_path": str(run_result.summary_path),
        "passed": run_result.passed,
        "duration_seconds": run_duration_seconds,
        "case_count": len(run_result.cases),
        "cases": [
            {
                "service": case.service,
                "mode": case.mode,
                "policy": case.policy,
                "model": case.model,
                "effort": case.effort,
                "artifact_path": case.artifact_path,
                "status": case.status,
                "provider_output": case.provider_output,
                "traceback": case.traceback,
                "duration_seconds": case.duration_seconds,
            }
            for case in run_result.cases
        ],
        "provider_plans": [
            {
                "service": plan.service,
                "status": plan.status,
                "model": plan.model,
                "effort": plan.effort,
            }
            for plan in provider_plans
        ],
        "warnings": list(run_result.warnings),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _write_required_summary(
    summary_payload: dict[str, Any], summary_path: Path
) -> bool:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary_payload, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    return summary_path.exists()


def _build_case_prompt(run_id: str, planned_case: Any) -> str:
    policy = planned_case.policy or "default"
    return f"{run_id}:{planned_case.service}:{planned_case.mode}:{policy}"


def _serialize_case_invocation_records(invocation_records: Any) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for record in invocation_records:
        serialized.append(
            {
                "run_kind": getattr(record, "run_kind", None),
                "service_name": getattr(record, "service_name", None),
                "provider_session_id": getattr(record, "provider_session_id", None),
                "prompt": getattr(record, "prompt", None),
                "provider_output": getattr(record, "provider_output", None),
                "usage": getattr(record, "usage", None),
            }
        )
    return serialized


def _serialize_live_turns(case_outcome: Any) -> list[dict[str, str]]:
    live_turns: list[dict[str, str]] = []
    for turn in getattr(case_outcome, "live_turns", ()):  # pragma: no branch
        live_turns.append(
            {
                "text": str(getattr(turn, "text", "")),
                "service_name": str(getattr(turn, "service_name", "")),
            }
        )
    return live_turns


def _serialize_case_output(case_outcome: Any) -> dict[str, Any]:
    return {
        "kind": getattr(case_outcome, "kind", "unknown"),
        "output": getattr(case_outcome, "output", ""),
        "service_name": getattr(case_outcome, "service_name", None),
        "account_label": getattr(case_outcome, "account_label", None),
        "invocation_records": _serialize_case_invocation_records(
            getattr(case_outcome, "invocation_records", ())
        ),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, default=str),
        encoding="utf-8",
    )


def _write_optional_case_artifacts(
    planned_case: Any,
    case_outcome: Any,
    artifact_dir: Path,
    prompt: str,
    duration_seconds: float,
    run_id: str,
    traceback_text: str | None = None,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    (artifact_dir / "final_output.txt").write_text(
        str(getattr(case_outcome, "output", "")),
        encoding="utf-8",
    )
    _write_json(artifact_dir / "live_turns.json", _serialize_live_turns(case_outcome))
    _write_json(artifact_dir / "outcome.json", _serialize_case_output(case_outcome))
    _write_json(
        artifact_dir / "invocation_records.json",
        _serialize_case_invocation_records(
            getattr(case_outcome, "invocation_records", ())
        ),
    )
    _write_json(
        artifact_dir / "timings.json",
        {
            "run_id": run_id,
            "service": planned_case.service,
            "mode": planned_case.mode,
            "policy": planned_case.policy,
            "duration_seconds": duration_seconds,
        },
    )
    if traceback_text is not None:
        (artifact_dir / "traceback.txt").write_text(traceback_text, encoding="utf-8")


def _write_optional_config_artifacts(
    summary_root: Path,
    run_id: str,
    dry_run_plan: Any,
) -> None:
    config_path = summary_root / run_id / "config_summary.json"
    _write_json(
        config_path,
        {
            "run_id": run_id,
            "provider_selection": [
                plan.service for plan in dry_run_plan.provider_plans
            ],
            "cases": [
                {
                    "service": plan.service,
                    "status": plan.status,
                    "model": plan.model,
                    "effort": plan.effort,
                }
                for plan in dry_run_plan.provider_plans
            ],
            "artifact_root": str(summary_root / run_id),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def _write_artifacts_marker(summary_root: Path, run_id: str) -> None:
    marker_path = summary_root / run_id / "ARTIFACTS.md"
    marker_path.write_text(
        "Potentially sensitive artifacts.\n"
        "These files may include raw provider output, prompts, and diagnostics.\n"
        "Review before sharing.",
        encoding="utf-8",
    )


def _build_summary_payload_path(artifact_root: Path, run_id: str) -> Path:
    return artifact_root / run_id / LIVE_SMOKE_SUMMARY_FILENAME


def run_live_smoke(
    provider_selection: str | tuple[str, ...],
    *,
    lifecycle_modes: tuple[str, ...],
    tool_policies: tuple[str, ...] = (),
    run_id: str | None = None,
    model_overrides: Mapping[str, str] | None = None,
    effort_overrides: Mapping[str, str] | None = None,
    artifact_root: Path | str | None = None,
    cleanup_artifact_root: bool = False,
    case_runner: Callable[..., Any] | None = None,
    env: Mapping[str, str] | None = None,
    claude_code_oauth_token: str | None = None,
    opencode_api_key: str | None = None,
    codex_auth_present: bool | None = None,
) -> LiveSmokeRunResult:
    resolved_artifact_root = _resolve_live_smoke_artifact_root(artifact_root)
    if cleanup_artifact_root:
        shutil.rmtree(resolved_artifact_root, ignore_errors=True)
    resolved_artifact_root.mkdir(parents=True, exist_ok=True)

    dry_run_plan = live_provider_smoke_plan.build_dry_run_plan(
        provider_selection,
        lifecycle_modes=lifecycle_modes,
        tool_policies=tool_policies,
        run_id=run_id,
        model_overrides=model_overrides,
        effort_overrides=effort_overrides,
        artifact_root=resolved_artifact_root,
        env=env,
        claude_code_oauth_token=claude_code_oauth_token,
        opencode_api_key=opencode_api_key,
        codex_auth_present=codex_auth_present,
    )

    warnings: list[str] = []
    case_results: list[LiveSmokeRunCaseResult] = []
    runner = case_runner or _run_case_stub
    run_started = time.perf_counter()

    for planned_case in dry_run_plan.cases:
        invocation_dir = (
            resolved_artifact_root
            / dry_run_plan.run_id
            / planned_case.service
            / planned_case.mode
            / (planned_case.policy or "default")
        )
        prompt = _build_case_prompt(dry_run_plan.run_id, planned_case)
        case_started = time.perf_counter()
        case_traceback: str | None = None

        try:
            case_outcome = runner(
                case=planned_case,
                artifact_dir=invocation_dir,
                run_id=dry_run_plan.run_id,
                prompt=prompt,
                model=planned_case.model,
                effort=planned_case.effort,
                env=env,
                claude_code_oauth_token=claude_code_oauth_token,
                opencode_api_key=opencode_api_key,
                codex_auth_present=codex_auth_present,
            )
            status = str(getattr(case_outcome, "kind", "completed"))
            provider_output = str(getattr(case_outcome, "output", ""))
        except Exception:
            case_outcome = _StubOutcome(kind="failed")
            status = "failed"
            provider_output = ""
            case_traceback = format_exc()
            warnings.append(
                f"runner exception for {planned_case.service}/{planned_case.mode}"
            )

        case_duration = time.perf_counter() - case_started
        case_results.append(
            LiveSmokeRunCaseResult(
                service=planned_case.service,
                mode=planned_case.mode,
                policy=planned_case.policy,
                model=planned_case.model,
                effort=planned_case.effort,
                artifact_path=str(invocation_dir),
                status=status,
                provider_output=provider_output,
                traceback=case_traceback,
                duration_seconds=case_duration,
            )
        )

        try:
            _write_optional_case_artifacts(
                planned_case=planned_case,
                case_outcome=case_outcome,
                artifact_dir=invocation_dir,
                prompt=prompt,
                duration_seconds=case_duration,
                run_id=dry_run_plan.run_id,
                traceback_text=case_traceback,
            )
        except Exception as exc:
            warnings.append(
                "optional case artifact write failed for "
                f"{planned_case.service}/{planned_case.mode}: {exc}"
            )

    summary_path = _build_summary_payload_path(
        resolved_artifact_root, dry_run_plan.run_id
    )
    run_result = LiveSmokeRunResult(
        run_id=dry_run_plan.run_id,
        artifact_root=resolved_artifact_root,
        summary_path=summary_path,
        summary_written=False,
        passed=all(case.status == "completed" for case in case_results),
        cases=tuple(case_results),
        warnings=tuple(warnings),
    )
    summary_payload = _build_summary_payload(
        run_result,
        time.perf_counter() - run_started,
        dry_run_plan.provider_plans,
    )

    summary_written = False
    try:
        summary_written = _write_required_summary(summary_payload, summary_path)
    except Exception:
        summary_written = False
        warnings.append("required summary write failed")

    try:
        _write_optional_config_artifacts(
            summary_root=resolved_artifact_root,
            run_id=dry_run_plan.run_id,
            dry_run_plan=dry_run_plan,
        )
    except Exception:
        warnings.append("optional config summary write failed")
    try:
        _write_artifacts_marker(
            summary_root=resolved_artifact_root,
            run_id=dry_run_plan.run_id,
        )
    except Exception:
        warnings.append("optional artifacts marker write failed")

    return LiveSmokeRunResult(
        run_id=run_result.run_id,
        artifact_root=run_result.artifact_root,
        summary_path=summary_path,
        summary_written=summary_written,
        passed=run_result.passed and summary_written,
        cases=run_result.cases,
        warnings=tuple(warnings),
    )


def _run_case_stub(*_: Any, **__: Any) -> Any:
    return _StubOutcome()
