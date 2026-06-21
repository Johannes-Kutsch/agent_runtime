from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping, Sequence, cast
from traceback import format_exc
from importlib import util as importlib_util

from agent_runtime.runtime import (
    Continuation,
    EphemeralRunRequest,
    NewSessionRunRequest,
    ProviderAuth,
    ProviderSelection,
    ResumedSessionRunRequest,
    RuntimeClient,
    ToolPolicy,
)
from agent_runtime.errors import RuntimeConfigurationError

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
_DEFAULT_CASE_TIMEOUT_SECONDS = 180
_LIVE_SMOKE_ENV_PATH = Path(__file__).resolve().parent / "live-smoke" / ".env"
_LIVE_SMOKE_ENV_PATH_ENV = "LIVE_SMOKE_ENV_PATH"


def _parse_service_map_arg(value: str, *, flag_name: str) -> tuple[str, str]:
    provider, _, payload = value.partition("=")
    if not provider or not payload:
        raise argparse.ArgumentTypeError(
            f"{flag_name} expects SERVICE=VALUE (example: claude=sonnet), got {value!r}"
        )
    return provider.strip(), payload.strip()


def _safe_list(value: Sequence[str]) -> str:
    return ", ".join(value)


def _portable_json_path(path: Path | str) -> str:
    return str(path).replace("\\", "/")


def _parse_live_smoke_env_lines(
    text: str,
    *,
    env_path: Path,
) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if "=" not in line:
            raise RuntimeConfigurationError(
                f"Malformed line in {env_path} at {line_number}: expected KEY=value"
            )
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            raise RuntimeConfigurationError(
                f"Malformed line in {env_path} at {line_number}: expected KEY=value"
            )
        values[key] = value.strip()
    return values


def _load_live_smoke_env(env_path: Path | None = None) -> dict[str, str]:
    path = env_path or Path(
        os.environ.get(_LIVE_SMOKE_ENV_PATH_ENV, str(_LIVE_SMOKE_ENV_PATH))
    )
    if not path.exists():
        return {}
    return _parse_live_smoke_env_lines(path.read_text(encoding="utf-8"), env_path=path)


def _resolve_live_smoke_env(env: Mapping[str, str] | None) -> Mapping[str, str]:
    if env is not None:
        return env
    return _load_live_smoke_env()


def _live_smoke_defaults_help_text() -> str:
    defaults = live_provider_smoke_plan.LIVE_SMOKE_DEFAULTS
    verified_on = live_provider_smoke_plan.LIVE_SMOKE_DEFAULTS_VERIFIED_ON
    default_tuples = ", ".join(
        f"{service}={defaults[service][0]}/{defaults[service][1]}"
        for service in live_provider_smoke_plan.SUPPORTED_PROVIDERS
    )
    return (
        "Model source precedence: CLI override, then Live Smoke Default.\n"
        "Effort source precedence: CLI override, then Live Smoke Default.\n"
        f"Live Smoke Defaults: {default_tuples}. "
        f"Verified {verified_on}."
    )


def _build_case_rerun_command(
    case: Any,
) -> str:
    def _stringify_command(parts: list[str]) -> str:
        if os.name == "nt":
            return subprocess.list2cmdline(parts)
        return " ".join(shlex.quote(part) for part in parts)

    def _command_parts_for_mode(mode: str) -> list[str]:
        command_parts = [
            "python",
            __file__,
            "--provider",
            case.service,
            "--mode",
            mode,
        ]
        if case.policy is not None and mode == case.mode:
            command_parts.extend(["--policy", str(case.policy)])
        if case.model:
            command_parts.extend(["--model", f"{case.service}={case.model}"])
        if case.effort:
            command_parts.extend(["--effort", f"{case.service}={case.effort}"])
        return command_parts

    command_parts = _command_parts_for_mode(case.mode)
    if case.mode == "resumed_session":
        prerequisite_parts = _command_parts_for_mode("new_session")
        return "\n".join(
            (
                "Start Session Run prerequisite:",
                _stringify_command(prerequisite_parts),
                "Resume Session Run:",
                _stringify_command(command_parts),
            )
        )
    return _stringify_command(command_parts)


def _build_summary_payload_with_reruns(
    run_result: LiveSmokeRunResult,
    run_duration_seconds: float,
    provider_plans: Any,
    *,
    failed_case_runs: Sequence[dict[str, str | None]],
) -> dict[str, Any]:
    payload = _build_summary_payload(run_result, run_duration_seconds, provider_plans)
    payload["failed_case_runs"] = list(failed_case_runs)
    return payload


def _case_label(case: LiveSmokeRunCaseResult) -> str:
    if case.policy is None:
        return f"{case.service}/{case.mode}"
    return f"{case.service}/{case.mode}/{case.policy}"


def _build_cli_summary_payload(
    run_result: LiveSmokeRunResult,
    provider_plans: Any = (),
    *,
    failed_case_runs: Sequence[dict[str, str | None]] = (),
) -> dict[str, Any]:
    if not provider_plans:
        provider_statuses: dict[str, dict[str, Any]] = {}
        for case in run_result.cases:
            provider_statuses.setdefault(
                case.service,
                {
                    "service": case.service,
                    "status": "runnable",
                    "model": case.model,
                    "effort": case.effort,
                },
            )
        provider_plans = tuple(provider_statuses.values())
    payload = _build_summary_payload(run_result, 0.0, provider_plans)
    payload["failed_case_runs"] = list(failed_case_runs)
    return payload


def _format_rerun_block(
    failed_case_runs: Sequence[dict[str, str | None]],
) -> str:
    if not failed_case_runs:
        return ""
    lines = ["To rerun failed cases:"]
    for entry in failed_case_runs:
        command = entry.get("command")
        if command:
            provider = str(entry.get("provider", "provider"))
            mode = str(entry.get("mode", "mode"))
            policy = entry.get("policy")
            label = f"{provider}/{mode}"
            if policy is not None:
                label = f"{label}/{policy}"
            command_lines = str(command).splitlines()
            lines.append(f"  - {label}: {command_lines[0]}")
            lines.extend(f"    {line}" for line in command_lines[1:])
    return "\n".join(lines)


@dataclass(frozen=True)
class LiveSmokeRunCaseResult:
    service: str
    mode: str
    policy: str | None
    model: str
    effort: str
    artifact_path: str
    status: str
    required: bool
    provider_output: str
    diagnostic: str | None
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


@dataclass(frozen=True)
class _ObservedRuntimeOutcome:
    kind: str
    output: str = ""
    live_turns: tuple[Any, ...] = ()
    invocation_records: tuple[Any, ...] = ()
    result: Any | None = None
    service_name: str | None = None
    account_label: str | None = None
    reset_time: Any | None = None
    invocation_progress: Any | None = None
    continuation: Any | None = None
    usage: Any | None = None


def _as_observed_runtime_outcome(
    runtime_outcome: Any,
    *,
    live_turns: tuple[Any, ...] = (),
) -> _ObservedRuntimeOutcome:
    return _ObservedRuntimeOutcome(
        kind=str(getattr(runtime_outcome, "kind", "failed")),
        output=str(getattr(runtime_outcome, "output", "")),
        live_turns=live_turns,
        invocation_records=tuple(getattr(runtime_outcome, "invocation_records", ())),
        result=getattr(runtime_outcome, "result", None),
        service_name=getattr(runtime_outcome, "service_name", None),
        account_label=getattr(runtime_outcome, "account_label", None),
        reset_time=getattr(runtime_outcome, "reset_time", None),
        invocation_progress=getattr(runtime_outcome, "invocation_progress", None),
        continuation=getattr(runtime_outcome, "continuation", None),
        usage=getattr(runtime_outcome, "usage", None),
    )


def _build_live_smoke_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Live Provider Smoke Test.",
        epilog=_live_smoke_defaults_help_text(),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    supported_policies = [policy.name for policy in ToolPolicy]
    supported_modes = ("ephemeral", "new_session", "resumed_session")
    supported_providers = tuple(live_provider_smoke_plan.SUPPORTED_PROVIDERS) + ("all",)

    parser.add_argument(
        "command",
        nargs="?",
        default=None,
        choices=("run",),
        help="Friendly run command for full-matrix execution.",
    )
    parser.add_argument(
        "run_provider",
        nargs="?",
        default=None,
        choices=tuple(live_provider_smoke_plan.SUPPORTED_PROVIDERS),
        help="Optional explicit provider for run command.",
    )

    parser.add_argument(
        "--provider",
        action="append",
        dest="providers",
        default=[],
        choices=supported_providers,
        help=(
            "Provider selection for smoke cases. Supported values: "
            f"{_safe_list(supported_providers)}. "
            "Can be repeated."
        ),
    )
    parser.add_argument(
        "--mode",
        action="append",
        dest="lifecycle_modes",
        default=[],
        choices=supported_modes,
        help=(
            "Lifecycle mode to execute. "
            "Can be repeated. "
            f"Supported: {_safe_list(supported_modes)}."
        ),
    )
    parser.add_argument(
        "--policy",
        action="append",
        dest="tool_policies",
        default=[],
        choices=supported_policies,
        help=(
            "Tool-policy mode. "
            "Supported values: "
            f"{_safe_list(supported_policies)}. "
            "If set, policy cases run with ephemeral mode."
        ),
    )
    parser.add_argument(
        "--model",
        action="append",
        metavar="SERVICE=MODEL",
        default=[],
        type=lambda value: _parse_service_map_arg(value, flag_name="--model"),
        help=(
            "Per-provider model override (for example: --model claude=sonnet). "
            "Defaults from Live Smoke Default."
        ),
    )
    parser.add_argument(
        "--effort",
        action="append",
        metavar="SERVICE=EFFORT",
        default=[],
        type=lambda value: _parse_service_map_arg(value, flag_name="--effort"),
        help=(
            "Per-provider effort override (for example: --effort claude=high). "
            "Defaults from Live Smoke Default."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON summary to stdout.",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=DEFAULT_LIVE_SMOKE_ARTIFACT_ROOT,
        help=(
            "Artifact root for preserved diagnostics. "
            "Artifacts may contain potentially sensitive provider output."
        ),
    )
    parser.add_argument(
        "--cleanup-artifact-root",
        action="store_true",
        help="Remove artifact root before execution.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=_DEFAULT_CASE_TIMEOUT_SECONDS,
        help=(
            "Case timeout in seconds. Useful for local smoke guardrails. "
            "The smoke runner does not stream raw subprocess output."
        ),
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run id (path-safe). Defaults to a random value.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Emit richer local diagnostics while leaving provider streams off-console.",
    )
    return parser


def _parse_cli_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = _build_live_smoke_parser()
    if argv is None:
        parsed = parser.parse_args()
    else:
        parsed = parser.parse_args(list(argv))
    if parsed.command == "run" and parsed.run_provider is not None and parsed.providers:
        parser.error("Cannot combine run <provider> with --provider.")
    return parsed


def _build_service_model_map(entries: Sequence[tuple[str, str]]) -> dict[str, str]:
    model_overrides: dict[str, str] = {}
    for service, value in entries:
        model_overrides[service.lower()] = value
    return model_overrides


def _coerce_list_provider_selection(parsed: argparse.Namespace) -> tuple[str, ...]:
    selected = tuple(parsed.providers or ("all",))
    if "all" in selected and len(selected) > 1:
        raise ValueError("Cannot combine --provider all with specific providers.")
    return selected


def _coerce_run_provider_selection(parsed: argparse.Namespace) -> tuple[str, ...]:
    if parsed.run_provider is not None:
        return (parsed.run_provider,)
    selected = _coerce_list_provider_selection(parsed)
    if len(selected) > 1 and parsed.command == "run":
        raise ValueError("run command accepts only one provider selector.")
    return selected


def _coerce_lifecycle_modes(parsed: argparse.Namespace) -> tuple[str, ...]:
    selected: tuple[str, ...]
    if parsed.tool_policies:
        selected = ("ephemeral",)
    elif parsed.lifecycle_modes:
        selected = tuple(parsed.lifecycle_modes)
    else:
        selected = ("ephemeral", "new_session", "resumed_session")
    return tuple(dict.fromkeys(selected))


def _build_smoke_plan(
    provider_selection: str | tuple[str, ...],
    *,
    lifecycle_modes: tuple[str, ...],
    tool_policies: tuple[str, ...],
    run_id: str | None,
    model_overrides: Mapping[str, str] | None,
    effort_overrides: Mapping[str, str] | None,
    artifact_root: Path,
    env: Mapping[str, str] | None = None,
    claude_code_oauth_token: str | None = None,
    opencode_api_key: str | None = None,
    codex_auth_present: bool | None = None,
) -> Any:
    lifecycle_tool_policy_mode = bool(tool_policies) and set(lifecycle_modes) != {
        "ephemeral"
    }
    if not lifecycle_tool_policy_mode:
        return live_provider_smoke_plan.build_dry_run_plan(
            provider_selection,
            lifecycle_modes=lifecycle_modes,
            tool_policies=tool_policies,
            run_id=run_id,
            model_overrides=model_overrides,
            effort_overrides=effort_overrides,
            artifact_root=artifact_root,
            env=env,
            claude_code_oauth_token=claude_code_oauth_token,
            opencode_api_key=opencode_api_key,
            codex_auth_present=codex_auth_present,
        )

    lifecycle_plan = live_provider_smoke_plan.build_dry_run_plan(
        provider_selection,
        lifecycle_modes=lifecycle_modes,
        tool_policies=(),
        run_id=run_id,
        model_overrides=model_overrides,
        effort_overrides=effort_overrides,
        artifact_root=artifact_root,
        env=env,
        claude_code_oauth_token=claude_code_oauth_token,
        opencode_api_key=opencode_api_key,
        codex_auth_present=codex_auth_present,
    )
    policy_plan = live_provider_smoke_plan.build_dry_run_plan(
        provider_selection,
        lifecycle_modes=("ephemeral",),
        tool_policies=tool_policies,
        run_id=run_id,
        model_overrides=model_overrides,
        effort_overrides=effort_overrides,
        artifact_root=artifact_root,
        env=env,
        claude_code_oauth_token=claude_code_oauth_token,
        opencode_api_key=opencode_api_key,
        codex_auth_present=codex_auth_present,
    )
    lifecycle_case_order = tuple(
        c
        for provider_plan in lifecycle_plan.provider_plans
        for c in lifecycle_plan.cases
        if c.service == provider_plan.service and c.policy is None
    )
    policy_case_order = tuple(
        c
        for provider_plan in lifecycle_plan.provider_plans
        for c in policy_plan.cases
        if c.service == provider_plan.service and c.policy is not None
    )
    return live_provider_smoke_plan.DryRunPlan(
        run_id=lifecycle_plan.run_id,
        cases=lifecycle_case_order + policy_case_order,
        provider_plans=lifecycle_plan.provider_plans,
        artifact_root=lifecycle_plan.artifact_root,
    )


def _print_run_result(
    run_result: LiveSmokeRunResult,
    *,
    verbose: bool,
    artifact_root: Path,
) -> tuple[str, int]:
    lines: list[str] = []
    lines.append(f"artifact root: {artifact_root}")
    for case in run_result.cases:
        status = case.status
        label = _case_label(case)
        lines.append(f"{label}: {status}")
        if verbose:
            if case.diagnostic:
                lines.append(f"  diagnostic: {case.diagnostic}")
            if case.traceback:
                lines.append(f"  traceback: {case.traceback}")
            lines.append(f"  artifact_path: {case.artifact_path}")
    failed_case_runs = _build_failed_case_runs(run_result.cases)
    lines.extend(_format_rerun_block(failed_case_runs).splitlines())
    for warning in run_result.warnings:
        lines.append(f"warning: {warning}")
    lines.append(f"final status: {'passed' if run_result.passed else 'failed'}")
    text = "\n".join(lines)
    if text:
        print(text)
    return text, 0 if run_result.passed else 1


def _build_failed_case_runs(
    cases: Sequence[LiveSmokeRunCaseResult],
) -> tuple[dict[str, str | None], ...]:
    failed_runs = []
    for case in cases:
        if case.status == "passed":
            continue
        if case.status == "skipped" and not case.required:
            continue
        failed_runs.append(
            {
                "provider": case.service,
                "mode": case.mode,
                "policy": case.policy,
                "status": case.status,
                "command": _build_case_rerun_command(case),
            }
        )
    return tuple(failed_runs)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_cli_args(argv)
    model_overrides = _build_service_model_map(args.model)
    effort_overrides = _build_service_model_map(args.effort)
    providers = (
        _coerce_run_provider_selection(args)
        if args.command == "run"
        else _coerce_list_provider_selection(args)
    )
    lifecycle_modes = _coerce_lifecycle_modes(args)
    tool_policies = tuple(args.tool_policies)
    live_smoke_env = _resolve_live_smoke_env(None)

    run_result = run_live_smoke(
        provider_selection=providers,
        lifecycle_modes=lifecycle_modes,
        tool_policies=tool_policies,
        run_id=args.run_id,
        model_overrides=model_overrides,
        effort_overrides=effort_overrides,
        artifact_root=args.artifact_root,
        env=live_smoke_env,
        cleanup_artifact_root=args.cleanup_artifact_root,
    )

    summary_payload: dict[str, Any] | None = None
    if run_result.summary_path.exists():
        try:
            summary_payload = json.loads(
                run_result.summary_path.read_text(encoding="utf-8")
            )
        except Exception:
            summary_payload = None

    if args.json:
        failed_case_runs = _build_failed_case_runs(run_result.cases)
        if summary_payload is None:
            summary_payload = _build_cli_summary_payload(
                run_result, failed_case_runs=failed_case_runs
            )
        else:
            summary_payload["failed_case_runs"] = list(failed_case_runs)
            summary_payload["run_id"] = run_result.run_id
            summary_payload["artifact_root"] = _portable_json_path(
                run_result.artifact_root
            )
            summary_payload["passed"] = run_result.passed
            summary_payload["warnings"] = list(run_result.warnings)
            summary_payload["duration_seconds"] = summary_payload.get(
                "duration_seconds", 0.0
            )
        print(json.dumps(summary_payload, sort_keys=True))
        return 0 if run_result.passed else 1

    _, status = _print_run_result(
        run_result,
        verbose=args.verbose,
        artifact_root=run_result.artifact_root,
    )
    return status


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
        "artifact_root": _portable_json_path(run_result.artifact_root),
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
                "artifact_path": _portable_json_path(case.artifact_path),
                "status": case.status,
                "required": case.required,
                "diagnostic": case.diagnostic,
                "provider_output": case.provider_output,
                "traceback": case.traceback,
                "duration_seconds": case.duration_seconds,
            }
            for case in run_result.cases
        ],
        "provider_plans": [
            {
                "service": plan["service"]
                if isinstance(plan, Mapping)
                else getattr(plan, "service"),
                "status": plan["status"]
                if isinstance(plan, Mapping)
                else getattr(plan, "status"),
                "model": plan["model"]
                if isinstance(plan, Mapping)
                else getattr(plan, "model"),
                "effort": plan["effort"]
                if isinstance(plan, Mapping)
                else getattr(plan, "effort"),
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


def _derive_session_continuation_sentinel(output: str) -> str:
    if not output:
        return output
    stripped_tokens = tuple(
        token.strip("`\"'[](){}.,:;!?").strip() for token in output.split()
    )
    for index, token in enumerate(stripped_tokens):
        lower = token.lower()
        if "token" in lower or "sentinel" in lower or "continuation" in lower:
            if lower in {"sentinel", "continuation", "token"} and index + 1 < len(
                stripped_tokens
            ):
                candidate = stripped_tokens[index + 1]
                if candidate:
                    return candidate
            return token
    return stripped_tokens[0]


def _resolve_plan_env(env: Mapping[str, str] | None) -> Mapping[str, str]:
    return _resolve_live_smoke_env(env)


def _resolve_tool_policy(case_policy: str | None) -> ToolPolicy:
    if case_policy is None:
        return ToolPolicy.UNRESTRICTED
    try:
        return ToolPolicy[case_policy]
    except KeyError:
        raise ValueError(f"Unsupported tool policy: {case_policy}") from None


def _resolve_provider_auth(
    *,
    service: str,
    env: Mapping[str, str] | None,
    claude_code_oauth_token: str | None,
    opencode_api_key: str | None,
) -> ProviderAuth:
    env_map = _resolve_plan_env(env)
    return ProviderAuth(
        claude_code_oauth_token=claude_code_oauth_token
        if service == "claude" and claude_code_oauth_token is not None
        else env_map.get("CLAUDE_CODE_OAUTH_TOKEN")
        if service == "claude"
        else None,
        opencode_api_key=opencode_api_key
        if service == "opencode" and opencode_api_key is not None
        else env_map.get("OPENCODE_GO_API_KEY")
        if service == "opencode"
        else None,
    )


def _resolve_case_provider_selection(
    resolved_case: Any,
    *,
    env: Mapping[str, str] | None,
    claude_code_oauth_token: str | None,
    opencode_api_key: str | None,
) -> ProviderSelection:
    planned_selection = getattr(resolved_case, "provider_selection", None)
    if isinstance(planned_selection, ProviderSelection):
        return planned_selection
    return ProviderSelection(
        service=resolved_case.service,
        model=resolved_case.model,
        effort=resolved_case.effort,
        auth=_resolve_provider_auth(
            service=resolved_case.service,
            env=env,
            claude_code_oauth_token=claude_code_oauth_token,
            opencode_api_key=opencode_api_key,
        ),
    )


def _run_public_smoke_case(
    planned_case: Any | None = None,
    artifact_dir: Path | None = None,
    *,
    case: Any | None = None,
    run_id: str | None = None,
    model: str | None = None,
    effort: str | None = None,
    prompt: str,
    env: Mapping[str, str] | None,
    claude_code_oauth_token: str | None,
    opencode_api_key: str | None,
    codex_auth_present: bool | None,
    continuation: Continuation | None = None,
) -> Any:
    del run_id, model, effort, codex_auth_present  # compatibility placeholders
    resolved_case = case if case is not None else planned_case
    if resolved_case is None:
        raise ValueError("case is required to run ephemeral smoke case")
    if artifact_dir is None:
        raise ValueError("artifact_dir is required to run ephemeral smoke case")
    live_turns: list[Any] = []

    def _on_live_output(turn: Any) -> None:
        live_turns.append(turn)

    tool_policy = _resolve_tool_policy(resolved_case.policy)
    provider_selection = _resolve_case_provider_selection(
        resolved_case,
        env=env,
        claude_code_oauth_token=claude_code_oauth_token,
        opencode_api_key=opencode_api_key,
    )
    auth = provider_selection.auth or ProviderAuth()

    request: Any
    if resolved_case.mode == "ephemeral":
        request = EphemeralRunRequest(
            prompt=prompt,
            invocation_dir=artifact_dir,
            provider_selection=provider_selection,
            tool_policy=tool_policy,
            on_live_output=_on_live_output,
        )
    elif resolved_case.mode == "new_session":
        request = NewSessionRunRequest(
            prompt=prompt,
            invocation_dir=artifact_dir,
            provider_selection=provider_selection,
            tool_policy=tool_policy,
            on_live_output=_on_live_output,
        )
    elif resolved_case.mode == "resumed_session":
        if continuation is None:
            raise ValueError("Resume session case requires prior continuation.")
        request = ResumedSessionRunRequest(
            prompt=prompt,
            invocation_dir=artifact_dir,
            continuation=continuation,
            provider_auth=auth,
            on_live_output=_on_live_output,
        )
    else:
        raise ValueError(
            f"public smoke runner only supports ephemeral, new_session, "
            f"and resumed_session modes; got {resolved_case.mode!r}"
        )

    runtime_outcome: Any
    runtime_client = RuntimeClient()
    if resolved_case.mode == "ephemeral":
        runtime_outcome = runtime_client.run_ephemeral(request)
    elif resolved_case.mode == "new_session":
        runtime_outcome = _resolve_runtime_outcome(
            runtime_client.run_new_session(request)
        )
    else:
        runtime_outcome = _resolve_runtime_outcome(
            runtime_client.run_resumed_session(request)
        )
    return _as_observed_runtime_outcome(
        runtime_outcome,
        live_turns=tuple(live_turns),
    )


def _resolve_runtime_outcome(runtime_outcome: Any) -> Any:
    if inspect.iscoroutine(runtime_outcome):
        return asyncio.run(runtime_outcome)
    return runtime_outcome


def _serialize_case_invocation_records(invocation_records: Any) -> list[dict[str, Any]]:
    if invocation_records is None:
        return []
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


def _serialize_case_output(planned_case: Any, case_outcome: Any) -> dict[str, Any]:
    return {
        "service": planned_case.service,
        "mode": planned_case.mode,
        "policy": planned_case.policy,
        "model": planned_case.model,
        "effort": planned_case.effort,
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
    _write_json(
        artifact_dir / "outcome.json",
        _serialize_case_output(planned_case, case_outcome),
    )
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
            "artifact_root": _portable_json_path(summary_root / run_id),
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


def _build_case_artifact_dir(
    artifact_root: Path,
    run_id: str,
    planned_case: Any,
    occurrence: int,
) -> Path:
    artifact_dir = (
        artifact_root
        / run_id
        / planned_case.service
        / planned_case.mode
        / (planned_case.policy or "default")
    )
    if occurrence == 1:
        return artifact_dir
    return artifact_dir.with_name(f"{artifact_dir.name}-{occurrence}")


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
    resolved_env = _resolve_live_smoke_env(env)
    if codex_auth_present is None:
        codex_auth_present = live_provider_smoke_plan.detect_codex_auth_present()
    if cleanup_artifact_root:
        shutil.rmtree(resolved_artifact_root, ignore_errors=True)
    resolved_artifact_root.mkdir(parents=True, exist_ok=True)

    dry_run_plan = _build_smoke_plan(
        provider_selection,
        lifecycle_modes=lifecycle_modes,
        tool_policies=tool_policies,
        run_id=run_id,
        model_overrides=model_overrides,
        effort_overrides=effort_overrides,
        artifact_root=resolved_artifact_root,
        env=resolved_env,
        claude_code_oauth_token=claude_code_oauth_token,
        opencode_api_key=opencode_api_key,
        codex_auth_present=codex_auth_present,
    )

    warnings: list[str] = []
    case_results: list[LiveSmokeRunCaseResult] = []
    case_occurrences: dict[tuple[str, str, str | None], int] = {}
    session_continuations: dict[tuple[str, str | None], Any] = {}
    session_turns: dict[tuple[str, str | None], str] = {}
    session_invocation_dirs: dict[tuple[str, str | None], Path] = {}
    lifecycle_failures: set[str] = set()
    lifecycle_tool_policy_mode = bool(tool_policies) and set(lifecycle_modes) != {
        "ephemeral"
    }
    runner = case_runner or _run_public_smoke_case
    run_started = time.perf_counter()

    for plan in dry_run_plan.provider_plans:
        if (
            plan.status
            is not live_provider_smoke_plan.LiveSmokeProviderSelectionStatus.RUNNABLE
        ):
            plan_status = getattr(plan.status, "value", plan.status)
            reason = getattr(plan, "reason", None)
            if reason:
                warnings.append(f"{plan.service}: {plan_status}: {reason}")
            else:
                warnings.append(f"{plan.service}: {plan_status}: provider not runnable")

    run_artifact_root = resolved_artifact_root / dry_run_plan.run_id
    run_artifact_root.mkdir(parents=True, exist_ok=True)

    for planned_case in dry_run_plan.cases:
        case_key = (
            planned_case.service,
            planned_case.mode,
            planned_case.policy,
        )
        case_occurrences[case_key] = case_occurrences.get(case_key, 0) + 1
        invocation_dir = _build_case_artifact_dir(
            resolved_artifact_root,
            dry_run_plan.run_id,
            planned_case,
            case_occurrences[case_key],
        )
        prompt = _build_case_prompt(dry_run_plan.run_id, planned_case)
        case_started = time.perf_counter()
        case_traceback: str | None = None

        case_continuation = None
        required_continuation_text = None
        case_state_key = (planned_case.service, planned_case.policy)
        invocation_dir_for_run = invocation_dir
        if (
            lifecycle_tool_policy_mode
            and planned_case.policy is not None
            and planned_case.service in lifecycle_failures
        ):
            case_classification = live_provider_smoke_plan.LiveSmokeCaseResult(
                service=planned_case.service,
                mode=planned_case.mode,
                policy=planned_case.policy,
                status=live_provider_smoke_plan.LiveSmokeCaseStatus.SKIPPED,
                required=False,
                diagnostic="lifecycle smoke failed earlier for provider",
            )
            provider_output = ""
            case_duration = 0.0
            case_results.append(
                LiveSmokeRunCaseResult(
                    service=planned_case.service,
                    mode=planned_case.mode,
                    policy=planned_case.policy,
                    model=planned_case.model,
                    effort=planned_case.effort,
                    artifact_path=str(invocation_dir),
                    status=case_classification.status.value,
                    required=case_classification.required,
                    diagnostic=case_classification.diagnostic,
                    provider_output=provider_output,
                    traceback=case_traceback,
                    duration_seconds=case_duration,
                )
            )
            try:
                _write_optional_case_artifacts(
                    planned_case=planned_case,
                    case_outcome=_StubOutcome(kind="failed"),
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
            continue

        try:
            if planned_case.mode == "resumed_session":
                case_continuation = session_continuations.get(case_state_key)
                required_continuation_text = session_turns.get(case_state_key)
                if case_continuation is None:
                    raise RuntimeError(
                        "Resume session case requires prior successful new session "
                        "continuation."
                    )
                if case_state_key not in session_invocation_dirs:
                    raise RuntimeError(
                        "Resume session case requires matching prior new session "
                        "invocation directory."
                    )
                invocation_dir_for_run = session_invocation_dirs[case_state_key]

            invocation_dir_for_run.mkdir(parents=True, exist_ok=True)
            case_outcome = runner(
                case=planned_case,
                artifact_dir=invocation_dir_for_run,
                run_id=dry_run_plan.run_id,
                prompt=prompt,
                model=planned_case.model,
                effort=planned_case.effort,
                env=resolved_env,
                claude_code_oauth_token=claude_code_oauth_token,
                opencode_api_key=opencode_api_key,
                codex_auth_present=codex_auth_present,
                continuation=case_continuation,
            )
            classify_kwargs = {
                "case": cast(Any, planned_case),
                "runtime_outcome": case_outcome,
            }
            if planned_case.mode == "new_session":
                classify_kwargs["required_continuation"] = True
            case_classification = (
                live_provider_smoke_plan.classify_live_smoke_case_result(
                    **classify_kwargs,
                )
            )
            provider_output = str(getattr(case_outcome, "output", ""))
            if (
                planned_case.mode == "resumed_session"
                and required_continuation_text is not None
                and case_classification.status
                is live_provider_smoke_plan.LiveSmokeCaseStatus.PASSED
            ):
                if required_continuation_text not in provider_output:
                    case_classification = live_provider_smoke_plan.LiveSmokeCaseResult(
                        service=planned_case.service,
                        mode=planned_case.mode,
                        policy=planned_case.policy,
                        status=live_provider_smoke_plan.LiveSmokeCaseStatus.FAILED,
                        diagnostic="completed outcome missing required continuation evidence",
                    )
            if (
                planned_case.mode == "new_session"
                and case_classification.status
                is live_provider_smoke_plan.LiveSmokeCaseStatus.PASSED
            ):
                continuation_value = getattr(case_outcome, "continuation", None)
                if continuation_value is not None:
                    session_continuations[case_state_key] = continuation_value
                    session_turns[case_state_key] = (
                        _derive_session_continuation_sentinel(
                            str(getattr(case_outcome, "output", ""))
                        )
                    )
                    session_invocation_dirs[case_state_key] = invocation_dir
            elif (
                lifecycle_tool_policy_mode
                and planned_case.policy is None
                and case_classification.status
                is not live_provider_smoke_plan.LiveSmokeCaseStatus.PASSED
            ):
                lifecycle_failures.add(planned_case.service)
        except Exception as exc:
            case_traceback = format_exc()
            case_classification = (
                live_provider_smoke_plan.classify_live_smoke_case_result(
                    case=cast(Any, planned_case),
                    runtime_exception=exc,
                )
            )
            case_outcome = _StubOutcome(kind="failed")
            provider_output = ""
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
                status=case_classification.status.value,
                required=case_classification.required,
                diagnostic=case_classification.diagnostic,
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
    if not case_results:
        if dry_run_plan.provider_plans and all(
            plan.status
            is live_provider_smoke_plan.LiveSmokeProviderSelectionStatus.SKIPPED
            for plan in dry_run_plan.provider_plans
        ):
            warnings.append("all selected providers are unconfigured")
        warnings.append("no runnable smoke cases planned")
    run_case_success = bool(case_results) and all(
        case.status == "passed" or (case.status == "skipped" and not case.required)
        for case in case_results
    )
    run_result = LiveSmokeRunResult(
        run_id=dry_run_plan.run_id,
        artifact_root=resolved_artifact_root,
        summary_path=summary_path,
        summary_written=False,
        passed=run_case_success,
        cases=tuple(case_results),
        warnings=tuple(warnings),
    )

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

    final_run_result = LiveSmokeRunResult(
        run_id=run_result.run_id,
        artifact_root=run_result.artifact_root,
        summary_path=summary_path,
        summary_written=False,
        passed=run_result.passed,
        cases=run_result.cases,
        warnings=tuple(warnings),
    )
    summary_payload = _build_summary_payload(
        final_run_result,
        time.perf_counter() - run_started,
        dry_run_plan.provider_plans,
    )

    summary_written = False
    try:
        summary_written = _write_required_summary(summary_payload, summary_path)
    except Exception:
        summary_written = False
        warnings.append("required summary write failed")

    return LiveSmokeRunResult(
        run_id=final_run_result.run_id,
        artifact_root=final_run_result.artifact_root,
        summary_path=summary_path,
        summary_written=summary_written,
        passed=final_run_result.passed and summary_written,
        cases=final_run_result.cases,
        warnings=tuple(warnings),
    )


def _run_case_stub(*_: Any, **__: Any) -> Any:
    return _StubOutcome()


if __name__ == "__main__":
    raise SystemExit(main())
