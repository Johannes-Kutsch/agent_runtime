"""Live Provider Probe — manual-debug-only runner.

Run by hand to watch a real built-in provider actually invoke through the
Runtime Public Surface. It is NOT CI, not part of the default test suite, and
not a Runtime Public Surface addition. There is no machine-readable output, no
exit-code contract, and no rerun-command suggestions: the one thing the probe
proves that pytest can't is that a real provider invocation reaches a
classified runtime outcome without an unexpected exception. See ADR 0013.

Per case the probe:
  * streams agent messages and tool calls live to the terminal,
  * writes ``live_feed.json`` (JSON-lines, appended as events arrive so a crash
    leaves a valid partial feed; carries the full ``raw_provider_output``), and
  * writes ``result.json`` (outcome kind/category, selected provider, output,
    usage, continuation, traceback).

Artifacts live under ``<artifact-root>/<service>/<mode>_<ToolPolicy>/``. A
service's entire directory is wiped and recreated before it reruns.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from traceback import format_exc
from types import ModuleType
from typing import Any, Mapping, Sequence, TextIO
from importlib import util as importlib_util

from agent_runtime.errors import (
    AgentCredentialFailureError,
    RuntimeConfigurationError,
)
from agent_runtime.runtime import (
    Continuation,
    EphemeralRunRequest,
    NewSessionRunRequest,
    ProviderAuth,
    ResumedSessionRunRequest,
    RuntimeClient,
    ToolPolicy,
)

try:
    import live_provider_probe_plan as plan
except ModuleNotFoundError:
    _plan_spec = importlib_util.spec_from_file_location(
        "live_provider_probe_plan",
        str(Path(__file__).resolve().parent / "live_provider_probe_plan.py"),
    )
    assert _plan_spec is not None
    assert _plan_spec.loader is not None
    plan = ModuleType("live_provider_probe_plan")
    sys.modules["live_provider_probe_plan"] = plan
    _plan_spec.loader.exec_module(plan)


DEFAULT_ARTIFACT_ROOT = "live-probe-artifacts"
LIVE_FEED_FILENAME = "live_feed.json"
RESULT_FILENAME = "result.json"
_DEFAULT_TIMEOUT_SECONDS = 300
_ENV_PATH = Path(__file__).resolve().parent / ".env"
_ENV_PATH_OVERRIDE = "LIVE_PROBE_ENV_PATH"

_DISPLAYED_EVENT_TYPES = ("agent_message", "agent_tool_call")

_RED = "\033[31m"
_DIM = "\033[2m"
_RESET = "\033[0m"

_PROMPTS = {
    "ephemeral": "Live probe check: reply with a one-word greeting.",
    "new_session": "Live probe check: reply with a one-word greeting.",
    "resumed_session": "Live probe check: reply again with a one-word greeting.",
}


# --------------------------------------------------------------------------- #
# Local credential loading (read-only; never written or mutated).
# --------------------------------------------------------------------------- #
def _parse_env_lines(text: str, *, env_path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
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


def _load_env(env_path: Path | None = None) -> dict[str, str]:
    path = env_path or Path(os.environ.get(_ENV_PATH_OVERRIDE, str(_ENV_PATH)))
    if not path.exists():
        return {}
    return _parse_env_lines(path.read_text(encoding="utf-8"), env_path=path)


def _resolve_env(env: Mapping[str, str] | None) -> Mapping[str, str]:
    return env if env is not None else _load_env()


# --------------------------------------------------------------------------- #
# Terminal output helpers.
# --------------------------------------------------------------------------- #
@dataclass
class _Console:
    stream: TextIO
    color: bool

    def line(self, text: str = "") -> None:
        print(text, file=self.stream)

    def red(self, text: str) -> None:
        self.line(f"{_RED}{text}{_RESET}" if self.color else text)

    def dim(self, text: str) -> None:
        self.line(f"{_DIM}{text}{_RESET}" if self.color else text)


# --------------------------------------------------------------------------- #
# Case execution.
# --------------------------------------------------------------------------- #
@dataclass
class _CaseExecution:
    category: str
    outcome: Any | None
    traceback: str | None


def _resolve_runtime_outcome(awaitable: Any) -> Any:
    return asyncio.run(awaitable)


def _run_single_case(
    case: Any,
    *,
    case_dir: Path,
    invocation_dir: Path,
    prompt: str,
    timeout_seconds: int,
    continuation: Continuation | None,
    console: _Console,
) -> _CaseExecution:
    case_dir.mkdir(parents=True, exist_ok=True)
    invocation_dir.mkdir(parents=True, exist_ok=True)
    feed_path = case_dir / LIVE_FEED_FILENAME
    feed_sink = feed_path.open("w", encoding="utf-8")

    def _on_live_output(event: Any) -> None:
        record = {
            "type": getattr(event, "type", ""),
            "display_message": getattr(event, "display_message", ""),
            "raw_provider_output": getattr(event, "raw_provider_output", ""),
        }
        feed_sink.write(json.dumps(record) + "\n")
        feed_sink.flush()
        if record["type"] in _DISPLAYED_EVENT_TYPES:
            console.line(f"  {record['display_message']}")

    selection = case.provider_selection
    auth = getattr(selection, "auth", None) or ProviderAuth()
    tool_policy = ToolPolicy[case.tool_policy]

    try:
        client = RuntimeClient()
        if case.mode == "ephemeral":
            request = EphemeralRunRequest(
                prompt=prompt,
                invocation_dir=invocation_dir,
                provider_selection=selection,
                tool_policy=tool_policy,
                timeout_seconds=timeout_seconds,
                on_live_output=_on_live_output,
            )
            outcome = _resolve_runtime_outcome(client.run_ephemeral(request))
        elif case.mode == "new_session":
            request = NewSessionRunRequest(
                prompt=prompt,
                invocation_dir=invocation_dir,
                provider_selection=selection,
                tool_policy=tool_policy,
                timeout_seconds=timeout_seconds,
                on_live_output=_on_live_output,
            )
            outcome = _resolve_runtime_outcome(client.run_new_session(request))
        elif case.mode == "resumed_session":
            if continuation is None:
                raise RuntimeError(
                    "resumed_session requires a continuation from new_session; "
                    "the new_session case did not produce one"
                )
            request = ResumedSessionRunRequest(
                prompt=prompt,
                invocation_dir=invocation_dir,
                continuation=continuation,
                provider_auth=auth,
                timeout_seconds=timeout_seconds,
                on_live_output=_on_live_output,
            )
            outcome = _resolve_runtime_outcome(client.run_resumed_session(request))
        else:
            raise ValueError(f"unsupported probe mode: {case.mode!r}")
    except AgentCredentialFailureError:
        return _CaseExecution(
            category="wrong_credentials", outcome=None, traceback=format_exc()
        )
    except Exception:
        return _CaseExecution(category="error", outcome=None, traceback=format_exc())
    finally:
        feed_sink.close()

    return _CaseExecution(
        category=plan.outcome_category(outcome), outcome=outcome, traceback=None
    )


def _selected_payload(selected: Any) -> dict[str, Any] | None:
    if selected is None:
        return None
    return {
        "service": getattr(selected, "service", None),
        "model": getattr(selected, "model", None),
        "effort": getattr(selected, "effort", None),
    }


def _usage_payload(usage: Any) -> dict[str, Any] | None:
    if usage is None:
        return None
    return {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", None),
        "cache_creation_input_tokens": getattr(
            usage, "cache_creation_input_tokens", None
        ),
        "cost_usd": getattr(usage, "cost_usd", None),
        "duration_seconds": getattr(usage, "duration_seconds", None),
    }


def _continuation_from_outcome(outcome: Any) -> Continuation | None:
    result = getattr(outcome, "result", None)
    return getattr(result, "continuation", None)


def _write_result_json(case_dir: Path, case: Any, execution: _CaseExecution) -> None:
    outcome = execution.outcome
    result = getattr(outcome, "result", None)
    kind = getattr(outcome, "kind", None)
    continuation = _continuation_from_outcome(outcome)
    payload = {
        "service": case.service,
        "mode": case.mode,
        "tool_policy": case.tool_policy,
        "category": execution.category,
        "kind": type(kind).__name__ if kind is not None else None,
        "selected": _selected_payload(getattr(result, "selected", None)),
        "output": getattr(result, "output", None),
        "usage": _usage_payload(getattr(result, "usage", None)),
        "continuation": continuation.serialized if continuation is not None else None,
        "traceback": execution.traceback,
    }
    (case_dir / RESULT_FILENAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


# --------------------------------------------------------------------------- #
# Probe orchestration.
# --------------------------------------------------------------------------- #
def run_probe(
    provider_selection: str | Sequence[str],
    *,
    model_overrides: Mapping[str, str] | None = None,
    effort_overrides: Mapping[str, str] | None = None,
    artifact_root: Path | str | None = None,
    env: Mapping[str, str] | None = None,
    claude_code_oauth_token: str | None = None,
    opencode_api_key: str | None = None,
    codex_auth_present: bool | None = None,
    timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
    stream: TextIO | None = None,
    color: bool | None = None,
) -> Path:
    """Probe the selected providers; return the resolved artifact root.

    There is no pass/fail return value by design — read the terminal and the
    per-case ``result.json`` artifacts.
    """

    out = stream if stream is not None else sys.stdout
    console = _Console(
        stream=out,
        color=color if color is not None else bool(getattr(out, "isatty", lambda: False)()),
    )
    root = Path(artifact_root or DEFAULT_ARTIFACT_ROOT).resolve()
    resolved_env = _resolve_env(env)
    if codex_auth_present is None:
        codex_auth_present = plan.detect_codex_auth_present()

    selection = plan.parse_provider_selection(provider_selection)
    provider_plans = plan.plan_selected_providers(
        selection,
        model_overrides=model_overrides,
        effort_overrides=effort_overrides,
        env=resolved_env,
        claude_code_oauth_token=claude_code_oauth_token,
        opencode_api_key=opencode_api_key,
        codex_auth_present=codex_auth_present,
    )

    console.line(f"Live Provider Probe — artifact root: {root}")
    root.mkdir(parents=True, exist_ok=True)

    for provider_plan in provider_plans:
        if provider_plan.status is plan.ProviderConfigStatus.SKIPPED:
            console.dim(
                f"{provider_plan.service}: skipped (unconfigured): "
                f"{provider_plan.reason}"
            )
            continue
        if provider_plan.status is plan.ProviderConfigStatus.CONFIG_ERROR:
            console.red(
                f"{provider_plan.service}: not configured: {provider_plan.reason}"
            )
            continue
        _run_provider(
            provider_plan,
            root=root,
            timeout_seconds=timeout_seconds,
            console=console,
        )

    return root


def _run_provider(
    provider_plan: Any,
    *,
    root: Path,
    timeout_seconds: int,
    console: _Console,
) -> None:
    service_dir = root / provider_plan.service
    shutil.rmtree(service_dir, ignore_errors=True)
    service_dir.mkdir(parents=True, exist_ok=True)

    console.line()
    console.line(
        f"== {provider_plan.service} "
        f"({provider_plan.model}/{provider_plan.effort}) =="
    )

    cases = plan.probe_cases_for_provider(provider_plan)
    new_session_continuation: Continuation | None = None
    new_session_invocation_dir: Path | None = None

    for case in cases:
        case_dir = service_dir / case.label
        console.line(f"-- {case.label}")

        if case.mode == "resumed_session":
            invocation_dir = new_session_invocation_dir or case_dir
            continuation = new_session_continuation
        else:
            invocation_dir = case_dir
            continuation = None

        execution = _run_single_case(
            case,
            case_dir=case_dir,
            invocation_dir=invocation_dir,
            prompt=_PROMPTS[case.mode],
            timeout_seconds=timeout_seconds,
            continuation=continuation,
            console=console,
        )

        try:
            _write_result_json(case_dir, case, execution)
        except Exception as exc:  # pragma: no cover - diagnostics best effort
            console.red(f"  (failed to write {RESULT_FILENAME}: {exc})")

        if case.mode == "new_session":
            new_session_continuation = _continuation_from_outcome(execution.outcome)
            new_session_invocation_dir = invocation_dir

        if execution.category == plan.SUCCESS_CATEGORY:
            console.line(f"  -> {execution.category}")
        else:
            console.red(f"  -> run not completed: {execution.category}")


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def _defaults_help_text() -> str:
    default_tuples = ", ".join(
        f"{service}={plan.LIVE_PROBE_DEFAULTS[service][0]}/"
        f"{plan.LIVE_PROBE_DEFAULTS[service][1]}"
        for service in plan.SUPPORTED_PROVIDERS
    )
    return (
        "Model/effort precedence: CLI override, then Live Probe Default.\n"
        f"Live Probe Defaults: {default_tuples}. "
        f"Verified {plan.LIVE_PROBE_DEFAULTS_VERIFIED_ON}."
    )


def _parse_service_map_arg(value: str, *, flag_name: str) -> tuple[str, str]:
    service, _, payload = value.partition("=")
    if not service or not payload:
        raise argparse.ArgumentTypeError(
            f"{flag_name} expects SERVICE=VALUE (example: claude=sonnet), got {value!r}"
        )
    return service.strip(), payload.strip()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Live Provider Probe (manual debugging only).",
        epilog=_defaults_help_text(),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "provider",
        nargs="?",
        default=None,
        choices=plan.SUPPORTED_PROVIDERS,
        help="Focus on one provider. Omit to probe all configured providers.",
    )
    parser.add_argument(
        "--model",
        action="append",
        metavar="SERVICE=MODEL",
        default=[],
        type=lambda value: _parse_service_map_arg(value, flag_name="--model"),
        help="Per-provider model override (example: --model claude=sonnet).",
    )
    parser.add_argument(
        "--effort",
        action="append",
        metavar="SERVICE=EFFORT",
        default=[],
        type=lambda value: _parse_service_map_arg(value, flag_name="--effort"),
        help="Per-provider effort override (example: --effort claude=high).",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
        help=(
            "Artifact root for per-case live feeds and results. "
            "Wiped per service on rerun; may contain sensitive provider output."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=_DEFAULT_TIMEOUT_SECONDS,
        help="Per-case timeout in seconds.",
    )
    return parser


def _service_map(entries: Sequence[tuple[str, str]]) -> dict[str, str]:
    return {service.lower(): value for service, value in entries}


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    provider_selection: str | Sequence[str]
    provider_selection = (args.provider,) if args.provider else "all"
    run_probe(
        provider_selection,
        model_overrides=_service_map(args.model),
        effort_overrides=_service_map(args.effort),
        artifact_root=args.artifact_root,
        timeout_seconds=args.timeout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
