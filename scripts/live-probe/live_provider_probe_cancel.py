"""Live Provider Probe — cancel-mid-turn scenarios (manual-debug-only).

Runs two cancel-mid-turn cases per selected provider:

  * ``cancel_ephemeral`` — starts a real ephemeral invocation and calls
    ``CancellationToken.cancel()`` the moment the first live output event
    arrives. The expected outcome is ``Cancelled`` with no continuation,
    confirming the provider subprocess was hard-killed.

  * ``cancel_new_session`` — starts a real session-backed invocation and
    cancels on first output. Because provider work had started, the expected
    outcome is ``Cancelled`` **with** a continuation, confirming #437's
    continuation-preservation logic fires against a real subprocess.

This script is NOT CI, not part of the default test suite, and not a Runtime
Public Surface addition. See ADR 0013 and ADR 0020.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence, TextIO
from importlib import util as importlib_util

from agent_runtime.errors import RuntimeConfigurationError

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

try:
    import _live_probe_case_runner as case_runner
except ModuleNotFoundError:
    _case_runner_spec = importlib_util.spec_from_file_location(
        "_live_probe_case_runner",
        str(Path(__file__).resolve().parent / "_live_probe_case_runner.py"),
    )
    assert _case_runner_spec is not None
    assert _case_runner_spec.loader is not None
    case_runner = ModuleType("_live_probe_case_runner")
    sys.modules["_live_probe_case_runner"] = case_runner
    _case_runner_spec.loader.exec_module(case_runner)

RuntimeClient = case_runner.RuntimeClient

DEFAULT_ARTIFACT_ROOT = "live-probe-artifacts"
_DEFAULT_TIMEOUT_SECONDS = 300
_ENV_PATH = Path(__file__).resolve().parent / ".env"
_ENV_PATH_OVERRIDE = "LIVE_PROBE_ENV_PATH"

_RED = "\033[31m"
_DIM = "\033[2m"
_RESET = "\033[0m"

# A prompt that generates substantial output so cancellation fires mid-turn.
_CANCEL_PROMPT = (
    "Please write a very detailed, multi-paragraph explanation of "
    "how neural networks work, including backpropagation and gradient descent."
)


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
# Cancel probe orchestration.
# --------------------------------------------------------------------------- #
def run_cancel_probe(
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
    """Run cancel-mid-turn probe cases for the selected providers.

    Returns the resolved artifact root. There is no pass/fail return value by
    design — read the terminal and the per-case ``result.json`` artifacts.
    """

    out = stream if stream is not None else sys.stdout
    console = _Console(
        stream=out,
        color=color
        if color is not None
        else bool(getattr(out, "isatty", lambda: False)()),
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

    console.line(f"Live Provider Probe (cancel-mid-turn) — artifact root: {root}")
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
        _run_cancel_provider(
            provider_plan,
            root=root,
            timeout_seconds=timeout_seconds,
            console=console,
        )

    return root


def _cancel_case_for_plan(provider_plan: Any, *, mode: str) -> Any:
    return next(
        c for c in plan.probe_cases_for_provider(provider_plan) if c.mode == mode
    )


def _run_cancel_provider(
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
        f"== {provider_plan.service} ({provider_plan.model}/{provider_plan.effort})"
        f" — cancel-mid-turn =="
    )

    # Case 1: ephemeral invocation cancelled on first output.
    _run_cancel_case(
        provider_plan,
        service_dir=service_dir,
        mode="ephemeral",
        timeout_seconds=timeout_seconds,
        console=console,
        session_store=None,
    )

    # Case 2: session-backed (new_session) invocation cancelled on first output.
    # The session store is created under the case dir.
    new_session_case_dir = service_dir / _cancel_label("new_session")
    session_store = new_session_case_dir / "_session_store"
    _run_cancel_case(
        provider_plan,
        service_dir=service_dir,
        mode="new_session",
        timeout_seconds=timeout_seconds,
        console=console,
        session_store=session_store,
    )


def _cancel_label(mode: str) -> str:
    return f"cancel_{mode}_UNRESTRICTED"


def _run_cancel_case(
    provider_plan: Any,
    *,
    service_dir: Path,
    mode: str,
    timeout_seconds: int,
    console: _Console,
    session_store: Path | None,
) -> None:
    probe_case = _cancel_case_for_plan(provider_plan, mode=mode)
    label = _cancel_label(mode)
    case_dir = service_dir / label
    console.line(f"-- {label}")

    result = case_runner.run_case(
        case_runner.ProbeCaseRunRequest(
            case=probe_case,
            case_dir=case_dir,
            invocation_dir=case_dir,
            prompt=_CANCEL_PROMPT,
            timeout_seconds=timeout_seconds,
            continuation=None,
            session_store=session_store,
            output=console,
            cancel_on_first_output=True,
        ),
        runtime_client_factory=RuntimeClient,
    )

    if result.category == "cancelled":
        if mode == "ephemeral":
            console.line(
                "  -> cancelled: subprocess terminated"
                " (subprocess terminated confirmed by Cancelled outcome)"
            )
        else:
            continuation_status = (
                "yes"
                if result.continuation is not None
                else "no (provider hadn't started)"
            )
            console.line(
                f"  -> cancelled: subprocess terminated;"
                f" continuation returned: {continuation_status}"
            )
    else:
        console.red(f"  -> unexpected outcome: {result.category}")


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
        description="Run the Live Provider Probe cancel-mid-turn scenarios (manual debugging only).",
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
    run_cancel_probe(
        provider_selection,
        model_overrides=_service_map(args.model),
        effort_overrides=_service_map(args.effort),
        artifact_root=args.artifact_root,
        timeout_seconds=args.timeout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
