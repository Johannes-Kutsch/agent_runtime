"""Deterministic tests for the Live Provider Probe cancel-mid-turn scenarios.

These tests fake RuntimeClient and never reach a live provider. They cover:
cancellation token creation and wiring, cancel-on-first-output triggering,
artifact layout, outcome reporting for ephemeral (no continuation) and
session-backed (continuation expected) cancel cases.
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from agent_runtime.contracts import ToolAccess
from agent_runtime import runtime as pr

CANCEL_PROBE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "live-probe"
    / "live_provider_probe_cancel.py"
)
CASE_RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "live-probe"
    / "_live_probe_case_runner.py"
)
PROBE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "live-probe"
    / "live_provider_probe.py"
)


@pytest.fixture
def cancel_probe() -> Any:
    spec = importlib.util.spec_from_file_location(
        "live_provider_probe_cancel", CANCEL_PROBE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # type: ignore[arg-type]
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def case_runner() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_live_probe_case_runner", CASE_RUNNER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # type: ignore[arg-type]
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def probe() -> Any:
    spec = importlib.util.spec_from_file_location("live_provider_probe", PROBE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # type: ignore[arg-type]
    spec.loader.exec_module(module)
    return module


class _OutputRecorder:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def line(self, text: str) -> None:
        self.lines.append(text)


def _continuation() -> pr.Continuation:
    return pr.Continuation(
        selected_service="codex",
        selected_model="gpt-5.4-mini",
        selected_effort="low",
        tool_access=ToolAccess.no_tools(),
        provider_resume_state={"provider_session_id": "codex-session"},
    )


def _cancelled_outcome(*, continuation: Any = None) -> Any:
    return pr.RuntimeOutcome(
        kind=pr.Cancelled(),
        result=pr.RunResult(
            output="",
            usage=None,
            continuation=continuation,
            selected=pr.ResolvedProvider(
                service="codex", model="gpt-5.4-mini", effort="low"
            ),
        ),
    )


def _codex_cancel_case(probe: Any, *, mode: str) -> Any:
    provider_plan = probe.plan.plan_selected_providers(
        probe.plan.parse_provider_selection("codex"),
        env={},
        codex_auth_present=True,
    )[0]
    return next(
        case
        for case in probe.plan.probe_cases_for_provider(provider_plan)
        if case.mode == mode
    )


# --------------------------------------------------------------------------- #
# Behavior 1: ephemeral cancel — token is cancelled on first output
# --------------------------------------------------------------------------- #


def test_cancel_probe_run_case_with_cancel_on_first_output_passes_token_and_cancels_on_first_event(
    probe: Any, case_runner: Any, tmp_path: Path
) -> None:
    case = _codex_cancel_case(probe, mode="ephemeral")
    output = _OutputRecorder()
    observed: dict[str, Any] = {}

    def _record(method: str, request: Any) -> Any:
        observed["token"] = request.token
        # Emit one live event (this should trigger cancellation)
        request.on_live_output(
            pr.AgentEvent(
                type="agent_message",
                display_message="hello",
                raw_provider_output='{"event":"message"}',
            )
        )
        # After on_live_output, the token should be cancelled
        observed["token_cancelled_after_first_event"] = (
            request.token is not None and request.token.is_cancelled
        )
        return _cancelled_outcome()

    adapter = case_runner.InMemoryRuntimeInvocationAdapter(record_handler=_record)

    result = case_runner.run_case(
        case_runner.ProbeCaseRunRequest(
            case=case,
            case_dir=tmp_path / case.label,
            invocation_dir=tmp_path / "workspace",
            prompt="prompt",
            timeout_seconds=30,
            continuation=None,
            output=output,
            cancel_on_first_output=True,
        ),
        runtime_client_factory=lambda: adapter,
    )

    assert result.category == "cancelled"
    assert observed["token"] is not None
    assert observed["token_cancelled_after_first_event"]
    assert result.continuation is None


def test_cancel_probe_run_case_with_cancel_on_first_output_false_passes_no_token(
    probe: Any, case_runner: Any, tmp_path: Path
) -> None:
    case = _codex_cancel_case(probe, mode="ephemeral")
    output = _OutputRecorder()
    observed: dict[str, Any] = {}

    def _record(method: str, request: Any) -> Any:
        observed["token"] = request.token
        return pr.RuntimeOutcome(
            kind=pr.Completed(),
            result=pr.RunResult(
                output="done",
                usage=None,
                continuation=None,
                selected=pr.ResolvedProvider(
                    service="codex", model="gpt-5.4-mini", effort="low"
                ),
            ),
        )

    adapter = case_runner.InMemoryRuntimeInvocationAdapter(record_handler=_record)

    case_runner.run_case(
        case_runner.ProbeCaseRunRequest(
            case=case,
            case_dir=tmp_path / case.label,
            invocation_dir=tmp_path / "workspace",
            prompt="prompt",
            timeout_seconds=30,
            continuation=None,
            output=output,
        ),
        runtime_client_factory=lambda: adapter,
    )

    assert observed["token"] is None


def test_cancel_probe_run_case_cancel_on_first_output_writes_live_feed_and_result_artifacts(
    probe: Any, case_runner: Any, tmp_path: Path
) -> None:
    case = _codex_cancel_case(probe, mode="ephemeral")
    output = _OutputRecorder()

    def _record(method: str, request: Any) -> Any:
        request.on_live_output(
            pr.AgentEvent(
                type="agent_message",
                display_message="hello",
                raw_provider_output='{"event":"message"}',
            )
        )
        return _cancelled_outcome()

    adapter = case_runner.InMemoryRuntimeInvocationAdapter(record_handler=_record)
    case_dir = tmp_path / "cancel_case"

    case_runner.run_case(
        case_runner.ProbeCaseRunRequest(
            case=case,
            case_dir=case_dir,
            invocation_dir=tmp_path / "workspace",
            prompt="prompt",
            timeout_seconds=30,
            continuation=None,
            output=output,
            cancel_on_first_output=True,
        ),
        runtime_client_factory=lambda: adapter,
    )

    assert (case_dir / case_runner.LIVE_FEED_FILENAME).exists()
    assert (case_dir / case_runner.RESULT_FILENAME).exists()
    result_payload = json.loads(
        (case_dir / case_runner.RESULT_FILENAME).read_text(encoding="utf-8")
    )
    assert result_payload["category"] == "cancelled"
    assert result_payload["kind"] == "Cancelled"
    assert result_payload["continuation"] is None


# --------------------------------------------------------------------------- #
# Behavior 2: session-backed cancel — continuation returned when work started
# --------------------------------------------------------------------------- #


def test_cancel_probe_run_case_new_session_cancel_after_output_captures_continuation(
    probe: Any, case_runner: Any, tmp_path: Path
) -> None:
    case = _codex_cancel_case(probe, mode="new_session")
    output = _OutputRecorder()
    session_store = tmp_path / "session-store"
    continuation = _continuation()

    def _record(method: str, request: Any) -> Any:
        request.on_live_output(
            pr.AgentEvent(
                type="agent_message",
                display_message="partial output",
                raw_provider_output='{"event":"message"}',
            )
        )
        return _cancelled_outcome(continuation=continuation)

    adapter = case_runner.InMemoryRuntimeInvocationAdapter(record_handler=_record)
    case_dir = tmp_path / "cancel_new_session"

    result = case_runner.run_case(
        case_runner.ProbeCaseRunRequest(
            case=case,
            case_dir=case_dir,
            invocation_dir=tmp_path / "workspace",
            prompt="prompt",
            timeout_seconds=30,
            continuation=None,
            session_store=session_store,
            output=output,
            cancel_on_first_output=True,
        ),
        runtime_client_factory=lambda: adapter,
    )

    assert result.category == "cancelled"
    assert result.continuation == continuation
    result_payload = json.loads(
        (case_dir / case_runner.RESULT_FILENAME).read_text(encoding="utf-8")
    )
    assert result_payload["category"] == "cancelled"
    assert result_payload["continuation"] == continuation.serialized


def test_cancel_probe_run_case_new_session_cancel_before_output_has_no_continuation(
    probe: Any, case_runner: Any, tmp_path: Path
) -> None:
    case = _codex_cancel_case(probe, mode="new_session")
    output = _OutputRecorder()
    session_store = tmp_path / "session-store"

    def _record(method: str, request: Any) -> Any:
        # Cancel before any provider output — no continuation expected
        return _cancelled_outcome(continuation=None)

    adapter = case_runner.InMemoryRuntimeInvocationAdapter(record_handler=_record)
    case_dir = tmp_path / "cancel_new_session"

    result = case_runner.run_case(
        case_runner.ProbeCaseRunRequest(
            case=case,
            case_dir=case_dir,
            invocation_dir=tmp_path / "workspace",
            prompt="prompt",
            timeout_seconds=30,
            continuation=None,
            session_store=session_store,
            output=output,
            cancel_on_first_output=True,
        ),
        runtime_client_factory=lambda: adapter,
    )

    assert result.category == "cancelled"
    assert result.continuation is None


# --------------------------------------------------------------------------- #
# Behavior 3: full cancel probe entry point writes two cancel cases
# --------------------------------------------------------------------------- #


def _install_cancel_client(
    cancel_probe: Any, monkeypatch: pytest.MonkeyPatch, handler: Any
) -> list:
    calls: list[tuple[str, Any]] = []

    class _FakeClient:
        async def run_ephemeral(self, request: Any) -> Any:
            return self._dispatch("run_ephemeral", request)

        async def run_new_session(self, request: Any) -> Any:
            return self._dispatch("run_new_session", request)

        async def run_resumed_session(self, request: Any) -> Any:
            return self._dispatch("run_resumed_session", request)

        def _dispatch(self, method: str, request: Any) -> Any:
            calls.append((method, request))
            return handler(method, request)

    monkeypatch.setattr(cancel_probe, "RuntimeClient", lambda: _FakeClient())
    return calls


def _default_cancel_handler(method: str, request: Any) -> Any:
    if request.on_live_output is not None:
        request.on_live_output(
            pr.AgentEvent(
                type="agent_message",
                display_message="partial output before cancel",
                raw_provider_output='{"raw": "payload"}',
            )
        )
    if method == "run_new_session":
        return _cancelled_outcome(continuation=_continuation())
    return _cancelled_outcome()


def test_cancel_probe_full_run_writes_two_cancel_case_artifacts_per_provider(
    cancel_probe: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_cancel_client(cancel_probe, monkeypatch, _default_cancel_handler)

    root = cancel_probe.run_cancel_probe(
        ("codex",),
        env={},
        codex_auth_present=True,
        artifact_root=tmp_path / "artifacts",
        stream=io.StringIO(),
    )

    service_dir = root / "codex"
    case_labels = [d.name for d in service_dir.iterdir() if d.is_dir()]
    assert len(case_labels) == 2
    for label in case_labels:
        case_dir = service_dir / label
        assert (case_dir / "live_feed.json").exists(), label
        assert (case_dir / "result.json").exists(), label


def test_cancel_probe_full_run_first_case_is_ephemeral_cancel(
    cancel_probe: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _install_cancel_client(cancel_probe, monkeypatch, _default_cancel_handler)

    cancel_probe.run_cancel_probe(
        ("codex",),
        env={},
        codex_auth_present=True,
        artifact_root=tmp_path / "artifacts",
        stream=io.StringIO(),
    )

    ephemeral_calls = [method for method, _ in calls if method == "run_ephemeral"]
    new_session_calls = [method for method, _ in calls if method == "run_new_session"]
    assert len(ephemeral_calls) == 1
    assert len(new_session_calls) == 1


def test_cancel_probe_full_run_second_case_reports_continuation_status(
    cancel_probe: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_cancel_client(cancel_probe, monkeypatch, _default_cancel_handler)
    stream = io.StringIO()

    root = cancel_probe.run_cancel_probe(
        ("codex",),
        env={},
        codex_auth_present=True,
        artifact_root=tmp_path / "artifacts",
        stream=stream,
    )

    text = stream.getvalue()
    assert "continuation" in text.lower()

    # new_session cancel result has continuation
    service_dir = root / "codex"
    cancel_dirs = sorted(service_dir.iterdir())
    payloads = [
        json.loads((d / "result.json").read_text(encoding="utf-8"))
        for d in cancel_dirs
        if (d / "result.json").exists()
    ]
    session_payload = next(p for p in payloads if p["mode"] == "new_session")
    assert session_payload["continuation"] == _continuation().serialized


def test_cancel_probe_full_run_service_dir_wiped_on_rerun(
    cancel_probe: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_cancel_client(cancel_probe, monkeypatch, _default_cancel_handler)
    root = tmp_path / "artifacts"
    stale = root / "codex" / "stale-from-previous-run.txt"
    stale.parent.mkdir(parents=True)
    stale.write_text("old", encoding="utf-8")

    cancel_probe.run_cancel_probe(
        ("codex",),
        env={},
        codex_auth_present=True,
        artifact_root=root,
        stream=io.StringIO(),
    )

    assert not stale.exists()


def test_cancel_probe_unconfigured_provider_skipped_in_all_mode(
    cancel_probe: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _install_cancel_client(cancel_probe, monkeypatch, _default_cancel_handler)
    stream = io.StringIO()

    cancel_probe.run_cancel_probe(
        "all",
        env={},
        codex_auth_present=False,
        artifact_root=tmp_path / "artifacts",
        stream=stream,
        color=True,
    )

    assert calls == []
    text = stream.getvalue()
    assert "skipped (unconfigured)" in text
    assert "\033[31m" not in text


def test_cancel_probe_main_uses_all_when_no_provider_given(
    cancel_probe: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, Any] = {}

    def _fake_run(provider_selection: Any, **kwargs: Any) -> Path:
        captured["selection"] = provider_selection
        return tmp_path

    monkeypatch.setattr(cancel_probe, "run_cancel_probe", _fake_run)

    exit_code = cancel_probe.main(["--artifact-root", str(tmp_path / "artifacts")])
    assert exit_code == 0
    assert captured["selection"] == "all"

    cancel_probe.main(["codex"])
    assert captured["selection"] == ("codex",)
