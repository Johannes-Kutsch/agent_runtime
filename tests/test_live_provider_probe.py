"""Deterministic tests for the Live Provider Probe runner.

The probe is manual-debug-only operator tooling (ADR 0013). These tests fake
``RuntimeClient`` and inject auth state, so they never reach a live provider or
use real credentials. They cover the parts pytest is meant to own: artifact
layout, the crash-survivable live feed, the result payload, wipe-on-rerun,
provider selection, the coupled new/resumed session pair, and the
outcome-category verdict (including red "run not completed" highlighting).
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
from agent_runtime.errors import AgentCredentialFailureError
from agent_runtime import runtime as pr
import agent_runtime._provider_invocation as provider_invocation_runtime
from agent_runtime.errors import ProviderUnavailableReason

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "live-probe"
    / "live_provider_probe.py"
)
CASE_RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "live-probe"
    / "_live_probe_case_runner.py"
)


@pytest.fixture
def probe() -> Any:
    spec = importlib.util.spec_from_file_location("live_provider_probe", SCRIPT_PATH)
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


def _completed(output: str, *, continuation: Any = None, usage: Any = None) -> Any:
    return pr.RuntimeOutcome(
        kind=pr.Completed(),
        result=pr.RunResult(
            output=output,
            usage=usage,
            continuation=continuation,
            selected=pr.ResolvedProvider(
                service="codex", model="gpt-5.4-mini", effort="low"
            ),
        ),
    )


def _install_client(probe: Any, monkeypatch: pytest.MonkeyPatch, handler: Any) -> list:
    """Install a fake async ``RuntimeClient``; return the recorded calls list."""

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

    monkeypatch.setattr(probe, "RuntimeClient", lambda: _FakeClient())
    return calls


def _default_handler(method: str, request: Any) -> Any:
    if request.on_live_output is not None:
        request.on_live_output(
            pr.AgentEvent(
                type="agent_message",
                display_message="hi there",
                raw_provider_output='{"raw": "full-payload", "method": "%s"}' % method,
            )
        )
    if method == "run_new_session":
        return _completed("new session output", continuation=_continuation())
    if method == "run_resumed_session":
        return _completed("resumed output")
    return _completed("ephemeral output")


def _codex_probe_case(probe: Any, *, mode: str) -> Any:
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


def _probe_case(
    probe: Any,
    *,
    provider: str,
    mode: str,
    env: dict[str, str] | None = None,
    codex_auth_present: bool | None = None,
) -> Any:
    provider_plan = probe.plan.plan_selected_providers(
        probe.plan.parse_provider_selection(provider),
        env=env or {},
        codex_auth_present=codex_auth_present,
    )[0]
    return next(
        case
        for case in probe.plan.probe_cases_for_provider(provider_plan)
        if case.mode == mode
    )


def test_probe_case_matrix_has_five_cases_for_codex(
    probe: Any,
) -> None:
    provider_plan = probe.plan.plan_selected_providers(
        probe.plan.parse_provider_selection("codex"),
        env={},
        codex_auth_present=True,
    )[0]
    cases = probe.plan.probe_cases_for_provider(provider_plan)

    assert [case.label for case in cases] == [
        "ephemeral_UNRESTRICTED",
        "new_session_UNRESTRICTED",
        "resumed_session_UNRESTRICTED",
        "ephemeral_NONE",
        "ephemeral_NO_FILE_MUTATION",
    ]


def test_full_run_writes_five_cases_with_feed_and_result(
    probe: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_client(probe, monkeypatch, _default_handler)

    root = probe.run_probe(
        ("codex",),
        env={},
        codex_auth_present=True,
        artifact_root=tmp_path / "artifacts",
        stream=io.StringIO(),
    )

    service_dir = root / "codex"
    expected = [
        "ephemeral_UNRESTRICTED",
        "new_session_UNRESTRICTED",
        "resumed_session_UNRESTRICTED",
        "ephemeral_NONE",
        "ephemeral_NO_FILE_MUTATION",
    ]
    for label in expected:
        case_dir = service_dir / label
        assert (case_dir / probe.LIVE_FEED_FILENAME).exists(), label
        assert (case_dir / probe.RESULT_FILENAME).exists(), label


def test_live_feed_is_json_lines_with_full_raw_output(
    probe: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_client(probe, monkeypatch, _default_handler)

    root = probe.run_probe(
        ("codex",),
        env={},
        codex_auth_present=True,
        artifact_root=tmp_path / "artifacts",
        stream=io.StringIO(),
    )

    feed_path = root / "codex" / "ephemeral_UNRESTRICTED" / probe.LIVE_FEED_FILENAME
    lines = [
        line for line in feed_path.read_text(encoding="utf-8").splitlines() if line
    ]
    records = [json.loads(line) for line in lines]
    assert records == [
        {
            "type": "agent_message",
            "display_message": "hi there",
            "raw_provider_output": '{"raw": "full-payload", "method": "run_ephemeral"}',
        }
    ]


def test_result_json_carries_outcome_facts(
    probe: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    usage = pr.ProviderUsage(input_tokens=10, output_tokens=3, cost_usd=0.01)

    def handler(method: str, request: Any) -> Any:
        if method == "run_new_session":
            return _completed(
                "new session output", continuation=_continuation(), usage=usage
            )
        if method == "run_resumed_session":
            return _completed("resumed output")
        return _completed("ephemeral output", usage=usage)

    _install_client(probe, monkeypatch, handler)

    root = probe.run_probe(
        ("codex",),
        env={},
        codex_auth_present=True,
        artifact_root=tmp_path / "artifacts",
        stream=io.StringIO(),
    )

    payload = json.loads(
        (root / "codex" / "ephemeral_UNRESTRICTED" / probe.RESULT_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert payload["service"] == "codex"
    assert payload["mode"] == "ephemeral"
    assert payload["tool_policy"] == "UNRESTRICTED"
    assert payload["category"] == "success"
    assert payload["kind"] == "Completed"
    assert payload["selected"] == {
        "service": "codex",
        "model": "gpt-5.4-mini",
        "effort": "low",
    }
    assert payload["output"] == "ephemeral output"
    assert payload["usage"]["input_tokens"] == 10
    assert payload["usage"]["cost_usd"] == 0.01
    assert payload["traceback"] is None

    # new_session result carries the continuation token.
    new_session = json.loads(
        (root / "codex" / "new_session_UNRESTRICTED" / probe.RESULT_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert new_session["continuation"] == _continuation().serialized


def test_resumed_session_receives_new_session_continuation(
    probe: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _install_client(probe, monkeypatch, _default_handler)

    probe.run_probe(
        ("codex",),
        env={},
        codex_auth_present=True,
        artifact_root=tmp_path / "artifacts",
        stream=io.StringIO(),
    )

    resumed_request = {
        method: request for method, request in calls if method == "run_resumed_session"
    }["run_resumed_session"]
    assert resumed_request.continuation.serialized == _continuation().serialized
    assert resumed_request.invocation_dir == (
        tmp_path / "artifacts" / "codex" / "new_session_UNRESTRICTED"
    )


def test_full_run_invokes_live_probe_case_matrix_in_order(
    probe: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _install_client(probe, monkeypatch, _default_handler)

    probe.run_probe(
        ("codex",),
        env={},
        codex_auth_present=True,
        artifact_root=tmp_path / "artifacts",
        stream=io.StringIO(),
    )

    assert [
        (
            method,
            request.invocation_dir.relative_to(tmp_path / "artifacts" / "codex"),
        )
        for method, request in calls
    ] == [
        ("run_ephemeral", Path("ephemeral_UNRESTRICTED")),
        ("run_new_session", Path("new_session_UNRESTRICTED")),
        ("run_resumed_session", Path("new_session_UNRESTRICTED")),
        ("run_ephemeral", Path("ephemeral_NONE")),
        ("run_ephemeral", Path("ephemeral_NO_FILE_MUTATION")),
    ]


def test_non_success_category_prints_red_and_records_category(
    probe: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def handler(method: str, request: Any) -> Any:
        return pr.RuntimeOutcome(
            kind=pr.UsageLimited(None),
            result=pr.RunResult(
                output="limited",
                usage=None,
                continuation=None,
                selected=pr.ResolvedProvider(
                    service="codex", model="gpt-5.4-mini", effort="low"
                ),
            ),
        )

    _install_client(probe, monkeypatch, handler)
    stream = io.StringIO()

    root = probe.run_probe(
        ("codex",),
        env={},
        codex_auth_present=True,
        artifact_root=tmp_path / "artifacts",
        stream=stream,
        color=True,
    )

    text = stream.getvalue()
    assert "run not completed: usage_limited" in text
    assert "\033[31m" in text  # red highlighting

    payload = json.loads(
        (root / "codex" / "ephemeral_UNRESTRICTED" / probe.RESULT_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert payload["category"] == "usage_limited"
    assert payload["kind"] == "UsageLimited"


def test_retryable_provider_unavailable_is_retryable_failure_category(
    probe: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def handler(method: str, request: Any) -> Any:
        if method == "run_ephemeral":
            return pr.RuntimeOutcome(
                kind=pr.ProviderUnavailable(
                    reason=ProviderUnavailableReason.TRANSIENT_API_ERROR,
                    detail="Selected model is at capacity. Please try a different model.",
                ),
                result=pr.RunResult(
                    output="",
                    usage=None,
                    continuation=None,
                    selected=pr.ResolvedProvider(
                        service="codex",
                        model="gpt-5.4-mini",
                        effort="low",
                    ),
                ),
            )
        if method == "run_new_session":
            return _completed("new session output")
        return _completed("resumed output")

    _install_client(probe, monkeypatch, handler)
    stream = io.StringIO()

    root = probe.run_probe(
        ("codex",),
        env={},
        codex_auth_present=True,
        artifact_root=tmp_path / "artifacts",
        stream=stream,
        color=True,
    )

    text = stream.getvalue()
    assert "run not completed: retryable_failure" in text
    assert "\033[31m" in text
    payload = json.loads(
        (root / "codex" / "ephemeral_UNRESTRICTED" / probe.RESULT_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert payload["category"] == "retryable_failure"
    assert payload["kind"] == "ProviderUnavailable"


def test_credential_failure_is_wrong_credentials_with_traceback(
    probe: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def handler(method: str, request: Any) -> Any:
        raise AgentCredentialFailureError("bad token", service_name="codex")

    _install_client(probe, monkeypatch, handler)

    root = probe.run_probe(
        ("codex",),
        env={},
        codex_auth_present=True,
        artifact_root=tmp_path / "artifacts",
        stream=io.StringIO(),
    )

    payload = json.loads(
        (root / "codex" / "ephemeral_UNRESTRICTED" / probe.RESULT_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert payload["category"] == "wrong_credentials"
    assert payload["kind"] is None
    assert "AgentCredentialFailureError" in payload["traceback"]


def test_unexpected_exception_is_error_category(
    probe: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def handler(method: str, request: Any) -> Any:
        raise RuntimeError("boom")

    _install_client(probe, monkeypatch, handler)

    root = probe.run_probe(
        ("codex",),
        env={},
        codex_auth_present=True,
        artifact_root=tmp_path / "artifacts",
        stream=io.StringIO(),
    )

    payload = json.loads(
        (root / "codex" / "ephemeral_UNRESTRICTED" / probe.RESULT_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert payload["category"] == "error"
    assert "RuntimeError: boom" in payload["traceback"]


def test_service_dir_is_wiped_on_rerun(
    probe: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_client(probe, monkeypatch, _default_handler)
    root = tmp_path / "artifacts"
    stale = root / "codex" / "stale-from-previous-run.txt"
    stale.parent.mkdir(parents=True)
    stale.write_text("old", encoding="utf-8")

    probe.run_probe(
        ("codex",),
        env={},
        codex_auth_present=True,
        artifact_root=root,
        stream=io.StringIO(),
    )

    assert not stale.exists()
    assert (root / "codex" / "ephemeral_UNRESTRICTED" / probe.RESULT_FILENAME).exists()


def test_all_mode_skips_unconfigured_without_wipe_or_client_call(
    probe: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _install_client(probe, monkeypatch, _default_handler)
    root = tmp_path / "artifacts"

    stream = io.StringIO()
    probe.run_probe(
        "all",
        env={},
        codex_auth_present=False,
        artifact_root=root,
        stream=stream,
        color=True,
    )

    # No provider configured -> nothing runs, no per-service dirs.
    assert calls == []
    assert not (root / "claude").exists()
    assert not (root / "codex").exists()
    assert not (root / "opencode").exists()
    text = stream.getvalue()
    assert "skipped (unconfigured)" in text
    # all-mode skips are expected, not failures -> not red.
    assert "\033[31m" not in text


def test_explicit_unconfigured_provider_surfaces_red_without_client_call(
    probe: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _install_client(probe, monkeypatch, _default_handler)
    root = tmp_path / "artifacts"
    stream = io.StringIO()

    probe.run_probe(
        ("claude",),
        env={},
        codex_auth_present=False,
        artifact_root=root,
        stream=stream,
        color=True,
    )

    assert calls == []
    assert not (root / "claude").exists()
    text = stream.getvalue()
    assert "claude: not configured" in text
    assert "\033[31m" in text


def test_artifacts_do_not_leak_credentials(
    probe: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_client(probe, monkeypatch, _default_handler)

    root = probe.run_probe(
        ("claude",),
        env={"CLAUDE_CODE_OAUTH_TOKEN": "super-secret-token"},
        artifact_root=tmp_path / "artifacts",
        stream=io.StringIO(),
    )

    for path in (root / "claude").rglob("*.json"):
        assert "super-secret-token" not in path.read_text(encoding="utf-8")


def test_main_uses_all_when_no_provider_given(
    probe: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, Any] = {}

    def _fake_run_probe(provider_selection: Any, **kwargs: Any) -> Path:
        captured["selection"] = provider_selection
        captured["kwargs"] = kwargs
        return tmp_path

    monkeypatch.setattr(probe, "run_probe", _fake_run_probe)

    exit_code = probe.main(["--artifact-root", str(tmp_path / "artifacts")])
    assert exit_code == 0
    assert captured["selection"] == "all"

    probe.main(["codex"])
    assert captured["selection"] == ("codex",)


def test_live_probe_case_runner_writes_feed_and_projects_case_result_facts(
    probe: Any, case_runner: Any, tmp_path: Path
) -> None:
    case = _codex_probe_case(probe, mode="new_session")
    output = _OutputRecorder()
    usage = pr.ProviderUsage(input_tokens=10, output_tokens=3, cost_usd=0.01)

    def _record(method: str, request: Any) -> Any:
        if method == "run_new_session":
            request.on_live_output(
                pr.AgentEvent(
                    type="agent_message",
                    display_message="hello",
                    raw_provider_output='{"event":"message"}',
                )
            )
            request.on_live_output(
                pr.AgentEvent(
                    type="other",
                    display_message="hidden",
                    raw_provider_output='{"event":"other"}',
                )
            )
            return _completed(
                "new session output", continuation=_continuation(), usage=usage
            )
        return None

    adapter = case_runner.InMemoryRuntimeInvocationAdapter(
        record_handler=_record,
    )

    result = case_runner.run_case(
        case_runner.ProbeCaseRunRequest(
            case=case,
            case_dir=tmp_path / case.label,
            invocation_dir=tmp_path / "workspace",
            prompt="prompt",
            timeout_seconds=123,
            continuation=None,
            output=output,
        ),
        runtime_client_factory=lambda: adapter,
    )

    assert result.category == "success"
    assert result.kind == "Completed"
    assert result.selected == {
        "service": "codex",
        "model": "gpt-5.4-mini",
        "effort": "low",
    }
    assert result.output == "new session output"
    assert result.usage is not None
    assert result.usage["input_tokens"] == 10
    assert result.continuation == _continuation()
    assert result.next_resumed_session_continuation == _continuation()
    assert result.next_resumed_session_invocation_dir == tmp_path / "workspace"
    assert result.traceback is None
    assert output.lines == ["  hello"]
    assert (tmp_path / "workspace").exists()
    assert json.loads(
        (tmp_path / case.label / case_runner.RESULT_FILENAME).read_text(
            encoding="utf-8"
        )
    ) == {
        "category": "success",
        "continuation": _continuation().serialized,
        "kind": "Completed",
        "mode": "new_session",
        "output": "new session output",
        "selected": {
            "service": "codex",
            "model": "gpt-5.4-mini",
            "effort": "low",
        },
        "service": "codex",
        "tool_policy": "UNRESTRICTED",
        "traceback": None,
        "usage": {
            "cache_creation_input_tokens": None,
            "cache_read_input_tokens": None,
            "cost_usd": 0.01,
            "duration_seconds": None,
            "input_tokens": 10,
            "output_tokens": 3,
        },
    }

    records = [
        json.loads(line)
        for line in (tmp_path / case.label / case_runner.LIVE_FEED_FILENAME)
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert records == [
        {
            "type": "agent_message",
            "display_message": "hello",
            "raw_provider_output": '{"event":"message"}',
        },
        {
            "type": "other",
            "display_message": "hidden",
            "raw_provider_output": '{"event":"other"}',
        },
    ]


def test_live_probe_case_runner_streams_tool_call_display_messages_and_persists_partial_feed_on_error(
    probe: Any, case_runner: Any, tmp_path: Path
) -> None:
    case = _codex_probe_case(probe, mode="ephemeral")
    output = _OutputRecorder()

    def _record(method: str, request: Any) -> Any:
        if method != "run_ephemeral":
            return None
        request.on_live_output(
            pr.AgentEvent(
                type="agent_message",
                display_message="hello",
                raw_provider_output='{"event":"message"}',
            )
        )
        request.on_live_output(
            pr.AgentEvent(
                type="agent_tool_call",
                display_message="tool call",
                raw_provider_output='{"event":"tool_call"}',
            )
        )
        request.on_live_output(
            pr.AgentEvent(
                type="other",
                display_message="hidden",
                raw_provider_output='{"event":"other"}',
            )
        )
        raise RuntimeError("boom")

    adapter = case_runner.InMemoryRuntimeInvocationAdapter(
        record_handler=_record,
    )

    result = case_runner.run_case(
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

    assert result.category == "error"
    assert result.kind is None
    assert result.traceback is not None
    assert "RuntimeError: boom" in result.traceback
    assert output.lines == ["  hello", "  tool call"]
    payload = json.loads(
        (tmp_path / case.label / case_runner.RESULT_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert payload["category"] == "error"
    assert payload["kind"] is None
    assert payload["traceback"] is not None
    assert "RuntimeError: boom" in payload["traceback"]
    records = [
        json.loads(line)
        for line in (tmp_path / case.label / case_runner.LIVE_FEED_FILENAME)
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert records == [
        {
            "type": "agent_message",
            "display_message": "hello",
            "raw_provider_output": '{"event":"message"}',
        },
        {
            "type": "agent_tool_call",
            "display_message": "tool call",
            "raw_provider_output": '{"event":"tool_call"}',
        },
        {
            "type": "other",
            "display_message": "hidden",
            "raw_provider_output": '{"event":"other"}',
        },
    ]


def test_live_probe_case_runner_flushes_each_observed_event(
    probe: Any, case_runner: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    case = _codex_probe_case(probe, mode="ephemeral")
    output = _OutputRecorder()

    class _RecordingSink:
        def __init__(self) -> None:
            self.flush_count = 0
            self.lines: list[str] = []

        def write(self, text: str) -> int:
            self.lines.append(text)
            return len(text)

        def flush(self) -> None:
            self.flush_count += 1

        def close(self) -> None:
            return None

    sink = _RecordingSink()
    original_open = Path.open

    def _open(self: Path, mode: str = "r", encoding: str | None = None) -> Any:
        if self == tmp_path / case.label / case_runner.LIVE_FEED_FILENAME:
            assert mode == "w"
            assert encoding == "utf-8"
            return sink
        return original_open(self, mode, encoding=encoding)

    monkeypatch.setattr(case_runner.Path, "open", _open)

    def _record(method: str, request: Any) -> Any:
        if method != "run_ephemeral":
            return None
        request.on_live_output(
            pr.AgentEvent(
                type="agent_message",
                display_message="hello",
                raw_provider_output='{"event":"message"}',
            )
        )
        request.on_live_output(
            pr.AgentEvent(
                type="other",
                display_message="hidden",
                raw_provider_output='{"event":"other"}',
            )
        )
        return _completed("ephemeral output")

    adapter = case_runner.InMemoryRuntimeInvocationAdapter(
        record_handler=_record,
    )

    result = case_runner.run_case(
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

    assert result.category == "success"
    assert sink.flush_count == 2
    assert [json.loads(line) for line in sink.lines] == [
        {
            "type": "agent_message",
            "display_message": "hello",
            "raw_provider_output": '{"event":"message"}',
        },
        {
            "type": "other",
            "display_message": "hidden",
            "raw_provider_output": '{"event":"other"}',
        },
    ]


def test_live_probe_case_runner_passes_continuation_and_default_provider_auth_for_resumed_session(
    probe: Any, case_runner: Any, tmp_path: Path
) -> None:
    case = _codex_probe_case(probe, mode="resumed_session")
    output = _OutputRecorder()
    observed: dict[str, Any] = {}

    def _record(method: str, request: Any) -> Any:
        if method != "run_resumed_session":
            return None
        observed["continuation"] = request.continuation
        observed["provider_auth"] = request.provider_auth
        observed["invocation_dir"] = request.invocation_dir
        return _completed("resumed output")

    adapter = case_runner.InMemoryRuntimeInvocationAdapter(
        record_handler=_record,
    )

    result = case_runner.run_case(
        case_runner.ProbeCaseRunRequest(
            case=case,
            case_dir=tmp_path / case.label,
            invocation_dir=tmp_path / "workspace",
            prompt="resume prompt",
            timeout_seconds=45,
            continuation=_continuation(),
            output=output,
        ),
        runtime_client_factory=lambda: adapter,
    )

    assert result.category == "success"
    assert observed["continuation"] == _continuation()
    assert observed["provider_auth"] == pr.ProviderAuth()
    assert observed["invocation_dir"] == tmp_path / "workspace"
    assert output.lines == []


def test_live_probe_case_runner_uses_returned_resumed_session_invocation_dir(
    probe: Any, case_runner: Any, tmp_path: Path
) -> None:
    case = _codex_probe_case(probe, mode="resumed_session")
    output = _OutputRecorder()
    observed: dict[str, Any] = {}
    resumed_session_invocation_dir = tmp_path / "new-session-workspace"

    def _record(method: str, request: Any) -> Any:
        if method != "run_resumed_session":
            return None
        observed["invocation_dir"] = request.invocation_dir
        return _completed("resumed output")

    adapter = case_runner.InMemoryRuntimeInvocationAdapter(record_handler=_record)

    result = case_runner.run_case(
        case_runner.ProbeCaseRunRequest(
            case=case,
            case_dir=tmp_path / case.label,
            invocation_dir=tmp_path / "resumed-session-workspace",
            resumed_session_invocation_dir=resumed_session_invocation_dir,
            prompt="resume prompt",
            timeout_seconds=45,
            continuation=_continuation(),
            output=output,
        ),
        runtime_client_factory=lambda: adapter,
    )

    assert result.category == "success"
    assert result.next_resumed_session_continuation is None
    assert result.next_resumed_session_invocation_dir is None
    assert observed["invocation_dir"] == resumed_session_invocation_dir
    assert resumed_session_invocation_dir.exists()


def test_live_probe_case_runner_reports_wrong_credentials_with_traceback(
    probe: Any, case_runner: Any, tmp_path: Path
) -> None:
    case = _codex_probe_case(probe, mode="ephemeral")
    output = _OutputRecorder()

    def _record(method: str, request: Any) -> Any:
        if method == "run_ephemeral":
            raise AgentCredentialFailureError("bad token", service_name="codex")
        return None

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
        ),
        runtime_client_factory=lambda: adapter,
    )

    assert result.category == "wrong_credentials"
    assert result.kind is None
    assert result.selected is None
    assert result.output is None
    assert result.usage is None
    assert result.continuation is None
    assert result.traceback is not None
    assert "AgentCredentialFailureError" in result.traceback
    assert output.lines == []
    payload = json.loads(
        (tmp_path / case.label / case_runner.RESULT_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert payload["category"] == "wrong_credentials"
    assert payload["kind"] is None
    assert payload["traceback"] is not None
    assert "AgentCredentialFailureError" in payload["traceback"]
    assert (tmp_path / case.label / case_runner.LIVE_FEED_FILENAME).read_text(
        encoding="utf-8"
    ) == ""


def test_live_probe_case_runner_in_memory_runtime_invocation_adapter_records_request_facts_for_all_lifecycle_modes(
    probe: Any, case_runner: Any, tmp_path: Path
) -> None:
    ephemeral_case = _probe_case(
        probe,
        provider="codex",
        mode="ephemeral",
        codex_auth_present=True,
    )
    new_session_case = _probe_case(
        probe,
        provider="codex",
        mode="new_session",
        codex_auth_present=True,
    )
    resumed_session_case = _probe_case(
        probe,
        provider="claude",
        mode="resumed_session",
        env={"CLAUDE_CODE_OAUTH_TOKEN": "claude-token"},
    )
    resumed_invocation_dir = tmp_path / "resumed-session-workspace"
    continuation = pr.Continuation(
        selected_service="claude",
        selected_model=resumed_session_case.model,
        selected_effort=resumed_session_case.effort,
        tool_access=ToolAccess.workspace_backed(
            resumed_invocation_dir,
            tool_policy=pr.ToolPolicy.UNRESTRICTED,
        ),
        provider_resume_state={"provider_session_id": "claude-session"},
    )
    adapter = case_runner.InMemoryRuntimeInvocationAdapter(
        prepared_outcomes=[
            _completed("ephemeral output"),
            _completed("new session output", continuation=_continuation()),
            _completed("resumed output"),
        ]
    )
    output = _OutputRecorder()

    for case, prompt, case_dir_name, case_continuation in (
        (ephemeral_case, "ephemeral prompt", "ephemeral", None),
        (new_session_case, "new session prompt", "new-session", None),
        (
            resumed_session_case,
            "resumed prompt",
            "resumed-session",
            continuation,
        ),
    ):
        result = case_runner.run_case(
            case_runner.ProbeCaseRunRequest(
                case=case,
                case_dir=tmp_path / case_dir_name,
                invocation_dir=tmp_path / f"{case_dir_name}-workspace",
                prompt=prompt,
                timeout_seconds=45,
                continuation=case_continuation,
                output=output,
            ),
            runtime_client_factory=lambda: adapter,
        )
        assert result.category == "success"

    assert {mode for mode, _ in adapter.recorded_requests} == {
        "run_ephemeral",
        "run_new_session",
        "run_resumed_session",
    }

    recorded_requests = {mode: request for mode, request in adapter.recorded_requests}
    ephemeral_request = recorded_requests["run_ephemeral"]
    assert isinstance(ephemeral_request, pr.EphemeralRunRequest)
    assert ephemeral_request.provider_selection == ephemeral_case.provider_selection
    assert ephemeral_request.tool_policy is pr.ToolPolicy.UNRESTRICTED
    assert ephemeral_request.prompt == "ephemeral prompt"
    assert ephemeral_request.timeout_seconds == 45
    assert ephemeral_request.invocation_dir == tmp_path / "ephemeral-workspace"

    new_session_request = recorded_requests["run_new_session"]
    assert isinstance(new_session_request, pr.NewSessionRunRequest)
    assert new_session_request.provider_selection == new_session_case.provider_selection
    assert new_session_request.tool_policy is pr.ToolPolicy.UNRESTRICTED
    assert new_session_request.prompt == "new session prompt"
    assert new_session_request.timeout_seconds == 45
    assert new_session_request.invocation_dir == tmp_path / "new-session-workspace"

    resumed_session_request = recorded_requests["run_resumed_session"]
    assert isinstance(resumed_session_request, pr.ResumedSessionRunRequest)
    assert resumed_session_request.continuation == continuation
    assert resumed_session_request.provider_auth == pr.ProviderAuth(
        claude_code_oauth_token="claude-token"
    )
    assert resumed_session_request.prompt == "resumed prompt"
    assert resumed_session_request.timeout_seconds == 45
    assert resumed_session_request.invocation_dir == (
        tmp_path / "resumed-session-workspace"
    )


def test_live_probe_case_runner_threads_public_session_store_between_new_and_resumed_session_cases(
    probe: Any, case_runner: Any, tmp_path: Path
) -> None:
    new_session_case = _probe_case(
        probe,
        provider="codex",
        mode="new_session",
        codex_auth_present=True,
    )
    resumed_session_case = _probe_case(
        probe,
        provider="codex",
        mode="resumed_session",
        codex_auth_present=True,
    )
    adapter = case_runner.InMemoryRuntimeInvocationAdapter(
        prepared_outcomes=[
            _completed("new session output", continuation=_continuation()),
            _completed("resumed output"),
        ]
    )
    output = _OutputRecorder()
    session_store = tmp_path / "session-store"

    first_result = case_runner.run_case(
        case_runner.ProbeCaseRunRequest(
            case=new_session_case,
            case_dir=tmp_path / "new-session",
            invocation_dir=tmp_path / "new-session-workspace",
            prompt="first prompt",
            timeout_seconds=45,
            continuation=None,
            session_store=session_store,
            output=output,
        ),
        runtime_client_factory=lambda: adapter,
    )
    assert first_result.category == "success"

    second_result = case_runner.run_case(
        case_runner.ProbeCaseRunRequest(
            case=resumed_session_case,
            case_dir=tmp_path / "resumed-session",
            invocation_dir=tmp_path / "resumed-session-workspace",
            resumed_session_invocation_dir=tmp_path / "new-session-workspace",
            prompt="second prompt",
            timeout_seconds=45,
            continuation=_continuation(),
            session_store=session_store,
            output=output,
        ),
        runtime_client_factory=lambda: adapter,
    )
    assert second_result.category == "success"

    requests = {method: request for method, request in adapter.recorded_requests}
    assert isinstance(requests["run_new_session"], pr.NewSessionRunRequest)
    assert isinstance(requests["run_resumed_session"], pr.ResumedSessionRunRequest)
    assert requests["run_new_session"].session_store == session_store
    assert requests["run_resumed_session"].session_store == session_store
    assert (
        requests["run_new_session"].session_store
        == requests["run_resumed_session"].session_store
    )


def test_live_probe_resumed_session_case_reaches_classified_outcome_when_session_store_is_threaded(
    probe: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[Any] = []

    def _record(method: str, request: Any) -> Any:
        calls.append((method, request))
        if method == "run_new_session":
            assert request.session_store is not None
            return _completed("new session output", continuation=_continuation())
        if method == "run_resumed_session":
            assert request.session_store is not None
            new_request = next(
                request_for_case
                for method_for_case, request_for_case in calls
                if method_for_case == "run_new_session"
            )
            assert request.session_store == new_request.session_store
            return _completed("resumed output")
        return _completed("ephemeral output")

    _install_client(probe, monkeypatch, _record)

    root = probe.run_probe(
        ("codex",),
        env={},
        codex_auth_present=True,
        artifact_root=tmp_path / "artifacts",
        stream=io.StringIO(),
    )

    payload = json.loads(
        (
            root / "codex" / "resumed_session_UNRESTRICTED" / probe.RESULT_FILENAME
        ).read_text(encoding="utf-8")
    )
    assert payload["category"] == "success"
    assert payload["kind"] == "Completed"
    assert len(calls) == 5


def test_live_probe_resumed_session_case_uses_sandbox_before_resume_token(
    probe: Any, case_runner: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    resumed_session_case = _probe_case(
        probe,
        provider="codex",
        mode="resumed_session",
        codex_auth_present=True,
    )
    session_store = tmp_path / "session-store"

    class _CheckingAdapter(
        provider_invocation_runtime.InMemoryProviderInvocationAdapter
    ):
        def execute(
            self,
            request: provider_invocation_runtime.ProviderInvocationRequest,
        ) -> (
            provider_invocation_runtime.ProviderInvocationResult
            | provider_invocation_runtime.ProviderInvocationFailure
        ):
            if request.argv[:3] != ("codex", "exec", "--sandbox"):
                raise RuntimeError("unexpected argument '--sandbox' found")
            return super().execute(request)

    adapter = _CheckingAdapter(
        prepared_invocations=[
            provider_invocation_runtime.ProviderInvocationResult(
                output="resumed output",
                stdout_lines=(
                    '{"type":"thread.started","thread_id":"codex-session"}\n',
                    '{"type":"item.completed","item":{"type":"agent_message","text":"resumed output"}}\n',
                    '{"type":"turn.completed"}\n',
                ),
                provider_session_id="codex-session",
            )
        ]
    )

    def _runtime_client_factory() -> pr.RuntimeClient:
        monkeypatch.setattr(
            pr._builtin_runtime_client_module,
            "_default_provider_invocation_adapter",
            lambda: adapter,
        )
        return pr.RuntimeClient()

    result = case_runner.run_case(
        case_runner.ProbeCaseRunRequest(
            case=resumed_session_case,
            case_dir=tmp_path / "resumed-session",
            invocation_dir=tmp_path / "resumed-session-workspace",
            prompt="resumed prompt",
            timeout_seconds=45,
            continuation=_continuation(),
            session_store=session_store,
            output=_OutputRecorder(),
        ),
        runtime_client_factory=_runtime_client_factory,
    )

    assert result.category == "success"


def test_live_probe_uses_one_per_session_store_for_new_and_resumed_session_cases(
    probe: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _install_client(probe, monkeypatch, _default_handler)

    root = probe.run_probe(
        ("codex",),
        env={},
        codex_auth_present=True,
        artifact_root=tmp_path / "artifacts",
        stream=io.StringIO(),
    )

    new_session_request = next(
        request for method, request in calls if method == "run_new_session"
    )
    resumed_session_request = next(
        request for method, request in calls if method == "run_resumed_session"
    )
    expected_session_store = (
        root / "codex" / "new_session_UNRESTRICTED" / "_session_store"
    )

    assert new_session_request.session_store == expected_session_store
    assert resumed_session_request.session_store == expected_session_store
