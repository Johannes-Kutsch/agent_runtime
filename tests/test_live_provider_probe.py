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

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "live-probe"
    / "live_provider_probe.py"
)


@pytest.fixture
def probe() -> Any:
    spec = importlib.util.spec_from_file_location("live_provider_probe", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # type: ignore[arg-type]
    spec.loader.exec_module(module)
    return module


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


def test_full_run_writes_six_cases_with_feed_and_result(
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
        "ephemeral_INSPECT_ONLY",
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

    methods = [method for method, _ in calls]
    assert methods.index("run_new_session") < methods.index("run_resumed_session")

    resumed_request = next(
        request for method, request in calls if method == "run_resumed_session"
    )
    assert resumed_request.continuation.serialized == _continuation().serialized


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
