from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_runtime._built_in_provider_session_invocation_dispatch import (
    dispatch_built_in_provider_session_invocation,
)
from agent_runtime._provider_invocation import (
    InMemoryProviderInvocationAdapter,
    ProviderInvocationPreparedStream,
)
from agent_runtime._runtime_lifecycle import ProviderAuth
from agent_runtime.contracts import ToolAccess
from agent_runtime.errors import AgentCredentialFailureError
from agent_runtime.session import RunKind


_CLAUDE_ASSISTANT_LINE = (
    json.dumps(
        {
            "type": "assistant",
            "message": {
                "content": [{"type": "text", "text": "hello from claude"}],
                "usage": {"input_tokens": 5},
            },
        }
    )
    + "\n"
)
_CLAUDE_RESULT_LINE = (
    json.dumps({"type": "result", "subtype": "success", "result": "hello from claude"})
    + "\n"
)
_CLAUDE_LINES = (_CLAUDE_ASSISTANT_LINE, _CLAUDE_RESULT_LINE)

_CODEX_THREAD_LINE = (
    json.dumps({"type": "thread.started", "thread_id": "thread-abc"}) + "\n"
)
_CODEX_ITEM_LINE = (
    json.dumps(
        {
            "type": "item.completed",
            "item": {"type": "agent_message", "text": "hello from codex"},
        }
    )
    + "\n"
)
_CODEX_USAGE_LINE = (
    json.dumps(
        {
            "type": "turn.completed",
            "usage": {"input_tokens": 8, "output_tokens": 4},
        }
    )
    + "\n"
)
_CODEX_LINES = (_CODEX_THREAD_LINE, _CODEX_ITEM_LINE, _CODEX_USAGE_LINE)

_OPENCODE_TEXT_LINE = (
    json.dumps(
        {
            "type": "text",
            "sessionID": "sess-obs",
            "part": {
                "type": "text",
                "text": "hello from opencode",
                "time": {"end": True},
            },
        }
    )
    + "\n"
)
_OPENCODE_IDLE_LINE = (
    json.dumps(
        {
            "type": "session.status",
            "sessionID": "sess-obs",
            "status": {"type": "idle"},
        }
    )
    + "\n"
)
_OPENCODE_LINES = (_OPENCODE_TEXT_LINE, _OPENCODE_IDLE_LINE)


def test_codex_dispatch_raises_credential_failure_when_auth_absent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    no_auth_home = tmp_path / "no-auth-home"
    no_auth_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: no_auth_home)

    adapter = InMemoryProviderInvocationAdapter()

    with pytest.raises(AgentCredentialFailureError):
        dispatch_built_in_provider_session_invocation(
            service_name="codex",
            run_kind=RunKind.FRESH,
            invocation_dir=tmp_path,
            prompt="hello",
            model="gpt-5.5",
            effort="medium",
            tool_access=ToolAccess.no_tools(),
            auth=None,
            provider_state_dir=None,
            provider_session_id=None,
            provider_invocation_adapter=adapter,
        )


def test_claude_dispatch_delivers_claude_argv_to_provider_invocation_adapter(
    tmp_path: Path,
) -> None:
    adapter = InMemoryProviderInvocationAdapter(
        prepared_invocations=[
            ProviderInvocationPreparedStream(stdout_lines=_CLAUDE_LINES)
        ]
    )

    dispatch_built_in_provider_session_invocation(
        service_name="claude",
        run_kind=RunKind.FRESH,
        invocation_dir=tmp_path,
        prompt="hello",
        model="sonnet",
        effort="medium",
        tool_access=ToolAccess.no_tools(),
        auth=ProviderAuth(claude_code_oauth_token="tok-test"),
        provider_state_dir=None,
        provider_session_id=None,
        provider_invocation_adapter=adapter,
    )

    assert len(adapter.recorded_requests) == 1
    assert adapter.recorded_requests[0].argv[0] == "claude"


def test_codex_dispatch_delivers_codex_argv_to_provider_invocation_adapter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    host_home = tmp_path / "host-home"
    auth_path = host_home / ".codex" / "auth.json"
    auth_path.parent.mkdir(parents=True)
    auth_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: host_home)

    adapter = InMemoryProviderInvocationAdapter(
        prepared_invocations=[
            ProviderInvocationPreparedStream(stdout_lines=_CODEX_LINES)
        ]
    )

    dispatch_built_in_provider_session_invocation(
        service_name="codex",
        run_kind=RunKind.FRESH,
        invocation_dir=tmp_path,
        prompt="hello",
        model="gpt-5.5",
        effort="medium",
        tool_access=ToolAccess.no_tools(),
        auth=None,
        provider_state_dir=None,
        provider_session_id=None,
        provider_invocation_adapter=adapter,
    )

    assert len(adapter.recorded_requests) == 1
    request = adapter.recorded_requests[0]
    assert request.argv[0] in {"codex", "codex.cmd"}
    assert request.argv[1] == "exec"


def test_codex_dispatch_with_argv_transform_uses_danger_full_access_sandbox(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    host_home = tmp_path / "host-home"
    auth_path = host_home / ".codex" / "auth.json"
    auth_path.parent.mkdir(parents=True)
    auth_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: host_home)

    adapter = InMemoryProviderInvocationAdapter(
        prepared_invocations=[
            ProviderInvocationPreparedStream(stdout_lines=_CODEX_LINES)
        ]
    )

    def identity_transform(
        argv: tuple[str, ...], prompt_path: Path, env: dict[str, str]
    ) -> tuple[str, ...]:
        return argv

    dispatch_built_in_provider_session_invocation(
        service_name="codex",
        run_kind=RunKind.FRESH,
        invocation_dir=tmp_path,
        prompt="hello",
        model="gpt-5.5",
        effort="medium",
        tool_access=ToolAccess.no_tools(),
        auth=None,
        provider_state_dir=None,
        provider_session_id=None,
        argv_transform=identity_transform,
        provider_invocation_adapter=adapter,
    )

    assert len(adapter.recorded_requests) == 1
    request = adapter.recorded_requests[0]
    assert "--sandbox" in request.argv
    sandbox_idx = request.argv.index("--sandbox")
    assert request.argv[sandbox_idx + 1] == "danger-full-access"


def test_opencode_dispatch_delivers_opencode_argv_to_provider_invocation_adapter(
    tmp_path: Path,
) -> None:
    adapter = InMemoryProviderInvocationAdapter(
        prepared_invocations=[
            ProviderInvocationPreparedStream(stdout_lines=_OPENCODE_LINES)
        ]
    )

    dispatch_built_in_provider_session_invocation(
        service_name="opencode",
        run_kind=RunKind.FRESH,
        invocation_dir=tmp_path,
        prompt="hello",
        model="kimi-k2.6",
        effort="medium",
        tool_access=ToolAccess.no_tools(),
        auth=ProviderAuth(opencode_api_key="key-xyz"),
        provider_state_dir=None,
        provider_session_id=None,
        provider_invocation_adapter=adapter,
    )

    assert len(adapter.recorded_requests) == 1
    assert adapter.recorded_requests[0].argv[0] in {"opencode", "opencode.cmd"}


def test_dispatch_raises_value_error_for_unknown_service_name(
    tmp_path: Path,
) -> None:
    adapter = InMemoryProviderInvocationAdapter()

    with pytest.raises(ValueError, match="unknown service 'bogus'"):
        dispatch_built_in_provider_session_invocation(
            service_name="bogus",
            run_kind=RunKind.FRESH,
            invocation_dir=tmp_path,
            prompt="hello",
            model="some-model",
            effort="medium",
            tool_access=ToolAccess.no_tools(),
            auth=None,
            provider_state_dir=None,
            provider_session_id=None,
            provider_invocation_adapter=adapter,
        )
