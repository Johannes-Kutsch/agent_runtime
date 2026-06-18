from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import pytest

import agent_runtime as runtime
import agent_runtime.runtime as prompt_runtime
from agent_runtime.errors import (
    AgentCredentialFailureError,
    HardAgentError,
    RuntimeConfigurationError,
    TransientAgentError,
)
from agent_runtime.provider_errors import ProviderErrorObservation
from agent_runtime.roles import InvocationRole
from agent_runtime.session import RunKind


def _stub_builtin_tmp_prompt_path(
    monkeypatch: pytest.MonkeyPatch,
    *,
    on_write: Callable[[str], None] | None = None,
    on_unlink: Callable[[], None] | None = None,
) -> None:
    prompt_path = Path("/tmp/.pycastle_prompt")
    original_write_text = Path.write_text
    original_unlink = Path.unlink

    def _fake_write_text(self: Path, data: str, *args: Any, **kwargs: Any) -> int:
        if self == prompt_path:
            if on_write is not None:
                on_write(data)
            return len(data)
        return original_write_text(self, data, *args, **kwargs)

    def _fake_unlink(self: Path, *args: Any, **kwargs: Any) -> None:
        if self == prompt_path:
            if on_unlink is not None:
                on_unlink()
            return None
        return original_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", _fake_write_text)
    monkeypatch.setattr(Path, "unlink", _fake_unlink)


def test_runtime_client_runs_claude_ephemeral_stage_through_builtin_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}

    class _ClaudeProcess:
        def __init__(self) -> None:
            self.stdout = iter(
                [
                    json.dumps(
                        {
                            "type": "assistant",
                            "message": {
                                "content": [{"type": "text", "text": "intermediate"}],
                                "usage": {
                                    "input_tokens": 5,
                                    "cache_creation_input_tokens": 0,
                                    "cache_read_input_tokens": 0,
                                },
                            },
                        }
                    )
                    + "\n",
                    json.dumps({"type": "result", "result": "final output"}) + "\n",
                ]
            )
            self.stderr = iter(())
            self.returncode = 0

        def wait(self) -> int:
            return 0

    def _fake_popen(
        command: str,
        *,
        shell: bool,
        cwd: Path,
        env: dict[str, str],
        stdout: Any,
        stderr: Any,
        text: bool,
    ) -> _ClaudeProcess:
        captured["command"] = command
        captured["shell"] = shell
        captured["cwd"] = cwd
        captured["env"] = env
        captured["stdout"] = stdout
        captured["stderr"] = stderr
        captured["text"] = text
        return _ClaudeProcess()

    monkeypatch.setattr(subprocess, "Popen", _fake_popen)

    outcome = runtime.RuntimeClient().run_ephemeral(
        prompt_runtime.EphemeralRunRequest(
            prompt="already rendered prompt",
            worktree=tmp_path,
            stage=runtime.StageSelection(
                service="claude",
                model="sonnet",
                effort="medium",
            ),
            role=InvocationRole("implementer"),
            tool_access=runtime.ToolAccess.no_tools(),
            auth=runtime.ProviderAuth(claude_code_oauth_token="oauth-token"),
        )
    )

    assert outcome == prompt_runtime.RuntimeOutcome.completed(
        output="final output",
        result=prompt_runtime.EphemeralRunResult(
            output="final output",
            selected_service="claude",
            selected_model="sonnet",
            selected_effort="medium",
            tool_access=runtime.ToolAccess.no_tools(),
            used_fallback=False,
            metadata=prompt_runtime.EphemeralResultMetadata(
                selected_service_path=("claude",),
                runtime=prompt_runtime.EphemeralRuntimeMetadata(
                    run_kind=RunKind.FRESH,
                    session_namespace="",
                ),
            ),
            usage=runtime.ProviderUsage(
                input_tokens=5,
                output_tokens=None,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
                cost_usd=None,
                duration_seconds=None,
            ),
        ),
        usage=runtime.ProviderUsage(
            input_tokens=5,
            output_tokens=None,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
            cost_usd=None,
            duration_seconds=None,
        ),
    )
    assert captured["cwd"] == tmp_path
    assert captured["env"]["CLAUDE_CODE_OAUTH_TOKEN"] == "oauth-token"
    assert "--output-format stream-json" in captured["command"]


def test_runtime_client_runs_rendered_prompt_through_claude_subprocess(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    claude_path = tmp_path / "fake-claude"
    claude_path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import json",
                "import sys",
                "prompt = sys.stdin.read()",
                'print(json.dumps({"type": "result", "result": prompt}))',
            ]
        )
        + "\n"
    )
    claude_path.chmod(0o755)
    monkeypatch.setattr(
        prompt_runtime,
        "_claude_command",
        lambda **kwargs: f"{claude_path} < {kwargs['prompt_path']}",
    )

    outcome = runtime.RuntimeClient().run_ephemeral(
        prompt_runtime.EphemeralRunRequest(
            prompt="already rendered prompt",
            worktree=tmp_path,
            stage=runtime.StageSelection(
                service="claude",
                model="sonnet",
                effort="medium",
            ),
            role=InvocationRole("implementer"),
            tool_access=runtime.ToolAccess.no_tools(),
            auth=runtime.ProviderAuth(claude_code_oauth_token="oauth-token"),
        )
    )

    assert outcome == prompt_runtime.RuntimeOutcome.completed(
        output="already rendered prompt",
        result=prompt_runtime.EphemeralRunResult(
            output="already rendered prompt",
            selected_service="claude",
            selected_model="sonnet",
            selected_effort="medium",
            tool_access=runtime.ToolAccess.no_tools(),
            used_fallback=False,
            metadata=prompt_runtime.EphemeralResultMetadata(
                selected_service_path=("claude",),
                runtime=prompt_runtime.EphemeralRuntimeMetadata(
                    run_kind=RunKind.FRESH,
                    session_namespace="",
                ),
            ),
        ),
    )


def test_runtime_client_runs_opencode_ephemeral_stage_through_builtin_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}

    class _OpenCodeProcess:
        def __init__(self) -> None:
            self.stdout = iter(
                [
                    json.dumps(
                        {
                            "type": "text",
                            "timestamp": 1,
                            "sessionID": "sess_123",
                            "part": {
                                "id": "part_1",
                                "sessionID": "sess_123",
                                "messageID": "msg_1",
                                "type": "text",
                                "text": "first assistant turn",
                                "time": {"start": 1, "end": 2},
                            },
                        }
                    )
                    + "\n",
                    json.dumps(
                        {
                            "type": "text",
                            "timestamp": 2,
                            "sessionID": "sess_123",
                            "part": {
                                "id": "part_2",
                                "sessionID": "sess_123",
                                "messageID": "msg_1",
                                "type": "text",
                                "text": "second assistant turn",
                                "time": {"start": 2, "end": 3},
                            },
                        }
                    )
                    + "\n",
                    json.dumps(
                        {
                            "type": "session.status",
                            "timestamp": 3,
                            "sessionID": "sess_123",
                            "status": {"type": "idle"},
                        }
                    )
                    + "\n",
                ]
            )
            self.stderr = iter(())
            self.returncode = 0

        def wait(self) -> int:
            return 0

    def _fake_popen(
        command: str,
        *,
        shell: bool,
        cwd: Path,
        env: dict[str, str],
        stdout: Any,
        stderr: Any,
        text: bool,
    ) -> _OpenCodeProcess:
        captured["command"] = command
        captured["shell"] = shell
        captured["cwd"] = cwd
        captured["env"] = env
        captured["stdout"] = stdout
        captured["stderr"] = stderr
        captured["text"] = text
        return _OpenCodeProcess()

    monkeypatch.setattr(subprocess, "Popen", _fake_popen)
    _stub_builtin_tmp_prompt_path(
        monkeypatch,
        on_write=lambda data: captured.__setitem__("prompt", data),
        on_unlink=lambda: captured.__setitem__("prompt_deleted", True),
    )

    outcome = runtime.RuntimeClient().run_ephemeral(
        prompt_runtime.EphemeralRunRequest(
            prompt="already rendered prompt",
            worktree=tmp_path,
            stage=runtime.StageSelection(
                service="opencode",
                model="kimi-k2.6",
                effort="medium",
            ),
            role=InvocationRole("implementer"),
            tool_access=runtime.ToolAccess.no_tools(),
            auth=runtime.ProviderAuth(opencode_api_key="go-key"),
        )
    )

    assert outcome == prompt_runtime.RuntimeOutcome.completed(
        output="first assistant turn\n\nsecond assistant turn",
        result=prompt_runtime.EphemeralRunResult(
            output="first assistant turn\n\nsecond assistant turn",
            selected_service="opencode",
            selected_model="kimi-k2.6",
            selected_effort="medium",
            tool_access=runtime.ToolAccess.no_tools(),
            used_fallback=False,
            metadata=prompt_runtime.EphemeralResultMetadata(
                selected_service_path=("opencode",),
                runtime=prompt_runtime.EphemeralRuntimeMetadata(
                    run_kind=RunKind.FRESH,
                    session_namespace="",
                ),
            ),
        ),
    )
    assert captured["cwd"] == tmp_path
    assert captured["prompt"] == "already rendered prompt"
    assert captured["prompt_deleted"] is True
    assert captured["env"]["TZ"] == "UTC"
    assert captured["env"]["OPENCODE_HOME"] == str(tmp_path)
    assert captured["env"]["OPENCODE_GO_API_KEY"] == "go-key"
    config = json.loads(captured["env"]["OPENCODE_CONFIG_CONTENT"])
    provider = config["provider"]["opencode-go"]
    assert provider["options"] == {
        "baseURL": "https://opencode.ai/zen/go/v1",
        "apiKey": "{env:OPENCODE_GO_API_KEY}",
    }
    assert "kimi-k2.6" in provider["models"]
    assert "deepseek-v4-flash" in provider["models"]
    assert captured["command"] == (
        "opencode run --format json --model opencode-go/kimi-k2.6 "
        '"$(cat /tmp/.pycastle_prompt)"'
    )


def test_runtime_client_passes_only_claude_specific_env_to_subprocess(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("SHOULD_NOT_LEAK", "host-value")
    captured: dict[str, Any] = {}

    class _ClaudeProcess:
        def __init__(self) -> None:
            self.stdout = iter(
                [json.dumps({"type": "result", "result": "final output"}) + "\n"]
            )
            self.stderr = iter(())
            self.returncode = 0

        def wait(self) -> int:
            return 0

    def _fake_popen(
        command: str,
        *,
        shell: bool,
        cwd: Path,
        env: dict[str, str],
        stdout: Any,
        stderr: Any,
        text: bool,
    ) -> _ClaudeProcess:
        captured["env"] = env
        return _ClaudeProcess()

    monkeypatch.setattr(subprocess, "Popen", _fake_popen)

    runtime.RuntimeClient().run_ephemeral(
        prompt_runtime.EphemeralRunRequest(
            prompt="already rendered prompt",
            worktree=tmp_path,
            stage=runtime.StageSelection(
                service="claude",
                model="sonnet",
                effort="medium",
            ),
            role=InvocationRole("implementer"),
            tool_access=runtime.ToolAccess.no_tools(),
            auth=runtime.ProviderAuth(claude_code_oauth_token="oauth-token"),
        )
    )

    assert captured["env"] == {"CLAUDE_CODE_OAUTH_TOKEN": "oauth-token"}


def test_runtime_client_reachable_opencode_stage_requires_api_key_without_falling_through(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    subprocess_calls = 0

    def _unexpected_popen(*args: Any, **kwargs: Any) -> Any:
        nonlocal subprocess_calls
        subprocess_calls += 1
        raise AssertionError("subprocess should not start without OpenCode auth")

    monkeypatch.setattr(subprocess, "Popen", _unexpected_popen)

    with pytest.raises(AgentCredentialFailureError) as exc_info:
        runtime.RuntimeClient().run_ephemeral(
            prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                worktree=tmp_path,
                stage=runtime.StageSelection(
                    service="missing",
                    model="ignored",
                    effort="low",
                    fallback=runtime.StageSelection(
                        service="opencode",
                        model="kimi-k2.6",
                        effort="medium",
                        fallback=runtime.StageSelection(
                            service="codex",
                            model="gpt-5.4",
                            effort="medium",
                        ),
                    ),
                ),
                role=InvocationRole("implementer"),
                tool_access=runtime.ToolAccess.no_tools(),
                auth=runtime.ProviderAuth(),
            )
        )

    assert str(exc_info.value) == "Missing OpenCode API key."
    assert exc_info.value.service_name == "opencode"
    assert exc_info.value.classification == (
        "operator_actionable_agent_credential_failure"
    )
    assert exc_info.value.observations == (
        ProviderErrorObservation(
            service_name="opencode",
            raw_provider_text="Missing OpenCode API key.",
            source_stream="pre-dispatch auth check",
            status_code=401,
        ),
    )
    assert subprocess_calls == 0


def test_runtime_client_preserves_opencode_invalid_api_key_observations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_builtin_tmp_prompt_path(monkeypatch)

    class _OpenCodeProcess:
        def __init__(self) -> None:
            self.stdout = iter(
                [
                    json.dumps(
                        {
                            "type": "error",
                            "timestamp": 1,
                            "sessionID": "sess_123",
                            "error": {
                                "name": "AuthenticationError",
                                "data": {
                                    "message": "invalid api key",
                                    "statusCode": 401,
                                    "isRetryable": False,
                                },
                            },
                        }
                    )
                    + "\n"
                ]
            )
            self.stderr = iter(())
            self.returncode = 0

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: _OpenCodeProcess())

    with pytest.raises(AgentCredentialFailureError) as exc_info:
        runtime.RuntimeClient().run_ephemeral(
            prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                worktree=tmp_path,
                stage=runtime.StageSelection(
                    service="opencode",
                    model="kimi-k2.6",
                    effort="medium",
                ),
                role=InvocationRole("implementer"),
                tool_access=runtime.ToolAccess.no_tools(),
                auth=runtime.ProviderAuth(opencode_api_key="go-key"),
            )
        )

    assert exc_info.value.service_name == "opencode"
    assert exc_info.value.status_code == 401
    assert exc_info.value.classification == (
        "operator_actionable_agent_credential_failure"
    )
    assert exc_info.value.observations == (
        ProviderErrorObservation(
            service_name="opencode",
            raw_provider_text="invalid api key",
            source_stream="json_event.error",
            status_code=401,
            error_name="AuthenticationError",
        ),
    )


@pytest.mark.parametrize(
    ("model", "effort", "expected_message"),
    [
        ("not-a-real-model", "medium", "Unsupported OpenCode model"),
        ("kimi-k2.6", "high", "Unsupported OpenCode effort"),
    ],
)
def test_runtime_client_validates_opencode_model_allowlist_and_medium_effort(
    tmp_path: Path,
    model: str,
    effort: str,
    expected_message: str,
) -> None:
    with pytest.raises(RuntimeConfigurationError, match=expected_message):
        runtime.RuntimeClient().run_ephemeral(
            prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                worktree=tmp_path,
                stage=runtime.StageSelection(
                    service="opencode",
                    model=model,
                    effort=effort,
                ),
                role=InvocationRole("implementer"),
                tool_access=runtime.ToolAccess.no_tools(),
                auth=runtime.ProviderAuth(opencode_api_key="go-key"),
            )
        )


def test_runtime_client_maps_opencode_usage_limit_after_ignoring_malformed_and_non_text_events(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_builtin_tmp_prompt_path(monkeypatch)
    monkeypatch.setattr(
        prompt_runtime._time_module,
        "now_local",
        lambda: datetime(2026, 4, 28, 20, 0, tzinfo=timezone.utc),
    )

    class _OpenCodeProcess:
        def __init__(self) -> None:
            self.stdout = iter(
                [
                    '"not a dict"\n',
                    "not json\n",
                    json.dumps(
                        {
                            "type": "text",
                            "timestamp": 1,
                            "sessionID": "sess_123",
                            "part": {
                                "type": "tool",
                                "text": "ignored",
                                "time": {"start": 1, "end": 2},
                            },
                        }
                    )
                    + "\n",
                    json.dumps(
                        {
                            "type": "error",
                            "timestamp": 2,
                            "sessionID": "sess_123",
                            "error": {
                                "name": "RateLimitError",
                                "data": {
                                    "message": (
                                        "You have reached your OpenCode Go usage limit. "
                                        "Try again at Apr 28th, 2026 9:02 PM."
                                    ),
                                    "statusCode": 429,
                                    "isRetryable": True,
                                },
                            },
                        }
                    )
                    + "\n",
                ]
            )
            self.stderr = iter(())
            self.returncode = 0

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: _OpenCodeProcess())

    outcome = runtime.RuntimeClient().run_ephemeral(
        prompt_runtime.EphemeralRunRequest(
            prompt="already rendered prompt",
            worktree=tmp_path,
            stage=runtime.StageSelection(
                service="opencode",
                model="kimi-k2.6",
                effort="medium",
            ),
            role=InvocationRole("implementer"),
            tool_access=runtime.ToolAccess.no_tools(),
            auth=runtime.ProviderAuth(opencode_api_key="go-key"),
        )
    )

    assert outcome == prompt_runtime.RuntimeOutcome.no_service_available(
        output="",
        reset_time=datetime(2026, 4, 28, 21, 4, tzinfo=timezone.utc),
        usage_limit_scope=runtime.UsageLimitScope("implementer"),
        invocation_progress=prompt_runtime.InvocationProgress.NOT_STARTED,
    )


def test_runtime_client_maps_opencode_missing_model_without_status_to_hard_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_builtin_tmp_prompt_path(monkeypatch)

    class _OpenCodeProcess:
        def __init__(self) -> None:
            self.stdout = iter(
                [
                    "not json\n",
                    json.dumps(
                        {
                            "type": "text",
                            "timestamp": 1,
                            "sessionID": "sess_123",
                            "part": {
                                "type": "image",
                                "text": "ignored",
                                "time": {"start": 1, "end": 2},
                            },
                        }
                    )
                    + "\n",
                    json.dumps(
                        {
                            "type": "error",
                            "timestamp": 2,
                            "sessionID": "sess_123",
                            "error": {
                                "name": "UnknownError",
                                "data": {
                                    "message": (
                                        "Model not found: "
                                        "opencode-go/deepseek-v4-flash. "
                                        "Did you mean: deepseek-v4-flash?"
                                    )
                                },
                            },
                        }
                    )
                    + "\n",
                ]
            )
            self.stderr = iter(())
            self.returncode = 0

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: _OpenCodeProcess())

    with pytest.raises(HardAgentError) as exc_info:
        runtime.RuntimeClient().run_ephemeral(
            prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                worktree=tmp_path,
                stage=runtime.StageSelection(
                    service="opencode",
                    model="kimi-k2.6",
                    effort="medium",
                ),
                role=InvocationRole("implementer"),
                tool_access=runtime.ToolAccess.no_tools(),
                auth=runtime.ProviderAuth(opencode_api_key="go-key"),
            )
        )

    assert str(exc_info.value) == (
        "Model not found: opencode-go/deepseek-v4-flash. "
        "Did you mean: deepseek-v4-flash?"
    )
    assert exc_info.value.service_name == "opencode"
    assert exc_info.value.status_code == 400


def test_runtime_client_maps_opencode_transient_error_stream_to_transient_exception(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_builtin_tmp_prompt_path(monkeypatch)

    class _OpenCodeProcess:
        def __init__(self) -> None:
            self.stdout = iter(
                [
                    json.dumps(
                        {
                            "type": "error",
                            "timestamp": 1,
                            "sessionID": "sess_123",
                            "error": {
                                "name": "InternalServerError",
                                "data": {
                                    "message": "temporary backend failure",
                                    "statusCode": 503,
                                    "isRetryable": True,
                                },
                            },
                        }
                    )
                    + "\n"
                ]
            )
            self.stderr = iter(())
            self.returncode = 0

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: _OpenCodeProcess())

    with pytest.raises(TransientAgentError) as exc_info:
        runtime.RuntimeClient().run_ephemeral(
            prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                worktree=tmp_path,
                stage=runtime.StageSelection(
                    service="opencode",
                    model="kimi-k2.6",
                    effort="medium",
                ),
                role=InvocationRole("implementer"),
                tool_access=runtime.ToolAccess.no_tools(),
                auth=runtime.ProviderAuth(opencode_api_key="go-key"),
            )
        )

    assert str(exc_info.value) == "temporary backend failure"
    assert exc_info.value.status_code == 503


def test_runtime_client_keeps_completed_opencode_result_after_idle_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_builtin_tmp_prompt_path(monkeypatch)

    class _OpenCodeProcess:
        def __init__(self) -> None:
            self.stdout = iter(
                [
                    json.dumps(
                        {
                            "type": "text",
                            "timestamp": 1,
                            "sessionID": "sess_123",
                            "part": {
                                "id": "part_1",
                                "sessionID": "sess_123",
                                "messageID": "msg_1",
                                "type": "text",
                                "text": "completed answer",
                                "time": {"start": 1, "end": 2},
                            },
                        }
                    )
                    + "\n",
                    json.dumps(
                        {
                            "type": "session.status",
                            "timestamp": 2,
                            "sessionID": "sess_123",
                            "status": {"type": "idle"},
                        }
                    )
                    + "\n",
                    json.dumps(
                        {
                            "type": "error",
                            "timestamp": 3,
                            "sessionID": "sess_123",
                            "error": {
                                "name": "InternalServerError",
                                "data": {
                                    "message": "should be ignored after idle result",
                                    "statusCode": 503,
                                    "isRetryable": True,
                                },
                            },
                        }
                    )
                    + "\n",
                ]
            )
            self.stderr = iter(())
            self.returncode = 0

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: _OpenCodeProcess())

    outcome = runtime.RuntimeClient().run_ephemeral(
        prompt_runtime.EphemeralRunRequest(
            prompt="already rendered prompt",
            worktree=tmp_path,
            stage=runtime.StageSelection(
                service="opencode",
                model="kimi-k2.6",
                effort="medium",
            ),
            role=InvocationRole("implementer"),
            tool_access=runtime.ToolAccess.no_tools(),
            auth=runtime.ProviderAuth(opencode_api_key="go-key"),
        )
    )

    assert outcome == prompt_runtime.RuntimeOutcome.completed(
        output="completed answer",
        result=prompt_runtime.EphemeralRunResult(
            output="completed answer",
            selected_service="opencode",
            selected_model="kimi-k2.6",
            selected_effort="medium",
            tool_access=runtime.ToolAccess.no_tools(),
            used_fallback=False,
            metadata=prompt_runtime.EphemeralResultMetadata(
                selected_service_path=("opencode",),
                runtime=prompt_runtime.EphemeralRuntimeMetadata(
                    run_kind=RunKind.FRESH,
                    session_namespace="",
                ),
            ),
        ),
    )


def test_runtime_client_maps_claude_usage_limit_stream_to_usage_limited_outcome(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _ClaudeProcess:
        def __init__(self) -> None:
            self.stdout = iter(
                [
                    json.dumps(
                        {
                            "type": "result",
                            "is_error": True,
                            "api_error_status": 429,
                            "result": "Claude usage limit reached.",
                        }
                    )
                    + "\n"
                ]
            )
            self.stderr = iter(())
            self.returncode = 0

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *args, **kwargs: _ClaudeProcess(),
    )
    monkeypatch.setattr(
        prompt_runtime._time_module,
        "now_local",
        lambda: datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
    )

    outcome = runtime.RuntimeClient().run_ephemeral(
        prompt_runtime.EphemeralRunRequest(
            prompt="already rendered prompt",
            worktree=tmp_path,
            stage=runtime.StageSelection(
                service="claude",
                model="sonnet",
                effort="medium",
            ),
            role=InvocationRole("implementer"),
            tool_access=runtime.ToolAccess.no_tools(),
            auth=runtime.ProviderAuth(claude_code_oauth_token="oauth-token"),
        )
    )

    assert outcome == prompt_runtime.RuntimeOutcome.no_service_available(
        output="",
        reset_time=datetime(2026, 1, 1, 13, 2, tzinfo=timezone.utc),
        usage_limit_scope=runtime.UsageLimitScope("implementer"),
        invocation_progress=prompt_runtime.InvocationProgress.NOT_STARTED,
    )


def test_runtime_client_reachable_claude_stage_requires_token_without_falling_through(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    subprocess_calls = 0

    def _unexpected_popen(*args: Any, **kwargs: Any) -> Any:
        nonlocal subprocess_calls
        subprocess_calls += 1
        raise AssertionError("subprocess should not start without Claude auth")

    monkeypatch.setattr(subprocess, "Popen", _unexpected_popen)

    with pytest.raises(AgentCredentialFailureError) as exc_info:
        runtime.RuntimeClient().run_ephemeral(
            prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                worktree=tmp_path,
                stage=runtime.StageSelection(
                    service="missing",
                    model="ignored",
                    effort="low",
                    fallback=runtime.StageSelection(
                        service="claude",
                        model="sonnet",
                        effort="medium",
                        fallback=runtime.StageSelection(
                            service="codex",
                            model="gpt-5",
                            effort="medium",
                        ),
                    ),
                ),
                role=InvocationRole("implementer"),
                tool_access=runtime.ToolAccess.no_tools(),
                auth=runtime.ProviderAuth(),
            )
        )

    assert exc_info.value.service_name == "claude"
    assert subprocess_calls == 0


def test_runtime_client_maps_claude_transient_error_stream_to_transient_exception(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _ClaudeProcess:
        def __init__(self) -> None:
            self.stdout = iter(
                [
                    json.dumps(
                        {
                            "type": "result",
                            "is_error": True,
                            "api_error_status": 500,
                            "result": "temporary Claude failure",
                        }
                    )
                    + "\n"
                ]
            )
            self.stderr = iter(())
            self.returncode = 0

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *args, **kwargs: _ClaudeProcess(),
    )

    with pytest.raises(TransientAgentError) as exc_info:
        runtime.RuntimeClient().run_ephemeral(
            prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                worktree=tmp_path,
                stage=runtime.StageSelection(
                    service="claude",
                    model="sonnet",
                    effort="medium",
                ),
                role=InvocationRole("implementer"),
                tool_access=runtime.ToolAccess.no_tools(),
                auth=runtime.ProviderAuth(claude_code_oauth_token="oauth-token"),
            )
        )

    assert exc_info.value.status_code == 500


def test_runtime_client_parses_claude_usage_limit_reset_time(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _ClaudeProcess:
        def __init__(self) -> None:
            self.stdout = iter(
                [
                    json.dumps(
                        {
                            "type": "result",
                            "is_error": True,
                            "api_error_status": 429,
                            "result": "Claude usage limit reached, resets Jan 2, 4pm (UTC).",
                        }
                    )
                    + "\n"
                ]
            )
            self.stderr = iter(())
            self.returncode = 0

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *args, **kwargs: _ClaudeProcess(),
    )
    monkeypatch.setattr(
        prompt_runtime._time_module,
        "now_local",
        lambda: datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
    )

    outcome = runtime.RuntimeClient().run_ephemeral(
        prompt_runtime.EphemeralRunRequest(
            prompt="already rendered prompt",
            worktree=tmp_path,
            stage=runtime.StageSelection(
                service="claude",
                model="sonnet",
                effort="medium",
            ),
            role=InvocationRole("implementer"),
            tool_access=runtime.ToolAccess.no_tools(),
            auth=runtime.ProviderAuth(claude_code_oauth_token="oauth-token"),
        )
    )

    assert outcome == prompt_runtime.RuntimeOutcome.no_service_available(
        output="",
        reset_time=datetime(2026, 1, 2, 16, 2, tzinfo=timezone.utc),
        usage_limit_scope=runtime.UsageLimitScope("implementer"),
        invocation_progress=prompt_runtime.InvocationProgress.NOT_STARTED,
    )


def test_runtime_client_keeps_runtime_reset_time_override_in_usage_limited_outcome(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    reset_time = datetime(2026, 1, 2, 16, 0, tzinfo=timezone.utc)

    class _ClaudeProcess:
        def __init__(self) -> None:
            self.stdout = iter(
                [
                    json.dumps(
                        {
                            "type": "result",
                            "is_error": True,
                            "api_error_status": 429,
                            "result": "Claude usage limit reached.",
                        }
                    )
                    + "\n"
                ]
            )
            self.stderr = iter(())
            self.returncode = 0

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *args, **kwargs: _ClaudeProcess(),
    )
    monkeypatch.setattr(
        prompt_runtime._time_module,
        "now_local",
        lambda: datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
    )
    monkeypatch.setattr(
        prompt_runtime,
        "_parse_claude_reset_time",
        lambda _text: reset_time,
    )

    outcome = runtime.RuntimeClient().run_ephemeral(
        prompt_runtime.EphemeralRunRequest(
            prompt="already rendered prompt",
            worktree=tmp_path,
            stage=runtime.StageSelection(
                service="claude",
                model="sonnet",
                effort="medium",
            ),
            role=InvocationRole("implementer"),
            tool_access=runtime.ToolAccess.no_tools(),
            auth=runtime.ProviderAuth(claude_code_oauth_token="oauth-token"),
        )
    )

    assert outcome == prompt_runtime.RuntimeOutcome.no_service_available(
        output="",
        reset_time=reset_time + timedelta(minutes=2),
        usage_limit_scope=runtime.UsageLimitScope("implementer"),
        invocation_progress=prompt_runtime.InvocationProgress.NOT_STARTED,
    )


def test_runtime_client_cleans_up_builtin_prompt_file_when_claude_subprocess_fails_to_start(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prompt_path = tmp_path / ".pycastle_prompt"

    def _raise_subprocess_start_failure(*args: Any, **kwargs: Any) -> Any:
        raise OSError("failed to start Claude subprocess")

    monkeypatch.setattr(subprocess, "Popen", _raise_subprocess_start_failure)

    with pytest.raises(OSError, match="failed to start Claude subprocess"):
        runtime.RuntimeClient().run_ephemeral(
            prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                worktree=tmp_path,
                stage=runtime.StageSelection(
                    service="claude",
                    model="sonnet",
                    effort="medium",
                ),
                role=InvocationRole("implementer"),
                tool_access=runtime.ToolAccess.no_tools(),
                auth=runtime.ProviderAuth(claude_code_oauth_token="oauth-token"),
            )
        )

    assert not prompt_path.exists()


def test_runtime_client_keeps_runtime_selected_service_path_override_in_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _ClaudeProcess:
        def __init__(self) -> None:
            self.stdout = iter(
                [json.dumps({"type": "result", "result": "final output"}) + "\n"]
            )
            self.stderr = iter(())
            self.returncode = 0

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *args, **kwargs: _ClaudeProcess(),
    )
    monkeypatch.setattr(
        prompt_runtime,
        "_selected_service_path",
        lambda *_args, **_kwargs: ("patched", "claude"),
    )

    outcome = runtime.RuntimeClient().run_ephemeral(
        prompt_runtime.EphemeralRunRequest(
            prompt="already rendered prompt",
            worktree=tmp_path,
            stage=runtime.StageSelection(
                service="claude",
                model="sonnet",
                effort="medium",
            ),
            role=InvocationRole("implementer"),
            tool_access=runtime.ToolAccess.no_tools(),
            auth=runtime.ProviderAuth(claude_code_oauth_token="oauth-token"),
        )
    )

    assert outcome == prompt_runtime.RuntimeOutcome.completed(
        output="final output",
        result=prompt_runtime.EphemeralRunResult(
            output="final output",
            selected_service="claude",
            selected_model="sonnet",
            selected_effort="medium",
            tool_access=runtime.ToolAccess.no_tools(),
            used_fallback=True,
            metadata=prompt_runtime.EphemeralResultMetadata(
                selected_service_path=("patched", "claude"),
                runtime=prompt_runtime.EphemeralRuntimeMetadata(
                    run_kind=RunKind.FRESH,
                    session_namespace="",
                ),
            ),
        ),
    )


def test_runtime_client_preserves_claude_credential_failure_observations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    denial_message = "Disabled Claude subscription access for Claude Code."

    class _ClaudeProcess:
        def __init__(self) -> None:
            self.stdout = iter(
                [
                    json.dumps(
                        {
                            "type": "result",
                            "is_error": True,
                            "api_error_status": 403,
                            "result": denial_message,
                        }
                    )
                    + "\n"
                ]
            )
            self.stderr = iter(())
            self.returncode = 0

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *args, **kwargs: _ClaudeProcess(),
    )

    with pytest.raises(AgentCredentialFailureError) as exc_info:
        runtime.RuntimeClient().run_ephemeral(
            prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                worktree=tmp_path,
                stage=runtime.StageSelection(
                    service="claude",
                    model="sonnet",
                    effort="medium",
                ),
                role=InvocationRole("implementer"),
                tool_access=runtime.ToolAccess.no_tools(),
                auth=runtime.ProviderAuth(claude_code_oauth_token="oauth-token"),
            )
        )

    assert exc_info.value.observations == (
        ProviderErrorObservation(
            service_name="claude",
            raw_provider_text=denial_message,
            source_stream="json_event.result",
            status_code=403,
        ),
    )
