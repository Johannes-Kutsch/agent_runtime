from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import pytest

import agent_runtime as runtime
import agent_runtime.runtime as prompt_runtime
from agent_runtime.contracts import UsageLimit
from agent_runtime.errors import AgentCredentialFailureError, TransientAgentError
from agent_runtime.provider_errors import ProviderErrorObservation
from agent_runtime.roles import InvocationRole
from agent_runtime.session import RunKind


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

    assert outcome == prompt_runtime.RuntimeOutcome.usage_limited(
        output="",
        service_name="claude",
        reset_time=None,
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

    assert outcome == prompt_runtime.RuntimeOutcome.usage_limited(
        output="",
        service_name="claude",
        reset_time=datetime(2026, 1, 2, 16, 0, tzinfo=timezone.utc),
        usage_limit_scope=runtime.UsageLimitScope("implementer"),
        invocation_progress=prompt_runtime.InvocationProgress.NOT_STARTED,
    )


def test_runtime_module_claude_event_parser_keeps_runtime_reset_time_override() -> None:
    reset_time = datetime(2026, 1, 2, 16, 0, tzinfo=timezone.utc)

    def _fake_parse_claude_reset_time(_text: object) -> datetime | None:
        return reset_time

    original_parse_reset_time = prompt_runtime._parse_claude_reset_time
    prompt_runtime._parse_claude_reset_time = cast(Any, _fake_parse_claude_reset_time)
    try:
        events = prompt_runtime._parse_claude_event(
            json.dumps(
                {
                    "type": "result",
                    "is_error": True,
                    "api_error_status": 429,
                    "result": "Claude usage limit reached.",
                }
            )
        )
    finally:
        prompt_runtime._parse_claude_reset_time = original_parse_reset_time

    assert events == [UsageLimit(reset_time=reset_time, raw_message=None)]


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
