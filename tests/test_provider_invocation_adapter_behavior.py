from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import agent_runtime._builtin_provider_parsed_output as builtin_provider_parsed_output
import agent_runtime._provider_invocation as provider_invocation_runtime
from agent_runtime._builtin_provider_stream_interpretation import reduce_codex_stream
from agent_runtime._runtime_lifecycle import CancellationToken
from agent_runtime.errors import (
    AgentCancelledError,
    HardAgentError,
    ProviderUnavailableError,
    ProviderUnavailableReason,
    UsageLimitError,
)
from agent_runtime.provider_usage import ProviderUsage
from agent_runtime.session import RunKind


def _assert_subprocess_is_dead(pid: int) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import ctypes, os, sys; "
                "pid = int(sys.argv[1]); "
                "alive = False; "
                "if os.name == 'nt': "
                " kernel32 = ctypes.windll.kernel32; "
                " handle = kernel32.OpenProcess(0x1000, False, pid); "
                " if handle: "
                "  exit_code = ctypes.c_ulong(); "
                "  kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)); "
                "  alive = exit_code.value == 259; "
                "  kernel32.CloseHandle(handle); "
                "else: "
                " try: os.kill(pid, 0); alive = True; "
                " except ProcessLookupError: alive = False; "
                " except PermissionError: alive = True; "
                "sys.exit(0 if alive else 1)"
            ),
            str(pid),
        ],
        check=False,
    )
    assert result.returncode == 1


def _assert_expected_process_env(
    actual_env: dict[str, str],
    provider_env: dict[str, str],
) -> None:
    if os.name != "nt":
        assert actual_env == provider_env
        return

    expected_env = {
        key: os.environ[key]
        for key in ("PATH", "PATHEXT", "SystemRoot", "ComSpec", "WINDIR")
    }
    expected_env.update(provider_env)
    assert actual_env == expected_env


def _execute_provider_invocation_at_adapter_seam(
    *,
    monkeypatch: pytest.MonkeyPatch | None,
    tmp_path: Path,
    adapter_kind: str,
    request: provider_invocation_runtime.ProviderInvocationRequest,
    stdout_lines: tuple[str, ...],
    stderr_lines: tuple[str, ...] = (),
    returncode: int = 0,
    prepared_provider_session_id: str | None = None,
) -> (
    provider_invocation_runtime.ProviderInvocationResult
    | provider_invocation_runtime.ProviderInvocationFailure
):
    if adapter_kind == "production":

        class _Process:
            def __init__(self) -> None:
                self.stdout = iter(stdout_lines)
                self.stderr = iter(stderr_lines)
                self.returncode = returncode

            def wait(self) -> int:
                return returncode

        assert monkeypatch is not None
        monkeypatch.setattr(
            provider_invocation_runtime.subprocess,
            "Popen",
            lambda *args, **kwargs: _Process(),
        )
        return (
            provider_invocation_runtime.ProductionProviderInvocationAdapter().execute(
                request
            )
        )

    if adapter_kind == "in_memory":
        return provider_invocation_runtime.InMemoryProviderInvocationAdapter(
            prepared_invocations=[
                provider_invocation_runtime.ProviderInvocationPreparedStream(
                    stdout_lines=stdout_lines,
                    provider_session_id=prepared_provider_session_id,
                )
            ]
        ).execute(request)

    raise AssertionError(f"Unsupported adapter kind: {adapter_kind}")


def test_production_adapter_executes_prepared_invocation_and_returns_reduced_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prompt_path = tmp_path / ".provider_prompt"
    captured: dict[str, Any] = {}

    class _Process:
        def __init__(self) -> None:
            self.stdout = iter(["line 1\n", "line 2\n"])
            self.stderr = iter(())
            self.returncode = 0
            self.wait_called = False

        def wait(self) -> int:
            self.wait_called = True
            return 0

    process = _Process()

    def _fake_popen(
        command: str,
        *,
        shell: bool,
        cwd: Path,
        env: dict[str, str],
        stdout: Any,
        stderr: Any,
        text: bool,
    ) -> _Process:
        captured["command"] = command
        captured["shell"] = shell
        captured["cwd"] = cwd
        captured["env"] = env
        captured["stdout"] = stdout
        captured["stderr"] = stderr
        captured["text"] = text
        return process

    monkeypatch.setattr(provider_invocation_runtime.subprocess, "Popen", _fake_popen)

    def _reduce_output(lines: list[str]) -> tuple[str, ProviderUsage | None]:
        captured["reduced_lines"] = lines
        return (
            "normalized output",
            ProviderUsage(input_tokens=3, output_tokens=5),
        )

    request = provider_invocation_runtime.ProviderInvocationRequest(
        command="provider --run",
        worktree=tmp_path,
        environment={"PROVIDER_TOKEN": "secret"},
        prompt=provider_invocation_runtime.ProviderInvocationPrompt(
            content="rendered prompt",
            path=prompt_path,
            cleanup_path=True,
        ),
        run_kind=RunKind.FRESH,
        provider_session_id=None,
        output_hooks=provider_invocation_runtime.ProviderOutputReductionHooks(
            reduce_output=_reduce_output,
            extract_provider_session_id=lambda _lines: None,
        ),
    )

    result = provider_invocation_runtime.ProductionProviderInvocationAdapter().execute(
        request
    )

    assert result == provider_invocation_runtime.ProviderInvocationResult(
        output="normalized output",
        usage=ProviderUsage(input_tokens=3, output_tokens=5),
        stdout_lines=("line 1\n", "line 2\n"),
        provider_session_id=None,
    )
    assert not prompt_path.exists()
    assert process.wait_called is True
    assert captured == {
        "command": "provider --run",
        "shell": True,
        "cwd": tmp_path,
        "env": captured["env"],
        "stdout": provider_invocation_runtime.subprocess.PIPE,
        "stderr": provider_invocation_runtime.subprocess.PIPE,
        "text": True,
        "reduced_lines": ["line 1\n", "line 2\n"],
    }
    _assert_expected_process_env(captured["env"], {"PROVIDER_TOKEN": "secret"})


def test_production_adapter_executes_argv_invocation_with_prompt_on_stdin(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prompt_path = tmp_path / ".provider_prompt"
    captured: dict[str, Any] = {}

    class _Stdin:
        def __init__(self) -> None:
            self.writes: list[str] = []
            self.closed = False

        def write(self, content: str) -> None:
            self.writes.append(content)

        def close(self) -> None:
            self.closed = True

    class _Process:
        def __init__(self) -> None:
            self.stdin = _Stdin()
            self.stdout = iter(["line 1\n"])
            self.stderr = iter(())
            self.returncode = 0
            self.wait_called = False

        def wait(self) -> int:
            self.wait_called = True
            return 0

    process = _Process()

    def _fake_popen(
        command: tuple[str, ...],
        *,
        shell: bool,
        cwd: Path,
        env: dict[str, str],
        stdout: Any,
        stderr: Any,
        text: bool,
        stdin: Any,
    ) -> _Process:
        captured["command"] = command
        captured["shell"] = shell
        captured["cwd"] = cwd
        captured["env"] = env
        captured["stdout"] = stdout
        captured["stderr"] = stderr
        captured["text"] = text
        captured["stdin"] = stdin
        return process

    monkeypatch.setattr(provider_invocation_runtime.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(provider_invocation_runtime.shutil, "which", lambda _name: None)

    request = provider_invocation_runtime.ProviderInvocationRequest(
        worktree=tmp_path,
        environment={"PROVIDER_TOKEN": "secret"},
        prompt=provider_invocation_runtime.ProviderInvocationPrompt(
            content="rendered prompt",
            path=prompt_path,
            cleanup_path=True,
        ),
        run_kind=RunKind.FRESH,
        provider_session_id=None,
        output_hooks=provider_invocation_runtime.ProviderOutputReductionHooks(
            reduce_output=lambda lines: ("".join(lines), None),
        ),
        argv=("provider", "--run"),
    )

    result = provider_invocation_runtime.ProductionProviderInvocationAdapter().execute(
        request
    )

    assert result == provider_invocation_runtime.ProviderInvocationResult(
        output="line 1\n",
        usage=None,
        stdout_lines=("line 1\n",),
        provider_session_id=None,
    )
    assert request.command == "provider --run"
    assert process.stdin.writes == ["rendered prompt"]
    assert process.stdin.closed is True
    assert process.wait_called is True
    assert not prompt_path.exists()
    assert captured == {
        "command": ["provider", "--run"],
        "shell": False,
        "cwd": tmp_path,
        "env": captured["env"],
        "stdout": provider_invocation_runtime.subprocess.PIPE,
        "stderr": provider_invocation_runtime.subprocess.PIPE,
        "text": True,
        "stdin": provider_invocation_runtime.subprocess.PIPE,
    }
    _assert_expected_process_env(captured["env"], {"PROVIDER_TOKEN": "secret"})


def test_production_adapter_applies_argv_transform_before_execution_and_forces_stdin_prompt_input(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prompt_path = tmp_path / ".provider_prompt"
    captured: dict[str, Any] = {}
    captured_env: dict[str, Any] = {}

    class _Stdin:
        def __init__(self) -> None:
            self.writes: list[str] = []
            self.closed = False

        def write(self, content: str) -> None:
            self.writes.append(content)

        def close(self) -> None:
            self.closed = True

    class _Process:
        def __init__(self) -> None:
            self.stdin = _Stdin()
            self.stdout = iter(["line 1\n"])
            self.stderr = iter(())
            self.returncode = 0
            self.wait_called = False

        def wait(self) -> int:
            self.wait_called = True
            return 0

    process = _Process()

    def _fake_popen(
        command: list[str] | tuple[str, ...],
        *,
        shell: bool,
        cwd: Path,
        env: dict[str, str],
        stdout: Any,
        stderr: Any,
        text: bool,
        stdin: Any,
    ) -> _Process:
        captured["command"] = command
        captured["shell"] = shell
        captured["cwd"] = cwd
        captured_env.update(env)
        captured["stdout"] = stdout
        captured["stderr"] = stderr
        captured["text"] = text
        captured["stdin"] = stdin
        return process

    monkeypatch.setattr(provider_invocation_runtime.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(provider_invocation_runtime.shutil, "which", lambda _name: None)

    captured_cwd: Path | None = None

    def _argv_transform(
        argv: tuple[str, ...],
        cwd: Path,
        _env: dict[str, str],
    ) -> tuple[str, ...]:
        nonlocal captured_cwd
        captured_cwd = cwd
        return ("transformed", "provider")

    request = provider_invocation_runtime.ProviderInvocationRequest(
        command="provider < /tmp/provider_prompt",
        argv=(),
        worktree=tmp_path,
        environment={"PROVIDER_TOKEN": "secret"},
        prompt=provider_invocation_runtime.ProviderInvocationPrompt(
            content="rendered prompt",
            path=prompt_path,
            cleanup_path=False,
        ),
        run_kind=RunKind.FRESH,
        provider_session_id=None,
        output_hooks=provider_invocation_runtime.ProviderOutputReductionHooks(
            reduce_output=lambda lines: ("".join(lines), None),
        ),
    )

    result = provider_invocation_runtime.ProductionProviderInvocationAdapter().execute(
        request,
        argv_transform=_argv_transform,
    )

    assert result == provider_invocation_runtime.ProviderInvocationResult(
        output="line 1\n",
        usage=None,
        stdout_lines=("line 1\n",),
        provider_session_id=None,
    )
    assert tuple(captured["command"]) == ("transformed", "provider")
    assert captured["shell"] is False
    assert captured["stdin"] is provider_invocation_runtime.subprocess.PIPE
    assert captured_cwd == tmp_path
    assert process.stdin.writes == ["rendered prompt"]
    assert process.stdin.closed is True
    assert not prompt_path.exists()


def test_production_adapter_prefers_argv_over_legacy_command_for_claude_prompt_input(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prompt_dir = tmp_path / "Users" / "Test User" / "Prompt Dir"
    prompt_dir.mkdir(parents=True)
    prompt_path = prompt_dir / ".provider_prompt"
    legacy_command = (
        "claude --verbose --dangerously-skip-permissions --output-format "
        "stream-json -p - --disable-slash-commands "
        "--exclude-dynamic-system-prompt-sections --strict-mcp-config "
        f'--mcp-config {{"mcpServers":{{}}}} --model sonnet --effort medium < {prompt_path}'
    )
    captured: dict[str, Any] = {}

    class _Stdin:
        def __init__(self) -> None:
            self.writes: list[str] = []
            self.closed = False

        def write(self, content: str) -> None:
            self.writes.append(content)

        def close(self) -> None:
            self.closed = True

    class _Process:
        def __init__(self) -> None:
            self.stdin = _Stdin()
            self.stdout = iter(['{"type":"result","result":"hello from claude"}\n'])
            self.stderr = iter(())
            self.returncode = 0

        def wait(self) -> int:
            return 0

    process = _Process()

    def _fake_popen(
        command: tuple[str, ...],
        *,
        shell: bool,
        cwd: Path,
        env: dict[str, str],
        stdout: Any,
        stderr: Any,
        text: bool,
        stdin: Any,
    ) -> _Process:
        captured["command"] = command
        captured["shell"] = shell
        captured["cwd"] = cwd
        captured["env"] = env
        captured["stdout"] = stdout
        captured["stderr"] = stderr
        captured["text"] = text
        captured["stdin"] = stdin
        return process

    monkeypatch.setattr(provider_invocation_runtime.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(provider_invocation_runtime.shutil, "which", lambda _name: None)

    request = provider_invocation_runtime.ProviderInvocationRequest(
        command=legacy_command,
        argv=(
            "claude",
            "--verbose",
            "--dangerously-skip-permissions",
            "--output-format",
            "stream-json",
            "-p",
            "-",
            "--disable-slash-commands",
            "--exclude-dynamic-system-prompt-sections",
            "--strict-mcp-config",
            "--mcp-config",
            '{"mcpServers":{}}',
            "--model",
            "sonnet",
            "--effort",
            "medium",
        ),
        prefer_argv=True,
        worktree=tmp_path,
        environment={"CLAUDE_CODE_OAUTH_TOKEN": "secret"},
        prompt=provider_invocation_runtime.ProviderInvocationPrompt(
            content="rendered prompt",
            path=prompt_path,
            cleanup_path=True,
        ),
        run_kind=RunKind.FRESH,
        provider_session_id=None,
        output_hooks=provider_invocation_runtime.ProviderOutputReductionHooks(
            reduce_output=lambda lines: ("".join(lines), None),
        ),
    )

    result = provider_invocation_runtime.ProductionProviderInvocationAdapter().execute(
        request
    )

    assert result == provider_invocation_runtime.ProviderInvocationResult(
        output='{"type":"result","result":"hello from claude"}\n',
        usage=None,
        stdout_lines=('{"type":"result","result":"hello from claude"}\n',),
        provider_session_id=None,
    )
    assert request.command == legacy_command
    assert process.stdin.writes == ["rendered prompt"]
    assert process.stdin.closed is True
    assert not prompt_path.exists()
    assert captured == {
        "command": list(request.argv),
        "shell": False,
        "cwd": tmp_path,
        "env": captured["env"],
        "stdout": provider_invocation_runtime.subprocess.PIPE,
        "stderr": provider_invocation_runtime.subprocess.PIPE,
        "text": True,
        "stdin": provider_invocation_runtime.subprocess.PIPE,
    }
    _assert_expected_process_env(
        captured["env"],
        {"CLAUDE_CODE_OAUTH_TOKEN": "secret"},
    )


def test_production_adapter_resolves_argv_executable_against_path_before_spawning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Regression: an argv invocation spawned with shell=False must resolve its
    # executable against PATH/PATHEXT first. On Windows, CreateProcess only
    # appends .exe and ignores PATHEXT, so a bare "claude" never finds the
    # npm-installed claude.cmd shim and raises FileNotFoundError. Resolving via
    # shutil.which yields the runnable shim path while preserving the rest of
    # the argv unchanged.
    resolved_path = r"C:\Users\agent\AppData\Roaming\npm\claude.CMD"
    captured: dict[str, Any] = {}
    which_calls: list[str] = []

    class _Stdin:
        def __init__(self) -> None:
            self.writes: list[str] = []
            self.closed = False

        def write(self, content: str) -> None:
            self.writes.append(content)

        def close(self) -> None:
            self.closed = True

    class _Process:
        def __init__(self) -> None:
            self.stdin = _Stdin()
            self.stdout = iter(['{"type":"result","result":"hi"}\n'])
            self.stderr = iter(())
            self.returncode = 0

        def wait(self) -> int:
            return 0

    process = _Process()

    def _fake_popen(
        command: tuple[str, ...],
        *,
        shell: bool,
        cwd: Path,
        env: dict[str, str],
        stdout: Any,
        stderr: Any,
        text: bool,
        stdin: Any,
    ) -> _Process:
        captured["command"] = command
        captured["shell"] = shell
        return process

    def _fake_which(name: str) -> str | None:
        which_calls.append(name)
        return resolved_path if name == "claude" else None

    monkeypatch.setattr(provider_invocation_runtime.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(provider_invocation_runtime.shutil, "which", _fake_which)

    request = provider_invocation_runtime.ProviderInvocationRequest(
        worktree=tmp_path,
        environment={"CLAUDE_CODE_OAUTH_TOKEN": "secret"},
        prompt=provider_invocation_runtime.ProviderInvocationPrompt(
            content="rendered prompt",
        ),
        run_kind=RunKind.FRESH,
        provider_session_id=None,
        output_hooks=provider_invocation_runtime.ProviderOutputReductionHooks(
            reduce_output=lambda lines: ("".join(lines), None),
        ),
        argv=("claude", "--model", "sonnet"),
        prefer_argv=True,
    )

    provider_invocation_runtime.ProductionProviderInvocationAdapter().execute(request)

    assert which_calls == ["claude"]
    assert captured["shell"] is False
    # argv[0] rewritten to the resolved shim; remaining args untouched.
    assert captured["command"] == [resolved_path, "--model", "sonnet"]
    # The unresolved request argv is left intact.
    assert request.argv == ("claude", "--model", "sonnet")


def test_production_adapter_falls_back_to_bare_argv_when_executable_unresolved(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # When shutil.which cannot resolve the executable (e.g. POSIX hosts where
    # the bare name already runs, or the tool is genuinely absent), the bare
    # argv must be passed through unchanged so existing behavior is preserved.
    captured: dict[str, Any] = {}

    class _Stdin:
        def write(self, content: str) -> None: ...

        def close(self) -> None: ...

    class _Process:
        def __init__(self) -> None:
            self.stdin = _Stdin()
            self.stdout = iter(['{"type":"result","result":"hi"}\n'])
            self.stderr = iter(())
            self.returncode = 0

        def wait(self) -> int:
            return 0

    def _fake_popen(command: tuple[str, ...], **_kwargs: Any) -> _Process:
        captured["command"] = command
        return _Process()

    monkeypatch.setattr(provider_invocation_runtime.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(provider_invocation_runtime.shutil, "which", lambda _name: None)

    request = provider_invocation_runtime.ProviderInvocationRequest(
        worktree=tmp_path,
        environment={},
        prompt=provider_invocation_runtime.ProviderInvocationPrompt(
            content="rendered prompt",
        ),
        run_kind=RunKind.FRESH,
        provider_session_id=None,
        output_hooks=provider_invocation_runtime.ProviderOutputReductionHooks(
            reduce_output=lambda lines: ("".join(lines), None),
        ),
        argv=("claude", "--model", "sonnet"),
        prefer_argv=True,
    )

    provider_invocation_runtime.ProductionProviderInvocationAdapter().execute(request)

    assert captured["command"] == ["claude", "--model", "sonnet"]


def test_production_adapter_uses_live_output_reducer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prompt_path = tmp_path / ".provider_prompt"
    process_lines = ['{"session":"provider-session-123"}\n', "final line\n"]

    class _Process:
        def __init__(self) -> None:
            self.stdout = iter(process_lines)
            self.stderr = iter(())
            self.returncode = 0

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(
        provider_invocation_runtime.subprocess,
        "Popen",
        lambda *args, **kwargs: _Process(),
    )

    observed = {"live_reducer_calls": 0}

    def _reduce_output(lines: list[str]) -> tuple[str, ProviderUsage | None]:
        observed["live_reducer_calls"] += 1
        return ("normalized output", ProviderUsage(output_tokens=7))

    request = provider_invocation_runtime.ProviderInvocationRequest(
        command="provider --run",
        worktree=tmp_path,
        environment={"PROVIDER_TOKEN": "secret"},
        prompt=provider_invocation_runtime.ProviderInvocationPrompt(
            content="rendered prompt",
            path=prompt_path,
            cleanup_path=True,
        ),
        run_kind=RunKind.RESUME,
        provider_session_id="existing-session",
        output_hooks=provider_invocation_runtime.ProviderOutputReductionHooks(
            reduce_output=_reduce_output,
            extract_provider_session_id=lambda _lines: "provider-session-123",
        ),
    )

    result = provider_invocation_runtime.ProductionProviderInvocationAdapter().execute(
        request
    )

    assert result == provider_invocation_runtime.ProviderInvocationResult(
        output="normalized output",
        usage=ProviderUsage(output_tokens=7),
        stdout_lines=tuple(process_lines),
        provider_session_id="provider-session-123",
    )
    assert observed["live_reducer_calls"] == 1


@pytest.mark.parametrize(
    ("adapter_kind", "prepared_provider_session_id"),
    [
        ("production", None),
        ("in_memory", "prepared-session"),
    ],
)
def test_provider_invocation_adapter_finalizes_completed_output_at_adapter_seam(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    adapter_kind: str,
    prepared_provider_session_id: str | None,
) -> None:
    process_lines = ['{"session":"provider-session-123"}\n', "final line\n"]
    observed = {"live_reducer_calls": 0}

    def _reduce_output(lines: list[str]) -> tuple[str, ProviderUsage | None]:
        observed["live_reducer_calls"] += 1
        return ("normalized output", ProviderUsage(output_tokens=7))

    request = provider_invocation_runtime.ProviderInvocationRequest(
        command="provider --run",
        worktree=tmp_path,
        environment={"PROVIDER_TOKEN": "secret"},
        prompt=provider_invocation_runtime.ProviderInvocationPrompt(
            content="rendered prompt",
        ),
        run_kind=RunKind.RESUME,
        provider_session_id="existing-session",
        output_hooks=provider_invocation_runtime.ProviderOutputReductionHooks(
            reduce_output=_reduce_output,
            extract_provider_session_id=lambda _lines: "provider-session-123",
        ),
    )

    result = _execute_provider_invocation_at_adapter_seam(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        adapter_kind=adapter_kind,
        request=request,
        stdout_lines=tuple(process_lines),
        prepared_provider_session_id=prepared_provider_session_id,
    )

    assert result == provider_invocation_runtime.ProviderInvocationResult(
        output="normalized output",
        usage=ProviderUsage(output_tokens=7),
        stdout_lines=tuple(process_lines),
        provider_session_id="provider-session-123",
    )
    assert observed["live_reducer_calls"] == 1


@pytest.mark.parametrize(
    ("adapter_kind", "prepared_provider_session_id"),
    [
        ("production", None),
        ("in_memory", "prepared-session"),
    ],
)
def test_provider_invocation_adapter_preserves_reducer_failure_with_extracted_provider_session_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    adapter_kind: str,
    prepared_provider_session_id: str | None,
) -> None:
    output_line = '{"session":"provider-session-123"}\n'
    request = provider_invocation_runtime.ProviderInvocationRequest(
        command="provider --run",
        worktree=tmp_path,
        environment={},
        prompt=provider_invocation_runtime.ProviderInvocationPrompt(
            content="rendered prompt",
        ),
        run_kind=RunKind.RESUME,
        provider_session_id="existing-session",
        output_hooks=provider_invocation_runtime.ProviderOutputReductionHooks(
            reduce_output=(
                lambda _lines: (_ for _ in ()).throw(
                    UsageLimitError(raw_message="usage limited")
                )
            ),
            extract_provider_session_id=lambda _lines: "provider-session-123",
        ),
    )

    result = _execute_provider_invocation_at_adapter_seam(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        adapter_kind=adapter_kind,
        request=request,
        stdout_lines=(output_line,),
        returncode=19,
        prepared_provider_session_id=prepared_provider_session_id,
    )

    assert result == provider_invocation_runtime.ProviderInvocationFailure(
        kind=provider_invocation_runtime.InvocationFailureKind.USAGE_LIMITED,
        detail="usage limited",
        stdout_lines=(output_line,),
        provider_session_id="provider-session-123",
        usage=None,
        reset_time=None,
    )


@pytest.mark.parametrize(
    ("adapter_kind", "prepared_provider_session_id", "expected_provider_session_id"),
    [
        ("production", None, None),
        ("in_memory", "prepared-session", "prepared-session"),
    ],
)
def test_provider_invocation_adapter_uses_adapter_session_fallback_on_reducer_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    adapter_kind: str,
    prepared_provider_session_id: str | None,
    expected_provider_session_id: str | None,
) -> None:
    output_line = "usage limit line\n"
    request = provider_invocation_runtime.ProviderInvocationRequest(
        command="provider --run",
        worktree=tmp_path,
        environment={},
        prompt=provider_invocation_runtime.ProviderInvocationPrompt(
            content="rendered prompt",
        ),
        run_kind=RunKind.RESUME,
        provider_session_id="existing-session",
        output_hooks=provider_invocation_runtime.ProviderOutputReductionHooks(
            reduce_output=(
                lambda _lines: (_ for _ in ()).throw(
                    UsageLimitError(raw_message="usage limited")
                )
            ),
        ),
    )

    result = _execute_provider_invocation_at_adapter_seam(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        adapter_kind=adapter_kind,
        request=request,
        stdout_lines=(output_line,),
        returncode=19,
        prepared_provider_session_id=prepared_provider_session_id,
    )

    assert result == provider_invocation_runtime.ProviderInvocationFailure(
        kind=provider_invocation_runtime.InvocationFailureKind.USAGE_LIMITED,
        detail="usage limited",
        stdout_lines=(output_line,),
        provider_session_id=expected_provider_session_id,
        usage=None,
        reset_time=None,
    )


def test_in_memory_adapter_records_request_before_empty_prepared_queue_failure(
    tmp_path: Path,
) -> None:
    adapter = provider_invocation_runtime.InMemoryProviderInvocationAdapter()
    request = provider_invocation_runtime.ProviderInvocationRequest(
        command="provider --run",
        worktree=tmp_path,
        environment={},
        prompt=provider_invocation_runtime.ProviderInvocationPrompt(
            content="rendered prompt",
        ),
        run_kind=RunKind.FRESH,
        provider_session_id=None,
        output_hooks=provider_invocation_runtime.ProviderOutputReductionHooks(
            reduce_output=lambda lines: ("".join(lines), None),
        ),
    )

    with pytest.raises(
        AssertionError,
        match="No prepared provider invocation remains.",
    ):
        adapter.execute(request)

    assert adapter.recorded_requests == [request]


@pytest.mark.parametrize(
    ("adapter_kind", "prepared_provider_session_id"),
    [
        ("production", None),
        ("in_memory", None),
    ],
)
def test_provider_invocation_adapter_supports_stream_observation_at_adapter_seam(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    adapter_kind: str,
    prepared_provider_session_id: str | None,
) -> None:
    process_lines = ("line 1\n", "line 2\n")

    class _ObservedReducer:
        def __init__(self) -> None:
            self.consumed_lines: list[str] = []

        def consume_stdout_lines(self, new_lines: list[str]) -> None:
            self.consumed_lines.extend(new_lines)

        def __call__(self, lines: list[str]) -> tuple[str, ProviderUsage | None]:
            return ("".join(self.consumed_lines), ProviderUsage(output_tokens=2))

    reducer = _ObservedReducer()
    request = provider_invocation_runtime.ProviderInvocationRequest(
        command="provider --run",
        worktree=tmp_path,
        environment={"PROVIDER_TOKEN": "secret"},
        prompt=provider_invocation_runtime.ProviderInvocationPrompt(
            content="rendered prompt",
        ),
        run_kind=RunKind.FRESH,
        provider_session_id=None,
        output_hooks=provider_invocation_runtime.ProviderOutputReductionHooks(
            reduce_output=reducer,
            extract_provider_session_id=lambda _lines: None,
        ),
    )

    result = _execute_provider_invocation_at_adapter_seam(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        adapter_kind=adapter_kind,
        request=request,
        stdout_lines=process_lines,
        prepared_provider_session_id=prepared_provider_session_id,
    )

    assert result == provider_invocation_runtime.ProviderInvocationResult(
        output="".join(process_lines),
        usage=ProviderUsage(output_tokens=2),
        stdout_lines=process_lines,
        provider_session_id=None,
    )


def test_production_adapter_keeps_first_observed_provider_session_id_on_hard_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_line = "provider output\n"

    class _Process:
        def __init__(self) -> None:
            self.stdout = iter([output_line])
            self.stderr = iter(())
            self.returncode = 23

        def wait(self) -> int:
            return 23

    monkeypatch.setattr(
        provider_invocation_runtime.subprocess,
        "Popen",
        lambda *args, **kwargs: _Process(),
    )
    observed_provider_session_ids = iter(
        ("provider-session-123", "provider-session-456")
    )
    request = provider_invocation_runtime.ProviderInvocationRequest(
        command="provider --run",
        worktree=tmp_path,
        environment={},
        prompt=provider_invocation_runtime.ProviderInvocationPrompt(
            content="rendered prompt",
        ),
        run_kind=RunKind.RESUME,
        provider_session_id="existing-session",
        output_hooks=provider_invocation_runtime.ProviderOutputReductionHooks(
            reduce_output=lambda lines: ("".join(lines), None),
            extract_provider_session_id=lambda _lines: next(
                observed_provider_session_ids
            ),
        ),
    )

    with pytest.raises(HardAgentError) as exc_info:
        provider_invocation_runtime.ProductionProviderInvocationAdapter().execute(
            request
        )

    assert getattr(exc_info.value, "provider_session_id") == "provider-session-123"


@pytest.mark.parametrize("failure_mode", ["start_failure", "reduction_failure"])
def test_production_adapter_cleans_up_prompt_file_on_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure_mode: str,
) -> None:
    prompt_path = tmp_path / ".provider_prompt"

    if failure_mode == "start_failure":

        def _raise_start_failure(*args: Any, **kwargs: Any) -> Any:
            raise OSError("failed to start provider")

        monkeypatch.setattr(
            provider_invocation_runtime.subprocess,
            "Popen",
            _raise_start_failure,
        )
    else:

        class _Process:
            def __init__(self) -> None:
                self.stdout = iter(["line 1\n"])
                self.stderr = iter(())
                self.returncode = 0

            def wait(self) -> int:
                return 0

        monkeypatch.setattr(
            provider_invocation_runtime.subprocess,
            "Popen",
            lambda *args, **kwargs: _Process(),
        )

    request = provider_invocation_runtime.ProviderInvocationRequest(
        command="provider --run",
        worktree=tmp_path,
        environment={"PROVIDER_TOKEN": "secret"},
        prompt=provider_invocation_runtime.ProviderInvocationPrompt(
            content="rendered prompt",
            path=prompt_path,
            cleanup_path=True,
        ),
        run_kind=RunKind.FRESH,
        provider_session_id=None,
        output_hooks=provider_invocation_runtime.ProviderOutputReductionHooks(
            reduce_output=(
                (lambda _lines: (_ for _ in ()).throw(RuntimeError("reduction failed")))
                if failure_mode == "reduction_failure"
                else (lambda lines: ("normalized output", None))
            )
        ),
    )

    with pytest.raises(
        OSError if failure_mode == "start_failure" else RuntimeError,
        match=(
            "failed to start provider"
            if failure_mode == "start_failure"
            else "reduction failed"
        ),
    ):
        provider_invocation_runtime.ProductionProviderInvocationAdapter().execute(
            request
        )

    assert not prompt_path.exists()


def test_production_adapter_classifies_usage_limit_emitted_only_on_stderr(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Codex prints its usage-limit notice to stderr with empty stdout.

    The adapter must merge stderr into the line stream it reduces, otherwise the
    usage-limit signal is lost and the run is misclassified as a clean
    completion. See the codex stderr-only path in ADR 0013's probe.
    """

    usage_limit_line = (
        json.dumps(
            {
                "type": "turn.failed",
                "error": {
                    "message": (
                        "You've hit your usage limit. "
                        "Try again at January 2, 5pm (UTC)."
                    )
                },
            }
        )
        + "\n"
    )

    class _Process:
        def __init__(self) -> None:
            self.stdout = iter(())
            self.stderr = iter([usage_limit_line])
            self.returncode = 1

        def wait(self) -> int:
            return 1

    monkeypatch.setattr(
        provider_invocation_runtime.subprocess,
        "Popen",
        lambda *args, **kwargs: _Process(),
    )
    fixed_local_tz = timezone(timedelta(hours=2))
    fixed_now_local = datetime(2027, 1, 2, 12, 0, tzinfo=fixed_local_tz)
    monkeypatch.setattr(
        builtin_provider_parsed_output._time_module,
        "now_local",
        lambda: fixed_now_local,
    )

    request = provider_invocation_runtime.ProviderInvocationRequest(
        command="codex exec --json",
        worktree=tmp_path,
        environment={},
        prompt=provider_invocation_runtime.ProviderInvocationPrompt(
            content="rendered prompt",
        ),
        run_kind=RunKind.FRESH,
        provider_session_id=None,
        output_hooks=provider_invocation_runtime.ProviderOutputReductionHooks(
            reduce_output=reduce_codex_stream,
        ),
    )

    result = provider_invocation_runtime.ProductionProviderInvocationAdapter().execute(
        request
    )
    expected_reset_time = datetime(
        2027,
        1,
        2,
        17,
        0,
        tzinfo=timezone.utc,
    ).astimezone(fixed_local_tz)

    assert result == provider_invocation_runtime.ProviderInvocationFailure(
        kind=provider_invocation_runtime.InvocationFailureKind.USAGE_LIMITED,
        detail=f"Usage limit reached (reset_time={expected_reset_time.isoformat()})",
        stdout_lines=(usage_limit_line,),
        provider_session_id=None,
        usage=None,
        reset_time=expected_reset_time,
    )


def test_production_adapter_streams_stderr_lines_to_live_output_and_reduction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Merged stderr must reach the live-output hook and the final reduction.

    A provider's stderr is part of its output stream: it should appear on the
    live feed (via ``consume_stdout_lines``), in the reduced line list, and in
    the returned ``stdout_lines`` alongside stdout.
    """

    class _Process:
        def __init__(self) -> None:
            self.stdout = iter(["out 1\n"])
            self.stderr = iter(["err 1\n", "err 2\n"])
            self.returncode = 0

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(
        provider_invocation_runtime.subprocess,
        "Popen",
        lambda *args, **kwargs: _Process(),
    )

    class _ObservedReducer:
        def __init__(self) -> None:
            self.consumed_lines: list[str] = []
            self.reduced_lines: list[str] | None = None

        def consume_stdout_lines(self, new_lines: list[str]) -> None:
            self.consumed_lines.extend(new_lines)

        def __call__(self, lines: list[str]) -> tuple[str, ProviderUsage | None]:
            self.reduced_lines = list(lines)
            return ("normalized output", None)

    reducer = _ObservedReducer()
    request = provider_invocation_runtime.ProviderInvocationRequest(
        command="provider --run",
        worktree=tmp_path,
        environment={},
        prompt=provider_invocation_runtime.ProviderInvocationPrompt(
            content="rendered prompt",
        ),
        run_kind=RunKind.FRESH,
        provider_session_id=None,
        output_hooks=provider_invocation_runtime.ProviderOutputReductionHooks(
            reduce_output=reducer,
        ),
    )

    result = provider_invocation_runtime.ProductionProviderInvocationAdapter().execute(
        request
    )

    assert reducer.consumed_lines == ["out 1\n", "err 1\n", "err 2\n"]
    assert reducer.reduced_lines == ["out 1\n", "err 1\n", "err 2\n"]
    assert result.stdout_lines == ("out 1\n", "err 1\n", "err 2\n")


def test_production_adapter_surfaces_provider_stderr_text_in_nonzero_shell_exit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _Process:
        def __init__(self) -> None:
            self.stdout = iter(())
            self.stderr = iter(["provider exploded\n"])
            self.returncode = 23

        def wait(self) -> int:
            return 23

    monkeypatch.setattr(
        provider_invocation_runtime.subprocess,
        "Popen",
        lambda *args, **kwargs: _Process(),
    )

    request = provider_invocation_runtime.ProviderInvocationRequest(
        command="provider --run",
        worktree=tmp_path,
        environment={},
        prompt=provider_invocation_runtime.ProviderInvocationPrompt(
            content="rendered prompt",
        ),
        run_kind=RunKind.FRESH,
        provider_session_id=None,
        output_hooks=provider_invocation_runtime.ProviderOutputReductionHooks(
            reduce_output=lambda lines: ("".join(lines), None),
        ),
    )

    with pytest.raises(
        HardAgentError,
        match=r"(?s)exit code 23.*provider exploded",
    ):
        provider_invocation_runtime.ProductionProviderInvocationAdapter().execute(
            request
        )


def test_production_adapter_surfaces_provider_stderr_text_in_nonzero_argv_exit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _Stdin:
        def __init__(self) -> None:
            self.writes: list[str] = []
            self.closed = False

        def write(self, content: str) -> None:
            self.writes.append(content)

        def close(self) -> None:
            self.closed = True

    class _Process:
        def __init__(self) -> None:
            self.stdin = _Stdin()
            self.stdout = iter(())
            self.stderr = iter(["argv path exploded\n"])
            self.returncode = 17

        def wait(self) -> int:
            return 17

    process = _Process()
    monkeypatch.setattr(
        provider_invocation_runtime.subprocess,
        "Popen",
        lambda *args, **kwargs: process,
    )
    monkeypatch.setattr(provider_invocation_runtime.shutil, "which", lambda _name: None)

    request = provider_invocation_runtime.ProviderInvocationRequest(
        worktree=tmp_path,
        environment={},
        prompt=provider_invocation_runtime.ProviderInvocationPrompt(
            content="rendered prompt",
        ),
        run_kind=RunKind.FRESH,
        provider_session_id=None,
        output_hooks=provider_invocation_runtime.ProviderOutputReductionHooks(
            reduce_output=lambda lines: ("".join(lines), None),
        ),
        argv=("provider", "--run"),
    )

    with pytest.raises(
        HardAgentError,
        match=r"(?s)exit code 17.*argv path exploded",
    ):
        provider_invocation_runtime.ProductionProviderInvocationAdapter().execute(
            request
        )

    assert process.stdin.writes == ["rendered prompt"]
    assert process.stdin.closed is True


def test_production_adapter_raises_hard_error_on_nonzero_exit_even_with_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _Process:
        def __init__(self) -> None:
            self.stdout = iter(["partial output\n"])
            self.stderr = iter(())
            self.returncode = 23

        def wait(self) -> int:
            return 23

    monkeypatch.setattr(
        provider_invocation_runtime.subprocess,
        "Popen",
        lambda *args, **kwargs: _Process(),
    )

    request = provider_invocation_runtime.ProviderInvocationRequest(
        command="provider --run",
        worktree=tmp_path,
        environment={},
        prompt=provider_invocation_runtime.ProviderInvocationPrompt(
            content="rendered prompt",
        ),
        run_kind=RunKind.FRESH,
        provider_session_id=None,
        output_hooks=provider_invocation_runtime.ProviderOutputReductionHooks(
            reduce_output=lambda lines: ("".join(lines), None),
        ),
    )

    with pytest.raises(HardAgentError, match="exit code 23"):
        provider_invocation_runtime.ProductionProviderInvocationAdapter().execute(
            request
        )


def test_production_adapter_raises_hard_error_on_zero_exit_with_empty_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _Process:
        def __init__(self) -> None:
            self.stdout = iter(["provider event with no final text\n"])
            self.stderr = iter(())
            self.returncode = 0

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(
        provider_invocation_runtime.subprocess,
        "Popen",
        lambda *args, **kwargs: _Process(),
    )

    request = provider_invocation_runtime.ProviderInvocationRequest(
        command="provider --run",
        worktree=tmp_path,
        environment={},
        prompt=provider_invocation_runtime.ProviderInvocationPrompt(
            content="rendered prompt",
        ),
        run_kind=RunKind.FRESH,
        provider_session_id=None,
        output_hooks=provider_invocation_runtime.ProviderOutputReductionHooks(
            reduce_output=lambda _lines: ("", None),
        ),
    )

    with pytest.raises(
        HardAgentError,
        match="without producing output",
    ):
        provider_invocation_runtime.ProductionProviderInvocationAdapter().execute(
            request
        )


def test_production_adapter_raises_hard_error_on_nonzero_exit_with_empty_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _Process:
        def __init__(self) -> None:
            self.stdout = iter(())
            self.stderr = iter(())
            self.returncode = 17

        def wait(self) -> int:
            return 17

    monkeypatch.setattr(
        provider_invocation_runtime.subprocess,
        "Popen",
        lambda *args, **kwargs: _Process(),
    )

    request = provider_invocation_runtime.ProviderInvocationRequest(
        command="provider --run",
        worktree=tmp_path,
        environment={},
        prompt=provider_invocation_runtime.ProviderInvocationPrompt(
            content="rendered prompt",
        ),
        run_kind=RunKind.FRESH,
        provider_session_id=None,
        output_hooks=provider_invocation_runtime.ProviderOutputReductionHooks(
            reduce_output=lambda _lines: ("", None),
        ),
    )

    with pytest.raises(HardAgentError, match="exit code 17"):
        provider_invocation_runtime.ProductionProviderInvocationAdapter().execute(
            request
        )


def test_production_adapter_preserves_exit_code_only_message_for_whitespace_stderr(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _Process:
        def __init__(self) -> None:
            self.stdout = iter(())
            self.stderr = iter(["   \n", "\t"])
            self.returncode = 19

        def wait(self) -> int:
            return 19

    monkeypatch.setattr(
        provider_invocation_runtime.subprocess,
        "Popen",
        lambda *args, **kwargs: _Process(),
    )

    request = provider_invocation_runtime.ProviderInvocationRequest(
        command="provider --run",
        worktree=tmp_path,
        environment={},
        prompt=provider_invocation_runtime.ProviderInvocationPrompt(
            content="rendered prompt",
        ),
        run_kind=RunKind.FRESH,
        provider_session_id=None,
        output_hooks=provider_invocation_runtime.ProviderOutputReductionHooks(
            reduce_output=lambda _lines: ("", None),
        ),
    )

    with pytest.raises(
        HardAgentError, match=r"^Provider subprocess exited with exit code 19\.$"
    ):
        provider_invocation_runtime.ProductionProviderInvocationAdapter().execute(
            request
        )


@pytest.mark.parametrize(
    "classified_failure",
    [
        UsageLimitError(),
        ProviderUnavailableError(
            "temporary provider failure",
            reason=ProviderUnavailableReason.TRANSIENT_API_ERROR,
            service_name="codex",
        ),
    ],
)
def test_production_adapter_preserves_reducer_classification_on_nonzero_exit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    classified_failure: UsageLimitError | ProviderUnavailableError,
) -> None:
    class _Process:
        def __init__(self) -> None:
            self.stdout = iter(["classified failure output\n"])
            self.stderr = iter(())
            self.returncode = 19

        def wait(self) -> int:
            return 19

    monkeypatch.setattr(
        provider_invocation_runtime.subprocess,
        "Popen",
        lambda *args, **kwargs: _Process(),
    )

    request = provider_invocation_runtime.ProviderInvocationRequest(
        command="provider --run",
        worktree=tmp_path,
        environment={},
        prompt=provider_invocation_runtime.ProviderInvocationPrompt(
            content="rendered prompt",
        ),
        run_kind=RunKind.FRESH,
        provider_session_id=None,
        output_hooks=provider_invocation_runtime.ProviderOutputReductionHooks(
            reduce_output=lambda _lines: (_ for _ in ()).throw(classified_failure),
        ),
    )

    result = provider_invocation_runtime.ProductionProviderInvocationAdapter().execute(
        request
    )

    if isinstance(classified_failure, UsageLimitError):
        expected = provider_invocation_runtime.ProviderInvocationFailure(
            kind=provider_invocation_runtime.InvocationFailureKind.USAGE_LIMITED,
            detail=str(classified_failure),
            stdout_lines=("classified failure output\n",),
            provider_session_id=None,
            usage=classified_failure.usage,
            reset_time=classified_failure.reset_time,
        )
    else:
        expected = provider_invocation_runtime.ProviderInvocationFailure(
            kind=provider_invocation_runtime.InvocationFailureKind.PROVIDER_UNAVAILABLE,
            detail=str(classified_failure),
            stdout_lines=("classified failure output\n",),
            provider_session_id=None,
            usage=classified_failure.usage,
            reset_time=None,
            provider_unavailable_reason=classified_failure.reason,
        )

    assert result == expected


def test_production_completed_output_returns_classified_interruption_before_hard_exit_handling(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_line = '{"session":"provider-session-123"}\n'

    class _Process:
        def __init__(self) -> None:
            self.stdout = iter([output_line])
            self.stderr = iter(())
            self.returncode = 19

        def wait(self) -> int:
            return 19

    monkeypatch.setattr(
        provider_invocation_runtime.subprocess,
        "Popen",
        lambda *args, **kwargs: _Process(),
    )

    request = provider_invocation_runtime.ProviderInvocationRequest(
        command="provider --run",
        worktree=tmp_path,
        environment={},
        prompt=provider_invocation_runtime.ProviderInvocationPrompt(
            content="rendered prompt",
        ),
        run_kind=RunKind.RESUME,
        provider_session_id="existing-session",
        output_hooks=provider_invocation_runtime.ProviderOutputReductionHooks(
            reduce_output=(
                lambda _lines: (_ for _ in ()).throw(
                    UsageLimitError(raw_message="usage limited")
                )
            ),
            extract_provider_session_id=lambda _lines: "provider-session-123",
        ),
    )

    result = provider_invocation_runtime.ProductionProviderInvocationAdapter().execute(
        request
    )

    assert result == provider_invocation_runtime.ProviderInvocationFailure(
        kind=provider_invocation_runtime.InvocationFailureKind.USAGE_LIMITED,
        detail="usage limited",
        stdout_lines=(output_line,),
        provider_session_id="provider-session-123",
        usage=None,
        reset_time=None,
    )


@pytest.mark.parametrize(
    ("adapter_factory", "needs_monkeypatch"),
    [
        (
            lambda output_line: (
                provider_invocation_runtime.ProductionProviderInvocationAdapter()
            ),
            True,
        ),
        (
            lambda output_line: (
                provider_invocation_runtime.InMemoryProviderInvocationAdapter(
                    prepared_invocations=[
                        provider_invocation_runtime.ProviderInvocationPreparedStream(
                            stdout_lines=(output_line,),
                            provider_session_id="prepared-session",
                        )
                    ]
                )
            ),
            False,
        ),
    ],
)
def test_provider_invocation_failure_preserves_provider_unavailable_reason_after_observed_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    adapter_factory: Any,
    needs_monkeypatch: bool,
) -> None:
    output_line = '{"session":"provider-session-123"}\n'
    usage = ProviderUsage(input_tokens=11, output_tokens=7)
    classified_failure = ProviderUnavailableError(
        "temporary provider failure",
        reason=ProviderUnavailableReason.TRANSIENT_API_ERROR,
        service_name="codex",
        usage=usage,
    )

    if needs_monkeypatch:

        class _Process:
            def __init__(self) -> None:
                self.stdout = iter([output_line])
                self.stderr = iter(())
                self.returncode = 0

            def wait(self) -> int:
                return 0

        monkeypatch.setattr(
            provider_invocation_runtime.subprocess,
            "Popen",
            lambda *args, **kwargs: _Process(),
        )

    request = provider_invocation_runtime.ProviderInvocationRequest(
        command="provider --run",
        worktree=tmp_path,
        environment={},
        prompt=provider_invocation_runtime.ProviderInvocationPrompt(
            content="rendered prompt",
        ),
        run_kind=RunKind.RESUME,
        provider_session_id="existing-session",
        output_hooks=provider_invocation_runtime.ProviderOutputReductionHooks(
            reduce_output=lambda _lines: (_ for _ in ()).throw(classified_failure),
            extract_provider_session_id=lambda _lines: "provider-session-123",
        ),
    )

    result = adapter_factory(output_line).execute(request)

    assert result == provider_invocation_runtime.ProviderInvocationFailure(
        kind=provider_invocation_runtime.InvocationFailureKind.PROVIDER_UNAVAILABLE,
        detail="temporary provider failure",
        stdout_lines=(output_line,),
        provider_session_id="provider-session-123",
        usage=usage,
        reset_time=None,
        provider_unavailable_reason=ProviderUnavailableReason.TRANSIENT_API_ERROR,
    )


@pytest.mark.parametrize(
    ("adapter_factory", "needs_monkeypatch"),
    [
        (
            lambda output_line: (
                provider_invocation_runtime.ProductionProviderInvocationAdapter()
            ),
            True,
        ),
        (
            lambda output_line: (
                provider_invocation_runtime.InMemoryProviderInvocationAdapter(
                    prepared_invocations=[
                        provider_invocation_runtime.ProviderInvocationPreparedStream(
                            stdout_lines=(output_line,),
                            provider_session_id="prepared-session",
                        )
                    ]
                )
            ),
            False,
        ),
    ],
)
def test_provider_invocation_failure_preserves_usage_limit_metadata_after_observed_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    adapter_factory: Any,
    needs_monkeypatch: bool,
) -> None:
    output_line = '{"session":"provider-session-123"}\n'
    usage = ProviderUsage(input_tokens=13, output_tokens=5)
    reset_time = datetime(2027, 1, 2, 17, 0, tzinfo=timezone.utc)
    classified_failure = UsageLimitError(
        reset_time=reset_time,
        usage=usage,
    )

    if needs_monkeypatch:

        class _Process:
            def __init__(self) -> None:
                self.stdout = iter([output_line])
                self.stderr = iter(())
                self.returncode = 19

            def wait(self) -> int:
                return 19

        monkeypatch.setattr(
            provider_invocation_runtime.subprocess,
            "Popen",
            lambda *args, **kwargs: _Process(),
        )

    request = provider_invocation_runtime.ProviderInvocationRequest(
        command="provider --run",
        worktree=tmp_path,
        environment={},
        prompt=provider_invocation_runtime.ProviderInvocationPrompt(
            content="rendered prompt",
        ),
        run_kind=RunKind.RESUME,
        provider_session_id="existing-session",
        output_hooks=provider_invocation_runtime.ProviderOutputReductionHooks(
            reduce_output=lambda _lines: (_ for _ in ()).throw(classified_failure),
            extract_provider_session_id=lambda _lines: "provider-session-123",
        ),
    )

    result = adapter_factory(output_line).execute(request)

    assert result == provider_invocation_runtime.ProviderInvocationFailure(
        kind=provider_invocation_runtime.InvocationFailureKind.USAGE_LIMITED,
        detail=str(classified_failure),
        stdout_lines=(output_line,),
        provider_session_id="provider-session-123",
        usage=usage,
        reset_time=reset_time,
    )


def test_provider_invocation_request_requires_command_or_argv() -> None:
    with pytest.raises(
        ValueError, match="ProviderInvocationRequest requires command or argv"
    ):
        provider_invocation_runtime.ProviderInvocationRequest(
            worktree=Path("/tmp/worktree"),
            environment={},
            prompt=provider_invocation_runtime.ProviderInvocationPrompt(
                content="rendered prompt"
            ),
            run_kind=RunKind.FRESH,
            provider_session_id=None,
            output_hooks=provider_invocation_runtime.ProviderOutputReductionHooks(
                reduce_output=lambda lines: ("".join(lines), None)
            ),
        )


def test_production_adapter_terminates_silent_subprocess_after_idle_timeout(
    tmp_path: Path,
) -> None:
    marker_path = tmp_path / "child-started"
    script_path = tmp_path / "silent_provider.py"
    script_path.write_text(
        "\n".join(
            [
                "import os",
                "import signal",
                "import sys",
                "import time",
                "",
                "marker_path = sys.argv[1]",
                "with open(marker_path, 'w', encoding='utf-8') as marker:",
                "    marker.write(str(os.getpid()))",
                "    marker.flush()",
                "",
                "signal.signal(signal.SIGTERM, lambda _signum, _frame: sys.exit(0))",
                "while True:",
                "    time.sleep(60)",
            ]
        ),
        encoding="utf-8",
    )

    request = provider_invocation_runtime.ProviderInvocationRequest(
        worktree=tmp_path,
        environment={},
        prompt=provider_invocation_runtime.ProviderInvocationPrompt(
            content="rendered prompt",
        ),
        run_kind=RunKind.FRESH,
        provider_session_id="provider-session-123",
        output_hooks=provider_invocation_runtime.ProviderOutputReductionHooks(
            reduce_output=lambda lines: ("".join(lines), None),
        ),
        argv=(sys.executable, str(script_path), str(marker_path)),
        prefer_argv=True,
        timeout_seconds=1,
    )

    with pytest.raises(provider_invocation_runtime.ProviderInvocationTimedOutError):
        provider_invocation_runtime.ProductionProviderInvocationAdapter().execute(
            request
        )

    child_pid = int(marker_path.read_text(encoding="utf-8"))
    _assert_subprocess_is_dead(child_pid)


def test_provider_invocation_request_layers_windows_host_process_allowlist_for_built_in_invocations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        provider_invocation_runtime,
        "os",
        SimpleNamespace(name="nt", environ=os.environ),
    )
    monkeypatch.setenv("PATH", "host/path")
    monkeypatch.setenv("PATHEXT", ".COM;.EXE;.BAT;.CMD")
    monkeypatch.setenv("SystemRoot", "C:\\Windows")
    monkeypatch.setenv("ComSpec", "C:\\Windows\\System32\\cmd.exe")
    monkeypatch.setenv("WINDIR", "C:\\Windows")

    captured: dict[str, Any] = {}

    class _Process:
        def __init__(self) -> None:
            self.stdin = None
            self.stdout = iter(["output line\n"])
            self.stderr = iter(())
            self.returncode = 0

        def wait(self) -> int:
            return 0

    def _fake_popen(
        command: list[str],
        *,
        shell: bool,
        cwd: Path,
        env: dict[str, str],
        stdout: Any,
        stderr: Any,
        text: bool,
        stdin: Any = None,
    ) -> _Process:
        captured["command"] = command
        captured["env"] = env
        return _Process()

    monkeypatch.setattr(provider_invocation_runtime.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(provider_invocation_runtime.shutil, "which", lambda _name: None)
    provider_invocation_runtime.ProductionProviderInvocationAdapter().execute(
        provider_invocation_runtime.ProviderInvocationRequest(
            worktree=tmp_path,
            environment={"TZ": "UTC", "PATH": "provider/path"},
            prompt=provider_invocation_runtime.ProviderInvocationPrompt(
                content="rendered prompt",
            ),
            run_kind=RunKind.FRESH,
            provider_session_id=None,
            output_hooks=provider_invocation_runtime.ProviderOutputReductionHooks(
                reduce_output=lambda lines: ("".join(lines), None)
            ),
            argv=("provider", "--run"),
            prefer_argv=True,
        )
    )

    assert captured["command"] == ["provider", "--run"]
    assert captured["env"] == {
        "PATH": "provider/path",
        "PATHEXT": ".COM;.EXE;.BAT;.CMD",
        "SystemRoot": "C:\\Windows",
        "ComSpec": "C:\\Windows\\System32\\cmd.exe",
        "WINDIR": "C:\\Windows",
        "TZ": "UTC",
    }


def test_provider_invocation_request_keeps_provider_environment_values_when_windows_keys_collide(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        provider_invocation_runtime,
        "os",
        SimpleNamespace(name="nt", environ=os.environ),
    )
    monkeypatch.setenv("PATH", "host/path")
    monkeypatch.setenv("PATHEXT", ".COM;.EXE;.BAT;.CMD")
    monkeypatch.setenv("SystemRoot", "C:\\Windows")
    monkeypatch.setenv("ComSpec", "C:\\Windows\\System32\\cmd.exe")
    monkeypatch.setenv("WINDIR", "C:\\Windows")

    captured: dict[str, dict[str, str]] = {}

    class _Process:
        def __init__(self) -> None:
            self.stdin = None
            self.stdout = iter(["output line\n"])
            self.stderr = iter(())
            self.returncode = 0

        def wait(self) -> int:
            return 0

    def _fake_popen(
        command: list[str],
        *,
        shell: bool,
        cwd: Path,
        env: dict[str, str],
        stdout: Any,
        stderr: Any,
        text: bool,
        stdin: Any = None,
    ) -> _Process:
        captured["env"] = env
        return _Process()

    monkeypatch.setattr(provider_invocation_runtime.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(provider_invocation_runtime.shutil, "which", lambda _name: None)
    provider_invocation_runtime.ProductionProviderInvocationAdapter().execute(
        provider_invocation_runtime.ProviderInvocationRequest(
            worktree=Path("/tmp"),
            environment={"PATH": "provider-path", "WINDIR": "provider-windir"},
            prompt=provider_invocation_runtime.ProviderInvocationPrompt(
                content="rendered prompt",
            ),
            run_kind=RunKind.FRESH,
            provider_session_id=None,
            output_hooks=provider_invocation_runtime.ProviderOutputReductionHooks(
                reduce_output=lambda lines: ("".join(lines), None)
            ),
            argv=("provider",),
            prefer_argv=True,
        )
    )

    assert captured["env"]["PATH"] == "provider-path"
    assert captured["env"]["WINDIR"] == "provider-windir"


def test_provider_invocation_request_keeps_provider_environment_unchanged_on_posix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        provider_invocation_runtime,
        "os",
        SimpleNamespace(name="posix", environ=os.environ),
    )
    monkeypatch.setenv("PATH", "host/path")

    captured: dict[str, dict[str, str]] = {}

    class _Process:
        def __init__(self) -> None:
            self.stdin = None
            self.stdout = iter(["output line\n"])
            self.stderr = iter(())
            self.returncode = 0

        def wait(self) -> int:
            return 0

    def _fake_popen(
        command: list[str],
        *,
        shell: bool,
        cwd: Path,
        env: dict[str, str],
        stdout: Any,
        stderr: Any,
        text: bool,
        stdin: Any = None,
    ) -> _Process:
        captured["env"] = env
        return _Process()

    monkeypatch.setattr(provider_invocation_runtime.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(provider_invocation_runtime.shutil, "which", lambda _name: None)
    provider_invocation_runtime.ProductionProviderInvocationAdapter().execute(
        provider_invocation_runtime.ProviderInvocationRequest(
            worktree=Path("/tmp"),
            environment={"TZ": "UTC", "PATH": "provider/path"},
            prompt=provider_invocation_runtime.ProviderInvocationPrompt(
                content="rendered prompt",
            ),
            run_kind=RunKind.FRESH,
            provider_session_id=None,
            output_hooks=provider_invocation_runtime.ProviderOutputReductionHooks(
                reduce_output=lambda lines: ("".join(lines), None)
            ),
            argv=("provider",),
            prefer_argv=True,
        )
    )

    assert captured["env"] == {"TZ": "UTC", "PATH": "provider/path"}


def test_production_adapter_resets_idle_timeout_on_stderr_activity(
    tmp_path: Path,
) -> None:
    script_path = tmp_path / "stderr_heartbeat_provider.py"
    script_path.write_text(
        "\n".join(
            [
                "import sys",
                "import time",
                "",
                "for line in ('heartbeat 1', 'heartbeat 2'):",
                "    print(line, file=sys.stderr, flush=True)",
                "    time.sleep(0.6)",
                "print('final output', flush=True)",
            ]
        ),
        encoding="utf-8",
    )

    request = provider_invocation_runtime.ProviderInvocationRequest(
        worktree=tmp_path,
        environment={},
        prompt=provider_invocation_runtime.ProviderInvocationPrompt(
            content="rendered prompt",
        ),
        run_kind=RunKind.FRESH,
        provider_session_id=None,
        output_hooks=provider_invocation_runtime.ProviderOutputReductionHooks(
            reduce_output=lambda _lines: ("normalized output", None),
        ),
        argv=(sys.executable, str(script_path)),
        prefer_argv=True,
        timeout_seconds=1,
    )

    result = provider_invocation_runtime.ProductionProviderInvocationAdapter().execute(
        request
    )

    assert result == provider_invocation_runtime.ProviderInvocationResult(
        output="normalized output",
        usage=None,
        stdout_lines=("final output\n", "heartbeat 1\n", "heartbeat 2\n"),
        provider_session_id=None,
    )


@pytest.mark.parametrize("timeout_seconds", [0, -1])
def test_production_adapter_disables_idle_timeout_for_non_positive_timeout_values(
    tmp_path: Path,
    timeout_seconds: int,
) -> None:
    script_path = tmp_path / "delayed_output_provider.py"
    script_path.write_text(
        "\n".join(
            [
                "import time",
                "",
                "time.sleep(1.2)",
                "print('final output', flush=True)",
            ]
        ),
        encoding="utf-8",
    )

    request = provider_invocation_runtime.ProviderInvocationRequest(
        worktree=tmp_path,
        environment={},
        prompt=provider_invocation_runtime.ProviderInvocationPrompt(
            content="rendered prompt",
        ),
        run_kind=RunKind.FRESH,
        provider_session_id=None,
        output_hooks=provider_invocation_runtime.ProviderOutputReductionHooks(
            reduce_output=lambda lines: ("".join(lines).strip(), None),
        ),
        argv=(sys.executable, str(script_path)),
        prefer_argv=True,
        timeout_seconds=timeout_seconds,
    )

    result = provider_invocation_runtime.ProductionProviderInvocationAdapter().execute(
        request
    )

    assert result == provider_invocation_runtime.ProviderInvocationResult(
        output="final output",
        usage=None,
        stdout_lines=("final output\n",),
        provider_session_id=None,
    )


def test_production_adapter_cancels_running_subprocess_on_token_cancellation(
    tmp_path: Path,
) -> None:
    marker_path = tmp_path / "child-started"
    script_path = tmp_path / "hanging_provider.py"
    script_path.write_text(
        "\n".join(
            [
                "import os",
                "import sys",
                "import time",
                "",
                "marker_path = sys.argv[1]",
                "with open(marker_path, 'w', encoding='utf-8') as f:",
                "    f.write(str(os.getpid()))",
                "    f.flush()",
                "while True:",
                "    time.sleep(60)",
            ]
        ),
        encoding="utf-8",
    )

    token = CancellationToken()

    def _cancel_after_start() -> None:
        while not marker_path.exists():
            time.sleep(0.05)
        time.sleep(0.1)
        token.cancel()

    cancel_thread = threading.Thread(target=_cancel_after_start, daemon=True)
    cancel_thread.start()

    request = provider_invocation_runtime.ProviderInvocationRequest(
        worktree=tmp_path,
        environment={},
        prompt=provider_invocation_runtime.ProviderInvocationPrompt(content="prompt"),
        run_kind=RunKind.FRESH,
        provider_session_id="session-123",
        output_hooks=provider_invocation_runtime.ProviderOutputReductionHooks(
            reduce_output=lambda lines: ("".join(lines), None),
        ),
        argv=(sys.executable, str(script_path), str(marker_path)),
        prefer_argv=True,
        timeout_seconds=10,
        token=token,
    )

    with pytest.raises(AgentCancelledError):
        provider_invocation_runtime.ProductionProviderInvocationAdapter().execute(
            request
        )

    cancel_thread.join(timeout=5)
    child_pid = int(marker_path.read_text(encoding="utf-8"))
    _assert_subprocess_is_dead(child_pid)


def test_production_adapter_cancel_takes_precedence_over_simultaneous_idle_timeout(
    tmp_path: Path,
) -> None:
    script_path = tmp_path / "hanging_provider.py"
    script_path.write_text(
        "import time; time.sleep(60)\n",
        encoding="utf-8",
    )

    token = CancellationToken()
    token.cancel()

    request = provider_invocation_runtime.ProviderInvocationRequest(
        worktree=tmp_path,
        environment={},
        prompt=provider_invocation_runtime.ProviderInvocationPrompt(content="prompt"),
        run_kind=RunKind.FRESH,
        provider_session_id="session-123",
        output_hooks=provider_invocation_runtime.ProviderOutputReductionHooks(
            reduce_output=lambda lines: ("".join(lines), None),
        ),
        argv=(sys.executable, str(script_path)),
        prefer_argv=True,
        timeout_seconds=1,
        token=token,
    )

    with pytest.raises(AgentCancelledError):
        provider_invocation_runtime.ProductionProviderInvocationAdapter().execute(
            request
        )


def test_production_adapter_fires_idle_timeout_when_token_present_but_not_cancelled(
    tmp_path: Path,
) -> None:
    marker_path = tmp_path / "child-started"
    script_path = tmp_path / "silent_provider.py"
    script_path.write_text(
        "\n".join(
            [
                "import os",
                "import sys",
                "import time",
                "",
                "marker_path = sys.argv[1]",
                "with open(marker_path, 'w', encoding='utf-8') as f:",
                "    f.write(str(os.getpid()))",
                "    f.flush()",
                "while True:",
                "    time.sleep(60)",
            ]
        ),
        encoding="utf-8",
    )

    token = CancellationToken()

    request = provider_invocation_runtime.ProviderInvocationRequest(
        worktree=tmp_path,
        environment={},
        prompt=provider_invocation_runtime.ProviderInvocationPrompt(content="prompt"),
        run_kind=RunKind.FRESH,
        provider_session_id="session-123",
        output_hooks=provider_invocation_runtime.ProviderOutputReductionHooks(
            reduce_output=lambda lines: ("".join(lines), None),
        ),
        argv=(sys.executable, str(script_path), str(marker_path)),
        prefer_argv=True,
        timeout_seconds=1,
        token=token,
    )

    with pytest.raises(provider_invocation_runtime.ProviderInvocationTimedOutError):
        provider_invocation_runtime.ProductionProviderInvocationAdapter().execute(
            request
        )

    child_pid = int(marker_path.read_text(encoding="utf-8"))
    _assert_subprocess_is_dead(child_pid)
