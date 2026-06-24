from __future__ import annotations

import inspect
import json
import re
from pathlib import Path
from typing import Any

import pytest

import agent_runtime._provider_invocation as provider_invocation_runtime
import agent_runtime.runtime as prompt_runtime
from agent_runtime._builtin_runtime_client import _reduce_codex_stream
from agent_runtime.agent_log import AgentInvocationLog
from agent_runtime.errors import (
    HardAgentError,
    ProviderUnavailableError,
    ProviderUnavailableReason,
    UsageLimitError,
)
from agent_runtime.provider_usage import ProviderUsage
from agent_runtime.session import RunKind


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
        log_context=None,
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
        "env": {"PROVIDER_TOKEN": "secret"},
        "stdout": provider_invocation_runtime.subprocess.PIPE,
        "stderr": provider_invocation_runtime.subprocess.PIPE,
        "text": True,
        "reduced_lines": ["line 1\n", "line 2\n"],
    }


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

    request = provider_invocation_runtime.ProviderInvocationRequest(
        worktree=tmp_path,
        environment={"PROVIDER_TOKEN": "secret"},
        prompt=provider_invocation_runtime.ProviderInvocationPrompt(
            content="rendered prompt",
            path=prompt_path,
            cleanup_path=True,
        ),
        run_kind=RunKind.FRESH,
        log_context=None,
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
        "command": ("provider", "--run"),
        "shell": False,
        "cwd": tmp_path,
        "env": {"PROVIDER_TOKEN": "secret"},
        "stdout": provider_invocation_runtime.subprocess.PIPE,
        "stderr": provider_invocation_runtime.subprocess.PIPE,
        "text": True,
        "stdin": provider_invocation_runtime.subprocess.PIPE,
    }


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
        log_context=None,
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
        "command": request.argv,
        "shell": False,
        "cwd": tmp_path,
        "env": {"CLAUDE_CODE_OAUTH_TOKEN": "secret"},
        "stdout": provider_invocation_runtime.subprocess.PIPE,
        "stderr": provider_invocation_runtime.subprocess.PIPE,
        "text": True,
        "stdin": provider_invocation_runtime.subprocess.PIPE,
    }


def test_production_adapter_records_provider_chunks_and_session_id_when_log_context_is_supplied(
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

    logs_dir = tmp_path / "logs"
    invocation_log = AgentInvocationLog().start_logical_session(
        log_name="implementer",
        logs_dir=logs_dir,
    )

    def _reduce_logged_output(
        lines: list[str],
        work_invocation_log: Any,
    ) -> tuple[str, ProviderUsage | None]:
        work_invocation_log.record_provider_session_id("provider-session-123")
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
        log_context=provider_invocation_runtime.ProviderInvocationLogContext(
            invocation_log=invocation_log,
        ),
        provider_session_id="existing-session",
        output_hooks=provider_invocation_runtime.ProviderOutputReductionHooks(
            reduce_output=lambda lines: ("unexpected", None),
            reduce_logged_output=_reduce_logged_output,
            extract_provider_session_id=lambda _lines: "provider-session-123",
        ),
    )

    result = provider_invocation_runtime.ProductionProviderInvocationAdapter().execute(
        request
    )

    log_path = next(logs_dir.glob("*.log"))
    log_text = log_path.read_text(encoding="utf-8")

    assert result == provider_invocation_runtime.ProviderInvocationResult(
        output="normalized output",
        usage=ProviderUsage(output_tokens=7),
        stdout_lines=tuple(process_lines),
        provider_session_id="provider-session-123",
    )
    assert '"provider_session_id": "provider-session-123"' in log_text
    assert '{"session":"provider-session-123"}\nfinal line\n' in log_text
    builtin_source = inspect.getsource(prompt_runtime._builtin_runtime_client_module)
    provider_source = inspect.getsource(provider_invocation_runtime)
    assert "subprocess.Popen(" not in builtin_source
    assert "prompt_path.write_text(request.prompt.content" not in builtin_source
    assert "prompt_path.unlink(missing_ok=True)" not in builtin_source
    assert "subprocess.Popen(" in provider_source
    assert "prompt_path.write_text(request.prompt.content" in provider_source
    assert "prompt_path.unlink(missing_ok=True)" in provider_source
    assert "append_provider_chunk" in provider_source
    assert "stdout_lines=tuple(stdout_lines)" in provider_source
    assert "provider_session_id=_observed_provider_session_id()" in provider_source


@pytest.mark.parametrize(
    ("adapter_factory", "needs_monkeypatch"),
    [
        (
            lambda: provider_invocation_runtime.ProductionProviderInvocationAdapter(),
            True,
        ),
        (
            lambda: provider_invocation_runtime.InMemoryProviderInvocationAdapter(
                prepared_invocations=[
                    provider_invocation_runtime.ProviderInvocationPreparedStream(
                        stdout_lines=("line 1\n", "line 2\n")
                    )
                ]
            ),
            False,
        ),
    ],
)
def test_provider_invocation_seam_consumes_stdout_lines_before_final_reduction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    adapter_factory: Any,
    needs_monkeypatch: bool,
) -> None:
    prompt_path = tmp_path / ".provider_prompt"
    observed_steps: list[str] = []

    if needs_monkeypatch:

        class _Process:
            def __init__(self) -> None:
                self.stdout = iter(["line 1\n", "line 2\n"])
                self.stderr = iter(())
                self.returncode = 0

            def wait(self) -> int:
                observed_steps.append("wait")
                return 0

        monkeypatch.setattr(
            provider_invocation_runtime.subprocess,
            "Popen",
            lambda *args, **kwargs: _Process(),
        )

    class _ObservedReducer:
        def __init__(self) -> None:
            self.consumed_lines: list[str] = []

        def consume_stdout_lines(self, new_lines: list[str]) -> None:
            self.consumed_lines.extend(new_lines)
            observed_steps.extend(f"consume:{line.rstrip()}" for line in new_lines)

        def __call__(self, lines: list[str]) -> tuple[str, ProviderUsage | None]:
            observed_steps.append("reduce")
            assert self.consumed_lines == lines
            return ("normalized output", ProviderUsage(output_tokens=2))

    reducer = _ObservedReducer()
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
        log_context=None,
        provider_session_id=None,
        output_hooks=provider_invocation_runtime.ProviderOutputReductionHooks(
            reduce_output=reducer,
            extract_provider_session_id=lambda _lines: None,
        ),
    )

    result = adapter_factory().execute(request)

    assert result == provider_invocation_runtime.ProviderInvocationResult(
        output="normalized output",
        usage=ProviderUsage(output_tokens=2),
        stdout_lines=("line 1\n", "line 2\n"),
        provider_session_id=None,
    )
    expected_steps = ["consume:line 1", "consume:line 2"]
    if needs_monkeypatch:
        expected_steps.append("wait")
    expected_steps.append("reduce")
    assert observed_steps == expected_steps


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
        log_context=None,
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

    request = provider_invocation_runtime.ProviderInvocationRequest(
        command="codex exec --json",
        worktree=tmp_path,
        environment={},
        prompt=provider_invocation_runtime.ProviderInvocationPrompt(
            content="rendered prompt",
        ),
        run_kind=RunKind.FRESH,
        log_context=None,
        provider_session_id=None,
        output_hooks=provider_invocation_runtime.ProviderOutputReductionHooks(
            reduce_output=lambda lines: _reduce_codex_stream(lines),
        ),
    )

    with pytest.raises(UsageLimitError):
        provider_invocation_runtime.ProductionProviderInvocationAdapter().execute(
            request
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
        log_context=None,
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
        log_context=None,
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
        log_context=None,
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
        log_context=None,
        provider_session_id=None,
        output_hooks=provider_invocation_runtime.ProviderOutputReductionHooks(
            reduce_output=lambda _lines: ("", None),
        ),
    )

    with pytest.raises(HardAgentError, match="exit code 17"):
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
        log_context=None,
        provider_session_id=None,
        output_hooks=provider_invocation_runtime.ProviderOutputReductionHooks(
            reduce_output=lambda _lines: (_ for _ in ()).throw(classified_failure),
        ),
    )

    with pytest.raises(
        type(classified_failure),
        match=re.escape(str(classified_failure)),
    ):
        provider_invocation_runtime.ProductionProviderInvocationAdapter().execute(
            request
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
            log_context=None,
            provider_session_id=None,
            output_hooks=provider_invocation_runtime.ProviderOutputReductionHooks(
                reduce_output=lambda lines: ("".join(lines), None)
            ),
        )
