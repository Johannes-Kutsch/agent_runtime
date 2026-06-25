from __future__ import annotations

import dataclasses
import os
import queue
import shutil
import subprocess
import shlex
import threading
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Protocol

from .agent_log import LogicalAgentInvocationLog, WorkInvocationLog
from .errors import (
    AgentTimeoutError,
    HardAgentError,
    ProviderUnavailableError,
    UsageLimitError,
)
from .provider_usage import ProviderUsage
from .session import RunKind

ProviderOutputReducer = Callable[[list[str]], tuple[str, ProviderUsage | None]]
ProviderLoggedOutputReducer = Callable[
    [list[str], WorkInvocationLog], tuple[str, ProviderUsage | None]
]
ProviderSessionIdExtractor = Callable[[list[str]], str | None]


def _consume_new_stdout_lines(
    reduce_output: Callable[[list[str]], tuple[str, ProviderUsage | None]],
    new_lines: list[str],
) -> None:
    consume_stdout_lines = getattr(reduce_output, "consume_stdout_lines", None)
    if callable(consume_stdout_lines):
        consume_stdout_lines(new_lines)


@dataclasses.dataclass(frozen=True, slots=True)
class ProviderInvocationPrompt:
    content: str
    path: Path | None = None
    cleanup_path: bool = False


@dataclasses.dataclass(frozen=True, slots=True)
class ProviderOutputReductionHooks:
    reduce_output: ProviderOutputReducer
    reduce_logged_output: ProviderLoggedOutputReducer | None = None
    extract_provider_session_id: ProviderSessionIdExtractor | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class ProviderInvocationLogContext:
    invocation_log: LogicalAgentInvocationLog


@dataclasses.dataclass(frozen=True, slots=True)
class ProviderInvocationRequest:
    worktree: Path
    environment: Mapping[str, str]
    prompt: ProviderInvocationPrompt
    run_kind: RunKind
    log_context: ProviderInvocationLogContext | None
    provider_session_id: str | None
    output_hooks: ProviderOutputReductionHooks
    command: str = ""
    argv: tuple[str, ...] = ()
    prefer_argv: bool = False
    timeout_seconds: int = 300

    def __post_init__(self) -> None:
        if not self.argv and not self.command:
            raise ValueError("ProviderInvocationRequest requires command or argv")
        if not self.command:
            object.__setattr__(
                self,
                "command",
                " ".join(shlex.quote(arg) for arg in self.argv),
            )
            object.__setattr__(self, "prefer_argv", True)


@dataclasses.dataclass(frozen=True, slots=True)
class ProviderInvocationResult:
    output: str
    usage: ProviderUsage | None = None
    stdout_lines: tuple[str, ...] = ()
    provider_session_id: str | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class ProviderInvocationPreparedStream:
    stdout_lines: tuple[str, ...] = ()
    provider_session_id: str | None = None


class ProviderInvocationTimedOutError(AgentTimeoutError):
    pass


class InvocationFailureKind(str, Enum):
    USAGE_LIMITED = "USAGE_LIMITED"
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"


@dataclasses.dataclass(frozen=True, slots=True)
class ProviderInvocationFailure:
    kind: InvocationFailureKind
    detail: str
    stdout_lines: tuple[str, ...] = ()
    provider_session_id: str | None = None
    usage: ProviderUsage | None = None
    reset_time: datetime | None = None


class ProviderInvocationAdapter(Protocol):
    def execute(
        self,
        request: ProviderInvocationRequest,
    ) -> ProviderInvocationResult | ProviderInvocationFailure: ...


def _provider_invocation_failure_from_error(
    error: UsageLimitError | ProviderUnavailableError,
    *,
    stdout_lines: tuple[str, ...],
    provider_session_id: str | None,
) -> ProviderInvocationFailure:
    if isinstance(error, UsageLimitError):
        return ProviderInvocationFailure(
            kind=InvocationFailureKind.USAGE_LIMITED,
            detail=error.raw_message or str(error),
            stdout_lines=stdout_lines,
            provider_session_id=provider_session_id,
            usage=error.usage,
            reset_time=error.reset_time,
        )
    return ProviderInvocationFailure(
        kind=InvocationFailureKind.PROVIDER_UNAVAILABLE,
        detail=str(error),
        stdout_lines=stdout_lines,
        provider_session_id=provider_session_id,
        usage=error.usage,
        reset_time=None,
    )


def _nonzero_exit_message(returncode: int, observed_lines: list[str]) -> str:
    message = f"Provider subprocess exited with exit code {returncode}."
    observed_output = "".join(observed_lines)
    if not observed_output.strip():
        return message
    return f"{message} Provider output:\n{observed_output}"


class ProductionProviderInvocationAdapter:
    def _windows_process_base_env(self) -> dict[str, str]:
        if os.name != "nt":
            return {}
        return {
            key: os.environ[key]
            for key in ("PATH", "PATHEXT", "SystemRoot", "ComSpec", "WINDIR")
            if key in os.environ and os.environ[key]
        }

    def execute(
        self,
        request: ProviderInvocationRequest,
    ) -> ProviderInvocationResult | ProviderInvocationFailure:
        use_shell = not (request.prefer_argv and request.argv)
        prompt_path = request.prompt.path
        prompt_file_created = use_shell and prompt_path is not None
        if prompt_file_created and prompt_path is not None:
            prompt_path.write_text(request.prompt.content, encoding="utf-8")

        environment = dict(request.environment)
        environment = {
            **self._windows_process_base_env(),
            **environment,
        }

        try:
            if use_shell:
                process = subprocess.Popen(
                    request.command,
                    shell=True,
                    cwd=request.worktree,
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            else:
                # Resolve argv[0] against PATH/PATHEXT before spawning with
                # shell=False. On Windows, CreateProcess only appends .exe to
                # a bare name and ignores PATHEXT, so an npm-installed CLI
                # shim (e.g. claude.cmd) is never found from the bare name
                # "claude". shutil.which returns the resolvable shim path;
                # fall back to the original name when it cannot be resolved.
                resolved_argv = list(request.argv)
                resolved_executable = shutil.which(resolved_argv[0])
                if resolved_executable is not None:
                    resolved_argv[0] = resolved_executable
                try:
                    process = subprocess.Popen(
                        resolved_argv,
                        shell=False,
                        stdin=subprocess.PIPE,
                        cwd=request.worktree,
                        env=environment,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                except TypeError as exc:
                    if "stdin" not in str(exc):
                        raise
                    process = subprocess.Popen(
                        resolved_argv,
                        shell=False,
                        cwd=request.worktree,
                        env=environment,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
            if not use_shell:
                process_stdin = getattr(process, "stdin", None)
                if process_stdin is not None:
                    process_stdin.write(request.prompt.content)
                    process_stdin.close()
            stdout_lines: list[str] = []
            stderr_lines: list[str] = []
            output_queue: queue.Queue[tuple[str, str | None]] = queue.Queue()

            def _drain_stream(
                stream: Iterable[str] | None,
                source: str,
            ) -> None:
                if stream is not None:
                    for line in stream:
                        output_queue.put((source, line))
                output_queue.put((source, None))

            stdout_thread = threading.Thread(
                target=_drain_stream,
                args=(process.stdout, "stdout"),
                daemon=True,
            )
            stderr_thread = threading.Thread(
                target=_drain_stream,
                args=(getattr(process, "stderr", None), "stderr"),
                daemon=True,
            )
            stdout_thread.start()
            stderr_thread.start()

            closed_streams: set[str] = set()
            while len(closed_streams) < 2:
                try:
                    if request.timeout_seconds > 0:
                        source, line = output_queue.get(timeout=request.timeout_seconds)
                    else:
                        source, line = output_queue.get()
                except queue.Empty as exc:
                    process.kill()
                    process.wait()
                    stdout_thread.join()
                    stderr_thread.join()
                    error = ProviderInvocationTimedOutError(
                        "Provider subprocess exceeded the idle timeout.",
                    )
                    setattr(error, "provider_session_id", request.provider_session_id)
                    raise error from exc

                if line is None:
                    closed_streams.add(source)
                    continue
                if source == "stdout":
                    stdout_lines.append(line)
                    _consume_new_stdout_lines(
                        request.output_hooks.reduce_output, [line]
                    )
                else:
                    stderr_lines.append(line)

            process.wait()

            if stderr_lines:
                _consume_new_stdout_lines(
                    request.output_hooks.reduce_output, stderr_lines
                )
                stdout_lines.extend(stderr_lines)

            def _extracted_provider_session_id() -> str | None:
                if request.output_hooks.extract_provider_session_id is None:
                    return None
                return request.output_hooks.extract_provider_session_id(stdout_lines)

            def _active_provider_session_id() -> str | None:
                provider_session_id = request.provider_session_id
                extracted_provider_session_id = _extracted_provider_session_id()
                if extracted_provider_session_id is not None:
                    provider_session_id = extracted_provider_session_id
                return provider_session_id

            try:
                output, usage = request.output_hooks.reduce_output(stdout_lines)
            except Exception as exc:
                if isinstance(exc, (UsageLimitError, ProviderUnavailableError)):
                    return _provider_invocation_failure_from_error(
                        exc,
                        stdout_lines=tuple(stdout_lines),
                        provider_session_id=_extracted_provider_session_id(),
                    )
                raise
            returncode = process.returncode
            observed_provider_session_id = _active_provider_session_id()
            if returncode != 0:
                hard_error = HardAgentError(
                    _nonzero_exit_message(returncode, stdout_lines),
                )
                setattr(
                    hard_error,
                    "provider_session_id",
                    observed_provider_session_id,
                )
                raise hard_error
            if not output.strip():
                hard_error = HardAgentError(
                    "Provider subprocess completed without producing output.",
                )
                setattr(
                    hard_error,
                    "provider_session_id",
                    observed_provider_session_id,
                )
                raise hard_error
            return ProviderInvocationResult(
                output=output,
                usage=usage,
                stdout_lines=tuple(stdout_lines),
                provider_session_id=observed_provider_session_id,
            )
        finally:
            if (
                request.prompt.cleanup_path
                and prompt_file_created
                and prompt_path is not None
            ):
                prompt_path.unlink(missing_ok=True)


@dataclasses.dataclass(slots=True)
class InMemoryProviderInvocationAdapter:
    prepared_invocations: list[
        ProviderInvocationResult
        | ProviderInvocationFailure
        | ProviderInvocationPreparedStream
    ] = dataclasses.field(default_factory=list)
    recorded_requests: list[ProviderInvocationRequest] = dataclasses.field(
        default_factory=list
    )

    def execute(
        self,
        request: ProviderInvocationRequest,
    ) -> ProviderInvocationResult | ProviderInvocationFailure:
        self.recorded_requests.append(request)
        if not self.prepared_invocations:
            raise AssertionError("No prepared provider invocation remains.")
        prepared = self.prepared_invocations.pop(0)
        if isinstance(prepared, ProviderInvocationFailure):
            return prepared
        if isinstance(prepared, ProviderInvocationPreparedStream):
            stdout_lines = list(prepared.stdout_lines)
            _consume_new_stdout_lines(request.output_hooks.reduce_output, stdout_lines)

            def _extracted_provider_session_id() -> str | None:
                if request.output_hooks.extract_provider_session_id is None:
                    return None
                return request.output_hooks.extract_provider_session_id(stdout_lines)

            def _active_provider_session_id() -> str | None:
                return (
                    _extracted_provider_session_id()
                    or prepared.provider_session_id
                    or request.provider_session_id
                )

            try:
                output, usage = request.output_hooks.reduce_output(stdout_lines)
            except Exception as exc:
                if isinstance(exc, (UsageLimitError, ProviderUnavailableError)):
                    return _provider_invocation_failure_from_error(
                        exc,
                        stdout_lines=tuple(stdout_lines),
                        provider_session_id=(
                            _extracted_provider_session_id()
                            or prepared.provider_session_id
                        ),
                    )
                raise
            return ProviderInvocationResult(
                output=output,
                usage=usage,
                stdout_lines=tuple(stdout_lines),
                provider_session_id=_active_provider_session_id(),
            )
        return prepared
