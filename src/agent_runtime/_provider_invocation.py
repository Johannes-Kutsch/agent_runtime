from __future__ import annotations

import dataclasses
import os
import queue
import shutil
import subprocess
import shlex
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Protocol

from .errors import (
    AgentCancelledError,
    AgentTimeoutError,
    HardAgentError,
    ProviderUnavailableError,
    ProviderUnavailableReason,
    UsageLimitError,
)
from .provider_usage import ProviderUsage
from .session import RunKind

_CANCEL_POLL_INTERVAL = 0.25


class _Cancellable(Protocol):
    @property
    def is_cancelled(self) -> bool: ...


ProviderOutputReducer = Callable[[list[str]], tuple[str, ProviderUsage | None]]
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
    extract_provider_session_id: ProviderSessionIdExtractor | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class ProviderInvocationRequest:
    worktree: Path
    environment: Mapping[str, str]
    prompt: ProviderInvocationPrompt
    run_kind: RunKind
    provider_session_id: str | None
    output_hooks: ProviderOutputReductionHooks
    command: str = ""
    argv: tuple[str, ...] = ()
    prefer_argv: bool = False
    timeout_seconds: int = 300
    token: _Cancellable | None = None

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
    provider_unavailable_reason: ProviderUnavailableReason | None = None


class ProviderInvocationAdapter(Protocol):
    def execute(
        self,
        request: ProviderInvocationRequest,
        argv_transform: (
            Callable[[tuple[str, ...], Path, dict[str, str]], tuple[str, ...]] | None
        ) = None,
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
        provider_unavailable_reason=error.reason,
    )


def _nonzero_exit_message(returncode: int, observed_lines: list[str]) -> str:
    message = f"Provider subprocess exited with exit code {returncode}."
    observed_output = "".join(observed_lines)
    if not observed_output.strip():
        return message
    return f"{message} Provider output:\n{observed_output}"


def _windows_process_base_env() -> dict[str, str]:
    if os.name != "nt":
        return {}
    return {
        key: os.environ[key]
        for key in ("PATH", "PATHEXT", "SystemRoot", "ComSpec", "WINDIR")
        if key in os.environ and os.environ[key]
    }


@dataclasses.dataclass(slots=True)
class _ProviderInvocationOutputFinalizer:
    request: ProviderInvocationRequest
    stdout_lines: list[str]
    success_fallback_provider_session_id: str | None = None
    failure_fallback_provider_session_id: str | None = None

    def consume_observed_lines(self) -> None:
        _consume_new_stdout_lines(
            self.request.output_hooks.reduce_output,
            self.stdout_lines,
        )

    def _extracted_provider_session_id(self) -> str | None:
        if self.request.output_hooks.extract_provider_session_id is None:
            return None
        return self.request.output_hooks.extract_provider_session_id(self.stdout_lines)

    def _success_provider_session_id(self) -> str | None:
        extracted_provider_session_id = self._extracted_provider_session_id()
        if extracted_provider_session_id is not None:
            return extracted_provider_session_id
        if self.success_fallback_provider_session_id is not None:
            return self.success_fallback_provider_session_id
        return self.request.provider_session_id

    def _failure_provider_session_id(self) -> str | None:
        extracted_provider_session_id = self._extracted_provider_session_id()
        if extracted_provider_session_id is not None:
            return extracted_provider_session_id
        return self.failure_fallback_provider_session_id

    def finalize(self) -> ProviderInvocationResult | ProviderInvocationFailure:
        try:
            output, usage = self.request.output_hooks.reduce_output(self.stdout_lines)
        except Exception as exc:
            if isinstance(exc, (UsageLimitError, ProviderUnavailableError)):
                return _provider_invocation_failure_from_error(
                    exc,
                    stdout_lines=tuple(self.stdout_lines),
                    provider_session_id=self._failure_provider_session_id(),
                )
            raise
        return ProviderInvocationResult(
            output=output,
            usage=usage,
            stdout_lines=tuple(self.stdout_lines),
            provider_session_id=self._success_provider_session_id(),
        )


def _finalize_provider_invocation_output(
    *,
    request: ProviderInvocationRequest,
    stdout_lines: list[str],
    success_fallback_provider_session_id: str | None = None,
    failure_fallback_provider_session_id: str | None = None,
    consume_observed_lines: bool = False,
) -> ProviderInvocationResult | ProviderInvocationFailure:
    output_finalizer = _ProviderInvocationOutputFinalizer(
        request=request,
        stdout_lines=stdout_lines,
        success_fallback_provider_session_id=success_fallback_provider_session_id,
        failure_fallback_provider_session_id=failure_fallback_provider_session_id,
    )
    if consume_observed_lines:
        output_finalizer.consume_observed_lines()
    return output_finalizer.finalize()


class ProductionProviderInvocationAdapter:
    def _windows_process_base_env(self) -> dict[str, str]:
        return _windows_process_base_env()

    def execute(
        self,
        request: ProviderInvocationRequest,
        argv_transform: (
            Callable[[tuple[str, ...], Path, dict[str, str]], tuple[str, ...]] | None
        ) = None,
    ) -> ProviderInvocationResult | ProviderInvocationFailure:
        requested_command = request.command
        requested_argv = request.argv
        use_shell = not (request.prefer_argv and request.argv)
        prompt_path = request.prompt.path
        environment = dict(request.environment)
        environment = {
            **_windows_process_base_env(),
            **environment,
        }
        if argv_transform is not None:
            requested_argv = tuple(
                argv_transform(
                    requested_argv,
                    request.worktree,
                    environment,
                )
            )
            requested_command = ""
            use_shell = False
        prompt_file_created = use_shell and prompt_path is not None
        if prompt_file_created and prompt_path is not None:
            prompt_path.write_text(request.prompt.content, encoding="utf-8")

        try:
            if use_shell:
                process = subprocess.Popen(
                    requested_command,
                    shell=True,
                    cwd=request.worktree,
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    encoding="utf-8",
                )
            else:
                # Resolve argv[0] against PATH/PATHEXT before spawning with
                # shell=False. On Windows, CreateProcess only appends .exe to
                # a bare name and ignores PATHEXT, so an npm-installed CLI
                # shim (e.g. claude.cmd) is never found from the bare name
                # "claude". shutil.which returns the resolvable shim path;
                # fall back to the original name when it cannot be resolved.
                resolved_argv = list(requested_argv)
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
                        encoding="utf-8",
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
                        encoding="utf-8",
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

            token = request.token
            if token is not None:
                idle_deadline: float | None = (
                    time.monotonic() + request.timeout_seconds
                    if request.timeout_seconds > 0
                    else None
                )
                poll_timeout: float | None = _CANCEL_POLL_INTERVAL
            else:
                idle_deadline = None
                poll_timeout = (
                    float(request.timeout_seconds)
                    if request.timeout_seconds > 0
                    else None
                )

            def _kill_process() -> None:
                if os.name == "nt":
                    subprocess.run(
                        [
                            "taskkill",
                            "/F",
                            "/T",
                            "/PID",
                            str(process.pid),
                        ],
                        capture_output=True,
                    )
                process.kill()
                process.wait()
                stdout_thread.join()
                stderr_thread.join()

            closed_streams: set[str] = set()
            while len(closed_streams) < 2:
                try:
                    if poll_timeout is not None:
                        source, line = output_queue.get(timeout=poll_timeout)
                    else:
                        source, line = output_queue.get()
                except queue.Empty as exc:
                    if token is not None and token.is_cancelled:
                        _kill_process()
                        cancel_error = AgentCancelledError()
                        setattr(
                            cancel_error,
                            "provider_session_id",
                            request.provider_session_id,
                        )
                        raise cancel_error
                    if token is not None:
                        if (
                            idle_deadline is not None
                            and time.monotonic() >= idle_deadline
                        ):
                            _kill_process()
                            timeout_error = ProviderInvocationTimedOutError(
                                "Provider subprocess exceeded the idle timeout.",
                            )
                            setattr(
                                timeout_error,
                                "provider_session_id",
                                request.provider_session_id,
                            )
                            raise timeout_error from exc
                        continue
                    _kill_process()
                    error = ProviderInvocationTimedOutError(
                        "Provider subprocess exceeded the idle timeout.",
                    )
                    setattr(error, "provider_session_id", request.provider_session_id)
                    raise error from exc

                if token is not None and idle_deadline is not None:
                    idle_deadline = time.monotonic() + request.timeout_seconds

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
            result = _finalize_provider_invocation_output(
                request=request,
                stdout_lines=stdout_lines,
                consume_observed_lines=False,
            )
            if isinstance(result, ProviderInvocationFailure):
                return result
            returncode = process.returncode
            observed_provider_session_id = result.provider_session_id
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
            if not result.output.strip():
                hard_error = HardAgentError(
                    "Provider subprocess completed without producing output.",
                )
                setattr(
                    hard_error,
                    "provider_session_id",
                    observed_provider_session_id,
                )
                raise hard_error
            return result
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
        argv_transform: (
            Callable[[tuple[str, ...], Path, dict[str, str]], tuple[str, ...]] | None
        ) = None,
    ) -> ProviderInvocationResult | ProviderInvocationFailure:
        if argv_transform is not None:
            request = dataclasses.replace(
                request,
                command="",
                argv=tuple(
                    argv_transform(
                        request.argv,
                        request.worktree,
                        {
                            **_windows_process_base_env(),
                            **dict(request.environment),
                        },
                    )
                ),
                prefer_argv=True,
            )
        self.recorded_requests.append(request)
        if not self.prepared_invocations:
            raise AssertionError("No prepared provider invocation remains.")
        prepared = self.prepared_invocations.pop(0)
        if isinstance(prepared, ProviderInvocationFailure):
            return prepared
        if isinstance(prepared, ProviderInvocationPreparedStream):
            stdout_lines = list(prepared.stdout_lines)
            return _finalize_provider_invocation_output(
                request=request,
                stdout_lines=stdout_lines,
                success_fallback_provider_session_id=prepared.provider_session_id,
                failure_fallback_provider_session_id=prepared.provider_session_id,
                consume_observed_lines=True,
            )
        return prepared
