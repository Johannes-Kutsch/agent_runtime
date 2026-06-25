"""Private runtime seam for one planned live probe case."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from traceback import format_exc
from typing import Any, Awaitable, Callable, Protocol, TextIO

from agent_runtime.runtime import (
    Continuation,
    EphemeralRunRequest,
    NewSessionRunRequest,
    ProviderAuth,
    ResumedSessionRunRequest,
    RuntimeClient as _PublicRuntimeClient,
    ToolPolicy,
)
from agent_runtime.errors import AgentCredentialFailureError
from agent_runtime.errors import ProviderUnavailableReason


class _LiveProbeOutput(Protocol):
    def line(self, text: str) -> None: ...


class _PlannedProbeCase(Protocol):
    @property
    def service(self) -> str: ...

    @property
    def mode(self) -> str: ...

    @property
    def tool_policy(self) -> str: ...

    @property
    def provider_selection(self) -> Any: ...


class _RuntimeInvocationPort(Protocol):
    def run_ephemeral(self, request: EphemeralRunRequest) -> Awaitable[Any]: ...

    def run_new_session(self, request: NewSessionRunRequest) -> Awaitable[Any]: ...

    def run_resumed_session(
        self, request: ResumedSessionRunRequest
    ) -> Awaitable[Any]: ...


@dataclass(frozen=True)
class ProbeCaseRunRequest:
    case: _PlannedProbeCase
    case_dir: Path
    invocation_dir: Path
    prompt: str
    timeout_seconds: int
    continuation: Continuation | None
    output: _LiveProbeOutput
    resumed_session_invocation_dir: Path | None = None


@dataclass(frozen=True)
class ProbeCaseRunResult:
    category: str
    kind: str | None
    selected: dict[str, Any] | None
    output: str | None
    usage: dict[str, Any] | None
    continuation: Continuation | None
    traceback: str | None
    next_resumed_session_continuation: Continuation | None = None
    next_resumed_session_invocation_dir: Path | None = None


class _RuntimeInvocationClient:
    """Private port for invoking runtime lifecycle methods."""

    def __init__(self, runtime_client: _PublicRuntimeClient | None = None) -> None:
        self._runtime_client = runtime_client or _PublicRuntimeClient()

    def run_ephemeral(self, request: EphemeralRunRequest) -> Awaitable[Any]:
        return self._runtime_client.run_ephemeral(request)

    def run_new_session(self, request: NewSessionRunRequest) -> Awaitable[Any]:
        return self._runtime_client.run_new_session(request)

    def run_resumed_session(self, request: ResumedSessionRunRequest) -> Awaitable[Any]:
        return self._runtime_client.run_resumed_session(request)


class InMemoryRuntimeInvocationAdapter:
    """In-memory runtime invocation adapter for deterministic probe tests."""

    def __init__(
        self,
        *,
        prepared_outcomes: list[Any] | None = None,
        record_handler: Callable[[str, Any], Any] | None = None,
    ) -> None:
        self.prepared_outcomes = prepared_outcomes or []
        self.record_handler = record_handler
        self.recorded_requests: list[tuple[str, Any]] = []

    def _record(self, mode: str, request: Any) -> None:
        self.recorded_requests.append((mode, request))

    def _next(self, mode: str, request: Any) -> Any:
        self._record(mode, request)
        if self.record_handler is not None:
            return self.record_handler(mode, request)
        if self.prepared_outcomes:
            return self.prepared_outcomes.pop(0)
        raise AssertionError("No prepared outcome for in-memory runtime invocation")

    async def run_ephemeral(self, request: EphemeralRunRequest) -> Any:
        return self._next("run_ephemeral", request)

    async def run_new_session(self, request: NewSessionRunRequest) -> Any:
        return self._next("run_new_session", request)

    async def run_resumed_session(self, request: ResumedSessionRunRequest) -> Any:
        return self._next("run_resumed_session", request)


RuntimeClient = _RuntimeInvocationClient
LIVE_FEED_FILENAME = "live_feed.json"
RESULT_FILENAME = "result.json"
_DISPLAYED_EVENT_TYPES = ("agent_message", "agent_tool_call")


class _LiveProbeFeedWriter:
    def __init__(self, path: Path, output: _LiveProbeOutput) -> None:
        self._sink: TextIO = path.open("w", encoding="utf-8")
        self._output = output

    def append(self, event: Any) -> None:
        record = {
            "type": getattr(event, "type", ""),
            "display_message": getattr(event, "display_message", ""),
            "raw_provider_output": getattr(event, "raw_provider_output", ""),
        }
        self._sink.write(json.dumps(record) + "\n")
        self._sink.flush()
        if record["type"] in _DISPLAYED_EVENT_TYPES:
            self._output.line(f"  {record['display_message']}")

    def close(self) -> None:
        self._sink.close()


def _outcome_category(runtime_outcome: Any) -> str:
    """Map a ``RuntimeOutcome`` to its probe verdict category."""

    kind = getattr(runtime_outcome, "kind", None)
    if type(kind).__name__ == "ProviderUnavailable":
        reason = getattr(kind, "reason", None)
        if reason is ProviderUnavailableReason.SERVICE_NOT_AVAILABLE:
            return "no_service_available"
        if reason is ProviderUnavailableReason.TRANSIENT_API_ERROR:
            return "retryable_failure"
    _outcome_map: dict[str, str] = {
        "Completed": "success",
        "UsageLimited": "usage_limited",
        "TimedOut": "timed_out",
        "Cancelled": "cancelled",
    }
    return _outcome_map.get(type(kind).__name__, "error")


def _resolve_runtime_outcome(awaitable: Any) -> Any:
    return asyncio.run(awaitable)


def _selected_payload(selected: Any) -> dict[str, Any] | None:
    if selected is None:
        return None
    return {
        "service": getattr(selected, "service", None),
        "model": getattr(selected, "model", None),
        "effort": getattr(selected, "effort", None),
    }


def _usage_payload(usage: Any) -> dict[str, Any] | None:
    if usage is None:
        return None
    return {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", None),
        "cache_creation_input_tokens": getattr(
            usage, "cache_creation_input_tokens", None
        ),
        "cost_usd": getattr(usage, "cost_usd", None),
        "duration_seconds": getattr(usage, "duration_seconds", None),
    }


def _continuation_from_outcome(outcome: Any) -> Continuation | None:
    result = getattr(outcome, "result", None)
    return getattr(result, "continuation", None)


def _invocation_dir_for_case(request: ProbeCaseRunRequest) -> Path:
    if request.case.mode == "resumed_session":
        return (
            request.resumed_session_invocation_dir
            if request.resumed_session_invocation_dir is not None
            else request.invocation_dir
        )
    return request.invocation_dir


def _result_payload(
    case: _PlannedProbeCase,
    outcome: ProbeCaseRunResult,
) -> dict[str, Any]:
    return {
        "service": case.service,
        "mode": case.mode,
        "tool_policy": case.tool_policy,
        "category": outcome.category,
        "kind": outcome.kind,
        "selected": outcome.selected,
        "output": outcome.output,
        "usage": outcome.usage,
        "continuation": (
            outcome.continuation.serialized
            if outcome.continuation is not None
            else None
        ),
        "traceback": outcome.traceback,
    }


def _write_result_json(
    case_dir: Path, case: _PlannedProbeCase, outcome: ProbeCaseRunResult, output: Any
) -> None:
    payload = _result_payload(case, outcome)
    try:
        (case_dir / RESULT_FILENAME).write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
    except Exception as exc:  # pragma: no cover - best effort diagnostics only
        output.line(f"  (failed to write {RESULT_FILENAME}: {exc})")


def run_case(
    request: ProbeCaseRunRequest,
    *,
    runtime_client_factory: Callable[[], _RuntimeInvocationPort] = RuntimeClient,
) -> ProbeCaseRunResult:
    request.case_dir.mkdir(parents=True, exist_ok=True)
    case_invocation_dir = _invocation_dir_for_case(request)
    case_invocation_dir.mkdir(parents=True, exist_ok=True)

    feed_path = request.case_dir / LIVE_FEED_FILENAME
    feed_writer = _LiveProbeFeedWriter(feed_path, request.output)
    outcome: Any | None = None
    category = "error"
    traceback: str | None = None

    selection = request.case.provider_selection
    auth = getattr(selection, "auth", None) or ProviderAuth()
    tool_policy = ToolPolicy[request.case.tool_policy]

    try:
        client = runtime_client_factory()
        if request.case.mode == "ephemeral":
            outcome = _resolve_runtime_outcome(
                client.run_ephemeral(
                    EphemeralRunRequest(
                        prompt=request.prompt,
                        invocation_dir=case_invocation_dir,
                        provider_selection=selection,
                        tool_policy=tool_policy,
                        timeout_seconds=request.timeout_seconds,
                        on_live_output=feed_writer.append,
                    )
                )
            )
        elif request.case.mode == "new_session":
            outcome = _resolve_runtime_outcome(
                client.run_new_session(
                    NewSessionRunRequest(
                        prompt=request.prompt,
                        invocation_dir=case_invocation_dir,
                        provider_selection=selection,
                        tool_policy=tool_policy,
                        timeout_seconds=request.timeout_seconds,
                        on_live_output=feed_writer.append,
                    )
                )
            )
        elif request.case.mode == "resumed_session":
            if request.continuation is None:
                raise RuntimeError(
                    "resumed_session requires a continuation from new_session; "
                    "the new_session case did not produce one"
                )
            outcome = _resolve_runtime_outcome(
                client.run_resumed_session(
                    ResumedSessionRunRequest(
                        prompt=request.prompt,
                        invocation_dir=case_invocation_dir,
                        continuation=request.continuation,
                        provider_auth=auth,
                        timeout_seconds=request.timeout_seconds,
                        on_live_output=feed_writer.append,
                    )
                )
            )
        else:
            raise ValueError(f"unsupported probe mode: {request.case.mode!r}")
        category = _outcome_category(outcome)
    except AgentCredentialFailureError:
        category = "wrong_credentials"
        traceback = format_exc()
    except Exception:
        traceback = format_exc()
    finally:
        feed_writer.close()

    if outcome is None:
        result = ProbeCaseRunResult(
            category=category,
            kind=None,
            selected=None,
            output=None,
            usage=None,
            continuation=None,
            next_resumed_session_continuation=None,
            next_resumed_session_invocation_dir=None,
            traceback=traceback,
        )
        _write_result_json(request.case_dir, request.case, result, request.output)
        return result

    outcome_result = getattr(outcome, "result", None)
    outcome_continuation = _continuation_from_outcome(outcome)
    result = ProbeCaseRunResult(
        category=category,
        kind=type(getattr(outcome, "kind", None)).__name__
        if getattr(outcome, "kind", None) is not None
        else None,
        selected=_selected_payload(getattr(outcome_result, "selected", None)),
        output=getattr(outcome_result, "output", None),
        usage=_usage_payload(getattr(outcome_result, "usage", None)),
        continuation=outcome_continuation,
        next_resumed_session_continuation=(
            outcome_continuation if request.case.mode == "new_session" else None
        ),
        next_resumed_session_invocation_dir=(
            case_invocation_dir if request.case.mode == "new_session" else None
        ),
        traceback=None,
    )
    _write_result_json(request.case_dir, request.case, result, request.output)
    return result
