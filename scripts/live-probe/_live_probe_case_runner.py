"""Private runtime seam for one planned live probe case."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from traceback import format_exc
from typing import Any, Awaitable, Callable, Protocol

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


@dataclass(frozen=True)
class ProbeCaseRunResult:
    category: str
    kind: str | None
    selected: dict[str, Any] | None
    output: str | None
    usage: dict[str, Any] | None
    continuation: Continuation | None
    traceback: str | None


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
_DISPLAYED_EVENT_TYPES = ("agent_message", "agent_tool_call")


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


def run_case(
    request: ProbeCaseRunRequest,
    *,
    outcome_category: Callable[[Any], str],
    runtime_client_factory: Callable[[], _RuntimeInvocationPort] = RuntimeClient,
) -> ProbeCaseRunResult:
    request.case_dir.mkdir(parents=True, exist_ok=True)
    request.invocation_dir.mkdir(parents=True, exist_ok=True)

    feed_path = request.case_dir / LIVE_FEED_FILENAME
    feed_sink = feed_path.open("w", encoding="utf-8")
    outcome: Any | None = None
    category = "error"
    traceback: str | None = None

    def _on_live_output(event: Any) -> None:
        record = {
            "type": getattr(event, "type", ""),
            "display_message": getattr(event, "display_message", ""),
            "raw_provider_output": getattr(event, "raw_provider_output", ""),
        }
        feed_sink.write(json.dumps(record) + "\n")
        feed_sink.flush()
        if record["type"] in _DISPLAYED_EVENT_TYPES:
            request.output.line(f"  {record['display_message']}")

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
                        invocation_dir=request.invocation_dir,
                        provider_selection=selection,
                        tool_policy=tool_policy,
                        timeout_seconds=request.timeout_seconds,
                        on_live_output=_on_live_output,
                    )
                )
            )
        elif request.case.mode == "new_session":
            outcome = _resolve_runtime_outcome(
                client.run_new_session(
                    NewSessionRunRequest(
                        prompt=request.prompt,
                        invocation_dir=request.invocation_dir,
                        provider_selection=selection,
                        tool_policy=tool_policy,
                        timeout_seconds=request.timeout_seconds,
                        on_live_output=_on_live_output,
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
                        invocation_dir=request.invocation_dir,
                        continuation=request.continuation,
                        provider_auth=auth,
                        timeout_seconds=request.timeout_seconds,
                        on_live_output=_on_live_output,
                    )
                )
            )
        else:
            raise ValueError(f"unsupported probe mode: {request.case.mode!r}")
        category = outcome_category(outcome)
    except AgentCredentialFailureError:
        category = "wrong_credentials"
        traceback = format_exc()
    except Exception:
        traceback = format_exc()
    finally:
        feed_sink.close()

    if outcome is None:
        return ProbeCaseRunResult(
            category=category,
            kind=None,
            selected=None,
            output=None,
            usage=None,
            continuation=None,
            traceback=traceback,
        )

    result = getattr(outcome, "result", None)
    return ProbeCaseRunResult(
        category=category,
        kind=type(getattr(outcome, "kind", None)).__name__
        if getattr(outcome, "kind", None) is not None
        else None,
        selected=_selected_payload(getattr(result, "selected", None)),
        output=getattr(result, "output", None),
        usage=_usage_payload(getattr(result, "usage", None)),
        continuation=_continuation_from_outcome(outcome),
        traceback=None,
    )
