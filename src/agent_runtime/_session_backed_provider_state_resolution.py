from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ._runtime_lifecycle import Continuation
from .contracts import ToolAccess
from .session import RunKind


@dataclass(frozen=True)
class ProviderIdentity:
    service: str
    model: str
    effort: str


@dataclass(frozen=True)
class ProviderStateDirectory:
    path: Path


@dataclass(frozen=True)
class ProviderStateRelpath:
    value: str


@dataclass(frozen=True)
class PreparedOrRecoveredProviderSessionId:
    value: str
    recovered: bool


@dataclass(frozen=True)
class ExactTranscriptMatch:
    value: bool


@dataclass(frozen=True)
class ContinuationInputFacts:
    provider_identity: ProviderIdentity
    provider_state_directory: ProviderStateDirectory
    provider_state_relpath: ProviderStateRelpath | None
    provider_session_id: PreparedOrRecoveredProviderSessionId | None
    run_kind: RunKind
    exact_transcript_match: ExactTranscriptMatch | None


def build_session_backed_continuation(
    continuation_input_facts: ContinuationInputFacts,
    *,
    tool_access: ToolAccess,
    provider_session_id: str | None = None,
) -> Continuation:
    service = continuation_input_facts.provider_identity.service
    continuation_run_kind: str | None = None
    exact_transcript_match: bool | None = None
    if service in {"claude", "codex"}:
        continuation_run_kind = RunKind.RESUME.value
        exact_transcript_match = False
    elif service == "opencode":
        exact_transcript_match = (
            continuation_input_facts.exact_transcript_match.value
            if continuation_input_facts.exact_transcript_match is not None
            else None
        )

    active_provider_session_id = provider_session_id
    if active_provider_session_id is None:
        active_provider_session = continuation_input_facts.provider_session_id
        active_provider_session_id = (
            active_provider_session.value
            if active_provider_session is not None
            else None
        )

    provider_state_relpath = continuation_input_facts.provider_state_relpath
    provider_state_dir_relpath = (
        provider_state_relpath.value if provider_state_relpath is not None else None
    )

    return Continuation.for_session_backed_provider(
        selected_service=service,
        selected_model=continuation_input_facts.provider_identity.model,
        selected_effort=continuation_input_facts.provider_identity.effort,
        tool_access=tool_access,
        provider_session_id=active_provider_session_id,
        provider_state_dir_relpath=provider_state_dir_relpath,
        exact_transcript_match=exact_transcript_match,
        run_kind=continuation_run_kind,
    )
