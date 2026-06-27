from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
