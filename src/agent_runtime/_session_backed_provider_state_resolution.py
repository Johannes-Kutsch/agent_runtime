from __future__ import annotations

import shutil
from dataclasses import dataclass
import json
from pathlib import Path

from . import _builtin_runtime_client as _builtin_runtime_client_module
from ._runtime_lifecycle import Continuation
from .contracts import ToolAccess
from .errors import RuntimeConfigurationError
from .session import RunKind, provider_state_relpath


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


@dataclass(frozen=True)
class CodexStartSessionStateResolution:
    provider_state_dir: Path
    provider_state_dir_relpath: str | None


@dataclass(frozen=True)
class ClaudeStartSessionStateResolution:
    provider_state_dir: Path
    provider_state_dir_relpath: str | None


@dataclass(frozen=True)
class OpenCodeStartSessionStateResolution:
    provider_state_dir: Path
    provider_state_dir_relpath: str | None


@dataclass(frozen=True)
class CodexResumedSessionResolution:
    provider_state_dir: Path | None
    continuation_input_facts: ContinuationInputFacts


@dataclass(frozen=True)
class ClaudeNewSessionResolution:
    provider_state_dir: Path
    continuation_input_facts: ContinuationInputFacts


@dataclass(frozen=True)
class OpenCodeNewSessionResolution:
    provider_state_dir: Path
    continuation_input_facts: ContinuationInputFacts


@dataclass(frozen=True)
class OpenCodeResumedSessionResolution:
    provider_state_dir: Path
    continuation_input_facts: ContinuationInputFacts


@dataclass(frozen=True)
class ClaudeResumedSessionResolution:
    provider_state_dir: Path
    continuation_input_facts: ContinuationInputFacts


@dataclass(frozen=True)
class CodexNewSessionResolution:
    provider_state_dir: Path
    continuation_input_facts: ContinuationInputFacts


def _opencode_session_id_path(provider_state_dir: Path) -> Path:
    return (
        provider_state_dir
        / _builtin_runtime_client_module._OPENCODE_SESSION_ID_FILENAME
    )


def load_opencode_stored_session_id(provider_state_dir: Path | None) -> str | None:
    if provider_state_dir is None:
        return None
    session_id_path = _opencode_session_id_path(provider_state_dir)
    if not session_id_path.is_file():
        return None
    try:
        return _normalize_provider_session_id(
            session_id_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError):
        return None


def _opencode_is_resumable(provider_state_dir: Path) -> bool:
    return (provider_state_dir / "resume.jsonl").is_file() or _opencode_session_id_path(
        provider_state_dir
    ).is_file()


def _opencode_exact_transcript_match(
    *,
    saved_exact_transcript_match: bool,
    active_provider_session_id: str | None,
    stored_provider_session_id: str | None,
) -> bool:
    return (
        saved_exact_transcript_match
        and active_provider_session_id is not None
        and stored_provider_session_id == active_provider_session_id
    )


def persist_opencode_provider_session_id(
    provider_state_dir: Path,
    provider_session_id: str,
) -> None:
    _opencode_session_id_path(provider_state_dir).write_text(
        f"{provider_session_id}\n",
        encoding="utf-8",
    )


def _codex_rollout_paths(provider_state_dir: Path) -> tuple[Path, ...]:
    sessions_dir = provider_state_dir / "sessions"
    if not sessions_dir.is_dir():
        return ()
    return tuple(sessions_dir.rglob("rollout-*.jsonl"))


def resolve_codex_start_session_state(
    *,
    runtime_state_dir: Path,
    session_namespace: str,
    caller_owned_session_store: bool,
    host_auth_path: Path,
) -> CodexStartSessionStateResolution:
    provider_state_dir_relpath = provider_state_relpath(
        "implementer",
        "codex",
        session_namespace,
    )
    provider_state_dir = runtime_state_dir / provider_state_dir_relpath
    provider_state_dir.mkdir(parents=True, exist_ok=True)
    provider_auth_path = provider_state_dir / "auth.json"
    if not provider_auth_path.exists():
        shutil.copyfile(host_auth_path, provider_auth_path)
    return CodexStartSessionStateResolution(
        provider_state_dir=provider_state_dir,
        provider_state_dir_relpath=(
            provider_state_dir_relpath if caller_owned_session_store else None
        ),
    )


def resolve_claude_start_session_state(
    *,
    runtime_state_dir: Path,
    session_namespace: str,
    caller_owned_session_store: bool,
) -> ClaudeStartSessionStateResolution:
    provider_state_dir_relpath = provider_state_relpath(
        "implementer",
        "claude",
        session_namespace,
    )
    provider_state_dir = runtime_state_dir / provider_state_dir_relpath
    provider_state_dir.mkdir(parents=True, exist_ok=True)
    return ClaudeStartSessionStateResolution(
        provider_state_dir=provider_state_dir,
        provider_state_dir_relpath=(
            provider_state_dir_relpath if caller_owned_session_store else None
        ),
    )


def resolve_opencode_start_session_state(
    *,
    runtime_state_dir: Path,
    session_namespace: str,
    caller_owned_session_store: bool,
) -> OpenCodeStartSessionStateResolution:
    provider_state_dir_relpath = provider_state_relpath(
        "implementer",
        "opencode",
        session_namespace,
    )
    provider_state_dir = runtime_state_dir / provider_state_dir_relpath
    provider_state_dir.mkdir(parents=True, exist_ok=True)
    return OpenCodeStartSessionStateResolution(
        provider_state_dir=provider_state_dir,
        provider_state_dir_relpath=(
            provider_state_dir_relpath if caller_owned_session_store else None
        ),
    )


def resolve_codex_new_session_facts(
    *,
    runtime_state_dir: Path,
    session_namespace: str,
    caller_owned_session_store: bool,
    model: str,
    effort: str,
    host_auth_path: Path,
) -> CodexNewSessionResolution:
    start_session_state = resolve_codex_start_session_state(
        runtime_state_dir=runtime_state_dir,
        session_namespace=session_namespace,
        caller_owned_session_store=caller_owned_session_store,
        host_auth_path=host_auth_path,
    )
    rollout_paths = _codex_rollout_paths(start_session_state.provider_state_dir)
    recovered_provider_session_id = _recover_codex_rollout_session_id(
        start_session_state.provider_state_dir
    )
    if rollout_paths and recovered_provider_session_id is None:
        raise RuntimeConfigurationError(
            "Codex continuation is not recoverable from provider state."
        )
    continuation_input_facts = codex_continuation_input_facts(
        model=model,
        effort=effort,
        provider_state_dir=start_session_state.provider_state_dir,
        provider_state_dir_relpath=start_session_state.provider_state_dir_relpath,
        provider_session_id=recovered_provider_session_id,
        recovered_provider_session_id=recovered_provider_session_id is not None,
        run_kind=(
            RunKind.RESUME
            if recovered_provider_session_id is not None
            else RunKind.FRESH
        ),
    )
    return CodexNewSessionResolution(
        provider_state_dir=start_session_state.provider_state_dir,
        continuation_input_facts=continuation_input_facts,
    )


def resolve_opencode_new_session_facts(
    *,
    runtime_state_dir: Path,
    session_namespace: str,
    caller_owned_session_store: bool,
    model: str,
    effort: str,
) -> OpenCodeNewSessionResolution:
    start_session_state = resolve_opencode_start_session_state(
        runtime_state_dir=runtime_state_dir,
        session_namespace=session_namespace,
        caller_owned_session_store=caller_owned_session_store,
    )
    stored_provider_session_id = load_opencode_stored_session_id(
        start_session_state.provider_state_dir
    )
    active_provider_session_id = (
        stored_provider_session_id
        or _builtin_runtime_client_module._new_provider_session_id()
    )
    run_kind = (
        RunKind.RESUME
        if _opencode_is_resumable(start_session_state.provider_state_dir)
        else RunKind.FRESH
    )
    return OpenCodeNewSessionResolution(
        provider_state_dir=start_session_state.provider_state_dir,
        continuation_input_facts=opencode_continuation_input_facts(
            model=model,
            effort=effort,
            provider_state_dir=start_session_state.provider_state_dir,
            provider_state_dir_relpath=start_session_state.provider_state_dir_relpath,
            provider_session_id=active_provider_session_id,
            run_kind=run_kind,
            exact_transcript_match=_opencode_exact_transcript_match(
                saved_exact_transcript_match=True,
                active_provider_session_id=active_provider_session_id,
                stored_provider_session_id=stored_provider_session_id,
            ),
        ),
    )


def _claude_is_resumable(state_dir: Path) -> bool:
    return state_dir.is_dir() and any(path.is_file() for path in state_dir.rglob("*"))


def resolve_claude_new_session_facts(
    *,
    runtime_state_dir: Path,
    session_namespace: str,
    caller_owned_session_store: bool,
    model: str,
    effort: str,
) -> ClaudeNewSessionResolution:
    start_session_state = resolve_claude_start_session_state(
        runtime_state_dir=runtime_state_dir,
        session_namespace=session_namespace,
        caller_owned_session_store=caller_owned_session_store,
    )
    run_kind = (
        RunKind.RESUME
        if _claude_is_resumable(start_session_state.provider_state_dir)
        else RunKind.FRESH
    )
    continuation_input_facts = claude_continuation_input_facts(
        model=model,
        effort=effort,
        provider_state_dir=start_session_state.provider_state_dir,
        provider_state_dir_relpath=start_session_state.provider_state_dir_relpath,
        provider_session_id=_builtin_runtime_client_module._new_provider_session_id(),
        run_kind=run_kind,
    )
    return ClaudeNewSessionResolution(
        provider_state_dir=start_session_state.provider_state_dir,
        continuation_input_facts=continuation_input_facts,
    )


def resolve_claude_resumed_session_facts(
    *,
    runtime_state_dir: Path,
    provider_state_dir_relpath: str | None,
    model: str,
    effort: str,
    provider_session_id: str | None,
) -> ClaudeResumedSessionResolution:
    if provider_state_dir_relpath is None:
        provider_state_dir = runtime_state_dir
        run_kind = RunKind.RESUME
    else:
        provider_state_dir = runtime_state_dir / provider_state_dir_relpath
        provider_state_dir.mkdir(parents=True, exist_ok=True)
        run_kind = (
            RunKind.RESUME
            if _claude_is_resumable(provider_state_dir)
            else RunKind.FRESH
        )
    active_provider_session_id = _normalize_provider_session_id(provider_session_id)
    if active_provider_session_id is None:
        active_provider_session_id = (
            _builtin_runtime_client_module._new_provider_session_id()
        )
    return ClaudeResumedSessionResolution(
        provider_state_dir=provider_state_dir,
        continuation_input_facts=claude_continuation_input_facts(
            model=model,
            effort=effort,
            provider_state_dir=provider_state_dir,
            provider_state_dir_relpath=provider_state_dir_relpath,
            provider_session_id=active_provider_session_id,
            run_kind=run_kind,
        ),
    )


def resolve_opencode_resumed_session_facts(
    *,
    runtime_state_dir: Path,
    session_namespace: str,
    continuation: Continuation,
    model: str,
    effort: str,
) -> OpenCodeResumedSessionResolution:
    continuation_facts = continuation.session_backed_facts
    provider_state_dir_relpath = continuation_facts.provider_state_dir_relpath
    if provider_state_dir_relpath is None:
        provider_state_dir_relpath = provider_state_relpath(
            "implementer",
            "opencode",
            session_namespace,
        )
    provider_state_dir = runtime_state_dir / provider_state_dir_relpath
    provider_state_dir.mkdir(parents=True, exist_ok=True)

    stored_provider_session_id = load_opencode_stored_session_id(provider_state_dir)
    active_provider_session_id = _normalize_provider_session_id(
        continuation_facts.provider_session_id
    )
    if active_provider_session_id is None:
        active_provider_session_id = stored_provider_session_id
    if active_provider_session_id is None:
        active_provider_session_id = (
            _builtin_runtime_client_module._new_provider_session_id()
        )

    return OpenCodeResumedSessionResolution(
        provider_state_dir=provider_state_dir,
        continuation_input_facts=opencode_continuation_input_facts(
            model=model,
            effort=effort,
            provider_state_dir=provider_state_dir,
            provider_state_dir_relpath=provider_state_dir_relpath,
            provider_session_id=active_provider_session_id,
            run_kind=RunKind.RESUME,
            exact_transcript_match=_opencode_exact_transcript_match(
                saved_exact_transcript_match=bool(
                    continuation_facts.exact_transcript_match
                ),
                active_provider_session_id=active_provider_session_id,
                stored_provider_session_id=stored_provider_session_id,
            ),
        ),
    )


def _seed_codex_auth(provider_state_dir: Path, host_auth_path: Path) -> None:
    provider_auth_path = provider_state_dir / "auth.json"
    if not provider_auth_path.exists():
        shutil.copyfile(host_auth_path, provider_auth_path)


def _read_codex_rollout_session_ids(rollout_path: Path) -> set[str]:
    session_ids: set[str] = set()
    if not rollout_path.is_file():
        return session_ids
    try:
        for line in rollout_path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict) or event.get("type") != "session_meta":
                continue
            payload = event.get("payload")
            if not isinstance(payload, dict):
                continue
            session_id = payload.get("id")
            if isinstance(session_id, str):
                stripped = session_id.strip()
                if stripped:
                    session_ids.add(stripped)
    except (OSError, UnicodeDecodeError):
        return set()
    return session_ids


def _recover_codex_rollout_session_id(provider_state_dir: Path) -> str | None:
    session_ids: set[str] = set()
    for rollout_path in _codex_rollout_paths(provider_state_dir):
        session_ids.update(_read_codex_rollout_session_ids(rollout_path))
        if len(session_ids) > 1:
            return None
    if len(session_ids) != 1:
        return None
    return next(iter(session_ids))


def _normalize_provider_session_id(provider_session_id: str | None) -> str | None:
    if provider_session_id is None:
        return None
    return provider_session_id.strip() or None


def resolve_codex_resumed_session_facts(
    *,
    runtime_state_dir: Path | None,
    provider_state_dir_relpath: str | None,
    model: str,
    effort: str,
    provider_session_id: str | None,
    host_auth_path: Path,
) -> CodexResumedSessionResolution:
    normalized_provider_session_id = _normalize_provider_session_id(provider_session_id)
    provider_state_dir: Path | None = None
    recovered_provider_session_id: str | None = None
    if runtime_state_dir is not None and provider_state_dir_relpath:
        provider_state_dir = runtime_state_dir / provider_state_dir_relpath
        provider_state_dir.mkdir(parents=True, exist_ok=True)
        _seed_codex_auth(provider_state_dir, host_auth_path)
        recovered_provider_session_id = _recover_codex_rollout_session_id(
            provider_state_dir
        )
        if recovered_provider_session_id is None:
            raise RuntimeConfigurationError(
                "Codex continuation is not recoverable from provider state."
            )
    active_provider_session_id = normalized_provider_session_id
    if active_provider_session_id is None:
        active_provider_session_id = recovered_provider_session_id
    if active_provider_session_id is None:
        raise RuntimeConfigurationError(
            "Codex continuation is missing `provider_session_id`."
        )
    return CodexResumedSessionResolution(
        provider_state_dir=provider_state_dir,
        continuation_input_facts=codex_continuation_input_facts(
            model=model,
            effort=effort,
            provider_state_dir=provider_state_dir or host_auth_path.parent,
            provider_state_dir_relpath=provider_state_dir_relpath,
            provider_session_id=active_provider_session_id,
            recovered_provider_session_id=normalized_provider_session_id is None,
            run_kind=RunKind.RESUME,
        ),
    )


def codex_continuation_input_facts(
    *,
    model: str,
    effort: str,
    provider_state_dir: Path,
    provider_state_dir_relpath: str | None,
    provider_session_id: str | None,
    recovered_provider_session_id: bool = False,
    run_kind: RunKind,
) -> ContinuationInputFacts:
    return ContinuationInputFacts(
        provider_identity=ProviderIdentity(
            service="codex",
            model=model,
            effort=effort,
        ),
        provider_state_directory=ProviderStateDirectory(path=provider_state_dir),
        provider_state_relpath=_provider_state_relpath(provider_state_dir_relpath),
        provider_session_id=_provider_session_id(
            provider_session_id,
            recovered=recovered_provider_session_id,
        ),
        run_kind=run_kind,
        exact_transcript_match=ExactTranscriptMatch(value=False),
    )


def claude_continuation_input_facts(
    *,
    model: str,
    effort: str,
    provider_state_dir: Path,
    provider_state_dir_relpath: str | None,
    provider_session_id: str | None,
    run_kind: RunKind,
) -> ContinuationInputFacts:
    return ContinuationInputFacts(
        provider_identity=ProviderIdentity(
            service="claude",
            model=model,
            effort=effort,
        ),
        provider_state_directory=ProviderStateDirectory(path=provider_state_dir),
        provider_state_relpath=_provider_state_relpath(provider_state_dir_relpath),
        provider_session_id=_provider_session_id(provider_session_id, recovered=False),
        run_kind=run_kind,
        exact_transcript_match=ExactTranscriptMatch(value=False),
    )


def opencode_continuation_input_facts(
    *,
    model: str,
    effort: str,
    provider_state_dir: Path,
    provider_state_dir_relpath: str | None,
    provider_session_id: str | None,
    run_kind: RunKind,
    exact_transcript_match: bool,
) -> ContinuationInputFacts:
    return ContinuationInputFacts(
        provider_identity=ProviderIdentity(
            service="opencode",
            model=model,
            effort=effort,
        ),
        provider_state_directory=ProviderStateDirectory(path=provider_state_dir),
        provider_state_relpath=_provider_state_relpath(provider_state_dir_relpath),
        provider_session_id=_provider_session_id(provider_session_id, recovered=False),
        run_kind=run_kind,
        exact_transcript_match=ExactTranscriptMatch(value=exact_transcript_match),
    )


def _provider_state_relpath(
    provider_state_dir_relpath: str | None,
) -> ProviderStateRelpath | None:
    if provider_state_dir_relpath is None:
        return None
    return ProviderStateRelpath(value=provider_state_dir_relpath)


def _provider_session_id(
    provider_session_id: str | None,
    *,
    recovered: bool,
) -> PreparedOrRecoveredProviderSessionId | None:
    if provider_session_id is None:
        return None
    return PreparedOrRecoveredProviderSessionId(
        value=provider_session_id,
        recovered=recovered,
    )


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
