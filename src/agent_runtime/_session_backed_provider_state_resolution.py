from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass
import json
from pathlib import Path

from . import _builtin_runtime_client as _builtin_runtime_client_module
from ._runtime_lifecycle import Continuation
from .contracts import ToolAccess
from .errors import ContinuationUnrecoverableError, RuntimeConfigurationError
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


@dataclass(frozen=True)
class _StartSessionState:
    provider_state_dir: Path
    provider_state_dir_relpath: str | None


@dataclass(frozen=True)
class _NewSessionResolution:
    provider_state_dir: Path
    continuation_input_facts: ContinuationInputFacts


@dataclass(frozen=True)
class _ResumedSessionResolution:
    provider_state_dir: Path | None
    continuation_input_facts: ContinuationInputFacts


# Backward-compatible module-level aliases so call sites in other modules and
# tests continue to reference the old per-service names without edits.
ClaudeNewSessionResolution = _NewSessionResolution
CodexNewSessionResolution = _NewSessionResolution
OpenCodeNewSessionResolution = _NewSessionResolution
ClaudeResumedSessionResolution = _ResumedSessionResolution
CodexResumedSessionResolution = _ResumedSessionResolution
OpenCodeResumedSessionResolution = _ResumedSessionResolution


@dataclass(frozen=True)
class _ServiceStateBundle:
    service: str
    start_session_hook: Callable[[Path, Path | None], None]
    make_exact_transcript_match: Callable[[bool], ExactTranscriptMatch]
    make_recovered: Callable[[bool], bool]
    # New-session resolution
    recover_new_session_id: Callable[[Path], str | None]
    has_prior_sessions: Callable[[Path], bool]
    probe_new_session_resumable: Callable[[Path], bool]
    generate_new_session_id: Callable[[], str | None]
    compute_new_session_exact_transcript_match: Callable[[str | None, str | None], bool]
    # Resumed-session resolution
    resumed_relpath_none_uses_root: bool
    assert_resumed_and_recover_id: Callable[[Path | None, Path | None], str | None]
    generate_resumed_session_id_or_raise: Callable[[], str]
    compute_resumed_exact_transcript_match: Callable[
        [str | None, str | None, bool], bool
    ]
    # Continuation building
    continuation_run_kind: str | None
    compute_continuation_exact_transcript_match: Callable[
        [ExactTranscriptMatch | None], bool | None
    ]


# ── Utility ───────────────────────────────────────────────────────────────────


def _normalize_provider_session_id(provider_session_id: str | None) -> str | None:
    if provider_session_id is None:
        return None
    return provider_session_id.strip() or None


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


# ── OpenCode helpers ──────────────────────────────────────────────────────────


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


def resolve_opencode_active_session_facts(
    continuation_input_facts: ContinuationInputFacts,
    *,
    provider_session_id: str | None,
) -> ContinuationInputFacts:
    active_provider_session_id = _normalize_provider_session_id(provider_session_id)
    if active_provider_session_id is None:
        return continuation_input_facts

    provider_state_dir = continuation_input_facts.provider_state_directory.path
    persist_opencode_provider_session_id(
        provider_state_dir,
        active_provider_session_id,
    )

    prepared_provider_session = continuation_input_facts.provider_session_id
    exact_transcript_match = bool(
        continuation_input_facts.exact_transcript_match is not None
        and continuation_input_facts.exact_transcript_match.value
        and prepared_provider_session is not None
        and prepared_provider_session.value == active_provider_session_id
    )
    provider_state_relpath = continuation_input_facts.provider_state_relpath

    return opencode_continuation_input_facts(
        model=continuation_input_facts.provider_identity.model,
        effort=continuation_input_facts.provider_identity.effort,
        provider_state_dir=provider_state_dir,
        provider_state_dir_relpath=(
            provider_state_relpath.value if provider_state_relpath is not None else None
        ),
        provider_session_id=active_provider_session_id,
        run_kind=continuation_input_facts.run_kind,
        exact_transcript_match=exact_transcript_match,
    )


# ── Codex helpers ─────────────────────────────────────────────────────────────


def _codex_rollout_paths(provider_state_dir: Path) -> tuple[Path, ...]:
    sessions_dir = provider_state_dir / "sessions"
    if not sessions_dir.is_dir():
        return ()
    return tuple(sessions_dir.rglob("rollout-*.jsonl"))


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


def _seed_codex_auth(provider_state_dir: Path, host_auth_path: Path) -> None:
    provider_auth_path = provider_state_dir / "auth.json"
    if not provider_auth_path.exists():
        shutil.copyfile(host_auth_path, provider_auth_path)


# ── Claude helpers ────────────────────────────────────────────────────────────


def _claude_is_resumable(state_dir: Path) -> bool:
    return state_dir.is_dir() and any(path.is_file() for path in state_dir.rglob("*"))


# ── Start-session hooks ───────────────────────────────────────────────────────


def _noop_start_session_hook(
    provider_state_dir: Path, host_auth_path: Path | None
) -> None:
    pass


def _codex_start_session_hook(
    provider_state_dir: Path, host_auth_path: Path | None
) -> None:
    if host_auth_path is not None:
        _seed_codex_auth(provider_state_dir, host_auth_path)


# ── Bundle field helpers: new-session ─────────────────────────────────────────


def _no_prior_sessions(provider_state_dir: Path) -> bool:
    return False


def _codex_has_prior_sessions(provider_state_dir: Path) -> bool:
    return bool(_codex_rollout_paths(provider_state_dir))


def _codex_probe_new_session_resumable(provider_state_dir: Path) -> bool:
    return False


def _compute_opencode_new_session_exact_transcript_match(
    active: str | None, recovered: str | None
) -> bool:
    return _opencode_exact_transcript_match(
        saved_exact_transcript_match=True,
        active_provider_session_id=active,
        stored_provider_session_id=recovered,
    )


# ── Bundle field helpers: resumed-session ─────────────────────────────────────


def _claude_assert_resumed_and_recover_id(
    provider_state_dir: Path | None, host_auth_path: Path | None
) -> str | None:
    if provider_state_dir is not None and not _claude_is_resumable(provider_state_dir):
        raise ContinuationUnrecoverableError(
            "Claude continuation is not recoverable from provider state.",
            service_name="claude",
        )
    return None


def _codex_assert_resumed_and_recover_id(
    provider_state_dir: Path | None, host_auth_path: Path | None
) -> str | None:
    if provider_state_dir is None:
        return None
    if host_auth_path is not None:
        _seed_codex_auth(provider_state_dir, host_auth_path)
    recovered = _recover_codex_rollout_session_id(provider_state_dir)
    if recovered is None:
        raise ContinuationUnrecoverableError(
            "Codex continuation is not recoverable from provider state.",
            service_name="codex",
        )
    return recovered


def _opencode_assert_resumed_and_recover_id(
    provider_state_dir: Path | None, host_auth_path: Path | None
) -> str | None:
    if provider_state_dir is not None and not _opencode_is_resumable(
        provider_state_dir
    ):
        raise ContinuationUnrecoverableError(
            "OpenCode continuation is not recoverable from provider state.",
            service_name="opencode",
        )
    return load_opencode_stored_session_id(provider_state_dir)


def _codex_generate_resumed_session_id_or_raise() -> str:
    raise RuntimeConfigurationError(
        "Codex continuation is missing `provider_session_id`."
    )


def _compute_opencode_resumed_exact_transcript_match(
    active: str | None, recovered: str | None, saved: bool
) -> bool:
    return _opencode_exact_transcript_match(
        saved_exact_transcript_match=saved,
        active_provider_session_id=active,
        stored_provider_session_id=recovered,
    )


# ── Bundle instances ──────────────────────────────────────────────────────────

_CLAUDE_STATE_BUNDLE = _ServiceStateBundle(
    service="claude",
    start_session_hook=_noop_start_session_hook,
    make_exact_transcript_match=lambda _: ExactTranscriptMatch(value=False),
    make_recovered=lambda _: False,
    recover_new_session_id=lambda _: None,
    has_prior_sessions=_no_prior_sessions,
    probe_new_session_resumable=_claude_is_resumable,
    generate_new_session_id=lambda: (
        _builtin_runtime_client_module._new_provider_session_id()
    ),
    compute_new_session_exact_transcript_match=lambda _a, _r: False,
    resumed_relpath_none_uses_root=True,
    assert_resumed_and_recover_id=_claude_assert_resumed_and_recover_id,
    generate_resumed_session_id_or_raise=lambda: (
        _builtin_runtime_client_module._new_provider_session_id()
    ),
    compute_resumed_exact_transcript_match=lambda _a, _r, _s: False,
    continuation_run_kind=RunKind.RESUME.value,
    compute_continuation_exact_transcript_match=lambda _: False,
)

_CODEX_STATE_BUNDLE = _ServiceStateBundle(
    service="codex",
    start_session_hook=_codex_start_session_hook,
    make_exact_transcript_match=lambda _: ExactTranscriptMatch(value=False),
    make_recovered=lambda r: r,
    recover_new_session_id=_recover_codex_rollout_session_id,
    has_prior_sessions=_codex_has_prior_sessions,
    probe_new_session_resumable=_codex_probe_new_session_resumable,
    generate_new_session_id=lambda: None,
    compute_new_session_exact_transcript_match=lambda _a, _r: False,
    resumed_relpath_none_uses_root=False,
    assert_resumed_and_recover_id=_codex_assert_resumed_and_recover_id,
    generate_resumed_session_id_or_raise=_codex_generate_resumed_session_id_or_raise,
    compute_resumed_exact_transcript_match=lambda _a, _r, _s: False,
    continuation_run_kind=RunKind.RESUME.value,
    compute_continuation_exact_transcript_match=lambda _: False,
)

_OPENCODE_STATE_BUNDLE = _ServiceStateBundle(
    service="opencode",
    start_session_hook=_noop_start_session_hook,
    make_exact_transcript_match=lambda v: ExactTranscriptMatch(value=v),
    make_recovered=lambda _: False,
    recover_new_session_id=load_opencode_stored_session_id,
    has_prior_sessions=_no_prior_sessions,
    probe_new_session_resumable=_opencode_is_resumable,
    generate_new_session_id=lambda: (
        _builtin_runtime_client_module._new_provider_session_id()
    ),
    compute_new_session_exact_transcript_match=_compute_opencode_new_session_exact_transcript_match,
    resumed_relpath_none_uses_root=False,
    assert_resumed_and_recover_id=_opencode_assert_resumed_and_recover_id,
    generate_resumed_session_id_or_raise=lambda: (
        _builtin_runtime_client_module._new_provider_session_id()
    ),
    compute_resumed_exact_transcript_match=_compute_opencode_resumed_exact_transcript_match,
    continuation_run_kind=None,
    compute_continuation_exact_transcript_match=lambda etm: (
        etm.value if etm is not None else None
    ),
)


_SERVICE_BUNDLES: dict[str, _ServiceStateBundle] = {
    b.service: b
    for b in (_CLAUDE_STATE_BUNDLE, _CODEX_STATE_BUNDLE, _OPENCODE_STATE_BUNDLE)
}


# ── Shared start-session body ─────────────────────────────────────────────────


def _resolve_start_session_state(
    bundle: _ServiceStateBundle,
    *,
    runtime_state_dir: Path,
    caller_owned_session_store: bool,
    host_auth_path: Path | None = None,
) -> _StartSessionState:
    provider_state_dir = runtime_state_dir
    provider_state_dir.mkdir(parents=True, exist_ok=True)
    bundle.start_session_hook(provider_state_dir, host_auth_path)
    return _StartSessionState(
        provider_state_dir=provider_state_dir,
        provider_state_dir_relpath=("" if caller_owned_session_store else None),
    )


def _continuation_input_facts(
    bundle: _ServiceStateBundle,
    *,
    model: str,
    effort: str,
    provider_state_dir: Path,
    provider_state_dir_relpath: str | None,
    provider_session_id: str | None,
    recovered_provider_session_id: bool = False,
    run_kind: RunKind,
    exact_transcript_match: bool = False,
) -> ContinuationInputFacts:
    return ContinuationInputFacts(
        provider_identity=ProviderIdentity(
            service=bundle.service,
            model=model,
            effort=effort,
        ),
        provider_state_directory=ProviderStateDirectory(path=provider_state_dir),
        provider_state_relpath=_provider_state_relpath(provider_state_dir_relpath),
        provider_session_id=_provider_session_id(
            provider_session_id,
            recovered=bundle.make_recovered(recovered_provider_session_id),
        ),
        run_kind=run_kind,
        exact_transcript_match=bundle.make_exact_transcript_match(
            exact_transcript_match
        ),
    )


# ── Shared new-session body ───────────────────────────────────────────────────


def _resolve_new_session_facts(
    bundle: _ServiceStateBundle,
    *,
    runtime_state_dir: Path,
    caller_owned_session_store: bool,
    model: str,
    effort: str,
    host_auth_path: Path | None = None,
) -> _NewSessionResolution:
    session_state = _resolve_start_session_state(
        bundle,
        runtime_state_dir=runtime_state_dir,
        caller_owned_session_store=caller_owned_session_store,
        host_auth_path=host_auth_path,
    )
    provider_state_dir = session_state.provider_state_dir

    recovered_id = bundle.recover_new_session_id(provider_state_dir)
    if bundle.has_prior_sessions(provider_state_dir) and recovered_id is None:
        raise ContinuationUnrecoverableError(
            f"{bundle.service.capitalize()} continuation is not recoverable from provider state.",
            service_name=bundle.service,
        )

    if recovered_id is not None:
        session_id: str | None = recovered_id
        recovered = True
        run_kind = RunKind.RESUME
    else:
        session_id = bundle.generate_new_session_id()
        recovered = False
        run_kind = (
            RunKind.RESUME
            if bundle.probe_new_session_resumable(provider_state_dir)
            else RunKind.FRESH
        )

    exact_transcript_match = bundle.compute_new_session_exact_transcript_match(
        session_id, recovered_id
    )

    return _NewSessionResolution(
        provider_state_dir=provider_state_dir,
        continuation_input_facts=_continuation_input_facts(
            bundle,
            model=model,
            effort=effort,
            provider_state_dir=provider_state_dir,
            provider_state_dir_relpath=session_state.provider_state_dir_relpath,
            provider_session_id=session_id,
            recovered_provider_session_id=recovered,
            run_kind=run_kind,
            exact_transcript_match=exact_transcript_match,
        ),
    )


# ── Shared resumed-session body ───────────────────────────────────────────────


def _resolve_resumed_session_facts(
    bundle: _ServiceStateBundle,
    *,
    runtime_state_dir: Path | None,
    provider_state_dir_relpath: str | None,
    model: str,
    effort: str,
    provider_session_id: str | None,
    host_auth_path: Path | None = None,
    saved_exact_transcript_match: bool = False,
) -> _ResumedSessionResolution:
    # Resolve provider_state_dir from relpath.  When relpath is None the
    # behaviour is service-specific: Claude uses the store root
    # (resumed_relpath_none_uses_root=True); Codex treats it as "no store"
    # and keeps provider_state_dir as None.
    provider_state_dir: Path | None
    if runtime_state_dir is not None and provider_state_dir_relpath is not None:
        provider_state_dir = runtime_state_dir / provider_state_dir_relpath
        provider_state_dir.mkdir(parents=True, exist_ok=True)
    elif runtime_state_dir is not None and bundle.resumed_relpath_none_uses_root:
        provider_state_dir = runtime_state_dir
    else:
        provider_state_dir = None

    # check_dir is None when relpath is None so per-service assertions are
    # skipped: Claude's empty-state-dir check only applies when a relpath is
    # present; Codex defers to the RuntimeConfigurationError path below.
    check_dir = provider_state_dir if provider_state_dir_relpath is not None else None
    recovered_id = bundle.assert_resumed_and_recover_id(check_dir, host_auth_path)

    normalized_session_id = _normalize_provider_session_id(provider_session_id)
    active_session_id: str | None = normalized_session_id or recovered_id
    if active_session_id is None:
        active_session_id = bundle.generate_resumed_session_id_or_raise()

    exact_transcript_match = bundle.compute_resumed_exact_transcript_match(
        active_session_id, recovered_id, saved_exact_transcript_match
    )

    # When provider_state_dir is None (Codex with no store), fall back to
    # host_auth_path.parent as the path recorded in ContinuationInputFacts,
    # preserving current behaviour.
    facts_state_dir: Path
    if provider_state_dir is not None:
        facts_state_dir = provider_state_dir
    elif host_auth_path is not None:
        facts_state_dir = host_auth_path.parent
    else:
        facts_state_dir = runtime_state_dir  # type: ignore[assignment]

    return _ResumedSessionResolution(
        provider_state_dir=provider_state_dir,
        continuation_input_facts=_continuation_input_facts(
            bundle,
            model=model,
            effort=effort,
            provider_state_dir=facts_state_dir,
            provider_state_dir_relpath=provider_state_dir_relpath,
            provider_session_id=active_session_id,
            recovered_provider_session_id=normalized_session_id is None,
            run_kind=RunKind.RESUME,
            exact_transcript_match=exact_transcript_match,
        ),
    )


# ── Per-service start-session state resolvers (public) ───────────────────────


def resolve_codex_start_session_state(
    *,
    runtime_state_dir: Path,
    caller_owned_session_store: bool,
    host_auth_path: Path,
) -> _StartSessionState:
    return _resolve_start_session_state(
        _CODEX_STATE_BUNDLE,
        runtime_state_dir=runtime_state_dir,
        caller_owned_session_store=caller_owned_session_store,
        host_auth_path=host_auth_path,
    )


def resolve_claude_start_session_state(
    *,
    runtime_state_dir: Path,
    caller_owned_session_store: bool,
) -> _StartSessionState:
    return _resolve_start_session_state(
        _CLAUDE_STATE_BUNDLE,
        runtime_state_dir=runtime_state_dir,
        caller_owned_session_store=caller_owned_session_store,
    )


def resolve_opencode_start_session_state(
    *,
    runtime_state_dir: Path,
    caller_owned_session_store: bool,
) -> _StartSessionState:
    return _resolve_start_session_state(
        _OPENCODE_STATE_BUNDLE,
        runtime_state_dir=runtime_state_dir,
        caller_owned_session_store=caller_owned_session_store,
    )


# ── Per-service new-session resolvers (public thin wrappers) ──────────────────


def resolve_codex_new_session_facts(
    *,
    runtime_state_dir: Path,
    caller_owned_session_store: bool,
    model: str,
    effort: str,
    host_auth_path: Path,
) -> _NewSessionResolution:
    return _resolve_new_session_facts(
        _CODEX_STATE_BUNDLE,
        runtime_state_dir=runtime_state_dir,
        caller_owned_session_store=caller_owned_session_store,
        model=model,
        effort=effort,
        host_auth_path=host_auth_path,
    )


def resolve_opencode_new_session_facts(
    *,
    runtime_state_dir: Path,
    caller_owned_session_store: bool,
    model: str,
    effort: str,
) -> _NewSessionResolution:
    return _resolve_new_session_facts(
        _OPENCODE_STATE_BUNDLE,
        runtime_state_dir=runtime_state_dir,
        caller_owned_session_store=caller_owned_session_store,
        model=model,
        effort=effort,
    )


def resolve_claude_new_session_facts(
    *,
    runtime_state_dir: Path,
    caller_owned_session_store: bool,
    model: str,
    effort: str,
) -> _NewSessionResolution:
    return _resolve_new_session_facts(
        _CLAUDE_STATE_BUNDLE,
        runtime_state_dir=runtime_state_dir,
        caller_owned_session_store=caller_owned_session_store,
        model=model,
        effort=effort,
    )


# ── Per-service resumed-session resolvers (public thin wrappers) ──────────────


def resolve_claude_resumed_session_facts(
    *,
    runtime_state_dir: Path,
    provider_state_dir_relpath: str | None,
    model: str,
    effort: str,
    provider_session_id: str | None,
) -> _ResumedSessionResolution:
    return _resolve_resumed_session_facts(
        _CLAUDE_STATE_BUNDLE,
        runtime_state_dir=runtime_state_dir,
        provider_state_dir_relpath=provider_state_dir_relpath,
        model=model,
        effort=effort,
        provider_session_id=provider_session_id,
    )


def resolve_opencode_resumed_session_facts(
    *,
    runtime_state_dir: Path,
    continuation: Continuation,
    model: str,
    effort: str,
) -> _ResumedSessionResolution:
    continuation_facts = continuation.session_backed_facts
    provider_state_dir_relpath = continuation_facts.provider_state_dir_relpath
    if provider_state_dir_relpath is None:
        provider_state_dir_relpath = ""
    return _resolve_resumed_session_facts(
        _OPENCODE_STATE_BUNDLE,
        runtime_state_dir=runtime_state_dir,
        provider_state_dir_relpath=provider_state_dir_relpath,
        model=model,
        effort=effort,
        provider_session_id=continuation_facts.provider_session_id,
        saved_exact_transcript_match=bool(continuation_facts.exact_transcript_match),
    )


def resolve_codex_resumed_session_facts(
    *,
    runtime_state_dir: Path | None,
    provider_state_dir_relpath: str | None,
    model: str,
    effort: str,
    provider_session_id: str | None,
    host_auth_path: Path,
) -> _ResumedSessionResolution:
    return _resolve_resumed_session_facts(
        _CODEX_STATE_BUNDLE,
        runtime_state_dir=runtime_state_dir,
        provider_state_dir_relpath=provider_state_dir_relpath,
        model=model,
        effort=effort,
        provider_session_id=provider_session_id,
        host_auth_path=host_auth_path,
    )


# ── Per-service continuation-input-facts builders (public) ───────────────────


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
    return _continuation_input_facts(
        _CODEX_STATE_BUNDLE,
        model=model,
        effort=effort,
        provider_state_dir=provider_state_dir,
        provider_state_dir_relpath=provider_state_dir_relpath,
        provider_session_id=provider_session_id,
        recovered_provider_session_id=recovered_provider_session_id,
        run_kind=run_kind,
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
    return _continuation_input_facts(
        _CLAUDE_STATE_BUNDLE,
        model=model,
        effort=effort,
        provider_state_dir=provider_state_dir,
        provider_state_dir_relpath=provider_state_dir_relpath,
        provider_session_id=provider_session_id,
        run_kind=run_kind,
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
    return _continuation_input_facts(
        _OPENCODE_STATE_BUNDLE,
        model=model,
        effort=effort,
        provider_state_dir=provider_state_dir,
        provider_state_dir_relpath=provider_state_dir_relpath,
        provider_session_id=provider_session_id,
        run_kind=run_kind,
        exact_transcript_match=exact_transcript_match,
    )


def build_session_backed_continuation(
    continuation_input_facts: ContinuationInputFacts,
    *,
    tool_access: ToolAccess,
    provider_session_id: str | None = None,
) -> Continuation:
    service = continuation_input_facts.provider_identity.service
    bundle = _SERVICE_BUNDLES[service]

    continuation_run_kind = bundle.continuation_run_kind
    exact_transcript_match = bundle.compute_continuation_exact_transcript_match(
        continuation_input_facts.exact_transcript_match
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
