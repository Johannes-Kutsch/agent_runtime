from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from . import _builtin_provider_rendering as _builtin_provider_rendering_module
from ._builtin_provider_stream_interpretation import (
    BuiltInProviderStreamInterpretation,
    claude_built_in_provider_stream_interpretation,
    codex_built_in_provider_stream_interpretation,
    opencode_built_in_provider_stream_interpretation,
)
from ._runtime_lifecycle import Continuation, ProviderAuth
from .errors import RuntimeConfigurationError
from .types import ProviderSelection

if TYPE_CHECKING:
    from . import _session_backed_provider_state_resolution as _state_res_module


@dataclass(frozen=True)
class NewSessionFactsResult:
    provider_state_dir: Path
    continuation_input_facts: _state_res_module.ContinuationInputFacts


@dataclass(frozen=True)
class NewSessionRedirect:
    continuation_input_facts: _state_res_module.ContinuationInputFacts


NewSessionFactsOutcome = NewSessionFactsResult | NewSessionRedirect


@dataclass(frozen=True)
class ResumedSessionFactsResult:
    provider_state_dir: Path | None
    continuation_input_facts: _state_res_module.ContinuationInputFacts


@dataclass(frozen=True)
class ResumedSessionFactsInput:
    runtime_state_dir: Path
    provider_state_dir_relpath: str | None
    provider_session_id: str | None
    exact_transcript_match: bool | None
    model: str
    effort: str
    continuation: Continuation | None = field(default=None)


class BuiltInProviderLifecyclePolicy:
    __slots__ = (
        "_stream_interpretation_fn",
        "_validate_stage_fn",
        "_require_auth_fn",
        "_resolve_new_session_facts_fn",
        "_resolve_resumed_session_facts_fn",
        "_refresh_active_session_facts_fn",
        "_resolve_ephemeral_provider_state_dir_fn",
    )

    def __init__(
        self,
        stream_interpretation_fn: Callable[[], BuiltInProviderStreamInterpretation],
        validate_stage_fn: Callable[[ProviderSelection], None],
        require_auth_fn: Callable[[ProviderAuth | None], None],
        resolve_new_session_facts_fn: Callable[
            [Path, bool, str, str], NewSessionFactsOutcome
        ],
        resolve_resumed_session_facts_fn: Callable[
            [ResumedSessionFactsInput], ResumedSessionFactsResult
        ],
        refresh_active_session_facts_fn: Callable[
            [_state_res_module.ContinuationInputFacts, str | None],
            _state_res_module.ContinuationInputFacts,
        ],
        resolve_ephemeral_provider_state_dir_fn: Callable[
            [Path], tuple[Path, Callable[[], None]]
        ],
    ) -> None:
        self._stream_interpretation_fn = stream_interpretation_fn
        self._validate_stage_fn = validate_stage_fn
        self._require_auth_fn = require_auth_fn
        self._resolve_new_session_facts_fn = resolve_new_session_facts_fn
        self._resolve_resumed_session_facts_fn = resolve_resumed_session_facts_fn
        self._refresh_active_session_facts_fn = refresh_active_session_facts_fn
        self._resolve_ephemeral_provider_state_dir_fn = (
            resolve_ephemeral_provider_state_dir_fn
        )

    def stream_interpretation(self) -> BuiltInProviderStreamInterpretation:
        return self._stream_interpretation_fn()

    def validate_stage(self, selection: ProviderSelection) -> None:
        self._validate_stage_fn(selection)

    def require_auth(self, auth: ProviderAuth | None) -> None:
        self._require_auth_fn(auth)

    def resolve_new_session_facts(
        self,
        runtime_state_dir: Path,
        caller_owned_session_store: bool,
        model: str,
        effort: str,
    ) -> NewSessionFactsOutcome:
        return self._resolve_new_session_facts_fn(
            runtime_state_dir, caller_owned_session_store, model, effort
        )

    def resolve_resumed_session_facts(
        self,
        inp: ResumedSessionFactsInput,
    ) -> ResumedSessionFactsResult:
        return self._resolve_resumed_session_facts_fn(inp)

    def refresh_active_session_facts(
        self,
        continuation_input_facts: _state_res_module.ContinuationInputFacts,
        provider_session_id: str | None,
    ) -> _state_res_module.ContinuationInputFacts:
        return self._refresh_active_session_facts_fn(
            continuation_input_facts, provider_session_id
        )

    def resolve_ephemeral_provider_state_dir(
        self,
        invocation_dir: Path,
    ) -> tuple[Path, Callable[[], None]]:
        return self._resolve_ephemeral_provider_state_dir_fn(invocation_dir)


def _claude_validate_stage(selection: ProviderSelection) -> None:
    _builtin_provider_rendering_module._validate_claude_selection(
        _builtin_provider_rendering_module.BuiltInProviderSelectionFacts(
            service=selection.service,
            model=selection.model,
            effort=selection.effort,
        )
    )


def _codex_validate_stage(selection: ProviderSelection) -> None:
    _builtin_provider_rendering_module._validate_codex_selection(
        _builtin_provider_rendering_module.BuiltInProviderSelectionFacts(
            service=selection.service,
            model=selection.model,
            effort=selection.effort,
        )
    )


def _opencode_validate_stage(selection: ProviderSelection) -> None:
    _builtin_provider_rendering_module._validate_opencode_selection(
        _builtin_provider_rendering_module.BuiltInProviderSelectionFacts(
            service=selection.service,
            model=selection.model,
            effort=selection.effort,
        )
    )


def _noop_require_auth(auth: ProviderAuth | None) -> None:
    pass


def _claude_resolve_new_session_facts(
    runtime_state_dir: Path,
    caller_owned_session_store: bool,
    model: str,
    effort: str,
) -> NewSessionFactsOutcome:
    from . import _session_backed_provider_state_resolution as _state_resolution
    from .session import RunKind

    resolution = _state_resolution.resolve_claude_new_session_facts(
        runtime_state_dir=runtime_state_dir,
        caller_owned_session_store=caller_owned_session_store,
        model=model,
        effort=effort,
    )
    if resolution.continuation_input_facts.run_kind is RunKind.RESUME:
        return NewSessionRedirect(
            continuation_input_facts=resolution.continuation_input_facts
        )
    return NewSessionFactsResult(
        provider_state_dir=resolution.provider_state_dir,
        continuation_input_facts=resolution.continuation_input_facts,
    )


def _codex_resolve_new_session_facts(
    runtime_state_dir: Path,
    caller_owned_session_store: bool,
    model: str,
    effort: str,
) -> NewSessionFactsOutcome:
    from . import _session_backed_provider_state_resolution as _state_resolution
    from .session import RunKind

    host_auth_path = _builtin_provider_rendering_module._codex_host_auth_path()
    if not host_auth_path.exists():
        raise _builtin_provider_rendering_module._missing_codex_auth_error()
    resolution = _state_resolution.resolve_codex_new_session_facts(
        runtime_state_dir=runtime_state_dir,
        caller_owned_session_store=caller_owned_session_store,
        model=model,
        effort=effort,
        host_auth_path=host_auth_path,
    )
    if resolution.continuation_input_facts.run_kind is RunKind.RESUME:
        return NewSessionRedirect(
            continuation_input_facts=resolution.continuation_input_facts
        )
    return NewSessionFactsResult(
        provider_state_dir=resolution.provider_state_dir,
        continuation_input_facts=resolution.continuation_input_facts,
    )


def _opencode_resolve_new_session_facts(
    runtime_state_dir: Path,
    caller_owned_session_store: bool,
    model: str,
    effort: str,
) -> NewSessionFactsOutcome:
    from . import _session_backed_provider_state_resolution as _state_resolution

    resolution = _state_resolution.resolve_opencode_new_session_facts(
        runtime_state_dir=runtime_state_dir,
        caller_owned_session_store=caller_owned_session_store,
        model=model,
        effort=effort,
    )
    return NewSessionFactsResult(
        provider_state_dir=resolution.provider_state_dir,
        continuation_input_facts=resolution.continuation_input_facts,
    )


def _claude_resolve_resumed_session_facts(
    inp: ResumedSessionFactsInput,
) -> ResumedSessionFactsResult:
    from . import _session_backed_provider_state_resolution as _state_resolution

    resolution = _state_resolution.resolve_claude_resumed_session_facts(
        runtime_state_dir=inp.runtime_state_dir,
        provider_state_dir_relpath=inp.provider_state_dir_relpath,
        model=inp.model,
        effort=inp.effort,
        provider_session_id=inp.provider_session_id,
    )
    return ResumedSessionFactsResult(
        provider_state_dir=resolution.provider_state_dir,
        continuation_input_facts=resolution.continuation_input_facts,
    )


def _codex_resolve_resumed_session_facts(
    inp: ResumedSessionFactsInput,
) -> ResumedSessionFactsResult:
    from . import _session_backed_provider_state_resolution as _state_resolution

    host_auth_path = _builtin_provider_rendering_module._codex_host_auth_path()
    if inp.provider_state_dir_relpath is not None and not host_auth_path.exists():
        raise _builtin_provider_rendering_module._missing_codex_auth_error()
    resolution = _state_resolution.resolve_codex_resumed_session_facts(
        runtime_state_dir=inp.runtime_state_dir,
        provider_state_dir_relpath=inp.provider_state_dir_relpath,
        model=inp.model,
        effort=inp.effort,
        provider_session_id=inp.provider_session_id,
        host_auth_path=host_auth_path,
    )
    return ResumedSessionFactsResult(
        provider_state_dir=resolution.provider_state_dir,
        continuation_input_facts=resolution.continuation_input_facts,
    )


def _opencode_resolve_resumed_session_facts(
    inp: ResumedSessionFactsInput,
) -> ResumedSessionFactsResult:
    from . import _session_backed_provider_state_resolution as _state_resolution

    resolution = _state_resolution.resolve_opencode_resumed_session_facts(
        runtime_state_dir=inp.runtime_state_dir,
        continuation=inp.continuation,  # type: ignore[arg-type]
        model=inp.model,
        effort=inp.effort,
    )
    return ResumedSessionFactsResult(
        provider_state_dir=resolution.provider_state_dir,
        continuation_input_facts=resolution.continuation_input_facts,
    )


def _temp_dir_ephemeral_provider_state_dir(
    invocation_dir: Path,
) -> tuple[Path, Callable[[], None]]:
    temp_dir = tempfile.TemporaryDirectory(prefix="ephemeral-provider-state-")
    return Path(temp_dir.name), temp_dir.cleanup


def _invocation_dir_ephemeral_provider_state_dir(
    invocation_dir: Path,
) -> tuple[Path, Callable[[], None]]:
    return invocation_dir, lambda: None


def _noop_refresh_active_session_facts(
    continuation_input_facts: _state_res_module.ContinuationInputFacts,
    provider_session_id: str | None,
) -> _state_res_module.ContinuationInputFacts:
    return continuation_input_facts


def _opencode_refresh_active_session_facts(
    continuation_input_facts: _state_res_module.ContinuationInputFacts,
    provider_session_id: str | None,
) -> _state_res_module.ContinuationInputFacts:
    from . import _session_backed_provider_state_resolution as _state_resolution

    return _state_resolution.resolve_opencode_active_session_facts(
        continuation_input_facts,
        provider_session_id=provider_session_id,
    )


_CLAUDE_POLICY = BuiltInProviderLifecyclePolicy(
    claude_built_in_provider_stream_interpretation,
    _claude_validate_stage,
    _builtin_provider_rendering_module._require_claude_auth,
    _claude_resolve_new_session_facts,
    _claude_resolve_resumed_session_facts,
    _noop_refresh_active_session_facts,
    _temp_dir_ephemeral_provider_state_dir,
)
_CODEX_POLICY = BuiltInProviderLifecyclePolicy(
    codex_built_in_provider_stream_interpretation,
    _codex_validate_stage,
    _noop_require_auth,
    _codex_resolve_new_session_facts,
    _codex_resolve_resumed_session_facts,
    _noop_refresh_active_session_facts,
    _temp_dir_ephemeral_provider_state_dir,
)
_OPENCODE_POLICY = BuiltInProviderLifecyclePolicy(
    opencode_built_in_provider_stream_interpretation,
    _opencode_validate_stage,
    _builtin_provider_rendering_module._require_opencode_auth,
    _opencode_resolve_new_session_facts,
    _opencode_resolve_resumed_session_facts,
    _opencode_refresh_active_session_facts,
    _invocation_dir_ephemeral_provider_state_dir,
)

_POLICIES: dict[str, BuiltInProviderLifecyclePolicy] = {
    "claude": _CLAUDE_POLICY,
    "codex": _CODEX_POLICY,
    "opencode": _OPENCODE_POLICY,
}


def policy_for_service(service_name: str) -> BuiltInProviderLifecyclePolicy:
    policy = _POLICIES.get(service_name)
    if policy is None:
        raise RuntimeConfigurationError(
            "RuntimeClient session-backed execution is only implemented for Claude, Codex, and OpenCode."
        )
    return policy
