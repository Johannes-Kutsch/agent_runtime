from __future__ import annotations

from typing import Callable

from . import _builtin_provider_rendering as _builtin_provider_rendering_module
from ._builtin_provider_stream_interpretation import (
    BuiltInProviderStreamInterpretation,
    claude_built_in_provider_stream_interpretation,
    codex_built_in_provider_stream_interpretation,
    opencode_built_in_provider_stream_interpretation,
)
from ._runtime_lifecycle import ProviderAuth
from .errors import RuntimeConfigurationError
from .types import ProviderSelection


class SessionBackedProviderLifecyclePolicy:
    __slots__ = ("_stream_interpretation_fn", "_validate_stage_fn", "_require_auth_fn")

    def __init__(
        self,
        stream_interpretation_fn: Callable[[], BuiltInProviderStreamInterpretation],
        validate_stage_fn: Callable[[ProviderSelection], None],
        require_auth_fn: Callable[[ProviderAuth | None], None],
    ) -> None:
        self._stream_interpretation_fn = stream_interpretation_fn
        self._validate_stage_fn = validate_stage_fn
        self._require_auth_fn = require_auth_fn

    def stream_interpretation(self) -> BuiltInProviderStreamInterpretation:
        return self._stream_interpretation_fn()

    def validate_stage(self, selection: ProviderSelection) -> None:
        self._validate_stage_fn(selection)

    def require_auth(self, auth: ProviderAuth | None) -> None:
        self._require_auth_fn(auth)


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


_CLAUDE_POLICY = SessionBackedProviderLifecyclePolicy(
    claude_built_in_provider_stream_interpretation,
    _claude_validate_stage,
    _builtin_provider_rendering_module._require_claude_auth,
)
_CODEX_POLICY = SessionBackedProviderLifecyclePolicy(
    codex_built_in_provider_stream_interpretation,
    _codex_validate_stage,
    _noop_require_auth,
)
_OPENCODE_POLICY = SessionBackedProviderLifecyclePolicy(
    opencode_built_in_provider_stream_interpretation,
    _opencode_validate_stage,
    _builtin_provider_rendering_module._require_opencode_auth,
)

_POLICIES: dict[str, SessionBackedProviderLifecyclePolicy] = {
    "claude": _CLAUDE_POLICY,
    "codex": _CODEX_POLICY,
    "opencode": _OPENCODE_POLICY,
}


def policy_for_service(service_name: str) -> SessionBackedProviderLifecyclePolicy:
    policy = _POLICIES.get(service_name)
    if policy is None:
        raise RuntimeConfigurationError(
            "RuntimeClient session-backed execution is only implemented for Claude, Codex, and OpenCode."
        )
    return policy
