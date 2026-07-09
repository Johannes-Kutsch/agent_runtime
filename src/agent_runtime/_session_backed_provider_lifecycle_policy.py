from __future__ import annotations

from ._builtin_provider_stream_interpretation import (
    BuiltInProviderStreamInterpretation,
    claude_built_in_provider_stream_interpretation,
    codex_built_in_provider_stream_interpretation,
    opencode_built_in_provider_stream_interpretation,
)
from .errors import RuntimeConfigurationError


class SessionBackedProviderLifecyclePolicy:
    __slots__ = ("_stream_interpretation_fn",)

    def __init__(
        self,
        stream_interpretation_fn: object,
    ) -> None:
        self._stream_interpretation_fn = stream_interpretation_fn

    def stream_interpretation(self) -> BuiltInProviderStreamInterpretation:
        fn = self._stream_interpretation_fn
        assert callable(fn)
        return fn()  # type: ignore[no-any-return]


_CLAUDE_POLICY = SessionBackedProviderLifecyclePolicy(
    claude_built_in_provider_stream_interpretation
)
_CODEX_POLICY = SessionBackedProviderLifecyclePolicy(
    codex_built_in_provider_stream_interpretation
)
_OPENCODE_POLICY = SessionBackedProviderLifecyclePolicy(
    opencode_built_in_provider_stream_interpretation
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
