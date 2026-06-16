from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import pytest

import agent_runtime as runtime
from agent_runtime._import_isolation import assert_runtime_import_isolation
from agent_runtime.contracts import AssistantTurn, CredentialFailure, HardError, PromptTokens, Result, TransientError, UnsupportedTokens, UsageLimit
from agent_runtime.errors import (
    AgentCredentialFailureError,
    AgentFailedError,
    AgentRuntimeError,
    AgentTimeoutError,
    HardAgentError,
    TransientAgentError,
    UsageLimitError,
)
from agent_runtime.provider_errors import ProviderErrorObservation
from agent_runtime.roles import AgentRole
from agent_runtime.session import (
    ProviderSessionSelection,
    is_exact_resumable_service_session,
    normalize_state_dir_relpath,
    provider_state_relpath,
    provider_state_session_id_path,
    select_resumable_provider_session_id,
)
from agent_runtime.stage_priority_chain import chain_entries, render_chain_label, select_configured_candidate_chain
from agent_runtime.work import reduce_text_output_events


class _Service:
    def __init__(self, name: str, *, available: bool, wake_time: datetime) -> None:
        self.name = name
        self._available = available
        self._wake_time = wake_time
        self.available_checks: list[datetime | None] = []

    def is_available(self, now: datetime | None = None) -> bool:
        self.available_checks.append(now)
        return self._available

    def next_wake_time(self) -> datetime:
        return self._wake_time

    def mark_exhausted(self, reset_time: datetime | None) -> None:
        del reset_time

    def state_dir_relpath(self, role: AgentRole, namespace: str = "") -> str | None:
        del role, namespace
        return None

    def is_resumable(self, state_dir: Path) -> bool:
        del state_dir
        return False

    def valid_models(self) -> frozenset[str]:
        return frozenset()

    def valid_efforts(self) -> frozenset[str]:
        return frozenset()


@dataclass
class _RoleSession:
    service_sessions: dict[str, str | None]
    service_metadata: dict[str, dict[str, str] | None]
    exact_transcript_service: str | None = None

    def session_uuid(self) -> str:
        return "session-uuid"

    def service_session_id(self, service_name: str) -> str | None:
        return self.service_sessions.get(service_name)

    def save_service_session_id(self, service_name: str, session_id: str) -> None:
        self.service_sessions[service_name] = session_id

    def service_session_metadata(self, service_name: str) -> dict[str, str] | None:
        return self.service_metadata.get(service_name)

    def exact_transcript_service_name(self) -> str | None:
        return self.exact_transcript_service


def test_package_exports_runtime_surface() -> None:
    assert runtime.StageOverride.__module__.startswith("agent_runtime")
    assert runtime.AgentRuntimeError is AgentRuntimeError
    assert runtime.ServiceRegistry.__module__.startswith("agent_runtime")


def test_import_isolation_helper_reports_forbidden_modules() -> None:
    with pytest.raises(ImportError) as excinfo:
        assert_runtime_import_isolation(
            importer="agent_runtime",
            newly_loaded_modules={"allowed.mod", "forbidden.pkg", "forbidden.pkg.sub"},
            forbidden_prefixes=("forbidden.pkg",),
        )

    assert "forbidden.pkg" in str(excinfo.value)


def test_stage_chain_resolution_prefers_first_available_configured_service() -> None:
    override = runtime.StageOverride(
        service="missing",
        model="ignored",
        effort="medium",
        fallback=runtime.StageOverride(
            service="codex",
            model="gpt-5.4",
            effort="medium",
            fallback=runtime.StageOverride(
                service="claude",
                model="sonnet",
                effort="high",
            ),
        ),
    )

    selection = select_configured_candidate_chain(
        override,
        configured_service_names=("codex", "claude"),
        available_service_names=("claude",),
    )

    assert selection.has_configured_candidate is True
    assert selection.selected_chain == runtime.StageOverride(
        service="claude",
        model="sonnet",
        effort="high",
    )
    assert render_chain_label(override) == "missing -> codex -> claude"
    assert [entry.service for entry in chain_entries(override)] == [
        "missing",
        "codex",
        "claude",
    ]


def test_service_registry_resolve_and_wake_time() -> None:
    services: dict[str, runtime.AgentService] = {
        "codex": cast(
            runtime.AgentService,
            _Service(
                "codex",
                available=False,
                wake_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
            ),
        ),
        "claude": cast(
            runtime.AgentService,
            _Service(
                "claude",
                available=True,
                wake_time=datetime(2026, 1, 2, tzinfo=timezone.utc),
            ),
        ),
    }
    registry = runtime.ServiceRegistry(
        services
    )
    override = runtime.StageOverride(
        service="codex",
        model="gpt-5.4",
        effort="medium",
        fallback=runtime.StageOverride(
            service="claude",
            model="sonnet",
            effort="high",
        ),
    )

    resolved = registry.resolve(override, datetime(2026, 1, 1, tzinfo=timezone.utc))

    assert resolved == runtime.StageOverride(
        service="claude",
        model="sonnet",
        effort="high",
    )
    assert registry.has_available(datetime(2026, 1, 1, tzinfo=timezone.utc)) is True
    assert registry.next_wake_time(datetime(2026, 1, 1, tzinfo=timezone.utc)) == datetime(
        2026, 1, 1, tzinfo=timezone.utc
    )


def test_provider_state_helpers_normalize_legacy_layout_and_build_session_id_path() -> None:
    legacy = ".runtime-session/implementer/main/codex/"

    assert provider_state_relpath(
        AgentRole.IMPLEMENTER,
        "codex",
        session_root=".runtime-session",
    ) == ".runtime-session/implementer/codex/"
    assert normalize_state_dir_relpath(
        AgentRole.IMPLEMENTER,
        "main",
        "codex",
        legacy,
    ) == ".runtime-session/implementer/main/codex/"
    assert provider_state_session_id_path(Path("state"), "codex") == Path(
        "state/thread_id"
    )


def test_select_resumable_provider_session_id_recovers_and_persists_state() -> None:
    state_dir = Path("state")
    role_session = _RoleSession(
        service_sessions={},
        service_metadata={},
    )

    selection = select_resumable_provider_session_id(
        role_session,
        "codex",
        provider_state_dir=state_dir,
        has_resumable_provider_state=True,
        recover_provider_session_id=lambda path: "provider-session" if path == state_dir else None,
    )

    assert selection == ProviderSessionSelection(
        provider_session_id="provider-session",
        persist_provider_session_id=True,
    )
    assert role_session.service_session_id("codex") == "provider-session"


def test_exact_resumable_service_session_requires_matching_metadata_and_maybe_matcher() -> None:
    role_session = _RoleSession(
        service_sessions={"codex": "provider-session"},
        service_metadata={"codex": {"provider_session_id": "provider-session"}},
        exact_transcript_service="codex",
    )

    assert is_exact_resumable_service_session(
        role_session,
        "codex",
        provider_session_id="provider-session",
        provider_state_dir=Path("state"),
    ) is True
    assert is_exact_resumable_service_session(
        role_session,
        "codex",
        provider_session_id="provider-session",
        provider_state_dir=Path("state"),
        exact_provider_session_matcher=lambda *_args: False,
    ) is False


def test_reduce_text_output_events_returns_result_and_maps_errors() -> None:
    token_counts: list[int] = []
    turns: list[str] = []
    result = reduce_text_output_events(
        [PromptTokens(2), UnsupportedTokens(3, "source"), AssistantTurn("hello"), Result("done")],
        turns.append,
        token_counts.append,
        provider="codex",
    )

    assert result == "done"
    assert turns == ["hello"]
    assert token_counts == [2]

    observation = ProviderErrorObservation(
        service_name="codex",
        raw_provider_text="bad credential",
        source_stream="stderr",
    )
    with pytest.raises(UsageLimitError):
        reduce_text_output_events([UsageLimit(reset_time=None)], turns.append, provider="codex")
    with pytest.raises(TransientAgentError):
        reduce_text_output_events([TransientError(status_code=503, raw_message="retry")], turns.append, provider="codex")
    with pytest.raises(HardAgentError):
        reduce_text_output_events([HardError(status_code=400, raw_message="bad", observations=(observation,))], turns.append, provider="codex")
    with pytest.raises(AgentCredentialFailureError):
        reduce_text_output_events(
            [CredentialFailure(raw_message="missing auth", service_name="codex", source_observations=(observation,))],
            turns.append,
            provider="codex",
        )


def test_runtime_errors_capture_context() -> None:
    timeout = AgentTimeoutError("timed out")
    usage_limit = UsageLimitError(reset_time=None)
    transient = TransientAgentError("transient", status_code=502)
    hard = HardAgentError("hard", status_code=400, service_name="codex")
    failed = AgentFailedError("implementer", Path("worktree"), service_name="codex")

    assert isinstance(timeout, AgentRuntimeError)
    assert usage_limit.provider is None
    assert transient.status_code == 502
    assert hard.service_name == "codex"
    assert failed.session_dir == "implementer/codex"
