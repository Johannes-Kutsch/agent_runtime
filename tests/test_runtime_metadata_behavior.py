from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from agent_runtime.agent_log import AgentInvocationLog
from agent_runtime.errors import (
    AgentFailedError,
    AgentRuntimeError,
    AgentTimeoutError,
    HardAgentError,
    TransientAgentError,
    UsageLimitError,
)
from agent_runtime.execution_contracts import PromptRunSession
from agent_runtime.roles import InvocationRole
from agent_runtime.usage_limit_scope import UsageLimitScope
from agent_runtime.session import RunKind
from agent_runtime.usage_limit_decision import (
    SleepUntil,
    Stop,
    UsageLimitOutcome,
    decide_usage_limit_continuation,
)


@pytest.mark.parametrize("label", ["", "has space", "a/b", "../escape"])
def test_invocation_role_rejects_unsafe_labels(label: str) -> None:
    with pytest.raises(ValueError):
        InvocationRole(label)


@pytest.mark.parametrize("label", ["", "has space", "a/b", "../escape"])
def test_usage_limit_scope_rejects_unsafe_labels(label: str) -> None:
    with pytest.raises(ValueError):
        UsageLimitScope(label)


@pytest.mark.parametrize("label", [" ", "a/b", "../escape"])
def test_prompt_run_session_namespace_preserves_empty_default_and_rejects_unsafe_non_empty_values(
    label: str,
) -> None:
    assert PromptRunSession().namespace == ""
    assert PromptRunSession(namespace="").namespace == ""

    with pytest.raises(ValueError):
        PromptRunSession(namespace=label)


def test_agent_failed_error_rejects_unsafe_session_namespace_before_building_diagnostics() -> (
    None
):
    with pytest.raises(ValueError):
        AgentFailedError(
            invocation_role="implementer",
            worktree_path=Path("."),
            namespace="../escape",
        )


def test_agent_failed_error_rejects_unsafe_service_name_before_building_diagnostics() -> (
    None
):
    with pytest.raises(ValueError):
        AgentFailedError(
            invocation_role="implementer",
            worktree_path=Path("."),
            service_name="a/b",
        )


def test_agent_timeout_error_exposes_invocation_role_metadata() -> None:
    timeout = AgentTimeoutError(
        "timed out",
        invocation_role="reviewer",
        worktree_path=Path("worktree"),
    )

    assert timeout.invocation_role == "reviewer"


def test_agent_failed_error_exposes_invocation_role_metadata() -> None:
    failed = AgentFailedError(
        invocation_role="reviewer",
        worktree_path=Path("worktree"),
    )

    assert failed.invocation_role == "reviewer"


def test_agent_failed_error_builds_session_dir_from_namespace_metadata() -> None:
    failed = AgentFailedError(
        invocation_role="reviewer",
        worktree_path=Path("worktree"),
        namespace="main",
    )

    assert failed.session_dir == "reviewer/main"


def test_agent_failed_error_builds_session_dir_from_namespace_and_service_name_metadata() -> (
    None
):
    failed = AgentFailedError(
        invocation_role="reviewer",
        worktree_path=Path("worktree"),
        namespace="main",
        service_name="codex",
    )

    assert failed.session_dir == "reviewer/main/codex"


def test_usage_limit_error_exposes_usage_limit_scope_metadata() -> None:
    error = UsageLimitError(
        reset_time=None,
        usage_limit_scope=UsageLimitScope("quota-review"),
    )

    assert error.usage_limit_scope == UsageLimitScope("quota-review")


def test_permanent_usage_limit_account_label_remains_diagnostic_metadata() -> None:
    decision = decide_usage_limit_continuation(
        UsageLimitOutcome(
            reset_time=None,
            service_name=None,
            account_label="team account",
            is_permanent=True,
        ),
        stage_override=None,
        service_registry=None,
        now=datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc),
        compute_wake_time=lambda reset_time, current_time: (current_time, False),
    )

    assert isinstance(decision, Stop)
    assert decision.message is not None
    assert "team account" in decision.message
    assert "claude" not in decision.message.lower()


@pytest.mark.parametrize("service_name", [" ", "a/b", "../escape"])
def test_hard_agent_error_rejects_unsafe_runtime_service_labels_before_recording_diagnostics(
    service_name: str,
) -> None:
    with pytest.raises(ValueError):
        HardAgentError("hard", service_name=service_name)


def test_agent_invocation_log_uses_invocation_role_header_key(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "agent.log"
    invocation_log = AgentInvocationLog(
        now_local=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc)
    )

    with invocation_log.open_work_invocation(
        log_path=log_path,
        role=InvocationRole("implementer"),
        run_kind=RunKind.FRESH,
        session_uuid=None,
        prompt="already rendered prompt",
    ):
        pass

    header = json.loads(log_path.read_text().splitlines()[0])

    assert header["invocation_role"] == "implementer"
    assert "role" not in header


def test_agent_invocation_log_uses_log_name_and_logs_dir_parameters(
    tmp_path: Path,
) -> None:
    invocation_log = AgentInvocationLog(
        now_local=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc)
    )

    reserved_path = invocation_log.reserve(
        log_name="Issue 51 Review",
        logs_dir=tmp_path,
    )
    logical_log = invocation_log.start_logical_session(
        log_name="Issue 51 Review",
        logs_dir=tmp_path,
    )

    assert reserved_path.name == "issue-51-review-20260101T0000.log"
    assert logical_log.log_path.name == "issue-51-review-20260101T0000-2.log"


def test_agent_invocation_log_omits_default_usage_limit_scope(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "agent.log"
    invocation_log = AgentInvocationLog(
        now_local=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc)
    )

    with invocation_log.open_work_invocation(
        log_path=log_path,
        role=InvocationRole("implementer"),
        usage_limit_scope=UsageLimitScope("implementer"),
        run_kind=RunKind.FRESH,
        session_uuid=None,
        prompt="same scope as role",
    ):
        pass

    header = json.loads(log_path.read_text().splitlines()[0])

    assert "usage_limit_scope" not in header


def test_agent_invocation_log_records_non_default_usage_limit_scope(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "agent.log"
    invocation_log = AgentInvocationLog(
        now_local=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc)
    )

    with invocation_log.open_work_invocation(
        log_path=log_path,
        role=InvocationRole("implementer"),
        usage_limit_scope=UsageLimitScope("repo-write"),
        run_kind=RunKind.RESUME,
        session_uuid=None,
        prompt="different scope from role",
    ):
        pass

    header = json.loads(log_path.read_text().splitlines()[0])

    assert header["invocation_role"] == "implementer"
    assert header["usage_limit_scope"] == "repo-write"


def test_agent_invocation_log_records_provider_session_id_in_header(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "agent.log"
    invocation_log = AgentInvocationLog(
        now_local=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc)
    )

    with invocation_log.open_work_invocation(
        log_path=log_path,
        role=InvocationRole("implementer"),
        usage_limit_scope=UsageLimitScope("repo-write"),
        run_kind=RunKind.RESUME,
        session_uuid=None,
        prompt="different scope from role",
    ) as work_invocation:
        work_invocation.record_provider_session_id("provider-session")

    header = json.loads(log_path.read_text().splitlines()[0])

    assert header["provider_session_id"] == "provider-session"


def test_usage_limit_continuation_exposes_selected_usage_limit_scope() -> None:
    now = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    wake_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

    decision = decide_usage_limit_continuation(
        UsageLimitOutcome(
            reset_time=None,
            service_name="codex",
            usage_limit_scope=UsageLimitScope("quota-review"),
        ),
        stage_override=None,
        service_registry=None,
        now=now,
        compute_wake_time=lambda reset_time, current_time: (wake_time, False),
    )

    assert decision == SleepUntil(
        wake_time=wake_time,
        message="Usage limit reached. Sleeping until 12:00. Press Ctrl+C to abort.",
        is_estimated=False,
        usage_limit_scope=UsageLimitScope("quota-review"),
    )


def test_agent_timeout_error_is_an_agent_runtime_error() -> None:
    timeout = AgentTimeoutError("timed out")

    assert isinstance(timeout, AgentRuntimeError)


def test_usage_limit_error_defaults_service_name_metadata_to_none() -> None:
    usage_limit = UsageLimitError(reset_time=None)

    assert usage_limit.service_name is None


def test_transient_agent_error_exposes_status_code_metadata() -> None:
    transient = TransientAgentError("transient", status_code=502)

    assert transient.status_code == 502


def test_hard_agent_error_exposes_service_name_metadata() -> None:
    hard = HardAgentError("hard", status_code=400, service_name="codex")

    assert hard.service_name == "codex"


def test_agent_failed_error_builds_session_dir_from_service_name_metadata() -> None:
    failed = AgentFailedError("implementer", Path("worktree"), service_name="codex")

    assert failed.session_dir == "implementer/codex"
