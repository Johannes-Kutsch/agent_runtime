from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from agent_runtime._builtin_provider_agent_event_building import (
    build_claude_agent_event,
    build_codex_agent_event,
)
from agent_runtime._builtin_provider_stream_interpretation import (
    BuiltInProviderStreamInterpretation,
)
from agent_runtime._runtime_lifecycle import AgentEvent, ProviderAuth
from agent_runtime._built_in_provider_lifecycle_policy import (
    NewSessionFactsResult,
    NewSessionRedirect,
    ResumedSessionFactsInput,
    ResumedSessionFactsResult,
    policy_for_service,
)
from agent_runtime._session_backed_provider_state_resolution import (
    ContinuationInputFacts,
    ExactTranscriptMatch,
    PreparedOrRecoveredProviderSessionId,
    ProviderIdentity,
    ProviderStateDirectory,
    load_opencode_stored_session_id,
    opencode_continuation_input_facts,
)
from agent_runtime.errors import (
    AgentCredentialFailureError,
    ContinuationUnrecoverableError,
    RuntimeConfigurationError,
    UsageLimitError,
)
from agent_runtime.invocation_progress import InvocationProgress
from agent_runtime.session import RunKind
from agent_runtime.types import ProviderSelection


def test_policy_for_service_returns_claude_stream_interpretation() -> None:
    result = policy_for_service("claude").stream_interpretation()
    assert isinstance(result, BuiltInProviderStreamInterpretation)


def test_policy_for_service_returns_codex_stream_interpretation() -> None:
    result = policy_for_service("codex").stream_interpretation()
    assert isinstance(result, BuiltInProviderStreamInterpretation)


def test_policy_for_service_returns_opencode_stream_interpretation() -> None:
    result = policy_for_service("opencode").stream_interpretation()
    assert isinstance(result, BuiltInProviderStreamInterpretation)


def test_policy_for_service_claude_stream_interpretation_attributes_errors_to_claude_service() -> (
    None
):
    interpretation = policy_for_service("claude").stream_interpretation()
    line = (
        json.dumps(
            {
                "type": "result",
                "is_error": True,
                "errors": [
                    {"message": "No conversation found with session ID abc-123"}
                ],
            }
        )
        + "\n"
    )
    with pytest.raises(ContinuationUnrecoverableError) as exc_info:
        interpretation.reduce_output([line])
    assert exc_info.value.service_name == "claude"


def test_policy_for_service_codex_stream_interpretation_raises_usage_limit_with_codex_service_name() -> (
    None
):
    interpretation = policy_for_service("codex").stream_interpretation()
    with pytest.raises(UsageLimitError) as exc_info:
        interpretation.reduce_output(
            [
                json.dumps({"type": "error", "message": "You've hit your usage limit."})
                + "\n"
            ]
        )
    assert exc_info.value.service_name == "codex"


def test_policy_for_service_raises_runtime_configuration_error_for_unknown_service() -> (
    None
):
    with pytest.raises(RuntimeConfigurationError) as exc_info:
        policy_for_service("unknown")
    assert str(exc_info.value) == (
        "RuntimeClient session-backed execution is only implemented for Claude, Codex, and OpenCode."
    )


@pytest.mark.parametrize("service_name", ["", "CLAUDE", "Claude", "gpt", "gemini"])
def test_policy_for_service_raises_for_any_unrecognized_service_name(
    service_name: str,
) -> None:
    with pytest.raises(RuntimeConfigurationError):
        policy_for_service(service_name)


# validate_stage tests


def test_claude_policy_validate_stage_raises_for_unsupported_model() -> None:
    selection = ProviderSelection(service="claude", model="gpt-5.5", effort="medium")
    with pytest.raises(RuntimeConfigurationError, match="Unsupported Claude model"):
        policy_for_service("claude").validate_stage(selection)


def test_claude_policy_validate_stage_raises_for_unsupported_effort() -> None:
    selection = ProviderSelection(service="claude", model="sonnet", effort="turbo")
    with pytest.raises(RuntimeConfigurationError, match="Unsupported Claude effort"):
        policy_for_service("claude").validate_stage(selection)


def test_claude_policy_validate_stage_passes_for_valid_selection() -> None:
    selection = ProviderSelection(service="claude", model="sonnet", effort="medium")
    policy_for_service("claude").validate_stage(selection)


def test_codex_policy_validate_stage_raises_for_unsupported_model() -> None:
    selection = ProviderSelection(
        service="codex", model="claude-sonnet", effort="medium"
    )
    with pytest.raises(RuntimeConfigurationError, match="Unsupported Codex model"):
        policy_for_service("codex").validate_stage(selection)


def test_codex_policy_validate_stage_raises_for_unsupported_effort() -> None:
    selection = ProviderSelection(service="codex", model="gpt-5.5", effort="max")
    with pytest.raises(RuntimeConfigurationError, match="Unsupported Codex effort"):
        policy_for_service("codex").validate_stage(selection)


def test_codex_policy_validate_stage_passes_for_valid_selection() -> None:
    selection = ProviderSelection(service="codex", model="gpt-5.5", effort="medium")
    policy_for_service("codex").validate_stage(selection)


def test_opencode_policy_validate_stage_raises_for_unsupported_model() -> None:
    selection = ProviderSelection(service="opencode", model="gpt-5.5", effort="medium")
    with pytest.raises(RuntimeConfigurationError, match="Unsupported OpenCode model"):
        policy_for_service("opencode").validate_stage(selection)


def test_opencode_policy_validate_stage_raises_for_unsupported_effort() -> None:
    selection = ProviderSelection(service="opencode", model="kimi-k2.6", effort="high")
    with pytest.raises(RuntimeConfigurationError, match="Unsupported OpenCode effort"):
        policy_for_service("opencode").validate_stage(selection)


def test_opencode_policy_validate_stage_passes_for_valid_selection() -> None:
    selection = ProviderSelection(
        service="opencode", model="kimi-k2.6", effort="medium"
    )
    policy_for_service("opencode").validate_stage(selection)


# require_auth tests


def test_claude_policy_require_auth_raises_when_auth_is_none() -> None:
    with pytest.raises(
        AgentCredentialFailureError, match="Missing Claude Code OAuth token"
    ) as exc_info:
        policy_for_service("claude").require_auth(None)
    assert exc_info.value.service_name == "claude"


def test_claude_policy_require_auth_raises_when_token_is_missing() -> None:
    with pytest.raises(
        AgentCredentialFailureError, match="Missing Claude Code OAuth token"
    ):
        policy_for_service("claude").require_auth(
            ProviderAuth(claude_code_oauth_token=None)
        )


def test_claude_policy_require_auth_passes_with_valid_token() -> None:
    policy_for_service("claude").require_auth(
        ProviderAuth(claude_code_oauth_token="tok-abc")
    )


def test_codex_policy_require_auth_is_noop_for_none() -> None:
    policy_for_service("codex").require_auth(None)


def test_codex_policy_require_auth_is_noop_for_any_auth_value() -> None:
    policy_for_service("codex").require_auth(ProviderAuth())


def test_opencode_policy_require_auth_raises_when_auth_is_none() -> None:
    with pytest.raises(
        AgentCredentialFailureError, match="Missing OpenCode API key"
    ) as exc_info:
        policy_for_service("opencode").require_auth(None)
    assert exc_info.value.service_name == "opencode"


def test_opencode_policy_require_auth_raises_when_key_is_missing() -> None:
    with pytest.raises(AgentCredentialFailureError, match="Missing OpenCode API key"):
        policy_for_service("opencode").require_auth(ProviderAuth(opencode_api_key=None))


def test_opencode_policy_require_auth_passes_with_valid_key() -> None:
    policy_for_service("opencode").require_auth(
        ProviderAuth(opencode_api_key="key-xyz")
    )


# resolve_new_session_facts tests


def test_codex_policy_resolve_new_session_facts_raises_credential_failure_when_host_auth_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    host_home = tmp_path / "no-auth-home"
    host_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: host_home)

    runtime_state_dir = tmp_path / "state"
    runtime_state_dir.mkdir()

    with pytest.raises(
        AgentCredentialFailureError, match="Codex authentication missing"
    ) as exc_info:
        policy_for_service("codex").resolve_new_session_facts(
            runtime_state_dir, True, "gpt-5.4", "medium"
        )
    assert exc_info.value.service_name == "codex"


def test_codex_policy_resolve_new_session_facts_returns_result_when_host_auth_present(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    host_home = tmp_path / "host-home"
    auth_path = host_home / ".codex" / "auth.json"
    auth_path.parent.mkdir(parents=True)
    auth_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: host_home)

    runtime_state_dir = tmp_path / "state"
    runtime_state_dir.mkdir()

    outcome = policy_for_service("codex").resolve_new_session_facts(
        runtime_state_dir, True, "gpt-5.4", "medium"
    )
    assert isinstance(outcome, NewSessionFactsResult)


def test_codex_policy_resolve_new_session_facts_returns_redirect_when_existing_session_recovered(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import json as _json

    host_home = tmp_path / "host-home"
    auth_path = host_home / ".codex" / "auth.json"
    auth_path.parent.mkdir(parents=True)
    auth_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: host_home)

    runtime_state_dir = tmp_path / "state"
    runtime_state_dir.mkdir()

    rollout_dir = runtime_state_dir / "sessions" / "2026" / "05" / "30"
    rollout_dir.mkdir(parents=True)
    rollout_path = rollout_dir / "rollout-001.jsonl"
    session_id = "recovered-thread"
    rollout_path.write_text(
        _json.dumps({"type": "session_meta", "payload": {"id": session_id}})
        + "\n"
        + _json.dumps({"type": "session_meta", "payload": {"id": session_id}})
        + "\n",
        encoding="utf-8",
    )

    outcome = policy_for_service("codex").resolve_new_session_facts(
        runtime_state_dir, True, "gpt-5.4", "medium"
    )
    assert isinstance(outcome, NewSessionRedirect)
    assert outcome.continuation_input_facts is not None


# resolve_resumed_session_facts tests


def test_codex_policy_resolve_resumed_session_facts_raises_credential_failure_when_relpath_set_and_host_auth_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    host_home = tmp_path / "no-auth-home"
    host_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: host_home)

    with pytest.raises(
        AgentCredentialFailureError, match="Codex authentication missing"
    ) as exc_info:
        policy_for_service("codex").resolve_resumed_session_facts(
            ResumedSessionFactsInput(
                runtime_state_dir=tmp_path / "state",
                provider_state_dir_relpath="",
                provider_session_id="some-thread",
                exact_transcript_match=None,
                model="gpt-5.4",
                effort="medium",
            )
        )
    assert exc_info.value.service_name == "codex"


def test_codex_policy_resolve_resumed_session_facts_skips_auth_check_when_relpath_is_none(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    host_home = tmp_path / "no-auth-home"
    host_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: host_home)

    result = policy_for_service("codex").resolve_resumed_session_facts(
        ResumedSessionFactsInput(
            runtime_state_dir=tmp_path / "state",
            provider_state_dir_relpath=None,
            provider_session_id="some-thread",
            exact_transcript_match=None,
            model="gpt-5.4",
            effort="medium",
        )
    )
    assert isinstance(result, ResumedSessionFactsResult)


# refresh_active_session_facts tests


@pytest.fixture
def minimal_continuation_input_facts(tmp_path: Path) -> ContinuationInputFacts:
    provider_state_dir = tmp_path / "state"
    provider_state_dir.mkdir()
    return ContinuationInputFacts(
        provider_identity=ProviderIdentity(
            service="opencode", model="kimi-k2.6", effort="medium"
        ),
        provider_state_directory=ProviderStateDirectory(path=provider_state_dir),
        provider_state_relpath=None,
        provider_session_id=PreparedOrRecoveredProviderSessionId(
            value="prepared-session-id", recovered=False
        ),
        run_kind=RunKind.FRESH,
        exact_transcript_match=None,
    )


def test_claude_policy_refresh_active_session_facts_returns_input_unchanged(
    minimal_continuation_input_facts: ContinuationInputFacts,
) -> None:
    result = policy_for_service("claude").refresh_active_session_facts(
        minimal_continuation_input_facts, "obs-session-id"
    )
    assert result is minimal_continuation_input_facts


def test_claude_policy_refresh_active_session_facts_returns_input_unchanged_when_session_id_is_none(
    minimal_continuation_input_facts: ContinuationInputFacts,
) -> None:
    result = policy_for_service("claude").refresh_active_session_facts(
        minimal_continuation_input_facts, None
    )
    assert result is minimal_continuation_input_facts


def test_codex_policy_refresh_active_session_facts_returns_input_unchanged(
    minimal_continuation_input_facts: ContinuationInputFacts,
) -> None:
    result = policy_for_service("codex").refresh_active_session_facts(
        minimal_continuation_input_facts, "obs-session-id"
    )
    assert result is minimal_continuation_input_facts


def test_codex_policy_refresh_active_session_facts_returns_input_unchanged_when_session_id_is_none(
    minimal_continuation_input_facts: ContinuationInputFacts,
) -> None:
    result = policy_for_service("codex").refresh_active_session_facts(
        minimal_continuation_input_facts, None
    )
    assert result is minimal_continuation_input_facts


def test_opencode_policy_refresh_active_session_facts_returns_input_unchanged_when_session_id_is_none(
    minimal_continuation_input_facts: ContinuationInputFacts,
) -> None:
    result = policy_for_service("opencode").refresh_active_session_facts(
        minimal_continuation_input_facts, None
    )
    assert result is minimal_continuation_input_facts


def test_opencode_policy_refresh_active_session_facts_persists_observed_session_id(
    tmp_path: Path,
) -> None:
    provider_state_dir = tmp_path / "state"
    provider_state_dir.mkdir()
    facts = opencode_continuation_input_facts(
        model="kimi-k2.6",
        effort="medium",
        provider_state_dir=provider_state_dir,
        provider_state_dir_relpath=None,
        provider_session_id="prepared-session-id",
        run_kind=RunKind.FRESH,
        exact_transcript_match=False,
    )

    policy_for_service("opencode").refresh_active_session_facts(
        facts, "observed-session-id"
    )

    assert load_opencode_stored_session_id(provider_state_dir) == "observed-session-id"


def test_opencode_policy_refresh_active_session_facts_updates_provider_session_id_in_returned_facts(
    tmp_path: Path,
) -> None:
    provider_state_dir = tmp_path / "state"
    provider_state_dir.mkdir()
    facts = opencode_continuation_input_facts(
        model="kimi-k2.6",
        effort="medium",
        provider_state_dir=provider_state_dir,
        provider_state_dir_relpath=None,
        provider_session_id="prepared-session-id",
        run_kind=RunKind.FRESH,
        exact_transcript_match=False,
    )

    result = policy_for_service("opencode").refresh_active_session_facts(
        facts, "observed-session-id"
    )

    assert result.provider_session_id is not None
    assert result.provider_session_id.value == "observed-session-id"


def test_opencode_policy_refresh_active_session_facts_marks_exact_transcript_match_when_observed_matches_prepared(
    tmp_path: Path,
) -> None:
    provider_state_dir = tmp_path / "state"
    provider_state_dir.mkdir()
    facts = opencode_continuation_input_facts(
        model="kimi-k2.6",
        effort="medium",
        provider_state_dir=provider_state_dir,
        provider_state_dir_relpath=None,
        provider_session_id="same-session-id",
        run_kind=RunKind.RESUME,
        exact_transcript_match=True,
    )

    result = policy_for_service("opencode").refresh_active_session_facts(
        facts, "same-session-id"
    )

    assert result.exact_transcript_match == ExactTranscriptMatch(value=True)


def test_opencode_policy_refresh_active_session_facts_clears_exact_transcript_match_when_observed_differs_from_prepared(
    tmp_path: Path,
) -> None:
    provider_state_dir = tmp_path / "state"
    provider_state_dir.mkdir()
    facts = opencode_continuation_input_facts(
        model="kimi-k2.6",
        effort="medium",
        provider_state_dir=provider_state_dir,
        provider_state_dir_relpath=None,
        provider_session_id="prepared-session-id",
        run_kind=RunKind.RESUME,
        exact_transcript_match=True,
    )

    result = policy_for_service("opencode").refresh_active_session_facts(
        facts, "different-observed-session-id"
    )

    assert result.exact_transcript_match == ExactTranscriptMatch(value=False)


# resolve_ephemeral_provider_state_dir tests


def test_opencode_policy_resolve_ephemeral_provider_state_dir_returns_invocation_dir(
    tmp_path: Path,
) -> None:
    invocation_dir = tmp_path / "invocation"
    invocation_dir.mkdir()

    provider_state_dir, _ = policy_for_service(
        "opencode"
    ).resolve_ephemeral_provider_state_dir(invocation_dir)

    assert provider_state_dir == invocation_dir


def test_opencode_policy_resolve_ephemeral_provider_state_dir_cleanup_is_noop(
    tmp_path: Path,
) -> None:
    invocation_dir = tmp_path / "invocation"
    invocation_dir.mkdir()

    _, cleanup = policy_for_service("opencode").resolve_ephemeral_provider_state_dir(
        invocation_dir
    )
    cleanup()

    assert invocation_dir.exists()


def test_claude_policy_resolve_ephemeral_provider_state_dir_returns_fresh_directory(
    tmp_path: Path,
) -> None:
    invocation_dir = tmp_path / "invocation"
    invocation_dir.mkdir()

    provider_state_dir, cleanup = policy_for_service(
        "claude"
    ).resolve_ephemeral_provider_state_dir(invocation_dir)

    try:
        assert provider_state_dir.exists()
        assert provider_state_dir != invocation_dir
    finally:
        cleanup()


def test_claude_policy_resolve_ephemeral_provider_state_dir_cleanup_removes_directory(
    tmp_path: Path,
) -> None:
    invocation_dir = tmp_path / "invocation"
    invocation_dir.mkdir()

    provider_state_dir, cleanup = policy_for_service(
        "claude"
    ).resolve_ephemeral_provider_state_dir(invocation_dir)
    cleanup()

    assert not provider_state_dir.exists()


def test_claude_policy_resolve_ephemeral_provider_state_dir_uses_correct_prefix(
    tmp_path: Path,
) -> None:
    invocation_dir = tmp_path / "invocation"
    invocation_dir.mkdir()

    provider_state_dir, cleanup = policy_for_service(
        "claude"
    ).resolve_ephemeral_provider_state_dir(invocation_dir)

    try:
        assert provider_state_dir.parent == Path(tempfile.gettempdir())
        assert provider_state_dir.name.startswith("ephemeral-provider-state-")
    finally:
        cleanup()


def test_codex_policy_resolve_ephemeral_provider_state_dir_returns_fresh_directory(
    tmp_path: Path,
) -> None:
    invocation_dir = tmp_path / "invocation"
    invocation_dir.mkdir()

    provider_state_dir, cleanup = policy_for_service(
        "codex"
    ).resolve_ephemeral_provider_state_dir(invocation_dir)

    try:
        assert provider_state_dir.exists()
        assert provider_state_dir != invocation_dir
    finally:
        cleanup()


def test_codex_policy_resolve_ephemeral_provider_state_dir_cleanup_removes_directory(
    tmp_path: Path,
) -> None:
    invocation_dir = tmp_path / "invocation"
    invocation_dir.mkdir()

    provider_state_dir, cleanup = policy_for_service(
        "codex"
    ).resolve_ephemeral_provider_state_dir(invocation_dir)
    cleanup()

    assert not provider_state_dir.exists()


def test_codex_policy_resolve_ephemeral_provider_state_dir_uses_correct_prefix(
    tmp_path: Path,
) -> None:
    invocation_dir = tmp_path / "invocation"
    invocation_dir.mkdir()

    provider_state_dir, cleanup = policy_for_service(
        "codex"
    ).resolve_ephemeral_provider_state_dir(invocation_dir)

    try:
        assert provider_state_dir.parent == Path(tempfile.gettempdir())
        assert provider_state_dir.name.startswith("ephemeral-provider-state-")
    finally:
        cleanup()


def test_claude_policy_resolve_ephemeral_render_invocation_dir_returns_invocation_dir(
    tmp_path: Path,
) -> None:
    invocation_dir = tmp_path / "invocation"
    invocation_dir.mkdir()

    result = policy_for_service("claude").resolve_ephemeral_render_invocation_dir(
        invocation_dir
    )

    assert result == invocation_dir


def test_codex_policy_resolve_ephemeral_render_invocation_dir_returns_invocation_dir(
    tmp_path: Path,
) -> None:
    invocation_dir = tmp_path / "invocation"
    invocation_dir.mkdir()

    result = policy_for_service("codex").resolve_ephemeral_render_invocation_dir(
        invocation_dir
    )

    assert result == invocation_dir


def test_opencode_policy_resolve_ephemeral_render_invocation_dir_returns_tmp(
    tmp_path: Path,
) -> None:
    invocation_dir = tmp_path / "invocation"
    invocation_dir.mkdir()

    result = policy_for_service("opencode").resolve_ephemeral_render_invocation_dir(
        invocation_dir
    )

    assert result == Path("/tmp")


# apply_ephemeral_pre_invocation_seeding tests


def test_claude_policy_apply_ephemeral_pre_invocation_seeding_leaves_dir_unchanged(
    tmp_path: Path,
) -> None:
    provider_state_dir = tmp_path / "state"
    provider_state_dir.mkdir()

    policy_for_service("claude").apply_ephemeral_pre_invocation_seeding(
        provider_state_dir
    )

    assert list(provider_state_dir.iterdir()) == []


def test_opencode_policy_apply_ephemeral_pre_invocation_seeding_leaves_dir_unchanged(
    tmp_path: Path,
) -> None:
    provider_state_dir = tmp_path / "state"
    provider_state_dir.mkdir()

    policy_for_service("opencode").apply_ephemeral_pre_invocation_seeding(
        provider_state_dir
    )

    assert list(provider_state_dir.iterdir()) == []


def test_codex_policy_apply_ephemeral_pre_invocation_seeding_seeds_auth_when_host_auth_present(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    host_home = tmp_path / "host-home"
    auth_path = host_home / ".codex" / "auth.json"
    auth_path.parent.mkdir(parents=True)
    auth_path.write_text('{"token":"abc"}', encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: host_home)

    provider_state_dir = tmp_path / "state"
    provider_state_dir.mkdir()

    policy_for_service("codex").apply_ephemeral_pre_invocation_seeding(
        provider_state_dir
    )

    assert (provider_state_dir / "auth.json").read_text(
        encoding="utf-8"
    ) == '{"token":"abc"}'


def test_codex_policy_apply_ephemeral_pre_invocation_seeding_raises_credential_failure_when_host_auth_absent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    host_home = tmp_path / "no-auth-home"
    host_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: host_home)

    provider_state_dir = tmp_path / "state"
    provider_state_dir.mkdir()

    with pytest.raises(
        AgentCredentialFailureError, match="Codex authentication missing"
    ) as exc_info:
        policy_for_service("codex").apply_ephemeral_pre_invocation_seeding(
            provider_state_dir
        )
    assert exc_info.value.service_name == "codex"


# build_session_dispatch_interpretation tests

_CLAUDE_ASSISTANT_LINE = (
    json.dumps(
        {
            "type": "assistant",
            "message": {
                "content": [{"type": "text", "text": "hello from claude"}],
                "usage": {"input_tokens": 5},
            },
        }
    )
    + "\n"
)
_CLAUDE_RESULT_LINE = (
    json.dumps({"type": "result", "subtype": "success", "result": "hello from claude"})
    + "\n"
)
_CLAUDE_LINES = [_CLAUDE_ASSISTANT_LINE, _CLAUDE_RESULT_LINE]

_CODEX_THREAD_LINE = (
    json.dumps({"type": "thread.started", "thread_id": "thread-abc"}) + "\n"
)
_CODEX_ITEM_LINE = (
    json.dumps(
        {
            "type": "item.completed",
            "item": {"type": "agent_message", "text": "hello from codex"},
        }
    )
    + "\n"
)
_CODEX_USAGE_LINE = (
    json.dumps(
        {
            "type": "turn.completed",
            "usage": {"input_tokens": 8, "output_tokens": 4},
        }
    )
    + "\n"
)
_CODEX_LINES = [_CODEX_THREAD_LINE, _CODEX_ITEM_LINE, _CODEX_USAGE_LINE]

_OPENCODE_TEXT_WITH_SESSION_LINE = (
    json.dumps(
        {
            "type": "text",
            "sessionID": "sess-obs",
            "part": {
                "type": "text",
                "text": "hello from opencode",
                "time": {"end": True},
            },
        }
    )
    + "\n"
)
_OPENCODE_IDLE_WITH_SESSION_LINE = (
    json.dumps(
        {
            "type": "session.status",
            "sessionID": "sess-obs",
            "status": {"type": "idle"},
        }
    )
    + "\n"
)
_OPENCODE_TEXT_NO_SESSION_LINE = (
    json.dumps(
        {
            "type": "text",
            "part": {
                "type": "text",
                "text": "hello from opencode",
                "time": {"end": True},
            },
        }
    )
    + "\n"
)
_OPENCODE_IDLE_NO_SESSION_LINE = (
    json.dumps({"type": "session.status", "status": {"type": "idle"}}) + "\n"
)
_OPENCODE_LINES_WITH_SESSION = [
    _OPENCODE_TEXT_WITH_SESSION_LINE,
    _OPENCODE_IDLE_WITH_SESSION_LINE,
]
_OPENCODE_LINES_NO_SESSION = [
    _OPENCODE_TEXT_NO_SESSION_LINE,
    _OPENCODE_IDLE_NO_SESSION_LINE,
]


def test_claude_policy_build_session_dispatch_interpretation_reduces_to_expected_output_and_usage() -> (
    None
):
    dispatch_interpretation, _ = policy_for_service(
        "claude"
    ).build_session_dispatch_interpretation(
        on_live_output=None,
        fallback_provider_session_id=None,
        on_provider_session_id=None,
    )

    output, usage = dispatch_interpretation.reduce_output(_CLAUDE_LINES)

    assert output == "hello from claude"
    assert usage is not None
    assert usage.input_tokens == 5


def test_claude_policy_build_session_dispatch_interpretation_emits_one_event_per_line_to_on_live_output() -> (
    None
):
    live_events: list[AgentEvent] = []
    dispatch_interpretation, _ = policy_for_service(
        "claude"
    ).build_session_dispatch_interpretation(
        on_live_output=live_events.append,
        fallback_provider_session_id=None,
        on_provider_session_id=None,
    )

    consume = getattr(dispatch_interpretation.reduce_output, "consume_stdout_lines")
    consume(_CLAUDE_LINES)

    assert live_events == [
        build_claude_agent_event(_CLAUDE_ASSISTANT_LINE),
        build_claude_agent_event(_CLAUDE_RESULT_LINE),
    ]


def test_claude_policy_build_session_dispatch_interpretation_timeout_state_record_sets_usage_and_progress() -> (
    None
):
    _, timeout_state = policy_for_service(
        "claude"
    ).build_session_dispatch_interpretation(
        on_live_output=None,
        fallback_provider_session_id=None,
        on_provider_session_id=None,
    )

    timeout_state.record(_CLAUDE_LINES)

    assert timeout_state.usage is not None
    assert timeout_state.usage.input_tokens == 5
    assert timeout_state.provider_session_id is None
    assert timeout_state.invocation_progress is InvocationProgress.STARTED


def test_codex_policy_build_session_dispatch_interpretation_reduces_to_expected_output_and_usage() -> (
    None
):
    dispatch_interpretation, _ = policy_for_service(
        "codex"
    ).build_session_dispatch_interpretation(
        on_live_output=None,
        fallback_provider_session_id=None,
        on_provider_session_id=None,
    )

    output, usage = dispatch_interpretation.reduce_output(_CODEX_LINES)

    assert output == "hello from codex"
    assert usage is not None
    assert usage.input_tokens == 8
    assert usage.output_tokens == 4


def test_codex_policy_build_session_dispatch_interpretation_emits_one_event_per_line_to_on_live_output() -> (
    None
):
    live_events: list[AgentEvent] = []
    dispatch_interpretation, _ = policy_for_service(
        "codex"
    ).build_session_dispatch_interpretation(
        on_live_output=live_events.append,
        fallback_provider_session_id=None,
        on_provider_session_id=None,
    )

    consume = getattr(dispatch_interpretation.reduce_output, "consume_stdout_lines")
    consume(_CODEX_LINES)

    assert live_events == [
        build_codex_agent_event(_CODEX_THREAD_LINE),
        build_codex_agent_event(_CODEX_ITEM_LINE),
        build_codex_agent_event(_CODEX_USAGE_LINE),
    ]


def test_codex_policy_build_session_dispatch_interpretation_timeout_state_record_sets_usage_session_id_and_progress() -> (
    None
):
    _, timeout_state = policy_for_service(
        "codex"
    ).build_session_dispatch_interpretation(
        on_live_output=None,
        fallback_provider_session_id=None,
        on_provider_session_id=None,
    )

    timeout_state.record(_CODEX_LINES)

    assert timeout_state.usage is not None
    assert timeout_state.usage.input_tokens == 8
    assert timeout_state.provider_session_id == "thread-abc"
    assert timeout_state.invocation_progress is InvocationProgress.STARTED


def test_opencode_policy_build_session_dispatch_interpretation_uses_fallback_session_id_when_stream_has_none() -> (
    None
):
    dispatch_interpretation, _ = policy_for_service(
        "opencode"
    ).build_session_dispatch_interpretation(
        on_live_output=None,
        fallback_provider_session_id="fallback-abc",
        on_provider_session_id=None,
    )

    assert dispatch_interpretation.extract_provider_session_id is not None
    provider_session_id = dispatch_interpretation.extract_provider_session_id(
        _OPENCODE_LINES_NO_SESSION
    )

    assert provider_session_id == "fallback-abc"


def test_opencode_policy_build_session_dispatch_interpretation_fires_on_provider_session_id_when_session_observed() -> (
    None
):
    observed_session_ids: list[str] = []
    dispatch_interpretation, _ = policy_for_service(
        "opencode"
    ).build_session_dispatch_interpretation(
        on_live_output=lambda _: None,
        fallback_provider_session_id=None,
        on_provider_session_id=observed_session_ids.append,
    )

    consume = getattr(dispatch_interpretation.reduce_output, "consume_stdout_lines")
    consume(_OPENCODE_LINES_WITH_SESSION)

    assert observed_session_ids == ["sess-obs"]


def test_opencode_policy_build_session_dispatch_interpretation_timeout_state_record_uses_fallback_when_no_stream_session() -> (
    None
):
    _, timeout_state = policy_for_service(
        "opencode"
    ).build_session_dispatch_interpretation(
        on_live_output=None,
        fallback_provider_session_id="fallback-xyz",
        on_provider_session_id=None,
    )

    timeout_state.record(_OPENCODE_LINES_NO_SESSION)

    assert timeout_state.provider_session_id == "fallback-xyz"
    assert timeout_state.invocation_progress is InvocationProgress.STARTED


def test_stream_interpretation_method_still_exists_on_claude_policy() -> None:
    result = policy_for_service("claude").stream_interpretation()
    assert isinstance(result, BuiltInProviderStreamInterpretation)
