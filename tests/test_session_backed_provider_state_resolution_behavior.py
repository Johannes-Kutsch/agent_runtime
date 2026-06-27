from pathlib import Path

import pytest

import agent_runtime._session_backed_provider_state_resolution as provider_state_resolution
import agent_runtime.contracts as contracts_runtime
import agent_runtime.runtime as prompt_runtime
from agent_runtime.errors import RuntimeConfigurationError
from agent_runtime.session import RunKind


@pytest.mark.parametrize(
    (
        "continuation_input_facts",
        "expected_continuation",
    ),
    [
        (
            provider_state_resolution.codex_continuation_input_facts(
                model="gpt-5.4",
                effort="medium",
                provider_state_dir=Path("/tmp/codex"),
                provider_state_dir_relpath="implementer/main/codex/",
                provider_session_id="recovered-thread",
                recovered_provider_session_id=True,
                run_kind=RunKind.RESUME,
            ),
            prompt_runtime.Continuation(
                selected_service="codex",
                selected_model="gpt-5.4",
                selected_effort="medium",
                tool_access=contracts_runtime.ToolAccess.no_tools(),
                provider_resume_state={
                    "run_kind": "resume",
                    "provider_session_id": "recovered-thread",
                    "provider_state_dir_relpath": "implementer/main/codex/",
                    "exact_transcript_match": False,
                },
            ),
        ),
        (
            provider_state_resolution.claude_continuation_input_facts(
                model="sonnet",
                effort="medium",
                provider_state_dir=Path("/tmp/claude"),
                provider_state_dir_relpath="implementer/main/claude/",
                provider_session_id="claude-session-123",
                run_kind=RunKind.FRESH,
            ),
            prompt_runtime.Continuation(
                selected_service="claude",
                selected_model="sonnet",
                selected_effort="medium",
                tool_access=contracts_runtime.ToolAccess.no_tools(),
                provider_resume_state={
                    "run_kind": "resume",
                    "provider_session_id": "claude-session-123",
                    "provider_state_dir_relpath": "implementer/main/claude/",
                    "exact_transcript_match": False,
                },
            ),
        ),
        (
            provider_state_resolution.opencode_continuation_input_facts(
                model="glm-5.2",
                effort="medium",
                provider_state_dir=Path("/tmp/opencode"),
                provider_state_dir_relpath="implementer/main/opencode/",
                provider_session_id="persisted-session-1",
                run_kind=RunKind.RESUME,
                exact_transcript_match=True,
            ),
            prompt_runtime.Continuation(
                selected_service="opencode",
                selected_model="glm-5.2",
                selected_effort="medium",
                tool_access=contracts_runtime.ToolAccess.no_tools(),
                provider_resume_state={
                    "provider_session_id": "persisted-session-1",
                    "provider_state_dir_relpath": "implementer/main/opencode/",
                    "exact_transcript_match": True,
                },
            ),
        ),
    ],
)
def test_session_backed_provider_state_resolution_builds_current_continuation_payload_through_module_interface(
    continuation_input_facts: provider_state_resolution.ContinuationInputFacts,
    expected_continuation: prompt_runtime.Continuation,
) -> None:
    assert (
        provider_state_resolution.build_session_backed_continuation(
            continuation_input_facts,
            tool_access=contracts_runtime.ToolAccess.no_tools(),
        )
        == expected_continuation
    )


@pytest.mark.parametrize(
    ("caller_owned_session_store", "expected_relpath"),
    [
        (True, "implementer/slice422/codex/"),
        (False, None),
    ],
)
def test_session_backed_provider_state_resolution_prepares_codex_start_session_state_through_module_interface(
    tmp_path: Path,
    caller_owned_session_store: bool,
    expected_relpath: str | None,
) -> None:
    host_auth_path = tmp_path / "host-home" / ".codex" / "auth.json"
    host_auth_path.parent.mkdir(parents=True, exist_ok=True)
    host_auth_path.write_text('{"token":"host-auth"}\n', encoding="utf-8")

    resolution = provider_state_resolution.resolve_codex_start_session_state(
        runtime_state_dir=tmp_path / "session-store",
        session_namespace="slice422",
        caller_owned_session_store=caller_owned_session_store,
        host_auth_path=host_auth_path,
    )

    assert resolution.provider_state_dir == (
        tmp_path / "session-store" / "implementer" / "slice422" / "codex"
    )
    assert resolution.provider_state_dir.is_dir()
    assert resolution.provider_state_dir_relpath == expected_relpath


def test_session_backed_provider_state_resolution_seeds_codex_auth_only_when_missing_through_module_interface(
    tmp_path: Path,
) -> None:
    host_auth_path = tmp_path / "host-home" / ".codex" / "auth.json"
    host_auth_path.parent.mkdir(parents=True, exist_ok=True)
    host_auth_path.write_text('{"token":"host-auth"}\n', encoding="utf-8")

    copied = provider_state_resolution.resolve_codex_start_session_state(
        runtime_state_dir=tmp_path / "copied-store",
        session_namespace="slice422",
        caller_owned_session_store=True,
        host_auth_path=host_auth_path,
    )
    preserved_store = tmp_path / "preserved-store"
    existing_auth_path = (
        preserved_store / "implementer" / "slice422" / "codex" / "auth.json"
    )
    existing_auth_path.parent.mkdir(parents=True, exist_ok=True)
    existing_auth_path.write_text('{"token":"existing-auth"}\n', encoding="utf-8")

    preserved = provider_state_resolution.resolve_codex_start_session_state(
        runtime_state_dir=preserved_store,
        session_namespace="slice422",
        caller_owned_session_store=True,
        host_auth_path=host_auth_path,
    )

    assert (copied.provider_state_dir / "auth.json").read_text(encoding="utf-8") == (
        '{"token":"host-auth"}\n'
    )
    assert (preserved.provider_state_dir / "auth.json").read_text(
        encoding="utf-8"
    ) == '{"token":"existing-auth"}\n'


@pytest.mark.parametrize(
    ("caller_owned_session_store", "expected_relpath"),
    [
        (True, "implementer/slice424/claude/"),
        (False, None),
    ],
)
def test_session_backed_provider_state_resolution_prepares_claude_start_session_state_through_module_interface(
    tmp_path: Path,
    caller_owned_session_store: bool,
    expected_relpath: str | None,
) -> None:
    resolution = provider_state_resolution.resolve_claude_start_session_state(
        runtime_state_dir=tmp_path / "session-store",
        session_namespace="slice424",
        caller_owned_session_store=caller_owned_session_store,
    )

    assert resolution.provider_state_dir == (
        tmp_path / "session-store" / "implementer" / "slice424" / "claude"
    )
    assert resolution.provider_state_dir.is_dir()
    assert resolution.provider_state_dir_relpath == expected_relpath


@pytest.mark.parametrize(
    ("caller_owned_session_store", "expected_relpath"),
    [
        (True, "implementer/slice426/opencode/"),
        (False, None),
    ],
)
def test_session_backed_provider_state_resolution_prepares_opencode_start_session_state_through_module_interface(
    tmp_path: Path,
    caller_owned_session_store: bool,
    expected_relpath: str | None,
) -> None:
    resolution = provider_state_resolution.resolve_opencode_start_session_state(
        runtime_state_dir=tmp_path / "session-store",
        session_namespace="slice426",
        caller_owned_session_store=caller_owned_session_store,
    )

    assert resolution.provider_state_dir == (
        tmp_path / "session-store" / "implementer" / "slice426" / "opencode"
    )
    assert resolution.provider_state_dir.is_dir()
    assert resolution.provider_state_dir_relpath == expected_relpath


@pytest.mark.parametrize(
    (
        "resume_jsonl_contents",
        "session_id_contents",
        "expected_run_kind",
        "expected_provider_session_id",
        "expected_exact_transcript_match",
    ),
    [
        ("[]", None, RunKind.RESUME, "prepared-opencode-session", False),
        (None, " persisted-session \n", RunKind.RESUME, "persisted-session", True),
        (None, None, RunKind.FRESH, "prepared-opencode-session", False),
    ],
)
def test_session_backed_provider_state_resolution_prepares_opencode_new_session_facts_from_native_state_through_module_interface(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    resume_jsonl_contents: str | None,
    session_id_contents: str | None,
    expected_run_kind: RunKind,
    expected_provider_session_id: str,
    expected_exact_transcript_match: bool,
) -> None:
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module,
        "_new_provider_session_id",
        lambda: "prepared-opencode-session",
    )
    provider_state_dir = (
        tmp_path / "session-store" / "implementer" / "slice426" / "opencode"
    )
    provider_state_dir.mkdir(parents=True, exist_ok=True)
    if resume_jsonl_contents is not None:
        (provider_state_dir / "resume.jsonl").write_text(
            resume_jsonl_contents,
            encoding="utf-8",
        )
    if session_id_contents is not None:
        (provider_state_dir / "session_id").write_text(
            session_id_contents,
            encoding="utf-8",
        )

    resolution = provider_state_resolution.resolve_opencode_new_session_facts(
        runtime_state_dir=tmp_path / "session-store",
        session_namespace="slice426",
        caller_owned_session_store=True,
        model="glm-5.2",
        effort="medium",
    )

    assert resolution.provider_state_dir == provider_state_dir
    assert resolution.continuation_input_facts == (
        provider_state_resolution.opencode_continuation_input_facts(
            model="glm-5.2",
            effort="medium",
            provider_state_dir=provider_state_dir,
            provider_state_dir_relpath="implementer/slice426/opencode/",
            provider_session_id=expected_provider_session_id,
            run_kind=expected_run_kind,
            exact_transcript_match=expected_exact_transcript_match,
        )
    )


def test_session_backed_provider_state_resolution_recovers_codex_resume_facts_from_rollout_state_through_module_interface(
    tmp_path: Path,
) -> None:
    host_auth_path = tmp_path / "host-home" / ".codex" / "auth.json"
    host_auth_path.parent.mkdir(parents=True, exist_ok=True)
    host_auth_path.write_text('{"token":"host-auth"}\n', encoding="utf-8")

    runtime_state_dir = tmp_path / "session-store"
    rollout_dir = (
        runtime_state_dir / "implementer" / "slice422" / "codex" / "sessions" / "2026"
    )
    rollout_dir.mkdir(parents=True, exist_ok=True)
    (rollout_dir / "rollout-001.jsonl").write_text(
        "{not-json\n"
        '{"type":"turn.completed"}\n'
        '{"type":"session_meta","payload":[]}\n'
        '{"type":"session_meta","payload":{"id":"recovered-thread"}}\n'
        '{"type":"session_meta","payload":{"id":"   "}}\n',
        encoding="utf-8",
    )

    resolution = provider_state_resolution.resolve_codex_resumed_session_facts(
        runtime_state_dir=runtime_state_dir,
        provider_state_dir_relpath="implementer/slice422/codex/",
        model="gpt-5.4",
        effort="medium",
        provider_session_id="   ",
        host_auth_path=host_auth_path,
    )

    assert resolution.provider_state_dir == (
        runtime_state_dir / "implementer" / "slice422" / "codex"
    )
    assert resolution.continuation_input_facts == (
        provider_state_resolution.codex_continuation_input_facts(
            model="gpt-5.4",
            effort="medium",
            provider_state_dir=runtime_state_dir / "implementer" / "slice422" / "codex",
            provider_state_dir_relpath="implementer/slice422/codex/",
            provider_session_id="recovered-thread",
            recovered_provider_session_id=True,
            run_kind=RunKind.RESUME,
        )
    )


def test_session_backed_provider_state_resolution_loads_trimmed_opencode_session_id_and_treats_nonvalues_as_missing_through_module_interface(
    tmp_path: Path,
) -> None:
    missing_state_dir = tmp_path / "missing"
    blank_state_dir = tmp_path / "blank"
    blank_state_dir.mkdir(parents=True, exist_ok=True)
    (blank_state_dir / "session_id").write_text("  \n", encoding="utf-8")

    unreadable_state_dir = tmp_path / "unreadable"
    unreadable_state_dir.mkdir(parents=True, exist_ok=True)
    (unreadable_state_dir / "session_id").write_bytes(b"\xff\xfe")

    present_state_dir = tmp_path / "present"
    present_state_dir.mkdir(parents=True, exist_ok=True)
    (present_state_dir / "session_id").write_text(
        " persisted-session \n",
        encoding="utf-8",
    )

    assert (
        provider_state_resolution.load_opencode_stored_session_id(missing_state_dir)
        is None
    )
    assert (
        provider_state_resolution.load_opencode_stored_session_id(blank_state_dir)
        is None
    )
    assert (
        provider_state_resolution.load_opencode_stored_session_id(unreadable_state_dir)
        is None
    )
    assert provider_state_resolution.load_opencode_stored_session_id(None) is None
    assert (
        provider_state_resolution.load_opencode_stored_session_id(present_state_dir)
        == "persisted-session"
    )


def test_session_backed_provider_state_resolution_persists_opencode_provider_session_id_in_native_state_through_module_interface(
    tmp_path: Path,
) -> None:
    provider_state_dir = (
        tmp_path / "session-store" / "implementer" / "slice426" / "opencode"
    )
    provider_state_dir.mkdir(parents=True, exist_ok=True)

    provider_state_resolution.persist_opencode_provider_session_id(
        provider_state_dir,
        "persisted-session",
    )

    assert (provider_state_dir / "session_id").read_text(encoding="utf-8") == (
        "persisted-session\n"
    )


@pytest.mark.parametrize(
    (
        "prepared_provider_session_id",
        "saved_exact_transcript_match",
        "active_provider_session_id",
        "expected_provider_session_id",
        "expected_exact_transcript_match",
    ),
    [
        (
            "persisted-session",
            True,
            " persisted-session \n",
            "persisted-session",
            True,
        ),
        (
            "persisted-session",
            True,
            "observed-session",
            "observed-session",
            False,
        ),
        (
            "persisted-session",
            False,
            "persisted-session",
            "persisted-session",
            False,
        ),
    ],
)
def test_session_backed_provider_state_resolution_resolves_opencode_active_session_facts_through_module_interface(
    tmp_path: Path,
    prepared_provider_session_id: str,
    saved_exact_transcript_match: bool,
    active_provider_session_id: str,
    expected_provider_session_id: str,
    expected_exact_transcript_match: bool,
) -> None:
    provider_state_dir = (
        tmp_path / "session-store" / "implementer" / "slice427" / "opencode"
    )
    provider_state_dir.mkdir(parents=True, exist_ok=True)
    continuation_input_facts = (
        provider_state_resolution.opencode_continuation_input_facts(
            model="glm-5.2",
            effort="medium",
            provider_state_dir=provider_state_dir,
            provider_state_dir_relpath="implementer/slice427/opencode/",
            provider_session_id=prepared_provider_session_id,
            run_kind=RunKind.RESUME,
            exact_transcript_match=saved_exact_transcript_match,
        )
    )

    resolved = provider_state_resolution.resolve_opencode_active_session_facts(
        continuation_input_facts,
        provider_session_id=active_provider_session_id,
    )

    assert resolved == provider_state_resolution.opencode_continuation_input_facts(
        model="glm-5.2",
        effort="medium",
        provider_state_dir=provider_state_dir,
        provider_state_dir_relpath="implementer/slice427/opencode/",
        provider_session_id=expected_provider_session_id,
        run_kind=RunKind.RESUME,
        exact_transcript_match=expected_exact_transcript_match,
    )
    assert (provider_state_dir / "session_id").read_text(encoding="utf-8") == (
        f"{expected_provider_session_id}\n"
    )


def test_session_backed_provider_state_resolution_keeps_opencode_active_session_facts_unchanged_when_provider_session_id_is_missing_through_module_interface(
    tmp_path: Path,
) -> None:
    provider_state_dir = (
        tmp_path / "session-store" / "implementer" / "slice427" / "opencode"
    )
    provider_state_dir.mkdir(parents=True, exist_ok=True)
    continuation_input_facts = (
        provider_state_resolution.opencode_continuation_input_facts(
            model="glm-5.2",
            effort="medium",
            provider_state_dir=provider_state_dir,
            provider_state_dir_relpath="implementer/slice427/opencode/",
            provider_session_id="prepared-session",
            run_kind=RunKind.RESUME,
            exact_transcript_match=True,
        )
    )

    resolved = provider_state_resolution.resolve_opencode_active_session_facts(
        continuation_input_facts,
        provider_session_id="  \n",
    )

    assert resolved is continuation_input_facts
    assert not (provider_state_dir / "session_id").exists()


@pytest.mark.parametrize(
    (
        "provider_state_dir_relpath",
        "continuation_provider_session_id",
        "stored_provider_session_id",
        "expected_provider_state_dir",
        "expected_provider_session_id",
        "expected_exact_transcript_match",
    ),
    [
        (
            "implementer/main/opencode/",
            None,
            "stored-session",
            ("implementer", "main", "opencode"),
            "stored-session",
            True,
        ),
        (
            None,
            None,
            "stored-session",
            ("implementer", "fallback", "opencode"),
            "stored-session",
            True,
        ),
        (
            "implementer/main/opencode/",
            "continuation-session",
            "stored-session",
            ("implementer", "main", "opencode"),
            "continuation-session",
            False,
        ),
    ],
)
def test_session_backed_provider_state_resolution_restores_opencode_resumed_session_facts_from_continuation_and_session_store_through_module_interface(
    tmp_path: Path,
    provider_state_dir_relpath: str | None,
    continuation_provider_session_id: str | None,
    stored_provider_session_id: str | None,
    expected_provider_state_dir: tuple[str, str, str],
    expected_provider_session_id: str,
    expected_exact_transcript_match: bool,
) -> None:
    runtime_state_dir = tmp_path / "session-store"
    provider_state_dir = runtime_state_dir.joinpath(*expected_provider_state_dir)
    provider_state_dir.mkdir(parents=True, exist_ok=True)
    if stored_provider_session_id is not None:
        (provider_state_dir / "session_id").write_text(
            f"{stored_provider_session_id}\n",
            encoding="utf-8",
        )

    provider_resume_state: dict[str, object] = {
        "exact_transcript_match": True,
        "provider_state": {
            "session_id": "forged-session",
            "resume_jsonl": "forged",
        },
    }
    if continuation_provider_session_id is not None:
        provider_resume_state["provider_session_id"] = continuation_provider_session_id
    if provider_state_dir_relpath is not None:
        provider_resume_state["provider_state_dir_relpath"] = provider_state_dir_relpath
    continuation = prompt_runtime.Continuation(
        selected_service="opencode",
        selected_model="glm-5.2",
        selected_effort="medium",
        tool_access=contracts_runtime.ToolAccess.no_tools(),
        provider_resume_state=provider_resume_state,
    )

    resolution = provider_state_resolution.resolve_opencode_resumed_session_facts(
        runtime_state_dir=runtime_state_dir,
        session_namespace="fallback",
        continuation=continuation,
        model="glm-5.2",
        effort="medium",
    )

    assert resolution.provider_state_dir == provider_state_dir
    assert resolution.continuation_input_facts == (
        provider_state_resolution.opencode_continuation_input_facts(
            model="glm-5.2",
            effort="medium",
            provider_state_dir=provider_state_dir,
            provider_state_dir_relpath=(
                provider_state_dir_relpath or "implementer/fallback/opencode/"
            ),
            provider_session_id=expected_provider_session_id,
            run_kind=RunKind.RESUME,
            exact_transcript_match=expected_exact_transcript_match,
        )
    )


def test_session_backed_provider_state_resolution_prepares_fresh_claude_new_session_facts_from_empty_state_through_module_interface(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module,
        "_new_provider_session_id",
        lambda: "prepared-claude-session",
    )

    resolution = provider_state_resolution.resolve_claude_new_session_facts(
        runtime_state_dir=tmp_path / "session-store",
        session_namespace="slice424",
        caller_owned_session_store=True,
        model="sonnet",
        effort="medium",
    )

    assert resolution.provider_state_dir == (
        tmp_path / "session-store" / "implementer" / "slice424" / "claude"
    )
    assert resolution.continuation_input_facts == (
        provider_state_resolution.claude_continuation_input_facts(
            model="sonnet",
            effort="medium",
            provider_state_dir=(
                tmp_path / "session-store" / "implementer" / "slice424" / "claude"
            ),
            provider_state_dir_relpath="implementer/slice424/claude/",
            provider_session_id="prepared-claude-session",
            run_kind=RunKind.FRESH,
        )
    )


def test_session_backed_provider_state_resolution_prepares_nonportable_claude_new_session_facts_without_relative_pointer_through_module_interface(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module,
        "_new_provider_session_id",
        lambda: "prepared-claude-session",
    )

    resolution = provider_state_resolution.resolve_claude_new_session_facts(
        runtime_state_dir=tmp_path / "session-store",
        session_namespace="slice424",
        caller_owned_session_store=False,
        model="sonnet",
        effort="medium",
    )

    continuation_input_facts = resolution.continuation_input_facts
    assert continuation_input_facts.provider_state_relpath is None
    assert continuation_input_facts.provider_session_id == (
        provider_state_resolution.PreparedOrRecoveredProviderSessionId(
            value="prepared-claude-session",
            recovered=False,
        )
    )
    assert continuation_input_facts.run_kind is RunKind.FRESH
    assert continuation_input_facts.exact_transcript_match == (
        provider_state_resolution.ExactTranscriptMatch(value=False)
    )
    assert provider_state_resolution.build_session_backed_continuation(
        continuation_input_facts,
        tool_access=contracts_runtime.ToolAccess.no_tools(),
    ) == prompt_runtime.Continuation(
        selected_service="claude",
        selected_model="sonnet",
        selected_effort="medium",
        tool_access=contracts_runtime.ToolAccess.no_tools(),
        provider_resume_state={
            "run_kind": "resume",
            "provider_session_id": "prepared-claude-session",
            "exact_transcript_match": False,
        },
    )


def test_session_backed_provider_state_resolution_recovers_resumable_claude_new_session_facts_from_native_state_through_module_interface(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module,
        "_new_provider_session_id",
        lambda: "prepared-claude-session",
    )
    provider_state_dir = (
        tmp_path / "session-store" / "implementer" / "slice424" / "claude"
    )
    provider_state_dir.mkdir(parents=True, exist_ok=True)
    (provider_state_dir / "projects" / "project.json").parent.mkdir(
        parents=True, exist_ok=True
    )
    (provider_state_dir / "projects" / "project.json").write_text(
        '{"native":"state"}\n',
        encoding="utf-8",
    )

    resolution = provider_state_resolution.resolve_claude_new_session_facts(
        runtime_state_dir=tmp_path / "session-store",
        session_namespace="slice424",
        caller_owned_session_store=True,
        model="sonnet",
        effort="medium",
    )

    assert resolution.provider_state_dir == provider_state_dir
    assert resolution.continuation_input_facts == (
        provider_state_resolution.claude_continuation_input_facts(
            model="sonnet",
            effort="medium",
            provider_state_dir=provider_state_dir,
            provider_state_dir_relpath="implementer/slice424/claude/",
            provider_session_id="prepared-claude-session",
            run_kind=RunKind.RESUME,
        )
    )


def test_session_backed_provider_state_resolution_restores_fresh_claude_resumed_session_facts_from_empty_state_pointer_through_module_interface(
    tmp_path: Path,
) -> None:
    runtime_state_dir = tmp_path / "session-store"

    resolution = provider_state_resolution.resolve_claude_resumed_session_facts(
        runtime_state_dir=runtime_state_dir,
        provider_state_dir_relpath="implementer/slice424/claude/",
        model="sonnet",
        effort="medium",
        provider_session_id="selected-session",
    )

    assert resolution.provider_state_dir == (
        runtime_state_dir / "implementer" / "slice424" / "claude"
    )
    assert resolution.provider_state_dir.is_dir()
    assert resolution.continuation_input_facts == (
        provider_state_resolution.claude_continuation_input_facts(
            model="sonnet",
            effort="medium",
            provider_state_dir=runtime_state_dir
            / "implementer"
            / "slice424"
            / "claude",
            provider_state_dir_relpath="implementer/slice424/claude/",
            provider_session_id="selected-session",
            run_kind=RunKind.FRESH,
        )
    )


def test_session_backed_provider_state_resolution_restores_claude_resumed_session_facts_from_continuation_state_pointer_through_module_interface(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module,
        "_new_provider_session_id",
        lambda: "prepared-claude-session",
    )
    runtime_state_dir = tmp_path / "session-store"
    provider_state_dir = runtime_state_dir / "implementer" / "slice424" / "claude"
    (provider_state_dir / "todos" / "todo.json").parent.mkdir(
        parents=True, exist_ok=True
    )
    (provider_state_dir / "todos" / "todo.json").write_text(
        '{"native":"state"}\n',
        encoding="utf-8",
    )

    resolution = provider_state_resolution.resolve_claude_resumed_session_facts(
        runtime_state_dir=runtime_state_dir,
        provider_state_dir_relpath="implementer/slice424/claude/",
        model="sonnet",
        effort="medium",
        provider_session_id=None,
    )

    assert resolution.provider_state_dir == provider_state_dir
    assert resolution.continuation_input_facts == (
        provider_state_resolution.claude_continuation_input_facts(
            model="sonnet",
            effort="medium",
            provider_state_dir=provider_state_dir,
            provider_state_dir_relpath="implementer/slice424/claude/",
            provider_session_id="prepared-claude-session",
            run_kind=RunKind.RESUME,
        )
    )


def test_session_backed_provider_state_resolution_recovers_codex_new_session_facts_from_rollout_state_through_module_interface(
    tmp_path: Path,
) -> None:
    host_auth_path = tmp_path / "host-home" / ".codex" / "auth.json"
    host_auth_path.parent.mkdir(parents=True, exist_ok=True)
    host_auth_path.write_text('{"token":"host-auth"}\n', encoding="utf-8")

    runtime_state_dir = tmp_path / "session-store"
    rollout_dir = (
        runtime_state_dir / "implementer" / "slice422" / "codex" / "sessions" / "2026"
    )
    rollout_dir.mkdir(parents=True, exist_ok=True)
    (rollout_dir / "rollout-001.jsonl").write_text(
        "{not-json\n"
        '{"type":"turn.completed"}\n'
        '{"type":"session_meta","payload":{"id":"recovered-thread"}}\n',
        encoding="utf-8",
    )

    resolution = provider_state_resolution.resolve_codex_new_session_facts(
        runtime_state_dir=runtime_state_dir,
        session_namespace="slice422",
        caller_owned_session_store=True,
        model="gpt-5.4",
        effort="medium",
        host_auth_path=host_auth_path,
    )

    assert resolution.provider_state_dir == (
        runtime_state_dir / "implementer" / "slice422" / "codex"
    )
    assert resolution.continuation_input_facts == (
        provider_state_resolution.codex_continuation_input_facts(
            model="gpt-5.4",
            effort="medium",
            provider_state_dir=runtime_state_dir / "implementer" / "slice422" / "codex",
            provider_state_dir_relpath="implementer/slice422/codex/",
            provider_session_id="recovered-thread",
            recovered_provider_session_id=True,
            run_kind=RunKind.RESUME,
        )
    )


@pytest.mark.parametrize(
    "rollout_content",
    [
        None,
        "",
        '{not-json\n{"type":"turn.completed"}\n{"type":"session_meta","payload":[]}\n',
        '{"type":"session_meta","payload":{"id":"thread-a"}}\n'
        '{"type":"session_meta","payload":{"id":"thread-b"}}\n',
    ],
)
def test_session_backed_provider_state_resolution_rejects_unrecoverable_codex_rollout_state_through_module_interface(
    tmp_path: Path,
    rollout_content: str | None,
) -> None:
    host_auth_path = tmp_path / "host-home" / ".codex" / "auth.json"
    host_auth_path.parent.mkdir(parents=True, exist_ok=True)
    host_auth_path.write_text('{"token":"host-auth"}\n', encoding="utf-8")

    runtime_state_dir = tmp_path / "session-store"
    rollout_dir = (
        runtime_state_dir / "implementer" / "slice422" / "codex" / "sessions" / "2026"
    )
    rollout_dir.mkdir(parents=True, exist_ok=True)
    if rollout_content is not None:
        (rollout_dir / "rollout-001.jsonl").write_text(
            rollout_content, encoding="utf-8"
        )

    with pytest.raises(
        RuntimeConfigurationError,
        match="Codex continuation is not recoverable from provider state.",
    ):
        provider_state_resolution.resolve_codex_resumed_session_facts(
            runtime_state_dir=runtime_state_dir,
            provider_state_dir_relpath="implementer/slice422/codex/",
            model="gpt-5.4",
            effort="medium",
            provider_session_id=None,
            host_auth_path=host_auth_path,
        )


@pytest.mark.parametrize(
    "rollout_content",
    [
        "",
        '{not-json\n{"type":"turn.completed"}\n{"type":"session_meta","payload":[]}\n',
        '{"type":"session_meta","payload":{"id":"thread-a"}}\n'
        '{"type":"session_meta","payload":{"id":"thread-b"}}\n',
    ],
)
def test_session_backed_provider_state_resolution_rejects_unrecoverable_existing_codex_new_session_rollout_state_through_module_interface(
    tmp_path: Path,
    rollout_content: str,
) -> None:
    host_auth_path = tmp_path / "host-home" / ".codex" / "auth.json"
    host_auth_path.parent.mkdir(parents=True, exist_ok=True)
    host_auth_path.write_text('{"token":"host-auth"}\n', encoding="utf-8")

    runtime_state_dir = tmp_path / "session-store"
    rollout_dir = (
        runtime_state_dir / "implementer" / "slice422" / "codex" / "sessions" / "2026"
    )
    rollout_dir.mkdir(parents=True, exist_ok=True)
    (rollout_dir / "rollout-001.jsonl").write_text(rollout_content, encoding="utf-8")

    with pytest.raises(
        RuntimeConfigurationError,
        match="Codex continuation is not recoverable from provider state.",
    ):
        provider_state_resolution.resolve_codex_new_session_facts(
            runtime_state_dir=runtime_state_dir,
            session_namespace="slice422",
            caller_owned_session_store=True,
            model="gpt-5.4",
            effort="medium",
            host_auth_path=host_auth_path,
        )
