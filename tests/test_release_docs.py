from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_repo_doc(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def test_readme_shows_ordinary_runtime_consumer_session_examples() -> None:
    readme = _read_repo_doc("README.md")

    assert "### Ephemeral Execution" in readme
    assert "### New-Session Execution" in readme
    assert "### Resumed-Session Execution" in readme
    assert "tool_policy=ToolPolicy.NONE" in readme
    assert "tool_policy=ToolPolicy.NO_FILE_MUTATION" in readme

    for legacy_term in (
        "InvocationRole",
        "RuntimeStateDir",
        "RuntimeLogsDir",
        "UsageLimitScope",
        "SessionNamespace",
        "ToolAccess",
        "worktree",
    ):
        assert legacy_term not in readme


def test_public_api_documents_ordinary_runtime_consumer_surface() -> None:
    public_api = _read_repo_doc("docs/public-api.md")

    assert "Invocation Directory" in public_api
    assert "`ToolPolicy`" in public_api
    assert "Usage-limit outcomes expose provider and service facts" in public_api
    assert "`InvocationRecord` is structured runtime output" in public_api
    assert "For callers, the ordinary surface remains `RuntimeClient`" in public_api
    assert "managed worktrees" not in public_api


def test_public_api_documents_live_runtime_output_consumer_surface() -> None:
    public_api = _read_repo_doc("docs/public-api.md")

    assert "### Live Runtime Output" in public_api
    assert "AgentMessageTurn(\n    text: str,\n    service_name: str," in public_api
    assert (
        "available from both `agent_runtime` and `agent_runtime.runtime`" in public_api
    )
    assert "on_live_output: Callable[[AgentMessageTurn], None] | None = None" in (
        public_api
    )
    assert "`on_live_output` is also available for Start Session Run." in public_api
    assert "`on_live_output` is also available for Resume Session Run." in public_api
    assert (
        "Live Runtime Output is agent-message-turn observation, not token-by-token"
        " streaming."
    ) in public_api
    assert (
        "It is provider-neutral and does not expose raw provider stdout, raw JSON,"
        " or provider DTO events."
    ) in public_api
    assert (
        "Completed runtime output and `RuntimeOutcome` interruption semantics remain"
        " authoritative and unchanged."
    ) in public_api
    assert (
        "`on_live_output` callbacks are synchronous and notification-only. Callback"
        " exceptions are propagated to the caller as consumer failures."
    ) in public_api
    assert (
        "Consumers own backpressure policy, queueing, display formatting,"
        " redaction, and persistence for observed turns."
    ) in public_api
    assert "pycastle (conceptual, not a runtime dependency)" in public_api
    assert (
        "requires consumers to apply their own protocol parsing and downstream policy."
    ) in public_api


def test_context_and_adrs_keep_legacy_runtime_storage_historical_only() -> None:
    context = _read_repo_doc("CONTEXT.md")
    adr_0005 = _read_repo_doc("docs/adr/0005-runtime-public-surface-narrowing.md")
    adr_0010 = _read_repo_doc("docs/adr/0010-portable-continuations.md")

    assert "| `ToolAccess` | Retired target vocabulary for the public API;" in context
    assert (
        "| `RuntimeStateDir` | Transitional caller-supplied directory root" in context
    )
    assert "| `RuntimeLogsDir` | Transitional caller-supplied directory root" in context
    assert (
        "| `SessionNamespace` | Transitional secondary label previously used" in context
    )
    assert "managed worktrees" not in context
    assert "execution-directory management" in context
    assert "This ADR is historical" in adr_0005
    assert "This supersedes ADR 0009's requirement" in adr_0010
    assert "and `ToolPolicy` belong in result metadata" in adr_0010
