from pathlib import Path

import pytest

import agent_runtime._session_backed_provider_state_resolution as provider_state_resolution
import agent_runtime.contracts as contracts_runtime
import agent_runtime.runtime as prompt_runtime
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
