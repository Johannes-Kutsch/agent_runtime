from __future__ import annotations

import re
from pathlib import Path

import pytest

import agent_runtime as runtime
import agent_runtime.contracts as contracts_runtime
import agent_runtime.runtime as prompt_runtime
from agent_runtime._runtime_lifecycle import CancellationToken


def _continuation(
    *,
    tool_access: contracts_runtime.ToolAccess | None = None,
) -> prompt_runtime.Continuation:
    return prompt_runtime.Continuation(
        selected_service="codex",
        selected_model="gpt-5.4",
        selected_effort="medium",
        tool_access=tool_access or contracts_runtime.ToolAccess.no_tools(),
        provider_resume_state={"run_kind": "resume"},
    )


def test_resumed_session_run_request_from_continuation_rejects_tool_access_override() -> (
    None
):
    with pytest.raises(
        TypeError,
        match=re.escape(
            "ResumedSessionRunRequest derives fixed tool access from `continuation` and does not accept `tool_access` or `tool_policy` overrides."
        ),
    ):
        prompt_runtime.ResumedSessionRunRequest(
            prompt="already rendered prompt",
            invocation_dir=Path("/repo"),
            continuation=_continuation(),
            tool_access=contracts_runtime.ToolAccess.workspace_backed(Path("/repo")),
        )


def test_resumed_session_run_request_from_continuation_accepts_minimal_fields() -> None:
    request = prompt_runtime.ResumedSessionRunRequest(
        prompt="already rendered prompt",
        invocation_dir=Path("/repo"),
        continuation=_continuation(),
    )

    assert request.model == "gpt-5.4"
    assert request.effort == "medium"
    assert request.provider_auth is None
    assert request.token is None
    assert request.tool_access == contracts_runtime.ToolAccess.no_tools()
    assert request.invocation_dir == Path("/repo")


def test_resumed_session_run_request_from_continuation_preserves_request_time_inputs() -> (
    None
):
    token = CancellationToken()
    provider_auth = prompt_runtime.ProviderAuth(opencode_api_key="go-key")
    live_events: list[prompt_runtime.AgentEvent] = []

    def on_live_output(event: prompt_runtime.AgentEvent) -> None:
        live_events.append(event)

    request = prompt_runtime.ResumedSessionRunRequest(
        prompt="already rendered prompt",
        invocation_dir=Path("/repo"),
        continuation=_continuation(),
        provider_auth=provider_auth,
        session_store=Path("/state"),
        timeout_seconds=123,
        on_live_output=on_live_output,
        token=token,
    )

    assert request.provider_auth == provider_auth
    assert request.session_store == Path("/state")
    assert request.timeout_seconds == 123
    assert request.on_live_output is on_live_output
    assert request.token is token
    assert live_events == []


def test_resumed_session_run_request_from_continuation_rejects_model_override() -> None:
    with pytest.raises(
        TypeError,
        match=re.escape(
            "ResumedSessionRunRequest got an unexpected keyword argument 'model'"
        ),
    ):
        prompt_runtime.ResumedSessionRunRequest(
            prompt="already rendered prompt",
            invocation_dir=Path("/repo"),
            continuation=_continuation(),
            model="gpt-5.5",
        )


def test_resumed_session_run_request_from_continuation_rejects_effort_override() -> (
    None
):
    with pytest.raises(
        TypeError,
        match=re.escape(
            "ResumedSessionRunRequest got an unexpected keyword argument 'effort'"
        ),
    ):
        prompt_runtime.ResumedSessionRunRequest(
            prompt="already rendered prompt",
            invocation_dir=Path("/repo"),
            continuation=_continuation(),
            effort="high",
        )


@pytest.mark.parametrize("label", ["", "../escape"])
def test_resumed_session_run_request_from_continuation_preserves_empty_internal_session_namespace_and_rejects_unsafe_values(
    label: str,
) -> None:
    if label == "":
        request = prompt_runtime.ResumedSessionRunRequest(
            prompt="already rendered prompt",
            invocation_dir=Path("/repo"),
            _session_namespace=label,
            continuation=_continuation(),
        )

        assert request._session_namespace == ""
        return

    with pytest.raises(ValueError):
        prompt_runtime.ResumedSessionRunRequest(
            prompt="already rendered prompt",
            invocation_dir=Path("/repo"),
            _session_namespace=label,
            continuation=_continuation(),
        )


def test_resumed_session_run_request_from_opaque_continuation_defaults_model_effort_and_tool_access() -> (
    None
):
    request = prompt_runtime.ResumedSessionRunRequest(
        prompt="already rendered prompt",
        invocation_dir=Path("/repo"),
        continuation=_continuation(
            tool_access=contracts_runtime.ToolAccess.workspace_backed(
                Path("/repo"),
                tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
            )
        ),
    )

    assert request.model == "gpt-5.4"
    assert request.effort == "medium"
    assert request.tool_access == contracts_runtime.ToolAccess.workspace_backed(
        Path("/repo"),
        tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
    )
    assert request.tool_policy == runtime.ToolPolicy.NO_FILE_MUTATION


def test_resumed_session_run_request_rejects_conflicting_continuation_and_session_plan() -> (
    None
):
    with pytest.raises(
        TypeError,
        match=re.escape(
            "ResumedSessionRunRequest got an unexpected keyword argument 'session_plan'"
        ),
    ):
        prompt_runtime.ResumedSessionRunRequest(
            prompt="already rendered prompt",
            invocation_dir=Path("/repo"),
            continuation=_continuation(),
            session_plan=object(),
        )


def test_resumed_session_run_request_rejects_conflicting_tool_access_and_tool_policy() -> (
    None
):
    with pytest.raises(
        TypeError,
        match=re.escape(
            "ResumedSessionRunRequest derives fixed tool access from `continuation` and does not accept `tool_access` or `tool_policy` overrides."
        ),
    ):
        prompt_runtime.ResumedSessionRunRequest(
            prompt="already rendered prompt",
            invocation_dir=Path("/repo"),
            continuation=_continuation(),
            tool_access=contracts_runtime.ToolAccess.no_tools(),
        )


def test_resumed_session_run_request_carries_workspace_backed_tool_access() -> None:
    tool_access = contracts_runtime.ToolAccess.workspace_backed(
        Path("/repo"),
        tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
    )

    request = prompt_runtime.ResumedSessionRunRequest(
        prompt="already rendered prompt",
        invocation_dir=Path("/repo"),
        continuation=_continuation(tool_access=tool_access),
    )

    assert request.tool_access == tool_access
    assert request.tool_access.workspace == Path("/repo")


def test_resumed_session_run_request_rejects_explicit_no_tools_tool_access() -> None:
    with pytest.raises(
        TypeError,
        match=re.escape(
            "ResumedSessionRunRequest derives fixed tool access from `continuation` and does not accept `tool_access` or `tool_policy` overrides."
        ),
    ):
        prompt_runtime.ResumedSessionRunRequest(
            prompt="already rendered prompt",
            invocation_dir=Path("/repo"),
            continuation=_continuation(),
            tool_access=contracts_runtime.ToolAccess.no_tools(),
        )


def test_resumed_session_run_request_from_continuation_rejects_tool_policy_override() -> (
    None
):
    with pytest.raises(
        TypeError,
        match=re.escape(
            "ResumedSessionRunRequest got an unexpected keyword argument 'tool_policy'"
        ),
    ):
        prompt_runtime.ResumedSessionRunRequest(
            prompt="already rendered prompt",
            invocation_dir=Path("/repo"),
            continuation=_continuation(),
            tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
        )


def test_resumed_session_run_request_from_continuation_rejects_workspace_backed_tool_access_for_other_invocation_dir() -> (
    None
):
    with pytest.raises(
        ValueError,
        match=re.escape(
            "ResumedSessionRunRequest workspace-backed tool access requires invocation_dir /repo, got /other."
        ),
    ):
        prompt_runtime.ResumedSessionRunRequest(
            prompt="already rendered prompt",
            invocation_dir=Path("/other"),
            continuation=_continuation(
                tool_access=contracts_runtime.ToolAccess.workspace_backed(
                    Path("/repo"),
                    tool_policy=runtime.ToolPolicy.UNRESTRICTED,
                )
            ),
        )


def test_resumed_session_run_request_rejects_missing_continuation() -> None:
    with pytest.raises(
        TypeError,
        match=re.escape("ResumedSessionRunRequest requires a `continuation` value."),
    ):
        prompt_runtime.ResumedSessionRunRequest(
            prompt="already rendered prompt",
            invocation_dir=Path("/repo"),
        )


@pytest.mark.parametrize(
    ("continuation", "message"),
    [
        (
            prompt_runtime.Continuation(serialized="{not-json"),
            "Continuation data is not valid JSON.",
        ),
        (
            prompt_runtime.Continuation(
                serialized='{"effort":"medium","model":"gpt-5.4","provider_resume_state":{"run_kind":"resume"},"service_name":"codex","tool_access":[]}'
            ),
            "Continuation data is malformed.",
        ),
    ],
)
def test_resumed_session_run_request_surfaces_malformed_continuation_through_runtime_configuration_error(
    continuation: prompt_runtime.Continuation,
    message: str,
) -> None:
    with pytest.raises(
        runtime.RuntimeConfigurationError,
        match=re.escape(message),
    ):
        prompt_runtime.ResumedSessionRunRequest(
            prompt="already rendered prompt",
            invocation_dir=Path("/repo"),
            continuation=continuation,
        )
