from __future__ import annotations

import inspect
import re
from collections.abc import Callable
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, cast

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


def test_ephemeral_run_request_only_accepts_minimal_ephemeral_fields(
    provider_selection_factory: Callable[..., runtime.ProviderSelection],
) -> None:
    token = CancellationToken()
    request = prompt_runtime.EphemeralRunRequest(
        prompt="already rendered prompt",
        invocation_dir=Path("/repo"),
        provider_selection=provider_selection_factory(
            service="codex",
            auth=runtime.ProviderAuth(opencode_api_key="go-key"),
        ),
        tool_access=contracts_runtime.ToolAccess.no_tools(),
        token=token,
    )

    assert request.provider_selection.auth == runtime.ProviderAuth(
        opencode_api_key="go-key"
    )
    assert request.token is token
    for field_name in ("role", "logs_dir", "usage_limit_scope", "session_namespace"):
        with pytest.raises(AttributeError, match=field_name):
            getattr(request, field_name)
    assert tuple(inspect.signature(prompt_runtime.EphemeralRunRequest).parameters) == (
        "prompt",
        "invocation_dir",
        "provider_selection",
        "tool_policy",
        "timeout_seconds",
        "token",
        "on_live_output",
    )


def test_resumed_session_run_request_has_minimal_public_signature() -> None:
    assert tuple(
        inspect.signature(prompt_runtime.ResumedSessionRunRequest).parameters
    ) == (
        "prompt",
        "invocation_dir",
        "continuation",
        "provider_auth",
        "session_store",
        "timeout_seconds",
        "on_live_output",
        "token",
    )


def test_new_session_run_request_signature_exposes_live_output_observer() -> None:
    parameters = inspect.signature(prompt_runtime.NewSessionRunRequest).parameters

    assert "on_live_output" in parameters
    assert "timeout_seconds" in parameters
    assert "session_store" in parameters
    assert "provider_session_adapter" not in parameters


@pytest.mark.parametrize(
    ("request_factory", "expected_request_type"),
    [
        (
            lambda: prompt_runtime.NewSessionRunRequest(
                prompt="already rendered prompt",
                invocation_dir=Path("/repo"),
                provider_selection=runtime.ProviderSelection(
                    service="codex",
                    model="gpt-5.4",
                    effort="medium",
                ),
                tool_policy=runtime.ToolPolicy.NONE,
                session_store=Path("/state"),
            ),
            "NewSessionRunRequest",
        ),
        (
            lambda: prompt_runtime.ResumedSessionRunRequest(
                prompt="already rendered prompt",
                invocation_dir=Path("/repo"),
                continuation=_continuation(),
                provider_auth=runtime.ProviderAuth(opencode_api_key="go-key"),
                session_store=Path("/state"),
            ),
            "ResumedSessionRunRequest",
        ),
    ],
)
def test_session_backed_run_requests_accept_public_session_store_input(
    request_factory: Callable[[], object],
    expected_request_type: str,
) -> None:
    request = cast(Any, request_factory())

    assert type(request).__name__ == expected_request_type
    assert request.session_store == Path("/state")
    assert request._runtime_state_dir == Path("/state")


def test_new_invocation_requests_take_provider_auth_from_provider_selection() -> None:
    provider_auth = runtime.ProviderAuth(opencode_api_key="go-key")
    provider_selection = runtime.ProviderSelection(
        service="opencode",
        model="gpt-5.4",
        effort="medium",
        auth=provider_auth,
    )

    ephemeral_request = prompt_runtime.EphemeralRunRequest(
        prompt="already rendered prompt",
        invocation_dir=Path("/repo"),
        provider_selection=provider_selection,
        tool_policy=runtime.ToolPolicy.NONE,
    )
    new_session_request = prompt_runtime.NewSessionRunRequest(
        prompt="already rendered prompt",
        invocation_dir=Path("/repo"),
        provider_selection=provider_selection,
        tool_policy=runtime.ToolPolicy.NONE,
    )

    assert ephemeral_request.provider_selection.auth == provider_auth
    assert new_session_request.provider_selection.auth == provider_auth


def test_public_root_and_runtime_modules_expose_provider_selection_only() -> None:
    assert runtime.ProviderSelection is prompt_runtime.ProviderSelection
    with pytest.raises(AttributeError):
        runtime.StageSelection
    with pytest.raises(AttributeError):
        prompt_runtime.StageSelection


def test_provider_value_objects_redact_credential_values_in_textual_representation() -> (
    None
):
    provider_auth = runtime.ProviderAuth(
        claude_code_oauth_token="claude-secret",
        opencode_api_key="opencode-secret",
    )
    provider_selection = runtime.ProviderSelection(
        service="opencode",
        model="gpt-5.4",
        effort="medium",
        auth=provider_auth,
    )

    for rendered in (
        repr(provider_auth),
        str(provider_auth),
        repr(provider_selection),
        str(provider_selection),
    ):
        assert "claude-secret" not in rendered
        assert "opencode-secret" not in rendered


@pytest.mark.parametrize(
    ("request_factory", "request_type"),
    [
        (
            lambda provider_selection: prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                invocation_dir=Path("/repo"),
                provider_selection=provider_selection,
                tool_policy=runtime.ToolPolicy.NONE,
            ),
            prompt_runtime.EphemeralRunRequest,
        ),
        (
            lambda provider_selection: prompt_runtime.NewSessionRunRequest(
                prompt="already rendered prompt",
                invocation_dir=Path("/repo"),
                provider_selection=provider_selection,
                tool_policy=runtime.ToolPolicy.NONE,
            ),
            prompt_runtime.NewSessionRunRequest,
        ),
    ],
)
def test_runtime_lifecycle_requests_use_provider_selection_public_field(
    provider_selection_factory: Callable[..., runtime.ProviderSelection],
    request_factory: Callable[[runtime.ProviderSelection], object],
    request_type: type[object],
) -> None:
    provider_selection = provider_selection_factory(service="codex")
    request = cast(Any, request_factory(provider_selection))

    assert request.provider_selection == provider_selection
    assert "provider_selection" in inspect.signature(request_type).parameters
    assert "stage" not in inspect.signature(request_type).parameters
    assert "override" not in inspect.signature(request_type).parameters


@pytest.mark.parametrize(
    ("request_factory", "request_name"),
    [
        (
            lambda on_live_output, provider_selection_factory, tmp_path: (
                prompt_runtime.EphemeralRunRequest(
                    prompt="already rendered prompt",
                    invocation_dir=tmp_path,
                    provider_selection=provider_selection_factory(service="codex"),
                    on_live_output=on_live_output,
                    tool_access=contracts_runtime.ToolAccess.no_tools(),
                )
            ),
            "EphemeralRunRequest",
        ),
        (
            lambda on_live_output, provider_selection_factory, tmp_path: (
                prompt_runtime.NewSessionRunRequest(
                    prompt="already rendered prompt",
                    invocation_dir=tmp_path,
                    provider_selection=provider_selection_factory(service="codex"),
                    on_live_output=on_live_output,
                    tool_access=contracts_runtime.ToolAccess.no_tools(),
                )
            ),
            "NewSessionRunRequest",
        ),
        (
            lambda on_live_output, _provider_selection_factory, tmp_path: (
                prompt_runtime.ResumedSessionRunRequest(
                    prompt="already rendered prompt",
                    invocation_dir=tmp_path,
                    continuation=_continuation(),
                    on_live_output=on_live_output,
                )
            ),
            "ResumedSessionRunRequest",
        ),
    ],
)
def test_runtime_lifecycle_request_values_accept_live_output_observer(
    provider_selection_factory: Callable[..., runtime.ProviderSelection],
    request_factory: Any,
    request_name: str,
    tmp_path: Path,
) -> None:
    observed: list[object] = []

    def on_live_output(value: object) -> None:
        observed.append(value)

    request = request_factory(on_live_output, provider_selection_factory, tmp_path)

    assert request.on_live_output is on_live_output
    assert request_name in request.__class__.__name__
    assert observed == []


@pytest.mark.parametrize(
    ("request_factory", "request_name"),
    [
        (prompt_runtime.EphemeralRunRequest, "EphemeralRunRequest"),
        (prompt_runtime.NewSessionRunRequest, "NewSessionRunRequest"),
    ],
)
@pytest.mark.parametrize("removed_name", ["stage", "override"])
def test_public_lifecycle_requests_reject_removed_request_selection_names(
    request_factory: type[object],
    request_name: str,
    removed_name: str,
) -> None:
    with pytest.raises(
        TypeError,
        match=f"{request_name} got an unexpected keyword argument '{removed_name}'.",
    ):
        kwargs: dict[str, Any] = {
            "prompt": "already rendered prompt",
            "invocation_dir": Path("/repo"),
            "tool_policy": runtime.ToolPolicy.NONE,
            removed_name: runtime.ProviderSelection(
                service="codex",
                model="gpt-5.4",
                effort="medium",
            ),
        }
        cast(Any, request_factory)(**kwargs)


@pytest.mark.parametrize(
    ("request_factory", "request_name"),
    [
        (
            lambda unexpected_name, unexpected_value: (
                prompt_runtime.EphemeralRunRequest(
                    prompt="already rendered prompt",
                    invocation_dir=Path("/repo"),
                    provider_selection=runtime.ProviderSelection(
                        service="codex",
                        model="gpt-5.4",
                        effort="medium",
                    ),
                    tool_policy=runtime.ToolPolicy.NONE,
                    **{unexpected_name: unexpected_value},
                )
            ),
            "EphemeralRunRequest",
        ),
        (
            lambda unexpected_name, unexpected_value: (
                prompt_runtime.NewSessionRunRequest(
                    prompt="already rendered prompt",
                    invocation_dir=Path("/repo"),
                    provider_selection=runtime.ProviderSelection(
                        service="codex",
                        model="gpt-5.4",
                        effort="medium",
                    ),
                    tool_policy=runtime.ToolPolicy.NONE,
                    **{unexpected_name: unexpected_value},
                )
            ),
            "NewSessionRunRequest",
        ),
        (
            lambda unexpected_name, unexpected_value: (
                prompt_runtime.ResumedSessionRunRequest(
                    prompt="already rendered prompt",
                    invocation_dir=Path("/repo"),
                    continuation=_continuation(),
                    **{unexpected_name: unexpected_value},
                )
            ),
            "ResumedSessionRunRequest",
        ),
    ],
)
@pytest.mark.parametrize(
    ("unexpected_name", "unexpected_value"),
    [
        ("logs_dir", Path("/tmp/runtime-logs")),
        ("log_name", "implementer"),
    ],
)
def test_ordinary_runtime_requests_reject_runtime_managed_inputs(
    request_factory: Callable[[str, object], object],
    request_name: str,
    unexpected_name: str,
    unexpected_value: object,
) -> None:
    with pytest.raises(
        TypeError,
        match=f"{request_name} got an unexpected keyword argument '{unexpected_name}'.",
    ):
        request_factory(unexpected_name, unexpected_value)


def test_lifecycle_requests_derive_workspace_backed_tool_access_from_tool_policy() -> (
    None
):
    assert prompt_runtime.EphemeralRunRequest(
        prompt="already rendered prompt",
        invocation_dir=Path("/repo"),
        provider_selection=runtime.ProviderSelection(
            service="codex",
            model="gpt-5.4",
            effort="medium",
        ),
        tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
    ).tool_access == contracts_runtime.ToolAccess.workspace_backed(
        Path("/repo"),
        tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
    )

    assert prompt_runtime.NewSessionRunRequest(
        prompt="already rendered prompt",
        invocation_dir=Path("/repo"),
        provider_selection=runtime.ProviderSelection(
            service="codex",
            model="gpt-5.4",
            effort="medium",
        ),
        tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
    ).tool_access == contracts_runtime.ToolAccess.workspace_backed(
        Path("/repo"),
        tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
    )

    assert prompt_runtime.ResumedSessionRunRequest(
        prompt="already rendered prompt",
        invocation_dir=Path("/repo"),
        continuation=_continuation(
            tool_access=contracts_runtime.ToolAccess.workspace_backed(
                Path("/repo"),
                tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
            )
        ),
    ).tool_access == contracts_runtime.ToolAccess.workspace_backed(
        Path("/repo"),
        tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
    )


def test_lifecycle_requests_keep_runtime_managed_compatibility_fields_internal_to_dataclass_surface() -> (
    None
):
    new_session_request = prompt_runtime.NewSessionRunRequest(
        prompt="already rendered prompt",
        invocation_dir=Path("/repo"),
        provider_selection=runtime.ProviderSelection(
            service="codex",
            model="gpt-5.4",
            effort="high",
        ),
        tool_access=contracts_runtime.ToolAccess.no_tools(),
        runtime_state_dir=Path("/state"),
        session_namespace="main",
    )
    resumed_request = prompt_runtime.ResumedSessionRunRequest(
        prompt="already rendered prompt",
        invocation_dir=Path("/repo"),
        continuation=_continuation(),
        runtime_state_dir=Path("/state"),
        session_namespace="main",
    )

    for request in (new_session_request, resumed_request):
        assert request._runtime_state_dir == Path("/state")
        assert request._session_namespace == "main"
        assert "_runtime_state_dir" not in repr(request)
        assert "_session_namespace" not in repr(request)
        assert "_runtime_state_dir" not in {field.name for field in fields(request)}
        assert "_session_namespace" not in {field.name for field in fields(request)}
        assert "_runtime_state_dir" not in asdict(request)
        assert "_session_namespace" not in asdict(request)


@pytest.mark.parametrize(
    ("request_factory", "expected_message"),
    [
        (
            lambda: prompt_runtime.NewSessionRunRequest(
                prompt="already rendered prompt",
                invocation_dir=Path("/repo"),
                provider_selection=runtime.ProviderSelection(
                    service="codex",
                    model="gpt-5.4",
                    effort="medium",
                ),
                tool_policy=runtime.ToolPolicy.NONE,
                session_store=Path("/state"),
                runtime_state_dir=Path("/other-state"),
            ),
            "NewSessionRunRequest received conflicting `runtime_state_dir` and `session_store` values.",
        ),
        (
            lambda: prompt_runtime.ResumedSessionRunRequest(
                prompt="already rendered prompt",
                invocation_dir=Path("/repo"),
                continuation=_continuation(),
                session_store=Path("/state"),
                runtime_state_dir=Path("/other-state"),
            ),
            "ResumedSessionRunRequest received conflicting `runtime_state_dir` and `session_store` values.",
        ),
    ],
)
def test_session_backed_run_request_rejects_conflicting_session_store_inputs(
    request_factory: Callable[[], object],
    expected_message: str,
) -> None:
    with pytest.raises(TypeError, match=re.escape(expected_message)):
        request_factory()


def test_lifecycle_request_construction_requires_explicit_tool_policy() -> None:
    with pytest.raises(
        TypeError,
        match=re.escape(
            "EphemeralRunRequest requires an explicit `tool_policy` value."
        ),
    ):
        prompt_runtime.EphemeralRunRequest(
            prompt="already rendered prompt",
            invocation_dir=Path("/repo"),
            provider_selection=runtime.ProviderSelection(
                service="codex",
                model="gpt-5.4",
                effort="medium",
            ),
        )

    with pytest.raises(
        TypeError,
        match=re.escape(
            "NewSessionRunRequest requires an explicit `tool_policy` value."
        ),
    ):
        prompt_runtime.NewSessionRunRequest(
            prompt="already rendered prompt",
            invocation_dir=Path("/repo"),
            provider_selection=runtime.ProviderSelection(
                service="codex",
                model="gpt-5.4",
                effort="medium",
            ),
        )


def test_lifecycle_request_construction_rejects_workspace_backed_tool_access_for_other_invocation_dir() -> (
    None
):
    with pytest.raises(
        ValueError,
        match=re.escape(
            "EphemeralRunRequest workspace-backed tool access requires invocation_dir /other, got /repo."
        ),
    ):
        prompt_runtime.EphemeralRunRequest(
            prompt="already rendered prompt",
            invocation_dir=Path("/repo"),
            provider_selection=runtime.ProviderSelection(
                service="codex",
                model="gpt-5.4",
                effort="medium",
            ),
            tool_access=contracts_runtime.ToolAccess.workspace_backed(
                Path("/other"),
                tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
            ),
        )


def test_resumed_session_run_request_keeps_path_invocation_dir() -> None:
    request = prompt_runtime.ResumedSessionRunRequest(
        prompt="already rendered prompt",
        invocation_dir=Path("/repo"),
        continuation=_continuation(),
    )

    assert request.invocation_dir == Path("/repo")
    assert request.mount_path == Path("/repo")


@pytest.mark.parametrize(
    "request_factory",
    [
        lambda: prompt_runtime.EphemeralRunRequest(
            prompt="already rendered prompt",
            worktree=Path("/repo"),
            provider_selection=runtime.ProviderSelection(
                service="codex",
                model="gpt-5.4",
                effort="medium",
            ),
            tool_access=contracts_runtime.ToolAccess.no_tools(),
        ),
        lambda: prompt_runtime.NewSessionRunRequest(
            prompt="already rendered prompt",
            worktree=Path("/repo"),
            provider_selection=runtime.ProviderSelection(
                service="codex",
                model="gpt-5.4",
                effort="medium",
            ),
            tool_access=contracts_runtime.ToolAccess.no_tools(),
        ),
        lambda: prompt_runtime.ResumedSessionRunRequest(
            prompt="already rendered prompt",
            worktree=Path("/repo"),
            continuation=_continuation(),
        ),
    ],
)
def test_lifecycle_request_construction_keeps_legacy_worktree_kwarg_outside_public_surface(
    request_factory: Callable[[], object],
) -> None:
    request = cast(Any, request_factory())

    assert request.invocation_dir == Path("/repo")


@pytest.mark.parametrize(
    "request_factory",
    [
        lambda: prompt_runtime.EphemeralRunRequest(
            prompt="already rendered prompt",
            invocation_dir=Path("/repo"),
            worktree=Path("/other"),
            provider_selection=runtime.ProviderSelection(
                service="codex",
                model="gpt-5.4",
                effort="medium",
            ),
            tool_access=contracts_runtime.ToolAccess.no_tools(),
        ),
        lambda: prompt_runtime.NewSessionRunRequest(
            prompt="already rendered prompt",
            invocation_dir=Path("/repo"),
            worktree=Path("/other"),
            provider_selection=runtime.ProviderSelection(
                service="codex",
                model="gpt-5.4",
                effort="medium",
            ),
            tool_access=contracts_runtime.ToolAccess.no_tools(),
        ),
        lambda: prompt_runtime.ResumedSessionRunRequest(
            prompt="already rendered prompt",
            invocation_dir=Path("/repo"),
            worktree=Path("/other"),
            continuation=_continuation(),
        ),
    ],
)
def test_lifecycle_request_construction_rejects_conflicting_invocation_dir_and_legacy_worktree(
    request_factory: Callable[[], object],
) -> None:
    with pytest.raises(
        TypeError,
        match=re.escape("received conflicting `invocation_dir` and `worktree` values."),
    ):
        request_factory()


def test_runtime_public_surface_keeps_request_normalization_module_private() -> None:
    assert "_request_normalization" not in runtime.__all__
    assert "_request_normalization" not in prompt_runtime.__all__
