from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, cast

import pytest

import agent_runtime as runtime
import agent_runtime._builtin_provider_rendering as built_in_provider_rendering
import agent_runtime.runtime as runtime_module
from agent_runtime._runtime_lifecycle import ProviderAuth
from agent_runtime.contracts import ToolAccess
from agent_runtime.session import RunKind


@pytest.mark.parametrize(
    ("module_name", "removed_name"),
    [
        ("agent_runtime", "BuiltInProviderRenderRequest"),
        ("agent_runtime", "BuiltInProviderRenderedInvocation"),
        ("agent_runtime", "BuiltInProviderSelectionFacts"),
        ("agent_runtime", "BuiltInProviderHostFacts"),
        ("agent_runtime.runtime", "BuiltInProviderRenderRequest"),
        ("agent_runtime.runtime", "BuiltInProviderRenderedInvocation"),
        ("agent_runtime.runtime", "BuiltInProviderSelectionFacts"),
        ("agent_runtime.runtime", "BuiltInProviderHostFacts"),
    ],
)
def test_built_in_provider_rendering_values_stay_off_runtime_public_surface(
    module_name: str,
    removed_name: str,
) -> None:
    with pytest.raises(ImportError):
        exec(f"from {module_name} import {removed_name}", {}, {})

    imported_module = runtime if module_name == "agent_runtime" else runtime_module
    assert not hasattr(imported_module, removed_name)


def test_built_in_provider_render_request_preserves_optional_rendering_facts() -> None:
    request = built_in_provider_rendering.BuiltInProviderRenderRequest(
        provider_selection=built_in_provider_rendering.BuiltInProviderSelectionFacts(
            service="claude",
            model="sonnet",
            effort="medium",
        ),
        run_kind=RunKind.RESUME,
        tool_access=ToolAccess.workspace_backed(Path("/tmp/invocation")),
        auth=ProviderAuth(claude_code_oauth_token="token"),
        invocation_dir=Path("/tmp/invocation"),
        provider_state_dir=Path("/tmp/provider-state"),
        provider_session_id="session-123",
        host_facts=built_in_provider_rendering.BuiltInProviderHostFacts(
            os_name="posix",
            environment={"HOME": "/tmp/home"},
        ),
    )

    assert request.provider_selection.service == "claude"
    assert request.provider_selection.model == "sonnet"
    assert request.provider_selection.effort == "medium"
    assert request.run_kind is RunKind.RESUME
    assert request.tool_access == ToolAccess.workspace_backed(Path("/tmp/invocation"))
    assert request.auth == ProviderAuth(claude_code_oauth_token="token")
    assert request.invocation_dir == Path("/tmp/invocation")
    assert request.provider_state_dir == Path("/tmp/provider-state")
    assert request.provider_session_id == "session-123"
    assert request.host_facts == built_in_provider_rendering.BuiltInProviderHostFacts(
        os_name="posix",
        environment={"HOME": "/tmp/home"},
    )


def test_built_in_provider_rendering_values_freeze_mutable_inputs() -> None:
    host_environment = {"HOME": "/tmp/home"}
    invocation_environment = {"PATH": "/usr/bin"}

    host_facts = built_in_provider_rendering.BuiltInProviderHostFacts(
        os_name="posix",
        environment=host_environment,
    )
    rendered_invocation = built_in_provider_rendering.BuiltInProviderRenderedInvocation(
        canonical_argv=cast(tuple[str, ...], ["provider", "--run"]),
        legacy_command_text="provider --run",
        environment=invocation_environment,
        prompt_path=Path("/tmp/prompt.txt"),
        prompt_cleanup_choice=(
            built_in_provider_rendering.PromptCleanupChoice.DELETE_AFTER_INVOCATION
        ),
        prompt_transport_preference=(
            built_in_provider_rendering.PromptTransportPreference.PROMPT_FILE
        ),
        provider_session_id_placement=(
            built_in_provider_rendering.ProviderSessionIdPlacement.CLI_FLAG
        ),
    )

    host_environment["HOME"] = "/mutated"
    invocation_environment["PATH"] = "/mutated"

    assert host_facts.environment == {"HOME": "/tmp/home"}
    assert rendered_invocation.canonical_argv == ("provider", "--run")
    assert rendered_invocation.environment == {"PATH": "/usr/bin"}

    with pytest.raises(FrozenInstanceError):
        setattr(host_facts, "os_name", "nt")
    with pytest.raises(TypeError):
        cast(Any, rendered_invocation.environment)["NEW_VAR"] = "value"


def test_built_in_provider_rendering_values_allow_missing_optional_facts() -> None:
    request = built_in_provider_rendering.BuiltInProviderRenderRequest(
        provider_selection=built_in_provider_rendering.BuiltInProviderSelectionFacts(
            service="codex",
            model="gpt-5-codex",
            effort="high",
        ),
        run_kind=RunKind.FRESH,
        tool_access=ToolAccess.no_tools(),
        auth=None,
        invocation_dir=Path("/tmp/invocation"),
    )
    rendered_invocation = built_in_provider_rendering.BuiltInProviderRenderedInvocation(
        canonical_argv=("provider", "--run"),
        legacy_command_text=None,
        environment={},
        prompt_path=None,
        prompt_cleanup_choice=built_in_provider_rendering.PromptCleanupChoice.KEEP,
        prompt_transport_preference=(
            built_in_provider_rendering.PromptTransportPreference.STDIN
        ),
        provider_session_id_placement=(
            built_in_provider_rendering.ProviderSessionIdPlacement.NONE
        ),
    )

    assert request.provider_state_dir is None
    assert request.provider_session_id is None
    assert request.host_facts is None
    assert rendered_invocation.legacy_command_text is None
    assert rendered_invocation.prompt_path is None
