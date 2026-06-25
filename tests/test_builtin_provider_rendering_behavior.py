from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, cast

import pytest

import agent_runtime as runtime
import agent_runtime._builtin_provider_rendering as built_in_provider_rendering
import agent_runtime.runtime as runtime_module
from agent_runtime._runtime_lifecycle import ProviderAuth
from agent_runtime.contracts import ToolAccess, ToolPolicyProfile
from agent_runtime.errors import AgentCredentialFailureError, RuntimeConfigurationError
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


def test_render_claude_invocation_returns_canonical_argv_and_compatibility_command() -> (
    None
):
    invocation_dir = Path("/tmp/invocation")

    rendered_invocation = (
        built_in_provider_rendering.render_built_in_provider_invocation(
            built_in_provider_rendering.BuiltInProviderRenderRequest(
                provider_selection=(
                    built_in_provider_rendering.BuiltInProviderSelectionFacts(
                        service="claude",
                        model="sonnet",
                        effort="medium",
                    )
                ),
                run_kind=RunKind.FRESH,
                tool_access=ToolAccess.workspace_backed(invocation_dir),
                auth=ProviderAuth(claude_code_oauth_token="token"),
                invocation_dir=invocation_dir,
            )
        )
    )

    assert (
        rendered_invocation
        == built_in_provider_rendering.BuiltInProviderRenderedInvocation(
            canonical_argv=(
                "claude",
                "--verbose",
                "--dangerously-skip-permissions",
                "--output-format",
                "stream-json",
                "-p",
                "-",
                "--disable-slash-commands",
                "--exclude-dynamic-system-prompt-sections",
                "--strict-mcp-config",
                "--mcp-config",
                '{"mcpServers":{}}',
                "--model",
                "sonnet",
                "--effort",
                "medium",
            ),
            legacy_command_text=(
                "claude --verbose --dangerously-skip-permissions --output-format "
                "stream-json -p - --disable-slash-commands "
                "--exclude-dynamic-system-prompt-sections --strict-mcp-config "
                "--mcp-config '{\"mcpServers\":{}}' --model sonnet --effort medium "
                "< /tmp/invocation/.provider_prompt"
            ),
            environment={"CLAUDE_CODE_OAUTH_TOKEN": "token"},
            prompt_path=invocation_dir / ".provider_prompt",
            prompt_cleanup_choice=(
                built_in_provider_rendering.PromptCleanupChoice.DELETE_AFTER_INVOCATION
            ),
            prompt_transport_preference=(
                built_in_provider_rendering.PromptTransportPreference.STDIN
            ),
            provider_session_id_placement=(
                built_in_provider_rendering.ProviderSessionIdPlacement.NONE
            ),
            prefer_argv=True,
        )
    )


def test_render_claude_invocation_uses_provider_prompt_path_and_claude_only_environment() -> (
    None
):
    invocation_dir = Path("/tmp/invocation")
    rendered_invocation = built_in_provider_rendering.render_built_in_provider_invocation(
        built_in_provider_rendering.BuiltInProviderRenderRequest(
            provider_selection=built_in_provider_rendering.BuiltInProviderSelectionFacts(
                service="claude",
                model="sonnet",
                effort="medium",
            ),
            run_kind=RunKind.FRESH,
            tool_access=ToolAccess.workspace_backed(invocation_dir),
            auth=ProviderAuth(claude_code_oauth_token="token"),
            invocation_dir=invocation_dir,
            provider_state_dir=Path("/tmp/provider-state"),
            host_facts=built_in_provider_rendering.BuiltInProviderHostFacts(
                os_name="posix",
                environment={
                    "HOME": "/tmp/home",
                    "PATH": "/usr/bin",
                    "CLAUDE_CODE_OAUTH_TOKEN": "host-token",
                },
            ),
        ),
    )

    assert rendered_invocation.prompt_path == invocation_dir / ".provider_prompt"
    assert (
        rendered_invocation.prompt_cleanup_choice
        is built_in_provider_rendering.PromptCleanupChoice.DELETE_AFTER_INVOCATION
    )
    assert rendered_invocation.environment == {
        "CLAUDE_CODE_OAUTH_TOKEN": "token",
        "CLAUDE_CONFIG_DIR": "/tmp/provider-state",
    }


def test_render_claude_invocation_maps_tool_policy_and_custom_profile_flags() -> None:
    invocation_dir = Path("/tmp/invocation")

    inspect_only_invocation = (
        built_in_provider_rendering.render_built_in_provider_invocation(
            built_in_provider_rendering.BuiltInProviderRenderRequest(
                provider_selection=(
                    built_in_provider_rendering.BuiltInProviderSelectionFacts(
                        service="claude",
                        model="sonnet",
                        effort="medium",
                    )
                ),
                run_kind=RunKind.FRESH,
                tool_access=ToolAccess.workspace_backed(
                    invocation_dir,
                    tool_policy=runtime.ToolPolicy.INSPECT_ONLY,
                ),
                auth=ProviderAuth(claude_code_oauth_token="token"),
                invocation_dir=invocation_dir,
            )
        )
    )
    custom_profile_invocation = (
        built_in_provider_rendering.render_built_in_provider_invocation(
            built_in_provider_rendering.BuiltInProviderRenderRequest(
                provider_selection=(
                    built_in_provider_rendering.BuiltInProviderSelectionFacts(
                        service="claude",
                        model="sonnet",
                        effort="medium",
                    )
                ),
                run_kind=RunKind.FRESH,
                tool_access=ToolAccess.workspace_backed(
                    invocation_dir,
                    tool_policy=ToolPolicyProfile(
                        allowed_tools=("Read", "Glob", "Bash"),
                        disallowed_tools=("Edit",),
                        strict_mcp_config=False,
                    ),
                ),
                auth=ProviderAuth(claude_code_oauth_token="token"),
                invocation_dir=invocation_dir,
            )
        )
    )

    assert inspect_only_invocation.canonical_argv == (
        "claude",
        "--verbose",
        "--dangerously-skip-permissions",
        "--output-format",
        "stream-json",
        "-p",
        "-",
        "--disable-slash-commands",
        "--exclude-dynamic-system-prompt-sections",
        "--tools",
        "Read Glob",
        "--strict-mcp-config",
        "--mcp-config",
        '{"mcpServers":{}}',
        "--model",
        "sonnet",
        "--effort",
        "medium",
    )
    assert custom_profile_invocation.canonical_argv == (
        "claude",
        "--verbose",
        "--dangerously-skip-permissions",
        "--output-format",
        "stream-json",
        "-p",
        "-",
        "--disable-slash-commands",
        "--exclude-dynamic-system-prompt-sections",
        "--tools",
        "Read Glob Bash",
        "--disallowedTools",
        "Edit",
        "--model",
        "sonnet",
        "--effort",
        "medium",
    )


def test_render_claude_invocation_places_provider_session_ids_for_fresh_and_resume() -> (
    None
):
    invocation_dir = Path("/tmp/invocation")

    fresh_render = built_in_provider_rendering.render_built_in_provider_invocation(
        built_in_provider_rendering.BuiltInProviderRenderRequest(
            provider_selection=built_in_provider_rendering.BuiltInProviderSelectionFacts(
                service="claude",
                model="sonnet",
                effort="medium",
            ),
            run_kind=RunKind.FRESH,
            tool_access=ToolAccess.workspace_backed(invocation_dir),
            auth=ProviderAuth(claude_code_oauth_token="token"),
            invocation_dir=invocation_dir,
            provider_session_id="session-fresh",
        )
    )
    resumed_render = built_in_provider_rendering.render_built_in_provider_invocation(
        built_in_provider_rendering.BuiltInProviderRenderRequest(
            provider_selection=built_in_provider_rendering.BuiltInProviderSelectionFacts(
                service="claude",
                model="sonnet",
                effort="medium",
            ),
            run_kind=RunKind.RESUME,
            tool_access=ToolAccess.workspace_backed(invocation_dir),
            auth=ProviderAuth(claude_code_oauth_token="token"),
            invocation_dir=invocation_dir,
            provider_session_id="session-resume",
        )
    )

    assert fresh_render.provider_session_id_placement is (
        built_in_provider_rendering.ProviderSessionIdPlacement.CLI_FLAG
    )
    assert fresh_render.canonical_argv[-2:] == ("--session-id", "session-fresh")
    assert resumed_render.provider_session_id_placement is (
        built_in_provider_rendering.ProviderSessionIdPlacement.CLI_FLAG
    )
    assert resumed_render.canonical_argv[-2:] == ("--resume", "session-resume")


def test_render_claude_invocation_fails_for_missing_credentials_and_unsupported_selection() -> (
    None
):
    invocation_dir = Path("/tmp/invocation")

    with pytest.raises(
        AgentCredentialFailureError, match="Missing Claude Code OAuth token."
    ):
        built_in_provider_rendering.render_built_in_provider_invocation(
            built_in_provider_rendering.BuiltInProviderRenderRequest(
                provider_selection=(
                    built_in_provider_rendering.BuiltInProviderSelectionFacts(
                        service="claude",
                        model="sonnet",
                        effort="medium",
                    )
                ),
                run_kind=RunKind.FRESH,
                tool_access=ToolAccess.workspace_backed(invocation_dir),
                auth=None,
                invocation_dir=invocation_dir,
            )
        )

    with pytest.raises(
        RuntimeConfigurationError, match=r"Unsupported Claude model 'invalid'\."
    ):
        built_in_provider_rendering.render_built_in_provider_invocation(
            built_in_provider_rendering.BuiltInProviderRenderRequest(
                provider_selection=(
                    built_in_provider_rendering.BuiltInProviderSelectionFacts(
                        service="claude",
                        model="invalid",
                        effort="medium",
                    )
                ),
                run_kind=RunKind.FRESH,
                tool_access=ToolAccess.workspace_backed(invocation_dir),
                auth=ProviderAuth(claude_code_oauth_token="token"),
                invocation_dir=invocation_dir,
            )
        )

    with pytest.raises(
        RuntimeConfigurationError, match=r"Unsupported Claude effort 'invalid'\."
    ):
        built_in_provider_rendering.render_built_in_provider_invocation(
            built_in_provider_rendering.BuiltInProviderRenderRequest(
                provider_selection=(
                    built_in_provider_rendering.BuiltInProviderSelectionFacts(
                        service="claude",
                        model="sonnet",
                        effort="invalid",
                    )
                ),
                run_kind=RunKind.FRESH,
                tool_access=ToolAccess.workspace_backed(invocation_dir),
                auth=ProviderAuth(claude_code_oauth_token="token"),
                invocation_dir=invocation_dir,
            )
        )


def test_render_codex_fresh_invocation_returns_canonical_argv_environment_and_prompt_facts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    host_home = tmp_path / "host-home"
    host_auth_path = host_home / ".codex" / "auth.json"
    host_auth_path.parent.mkdir(parents=True)
    host_auth_path.write_text('{"token":"host-auth"}\n', encoding="utf-8")
    monkeypatch.setattr(built_in_provider_rendering.Path, "home", lambda: host_home)

    rendered_invocation = built_in_provider_rendering.render_built_in_provider_invocation(
        built_in_provider_rendering.BuiltInProviderRenderRequest(
            provider_selection=built_in_provider_rendering.BuiltInProviderSelectionFacts(
                service="codex",
                model="gpt-5.4",
                effort="medium",
            ),
            run_kind=RunKind.FRESH,
            tool_access=ToolAccess.workspace_backed(
                tmp_path,
                tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
            ),
            auth=None,
            invocation_dir=tmp_path,
            provider_state_dir=tmp_path / "provider-state",
        )
    )

    assert (
        rendered_invocation
        == built_in_provider_rendering.BuiltInProviderRenderedInvocation(
            canonical_argv=(
                "codex",
                "exec",
                "-m",
                "gpt-5.4",
                "-c",
                "model_reasoning_effort=medium",
                "-c",
                "approval_policy=never",
                "--sandbox",
                "read-only",
                "--json",
            ),
            legacy_command_text=(
                "codex exec -m gpt-5.4 -c model_reasoning_effort=medium "
                "-c approval_policy=never --sandbox read-only --json"
            ),
            environment={
                "TZ": "UTC",
                "CODEX_HOME": str(tmp_path / "provider-state"),
            },
            prompt_path=Path("/tmp/.provider_prompt"),
            prompt_cleanup_choice=(
                built_in_provider_rendering.PromptCleanupChoice.DELETE_AFTER_INVOCATION
            ),
            prompt_transport_preference=(
                built_in_provider_rendering.PromptTransportPreference.STDIN
            ),
            provider_session_id_placement=(
                built_in_provider_rendering.ProviderSessionIdPlacement.NONE
            ),
            prefer_argv=True,
        )
    )


def test_render_codex_resumed_invocation_places_and_carries_provider_session_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    host_home = tmp_path / "host-home"
    host_auth_path = host_home / ".codex" / "auth.json"
    host_auth_path.parent.mkdir(parents=True)
    host_auth_path.write_text('{"token":"host-auth"}\n', encoding="utf-8")
    monkeypatch.setattr(built_in_provider_rendering.Path, "home", lambda: host_home)

    rendered_invocation = built_in_provider_rendering.render_built_in_provider_invocation(
        built_in_provider_rendering.BuiltInProviderRenderRequest(
            provider_selection=built_in_provider_rendering.BuiltInProviderSelectionFacts(
                service="codex",
                model="gpt-5.4",
                effort="medium",
            ),
            run_kind=RunKind.RESUME,
            tool_access=ToolAccess.workspace_backed(
                tmp_path,
                tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
            ),
            auth=None,
            invocation_dir=tmp_path,
            provider_session_id="thread-123",
        )
    )

    assert rendered_invocation.canonical_argv[:4] == (
        "codex",
        "exec",
        "resume",
        "thread-123",
    )
    assert rendered_invocation.provider_session_id_placement is (
        built_in_provider_rendering.ProviderSessionIdPlacement.CLI_FLAG
    )
    assert rendered_invocation.provider_session_id == "thread-123"


@pytest.mark.parametrize(
    ("tool_policy", "expected_sandbox"),
    [
        pytest.param(runtime.ToolPolicy.NONE, "read-only", id="none"),
        pytest.param(runtime.ToolPolicy.INSPECT_ONLY, "read-only", id="inspect-only"),
        pytest.param(
            runtime.ToolPolicy.NO_FILE_MUTATION, "read-only", id="no-file-mutation"
        ),
        pytest.param(
            runtime.ToolPolicy.UNRESTRICTED,
            "danger-full-access",
            id="unrestricted",
        ),
        pytest.param(
            ToolPolicyProfile(),
            "danger-full-access",
            id="custom-unrestricted-profile",
        ),
    ],
)
def test_render_codex_uses_windows_executable_and_tool_policy_sandbox_mapping(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tool_policy: runtime.ToolPolicy | ToolPolicyProfile,
    expected_sandbox: str,
) -> None:
    host_home = tmp_path / "host-home"
    host_auth_path = host_home / ".codex" / "auth.json"
    host_auth_path.parent.mkdir(parents=True)
    host_auth_path.write_text('{"token":"host-auth"}\n', encoding="utf-8")
    monkeypatch.setattr(built_in_provider_rendering.Path, "home", lambda: host_home)

    rendered_invocation = built_in_provider_rendering.render_built_in_provider_invocation(
        built_in_provider_rendering.BuiltInProviderRenderRequest(
            provider_selection=built_in_provider_rendering.BuiltInProviderSelectionFacts(
                service="codex",
                model="gpt-5.4",
                effort="medium",
            ),
            run_kind=RunKind.FRESH,
            tool_access=ToolAccess.workspace_backed(
                tmp_path,
                tool_policy=tool_policy,
            ),
            auth=None,
            invocation_dir=tmp_path,
            host_facts=built_in_provider_rendering.BuiltInProviderHostFacts(
                os_name="nt",
            ),
        )
    )

    assert rendered_invocation.canonical_argv[:2] == ("codex.cmd", "exec")
    assert rendered_invocation.canonical_argv[-3:] == (
        "--sandbox",
        expected_sandbox,
        "--json",
    )


def test_render_codex_fails_for_missing_host_auth_and_unsupported_selection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    host_home = tmp_path / "host-home"
    monkeypatch.setattr(built_in_provider_rendering.Path, "home", lambda: host_home)

    with pytest.raises(
        AgentCredentialFailureError,
        match="Codex authentication missing: run `codex login` on the host.",
    ):
        built_in_provider_rendering.render_built_in_provider_invocation(
            built_in_provider_rendering.BuiltInProviderRenderRequest(
                provider_selection=(
                    built_in_provider_rendering.BuiltInProviderSelectionFacts(
                        service="codex",
                        model="gpt-5.4",
                        effort="medium",
                    )
                ),
                run_kind=RunKind.FRESH,
                tool_access=ToolAccess.workspace_backed(tmp_path),
                auth=None,
                invocation_dir=tmp_path,
            )
        )

    host_auth_path = host_home / ".codex" / "auth.json"
    host_auth_path.parent.mkdir(parents=True)
    host_auth_path.write_text('{"token":"host-auth"}\n', encoding="utf-8")

    with pytest.raises(
        RuntimeConfigurationError, match=r"Unsupported Codex model 'invalid'\."
    ):
        built_in_provider_rendering.render_built_in_provider_invocation(
            built_in_provider_rendering.BuiltInProviderRenderRequest(
                provider_selection=(
                    built_in_provider_rendering.BuiltInProviderSelectionFacts(
                        service="codex",
                        model="invalid",
                        effort="medium",
                    )
                ),
                run_kind=RunKind.FRESH,
                tool_access=ToolAccess.workspace_backed(tmp_path),
                auth=None,
                invocation_dir=tmp_path,
            )
        )

    with pytest.raises(
        RuntimeConfigurationError, match=r"Unsupported Codex effort 'invalid'\."
    ):
        built_in_provider_rendering.render_built_in_provider_invocation(
            built_in_provider_rendering.BuiltInProviderRenderRequest(
                provider_selection=(
                    built_in_provider_rendering.BuiltInProviderSelectionFacts(
                        service="codex",
                        model="gpt-5.4",
                        effort="invalid",
                    )
                ),
                run_kind=RunKind.FRESH,
                tool_access=ToolAccess.workspace_backed(tmp_path),
                auth=None,
                invocation_dir=tmp_path,
            )
        )
