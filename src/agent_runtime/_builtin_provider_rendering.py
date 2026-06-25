from __future__ import annotations

import dataclasses
import enum
import shlex
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

from .errors import AgentCredentialFailureError, RuntimeConfigurationError
from ._runtime_lifecycle import ProviderAuth
from .contracts import ToolAccess, ToolPolicy, ToolPolicyProfile
from .session import RunKind

_CLAUDE_VALID_MODELS = frozenset({"haiku", "sonnet", "opus"})
_CLAUDE_VALID_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})
_BUILTIN_PROVIDER_PROMPT_FILENAME = ".provider_prompt"


def _freeze_mapping(values: Mapping[str, str]) -> Mapping[str, str]:
    return MappingProxyType(dict(values))


@dataclasses.dataclass(frozen=True, slots=True)
class BuiltInProviderSelectionFacts:
    service: str
    model: str
    effort: str


@dataclasses.dataclass(frozen=True, slots=True)
class BuiltInProviderHostFacts:
    os_name: str | None = None
    environment: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if self.environment is not None:
            object.__setattr__(
                self,
                "environment",
                _freeze_mapping(self.environment),
            )


class PromptCleanupChoice(str, enum.Enum):
    KEEP = "KEEP"
    DELETE_AFTER_INVOCATION = "DELETE_AFTER_INVOCATION"


class PromptTransportPreference(str, enum.Enum):
    STDIN = "STDIN"
    PROMPT_FILE = "PROMPT_FILE"


class ProviderSessionIdPlacement(str, enum.Enum):
    NONE = "NONE"
    CLI_FLAG = "CLI_FLAG"
    ENVIRONMENT = "ENVIRONMENT"


@dataclasses.dataclass(frozen=True, slots=True)
class BuiltInProviderRenderRequest:
    provider_selection: BuiltInProviderSelectionFacts
    run_kind: RunKind
    tool_access: ToolAccess
    auth: ProviderAuth | None
    invocation_dir: Path
    provider_state_dir: Path | None = None
    provider_session_id: str | None = None
    host_facts: BuiltInProviderHostFacts | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class BuiltInProviderRenderedInvocation:
    canonical_argv: tuple[str, ...]
    legacy_command_text: str | None
    environment: Mapping[str, str]
    prompt_path: Path | None
    prompt_cleanup_choice: PromptCleanupChoice
    prompt_transport_preference: PromptTransportPreference
    provider_session_id_placement: ProviderSessionIdPlacement
    prefer_argv: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "canonical_argv", tuple(self.canonical_argv))
        object.__setattr__(self, "environment", _freeze_mapping(self.environment))


def _builtin_provider_prompt_path(invocation_dir: Path) -> Path:
    return invocation_dir / _BUILTIN_PROVIDER_PROMPT_FILENAME


def _claude_tool_policy_profile(tool_access: ToolAccess) -> ToolPolicyProfile:
    if isinstance(tool_access.tool_policy, ToolPolicy):
        if tool_access.tool_policy is ToolPolicy.NONE:
            return ToolPolicyProfile(disallowed_tools=("all",))
        if tool_access.tool_policy is ToolPolicy.INSPECT_ONLY:
            return ToolPolicyProfile(allowed_tools=("Read", "Glob"))
        if tool_access.tool_policy is ToolPolicy.NO_FILE_MUTATION:
            return ToolPolicyProfile(disallowed_tools=("Edit", "Write", "NotebookEdit"))
        return ToolPolicyProfile()
    return tool_access.tool_policy


def _validate_claude_selection(
    provider_selection: BuiltInProviderSelectionFacts,
) -> None:
    if provider_selection.model not in _CLAUDE_VALID_MODELS:
        raise RuntimeConfigurationError(
            f"Unsupported Claude model {provider_selection.model!r}."
        )
    if provider_selection.effort not in _CLAUDE_VALID_EFFORTS:
        raise RuntimeConfigurationError(
            f"Unsupported Claude effort {provider_selection.effort!r}."
        )


def _require_claude_auth(auth: ProviderAuth | None) -> None:
    if auth is not None and auth.claude_code_oauth_token:
        return
    raise AgentCredentialFailureError(
        "Missing Claude Code OAuth token.",
        service_name="claude",
    )


def _render_claude_invocation(
    request: BuiltInProviderRenderRequest,
) -> BuiltInProviderRenderedInvocation:
    _validate_claude_selection(request.provider_selection)
    _require_claude_auth(request.auth)
    assert request.auth is not None
    assert request.auth.claude_code_oauth_token is not None
    profile = _claude_tool_policy_profile(request.tool_access)
    prompt_path = _builtin_provider_prompt_path(request.invocation_dir)
    flags = [
        "--verbose",
        "--dangerously-skip-permissions",
        "--output-format",
        "stream-json",
        "-p",
        "-",
        "--disable-slash-commands",
        "--exclude-dynamic-system-prompt-sections",
    ]
    if profile.allowed_tools is not None:
        flags.extend(["--tools", " ".join(profile.allowed_tools)])
    if profile.disallowed_tools:
        flags.extend(["--disallowedTools", " ".join(profile.disallowed_tools)])
    if profile.strict_mcp_config:
        flags.extend(["--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}'])
    flags.extend(
        [
            "--model",
            request.provider_selection.model,
            "--effort",
            request.provider_selection.effort,
        ]
    )
    provider_session_id_placement = ProviderSessionIdPlacement.NONE
    if request.provider_session_id:
        provider_session_id_placement = ProviderSessionIdPlacement.CLI_FLAG
        if request.run_kind is RunKind.RESUME:
            flags.extend(["--resume", request.provider_session_id])
        else:
            flags.extend(["--session-id", request.provider_session_id])
    legacy_flags = (
        "--verbose --dangerously-skip-permissions --output-format stream-json -p -"
        " --disable-slash-commands --exclude-dynamic-system-prompt-sections"
    )
    if profile.allowed_tools is not None:
        legacy_flags += f" --tools {shlex.quote(' '.join(profile.allowed_tools))}"
    if profile.disallowed_tools:
        legacy_flags += f' --disallowedTools "{" ".join(profile.disallowed_tools)}"'
    if profile.strict_mcp_config:
        legacy_flags += " --strict-mcp-config --mcp-config '{\"mcpServers\":{}}'"
    legacy_flags += (
        f" --model {request.provider_selection.model}"
        f" --effort {request.provider_selection.effort}"
    )
    if request.provider_session_id:
        if request.run_kind is RunKind.RESUME:
            legacy_flags += f" --resume {shlex.quote(request.provider_session_id)}"
        else:
            legacy_flags += f" --session-id {shlex.quote(request.provider_session_id)}"
    environment = {"CLAUDE_CODE_OAUTH_TOKEN": request.auth.claude_code_oauth_token}
    if request.provider_state_dir is not None:
        environment["CLAUDE_CONFIG_DIR"] = str(request.provider_state_dir)
    return BuiltInProviderRenderedInvocation(
        canonical_argv=("claude", *flags),
        legacy_command_text=f"claude {legacy_flags} < {shlex.quote(str(prompt_path))}",
        environment=environment,
        prompt_path=prompt_path,
        prompt_cleanup_choice=PromptCleanupChoice.DELETE_AFTER_INVOCATION,
        prompt_transport_preference=PromptTransportPreference.STDIN,
        provider_session_id_placement=provider_session_id_placement,
        prefer_argv=True,
    )


def render_built_in_provider_invocation(
    request: BuiltInProviderRenderRequest,
) -> BuiltInProviderRenderedInvocation:
    if request.provider_selection.service == "claude":
        return _render_claude_invocation(request)
    raise RuntimeConfigurationError(
        f"Unsupported built-in provider {request.provider_selection.service!r}."
    )
