from __future__ import annotations

import dataclasses
import enum
import json
import os
import shlex
from collections.abc import Callable, Mapping
from pathlib import Path
from types import MappingProxyType

from .errors import AgentCredentialFailureError, RuntimeConfigurationError
from ._runtime_lifecycle import ProviderAuth
from .contracts import ToolAccess, ToolPolicy, ToolPolicyProfile
from .session import RunKind

_CLAUDE_VALID_MODELS = frozenset({"haiku", "sonnet", "opus"})
_CLAUDE_VALID_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})
_CODEX_VALID_MODELS = frozenset(
    {
        "gpt-5.5",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.3-codex",
        "gpt-5.3-codex-spark",
        "gpt-5.2",
    }
)
_CODEX_VALID_EFFORTS = frozenset({"low", "medium", "high", "xhigh"})
_OPENCODE_GO_PROVIDER_ID = "opencode-go"
_OPENCODE_GO_BASE_URL = "https://opencode.ai/zen/go/v1"
_OPENCODE_GO_MODELS = frozenset(
    {
        "deepseek-v4-flash",
        "deepseek-v4-pro",
        "glm-5.1",
        "glm-5.2",
        "kimi-k2.6",
        "kimi-k2.7-code",
        "mimo-v2.5",
        "mimo-v2.5-pro",
        "minimax-m2.7",
        "minimax-m3",
        "qwen3.6-plus",
        "qwen3.7-max",
        "qwen3.7-plus",
    }
)
_OPENCODE_VALID_EFFORTS = frozenset({"medium"})
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
    provider_session_id: str | None = None
    prefer_argv: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "canonical_argv", tuple(self.canonical_argv))
        object.__setattr__(self, "environment", _freeze_mapping(self.environment))


@dataclasses.dataclass(frozen=True, slots=True)
class BuiltInProviderRenderedToolPolicy:
    claude_profile: ToolPolicyProfile
    codex_sandbox: str
    opencode_permission: dict[str, str] | str | None


def _builtin_provider_prompt_path(invocation_dir: Path) -> Path:
    return invocation_dir / _BUILTIN_PROVIDER_PROMPT_FILENAME


def _codex_provider_prompt_path() -> Path:
    return _builtin_provider_prompt_path(Path("/tmp"))


def _claude_tool_policy_profile(tool_access: ToolAccess) -> ToolPolicyProfile:
    return render_built_in_provider_tool_policy(tool_access.tool_policy).claude_profile


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


def _validate_codex_selection(
    provider_selection: BuiltInProviderSelectionFacts,
) -> None:
    if provider_selection.model not in _CODEX_VALID_MODELS:
        raise RuntimeConfigurationError(
            f"Unsupported Codex model {provider_selection.model!r}."
        )
    if provider_selection.effort not in _CODEX_VALID_EFFORTS:
        raise RuntimeConfigurationError(
            f"Unsupported Codex effort {provider_selection.effort!r}."
        )


def _validate_opencode_selection(
    provider_selection: BuiltInProviderSelectionFacts,
) -> None:
    if provider_selection.model not in _OPENCODE_GO_MODELS:
        raise RuntimeConfigurationError(
            f"Unsupported OpenCode model {provider_selection.model!r}."
        )
    if provider_selection.effort not in _OPENCODE_VALID_EFFORTS:
        raise RuntimeConfigurationError(
            f"Unsupported OpenCode effort {provider_selection.effort!r}."
        )


def _require_claude_auth(auth: ProviderAuth | None) -> None:
    if auth is not None and auth.claude_code_oauth_token:
        return
    raise AgentCredentialFailureError(
        "Missing Claude Code OAuth token.",
        service_name="claude",
    )


def _codex_host_auth_path() -> Path:
    return Path.home() / ".codex" / "auth.json"


def _codex_host_home() -> Path:
    return _codex_host_auth_path().parent


def _missing_codex_auth_error() -> AgentCredentialFailureError:
    return AgentCredentialFailureError(
        "Codex authentication missing: run `codex login` on the host.",
        service_name="codex",
    )


def _require_codex_auth() -> None:
    if _codex_host_auth_path().exists():
        return
    raise _missing_codex_auth_error()


def _require_opencode_auth(auth: ProviderAuth | None) -> None:
    if auth is not None and auth.opencode_api_key:
        return
    raise AgentCredentialFailureError(
        "Missing OpenCode API key.",
        service_name="opencode",
    )


def _claude_environment(
    auth: ProviderAuth | None,
    provider_state_dir: Path | None,
) -> dict[str, str]:
    environment: dict[str, str] = {}
    token = None if auth is None else auth.claude_code_oauth_token
    if token:
        environment["CLAUDE_CODE_OAUTH_TOKEN"] = token
    if provider_state_dir is not None:
        environment["CLAUDE_CONFIG_DIR"] = str(provider_state_dir)
    return environment


def _codex_environment(provider_state_dir: Path | None) -> dict[str, str]:
    return {
        "TZ": "UTC",
        "CODEX_HOME": str(
            provider_state_dir if provider_state_dir is not None else _codex_host_home()
        ),
    }


def _opencode_go_model_ref(model: str) -> str:
    if "/" in model:
        return model
    return f"{_OPENCODE_GO_PROVIDER_ID}/{model}"


def _opencode_tool_policy_permission(
    tool_policy: ToolPolicy | ToolPolicyProfile,
) -> dict[str, str] | str | None:
    return render_built_in_provider_tool_policy(tool_policy).opencode_permission


def _opencode_go_config_content(
    *,
    tool_policy: ToolPolicy | ToolPolicyProfile | None = None,
) -> str:
    config: dict[str, object] = {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            _OPENCODE_GO_PROVIDER_ID: {
                "npm": "@ai-sdk/openai-compatible",
                "name": "OpenCode Go",
                "options": {
                    "baseURL": _OPENCODE_GO_BASE_URL,
                    "apiKey": "{env:OPENCODE_GO_API_KEY}",
                },
                "models": {
                    model: {"name": model} for model in sorted(_OPENCODE_GO_MODELS)
                },
            }
        },
    }
    if tool_policy is not None:
        permission = _opencode_tool_policy_permission(tool_policy)
        if permission is not None:
            config["permission"] = permission
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def _opencode_environment(
    *,
    auth: ProviderAuth | None,
    provider_state_dir: Path | None,
    tool_policy: ToolPolicy | ToolPolicyProfile | None,
) -> dict[str, str]:
    environment: dict[str, str] = {"TZ": "UTC"}
    if provider_state_dir is not None:
        environment["OPENCODE_HOME"] = str(provider_state_dir)
    api_key = None if auth is None else auth.opencode_api_key
    if api_key:
        environment["OPENCODE_GO_API_KEY"] = api_key
        environment["OPENCODE_CONFIG_CONTENT"] = _opencode_go_config_content(
            tool_policy=tool_policy
        )
    return environment


def _codex_sandbox(tool_policy: ToolPolicy | ToolPolicyProfile) -> str:
    return render_built_in_provider_tool_policy(tool_policy).codex_sandbox


def render_built_in_provider_tool_policy(
    tool_policy: ToolPolicy | ToolPolicyProfile,
) -> BuiltInProviderRenderedToolPolicy:
    profile = (
        tool_policy.profile if isinstance(tool_policy, ToolPolicy) else tool_policy
    )
    if isinstance(tool_policy, ToolPolicy):
        claude_profile = (
            ToolPolicyProfile(disallowed_tools=("all",))
            if tool_policy is ToolPolicy.NONE
            else profile
        )
        codex_sandbox = (
            "read-only"
            if tool_policy
            in {
                ToolPolicy.NONE,
                ToolPolicy.NO_FILE_MUTATION,
            }
            else "danger-full-access"
        )
    else:
        claude_profile = profile
        codex_sandbox = (
            "read-only"
            if profile
            in {
                ToolPolicy.NONE.profile,
                ToolPolicy.NO_FILE_MUTATION.profile,
            }
            else "danger-full-access"
        )
    opencode_permission: dict[str, str] | str | None
    if profile == ToolPolicy.NONE.profile:
        opencode_permission = "deny"
    elif profile == ToolPolicy.NO_FILE_MUTATION.profile:
        opencode_permission = {"edit": "deny"}
    else:
        opencode_permission = None
    return BuiltInProviderRenderedToolPolicy(
        claude_profile=claude_profile,
        codex_sandbox=codex_sandbox,
        opencode_permission=opencode_permission,
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
    return BuiltInProviderRenderedInvocation(
        canonical_argv=("claude", *flags),
        legacy_command_text=f"claude {legacy_flags} < {shlex.quote(str(prompt_path))}",
        environment=_claude_environment(request.auth, request.provider_state_dir),
        prompt_path=prompt_path,
        prompt_cleanup_choice=PromptCleanupChoice.DELETE_AFTER_INVOCATION,
        prompt_transport_preference=PromptTransportPreference.STDIN,
        provider_session_id_placement=provider_session_id_placement,
        provider_session_id=request.provider_session_id,
        prefer_argv=True,
    )


def _render_codex_invocation(
    request: BuiltInProviderRenderRequest,
    *,
    validate_auth: bool = True,
    argv_transform: (
        Callable[[tuple[str, ...], Path, dict[str, str]], tuple[str, ...]] | None
    ) = None,
) -> BuiltInProviderRenderedInvocation:
    _validate_codex_selection(request.provider_selection)
    if validate_auth:
        _require_codex_auth()
    executable = (
        "codex.cmd"
        if ((request.host_facts.os_name if request.host_facts else None) or os.name)
        == "nt"
        else "codex"
    )
    prompt_path = _codex_provider_prompt_path()
    flags: list[str] = []
    codex_sandbox = (
        "danger-full-access"
        if argv_transform is not None
        else _codex_sandbox(request.tool_access.tool_policy)
    )
    is_resumed_session = (
        request.run_kind is RunKind.RESUME and request.provider_session_id is not None
    )
    if is_resumed_session:
        assert request.provider_session_id is not None
        flags.extend(
            ["--sandbox", codex_sandbox, "resume", request.provider_session_id]
        )
    if request.provider_selection.model:
        flags.extend(["-m", request.provider_selection.model])
    if request.provider_selection.effort:
        flags.extend(
            ["-c", f"model_reasoning_effort={request.provider_selection.effort}"]
        )
    flags.extend(["-c", "approval_policy=never"])
    if not is_resumed_session:
        flags.extend(["--sandbox", codex_sandbox])
    flags.append("--json")
    provider_session_id_placement = (
        ProviderSessionIdPlacement.CLI_FLAG
        if is_resumed_session
        else ProviderSessionIdPlacement.NONE
    )
    return BuiltInProviderRenderedInvocation(
        canonical_argv=(executable, "exec", *flags),
        legacy_command_text=" ".join(
            shlex.quote(part) for part in (executable, "exec", *flags)
        ),
        environment=_codex_environment(request.provider_state_dir),
        prompt_path=prompt_path,
        prompt_cleanup_choice=PromptCleanupChoice.DELETE_AFTER_INVOCATION,
        prompt_transport_preference=PromptTransportPreference.STDIN,
        provider_session_id_placement=provider_session_id_placement,
        provider_session_id=request.provider_session_id,
        prefer_argv=True,
    )


def _render_opencode_invocation(
    request: BuiltInProviderRenderRequest,
) -> BuiltInProviderRenderedInvocation:
    _validate_opencode_selection(request.provider_selection)
    _require_opencode_auth(request.auth)
    executable = (
        "opencode.cmd"
        if ((request.host_facts.os_name if request.host_facts else None) or os.name)
        == "nt"
        else "opencode"
    )
    prompt_path = _builtin_provider_prompt_path(request.invocation_dir)
    flags = ["run", "--format", "json"]
    if request.run_kind is RunKind.RESUME and request.provider_session_id:
        flags.extend(["--session", request.provider_session_id])
    flags.extend(["--model", _opencode_go_model_ref(request.provider_selection.model)])
    return BuiltInProviderRenderedInvocation(
        canonical_argv=(executable, *flags),
        legacy_command_text=(
            " ".join(shlex.quote(part) for part in (executable, *flags))
            + f' "$(cat {shlex.quote(str(prompt_path))})"'
        ),
        environment=_opencode_environment(
            auth=request.auth,
            provider_state_dir=request.provider_state_dir,
            tool_policy=request.tool_access.tool_policy,
        ),
        prompt_path=prompt_path,
        prompt_cleanup_choice=PromptCleanupChoice.DELETE_AFTER_INVOCATION,
        prompt_transport_preference=PromptTransportPreference.PROMPT_FILE,
        provider_session_id_placement=(
            ProviderSessionIdPlacement.CLI_FLAG
            if request.run_kind is RunKind.RESUME and request.provider_session_id
            else ProviderSessionIdPlacement.NONE
        ),
        provider_session_id=request.provider_session_id,
        prefer_argv=True,
    )
