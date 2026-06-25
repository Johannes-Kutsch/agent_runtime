from __future__ import annotations

import dataclasses
import enum
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

from ._runtime_lifecycle import ProviderAuth
from .contracts import ToolAccess
from .session import RunKind


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

    def __post_init__(self) -> None:
        object.__setattr__(self, "canonical_argv", tuple(self.canonical_argv))
        object.__setattr__(self, "environment", _freeze_mapping(self.environment))
