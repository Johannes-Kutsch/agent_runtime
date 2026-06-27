from __future__ import annotations

import asyncio
import dataclasses
import inspect
import json
import math
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, Mapping, cast

from .contracts import ToolAccess, ToolPolicy, ToolPolicyProfile
from .provider_usage import ProviderUsage
from . import _lifecycle_request_facts as _lifecycle_request_facts_module
from .types import ProviderSelection, ResolvedProvider
from .errors import ProviderUnavailableReason

__all__ = [
    "AgentEvent",
    "Cancelled",
    "Completed",
    "Continuation",
    "EphemeralRunRequest",
    "NewSessionRunRequest",
    "ProviderUnavailable",
    "ProviderUsage",
    "ProviderAuth",
    "ResolvedProvider",
    "ResumedSessionRunRequest",
    "RunResult",
    "RuntimeOutcome",
    "TimedOut",
    "UsageLimited",
]

_MISSING_TOOL_POLICY = object()
_DEFAULT_EPHEMERAL_SESSION_NAMESPACE = ""
_PUBLIC_INVOCATION_DIR_NAME = "invocation_dir"


@dataclasses.dataclass
class CancellationToken:
    _event: asyncio.Event = dataclasses.field(
        default_factory=asyncio.Event,
        init=False,
        repr=False,
    )

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> None:
        self._event.set()


def _redacted_credential_value(value: str | None) -> str:
    return "None" if value is None else "'<redacted>'"


def _public_request_signature(
    *parameter_names: str,
) -> inspect.Signature:
    return inspect.Signature(
        [
            inspect.Parameter(
                name,
                kind=(
                    inspect.Parameter.KEYWORD_ONLY
                    if name == "override"
                    else inspect.Parameter.POSITIONAL_OR_KEYWORD
                ),
            )
            for name in parameter_names
        ]
    )


def _require_json_compatible_resume_state(
    value: Any,
    *,
    path: str = "provider_resume_state",
) -> None:
    if value is None or isinstance(value, str | bool | int):
        return
    if isinstance(value, float):
        if math.isfinite(value):
            return
        raise TypeError("Continuation provider_resume_state must be JSON-compatible.")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _require_json_compatible_resume_state(item, path=f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    "Continuation provider_resume_state must be JSON-compatible."
                )
            _require_json_compatible_resume_state(item, path=f"{path}.{key}")
        return
    raise TypeError("Continuation provider_resume_state must be JSON-compatible.")


@dataclasses.dataclass(frozen=True)
class _PortableContinuationPayload:
    service_name: str
    model: str
    effort: str
    tool_access: ToolAccess
    provider_resume_state: dict[str, Any]

    @property
    def serialized(self) -> str:
        return json.dumps(self._payload_state(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_serialized(cls, serialized: str) -> _PortableContinuationPayload:
        try:
            payload = json.loads(serialized)
        except json.JSONDecodeError as exc:
            raise TypeError("Continuation data is not valid JSON.") from exc
        if not isinstance(payload, dict):
            raise TypeError("Continuation data must be a JSON object.")
        serialized_provider_resume_state = payload.get("provider_resume_state")
        if not isinstance(serialized_provider_resume_state, dict):
            raise TypeError("Continuation provider_resume_state must be a JSON object.")
        return cls(
            service_name=_payload_string_field(payload.get("service_name")),
            model=_payload_string_field(payload.get("model")),
            effort=_payload_string_field(payload.get("effort")),
            tool_access=_deserialize_tool_access(payload.get("tool_access")),
            provider_resume_state=serialized_provider_resume_state,
        )

    def _payload_state(self) -> dict[str, Any]:
        return {
            "service_name": self.service_name,
            "model": self.model,
            "effort": self.effort,
            "tool_access": _serialize_tool_access(self.tool_access),
            "provider_resume_state": self.provider_resume_state,
        }


def _payload_string_field(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("Continuation data is malformed.")
    return value


def _serialize_tool_access(tool_access: ToolAccess) -> dict[str, Any]:
    policy = tool_access.tool_policy
    policy_payload: dict[str, Any]
    if isinstance(policy, ToolPolicy):
        policy_payload = {"kind": "tool_policy", "value": policy.value}
    else:
        policy_payload = {
            "kind": "tool_policy_profile",
            "allowed_tools": policy.allowed_tools,
            "disallowed_tools": policy.disallowed_tools,
            "strict_mcp_config": policy.strict_mcp_config,
        }
    return {
        "kind": tool_access.kind,
        "workspace": str(tool_access.workspace) if tool_access.workspace else None,
        "tool_policy": policy_payload,
    }


def _deserialize_tool_access(value: Any) -> ToolAccess:
    if not isinstance(value, dict):
        raise TypeError("Continuation data is malformed.")
    kind = value.get("kind")
    workspace = value.get("workspace")
    policy = value.get("tool_policy")
    if kind not in {"none", "workspace_backed"}:
        raise TypeError("Continuation data is malformed.")
    if not isinstance(policy, dict):
        raise TypeError("Continuation data is malformed.")
    profile_type = policy.get("kind")
    tool_policy: ToolPolicy | ToolPolicyProfile
    if profile_type == "tool_policy":
        if not isinstance(policy.get("value"), str):
            raise TypeError("Continuation data is malformed.")
        policy_value = policy["value"]
        if policy_value == "inspect_only":
            raise TypeError(
                "Continuation data contains legacy tool-policy value `inspect_only`."
            )
        try:
            tool_policy = ToolPolicy(policy_value)
        except ValueError as exc:
            raise TypeError(
                f"Continuation data contains unsupported tool-policy value {policy_value!r}."
            ) from exc
    elif profile_type == "tool_policy_profile":
        tool_policy = ToolPolicyProfile(
            allowed_tools=tuple(policy.get("allowed_tools") or ()),
            disallowed_tools=tuple(policy.get("disallowed_tools") or ()),
            strict_mcp_config=bool(policy.get("strict_mcp_config", True)),
        )
    else:
        raise TypeError("Continuation data is malformed.")
    return ToolAccess(
        kind=cast(str, kind),
        workspace=Path(workspace) if workspace is not None else None,
        tool_policy=tool_policy,
    )


@dataclasses.dataclass(frozen=True, init=False)
class Continuation:
    serialized: str

    @classmethod
    def for_session_backed_provider(
        cls,
        *,
        selected_service: str,
        selected_model: str,
        selected_effort: str,
        tool_access: ToolAccess,
        provider_session_id: str | None = None,
        provider_state_dir_relpath: str | None = None,
        exact_transcript_match: bool | None = None,
        run_kind: str | None = None,
    ) -> Continuation:
        provider_resume_state: dict[str, Any] = {}
        if run_kind is not None:
            provider_resume_state["run_kind"] = run_kind
        if provider_session_id is not None:
            provider_resume_state["provider_session_id"] = provider_session_id
        if provider_state_dir_relpath is not None:
            provider_resume_state["provider_state_dir_relpath"] = (
                provider_state_dir_relpath
            )
        if exact_transcript_match is not None:
            provider_resume_state["exact_transcript_match"] = exact_transcript_match
        return cls(
            selected_service=selected_service,
            selected_model=selected_model,
            selected_effort=selected_effort,
            tool_access=tool_access,
            provider_resume_state=provider_resume_state,
        )

    def __init__(
        self,
        serialized: str | None = None,
        *,
        selected_service: str | None = None,
        selected_model: str | None = None,
        selected_effort: str | None = None,
        tool_access: ToolAccess | None = None,
        provider_resume_state: Any = None,
    ) -> None:
        if serialized is None:
            if (
                selected_service is None
                or selected_model is None
                or selected_effort is None
                or tool_access is None
                or provider_resume_state is None
            ):
                raise TypeError(
                    "Continuation requires either `serialized` or all legacy "
                    "continuation fields."
                )
            _require_json_compatible_resume_state(provider_resume_state)
            serialized = _PortableContinuationPayload(
                service_name=selected_service,
                model=selected_model,
                effort=selected_effort,
                tool_access=tool_access,
                provider_resume_state=provider_resume_state,
            ).serialized

        object.__setattr__(self, "serialized", serialized)

    @property
    def resume_facts(self) -> ContinuationResumeFacts:
        payload = self._payload()
        return ContinuationResumeFacts(
            selected=ResolvedProvider(
                service=payload.service_name,
                model=payload.model,
                effort=payload.effort,
            ),
            tool_access=payload.tool_access,
            provider_resume_state=payload.provider_resume_state,
        )

    @property
    def session_backed_facts(self) -> SessionBackedContinuationFacts:
        resume_facts = self.resume_facts
        provider_resume_state = resume_facts.provider_resume_state
        if not isinstance(provider_resume_state, dict):
            raise TypeError("Continuation provider_resume_state must be a JSON object.")
        return SessionBackedContinuationFacts(
            selected=resume_facts.selected,
            tool_access=resume_facts.tool_access,
            provider_resume_state=provider_resume_state,
            provider_session_id=cast(
                str | None,
                provider_resume_state.get("provider_session_id"),
            ),
            provider_state_dir_relpath=cast(
                str | None,
                provider_resume_state.get("provider_state_dir_relpath"),
            ),
            exact_transcript_match=cast(
                bool | None,
                provider_resume_state.get("exact_transcript_match"),
            ),
            run_kind=cast(str | None, provider_resume_state.get("run_kind")),
        )

    @property
    def service_name(self) -> str:
        return self.resume_facts.selected.service

    @property
    def model(self) -> str:
        return self.resume_facts.selected.model

    @property
    def effort(self) -> str:
        return self.resume_facts.selected.effort

    @property
    def provider_resume_state(self) -> Any:
        return self.resume_facts.provider_resume_state

    @property
    def tool_access(self) -> ToolAccess:
        return self.resume_facts.tool_access

    def _payload(self) -> "_PortableContinuationPayload":
        return _PortableContinuationPayload.from_serialized(self.serialized)


@dataclasses.dataclass(frozen=True)
class ContinuationResumeFacts:
    selected: ResolvedProvider
    tool_access: ToolAccess
    provider_resume_state: Any


@dataclasses.dataclass(frozen=True)
class SessionBackedContinuationFacts:
    selected: ResolvedProvider
    tool_access: ToolAccess
    provider_resume_state: dict[str, Any]
    provider_session_id: str | None
    provider_state_dir_relpath: str | None
    exact_transcript_match: bool | None
    run_kind: str | None


@dataclasses.dataclass(frozen=True)
class ProviderAuth:
    claude_code_oauth_token: str | None = None
    opencode_api_key: str | None = None

    def __repr__(self) -> str:
        return (
            "ProviderAuth("
            "claude_code_oauth_token="
            f"{_redacted_credential_value(self.claude_code_oauth_token)}, "
            f"opencode_api_key={_redacted_credential_value(self.opencode_api_key)})"
        )


@dataclasses.dataclass(frozen=True)
class AgentEvent:
    type: Literal["agent_message", "agent_tool_call", "turn_summary", "other"]
    display_message: str
    raw_provider_output: str


@dataclasses.dataclass(frozen=True)
class RunResult:
    """Run facts carried by every ``RuntimeOutcome``, present even after interruption."""

    output: str
    usage: ProviderUsage | None
    continuation: Continuation | None
    selected: ResolvedProvider


@dataclasses.dataclass(frozen=True)
class Completed:
    pass


@dataclasses.dataclass(frozen=True)
class UsageLimited:
    reset_time: datetime | None


@dataclasses.dataclass(frozen=True)
class ProviderUnavailable:
    reason: ProviderUnavailableReason
    detail: str


@dataclasses.dataclass(frozen=True)
class Cancelled:
    pass


@dataclasses.dataclass(frozen=True)
class TimedOut:
    pass


OutcomeKind = Completed | UsageLimited | ProviderUnavailable | Cancelled | TimedOut


@dataclasses.dataclass(frozen=True)
class RuntimeOutcome:
    kind: OutcomeKind
    result: RunResult


@dataclasses.dataclass(frozen=True, init=False)
class EphemeralRunRequest:
    prompt: str
    invocation_dir: Path
    provider_selection: ProviderSelection
    tool_access: ToolAccess
    timeout_seconds: int = 300
    on_live_output: Callable[[AgentEvent], None] | None = None
    token: CancellationToken | None = None
    argv_transform: (
        Callable[[tuple[str, ...], Path, Mapping[str, str]], tuple[str, ...]] | None
    ) = None

    def __init__(
        self,
        prompt: str,
        invocation_dir: Path | None = None,
        provider_selection: ProviderSelection | None = None,
        tool_policy: ToolPolicy | ToolPolicyProfile | object = _MISSING_TOOL_POLICY,
        tool_access: ToolAccess | object = _MISSING_TOOL_POLICY,
        timeout_seconds: int = 300,
        token: CancellationToken | None = None,
        on_live_output: Callable[[AgentEvent], None] | None = None,
        **compatibility_kwargs: Any,
    ) -> None:
        normalized_request = _lifecycle_request_facts_module._ephemeral_run_request_facts(
            invocation_dir=invocation_dir,
            compatibility_kwargs=compatibility_kwargs,
            provider_selection=provider_selection,
            tool_access=tool_access,
            tool_policy=tool_policy,
            missing_sentinel=_MISSING_TOOL_POLICY,
            session_namespace=_DEFAULT_EPHEMERAL_SESSION_NAMESPACE,
            context="EphemeralRunRequest",
            missing_message="EphemeralRunRequest requires an explicit `tool_policy` value.",
            public_invocation_dir_name=_PUBLIC_INVOCATION_DIR_NAME,
        )

        object.__setattr__(self, "prompt", prompt)
        object.__setattr__(
            self,
            _PUBLIC_INVOCATION_DIR_NAME,
            normalized_request.invocation_dir,
        )
        object.__setattr__(
            self,
            "provider_selection",
            normalized_request.provider_selection,
        )
        object.__setattr__(self, "tool_access", normalized_request.tool_access)
        object.__setattr__(self, "timeout_seconds", timeout_seconds)
        object.__setattr__(self, "on_live_output", on_live_output)
        object.__setattr__(self, "token", token)
        object.__setattr__(self, "argv_transform", normalized_request.argv_transform)

    @property
    def mount_path(self) -> Path:
        return self.invocation_dir

    @property
    def tool_policy(self) -> ToolPolicy | ToolPolicyProfile:
        return self.tool_access.tool_policy


@dataclasses.dataclass(frozen=True, init=False)
class NewSessionRunRequest:
    prompt: str
    invocation_dir: Path
    provider_selection: ProviderSelection
    tool_access: ToolAccess
    timeout_seconds: int = 300
    name: str = "Runtime Agent"
    status_display: Any = None
    work_body: str = ""
    on_live_output: Callable[[AgentEvent], None] | None = None
    token: CancellationToken | None = None
    argv_transform: (
        Callable[[tuple[str, ...], Path, Mapping[str, str]], tuple[str, ...]] | None
    ) = None

    if TYPE_CHECKING:
        session_store: Path | None = None
        _session_namespace: str = ""

    def __init__(
        self,
        prompt: str,
        invocation_dir: Path | None = None,
        provider_selection: ProviderSelection | None = None,
        tool_policy: ToolPolicy | ToolPolicyProfile | object = _MISSING_TOOL_POLICY,
        tool_access: ToolAccess | object = _MISSING_TOOL_POLICY,
        session_store: Path | None = None,
        _session_namespace: str = "",
        timeout_seconds: int = 300,
        name: str = "Runtime Agent",
        status_display: Any = None,
        work_body: str = "",
        on_live_output: Callable[[AgentEvent], None] | None = None,
        token: CancellationToken | None = None,
        **compatibility_kwargs: Any,
    ) -> None:
        normalized_request = _lifecycle_request_facts_module._new_session_run_request_facts(
            invocation_dir=invocation_dir,
            compatibility_kwargs=compatibility_kwargs,
            provider_selection=provider_selection,
            tool_access=tool_access,
            tool_policy=tool_policy,
            session_store=session_store,
            session_namespace=_session_namespace,
            missing_sentinel=_MISSING_TOOL_POLICY,
            context="NewSessionRunRequest",
            missing_message="NewSessionRunRequest requires an explicit `tool_policy` value.",
            public_invocation_dir_name=_PUBLIC_INVOCATION_DIR_NAME,
        )

        object.__setattr__(self, "prompt", prompt)
        object.__setattr__(
            self,
            _PUBLIC_INVOCATION_DIR_NAME,
            normalized_request.invocation_dir,
        )
        object.__setattr__(self, "session_store", normalized_request.session_store)
        object.__setattr__(
            self,
            "provider_selection",
            normalized_request.provider_selection,
        )
        object.__setattr__(self, "tool_access", normalized_request.tool_access)
        object.__setattr__(
            self, "_session_namespace", normalized_request.session_namespace
        )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "status_display", status_display)
        object.__setattr__(self, "work_body", work_body)
        object.__setattr__(self, "timeout_seconds", timeout_seconds)
        object.__setattr__(self, "on_live_output", on_live_output)
        object.__setattr__(self, "token", token)
        object.__setattr__(self, "argv_transform", normalized_request.argv_transform)

    @property
    def mount_path(self) -> Path:
        return self.invocation_dir

    @property
    def tool_policy(self) -> ToolPolicy | ToolPolicyProfile:
        return self.tool_access.tool_policy

    @property
    def _runtime_state_dir(self) -> Path | None:
        return self.session_store


@dataclasses.dataclass(frozen=True, init=False)
class ResumedSessionRunRequest:
    prompt: str
    invocation_dir: Path
    model: str
    effort: str
    continuation: Continuation | None
    provider_auth: ProviderAuth | None
    tool_access: ToolAccess
    timeout_seconds: int = 300
    name: str = "Runtime Agent"
    status_display: Any = None
    work_body: str = ""
    on_live_output: Callable[[AgentEvent], None] | None = None
    token: CancellationToken | None = None
    argv_transform: (
        Callable[[tuple[str, ...], Path, Mapping[str, str]], tuple[str, ...]] | None
    ) = None

    if TYPE_CHECKING:
        session_store: Path | None = None
        _session_namespace: str = ""

    def __init__(
        self,
        prompt: str,
        invocation_dir: Path | None = None,
        continuation: Continuation | None = None,
        provider_auth: ProviderAuth | None = None,
        session_store: Path | None = None,
        _session_namespace: str = "",
        tool_access: ToolAccess | object = _MISSING_TOOL_POLICY,
        timeout_seconds: int = 300,
        name: str = "Runtime Agent",
        status_display: Any = None,
        work_body: str = "",
        on_live_output: Callable[[AgentEvent], None] | None = None,
        token: CancellationToken | None = None,
        **compatibility_kwargs: Any,
    ) -> None:
        normalized_request = (
            _lifecycle_request_facts_module._resumed_session_run_request_facts(
                invocation_dir=invocation_dir,
                compatibility_kwargs=compatibility_kwargs,
                continuation=continuation,
                tool_access=tool_access,
                session_store=session_store,
                session_namespace=_session_namespace,
                context="ResumedSessionRunRequest",
                public_invocation_dir_name=_PUBLIC_INVOCATION_DIR_NAME,
            )
        )

        object.__setattr__(self, "prompt", prompt)
        object.__setattr__(
            self,
            _PUBLIC_INVOCATION_DIR_NAME,
            normalized_request.invocation_dir,
        )
        object.__setattr__(self, "session_store", normalized_request.session_store)
        object.__setattr__(self, "model", normalized_request.model)
        object.__setattr__(self, "effort", normalized_request.effort)
        object.__setattr__(
            self,
            "_session_namespace",
            normalized_request.session_namespace,
        )
        object.__setattr__(self, "continuation", continuation)
        object.__setattr__(self, "provider_auth", provider_auth)
        object.__setattr__(self, "tool_access", normalized_request.tool_access)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "status_display", status_display)
        object.__setattr__(self, "work_body", work_body)
        object.__setattr__(self, "timeout_seconds", timeout_seconds)
        object.__setattr__(self, "on_live_output", on_live_output)
        object.__setattr__(self, "token", token)
        object.__setattr__(self, "argv_transform", normalized_request.argv_transform)

    @property
    def mount_path(self) -> Path:
        return self.invocation_dir

    @property
    def tool_policy(self) -> ToolPolicy | ToolPolicyProfile:
        return self.tool_access.tool_policy

    @property
    def _runtime_state_dir(self) -> Path | None:
        return self.session_store


cast(Any, EphemeralRunRequest).__signature__ = _public_request_signature(
    "prompt",
    "invocation_dir",
    "provider_selection",
    "tool_policy",
    "timeout_seconds",
    "token",
    "on_live_output",
    "argv_transform",
)
cast(Any, NewSessionRunRequest).__signature__ = _public_request_signature(
    "prompt",
    "invocation_dir",
    "provider_selection",
    "tool_policy",
    "session_store",
    "timeout_seconds",
    "name",
    "status_display",
    "work_body",
    "token",
    "on_live_output",
    "argv_transform",
)
cast(Any, ResumedSessionRunRequest).__signature__ = _public_request_signature(
    "prompt",
    "invocation_dir",
    "continuation",
    "provider_auth",
    "session_store",
    "timeout_seconds",
    "on_live_output",
    "token",
    "argv_transform",
)
