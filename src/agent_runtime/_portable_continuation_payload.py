from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any
from typing import cast

from .contracts import ToolPolicy, ToolPolicyProfile, ToolAccess
from ._runtime_lifecycle import Continuation


@dataclasses.dataclass(frozen=True)
class PortableContinuationPayload:
    service_name: str
    model: str
    effort: str
    tool_access: ToolAccess
    provider_resume_state: dict[str, Any]

    @property
    def serialized(self) -> str:
        return json.dumps(self._payload_state(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_continuation(
        cls,
        continuation: Continuation,
    ) -> PortableContinuationPayload:
        return cls.from_serialized(continuation.serialized)

    @classmethod
    def from_serialized(cls, serialized: str) -> PortableContinuationPayload:
        try:
            payload = json.loads(serialized)
        except json.JSONDecodeError as exc:
            raise TypeError("Continuation data is not valid JSON.") from exc
        if not isinstance(payload, dict):
            raise TypeError("Continuation data must be a JSON object.")
        tool_access = payload.get("tool_access")
        serialized_provider_resume_state = payload.get("provider_resume_state")
        if not isinstance(serialized_provider_resume_state, dict):
            raise TypeError("Continuation provider_resume_state must be a JSON object.")
        return cls(
            service_name=_payload_service_name(payload.get("service_name")),
            model=_payload_service_name(payload.get("model")),
            effort=_payload_service_name(payload.get("effort")),
            tool_access=_deserialize_tool_access(tool_access),
            provider_resume_state=serialized_provider_resume_state,
        )

    def to_continuation(self) -> Continuation:
        return Continuation(serialized=self.serialized)

    def _payload_state(self) -> dict[str, Any]:
        return {
            "service_name": self.service_name,
            "model": self.model,
            "effort": self.effort,
            "tool_access": _serialize_tool_access(self.tool_access),
            "provider_resume_state": self.provider_resume_state,
        }


def read_portable_continuation_payload(
    continuation: Continuation,
) -> PortableContinuationPayload:
    return PortableContinuationPayload.from_serialized(continuation.serialized)


def create_portable_continuation_payload(
    *,
    service_name: str,
    model: str,
    effort: str,
    tool_access: ToolAccess,
    provider_resume_state: dict[str, Any],
) -> PortableContinuationPayload:
    return PortableContinuationPayload(
        service_name=service_name,
        model=model,
        effort=effort,
        tool_access=tool_access,
        provider_resume_state=provider_resume_state,
    )


def _payload_service_name(value: Any) -> str:
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
