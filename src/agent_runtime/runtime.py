from __future__ import annotations

import json
import re
import shlex
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

from . import _time as _time_module
from .contracts import (
    AssistantTurn,
    CredentialFailure,
    HardError,
    PromptTokens,
    Result,
    ToolAccess,
    ToolPolicy,
    ToolPolicyProfile,
    TransientError,
    UsageLimit,
)
from .execution_contracts import (
    PromptRunRequest as _PromptRunRequest,
    PromptRuntimeExecutionAdapter as _PromptRuntimeExecutionAdapter,
    TextOutputAdapter,
    WorkInvocationPresentation,
    WorktreeMount,
)
from .errors import (
    AgentCredentialFailureError,
    RuntimeConfigurationError,
    UsageLimitError,
)
from .invocation_progress import InvocationProgress
from .provider_errors import ProviderErrorObservation
from .provider_output import reduce_text_output_events
from . import _runtime_facade_lifecycle as _runtime_facade_lifecycle_module
from ._runtime_lifecycle import (
    Continuation,
    EphemeralResultMetadata,
    EphemeralRunRequest,
    EphemeralRunResult,
    EphemeralRuntimeMetadata,
    NewSessionRunRequest,
    ProviderAuth,
    ResumedSessionRunRequest,
    RuntimeOutcome,
    SessionRunResult,
    SessionRuntimeMetadata,
)
from .service_registry import ServiceRegistry
from .session import RunKind
from .stage_priority_chain import iter_stage_chain
from .types import StageSelection
from .usage_limit_scope import UsageLimitScope

__all__ = [
    "Continuation",
    "EphemeralRunRequest",
    "EphemeralRunResult",
    "EphemeralResultMetadata",
    "EphemeralRuntime",
    "EphemeralRuntimeExecutionAdapter",
    "EphemeralRuntimeMetadata",
    "NewSessionRunRequest",
    "NewSessionRuntime",
    "NewSessionRuntimeExecutionAdapter",
    "InvocationProgress",
    "ProviderAuth",
    "ResumedSessionRunRequest",
    "ResumedSessionRuntime",
    "ResumedSessionRuntimeExecutionAdapter",
    "RuntimeClient",
    "RuntimeOutcome",
    "SessionRunResult",
    "SessionRuntimeMetadata",
    "ToolAccess",
    "ToolPolicy",
    "ToolPolicyProfile",
    "WorktreeMount",
]

EphemeralRuntimeExecutionAdapter = _PromptRuntimeExecutionAdapter
NewSessionRuntimeExecutionAdapter = _PromptRuntimeExecutionAdapter
ResumedSessionRuntimeExecutionAdapter = _PromptRuntimeExecutionAdapter

_RuntimeIntent = _runtime_facade_lifecycle_module._RuntimeIntent
_EphemeralPreparedProviderRunSession = (
    _runtime_facade_lifecycle_module._EphemeralPreparedProviderRunSession
)
_EphemeralPreparedRunSessionState = (
    _runtime_facade_lifecycle_module._EphemeralPreparedRunSessionState
)
_TrackedPreparedSessionState = (
    _runtime_facade_lifecycle_module._TrackedPreparedSessionState
)
_require_execution_adapter_method = (
    _runtime_facade_lifecycle_module._require_execution_adapter_method
)
_build_run_session = _runtime_facade_lifecycle_module._build_run_session
_latest_provider_run_session = (
    _runtime_facade_lifecycle_module._latest_provider_run_session
)
_invoke_runtime_intent = _runtime_facade_lifecycle_module._invoke_runtime_intent
_run_ephemeral = _runtime_facade_lifecycle_module._run_ephemeral
_run_new_session = _runtime_facade_lifecycle_module._run_new_session
_run_resumed_session = _runtime_facade_lifecycle_module._run_resumed_session
_run_resumed_session_outcome = (
    _runtime_facade_lifecycle_module._run_resumed_session_outcome
)
_provider_state_dir_container_path = (
    _runtime_facade_lifecycle_module._provider_state_dir_container_path
)
_continuation_resume_state = _runtime_facade_lifecycle_module._continuation_resume_state
_build_continuation = _runtime_facade_lifecycle_module._build_continuation
_interruption_continuation = _runtime_facade_lifecycle_module._interruption_continuation
_coerce_service_registry = _runtime_facade_lifecycle_module._coerce_service_registry
_run_ephemeral_outcome = _runtime_facade_lifecycle_module._run_ephemeral_outcome
_run_new_session_outcome = _runtime_facade_lifecycle_module._run_new_session_outcome

for _runtime_export in (
    Continuation,
    EphemeralResultMetadata,
    EphemeralRunRequest,
    EphemeralRunResult,
    EphemeralRuntimeMetadata,
    NewSessionRunRequest,
    ProviderAuth,
    ResumedSessionRunRequest,
    RuntimeOutcome,
    SessionRunResult,
    SessionRuntimeMetadata,
):
    _runtime_export.__module__ = __name__

_CLAUDE_VALID_MODELS = frozenset({"haiku", "sonnet", "opus"})
_CLAUDE_VALID_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})
_CLAUDE_SUBSCRIPTION_ACCESS_DENIAL_PHRASE = (
    "disabled Claude subscription access for Claude Code"
)
_CLAUDE_RESET_PATTERN = re.compile(
    r"resets\s+"
    r"(?:(?P<month>[A-Za-z]+)\s+(?P<day>\d{1,2}),\s+)?"
    r"(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?(?P<ampm>am|pm)\s+\(UTC\)",
    re.IGNORECASE,
)
_CLAUDE_MONTHS = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sept": 9,
    "sep": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}


def _selected_service_path(
    override: StageSelection,
    *,
    selected_service: str,
) -> tuple[str, ...]:
    path: list[str] = []
    for node in iter_stage_chain(override):
        if not node.service:
            continue
        path.append(node.service)
        if node.service == selected_service:
            return tuple(path)
    return (selected_service,)


def _validate_claude_stage(stage: StageSelection) -> None:
    if stage.model not in _CLAUDE_VALID_MODELS:
        raise RuntimeConfigurationError(f"Unsupported Claude model {stage.model!r}.")
    if stage.effort not in _CLAUDE_VALID_EFFORTS:
        raise RuntimeConfigurationError(f"Unsupported Claude effort {stage.effort!r}.")


def _claude_command(
    *,
    model: str,
    effort: str,
    tool_access: ToolAccess,
    prompt_path: Path,
) -> str:
    profile = (
        tool_access.tool_policy.profile
        if isinstance(tool_access.tool_policy, ToolPolicy)
        else tool_access.tool_policy
    )
    flags = (
        "--verbose --dangerously-skip-permissions --output-format stream-json -p -"
        " --disable-slash-commands --exclude-dynamic-system-prompt-sections"
    )
    if profile.allowed_tools is not None:
        flags += f" --tools {shlex.quote(' '.join(profile.allowed_tools))}"
    if profile.disallowed_tools:
        flags += f' --disallowedTools "{" ".join(profile.disallowed_tools)}"'
    if profile.strict_mcp_config:
        flags += " --strict-mcp-config --mcp-config '{\"mcpServers\":{}}'"
    if model:
        flags += f" --model {model}"
    if effort:
        flags += f" --effort {effort}"
    return f"claude {flags} < {shlex.quote(str(prompt_path))}"


def _claude_env(
    *,
    auth: ProviderAuth | None,
    state_dir_container_path: str | None = None,
) -> dict[str, str]:
    env: dict[str, str] = {}
    token = None if auth is None else auth.claude_code_oauth_token
    if token:
        env["CLAUDE_CODE_OAUTH_TOKEN"] = token
    if state_dir_container_path:
        env["CLAUDE_CONFIG_DIR"] = state_dir_container_path
    return env


def _is_claude_subscription_access_denial(event: dict[str, Any]) -> bool:
    result = event.get("result")
    return (
        event.get("is_error") is True
        and event.get("api_error_status") == 403
        and isinstance(result, str)
        and _CLAUDE_SUBSCRIPTION_ACCESS_DENIAL_PHRASE.lower() in result.lower()
    )


def _parse_claude_event(line: str) -> list[Any]:
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return []
    if not isinstance(event, dict):
        return []
    if event.get("api_error_status") == 429:
        reset_time = _parse_claude_reset_time(event.get("result"))
        return [
            UsageLimit(
                reset_time=reset_time,
                raw_message=None if reset_time is not None else line,
            )
        ]
    if _is_claude_subscription_access_denial(event):
        denial_message = event.get("result")
        return [
            CredentialFailure(
                raw_message=line,
                service_name="claude",
                source_observations=(
                    ProviderErrorObservation(
                        service_name="claude",
                        raw_provider_text=(
                            denial_message if isinstance(denial_message, str) else line
                        ),
                        source_stream="json_event.result",
                        status_code=403,
                    ),
                ),
                status_code=403,
            )
        ]
    if event.get("is_error") and event.get("type") == "result":
        status = event.get("api_error_status")
        if status is None or (isinstance(status, int) and status >= 500):
            return [
                TransientError(
                    status_code=status if isinstance(status, int) else None,
                    raw_message=line,
                )
            ]
        if isinstance(status, int) and 400 <= status < 500:
            return [HardError(status_code=status, raw_message=line)]
        return []
    if event.get("type") == "assistant":
        message = event.get("message") or {}
        content = message.get("content") or []
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        usage = message.get("usage") or {}
        total_tokens = (
            int(usage.get("input_tokens") or 0)
            + int(usage.get("cache_creation_input_tokens") or 0)
            + int(usage.get("cache_read_input_tokens") or 0)
        )
        parsed_events: list[Any] = []
        if total_tokens > 0:
            parsed_events.append(PromptTokens(count=total_tokens))
        if parts:
            parsed_events.append(AssistantTurn(text="\n\n".join(parts)))
        return parsed_events
    if (
        event.get("type") == "result"
        and event.get("is_error") is not True
        and isinstance(event.get("result"), str)
    ):
        return [Result(text=cast(str, event["result"]))]
    return []


def _reduce_claude_stream(lines: list[str]) -> str:
    parsed_events: list[Any] = []
    for line in lines:
        parsed_events.extend(_parse_claude_event(line))
    return reduce_text_output_events(
        parsed_events,
        lambda _turn: None,
        provider="claude",
    )


def _parse_claude_reset_time(retry_text: object) -> datetime | None:
    if not isinstance(retry_text, str):
        return None
    match = _CLAUDE_RESET_PATTERN.search(retry_text)
    if match is None:
        return None
    hour = int(match.group("hour"))
    if not 1 <= hour <= 12:
        return None
    ampm = match.group("ampm").lower()
    if ampm == "pm" and hour != 12:
        hour += 12
    elif ampm == "am" and hour == 12:
        hour = 0
    minute = int(match.group("minute") or 0)
    if not 0 <= minute <= 59:
        return None
    now_local = _time_module.now_local()
    utc_now = now_local.astimezone(timezone.utc)
    month_text = match.group("month")
    day_text = match.group("day")
    if month_text is not None or day_text is not None:
        if month_text is None or day_text is None:
            return None
        month = _CLAUDE_MONTHS.get(month_text.lower())
        if month is None:
            return None
        utc_dt = datetime(
            utc_now.year,
            month,
            int(day_text),
            hour,
            minute,
            tzinfo=timezone.utc,
        )
        local_dt = utc_dt.astimezone(now_local.tzinfo)
        if local_dt < now_local - timedelta(days=31):
            return datetime(
                utc_dt.year + 1,
                month,
                int(day_text),
                hour,
                minute,
                tzinfo=timezone.utc,
            ).astimezone(now_local.tzinfo)
        return local_dt
    utc_dt = datetime.combine(
        utc_now.date(),
        datetime.min.time(),
        tzinfo=timezone.utc,
    ).replace(hour=hour, minute=minute)
    if utc_dt < utc_now - timedelta(minutes=2):
        utc_dt += timedelta(days=1)
    return utc_dt.astimezone(now_local.tzinfo)


def _select_builtin_stage(stage: StageSelection) -> StageSelection:
    for candidate in iter_stage_chain(stage):
        if candidate.service == "claude":
            return candidate
    raise RuntimeConfigurationError(
        "RuntimeClient requires at least one supported built-in service candidate."
    )


class EphemeralRuntime:
    def __init__(
        self,
        *,
        execution_adapter: EphemeralRuntimeExecutionAdapter,
        service_registry: ServiceRegistry | dict[str, Any] | None = None,
    ) -> None:
        self._service_registry = _coerce_service_registry(service_registry)
        self._execution_adapter = execution_adapter

    async def run_ephemeral(self, request: EphemeralRunRequest) -> RuntimeOutcome:
        return await _run_ephemeral_outcome(
            runner=self._execution_adapter,
            service_registry=self._service_registry,
            request=request,
        )


class NewSessionRuntime:
    def __init__(
        self,
        *,
        execution_adapter: NewSessionRuntimeExecutionAdapter,
        service_registry: ServiceRegistry | dict[str, Any] | None = None,
    ) -> None:
        self._service_registry = _coerce_service_registry(service_registry)
        self._execution_adapter = execution_adapter

    async def run_new_session(self, request: NewSessionRunRequest) -> RuntimeOutcome:
        return await _run_new_session_outcome(
            runner=self._execution_adapter,
            service_registry=self._service_registry,
            request=request,
        )


class ResumedSessionRuntime:
    def __init__(
        self,
        *,
        execution_adapter: ResumedSessionRuntimeExecutionAdapter,
    ) -> None:
        self._execution_adapter = execution_adapter

    async def run_resumed_session(
        self,
        request: ResumedSessionRunRequest,
    ) -> RuntimeOutcome:
        return await _run_resumed_session_outcome(
            runner=self._execution_adapter,
            request=request,
        )


class RuntimeClient:
    def run_ephemeral(self, request: EphemeralRunRequest) -> RuntimeOutcome:
        try:
            result = _run_builtin_ephemeral(request)
        except UsageLimitError as exc:
            return RuntimeOutcome.usage_limited(
                output="",
                service_name=exc.service_name,
                reset_time=exc.reset_time,
                usage_limit_scope=exc.usage_limit_scope
                or UsageLimitScope(request.role.value),
                invocation_progress=exc.invocation_progress,
                continuation=exc.continuation,
            )
        return RuntimeOutcome.completed(output=result.output, result=result)


def _run_builtin_ephemeral(request: EphemeralRunRequest) -> EphemeralRunResult:
    selected_stage = _select_builtin_stage(request.stage)
    _validate_claude_stage(selected_stage)
    if request.auth is None or not request.auth.claude_code_oauth_token:
        raise AgentCredentialFailureError(
            message="Missing Claude Code OAuth token.",
            service_name="claude",
            observations=(),
        )
    prompt_path = request.worktree / ".pycastle_prompt"
    prompt_path.write_text(request.prompt)
    try:
        process = subprocess.Popen(
            _claude_command(
                model=selected_stage.model,
                effort=selected_stage.effort,
                tool_access=request.tool_access,
                prompt_path=prompt_path,
            ),
            shell=True,
            cwd=request.worktree,
            env=_claude_env(auth=request.auth),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout_lines = [] if process.stdout is None else list(process.stdout)
        result_text = _reduce_claude_stream(stdout_lines)
        process.wait()
    finally:
        prompt_path.unlink(missing_ok=True)
    selected_service_path = _selected_service_path(
        request.stage,
        selected_service="claude",
    )
    result = EphemeralRunResult(
        output=result_text,
        selected_service="claude",
        selected_model=selected_stage.model,
        selected_effort=selected_stage.effort,
        tool_access=request.tool_access,
        used_fallback=len(selected_service_path) > 1,
        metadata=EphemeralResultMetadata(
            selected_service_path=selected_service_path,
            runtime=EphemeralRuntimeMetadata(
                run_kind=RunKind.FRESH,
                session_namespace=request.session_namespace,
            ),
        ),
    )
    return result


async def _run_prompt(
    *,
    runner: _PromptRuntimeExecutionAdapter,
    service_registry: ServiceRegistry,
    request: _PromptRunRequest,
) -> str:
    resolved_override = service_registry.resolve(
        request.stage,
        _time_module.now_local(),
    )
    role = request.role
    resolve_service = _require_execution_adapter_method(runner, "resolve_service")
    build_work_dependencies = _require_execution_adapter_method(
        runner,
        "build_work_dependencies",
    )
    resolved_service = resolve_service(resolved_override.service)
    dependencies = build_work_dependencies(
        name=request.name,
        model=resolved_override.model,
        effort=resolved_override.effort,
        service=resolved_service,
    )
    return await _invoke_runtime_intent(
        _RuntimeIntent(
            run_session=_build_run_session(
                mount_path=request.mount_path,
                role=role,
                session_namespace=request.session_namespace,
                service=resolved_service,
                container_workspace=dependencies.execution.container_workspace,
                usage_limit_scope=request.usage_limit_scope,
            ),
            model=resolved_override.model,
            effort=resolved_override.effort,
            output_adapter=TextOutputAdapter(
                prompt=request.prompt,
                tool_access=request.tool_access,
                workspace=request.worktree.host_path,
            ),
            dependencies=dependencies,
            presentation=WorkInvocationPresentation(
                name=request.name,
                status_display=request.status_display,
                work_body=request.work_body,
            ),
            token=request.token,
        )
    )
