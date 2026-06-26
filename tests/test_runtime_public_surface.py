from __future__ import annotations

import importlib
import inspect
import asyncio
from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from typing import Any
import unittest.mock

import pytest

import agent_runtime as runtime
import agent_runtime.contracts as contracts_runtime
import agent_runtime._provider_invocation as provider_invocation_runtime
import agent_runtime._portable_continuation_payload as continuation_payload_module
import agent_runtime.runtime as prompt_runtime
import agent_runtime.session as session_runtime
from agent_runtime.errors import AgentRuntimeError
from agent_runtime.provider_usage import ProviderUsage
from agent_runtime.session import RunKind


def test_package_exports_runtime_surface() -> None:
    assert runtime.__all__ == [
        "AgentCredentialFailureError",
        "AgentEvent",
        "AgentFailedError",
        "AgentRuntimeError",
        "AgentTimeoutError",
        "Cancelled",
        "ClaudeCodeOAuthToken",
        "Completed",
        "Continuation",
        "HardAgentError",
        "ProviderUnavailable",
        "ProviderAuth",
        "ProviderSelection",
        "ProviderUsage",
        "ResolvedProvider",
        "RunResult",
        "RuntimeClient",
        "RuntimeConfigurationError",
        "RuntimeOutcome",
        "RunKind",
        "TimedOut",
        "ToolPolicy",
        "TransientAgentError",
        "UsageLimited",
        "UsageLimitError",
    ]
    assert not hasattr(runtime, "InvocationRecord")
    assert not hasattr(runtime, "InvocationProgress")
    assert runtime.ProviderSelection.__module__.startswith("agent_runtime")
    assert not hasattr(runtime, "StageOverride")
    assert runtime.AgentRuntimeError is AgentRuntimeError
    assert runtime.RuntimeOutcome is prompt_runtime.RuntimeOutcome
    assert "ToolAccess" not in runtime.__all__
    assert "ToolPolicyProfile" not in runtime.__all__
    assert "InvocationRole" not in runtime.__all__
    assert "UsageLimitScope" not in runtime.__all__
    assert not hasattr(runtime, "ToolAccess")
    assert not hasattr(runtime, "ToolPolicyProfile")
    assert not hasattr(runtime, "InvocationRole")
    assert not hasattr(runtime, "UsageLimitScope")
    assert hasattr(contracts_runtime, "ToolAccess")
    assert hasattr(contracts_runtime, "ToolPolicyProfile")
    assert not hasattr(runtime, "assert_runtime_import_isolation")
    assert not hasattr(runtime, "run_prompt")
    assert not hasattr(runtime, "ExecutionProvider")
    assert not hasattr(runtime, "ServiceRegistry")
    assert not hasattr(runtime, "ProviderSessionAdapter")
    assert not hasattr(runtime, "ProviderSessionPreferences")
    assert not hasattr(runtime, "ProviderSessionPreferencesRequest")
    assert not hasattr(runtime, "ProviderSessionState")
    assert not hasattr(runtime, "ProviderSessionStateRequest")
    assert not hasattr(prompt_runtime, "PromptRuntime")
    assert not hasattr(prompt_runtime, "PromptRunRequest")
    assert not hasattr(prompt_runtime, "PromptRuntimeExecutionAdapter")
    assert not hasattr(prompt_runtime, "run_ephemeral")
    assert not hasattr(prompt_runtime, "run_prompt")
    assert not hasattr(prompt_runtime, "run_resumable_prompt")
    assert not hasattr(prompt_runtime, "ResidentRunRequest")
    assert not hasattr(prompt_runtime, "ResidentRunResult")
    assert not hasattr(prompt_runtime, "ResidentRuntime")
    assert not hasattr(prompt_runtime, "ResidentRuntimeExecutionAdapter")
    assert not hasattr(prompt_runtime, "ResidentRuntimeMetadata")
    assert {
        "AgentEvent",
        "Cancelled",
        "Completed",
        "Continuation",
        "EphemeralRunRequest",
        "NewSessionRunRequest",
        "ProviderUnavailable",
        "ProviderAuth",
        "ProviderUsage",
        "ResolvedProvider",
        "ResumedSessionRunRequest",
        "RunResult",
        "RuntimeClient",
        "RuntimeOutcome",
        "TimedOut",
        "UsageLimited",
    } <= set(prompt_runtime.__all__)
    assert "ToolAccess" not in prompt_runtime.__all__
    assert "ToolPolicyProfile" not in prompt_runtime.__all__
    assert not hasattr(prompt_runtime, "ToolAccess")
    assert not hasattr(prompt_runtime, "ToolPolicyProfile")
    for removed in (
        "InvocationRecord",
        "EphemeralRunResult",
        "EphemeralResultMetadata",
        "EphemeralRuntimeMetadata",
        "SessionRunResult",
        "SessionRuntimeMetadata",
    ):
        assert removed not in prompt_runtime.__all__
        assert not hasattr(prompt_runtime, removed)
    assert "EphemeralRuntime" not in prompt_runtime.__all__
    assert "NewSessionRuntime" not in prompt_runtime.__all__
    assert "ResumedSessionRuntime" not in prompt_runtime.__all__
    assert "EphemeralRuntimeExecutionAdapter" not in prompt_runtime.__all__
    assert "NewSessionRuntimeExecutionAdapter" not in prompt_runtime.__all__
    assert "ResumedSessionRuntimeExecutionAdapter" not in prompt_runtime.__all__
    assert not hasattr(prompt_runtime, "ResumableRunResult")
    assert not hasattr(prompt_runtime, "ResumableRuntimeMetadata")
    assert "ResumableRunRequest" not in prompt_runtime.__all__
    assert not hasattr(prompt_runtime, "ResumableRunRequest")
    assert "ResumableRuntime" not in prompt_runtime.__all__
    assert "ResumableRuntimeExecutionAdapter" not in prompt_runtime.__all__
    assert "OneShotRunRequest" not in prompt_runtime.__all__
    assert "OneShotRunResult" not in prompt_runtime.__all__
    assert "OneShotResultMetadata" not in prompt_runtime.__all__
    assert "OneShotRuntime" not in prompt_runtime.__all__
    assert "OneShotRuntimeExecutionAdapter" not in prompt_runtime.__all__
    assert "OneShotRuntimeMetadata" not in prompt_runtime.__all__
    assert not hasattr(runtime, "ProviderInvocationRequest")
    assert not hasattr(runtime, "ProviderInvocationResult")
    assert not hasattr(runtime, "ProviderInvocationAdapter")
    assert not hasattr(prompt_runtime, "ProviderInvocationRequest")
    assert not hasattr(prompt_runtime, "ProviderInvocationResult")
    assert not hasattr(prompt_runtime, "ProviderInvocationAdapter")


def test_built_in_provider_invocation_seam_stays_private_to_runtime_public_surface() -> (
    None
):
    with pytest.raises(ImportError):
        exec("from agent_runtime import ProviderInvocationRequest", {}, {})
    with pytest.raises(ImportError):
        exec("from agent_runtime.runtime import ProviderInvocationRequest", {}, {})
    with pytest.raises(ImportError):
        exec("from agent_runtime.runtime import ProviderInvocationResult", {}, {})
    with pytest.raises(ImportError):
        exec("from agent_runtime.runtime import ProviderInvocationAdapter", {}, {})


def test_session_backed_provider_execution_module_stays_private_to_runtime_public_surface() -> (
    None
):
    assert not hasattr(prompt_runtime, "_session_backed_provider_execution_module")
    assert not hasattr(prompt_runtime, "_session_backed_provider_execution")
    assert "_session_backed_provider_execution_module" not in runtime.__all__
    assert "_session_backed_provider_execution_module" not in prompt_runtime.__all__


@pytest.mark.parametrize(
    "module_name",
    [
        "agent_runtime.execution_contracts",
        "agent_runtime.provider_session_adapter",
        "agent_runtime.service_registry",
    ],
)
def test_retired_public_adapter_modules_do_not_expose_runtime_seams(
    module_name: str,
) -> None:
    module = importlib.import_module(module_name)
    assert module.__all__ == []


def test_session_module_exports_only_active_provider_state_helpers() -> None:
    assert session_runtime.__all__ == ["RunKind", "provider_state_relpath"]
    assert session_runtime.RunKind is RunKind
    assert session_runtime.provider_state_relpath("implementer", "codex") == (
        "implementer/codex/"
    )

    for removed_name in (
        "ProviderSessionState",
        "ProviderSessionStateRequest",
        "normalize_state_dir_relpath",
        "provider_state_session_id_path",
        "load_provider_state_session_id",
        "load_state_dir_provider_session_id",
    ):
        with pytest.raises(ImportError):
            exec(f"from agent_runtime.session import {removed_name}", {}, {})
        assert not hasattr(session_runtime, removed_name)


@pytest.mark.parametrize(
    ("module_name", "removed_name"),
    [
        ("agent_runtime", "ToolAccess"),
        ("agent_runtime", "ToolPolicyProfile"),
        ("agent_runtime.runtime", "ToolAccess"),
        ("agent_runtime.runtime", "ToolPolicyProfile"),
    ],
)
def test_removed_tool_policy_compatibility_names_fail_on_ordinary_runtime_surface(
    module_name: str,
    removed_name: str,
) -> None:
    with pytest.raises(ImportError):
        exec(f"from {module_name} import {removed_name}", {}, {})

    imported_module = importlib.import_module(module_name)
    with pytest.raises(AttributeError, match="Runtime Public Surface"):
        getattr(imported_module, removed_name)

    compatibility_imports: dict[str, object] = {}
    exec(
        f"from agent_runtime.contracts import {removed_name}",
        {},
        compatibility_imports,
    )
    assert compatibility_imports[removed_name] is getattr(
        contracts_runtime,
        removed_name,
    )


@pytest.mark.parametrize(
    ("module_name", "removed_name"),
    [("agent_runtime", "InvocationRole"), ("agent_runtime", "UsageLimitScope")],
)
def test_removed_value_object_compatibility_names_fail_on_ordinary_runtime_surface(
    module_name: str,
    removed_name: str,
) -> None:
    with pytest.raises(ImportError):
        exec(f"from {module_name} import {removed_name}", {}, {})

    imported_module = importlib.import_module(module_name)
    with pytest.raises(AttributeError, match="Runtime Public Surface"):
        getattr(imported_module, removed_name)


def test_runtime_client_constructor_stays_on_public_default_surface() -> None:
    signature = inspect.signature(runtime.RuntimeClient)
    assert list(signature.parameters) == []
    unexpected_kwargs: dict[str, object] = {"_provider_invocation_adapter": None}
    with pytest.raises(TypeError):
        runtime.RuntimeClient(**unexpected_kwargs)


def test_built_in_provider_invocation_seam_uses_frozen_contract_values() -> None:
    def reduce_output(lines: list[str]) -> tuple[str, None]:
        return "".join(lines), None

    hooks = provider_invocation_runtime.ProviderOutputReductionHooks(
        reduce_output=reduce_output
    )
    prompt = provider_invocation_runtime.ProviderInvocationPrompt(
        content="prompt body",
        path=Path("/tmp/prompt.txt"),
        cleanup_path=True,
    )
    request = provider_invocation_runtime.ProviderInvocationRequest(
        command="provider --run",
        worktree=Path("/tmp/worktree"),
        environment={"PATH": "/usr/bin"},
        prompt=prompt,
        run_kind=RunKind.FRESH,
        provider_session_id="session-123",
        output_hooks=hooks,
    )
    result = provider_invocation_runtime.ProviderInvocationResult(
        output="ok",
        usage=ProviderUsage(output_tokens=3),
        stdout_lines=("line 1", "line 2"),
        provider_session_id="session-123",
    )

    assert [field.name for field in fields(prompt)] == [
        "content",
        "path",
        "cleanup_path",
    ]
    assert [field.name for field in fields(hooks)] == [
        "reduce_output",
        "extract_provider_session_id",
    ]
    assert [field.name for field in fields(request)] == [
        "worktree",
        "environment",
        "prompt",
        "run_kind",
        "provider_session_id",
        "output_hooks",
        "command",
        "argv",
        "prefer_argv",
        "timeout_seconds",
    ]
    assert [field.name for field in fields(result)] == [
        "output",
        "usage",
        "stdout_lines",
        "provider_session_id",
    ]
    assert request.prompt.cleanup_path is True
    assert request.output_hooks.reduce_output(["a", "b"]) == ("ab", None)
    assert result.stdout_lines == ("line 1", "line 2")
    with pytest.raises(FrozenInstanceError):
        setattr(request, "command", "changed")
    with pytest.raises(FrozenInstanceError):
        setattr(result, "output", "changed")


def test_built_in_provider_invocation_request_signature_excludes_logging_context() -> (
    None
):
    assert tuple(
        inspect.signature(
            provider_invocation_runtime.ProviderInvocationRequest
        ).parameters
    ) == (
        "worktree",
        "environment",
        "prompt",
        "run_kind",
        "provider_session_id",
        "output_hooks",
        "command",
        "argv",
        "prefer_argv",
        "timeout_seconds",
    )


def test_runtime_star_import_uses_lifecycle_surface_while_removed_legacy_aliases_fail_direct_import() -> (
    None
):
    exported_names: dict[str, object] = {}

    exec("from agent_runtime.runtime import *", {}, exported_names)

    assert "EphemeralRunRequest" in exported_names
    assert "RuntimeClient" in exported_names
    assert "ResumedSessionRunRequest" in exported_names
    assert "EphemeralRuntime" not in exported_names
    assert "NewSessionRuntime" not in exported_names
    assert "ResumedSessionRuntime" not in exported_names
    assert "EphemeralRuntimeExecutionAdapter" not in exported_names
    assert "NewSessionRuntimeExecutionAdapter" not in exported_names
    assert "ResumedSessionRuntimeExecutionAdapter" not in exported_names
    assert "ResumableRuntime" not in exported_names
    assert "ResumableRunRequest" not in exported_names
    assert "OneShotRuntime" not in exported_names
    assert "OneShotRunRequest" not in exported_names
    with pytest.raises(ImportError):
        exec("from agent_runtime.runtime import EphemeralRuntime", {}, {})
    with pytest.raises(ImportError):
        exec("from agent_runtime.runtime import NewSessionRuntime", {}, {})
    with pytest.raises(ImportError):
        exec("from agent_runtime.runtime import ResumedSessionRuntime", {}, {})
    with pytest.raises(ImportError):
        exec("from agent_runtime.runtime import ResumableRunRequest", {}, {})
    with pytest.raises(ImportError):
        exec("from agent_runtime.runtime import OneShotRuntime", {}, {})
    with pytest.raises(ImportError):
        exec("from agent_runtime.runtime import OneShotRunRequest", {}, {})


def test_runtime_surfaces_do_not_expose_retired_agent_log_names() -> None:
    exported_root_names: dict[str, object] = {}
    exported_runtime_names: dict[str, object] = {}

    exec("from agent_runtime import *", {}, exported_root_names)
    exec("from agent_runtime.runtime import *", {}, exported_runtime_names)

    for removed_name in (
        "AgentInvocationLog",
        "LogicalAgentInvocationLog",
        "WorkInvocationLog",
    ):
        assert removed_name not in runtime.__all__
        assert removed_name not in prompt_runtime.__all__
        assert removed_name not in exported_root_names
        assert removed_name not in exported_runtime_names
        assert not hasattr(runtime, removed_name)
        assert not hasattr(prompt_runtime, removed_name)
        with pytest.raises(ImportError):
            exec(f"from agent_runtime import {removed_name}", {}, {})
        with pytest.raises(ImportError):
            exec(f"from agent_runtime.runtime import {removed_name}", {}, {})

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("agent_runtime.agent_log")


@pytest.mark.parametrize(
    "removed_name",
    [
        "OneShotRunRequest",
        "OneShotRunResult",
        "OneShotResultMetadata",
        "OneShotRuntime",
        "OneShotRuntimeExecutionAdapter",
        "OneShotRuntimeMetadata",
    ],
)
def test_runtime_direct_import_rejects_removed_legacy_names(
    removed_name: str,
) -> None:
    with pytest.raises(ImportError):
        exec(f"from agent_runtime.runtime import {removed_name}", {}, {})

    with pytest.raises(AttributeError):
        getattr(prompt_runtime, removed_name)


def test_runtime_direct_import_rejects_removed_resumable_completed_result_names() -> (
    None
):
    with pytest.raises(AttributeError):
        getattr(prompt_runtime, "ResumableRuntime")
    with pytest.raises(AttributeError):
        getattr(prompt_runtime, "ResumableRuntimeExecutionAdapter")
    with pytest.raises(ImportError):
        exec("from agent_runtime.runtime import ResumableRuntime", {}, {})
    with pytest.raises(ImportError):
        exec(
            "from agent_runtime.runtime import ResumableRuntimeExecutionAdapter", {}, {}
        )
    with pytest.raises(ImportError):
        exec("from agent_runtime.runtime import ResumableRunResult", {}, {})
    with pytest.raises(ImportError):
        exec("from agent_runtime.runtime import ResumableRuntimeMetadata", {}, {})


def test_types_module_hides_removed_legacy_stage_names() -> None:
    types_module = importlib.import_module("agent_runtime.types")

    assert types_module.ProviderSelection.__module__.startswith("agent_runtime")
    assert not hasattr(types_module, "StageSelection")
    assert not hasattr(types_module, "StageOverride")
    with pytest.raises(ImportError, match="StageOverride"):
        exec("from agent_runtime.types import StageOverride", {})
    with pytest.raises(ImportError, match="StageSelection"):
        exec("from agent_runtime.types import StageSelection", {})


def test_deleted_stage_priority_chain_module_is_not_importable() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("agent_runtime.stage_priority_chain")


def test_runtime_surface_exposes_resumed_session_lifecycle_names() -> None:
    assert {
        "NewSessionRunRequest",
        "ResumedSessionRunRequest",
        "RuntimeClient",
    } <= set(prompt_runtime.__all__)
    assert hasattr(prompt_runtime, "ResumedSessionRunRequest")
    assert hasattr(prompt_runtime, "RuntimeClient")
    assert prompt_runtime.ResumedSessionRunRequest.__name__ == (
        "ResumedSessionRunRequest"
    )


def test_runtime_surface_exports_agent_event_public_vocabulary() -> None:
    assert hasattr(runtime, "AgentEvent")
    assert runtime.AgentEvent is prompt_runtime.AgentEvent
    assert {field.name for field in fields(runtime.AgentEvent)} == {
        "type",
        "display_message",
        "raw_provider_output",
    }
    with pytest.raises(FrozenInstanceError):
        setattr(
            runtime.AgentEvent(
                type="agent_message", display_message="hi", raw_provider_output=""
            ),
            "display_message",
            "changed",
        )


def test_runtime_lifecycle_request_values_expose_invocation_dir_without_public_worktree_alias(
    stage_selection_factory,
) -> None:
    ephemeral_request = prompt_runtime.EphemeralRunRequest(
        prompt="already rendered prompt",
        invocation_dir=Path("/tmp/worktree"),
        provider_selection=stage_selection_factory(service="codex"),
        tool_access=contracts_runtime.ToolAccess.no_tools(),
    )
    new_session_request = prompt_runtime.NewSessionRunRequest(
        prompt="already rendered prompt",
        invocation_dir=Path("/tmp/worktree"),
        provider_selection=stage_selection_factory(service="codex"),
        tool_access=contracts_runtime.ToolAccess.no_tools(),
    )
    resumed_session_request = prompt_runtime.ResumedSessionRunRequest(
        prompt="already rendered prompt",
        invocation_dir=Path("/tmp/worktree"),
        continuation=prompt_runtime.Continuation(
            selected_service="codex",
            selected_model="gpt-5.4",
            selected_effort="medium",
            tool_access=contracts_runtime.ToolAccess.no_tools(),
            provider_resume_state={},
        ),
    )

    assert ephemeral_request.invocation_dir == Path("/tmp/worktree")
    assert new_session_request.invocation_dir == Path("/tmp/worktree")
    assert resumed_session_request.invocation_dir == Path("/tmp/worktree")
    for request in (
        ephemeral_request,
        new_session_request,
        resumed_session_request,
    ):
        with pytest.raises(AttributeError):
            getattr(request, "worktree")


def test_runtime_lifecycle_values_keep_runtime_module_names_after_extraction() -> None:
    for exported_name in (
        "Cancelled",
        "Completed",
        "Continuation",
        "EphemeralRunRequest",
        "NewSessionRunRequest",
        "ProviderUnavailable",
        "ProviderAuth",
        "ResolvedProvider",
        "ResumedSessionRunRequest",
        "RunResult",
        "RuntimeOutcome",
        "TimedOut",
        "UsageLimited",
    ):
        assert getattr(prompt_runtime, exported_name).__module__ == (
            "agent_runtime.runtime"
        )


def test_runtime_package_root_exports_keep_runtime_lifecycle_identity() -> None:
    assert runtime.Continuation is prompt_runtime.Continuation
    assert runtime.ProviderAuth is prompt_runtime.ProviderAuth
    assert runtime.RuntimeOutcome is prompt_runtime.RuntimeOutcome


def test_runtime_client_lifecycle_entrypoints_do_not_read_live_probe_env(
    tmp_path: Path,
) -> None:
    env_path = Path(__file__).resolve().parents[1] / "scripts" / "live-probe" / ".env"
    original_env = None
    if env_path.exists():
        original_env = env_path.read_text(encoding="utf-8")
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text("invalid", encoding="utf-8")

    _fake_result = prompt_runtime.RunResult(
        output="ok",
        usage=None,
        continuation=None,
        selected=runtime.ResolvedProvider(
            service="claude", model="haiku", effort="low"
        ),
    )
    invoked = {"ephemeral": 0, "new_session": 0, "resumed_session": 0}

    try:

        def _fake_ephemeral(*_args: object, **_kwargs: object) -> object:
            invoked["ephemeral"] += 1
            return _fake_result

        def _fake_new_session(*_args: object, **_kwargs: object) -> object:
            invoked["new_session"] += 1
            return _fake_result

        def _fake_resumed_session(*_args: object, **_kwargs: object) -> object:
            invoked["resumed_session"] += 1
            return _fake_result

        provider_selection = prompt_runtime.ProviderSelection(
            service="claude",
            model="haiku",
            effort="low",
            auth=prompt_runtime.ProviderAuth(claude_code_oauth_token="token"),
        )
        ephemeral_request = prompt_runtime.EphemeralRunRequest(
            prompt="hello",
            invocation_dir=tmp_path / "invocation",
            provider_selection=provider_selection,
            tool_policy=prompt_runtime.ToolPolicy.UNRESTRICTED,
        )
        new_session_request = prompt_runtime.NewSessionRunRequest(
            prompt="hello",
            invocation_dir=tmp_path / "new-session",
            runtime_state_dir=tmp_path / "new-session" / "runtime-state",
            provider_selection=provider_selection,
            tool_policy=prompt_runtime.ToolPolicy.UNRESTRICTED,
        )
        continuation_payload = (
            continuation_payload_module.create_portable_continuation_payload(
                service_name="claude",
                model="haiku",
                effort="low",
                tool_access=contracts_runtime.ToolAccess.no_tools(),
                provider_resume_state={},
            ).serialized
        )
        resumed_request = prompt_runtime.ResumedSessionRunRequest(
            prompt="hello",
            invocation_dir=tmp_path / "resumed-session",
            runtime_state_dir=tmp_path / "resumed-session" / "runtime-state",
            continuation=prompt_runtime.Continuation(serialized=continuation_payload),
        )

        client = prompt_runtime.RuntimeClient()

        with (
            unittest.mock.patch.object(
                prompt_runtime, "_run_builtin_ephemeral", _fake_ephemeral
            ),
            unittest.mock.patch.object(
                prompt_runtime, "_run_builtin_new_session", _fake_new_session
            ),
            unittest.mock.patch.object(
                prompt_runtime, "_run_builtin_resumed_session", _fake_resumed_session
            ),
        ):
            run_ephemeral = asyncio.run(client.run_ephemeral(ephemeral_request))
            run_new_session = asyncio.run(client.run_new_session(new_session_request))
            run_resumed_session = asyncio.run(
                client.run_resumed_session(resumed_request)
            )

        assert isinstance(run_ephemeral.kind, prompt_runtime.Completed)
        assert isinstance(run_new_session.kind, prompt_runtime.Completed)
        assert isinstance(run_resumed_session.kind, prompt_runtime.Completed)
        assert invoked == {
            "ephemeral": 1,
            "new_session": 1,
            "resumed_session": 1,
        }
    finally:
        if original_env is None:
            env_path.unlink(missing_ok=True)
        else:
            env_path.write_text(original_env, encoding="utf-8")


def test_contracts_expose_execution_provider_as_canonical_public_protocol_name() -> (
    None
):
    contracts = importlib.import_module("agent_runtime.contracts")

    assert "ExecutionProvider" in contracts.__all__
    assert "ResumableExecutionProvider" in contracts.__all__
    assert not hasattr(contracts, "ExecutionService")
    assert not hasattr(contracts, "ResidentExecutionProvider")
    with pytest.raises(AttributeError):
        getattr(runtime, "ExecutionProvider")
    with pytest.raises(ImportError):
        exec("from agent_runtime import ExecutionProvider", {}, {})


@pytest.mark.parametrize(
    ("entrypoint_name", "request_factory", "delegate_name"),
    [
        (
            "new_session",
            lambda tmp_path: prompt_runtime.NewSessionRunRequest(
                prompt="hello",
                invocation_dir=tmp_path / "new-session",
                runtime_state_dir=tmp_path / "runtime-state",
                provider_selection=runtime.ProviderSelection(
                    service="claude",
                    model="haiku",
                    effort="low",
                ),
                tool_policy=prompt_runtime.ToolPolicy.NONE,
                timeout_seconds=7,
            ),
            "_run_builtin_new_session",
        ),
        (
            "resumed_session",
            lambda tmp_path: prompt_runtime.ResumedSessionRunRequest(
                prompt="hello",
                invocation_dir=tmp_path / "resumed-session",
                runtime_state_dir=tmp_path / "runtime-state",
                continuation=prompt_runtime.Continuation(
                    serialized=continuation_payload_module.create_portable_continuation_payload(
                        service_name="claude",
                        model="haiku",
                        effort="low",
                        tool_access=contracts_runtime.ToolAccess.no_tools(),
                        provider_resume_state={},
                    ).serialized
                ),
                timeout_seconds=7,
            ),
            "_run_builtin_resumed_session",
        ),
    ],
)
def test_runtime_client_session_entrypoints_delegate_directly_to_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    entrypoint_name: str,
    request_factory: Any,
    delegate_name: str,
) -> None:
    delegated_calls: list[tuple[Any, Any]] = []

    def _delegate(request: Any, *, on_live_output: Any) -> prompt_runtime.RunResult:
        delegated_calls.append((request, on_live_output))
        return prompt_runtime.RunResult(
            output="delegated output",
            usage=None,
            continuation=None,
            selected=prompt_runtime.ResolvedProvider(
                service="claude",
                model="haiku",
                effort="low",
            ),
        )

    monkeypatch.setattr(prompt_runtime, delegate_name, _delegate)

    client = prompt_runtime.RuntimeClient()
    request = request_factory(tmp_path)
    outcome = asyncio.run(getattr(client, f"run_{entrypoint_name}")(request))

    assert isinstance(outcome.kind, prompt_runtime.Completed)
    assert outcome.result.output == "delegated output"
    assert delegated_calls == [(request, request.on_live_output)]


@pytest.mark.parametrize(
    ("entrypoint_name", "request_factory", "delegate_name"),
    [
        (
            "new_session",
            lambda tmp_path: prompt_runtime.NewSessionRunRequest(
                prompt="hello",
                invocation_dir=tmp_path / "new-session",
                runtime_state_dir=tmp_path / "runtime-state",
                provider_selection=runtime.ProviderSelection(
                    service="claude",
                    model="haiku",
                    effort="low",
                ),
                tool_policy=prompt_runtime.ToolPolicy.NONE,
                timeout_seconds=7,
            ),
            "_run_builtin_new_session",
        ),
        (
            "resumed_session",
            lambda tmp_path: prompt_runtime.ResumedSessionRunRequest(
                prompt="hello",
                invocation_dir=tmp_path / "resumed-session",
                runtime_state_dir=tmp_path / "runtime-state",
                continuation=prompt_runtime.Continuation(
                    serialized=continuation_payload_module.create_portable_continuation_payload(
                        service_name="claude",
                        model="haiku",
                        effort="low",
                        tool_access=contracts_runtime.ToolAccess.no_tools(),
                        provider_resume_state={},
                    ).serialized
                ),
                timeout_seconds=7,
            ),
            "_run_builtin_resumed_session",
        ),
    ],
)
def test_runtime_client_session_entrypoints_propagate_delegated_execution_errors_directly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    entrypoint_name: str,
    request_factory: Any,
    delegate_name: str,
) -> None:
    def _raise_delegate(
        request: Any, *, on_live_output: Any
    ) -> prompt_runtime.RunResult:
        raise prompt_runtime.RuntimeConfigurationError("delegated failure")

    monkeypatch.setattr(prompt_runtime, delegate_name, _raise_delegate)

    client = prompt_runtime.RuntimeClient()
    request = request_factory(tmp_path)

    with pytest.raises(
        prompt_runtime.RuntimeConfigurationError, match="delegated failure"
    ):
        asyncio.run(getattr(client, f"run_{entrypoint_name}")(request))


def test_provider_session_seams_consolidate_public_session_store_vocabulary() -> None:
    assert "SessionStore" not in session_runtime.__all__
    assert not hasattr(session_runtime, "ServiceResumeIdentityStore")
    assert not hasattr(
        importlib.import_module("agent_runtime.contracts"),
        "ProviderSessionRecordingStore",
    )


def test_provider_session_adapter_public_seam_stays_narrow() -> None:
    public_module = importlib.import_module("agent_runtime.provider_session_adapter")
    assert public_module.__all__ == []
    for removed_name in (
        "ProviderSessionAdapter",
        "ProviderSessionPlanningFacts",
        "ProviderSessionPlanningRequest",
        "provider_session_planning_facts",
    ):
        with pytest.raises(ImportError):
            exec(
                f"from agent_runtime.provider_session_adapter import {removed_name}",
                {},
                {},
            )

    internal_module = importlib.import_module("agent_runtime._provider_session_adapter")
    for removed_name in (
        "ProviderSessionAdapter",
        "ProviderSessionPlanningFacts",
        "ProviderSessionPlanningRequest",
        "provider_session_planning_facts",
    ):
        assert not hasattr(internal_module, removed_name)


def test_tool_policy_has_three_members_on_public_surface() -> None:
    assert len(list(runtime.ToolPolicy)) == 3
    assert {policy.name for policy in runtime.ToolPolicy} == {
        "NONE",
        "NO_FILE_MUTATION",
        "UNRESTRICTED",
    }


def test_tool_policy_does_not_include_inspect_only_value() -> None:
    values = {policy.value for policy in runtime.ToolPolicy}
    assert "inspect_only" not in values


def test_tool_policy_none_resolves_to_closed_no_tools_profile() -> None:
    profile = runtime.ToolPolicy.NONE.profile

    assert profile.allowed_tools == ("none",)
    assert profile.disallowed_tools == ("all",)
    assert profile.strict_mcp_config is True


def test_runtime_surface_exposes_tool_policy_profiles_for_no_file_mutation_and_unrestricted() -> (
    None
):
    partial = runtime.ToolPolicy.NO_FILE_MUTATION.profile
    full = runtime.ToolPolicy.UNRESTRICTED.profile

    assert isinstance(partial, contracts_runtime.ToolPolicyProfile)
    assert partial.allowed_tools is None
    assert partial.disallowed_tools == ("Edit", "Write", "NotebookEdit")
    assert partial.strict_mcp_config is True
    assert isinstance(full, contracts_runtime.ToolPolicyProfile)
    assert full.allowed_tools is None
    assert full.disallowed_tools == ()
    assert full.strict_mcp_config is True


def test_tool_policy_profiles_stay_provider_neutral() -> None:
    for policy in runtime.ToolPolicy:
        profile = policy.profile
        rendered_values = (profile.allowed_tools or ()) + profile.disallowed_tools

        assert profile.strict_mcp_config is True
        assert all(not value.startswith("-") for value in rendered_values)
        assert all(
            provider not in value.lower()
            for value in rendered_values
            for provider in ("claude", "codex", "opencode")
        )
