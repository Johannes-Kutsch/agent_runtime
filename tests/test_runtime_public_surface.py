from __future__ import annotations

import importlib
import inspect
import asyncio
from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from typing import Any, cast
import unittest.mock

import pytest

import agent_runtime as runtime
import agent_runtime._session_backed_provider_state_resolution as provider_state_resolution_runtime
import agent_runtime.contracts as contracts_runtime
import agent_runtime._provider_invocation as provider_invocation_runtime
import agent_runtime.runtime as prompt_runtime
import agent_runtime.session as session_runtime
from agent_runtime.errors import AgentRuntimeError
from agent_runtime.provider_usage import ProviderUsage
from agent_runtime.session import RunKind


def test_package_exports_runtime_surface() -> None:
    assert runtime.__all__ == [
        "AgentCredentialFailureError",
        "AgentEvent",
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
    assert runtime.AgentRuntimeError is AgentRuntimeError
    assert runtime.RuntimeOutcome is prompt_runtime.RuntimeOutcome
    assert not hasattr(runtime, "AgentFailedError")
    assert "ToolAccess" not in runtime.__all__
    assert "ToolPolicyProfile" not in runtime.__all__
    assert not hasattr(runtime, "ToolAccess")
    assert not hasattr(runtime, "ToolPolicyProfile")
    assert hasattr(contracts_runtime, "ToolAccess")
    assert hasattr(contracts_runtime, "ToolPolicyProfile")
    assert not hasattr(runtime, "ExecutionProvider")
    assert not hasattr(runtime, "ServiceRegistry")
    assert not hasattr(prompt_runtime, "PromptRuntime")
    assert not hasattr(prompt_runtime, "PromptRunRequest")
    assert not hasattr(prompt_runtime, "PromptRuntimeExecutionAdapter")
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


@pytest.mark.parametrize(
    ("module_name", "removed_name"),
    [
        ("agent_runtime", "AgentFailedError"),
        ("agent_runtime.errors", "AgentFailedError"),
    ],
)
def test_retired_agent_failed_error_is_not_importable_from_runtime_surface(
    module_name: str,
    removed_name: str,
) -> None:
    with pytest.raises(ImportError):
        exec(f"from {module_name} import {removed_name}", {}, {})

    imported_module = importlib.import_module(module_name)
    with pytest.raises(AttributeError):
        getattr(imported_module, removed_name)


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


def test_session_backed_provider_state_resolution_module_stays_private_to_runtime_public_surface() -> (
    None
):
    with pytest.raises(ImportError):
        exec("from agent_runtime import ProviderIdentity", {}, {})
    with pytest.raises(ImportError):
        exec("from agent_runtime.runtime import ProviderIdentity", {}, {})
    with pytest.raises(ImportError):
        exec("from agent_runtime.contracts import ProviderIdentity", {}, {})

    assert not hasattr(runtime, "ProviderIdentity")
    assert not hasattr(prompt_runtime, "ProviderIdentity")
    assert not hasattr(contracts_runtime, "ProviderIdentity")
    assert "ProviderIdentity" not in runtime.__all__
    assert "ProviderIdentity" not in prompt_runtime.__all__
    assert "ProviderIdentity" not in contracts_runtime.__all__


def test_session_backed_provider_state_resolution_module_exposes_immutable_internal_fact_values() -> (
    None
):
    provider_identity = provider_state_resolution_runtime.ProviderIdentity(
        service="claude",
        model="sonnet",
        effort="high",
    )
    provider_state_directory = provider_state_resolution_runtime.ProviderStateDirectory(
        path=Path("/tmp/store")
    )
    provider_state_relpath = provider_state_resolution_runtime.ProviderStateRelpath(
        value="implementer/claude/"
    )
    provider_session_id = (
        provider_state_resolution_runtime.PreparedOrRecoveredProviderSessionId(
            value="session-123",
            recovered=True,
        )
    )
    exact_transcript_match = provider_state_resolution_runtime.ExactTranscriptMatch(
        value=True
    )
    continuation_input_facts = provider_state_resolution_runtime.ContinuationInputFacts(
        provider_identity=provider_identity,
        provider_state_directory=provider_state_directory,
        provider_state_relpath=provider_state_relpath,
        provider_session_id=provider_session_id,
        run_kind=RunKind.RESUME,
        exact_transcript_match=exact_transcript_match,
    )

    assert continuation_input_facts.provider_identity is provider_identity
    assert continuation_input_facts.provider_state_directory is provider_state_directory
    assert continuation_input_facts.provider_state_relpath is provider_state_relpath
    assert continuation_input_facts.provider_session_id is provider_session_id
    assert continuation_input_facts.run_kind is RunKind.RESUME
    assert continuation_input_facts.exact_transcript_match is exact_transcript_match

    with pytest.raises(FrozenInstanceError):
        cast(Any, provider_identity).service = "codex"
    with pytest.raises(FrozenInstanceError):
        cast(Any, provider_state_directory).path = Path("/tmp/other")
    with pytest.raises(FrozenInstanceError):
        cast(Any, provider_state_relpath).value = "implementer/codex/"
    with pytest.raises(FrozenInstanceError):
        cast(Any, provider_session_id).value = "session-456"
    with pytest.raises(FrozenInstanceError):
        cast(Any, exact_transcript_match).value = False
    with pytest.raises(FrozenInstanceError):
        cast(Any, continuation_input_facts).run_kind = RunKind.FRESH


def test_execution_contracts_module_is_absent_without_changing_runtime_public_surface() -> (
    None
):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("agent_runtime.execution_contracts")

    with pytest.raises(ModuleNotFoundError):
        exec(
            "from agent_runtime.execution_contracts import ExecutionProvider",
            {},
            {},
        )

    for module_name, exported_name in (
        ("agent_runtime", "RuntimeClient"),
        ("agent_runtime", "ProviderSelection"),
        ("agent_runtime", "ProviderAuth"),
        ("agent_runtime", "Continuation"),
        ("agent_runtime", "ToolPolicy"),
        ("agent_runtime", "RuntimeOutcome"),
        ("agent_runtime", "RunResult"),
        ("agent_runtime", "RunKind"),
        ("agent_runtime.runtime", "RuntimeClient"),
        ("agent_runtime.runtime", "ProviderSelection"),
        ("agent_runtime.runtime", "ProviderAuth"),
        ("agent_runtime.runtime", "Continuation"),
        ("agent_runtime.runtime", "ToolPolicy"),
        ("agent_runtime.runtime", "RuntimeOutcome"),
        ("agent_runtime.runtime", "RunResult"),
        ("agent_runtime.runtime", "EphemeralRunRequest"),
        ("agent_runtime.runtime", "NewSessionRunRequest"),
        ("agent_runtime.runtime", "ResumedSessionRunRequest"),
    ):
        imported_module = importlib.import_module(module_name)
        assert hasattr(imported_module, exported_name)


def test_provider_session_adapter_module_is_absent() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("agent_runtime.provider_session_adapter")


def test_retired_service_registry_module_is_absent() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("agent_runtime.service_registry")

    with pytest.raises(ModuleNotFoundError):
        exec("from agent_runtime.service_registry import ServiceRegistry", {}, {})


@pytest.mark.parametrize(
    "imported_name",
    [
        "ExecutionProvider",
        "ProviderStatePreparationAction",
        "ResumabilityProvider",
        "ResumableExecutionProvider",
        "ServiceSelectionProvider",
        "ParsedTurn",
        "ToolAccess",
        "ToolPolicy",
        "ToolPolicyProfile",
    ],
)
def test_provider_session_planning_compatibility_module_is_absent(
    imported_name: str,
) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("agent_runtime.session_planning")

    with pytest.raises(ModuleNotFoundError):
        exec(
            f"from agent_runtime.session_planning import {imported_name}",
            {},
            {},
        )


def test_session_module_exports_only_active_provider_state_helpers() -> None:
    assert session_runtime.__all__ == ["RunKind", "provider_state_relpath"]
    assert session_runtime.RunKind is RunKind
    assert session_runtime.provider_state_relpath("implementer", "codex") == (
        "implementer/codex/"
    )


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


def test_runtime_client_constructor_rejects_already_sandboxed_keyword_argument() -> (
    None
):
    signature = inspect.signature(runtime.RuntimeClient)
    assert list(signature.parameters.keys()) == []
    unexpected_kwargs: dict[str, object] = {"_provider_invocation_adapter": None}
    with pytest.raises(TypeError):
        cast(Any, runtime.RuntimeClient)(**unexpected_kwargs)
    with pytest.raises(TypeError):
        cast(Any, runtime.RuntimeClient)(already_sandboxed=True)


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


def test_runtime_star_import_uses_lifecycle_surface() -> None:
    exported_names: dict[str, object] = {}

    exec("from agent_runtime.runtime import *", {}, exported_names)

    assert "EphemeralRunRequest" in exported_names
    assert "RuntimeClient" in exported_names
    assert "ResumedSessionRunRequest" in exported_names


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
        resumed_request = prompt_runtime.ResumedSessionRunRequest(
            prompt="hello",
            invocation_dir=tmp_path / "resumed-session",
            runtime_state_dir=tmp_path / "resumed-session" / "runtime-state",
            continuation=prompt_runtime.Continuation(
                selected_service="claude",
                selected_model="haiku",
                selected_effort="low",
                tool_access=contracts_runtime.ToolAccess.no_tools(),
                provider_resume_state={},
            ),
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


@pytest.mark.parametrize(
    "removed_name",
    [
        "ExecutionProvider",
        "ProviderStatePreparationAction",
        "ResumabilityProvider",
        "ResumableExecutionProvider",
        "ServiceSelectionProvider",
    ],
)
def test_contracts_expose_only_active_runtime_contracts(
    removed_name: str,
) -> None:
    contracts = importlib.import_module("agent_runtime.contracts")

    assert {"ParsedTurn", "ToolAccess", "ToolPolicy", "ToolPolicyProfile"} <= set(
        contracts.__all__
    )
    assert removed_name not in contracts.__all__
    with pytest.raises(AttributeError):
        getattr(contracts, removed_name)
    with pytest.raises(ImportError):
        exec(f"from agent_runtime.contracts import {removed_name}", {}, {})
    with pytest.raises(AttributeError):
        getattr(runtime, removed_name)
    with pytest.raises(ImportError):
        exec(f"from agent_runtime import {removed_name}", {}, {})


@pytest.mark.parametrize(
    "removed_name",
    [
        "ExecutionProvider",
        "ProviderStatePreparationAction",
        "ResumabilityProvider",
        "ResumableExecutionProvider",
        "ServiceSelectionProvider",
    ],
)
def test_runtime_module_omits_retired_provider_protocol_names_from_public_surface(
    removed_name: str,
) -> None:
    assert removed_name not in prompt_runtime.__all__
    with pytest.raises(AttributeError):
        getattr(prompt_runtime, removed_name)
    with pytest.raises(ImportError):
        exec(f"from agent_runtime.runtime import {removed_name}", {}, {})


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
                    selected_service="claude",
                    selected_model="haiku",
                    selected_effort="low",
                    tool_access=contracts_runtime.ToolAccess.no_tools(),
                    provider_resume_state={},
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

    def _delegate(
        request: Any,
        *,
        on_live_output: Any,
        **_kwargs: Any,
    ) -> prompt_runtime.RunResult:
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
                    selected_service="claude",
                    selected_model="haiku",
                    selected_effort="low",
                    tool_access=contracts_runtime.ToolAccess.no_tools(),
                    provider_resume_state={},
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
        request: Any,
        *,
        on_live_output: Any,
        **_kwargs: Any,
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


@pytest.mark.parametrize(
    ("module_name", "removed_name"),
    [
        ("agent_runtime", "ProviderSessionAdapter"),
        ("agent_runtime", "ProviderSessionPlanningFacts"),
        ("agent_runtime", "ProviderSessionPlanningRequest"),
        ("agent_runtime", "provider_session_planning_facts"),
        ("agent_runtime.runtime", "ProviderSessionAdapter"),
        ("agent_runtime.runtime", "ProviderSessionPlanningFacts"),
        ("agent_runtime.runtime", "ProviderSessionPlanningRequest"),
        ("agent_runtime.runtime", "provider_session_planning_facts"),
        ("agent_runtime.contracts", "ProviderSessionAdapter"),
        ("agent_runtime.contracts", "ProviderSessionPlanningFacts"),
        ("agent_runtime.contracts", "ProviderSessionPlanningRequest"),
        ("agent_runtime.contracts", "provider_session_planning_facts"),
    ],
)
def test_provider_session_adapter_names_are_absent_from_runtime_public_surface(
    module_name: str,
    removed_name: str,
) -> None:
    imported_module = importlib.import_module(module_name)
    assert removed_name not in getattr(imported_module, "__all__", [])
    assert not hasattr(imported_module, removed_name)


def test_provider_session_adapter_public_seam_is_absent() -> None:
    for removed_name in (
        "ProviderSessionAdapter",
        "ProviderSessionPlanningFacts",
        "ProviderSessionPlanningRequest",
        "provider_session_planning_facts",
    ):
        with pytest.raises(ModuleNotFoundError):
            exec(
                f"from agent_runtime.provider_session_adapter import {removed_name}",
                {},
                {},
            )

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("agent_runtime._provider_session_adapter")


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


def test_tool_policy_inspect_only_attribute_is_removed_from_public_surface() -> None:
    with pytest.raises(AttributeError):
        getattr(runtime.ToolPolicy, "INSPECT_ONLY")


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
