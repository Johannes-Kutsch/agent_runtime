from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, cast

import pytest

import agent_runtime.session_planning as session_planning_runtime
from agent_runtime.contracts import (
    ExecutionProvider,
    ResumabilityProvider,
    ServiceSelectionProvider,
)
from agent_runtime._provider_session_adapter import ProviderSessionPlanningRequest
from agent_runtime.roles import InvocationRole
from agent_runtime._service_registry import ServiceRegistry
from agent_runtime.session import (
    RunKind,
    normalize_state_dir_relpath,
    provider_state_relpath,
    provider_state_session_id_path,
)
from agent_runtime.session_planning import (
    ResumableSessionPlanRequest,
    plan_resumable_session,
)
from agent_runtime.types import ProviderSelection as InternalStageSelection
from tests.runtime_boundary_fakes import (
    ResidentPlanningProviderSessionAdapterFake as _ResidentPlanningProviderSessionAdapter,
    SelectionServiceFake as _Service,
)


@pytest.mark.parametrize("label", ["", " ", "a/b", "../escape"])
def test_runtime_service_identities_reject_unsafe_labels(label: str) -> None:
    with pytest.raises(ValueError):
        InternalStageSelection(
            service=label,
            model="provider model / ../ still allowed",
            effort="high effort / ../ still allowed",
        )

    with pytest.raises(ValueError):
        ServiceRegistry(
            {
                label: cast(
                    ServiceSelectionProvider,
                    _Service(
                        "codex",
                        available=True,
                        wake_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
                    ),
                )
            }
        )


@pytest.mark.parametrize("label", [" ", "a/b", "../escape"])
def test_provider_session_namespace_seams_preserve_empty_default_and_reject_unsafe_non_empty_values(
    label: str,
    execution_service_factory: Callable[..., ExecutionProvider],
    resident_provider_session_adapter: _ResidentPlanningProviderSessionAdapter,
) -> None:
    assert (
        ProviderSessionPlanningRequest(
            worktree=Path("."),
            role=InvocationRole("implementer"),
            namespace="",
        ).namespace
        == ""
    )
    assert (
        ResumableSessionPlanRequest(
            worktree=Path("."),
            role=InvocationRole("implementer"),
            namespace="",
            service=execution_service_factory(),
            provider_session_adapter=resident_provider_session_adapter,
        ).namespace
        == ""
    )

    with pytest.raises(ValueError):
        ProviderSessionPlanningRequest(
            worktree=Path("."),
            role=InvocationRole("implementer"),
            namespace=label,
        )

    with pytest.raises(ValueError):
        ResumableSessionPlanRequest(
            worktree=Path("."),
            role=InvocationRole("implementer"),
            namespace=label,
            service=execution_service_factory(),
            provider_session_adapter=resident_provider_session_adapter,
        )


def test_provider_session_planning_returns_immutable_decision_value(
    execution_service_factory: Callable[..., ExecutionProvider],
    resident_provider_session_adapter: _ResidentPlanningProviderSessionAdapter,
) -> None:
    provider_session_decision = session_planning_runtime.plan_provider_session(
        session_planning_runtime.ProviderSessionPlanRequest(
            worktree=Path("."),
            role=InvocationRole("implementer"),
            namespace="main",
            resumability_service=cast(
                ResumabilityProvider,
                execution_service_factory(),
            ),
            provider_session_adapter=resident_provider_session_adapter,
        )
    )

    assert (
        provider_session_decision
        == session_planning_runtime.ProviderSessionDecision(
            run_kind=RunKind.RESUME,
            provider_session_id="recovered-session",
            state_dir_relpath="state/",
            state_dir_path=Path("state"),
            service_state_dir=Path("state"),
            exact_transcript_match=False,
        )
    )
    with pytest.raises(FrozenInstanceError):
        setattr(provider_session_decision, "provider_session_id", "other")


def test_resumable_session_plan_exposes_public_value_fields_only(
    execution_service_factory: Callable[..., ExecutionProvider],
    resident_provider_session_adapter: _ResidentPlanningProviderSessionAdapter,
) -> None:
    service = execution_service_factory()

    session_plan = plan_resumable_session(
        ResumableSessionPlanRequest(
            worktree=Path("."),
            role=InvocationRole("implementer"),
            namespace="main",
            service=service,
            provider_session_adapter=resident_provider_session_adapter,
        )
    )

    assert session_plan.role == InvocationRole("implementer")
    assert session_plan.worktree == Path(".")
    assert session_plan.namespace == "main"
    assert session_plan.service is service
    assert session_plan.run_kind is RunKind.RESUME
    assert session_plan.provider_state_dir == Path("state")
    assert session_plan.provider_session_id == "recovered-session"
    assert session_plan.exact_transcript_match is False
    assert session_plan.usage_limit_scope is None
    with pytest.raises(FrozenInstanceError):
        setattr(session_plan, "provider_state_dir", Path("other-state"))


def test_resumable_session_plan_hides_container_state_selection_metadata(
    execution_service_factory: Callable[..., ExecutionProvider],
    resident_provider_session_adapter: _ResidentPlanningProviderSessionAdapter,
) -> None:
    service = execution_service_factory()

    session_plan = plan_resumable_session(
        ResumableSessionPlanRequest(
            worktree=Path("."),
            role=InvocationRole("implementer"),
            namespace="main",
            service=service,
            provider_session_adapter=resident_provider_session_adapter,
        )
    )

    field_names = {field.name for field in fields(session_plan)}

    assert "service_state_dir" not in field_names
    assert "use_service_state_dir_for_container" not in field_names


@pytest.mark.parametrize("service_name", ["", " ", "a/b", "../escape"])
def test_provider_state_path_helpers_reject_unsafe_runtime_service_labels(
    service_name: str,
) -> None:
    role = InvocationRole("implementer")

    with pytest.raises(ValueError):
        provider_state_relpath(role, service_name, namespace="main")

    with pytest.raises(ValueError):
        normalize_state_dir_relpath(
            role,
            "main",
            service_name,
            ".runtime-session/implementer/main/codex/",
        )


def test_public_provider_selection_requires_non_empty_candidate_configuration() -> None:
    with pytest.raises(ValueError, match="service"):
        InternalStageSelection(
            service="",
            model="gpt-5.4",
            effort="medium",
        )

    with pytest.raises(ValueError, match="model"):
        InternalStageSelection(
            service="codex",
            model="",
            effort="medium",
        )

    with pytest.raises(ValueError, match="effort"):
        InternalStageSelection(
            service="codex",
            model="gpt-5.4",
            effort="",
        )


def test_public_provider_selection_rejects_path_like_service_name() -> None:
    with pytest.raises(ValueError, match="ProviderSelection service"):
        InternalStageSelection(
            service="bad/name",
            model="gpt-5.4",
            effort="medium",
        )


def test_service_registry_rejects_invalid_public_service_name_configuration() -> None:
    with pytest.raises(ValueError, match="ServiceRegistry service name"):
        ServiceRegistry(
            {
                "bad/name": cast(
                    ServiceSelectionProvider,
                    _Service(
                        "bad/name",
                        available=True,
                        wake_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
                    ),
                ),
            }
        )


def test_application_can_render_service_availability_summary_from_registry(
    service_registry_factory: Callable[..., ServiceRegistry],
) -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    registry = service_registry_factory(
        "codex",
        "claude",
        unavailable={"codex"},
        wake_times={
            "codex": datetime(2026, 1, 2, tzinfo=timezone.utc),
            "claude": datetime(2026, 1, 3, tzinfo=timezone.utc),
        },
    )

    summary_lines = [
        f"{name}: {'available' if service.is_available(now=now) else 'unavailable'}"
        for name, service in registry.services.items()
    ]

    assert summary_lines == [
        "codex: unavailable",
        "claude: available",
    ]


def test_service_registry_resolves_single_provider_selection_unchanged(
    service_registry_factory: Callable[..., ServiceRegistry],
) -> None:
    registry = service_registry_factory("codex", unavailable={"codex"})
    provider_selection = InternalStageSelection(
        service="codex",
        model="gpt-5.4",
        effort="medium",
    )

    assert (
        registry.resolve(
            provider_selection,
            datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        is provider_selection
    )


def test_service_registry_scopes_availability_to_selected_provider(
    service_registry_factory: Callable[..., ServiceRegistry],
) -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    registry = service_registry_factory(
        "codex",
        "claude",
        unavailable={"codex"},
        wake_times={"codex": datetime(2026, 1, 2, tzinfo=timezone.utc)},
    )

    assert (
        registry.has_available_for(
            InternalStageSelection(
                service="codex",
                model="gpt-5.4",
                effort="medium",
            ),
            now,
        )
        is False
    )
    assert registry.next_wake_time_for(
        InternalStageSelection(
            service="codex",
            model="gpt-5.4",
            effort="medium",
        ),
        now,
    ) == datetime(2026, 1, 2, tzinfo=timezone.utc)
    assert (
        registry.has_available_for(
            InternalStageSelection(
                service="claude",
                model="sonnet",
                effort="high",
            ),
            now,
        )
        is True
    )
    assert (
        registry.next_wake_time_for(
            InternalStageSelection(
                service="claude",
                model="sonnet",
                effort="high",
            ),
            now,
        )
        is None
    )


def test_provider_state_helpers_normalize_legacy_layout_and_build_session_id_path() -> (
    None
):
    legacy = ".runtime-session/implementer/main/codex/"

    assert (
        provider_state_relpath(
            InvocationRole("implementer"),
            "codex",
            session_root=".runtime-session",
        )
        == ".runtime-session/implementer/codex/"
    )
    assert (
        normalize_state_dir_relpath(
            InvocationRole("implementer"),
            "main",
            "codex",
            legacy,
        )
        == ".runtime-session/implementer/main/codex/"
    )
    assert provider_state_session_id_path(Path("state"), "codex") == Path(
        "state/thread_id"
    )
