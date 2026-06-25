from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, cast

import pytest

from agent_runtime.contracts import (
    ServiceSelectionProvider,
)
from agent_runtime._service_registry import ServiceRegistry
from agent_runtime.session import (
    normalize_state_dir_relpath,
    provider_state_relpath,
    provider_state_session_id_path,
)
from agent_runtime.types import ProviderSelection as InternalStageSelection
from tests.runtime_boundary_fakes import (
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


@pytest.mark.parametrize("service_name", ["", " ", "a/b", "../escape"])
def test_provider_state_path_helpers_reject_unsafe_runtime_service_labels(
    service_name: str,
) -> None:
    with pytest.raises(ValueError):
        provider_state_relpath("implementer", service_name, namespace="main")

    with pytest.raises(ValueError):
        normalize_state_dir_relpath(
            "implementer",
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
            "implementer",
            "codex",
            session_root=".runtime-session",
        )
        == ".runtime-session/implementer/codex/"
    )
    assert (
        normalize_state_dir_relpath(
            "implementer",
            "main",
            "codex",
            legacy,
        )
        == ".runtime-session/implementer/main/codex/"
    )
    assert provider_state_session_id_path(Path("state"), "codex") == Path(
        "state/thread_id"
    )
