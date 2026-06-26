from __future__ import annotations

import pytest

from agent_runtime.session import (
    provider_state_relpath,
)
from agent_runtime.types import ProviderSelection as InternalStageSelection


@pytest.mark.parametrize("label", ["", " ", "a/b", "../escape"])
def test_runtime_service_identities_reject_unsafe_labels(label: str) -> None:
    with pytest.raises(ValueError):
        InternalStageSelection(
            service=label,
            model="provider model / ../ still allowed",
            effort="high effort / ../ still allowed",
        )


@pytest.mark.parametrize("service_name", ["", " ", "a/b", "../escape"])
def test_provider_state_path_helpers_reject_unsafe_runtime_service_labels(
    service_name: str,
) -> None:
    with pytest.raises(ValueError):
        provider_state_relpath("implementer", service_name, namespace="main")


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


def test_provider_state_relpath_supports_session_root_layout() -> None:
    assert (
        provider_state_relpath(
            "implementer",
            "codex",
            session_root=".runtime-session",
        )
        == ".runtime-session/implementer/codex/"
    )
