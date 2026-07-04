from __future__ import annotations

import pytest

from agent_runtime.types import ProviderSelection as InternalStageSelection


@pytest.mark.parametrize("label", ["", " ", "a/b", "../escape"])
def test_runtime_service_identities_reject_unsafe_labels(label: str) -> None:
    with pytest.raises(ValueError):
        InternalStageSelection(
            service=label,
            model="provider model / ../ still allowed",
            effort="high effort / ../ still allowed",
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
