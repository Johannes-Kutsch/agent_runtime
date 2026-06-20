from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from .identity import validate_runtime_identity_label

if TYPE_CHECKING:
    from ._runtime_lifecycle import ProviderAuth


@dataclasses.dataclass(frozen=True)
class ProviderSelection:
    service: str
    model: str
    effort: str
    auth: ProviderAuth | None = None
    fallback: ProviderSelection | None = dataclasses.field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        validate_provider_selection(self)

    def __repr__(self) -> str:
        return (
            "ProviderSelection("
            f"service={self.service!r}, "
            f"model={self.model!r}, "
            f"effort={self.effort!r}, "
            f"auth={self.auth!r})"
        )


@dataclasses.dataclass(frozen=True)
class StageSelection(ProviderSelection):
    fallback: StageSelection | None = None

    def __post_init__(self) -> None:
        validate_stage_selection(self)


SelectionLike = ProviderSelection | StageSelection


def validate_provider_selection(selection: SelectionLike) -> None:
    _validate_selection_chain(selection, kind="ProviderSelection")


def validate_stage_selection(stage: StageSelection) -> None:
    _validate_selection_chain(stage, kind="StageSelection")


def _validate_selection_chain(
    selection: SelectionLike,
    *,
    kind: str,
) -> None:
    node: ProviderSelection | StageSelection | None = selection
    index = 0
    while node is not None:
        _require_stage_value("service", node.service, index=index)
        _require_stage_value("model", node.model, index=index)
        _require_stage_value("effort", node.effort, index=index)
        node = node.fallback
        index += 1


def _require_stage_value(field_name: str, value: str, *, index: int) -> None:
    if value.strip():
        if field_name == "service":
            validate_runtime_identity_label(
                value,
                kind="StageSelection service",
            )
        return
    node_label = "stage" if index == 0 else f"stage fallback #{index}"
    raise ValueError(f"StageSelection {node_label} requires a non-empty {field_name}.")


__all__ = ["ProviderSelection"]
