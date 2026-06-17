from __future__ import annotations

import dataclasses

from .identity import validate_runtime_identity_label


@dataclasses.dataclass(frozen=True)
class StageSelection:
    model: str = ""
    effort: str = ""
    service: str = ""
    fallback: StageSelection | None = None

    def __post_init__(self) -> None:
        validate_stage_selection(self)


def validate_stage_selection(stage: StageSelection) -> None:
    node: StageSelection | None = stage
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


StageOverride = StageSelection


__all__ = ["StageSelection"]
