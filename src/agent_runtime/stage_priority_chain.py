from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from .types import StageOverride


@dataclass(frozen=True)
class ChainEntry:
    service: str
    model: str
    effort: str
    fallback: StageOverride | None


@dataclass(frozen=True)
class ConfiguredCandidateSelection:
    has_configured_candidate: bool
    selected_chain: StageOverride | None


@dataclass(frozen=True)
class ConfiguredCandidateChain:
    candidates: tuple[StageOverride, ...]

    @property
    def has_configured_candidate(self) -> bool:
        return bool(self.candidates)


def iter_stage_chain(override: StageOverride) -> Iterator[StageOverride]:
    node: StageOverride | None = override
    while node is not None:
        yield node
        node = node.fallback


def chain_entries(override: StageOverride) -> tuple[ChainEntry, ...]:
    return tuple(
        ChainEntry(
            service=node.service,
            model=node.model,
            effort=node.effort,
            fallback=node.fallback,
        )
        for node in iter_stage_chain(override)
    )


def validation_labels(stage_name: str, override: StageOverride) -> tuple[str, ...]:
    return tuple(
        stage_name if index == 0 else f"{stage_name} fallback"
        for index, _entry in enumerate(chain_entries(override))
    )


def render_chain_label(override: StageOverride) -> str:
    return " -> ".join(
        entry.service if entry.service else "<missing>"
        for entry in chain_entries(override)
    )


def referenced_service_names(override: StageOverride) -> tuple[str, ...]:
    names: list[str] = []
    seen: set[str] = set()
    for node in iter_stage_chain(override):
        service = node.service.strip()
        if not service or service in seen:
            continue
        names.append(service)
        seen.add(service)
    return tuple(names)


def configured_candidate_chain(
    override: StageOverride, *, configured_service_names: tuple[str, ...]
) -> ConfiguredCandidateChain:
    configured = set(configured_service_names)
    return ConfiguredCandidateChain(
        candidates=tuple(
            node for node in iter_stage_chain(override) if node.service in configured
        )
    )


def _build_chain(nodes: tuple[StageOverride, ...]) -> StageOverride | None:
    chain: StageOverride | None = None
    for node in reversed(nodes):
        chain = StageOverride(
            service=node.service,
            model=node.model,
            effort=node.effort,
            fallback=chain,
        )
    return chain


def _remaining_chain_is_fully_configured(
    override: StageOverride, configured: set[str]
) -> bool:
    return all(node.service in configured for node in iter_stage_chain(override))


def select_configured_candidate_chain(
    override: StageOverride,
    *,
    configured_service_names: tuple[str, ...],
    available_service_names: tuple[str, ...],
) -> ConfiguredCandidateSelection:
    configured = set(configured_service_names)
    available = set(available_service_names)
    configured_candidates = configured_candidate_chain(
        override, configured_service_names=configured_service_names
    )
    if not configured_candidates.has_configured_candidate:
        return ConfiguredCandidateSelection(
            has_configured_candidate=False,
            selected_chain=None,
        )
    for index, node in enumerate(configured_candidates.candidates):
        if node.service in available:
            if _remaining_chain_is_fully_configured(node, configured):
                return ConfiguredCandidateSelection(
                    has_configured_candidate=True,
                    selected_chain=node,
                )
            return ConfiguredCandidateSelection(
                has_configured_candidate=True,
                selected_chain=_build_chain(configured_candidates.candidates[index:]),
            )
    first_configured = configured_candidates.candidates[0]
    if _remaining_chain_is_fully_configured(first_configured, configured):
        return ConfiguredCandidateSelection(
            has_configured_candidate=True,
            selected_chain=first_configured,
        )
    return ConfiguredCandidateSelection(
        has_configured_candidate=True,
        selected_chain=_build_chain(configured_candidates.candidates),
    )
