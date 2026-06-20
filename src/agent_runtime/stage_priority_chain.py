from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from .types import (
    ProviderSelection,
    StageSelection,
    validate_provider_selection,
)


@dataclass(frozen=True)
class ChainEntry:
    service: str
    model: str
    effort: str
    fallback: ProviderSelection | None


@dataclass(frozen=True)
class ConfiguredCandidateSelection:
    has_configured_candidate: bool
    selected_provider_selection: ProviderSelection | None

    @property
    def selected_chain(self) -> StageSelection | None:
        return self.selected_provider_selection


@dataclass(frozen=True)
class ConfiguredCandidateChain:
    candidates: tuple[ProviderSelection, ...]

    @property
    def has_configured_candidate(self) -> bool:
        return bool(self.candidates)


def iter_stage_chain(override: StageSelection) -> Iterator[StageSelection]:
    return iter_provider_selection_chain(override)


def iter_provider_selection_chain(
    provider_selection: ProviderSelection,
) -> Iterator[ProviderSelection]:
    validate_provider_selection(provider_selection)
    node: ProviderSelection | None = provider_selection
    while node is not None:
        yield node
        node = node.fallback


def chain_entries(override: StageSelection) -> tuple[ChainEntry, ...]:
    return provider_selection_entries(override)


def provider_selection_entries(
    provider_selection: ProviderSelection,
) -> tuple[ChainEntry, ...]:
    return tuple(
        ChainEntry(
            service=node.service,
            model=node.model,
            effort=node.effort,
            fallback=node.fallback,
        )
        for node in iter_provider_selection_chain(provider_selection)
    )


def validation_labels(stage_name: str, override: StageSelection) -> tuple[str, ...]:
    return provider_selection_validation_labels(stage_name, override)


def provider_selection_validation_labels(
    selection_name: str,
    provider_selection: ProviderSelection,
) -> tuple[str, ...]:
    return tuple(
        selection_name if index == 0 else f"{selection_name} fallback"
        for index, _entry in enumerate(provider_selection_entries(provider_selection))
    )


def render_chain_label(override: StageSelection) -> str:
    return render_provider_selection_label(override)


def render_provider_selection_label(
    provider_selection: ProviderSelection,
) -> str:
    return " -> ".join(
        entry.service if entry.service else "<missing>"
        for entry in provider_selection_entries(provider_selection)
    )


def referenced_service_names(override: StageSelection) -> tuple[str, ...]:
    return referenced_provider_service_names(override)


def referenced_provider_service_names(
    provider_selection: ProviderSelection,
) -> tuple[str, ...]:
    names: list[str] = []
    seen: set[str] = set()
    for node in iter_provider_selection_chain(provider_selection):
        service = node.service.strip()
        if not service or service in seen:
            continue
        names.append(service)
        seen.add(service)
    return tuple(names)


def configured_candidate_chain(
    override: StageSelection, *, configured_service_names: tuple[str, ...]
) -> ConfiguredCandidateChain:
    return configured_provider_selection_chain(
        override,
        configured_service_names=configured_service_names,
    )


def configured_provider_selection_chain(
    provider_selection: ProviderSelection,
    *,
    configured_service_names: tuple[str, ...],
) -> ConfiguredCandidateChain:
    configured = set(configured_service_names)
    return ConfiguredCandidateChain(
        candidates=tuple(
            node
            for node in iter_provider_selection_chain(provider_selection)
            if node.service in configured
        )
    )


def _build_chain(nodes: tuple[ProviderSelection, ...]) -> ProviderSelection | None:
    chain: ProviderSelection | None = None
    for node in reversed(nodes):
        chain = StageSelection(
            service=node.service,
            model=node.model,
            effort=node.effort,
            fallback=chain,
        )
    return chain


def _remaining_chain_is_fully_configured(
    override: StageSelection, configured: set[str]
) -> bool:
    return all(
        node.service in configured for node in iter_provider_selection_chain(override)
    )


def select_configured_candidate_chain(
    override: StageSelection,
    *,
    configured_service_names: tuple[str, ...],
    available_service_names: tuple[str, ...],
) -> ConfiguredCandidateSelection:
    return select_configured_provider_selection_chain(
        override,
        configured_service_names=configured_service_names,
        available_service_names=available_service_names,
    )


def select_configured_provider_selection_chain(
    provider_selection: ProviderSelection,
    *,
    configured_service_names: tuple[str, ...],
    available_service_names: tuple[str, ...],
) -> ConfiguredCandidateSelection:
    configured = set(configured_service_names)
    available = set(available_service_names)
    configured_candidates = configured_provider_selection_chain(
        provider_selection,
        configured_service_names=configured_service_names,
    )
    if not configured_candidates.has_configured_candidate:
        return ConfiguredCandidateSelection(
            has_configured_candidate=False,
            selected_provider_selection=None,
        )
    for index, node in enumerate(configured_candidates.candidates):
        if node.service in available:
            if _remaining_chain_is_fully_configured(node, configured):
                return ConfiguredCandidateSelection(
                    has_configured_candidate=True,
                    selected_provider_selection=node,
                )
            return ConfiguredCandidateSelection(
                has_configured_candidate=True,
                selected_provider_selection=_build_chain(
                    configured_candidates.candidates[index:]
                ),
            )
    first_configured = configured_candidates.candidates[0]
    if _remaining_chain_is_fully_configured(first_configured, configured):
        return ConfiguredCandidateSelection(
            has_configured_candidate=True,
            selected_provider_selection=first_configured,
        )
    return ConfiguredCandidateSelection(
        has_configured_candidate=True,
        selected_provider_selection=_build_chain(configured_candidates.candidates),
    )
