from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

from .contracts import ServiceSelectionProvider
from .identity import validate_runtime_identity_label
from .stage_priority_chain import (
    configured_provider_selection_chain,
    select_configured_provider_selection_chain,
)
from .types import ProviderSelection, StageSelection, validate_provider_selection


class ServiceRegistry:
    def __init__(self, services: Mapping[str, ServiceSelectionProvider]) -> None:
        for service_name in services:
            validate_runtime_identity_label(
                service_name,
                kind="ServiceRegistry service name",
            )
        self._services = dict(services)

    @property
    def services(self) -> dict[str, ServiceSelectionProvider]:
        return dict(self._services)

    def _configured_candidate_provider_selections(
        self, provider_selection: ProviderSelection
    ) -> tuple[ProviderSelection, ...]:
        validate_provider_selection(provider_selection)
        return configured_provider_selection_chain(
            provider_selection,
            configured_service_names=tuple(self._services),
        ).candidates

    def _availability_by_service(
        self, provider_selections: tuple[ProviderSelection, ...], now: datetime
    ) -> dict[str, bool]:
        availability: dict[str, bool] = {}
        for node in provider_selections:
            if node.service in availability:
                continue
            availability[node.service] = self._services[node.service].is_available(
                now=now
            )
        return availability

    def _exhausted_services_for(
        self, provider_selection: ProviderSelection, now: datetime
    ) -> tuple[ServiceSelectionProvider, ...]:
        configured_provider_selections = self._configured_candidate_provider_selections(
            provider_selection
        )
        availability = self._availability_by_service(
            configured_provider_selections, now
        )
        return tuple(
            self._services[node.service]
            for node in configured_provider_selections
            if not availability[node.service]
        )

    def has_configured_candidate(self, override: StageSelection) -> bool:
        return configured_provider_selection_chain(
            override,
            configured_service_names=tuple(self._services),
        ).has_configured_candidate

    def resolve(self, override: StageSelection, now: datetime) -> StageSelection:
        configured_provider_selections = self._configured_candidate_provider_selections(
            override
        )
        availability = self._availability_by_service(
            configured_provider_selections, now
        )
        selection = select_configured_provider_selection_chain(
            override,
            configured_service_names=tuple(
                node.service for node in configured_provider_selections
            ),
            available_service_names=tuple(
                node.service
                for node in configured_provider_selections
                if availability[node.service]
            ),
        )
        return selection.selected_provider_selection or override

    def has_available(self, now: datetime) -> bool:
        return any(svc.is_available(now=now) for svc in self._services.values())

    def has_available_for(self, override: StageSelection, now: datetime) -> bool:
        configured_provider_selections = self._configured_candidate_provider_selections(
            override
        )
        availability = self._availability_by_service(
            configured_provider_selections, now
        )
        return any(
            availability[node.service] for node in configured_provider_selections
        )

    def next_wake_time(self, now: datetime) -> datetime | None:
        exhausted = [
            svc for svc in self._services.values() if not svc.is_available(now=now)
        ]
        if not exhausted:
            return None
        return min(svc.next_wake_time() for svc in exhausted)

    def next_wake_time_for(
        self, override: StageSelection, now: datetime
    ) -> datetime | None:
        exhausted = self._exhausted_services_for(override, now)
        if not exhausted:
            return None
        return min(service.next_wake_time() for service in exhausted)

    def mark_exhausted(self, service_name: str, *, reset_time: datetime | None) -> None:
        service = self._services.get(service_name)
        if service is None:
            return
        service.mark_exhausted(reset_time)

    def __getitem__(self, key: str) -> ServiceSelectionProvider | None:
        return self._services.get(key)
