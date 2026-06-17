from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

from .contracts import ServiceSelectionProvider
from .stage_priority_chain import (
    configured_candidate_chain,
    select_configured_candidate_chain,
)
from .types import StageSelection, validate_stage_selection


class ServiceRegistry:
    def __init__(self, services: Mapping[str, ServiceSelectionProvider]) -> None:
        self._services = dict(services)

    @property
    def services(self) -> dict[str, ServiceSelectionProvider]:
        return dict(self._services)

    def _configured_candidate_overrides(
        self, override: StageSelection
    ) -> tuple[StageSelection, ...]:
        validate_stage_selection(override)
        return configured_candidate_chain(
            override, configured_service_names=tuple(self._services)
        ).candidates

    def _availability_by_service(
        self, overrides: tuple[StageSelection, ...], now: datetime
    ) -> dict[str, bool]:
        availability: dict[str, bool] = {}
        for node in overrides:
            if node.service in availability:
                continue
            availability[node.service] = self._services[node.service].is_available(
                now=now
            )
        return availability

    def _exhausted_services_for(
        self, override: StageSelection, now: datetime
    ) -> tuple[ServiceSelectionProvider, ...]:
        configured_overrides = self._configured_candidate_overrides(override)
        availability = self._availability_by_service(configured_overrides, now)
        return tuple(
            self._services[node.service]
            for node in configured_overrides
            if not availability[node.service]
        )

    def has_configured_candidate(self, override: StageSelection) -> bool:
        return configured_candidate_chain(
            override, configured_service_names=tuple(self._services)
        ).has_configured_candidate

    def resolve(self, override: StageSelection, now: datetime) -> StageSelection:
        configured_overrides = self._configured_candidate_overrides(override)
        availability = self._availability_by_service(configured_overrides, now)
        selection = select_configured_candidate_chain(
            override,
            configured_service_names=tuple(
                node.service for node in configured_overrides
            ),
            available_service_names=tuple(
                node.service
                for node in configured_overrides
                if availability[node.service]
            ),
        )
        return selection.selected_chain or override

    def has_available(self, now: datetime) -> bool:
        return any(svc.is_available(now=now) for svc in self._services.values())

    def has_available_for(self, override: StageSelection, now: datetime) -> bool:
        configured_overrides = self._configured_candidate_overrides(override)
        availability = self._availability_by_service(configured_overrides, now)
        return any(availability[node.service] for node in configured_overrides)

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
