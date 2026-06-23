from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

from .contracts import ServiceSelectionProvider
from .identity import validate_runtime_identity_label
from .types import ProviderSelection, validate_provider_selection


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

    def _availability_for(
        self, provider_selection: ProviderSelection, now: datetime
    ) -> bool:
        service = self._services.get(provider_selection.service)
        if service is None:
            return False
        return service.is_available(now=now)

    def has_configured_candidate(self, provider_selection: ProviderSelection) -> bool:
        return provider_selection.service in self._services

    def resolve(
        self, provider_selection: ProviderSelection, now: datetime
    ) -> ProviderSelection:
        validate_provider_selection(provider_selection)
        return provider_selection

    def has_available(self, now: datetime) -> bool:
        return any(svc.is_available(now=now) for svc in self._services.values())

    def has_available_for(
        self, provider_selection: ProviderSelection, now: datetime
    ) -> bool:
        return self._availability_for(provider_selection, now)

    def next_wake_time(self, now: datetime) -> datetime | None:
        exhausted = [
            svc for svc in self._services.values() if not svc.is_available(now=now)
        ]
        if not exhausted:
            return None
        return min(svc.next_wake_time() for svc in exhausted)

    def next_wake_time_for(
        self, provider_selection: ProviderSelection, now: datetime
    ) -> datetime | None:
        service = self._services.get(provider_selection.service)
        if service is None or service.is_available(now=now):
            return None
        return service.next_wake_time()

    def mark_exhausted(self, service_name: str, *, reset_time: datetime | None) -> None:
        service = self._services.get(service_name)
        if service is None:
            return
        service.mark_exhausted(reset_time)

    def __getitem__(self, key: str) -> ServiceSelectionProvider | None:
        return self._services.get(key)
