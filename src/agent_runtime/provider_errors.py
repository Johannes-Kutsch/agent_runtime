from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class ProviderErrorObservation:
    service_name: str
    raw_provider_text: str
    source_stream: str
    status_code: int | None = None
    provider_code: str | None = None
    error_name: str | None = None


__all__ = ["ProviderErrorObservation"]
