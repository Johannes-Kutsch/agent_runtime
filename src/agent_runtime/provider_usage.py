from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class ProviderUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    cost_usd: float | None = None
    duration_seconds: float | None = None


__all__ = ["ProviderUsage"]
