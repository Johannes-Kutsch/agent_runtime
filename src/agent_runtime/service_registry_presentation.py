from __future__ import annotations

from collections.abc import Callable

from .contracts import ServiceSelectionProvider
from .service_registry import ServiceRegistry

ServiceSummaryRenderer = Callable[[str, ServiceSelectionProvider], str | None]


def summary_lines(
    registry: ServiceRegistry,
    render_summary_line: ServiceSummaryRenderer,
) -> list[str]:
    lines = []
    for name, service in registry.services.items():
        line = render_summary_line(name, service)
        if line is None:
            continue
        lines.append(line)
    return lines


__all__ = ["ServiceSummaryRenderer", "summary_lines"]
