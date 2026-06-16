from __future__ import annotations

import enum


class AgentRole(enum.Enum):
    PLANNER = "planner"
    PREFLIGHT_ISSUE = "preflight_issue"
    IMPLEMENTER = "implementer"
    REVIEWER = "reviewer"
    MERGER = "merger"
    IMPROVE = "improve"
    FAILURE_REPORT = "failure_report"
    DIVERGENCE_RESOLVER = "divergence_resolver"


__all__ = ["AgentRole"]
