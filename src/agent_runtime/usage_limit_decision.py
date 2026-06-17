from __future__ import annotations

import dataclasses
from datetime import datetime
from typing import Callable, TypeAlias

from .service_registry import ServiceRegistry
from .types import StageOverride
from .usage_limit_scope import UsageLimitScope


@dataclasses.dataclass(frozen=True)
class UsageLimitOutcome:
    reset_time: datetime | None = None
    provider: str | None = None
    raw_message: str | None = None
    account_label: str | None = None
    is_permanent: bool = False
    usage_limit_scope: UsageLimitScope | None = None


@dataclasses.dataclass(frozen=True)
class ContinueNow:
    message: str | None = None
    exhausted_wake_time: datetime | None = None
    usage_limit_scope: UsageLimitScope | None = None


@dataclasses.dataclass(frozen=True)
class SleepUntil:
    wake_time: datetime
    message: str
    is_estimated: bool = False
    usage_limit_scope: UsageLimitScope | None = None


@dataclasses.dataclass(frozen=True)
class Stop:
    message: str | None = None
    usage_limit_scope: UsageLimitScope | None = None


UsageLimitContinuationDecision: TypeAlias = ContinueNow | SleepUntil | Stop
WakeTimeComputer = Callable[[datetime | None, datetime], tuple[datetime, bool]]


def _fmt_wake(wake: datetime, now: datetime) -> str:
    local_wake = wake.astimezone(now.tzinfo) if now.tzinfo is not None else wake
    if local_wake.date() != now.date():
        return f"{local_wake:%b} {local_wake.day}, {local_wake:%H:%M}"
    return local_wake.strftime("%H:%M")


def _sleep_message(wake: datetime, now: datetime, *, is_estimated: bool) -> str:
    suffix = " (estimated)" if is_estimated else ""
    return (
        f"Usage limit reached. Sleeping until {_fmt_wake(wake, now)}{suffix}."
        " Press Ctrl+C to abort."
    )


def _permanent_exhaustion_message(outcome: UsageLimitOutcome) -> str:
    provider_label = outcome.provider or "claude"
    account = outcome.account_label or "unknown"
    message = (
        f"{provider_label} {account} account retired for this run and will be retried "
        "on the next run."
    )
    if outcome.raw_message:
        message += (
            f" {_provider_message_label(provider_label)} said: {outcome.raw_message}"
        )
    return message


def _provider_message_label(provider_label: str) -> str:
    known_labels = {
        "claude": "Claude",
        "codex": "Codex",
        "opencode": "OpenCode",
    }
    return known_labels.get(provider_label, provider_label)


def decide_usage_limit_continuation(
    outcome: UsageLimitOutcome,
    *,
    stage_override: StageOverride | None,
    service_registry: ServiceRegistry | None,
    now: datetime,
    compute_wake_time: WakeTimeComputer,
) -> UsageLimitContinuationDecision:
    use_stage_scope = service_registry is not None and stage_override is not None

    if service_registry is None:
        has_available = False
    elif use_stage_scope:
        assert stage_override is not None
        has_available = service_registry.has_available_for(stage_override, now)
    else:
        has_available = service_registry.has_available(now)

    if has_available:
        if service_registry is None:
            exhausted_wake_time = None
        elif use_stage_scope:
            assert stage_override is not None
            exhausted_wake_time = service_registry.next_wake_time_for(
                stage_override, now
            )
        else:
            exhausted_wake_time = service_registry.next_wake_time(now)
        message = None
        if outcome.is_permanent:
            message = _permanent_exhaustion_message(outcome)
        elif exhausted_wake_time is not None:
            message = (
                f"Account exhausted until {_fmt_wake(exhausted_wake_time, now)}, "
                "switching to next available."
            )
        return ContinueNow(
            message=message,
            exhausted_wake_time=exhausted_wake_time,
            usage_limit_scope=outcome.usage_limit_scope,
        )

    if outcome.is_permanent:
        return Stop(
            message=_permanent_exhaustion_message(outcome),
            usage_limit_scope=outcome.usage_limit_scope,
        )

    if service_registry is None:
        next_wake = None
    elif use_stage_scope:
        assert stage_override is not None
        next_wake = service_registry.next_wake_time_for(stage_override, now)
    else:
        next_wake = service_registry.next_wake_time(now)

    if next_wake is not None:
        return SleepUntil(
            wake_time=next_wake,
            message=_sleep_message(next_wake, now, is_estimated=False),
            usage_limit_scope=outcome.usage_limit_scope,
        )

    wake_time, is_estimated = compute_wake_time(outcome.reset_time, now)
    return SleepUntil(
        wake_time=wake_time,
        is_estimated=is_estimated,
        message=_sleep_message(wake_time, now, is_estimated=is_estimated),
        usage_limit_scope=outcome.usage_limit_scope,
    )
