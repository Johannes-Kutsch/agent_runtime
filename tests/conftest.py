from __future__ import annotations

from collections.abc import Callable
from collections.abc import Generator
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any

import pytest
import time_machine

import agent_runtime as runtime
import agent_runtime.contracts as contracts_runtime
import agent_runtime.runtime as prompt_runtime
from agent_runtime.types import ProviderSelection as InternalStageSelection


@pytest.fixture
def stage_selection_factory() -> Callable[..., InternalStageSelection]:
    def _factory(
        service: str = "codex",
        *,
        model: str = "gpt-5.4",
        effort: str = "medium",
        auth: runtime.ProviderAuth | None = None,
    ) -> InternalStageSelection:
        return InternalStageSelection(
            service=service,
            model=model,
            effort=effort,
            auth=auth,
        )

    return _factory


@pytest.fixture
def provider_selection_factory() -> Callable[..., runtime.ProviderSelection]:
    def _factory(
        service: str = "codex",
        *,
        model: str = "gpt-5.4",
        effort: str = "medium",
        auth: runtime.ProviderAuth | None = None,
    ) -> runtime.ProviderSelection:
        return runtime.ProviderSelection(
            service=service,
            model=model,
            effort=effort,
            auth=auth,
        )

    return _factory


@pytest.fixture(autouse=True)
def frozen_clock() -> Generator[None, None, None]:
    with time_machine.travel(datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc), tick=True):
        yield


@pytest.fixture
def ephemeral_request_factory(
    provider_selection_factory: Callable[..., runtime.ProviderSelection],
) -> Callable[..., prompt_runtime.EphemeralRunRequest]:
    def _factory(
        *,
        prompt: str = "already rendered prompt",
        invocation_dir: Path = Path("."),
        stage: runtime.ProviderSelection | None = None,
        tool_access: contracts_runtime.ToolAccess | None = None,
        tool_policy: runtime.ToolPolicy = runtime.ToolPolicy.NONE,
        token: Any = None,
    ) -> prompt_runtime.EphemeralRunRequest:
        kwargs: dict[str, Any] = {"tool_policy": tool_policy}
        if tool_access is not None:
            kwargs["tool_access"] = tool_access
        return prompt_runtime.EphemeralRunRequest(
            prompt=prompt,
            invocation_dir=invocation_dir,
            provider_selection=stage or provider_selection_factory(),
            **kwargs,
            token=token,
        )

    return _factory
