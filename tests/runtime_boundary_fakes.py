from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import agent_runtime.session as session_runtime
from agent_runtime._provider_session_adapter import (
    ProviderSessionPlanningFacts,
    ProviderSessionPlanningRequest,
)
from agent_runtime.session import RunKind


class SelectionServiceFake:
    def __init__(self, name: str, *, available: bool, wake_time: datetime) -> None:
        self.name = name
        self._available = available
        self._wake_time = wake_time

    def is_available(self, now: datetime | None = None) -> bool:
        del now
        return self._available

    def next_wake_time(self) -> datetime:
        return self._wake_time

    def mark_exhausted(self, reset_time: datetime | None) -> None:
        self._available = False
        if reset_time is not None:
            self._wake_time = reset_time

    def state_dir_relpath(
        self, role: str, namespace: str = ""
    ) -> str | None:
        del role, namespace
        return None

    def is_resumable(self, state_dir: Path) -> bool:
        del state_dir
        return False

    def valid_models(self) -> frozenset[str]:
        return frozenset()

    def valid_efforts(self) -> frozenset[str]:
        return frozenset()


class ExecutionServiceFake:
    def __init__(self, name: str) -> None:
        self.name = name

    def mark_exhausted(self, reset_time: datetime | None) -> None:
        del reset_time

    def build_command(
        self,
        model: str,
        effort: str,
        run_kind: RunKind,
        session_uuid: str | None,
        *,
        tool_policy: Any | None = None,
    ) -> str:
        del model, effort, run_kind, session_uuid, tool_policy
        return ""

    def build_env(
        self,
        state_dir_container_path: str | None = None,
        token: str | None = None,
    ) -> dict[str, str]:
        del state_dir_container_path, token
        return {}

    def run(
        self,
        lines: Any,
        on_provider_session_id: Any = None,
    ) -> Any:
        del lines, on_provider_session_id
        return iter(())

    def state_dir_relpath(
        self, role: str, namespace: str = ""
    ) -> str | None:
        del role, namespace
        return None

    def is_resumable(self, state_dir: Path) -> bool:
        del state_dir
        return False

    def valid_models(self) -> frozenset[str]:
        return frozenset()

    def valid_efforts(self) -> frozenset[str]:
        return frozenset()


class ResidentPlanningProviderSessionAdapterFake:
    @property
    def service_name(self) -> str:
        return "codex"

    def provider_session_planning_facts(
        self,
        request: ProviderSessionPlanningRequest,
    ) -> ProviderSessionPlanningFacts:
        del request
        return ProviderSessionPlanningFacts(
            state_dir_relpath="state/",
            provider_state_dir=Path("state"),
            has_resumable_provider_state=True,
        )

    def provider_session_state(self, request: Any) -> Any:
        del request
        return session_runtime.ProviderSessionState(
            run_kind=RunKind.RESUME,
            provider_session_id="recovered-session",
            state_dir_relpath="state/",
            state_dir_path=Path("state"),
        )

    def prepare_local_provider_run_state(
        self,
        provider_state_dir: Path | None,
    ) -> None:
        del provider_state_dir


class ExternalStateResidentPlanningProviderSessionAdapterFake:
    @property
    def service_name(self) -> str:
        return "codex"

    def provider_session_planning_facts(
        self,
        request: ProviderSessionPlanningRequest,
    ) -> ProviderSessionPlanningFacts:
        del request
        return ProviderSessionPlanningFacts(
            state_dir_relpath="runtime-state/",
            provider_state_dir=Path("/host/runtime-state"),
            has_resumable_provider_state=True,
        )

    def provider_session_state(self, request: Any) -> Any:
        del request
        return session_runtime.ProviderSessionState(
            run_kind=RunKind.RESUME,
            provider_session_id="recovered-session",
            state_dir_relpath="runtime-state/",
            state_dir_path=Path("/host/runtime-state"),
        )

    def prepare_local_provider_run_state(
        self,
        provider_state_dir: Path | None,
    ) -> None:
        del provider_state_dir
