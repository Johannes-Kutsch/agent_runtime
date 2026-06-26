from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Any

from agent_runtime.session import RunKind


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

    def state_dir_relpath(self, role: str, namespace: str = "") -> str | None:
        del role, namespace
        return None

    def is_resumable(self, state_dir: Path) -> bool:
        del state_dir
        return False

    def valid_models(self) -> frozenset[str]:
        return frozenset()

    def valid_efforts(self) -> frozenset[str]:
        return frozenset()
