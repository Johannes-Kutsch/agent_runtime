import json
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import BinaryIO

from . import _time as _time_module
from .session import RunKind


class WorkInvocationLog:
    def __init__(
        self,
        log: BinaryIO,
        *,
        log_path: Path,
        header_start: int,
        header_record: dict[str, object],
    ) -> None:
        self._log = log
        self._log_path = log_path
        self._header_start = header_start
        self._header_record = header_record

    def append_provider_chunk(self, provider_bytes: bytes) -> None:
        self._log.write(provider_bytes)
        self._log.flush()

    def record_provider_session_id(self, provider_session_id: str | None) -> None:
        if self._header_record["provider_session_id"] == provider_session_id:
            return
        self._header_record["provider_session_id"] = provider_session_id
        header_bytes = json.dumps(self._header_record).encode() + b"\n"
        with open(self._log_path, "r+b") as log_file:
            log_file.seek(self._header_start)
            remainder = log_file.read()
            newline_index = remainder.find(b"\n")
            if newline_index < 0:
                raise RuntimeError(
                    "agent invocation header missing terminating newline"
                )
            tail = remainder[newline_index + 1 :]
            log_file.seek(self._header_start)
            log_file.write(header_bytes)
            log_file.write(tail)
            log_file.truncate()


class LogicalAgentInvocationLog:
    def __init__(self, owner: "AgentInvocationLog", *, log_path: Path) -> None:
        self._owner = owner
        self.log_path = log_path
        self._latest_work_invocation: WorkInvocationLog | None = None

    @contextmanager
    def open_work_invocation(
        self,
        *,
        run_kind: RunKind,
        session_uuid: str | None,
        prompt: str,
    ) -> Iterator[WorkInvocationLog]:
        with self._owner.open_work_invocation(
            log_path=self.log_path,
            run_kind=run_kind,
            session_uuid=session_uuid,
            prompt=prompt,
        ) as work_invocation:
            self._latest_work_invocation = work_invocation
            yield work_invocation

    def append_work_invocation(
        self,
        *,
        run_kind: RunKind,
        session_uuid: str | None,
        prompt: str,
        provider_bytes: bytes,
    ) -> None:
        self._owner.append_work_invocation(
            log_path=self.log_path,
            run_kind=run_kind,
            session_uuid=session_uuid,
            prompt=prompt,
            provider_bytes=provider_bytes,
        )

    def record_provider_session_id(self, provider_session_id: str | None) -> None:
        if self._latest_work_invocation is None:
            raise RuntimeError(
                "no work invocation has been opened for this log session"
            )
        self._latest_work_invocation.record_provider_session_id(provider_session_id)


class AgentInvocationLog:
    def __init__(
        self,
        *,
        now_local: Callable[[], datetime] | None = None,
    ) -> None:
        self._now_local = _time_module.now_local if now_local is None else now_local

    def reserve(self, *, log_name: str, logs_dir: Path) -> Path:
        logs_dir.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^a-z0-9]+", "-", log_name.lower()).strip("-")
        timestamp = self._now_local().strftime("%Y%m%dT%H%M")
        stem = f"{slug}-{timestamp}"
        for suffix in ["", *[f"-{n}" for n in range(2, 10_000)]]:
            path = logs_dir / f"{stem}{suffix}.log"
            try:
                with open(path, "xb"):
                    pass
                return path
            except FileExistsError:
                continue
        raise RuntimeError(f"could not reserve unique agent log path for {stem}")

    def start_logical_session(
        self,
        *,
        log_name: str,
        logs_dir: Path,
    ) -> LogicalAgentInvocationLog:
        return LogicalAgentInvocationLog(
            self,
            log_path=self.reserve(
                log_name=log_name,
                logs_dir=logs_dir,
            ),
        )

    @contextmanager
    def open_work_invocation(
        self,
        *,
        log_path: Path,
        run_kind: RunKind,
        session_uuid: str | None,
        prompt: str,
    ) -> Iterator[WorkInvocationLog]:
        with open(log_path, "ab") as log:
            separator = self._separator_for_next_invocation(log_path)
            if separator:
                log.write(separator)
            header_start = log.tell()
            header_record: dict[str, object] = {
                "type": "agent_invocation",
                "run_kind": run_kind.value,
                "provider_session_id": session_uuid,
                "prompt": prompt,
            }
            log.write(json.dumps(header_record).encode() + b"\n")
            log.flush()
            yield WorkInvocationLog(
                log,
                log_path=log_path,
                header_start=header_start,
                header_record=header_record,
            )

    def append_work_invocation(
        self,
        *,
        log_path: Path,
        run_kind: RunKind,
        session_uuid: str | None,
        prompt: str,
        provider_bytes: bytes,
    ) -> None:
        with self.open_work_invocation(
            log_path=log_path,
            run_kind=run_kind,
            session_uuid=session_uuid,
            prompt=prompt,
        ) as log:
            log.append_provider_chunk(provider_bytes)

    def _separator_for_next_invocation(self, log_path: Path) -> bytes:
        if not log_path.exists() or log_path.stat().st_size == 0:
            return b""
        with open(log_path, "rb") as existing_log:
            existing_log.seek(-1, 2)
            return b"\n\n" if existing_log.read(1) != b"\n" else b"\n"
