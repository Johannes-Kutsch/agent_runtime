from __future__ import annotations

import dataclasses
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from .identity import validate_runtime_identity_label, validate_session_namespace

if TYPE_CHECKING:
    from .roles import InvocationRole


_DEFAULT_PROVIDER_SESSION_ID_FILENAME = "thread_id"


class RunKind(Enum):
    FRESH = "fresh"
    RESUME = "resume"


@dataclasses.dataclass(frozen=True)
class ProviderSessionStateRequest:
    provider_state_dir: Path | None
    has_resumable_provider_state: bool
    state_dir_relpath: str | None = None
    require_exact_transcript_match: bool = False


@dataclasses.dataclass(frozen=True)
class ProviderSessionState:
    run_kind: RunKind
    provider_session_id: str | None
    state_dir_relpath: str | None = None
    state_dir_path: Path | None = None
    exact_transcript_match: bool = False
    use_service_state_dir_for_container: bool = False


def provider_state_relpath(
    role: "InvocationRole",
    provider_name: str,
    namespace: str = "",
    *,
    session_root: str = "",
) -> str:
    validate_runtime_identity_label(
        provider_name,
        kind="Provider state service name",
    )
    validate_session_namespace(namespace)
    base = f"{role.value}/{provider_name}/"
    if namespace:
        base = f"{role.value}/{namespace}/{provider_name}/"
    return f"{session_root}/{base}" if session_root else base


def normalize_state_dir_relpath(
    role: "InvocationRole",
    namespace: str,
    service_name: str,
    state_dir_relpath: str | None,
    *,
    session_root: str | None = None,
) -> str | None:
    validate_runtime_identity_label(
        service_name,
        kind="Provider state service name",
    )
    validate_session_namespace(namespace)
    if state_dir_relpath is None or not namespace:
        return state_dir_relpath
    session_root = session_root or _session_root_for_relpath(state_dir_relpath)
    legacy_relpath = provider_state_relpath(
        role, service_name, session_root=session_root
    )
    if state_dir_relpath == legacy_relpath:
        return provider_state_relpath(
            role,
            service_name,
            namespace,
            session_root=session_root,
        )
    return state_dir_relpath


def _session_root_for_relpath(state_dir_relpath: str) -> str:
    stripped = state_dir_relpath.strip("/")
    parts = stripped.split("/")
    if len(parts) >= 3:
        return parts[0]
    return ""


def provider_state_session_id_path(
    state_dir: Path,
    service_name: str,
    *,
    session_id_filename: str = _DEFAULT_PROVIDER_SESSION_ID_FILENAME,
) -> Path:
    del service_name
    return state_dir / session_id_filename


def load_provider_state_session_id(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        value = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return None
    return value or None


def load_state_dir_provider_session_id(
    state_dir: Path | None,
    service_name: str,
    *,
    session_id_filename: str = _DEFAULT_PROVIDER_SESSION_ID_FILENAME,
) -> str | None:
    if state_dir is None:
        return None
    return load_provider_state_session_id(
        provider_state_session_id_path(
            state_dir,
            service_name,
            session_id_filename=session_id_filename,
        )
    )


__all__ = [
    "ProviderSessionState",
    "ProviderSessionStateRequest",
    "RunKind",
    "load_provider_state_session_id",
    "load_state_dir_provider_session_id",
    "normalize_state_dir_relpath",
    "provider_state_session_id_path",
    "provider_state_relpath",
]
