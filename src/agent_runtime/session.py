from __future__ import annotations

import dataclasses
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Protocol

from .identity import validate_runtime_identity_label, validate_session_namespace

if TYPE_CHECKING:
    from .roles import InvocationRole
    from agent_runtime.session_planning import (
        AuthSeedingRequirement,
        LocalAuthSeedAction,
    )
else:
    AuthSeedingRequirement = object
    LocalAuthSeedAction = object


class ServiceResumeIdentityStore(Protocol):
    def session_uuid(self) -> str: ...

    def service_session_id(self, service_name: str) -> str | None: ...

    def save_service_session_id(self, service_name: str, session_id: str) -> None: ...

    def service_session_metadata(self, service_name: str) -> dict[str, str] | None: ...

    def exact_transcript_service_name(self) -> str | None: ...


_DEFAULT_PROVIDER_SESSION_ID_FILENAME = "thread_id"


class RunKind(Enum):
    FRESH = "fresh"
    RESUME = "resume"


@dataclasses.dataclass(frozen=True)
class ProviderSessionSelection:
    provider_session_id: str | None
    persist_provider_session_id: bool = False


@dataclasses.dataclass(frozen=True)
class ProviderSessionStateRequest:
    role_session: ServiceResumeIdentityStore
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
    persist_provider_session_id: bool = False
    auth_seeding_requirement: AuthSeedingRequirement | None = None
    auth_seed_action: LocalAuthSeedAction | None = None
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


def select_resumable_provider_session_id(
    role_session: ServiceResumeIdentityStore,
    service_name: str,
    *,
    provider_state_dir: Path | None,
    has_resumable_provider_state: bool,
    recover_provider_session_id: Callable[[Path | None], str | None] | None = None,
) -> ProviderSessionSelection:
    if not has_resumable_provider_state:
        return ProviderSessionSelection(provider_session_id=None)

    provider_session_id = role_session.service_session_id(service_name)
    if provider_session_id is not None:
        return ProviderSessionSelection(provider_session_id=provider_session_id)

    recover_provider_session_id = recover_provider_session_id or (
        lambda state_dir: load_state_dir_provider_session_id(
            state_dir,
            service_name,
        )
    )
    provider_session_id = recover_provider_session_id(provider_state_dir)
    if provider_session_id is None:
        return ProviderSessionSelection(provider_session_id=None)

    role_session.save_service_session_id(service_name, provider_session_id)
    return ProviderSessionSelection(
        provider_session_id=provider_session_id,
        persist_provider_session_id=True,
    )


def is_exact_resumable_service_session(
    role_session: ServiceResumeIdentityStore,
    service_name: str,
    *,
    provider_session_id: str | None,
    provider_state_dir: Path | None,
    exact_provider_session_matcher: Callable[[str | None, Path | None], bool]
    | None = None,
) -> bool:
    metadata = role_session.service_session_metadata(service_name)
    if (
        role_session.exact_transcript_service_name() != service_name
        or metadata is None
        or metadata["provider_session_id"] != provider_session_id
    ):
        return False
    if exact_provider_session_matcher is not None:
        return exact_provider_session_matcher(provider_session_id, provider_state_dir)
    return provider_session_id is not None and provider_state_dir is not None


__all__ = [
    "ProviderSessionSelection",
    "ProviderSessionState",
    "ProviderSessionStateRequest",
    "RunKind",
    "is_exact_resumable_service_session",
    "load_provider_state_session_id",
    "load_state_dir_provider_session_id",
    "normalize_state_dir_relpath",
    "provider_state_session_id_path",
    "provider_state_relpath",
    "select_resumable_provider_session_id",
]
