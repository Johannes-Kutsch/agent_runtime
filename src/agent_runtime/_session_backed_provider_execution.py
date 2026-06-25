from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, cast

from . import _builtin_runtime_client as _builtin_runtime_client_module
from . import _builtin_provider_rendering as _builtin_provider_rendering_module
from ._builtin_provider_stream_interpretation import BuiltInProviderStreamInterpretation
from ._portable_continuation_payload import (
    create_portable_continuation_payload,
    read_portable_continuation_payload,
)
from ._provider_invocation import (
    ProviderInvocationAdapter,
    ProviderInvocationFailure,
    ProviderInvocationResult,
)
from ._runtime_lifecycle import (
    Continuation,
    NewSessionRunRequest,
    ProviderAuth,
    ProviderUsage,
    ResumedSessionRunRequest,
    RunResult,
)
from .errors import RuntimeConfigurationError
from .invocation_progress import InvocationProgress
from .session import RunKind, provider_state_relpath
from .contracts import ToolAccess
from .types import ProviderSelection, ResolvedProvider


def _codex_provider_state_dir_relpath(
    *,
    role: Any,
    session_namespace: str,
) -> str:
    return cast(
        str,
        _builtin_runtime_client_module.provider_state_relpath(
            role,
            "codex",
            session_namespace,
        ),
    )


def _codex_prepare_runtime_state(
    runtime_state_dir: Path,
    *,
    role: Any,
    session_namespace: str,
) -> tuple[str, Path]:
    provider_state_dir_relpath = _codex_provider_state_dir_relpath(
        role=role,
        session_namespace=session_namespace,
    )
    provider_state_dir = runtime_state_dir / provider_state_dir_relpath
    provider_state_dir.mkdir(parents=True, exist_ok=True)
    return provider_state_dir_relpath, provider_state_dir


def _opencode_provider_state_dir_relpath(
    *,
    role: Any,
    session_namespace: str,
) -> str:
    return cast(
        str,
        _builtin_runtime_client_module.provider_state_relpath(
            role,
            "opencode",
            session_namespace,
        ),
    )


def _opencode_prepare_runtime_state(
    runtime_state_dir: Path,
    *,
    role: Any,
    session_namespace: str,
) -> tuple[str, Path]:
    provider_state_dir_relpath = _opencode_provider_state_dir_relpath(
        role=role,
        session_namespace=session_namespace,
    )
    provider_state_dir = runtime_state_dir / provider_state_dir_relpath
    provider_state_dir.mkdir(parents=True, exist_ok=True)
    return provider_state_dir_relpath, provider_state_dir


def _load_opencode_state_dir_session_id(state_dir: Path | None) -> str | None:
    if state_dir is None:
        return None
    path = state_dir / _builtin_runtime_client_module._OPENCODE_SESSION_ID_FILENAME
    if not path.is_file():
        return None
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return value or None


def _opencode_is_resumable(state_dir: Path) -> bool:
    return (state_dir / "resume.jsonl").is_file() or (
        state_dir / _builtin_runtime_client_module._OPENCODE_SESSION_ID_FILENAME
    ).is_file()


def _opencode_provider_state_from_runtime_dir(
    state_dir: Path | None,
) -> dict[str, Any]:
    if state_dir is None:
        return {}
    provider_state: dict[str, Any] = {}
    session_id = _load_opencode_state_dir_session_id(state_dir)
    if session_id is not None:
        provider_state["session_id"] = session_id
    resume_jsonl_path = state_dir / "resume.jsonl"
    if resume_jsonl_path.is_file():
        try:
            provider_state["resume_jsonl"] = resume_jsonl_path.read_text(
                encoding="utf-8"
            )
        except OSError:
            pass
    return provider_state


def _build_opencode_continuation(
    *,
    model: str,
    effort: str,
    tool_access: ToolAccess,
    provider_session_id: str,
    provider_state: dict[str, Any] | None = None,
    provider_state_dir: Path | None = None,
    exact_transcript_match: bool | None = None,
) -> Continuation:
    if provider_state is None and provider_state_dir is not None:
        provider_state = _opencode_provider_state_from_runtime_dir(provider_state_dir)
    if provider_state is None:
        provider_state = {}
    provider_resume_state: dict[str, Any] = {
        "provider_session_id": provider_session_id,
        "provider_state": provider_state,
    }
    if exact_transcript_match is not None:
        provider_resume_state["exact_transcript_match"] = exact_transcript_match
    return create_portable_continuation_payload(
        service_name="opencode",
        model=model,
        effort=effort,
        tool_access=tool_access,
        provider_resume_state=provider_resume_state,
    ).to_continuation()


def _persist_opencode_session_id(state_dir: Path, provider_session_id: str) -> None:
    (
        state_dir / _builtin_runtime_client_module._OPENCODE_SESSION_ID_FILENAME
    ).write_text(
        f"{provider_session_id}\n",
        encoding="utf-8",
    )


def _seed_opencode_provider_state_dir(
    state_dir: Path,
    provider_state: dict[str, Any] | None,
) -> None:
    for state_filename in (
        _builtin_runtime_client_module._OPENCODE_SESSION_ID_FILENAME,
        "resume.jsonl",
    ):
        state_path = state_dir / state_filename
        if state_path.exists():
            state_path.unlink()
    if not isinstance(provider_state, dict):
        return
    provider_session_id = provider_state.get("session_id")
    if isinstance(provider_session_id, str) and provider_session_id:
        _persist_opencode_session_id(state_dir, provider_session_id)
    resume_jsonl = provider_state.get("resume_jsonl")
    if isinstance(resume_jsonl, str) and resume_jsonl:
        (state_dir / "resume.jsonl").write_text(
            resume_jsonl,
            encoding="utf-8",
        )


def _restore_opencode_state_dir(
    request: ResumedSessionRunRequest,
    continuation_provider_state: dict[str, Any] | None,
) -> tuple[Path, Callable[[], None]]:
    if request._runtime_state_dir is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="opencode-provider-state-")

        def cleanup() -> None:
            temp_dir.cleanup()

        state_dir = Path(temp_dir.name)
        _seed_opencode_provider_state_dir(state_dir, continuation_provider_state)
        return state_dir, cleanup
    provider_state_dir_relpath = _opencode_provider_state_dir_relpath(
        role="implementer",
        session_namespace=request._session_namespace,
    )
    state_dir = request._runtime_state_dir / provider_state_dir_relpath
    state_dir.mkdir(parents=True, exist_ok=True)
    _seed_opencode_provider_state_dir(state_dir, continuation_provider_state)
    return state_dir, lambda: None


def _opencode_exact_transcript_match(
    *,
    saved_exact_transcript_match: bool,
    provider_session_id: str | None,
    state_dir_session_id: str | None,
) -> bool:
    return (
        saved_exact_transcript_match
        and provider_session_id is not None
        and state_dir_session_id == provider_session_id
    )


def _read_codex_rollout_thread_ids(rollout_path: Path) -> set[str]:
    thread_ids: set[str] = set()
    if not rollout_path.is_file():
        return thread_ids
    try:
        for line in rollout_path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict) or event.get("type") != "thread.started":
                continue
            thread_id = event.get("thread_id")
            if isinstance(thread_id, str):
                stripped = thread_id.strip()
                if stripped:
                    thread_ids.add(stripped)
    except (OSError, UnicodeDecodeError):
        return set()
    return thread_ids


def _recover_codex_rollout_thread_id(state_dir: Path | None) -> str | None:
    if state_dir is None:
        return None
    sessions_dir = state_dir / "sessions"
    if not sessions_dir.is_dir():
        return None
    thread_ids: set[str] = set()
    for rollout_path in sessions_dir.rglob("rollout-*.jsonl"):
        thread_ids.update(_read_codex_rollout_thread_ids(rollout_path))
        if len(thread_ids) > 1:
            return None
    if len(thread_ids) != 1:
        return None
    return next(iter(thread_ids))


def _codex_is_resumable(state_dir: Path) -> bool:
    sessions_dir = state_dir / "sessions"
    if not sessions_dir.is_dir():
        return False
    return any(sessions_dir.rglob("rollout-*.jsonl"))


def _resolve_recoverable_codex_session_id(
    *,
    provider_state_dir: Path,
    provider_session_id: str | None,
) -> str:
    recovered_thread_id = _recover_codex_rollout_thread_id(provider_state_dir)
    if not _codex_is_resumable(provider_state_dir) or recovered_thread_id is None:
        raise RuntimeConfigurationError(
            "Codex continuation is not recoverable from provider state."
        )
    if provider_session_id:
        return provider_session_id
    return recovered_thread_id


def _codex_seed_auth(provider_state_dir: Path) -> None:
    provider_auth_path = provider_state_dir / "auth.json"
    if provider_auth_path.exists():
        return
    host_auth_path = _builtin_runtime_client_module._codex_host_auth_path()
    if not host_auth_path.exists():
        raise _builtin_runtime_client_module._missing_codex_auth_error()
    shutil.copyfile(host_auth_path, provider_auth_path)


def _build_codex_continuation(
    *,
    model: str,
    effort: str,
    tool_access: ToolAccess,
    provider_session_id: str,
    provider_state_dir_relpath: str | None = None,
) -> Continuation:
    provider_resume_state: dict[str, Any] = {
        "run_kind": RunKind.RESUME.value,
        "provider_session_id": provider_session_id,
        "exact_transcript_match": False,
    }
    if provider_state_dir_relpath is not None:
        provider_resume_state["provider_state_dir_relpath"] = provider_state_dir_relpath
    return create_portable_continuation_payload(
        service_name="codex",
        model=model,
        effort=effort,
        tool_access=tool_access,
        provider_resume_state=provider_resume_state,
    ).to_continuation()


def _claude_provider_state_dir_relpath(
    *,
    role: Any,
    session_namespace: str,
) -> str:
    return cast(str, provider_state_relpath(role, "claude", session_namespace))


def _claude_is_resumable(state_dir: Path) -> bool:
    return state_dir.is_dir() and any(path.is_file() for path in state_dir.rglob("*"))


def _claude_prepare_runtime_state(
    runtime_state_dir: Path,
    *,
    role: Any,
    session_namespace: str,
) -> tuple[str, Path]:
    provider_state_dir_relpath = _claude_provider_state_dir_relpath(
        role=role,
        session_namespace=session_namespace,
    )
    provider_state_dir = runtime_state_dir / provider_state_dir_relpath
    provider_state_dir.mkdir(parents=True, exist_ok=True)
    return provider_state_dir_relpath, provider_state_dir


def _claude_run_kind_for_state_dir(state_dir: Path | None) -> RunKind:
    if state_dir is None:
        return RunKind.RESUME
    if _claude_is_resumable(state_dir):
        return RunKind.RESUME
    return RunKind.FRESH


def _build_claude_continuation(
    *,
    model: str,
    effort: str,
    tool_access: ToolAccess,
    provider_session_id: str,
) -> Continuation:
    return create_portable_continuation_payload(
        service_name="claude",
        model=model,
        effort=effort,
        tool_access=tool_access,
        provider_resume_state={
            "run_kind": RunKind.RESUME.value,
            "provider_session_id": provider_session_id,
            "exact_transcript_match": False,
        },
    ).to_continuation()


def _require_claude_auth(auth: ProviderAuth | None) -> None:
    _builtin_provider_rendering_module._require_claude_auth(auth)


def _session_backed_service_name(request: ResumedSessionRunRequest) -> str:
    if request.continuation is not None:
        continuation_payload = read_portable_continuation_payload(request.continuation)
        return continuation_payload.service_name
    assert request.session_plan is not None
    return request.session_plan.service.name


def _resolve_active_provider_session_id(
    *,
    stream_interpretation: BuiltInProviderStreamInterpretation,
    invocation_result: ProviderInvocationResult | ProviderInvocationFailure,
    prepared_or_continuation_provider_session_id: str | None,
) -> str | None:
    if stream_interpretation.extract_provider_session_id is not None:
        observed_provider_session_id = (
            stream_interpretation.extract_provider_session_id(
                list(invocation_result.stdout_lines)
            )
        )
        if observed_provider_session_id is not None:
            return observed_provider_session_id
    if invocation_result.provider_session_id is not None:
        return invocation_result.provider_session_id
    return prepared_or_continuation_provider_session_id


def _interruption_continuation(
    *,
    provider_work_started: bool,
    provider_session_id: str | None,
    build_continuation: Callable[[str], Continuation],
) -> Continuation | None:
    if not provider_work_started or provider_session_id is None:
        return None
    return build_continuation(provider_session_id)


def _completed_result(
    *,
    output: str,
    usage: ProviderUsage | None,
    continuation: Continuation | None,
    service: str,
    model: str,
    effort: str,
) -> RunResult:
    return RunResult(
        output=output,
        usage=usage,
        continuation=continuation,
        selected=ResolvedProvider(service=service, model=model, effort=effort),
    )


def _run_builtin_new_session(
    request: NewSessionRunRequest,
    *,
    provider_invocation_adapter: ProviderInvocationAdapter | None = None,
    on_live_output: Callable[[Any], None] | None = None,
):
    invocation_adapter = (
        _builtin_runtime_client_module._default_provider_invocation_adapter()
        if provider_invocation_adapter is None
        else provider_invocation_adapter
    )
    runtime_state_dir, cleanup_runtime_state_dir, is_caller_managed_runtime_state = (
        _builtin_runtime_client_module._new_session_runtime_state_dir(
            request,
            context="new-session",
        )
    )
    try:
        if (
            _builtin_runtime_client_module.supported_builtin_provider_selection(
                request.provider_selection
            )
            is None
        ):
            raise RuntimeConfigurationError(
                "RuntimeClient requires at least one supported built-in service candidate."
            )
        selected_stage = _builtin_runtime_client_module._select_builtin_stage(
            request.provider_selection
        )
        selected_stage_auth = _builtin_runtime_client_module._selection_auth(
            selected_stage
        )
        _builtin_runtime_client_module._require_portable_continuation_support(
            selected_stage.service
        )

        def _portable_codex_state_dir_relpath(
            provider_state_dir_relpath: str | None,
        ) -> str | None:
            if is_caller_managed_runtime_state:
                return provider_state_dir_relpath
            return None

        if selected_stage.service == "codex":
            _builtin_runtime_client_module._validate_codex_stage(selected_stage)
            provider_state_dir_relpath, provider_state_dir = (
                _codex_prepare_runtime_state(
                    runtime_state_dir,
                    role="implementer",
                    session_namespace=request._session_namespace,
                )
            )
            _codex_seed_auth(provider_state_dir)
            recovered_thread_id = _recover_codex_rollout_thread_id(provider_state_dir)
            if (
                _codex_is_resumable(provider_state_dir)
                and recovered_thread_id is not None
            ):
                return _run_builtin_resumed_session(
                    _builtin_runtime_client_module.ResumedSessionRunRequest(
                        prompt=request.prompt,
                        invocation_dir=request.invocation_dir,
                        _runtime_state_dir=runtime_state_dir,
                        continuation=_build_codex_continuation(
                            model=selected_stage.model,
                            effort=selected_stage.effort,
                            tool_access=request.tool_access,
                            provider_session_id=recovered_thread_id,
                            provider_state_dir_relpath=_portable_codex_state_dir_relpath(
                                provider_state_dir_relpath
                            ),
                        ),
                        provider_auth=selected_stage_auth,
                        on_live_output=on_live_output,
                        timeout_seconds=0,
                        _session_namespace=request._session_namespace,
                    ),
                    provider_invocation_adapter=invocation_adapter,
                    on_live_output=on_live_output,
                )
            provider_session_id: str | None = None
            invocation_result = (
                _builtin_runtime_client_module._invoke_codex_new_session_provider(
                    provider_invocation_adapter=invocation_adapter,
                    request=request,
                    stage=selected_stage,
                    provider_state_dir=provider_state_dir,
                    on_live_output=on_live_output,
                )
            )
            if isinstance(invocation_result, ProviderInvocationFailure):
                provider_session_id = _resolve_active_provider_session_id(
                    stream_interpretation=(
                        _builtin_runtime_client_module._codex_stream_interpretation()
                    ),
                    invocation_result=invocation_result,
                    prepared_or_continuation_provider_session_id=provider_session_id,
                )
                failure_error = _builtin_runtime_client_module._provider_invocation_error_from_failure(
                    "codex",
                    invocation_result,
                )
                failure_error.continuation = _interruption_continuation(
                    provider_work_started=(
                        failure_error.invocation_progress is InvocationProgress.STARTED
                    ),
                    provider_session_id=provider_session_id,
                    build_continuation=lambda active_provider_session_id: (
                        _build_codex_continuation(
                            model=selected_stage.model,
                            effort=selected_stage.effort,
                            tool_access=request.tool_access,
                            provider_session_id=active_provider_session_id,
                            provider_state_dir_relpath=(
                                _portable_codex_state_dir_relpath(
                                    provider_state_dir_relpath
                                )
                            ),
                        )
                    ),
                )
                raise failure_error
            else:
                provider_session_id = _resolve_active_provider_session_id(
                    stream_interpretation=(
                        _builtin_runtime_client_module._codex_stream_interpretation()
                    ),
                    invocation_result=invocation_result,
                    prepared_or_continuation_provider_session_id=None,
                )
                result_text = invocation_result.output
                usage = invocation_result.usage
            return _completed_result(
                output=result_text,
                usage=usage,
                continuation=(
                    _build_codex_continuation(
                        model=selected_stage.model,
                        effort=selected_stage.effort,
                        tool_access=request.tool_access,
                        provider_session_id=provider_session_id,
                        provider_state_dir_relpath=_portable_codex_state_dir_relpath(
                            provider_state_dir_relpath
                        ),
                    )
                    if provider_session_id is not None
                    else None
                ),
                service="codex",
                model=selected_stage.model,
                effort=selected_stage.effort,
            )
        elif selected_stage.service == "claude":
            provider_state_dir_relpath, provider_state_dir = (
                _claude_prepare_runtime_state(
                    runtime_state_dir,
                    role="implementer",
                    session_namespace=request._session_namespace,
                )
            )
            if _claude_is_resumable(provider_state_dir):
                return _run_builtin_resumed_session(
                    _builtin_runtime_client_module.ResumedSessionRunRequest(
                        prompt=request.prompt,
                        invocation_dir=request.invocation_dir,
                        _runtime_state_dir=runtime_state_dir,
                        on_live_output=on_live_output,
                        timeout_seconds=0,
                        continuation=_build_claude_continuation(
                            model=selected_stage.model,
                            effort=selected_stage.effort,
                            tool_access=request.tool_access,
                            provider_session_id=_builtin_runtime_client_module._new_provider_session_id(),
                        ),
                        provider_auth=selected_stage_auth,
                        _session_namespace=request._session_namespace,
                    ),
                    provider_invocation_adapter=invocation_adapter,
                    on_live_output=on_live_output,
                )
            _builtin_runtime_client_module._validate_claude_stage(selected_stage)
            _require_claude_auth(selected_stage_auth)
            provider_session_id = (
                _builtin_runtime_client_module._new_provider_session_id()
            )
            run_kind = RunKind.FRESH
            exact_transcript_match = False
        elif selected_stage.service == "opencode":
            provider_state_dir_relpath, provider_state_dir = (
                _opencode_prepare_runtime_state(
                    runtime_state_dir,
                    role="implementer",
                    session_namespace=request._session_namespace,
                )
            )
            _builtin_runtime_client_module._validate_opencode_stage(selected_stage)
            _builtin_runtime_client_module._require_opencode_auth(selected_stage_auth)
            recovered_state_dir_session_id = _load_opencode_state_dir_session_id(
                provider_state_dir
            )
            if (
                _opencode_is_resumable(provider_state_dir)
                and recovered_state_dir_session_id
            ):
                provider_session_id = recovered_state_dir_session_id
                run_kind = RunKind.RESUME
                exact_transcript_match = True
            else:
                provider_session_id = (
                    _builtin_runtime_client_module._new_provider_session_id()
                )
                run_kind = RunKind.FRESH
                exact_transcript_match = False
        else:
            raise RuntimeConfigurationError(
                "RuntimeClient session-backed execution is only implemented for Claude, Codex, and OpenCode."
            )
        if selected_stage.service == "claude":
            assert provider_session_id is not None
            invocation_result = (
                _builtin_runtime_client_module._invoke_claude_new_session_provider(
                    provider_invocation_adapter=invocation_adapter,
                    request=request,
                    stage=selected_stage,
                    provider_state_dir=provider_state_dir,
                    run_kind=run_kind,
                    provider_session_id=provider_session_id,
                    on_live_output=on_live_output,
                )
            )
        else:
            assert provider_session_id is not None
            invocation_result = (
                _builtin_runtime_client_module._invoke_opencode_new_session_provider(
                    provider_invocation_adapter=invocation_adapter,
                    request=request,
                    stage=selected_stage,
                    provider_state_dir=provider_state_dir,
                    run_kind=run_kind,
                    provider_session_id=provider_session_id,
                    on_live_output=on_live_output,
                )
            )
        stream_interpretation = (
            _builtin_runtime_client_module._stream_interpretation_for_service(
                selected_stage.service
            )
        )
        if isinstance(invocation_result, ProviderInvocationFailure):
            provider_session_id = _resolve_active_provider_session_id(
                stream_interpretation=stream_interpretation,
                invocation_result=invocation_result,
                prepared_or_continuation_provider_session_id=provider_session_id,
            )
            if selected_stage.service == "opencode" and provider_session_id is not None:
                _persist_opencode_session_id(provider_state_dir, provider_session_id)
            failure_error = (
                _builtin_runtime_client_module._provider_invocation_error_from_failure(
                    selected_stage.service,
                    invocation_result,
                )
            )
            failure_error.continuation = _interruption_continuation(
                provider_work_started=(
                    failure_error.invocation_progress is InvocationProgress.STARTED
                ),
                provider_session_id=provider_session_id,
                build_continuation=lambda active_provider_session_id: (
                    _build_claude_continuation(
                        model=selected_stage.model,
                        effort=selected_stage.effort,
                        tool_access=request.tool_access,
                        provider_session_id=active_provider_session_id,
                    )
                    if selected_stage.service == "claude"
                    else _build_opencode_continuation(
                        model=selected_stage.model,
                        effort=selected_stage.effort,
                        tool_access=request.tool_access,
                        provider_session_id=active_provider_session_id,
                        provider_state_dir=provider_state_dir,
                        exact_transcript_match=exact_transcript_match,
                    )
                ),
            )
            raise failure_error
        provider_session_id = _resolve_active_provider_session_id(
            stream_interpretation=stream_interpretation,
            invocation_result=invocation_result,
            prepared_or_continuation_provider_session_id=provider_session_id,
        )
        if selected_stage.service == "opencode" and provider_session_id is not None:
            _persist_opencode_session_id(provider_state_dir, provider_session_id)
        assert provider_session_id is not None
        result_text = invocation_result.output
        usage = invocation_result.usage
        return _completed_result(
            output=result_text,
            usage=usage,
            continuation=(
                _build_claude_continuation(
                    model=selected_stage.model,
                    effort=selected_stage.effort,
                    tool_access=request.tool_access,
                    provider_session_id=provider_session_id,
                )
                if selected_stage.service == "claude"
                else _build_opencode_continuation(
                    model=selected_stage.model,
                    effort=selected_stage.effort,
                    tool_access=request.tool_access,
                    provider_session_id=provider_session_id,
                    provider_state_dir=provider_state_dir,
                    exact_transcript_match=exact_transcript_match,
                )
            ),
            service=selected_stage.service,
            model=selected_stage.model,
            effort=selected_stage.effort,
        )
    finally:
        cleanup_runtime_state_dir()


def _run_builtin_resumed_session(
    request: ResumedSessionRunRequest,
    *,
    provider_invocation_adapter: ProviderInvocationAdapter | None = None,
    on_live_output: Callable[[Any], None] | None = None,
):
    invocation_adapter = (
        _builtin_runtime_client_module._default_provider_invocation_adapter()
        if provider_invocation_adapter is None
        else provider_invocation_adapter
    )
    runtime_state_dir = request._runtime_state_dir
    continuation = request.continuation
    if continuation is None:
        raise RuntimeConfigurationError(
            "RuntimeClient resumed-session execution requires a continuation."
        )
    try:
        continuation_payload = read_portable_continuation_payload(continuation)
    except TypeError as exc:
        raise RuntimeConfigurationError(str(exc)) from exc
    continuation_service = continuation_payload.service_name
    _builtin_runtime_client_module._require_portable_continuation_support(
        continuation_service
    )
    provider_resume_state = continuation_payload.provider_resume_state
    provider_session_id: str | None
    provider_state_dir_relpath: str | None = None
    provider_state_dir: Path | None = None

    def _no_cleanup() -> None:
        return None

    if continuation_service == "codex":
        _builtin_runtime_client_module._validate_codex_stage(
            ProviderSelection(
                service="codex",
                model=request.model,
                effort=request.effort,
            )
        )
        provider_state_dir_relpath = cast(
            str | None,
            provider_resume_state.get("provider_state_dir_relpath"),
        )
        provider_session_id = cast(
            str | None,
            provider_resume_state.get("provider_session_id"),
        )
        if provider_session_id is not None:
            provider_session_id = provider_session_id.strip() or None
        if runtime_state_dir is not None and provider_state_dir_relpath:
            provider_state_dir = runtime_state_dir / provider_state_dir_relpath
            provider_state_dir.mkdir(parents=True, exist_ok=True)
            _codex_seed_auth(provider_state_dir)
            provider_session_id = _resolve_recoverable_codex_session_id(
                provider_state_dir=provider_state_dir,
                provider_session_id=provider_session_id,
            )
        elif provider_session_id is None:
            raise RuntimeConfigurationError(
                "Codex continuation is missing `provider_session_id`."
            )
        active_provider_session_id: str | None = provider_session_id
        invocation_result = (
            _builtin_runtime_client_module._invoke_codex_resumed_session_provider(
                provider_invocation_adapter=invocation_adapter,
                provider_session_id=provider_session_id,
                request=request,
                provider_state_dir=provider_state_dir,
                on_live_output=on_live_output,
            )
        )
        if isinstance(invocation_result, ProviderInvocationFailure):
            active_provider_session_id = _resolve_active_provider_session_id(
                stream_interpretation=(
                    _builtin_runtime_client_module._codex_stream_interpretation()
                ),
                invocation_result=invocation_result,
                prepared_or_continuation_provider_session_id=(
                    active_provider_session_id
                ),
            )
            failure_error = (
                _builtin_runtime_client_module._provider_invocation_error_from_failure(
                    "codex",
                    invocation_result,
                )
            )
            failure_error.continuation = _interruption_continuation(
                provider_work_started=(
                    failure_error.invocation_progress is InvocationProgress.STARTED
                ),
                provider_session_id=active_provider_session_id,
                build_continuation=lambda resumed_provider_session_id: (
                    _build_codex_continuation(
                        model=request.model,
                        effort=request.effort,
                        tool_access=request.tool_access,
                        provider_session_id=resumed_provider_session_id,
                        provider_state_dir_relpath=provider_state_dir_relpath,
                    )
                ),
            )
            raise failure_error
        else:
            active_provider_session_id = _resolve_active_provider_session_id(
                stream_interpretation=(
                    _builtin_runtime_client_module._codex_stream_interpretation()
                ),
                invocation_result=invocation_result,
                prepared_or_continuation_provider_session_id=provider_session_id,
            )
            result_text = invocation_result.output
            usage = invocation_result.usage
        return _completed_result(
            output=result_text,
            usage=usage,
            continuation=(
                _build_codex_continuation(
                    model=request.model,
                    effort=request.effort,
                    tool_access=request.tool_access,
                    provider_session_id=active_provider_session_id,
                    provider_state_dir_relpath=provider_state_dir_relpath,
                )
                if active_provider_session_id is not None
                else None
            ),
            service="codex",
            model=request.model,
            effort=request.effort,
        )
    if continuation_service not in {"claude", "opencode"}:
        raise RuntimeConfigurationError(
            "RuntimeClient session-backed execution is only implemented for Claude, Codex, and OpenCode."
        )
    provider_session_id = cast(
        str | None,
        provider_resume_state.get("provider_session_id"),
    )
    if continuation_service == "claude":
        _require_claude_auth(request.provider_auth)
        provider_state_dir_relpath = cast(
            str | None,
            provider_resume_state.get("provider_state_dir_relpath"),
        )
        if provider_state_dir_relpath and request._runtime_state_dir is not None:
            provider_state_dir = request._runtime_state_dir / provider_state_dir_relpath
            provider_state_dir.mkdir(parents=True, exist_ok=True)
        state_dir_session_id = None
        run_kind = _claude_run_kind_for_state_dir(provider_state_dir)
        exact_transcript_match = False
        if not provider_session_id:
            provider_session_id = (
                _builtin_runtime_client_module._new_provider_session_id()
            )
        cleanup_opencode_state_dir = _no_cleanup
    else:
        _builtin_runtime_client_module._require_opencode_auth(request.provider_auth)
        continuation_provider_state = provider_resume_state.get("provider_state")
        if not isinstance(continuation_provider_state, dict):
            continuation_provider_state = None
        provider_state_dir, cleanup_opencode_state_dir = _restore_opencode_state_dir(
            request=request,
            continuation_provider_state=continuation_provider_state,
        )
        state_dir_session_id = _load_opencode_state_dir_session_id(provider_state_dir)
        saved_exact_transcript_match = bool(
            provider_resume_state.get("exact_transcript_match", False)
        )
        if provider_session_id is None:
            provider_session_id = state_dir_session_id
        if provider_session_id is None:
            provider_session_id = (
                _builtin_runtime_client_module._new_provider_session_id()
            )
        exact_transcript_match = _opencode_exact_transcript_match(
            saved_exact_transcript_match=saved_exact_transcript_match,
            provider_session_id=provider_session_id,
            state_dir_session_id=state_dir_session_id,
        )
        run_kind = RunKind.RESUME
    prompt_path = _builtin_runtime_client_module._builtin_provider_prompt_path(
        request.invocation_dir
    )

    if continuation_service == "claude":
        command_argv = _builtin_runtime_client_module._claude_command(
            model=request.model,
            effort=request.effort,
            tool_access=request.tool_access,
            run_kind=run_kind,
            session_uuid=provider_session_id,
        )
        command = _builtin_runtime_client_module._claude_legacy_command_text(
            model=request.model,
            effort=request.effort,
            tool_access=request.tool_access,
            prompt_path=prompt_path,
            run_kind=run_kind,
            session_uuid=provider_session_id,
        )
        environment = _builtin_runtime_client_module._claude_env(
            auth=request.provider_auth,
            state_dir_container_path=(
                str(provider_state_dir) if provider_state_dir is not None else None
            ),
        )

        stream_interpretation = _builtin_runtime_client_module._with_observed_output(
            _builtin_runtime_client_module._claude_stream_interpretation(),
            on_live_output,
        )
    else:
        command_argv = _builtin_runtime_client_module._opencode_command(
            model=request.model,
            effort=request.effort,
            run_kind=run_kind,
            session_uuid=provider_session_id,
        )
        command = _builtin_runtime_client_module._legacy_command_text(
            command_argv,
            prompt_path,
            opencode_prompt_substitution=True,
        )
        environment = _builtin_runtime_client_module._opencode_env(
            auth=request.provider_auth,
            state_dir_container_path=str(provider_state_dir),
            tool_policy=request.tool_access.tool_policy,
        )
        stream_interpretation = (
            _builtin_runtime_client_module._opencode_stream_interpretation(
                on_live_output=on_live_output,
                fallback_provider_session_id=provider_session_id,
            )
        )
    active_provider_session_interpretation = (
        _builtin_runtime_client_module._stream_interpretation_for_service(
            continuation_service
        )
    )
    invocation_result = _builtin_runtime_client_module._invoke_provider(
        provider_invocation_adapter=invocation_adapter,
        command="" if continuation_service == "opencode" else command,
        command_argv=command_argv,
        prefer_argv=(continuation_service in {"claude", "opencode"}),
        worktree=request.invocation_dir,
        environment=environment,
        prompt_content=request.prompt,
        prompt_path=prompt_path,
        cleanup_prompt_path=True,
        run_kind=run_kind,
        provider_session_id=provider_session_id,
        stream_interpretation=stream_interpretation,
    )
    if isinstance(invocation_result, ProviderInvocationFailure):
        provider_session_id = _resolve_active_provider_session_id(
            stream_interpretation=active_provider_session_interpretation,
            invocation_result=invocation_result,
            prepared_or_continuation_provider_session_id=provider_session_id,
        )
        if continuation_service == "opencode":
            if provider_session_id is not None:
                assert provider_state_dir is not None
                _persist_opencode_session_id(provider_state_dir, provider_session_id)
            exact_transcript_match = _opencode_exact_transcript_match(
                saved_exact_transcript_match=saved_exact_transcript_match,
                provider_session_id=provider_session_id,
                state_dir_session_id=state_dir_session_id,
            )
        failure_error = (
            _builtin_runtime_client_module._provider_invocation_error_from_failure(
                continuation_service,
                invocation_result,
            )
        )
        failure_error.continuation = _interruption_continuation(
            provider_work_started=(
                failure_error.invocation_progress is InvocationProgress.STARTED
            ),
            provider_session_id=provider_session_id,
            build_continuation=lambda active_provider_session_id: (
                (
                    _build_claude_continuation(
                        model=request.model,
                        effort=request.effort,
                        tool_access=request.tool_access,
                        provider_session_id=active_provider_session_id,
                    )
                )
                if continuation_service == "claude"
                else (
                    _build_opencode_continuation(
                        model=request.model,
                        effort=request.effort,
                        tool_access=request.tool_access,
                        provider_session_id=active_provider_session_id,
                        provider_state_dir=provider_state_dir,
                        exact_transcript_match=exact_transcript_match,
                    )
                )
            ),
        )
        cleanup_opencode_state_dir()
        raise failure_error
    provider_session_id = _resolve_active_provider_session_id(
        stream_interpretation=active_provider_session_interpretation,
        invocation_result=invocation_result,
        prepared_or_continuation_provider_session_id=provider_session_id,
    )
    if continuation_service == "opencode":
        assert provider_session_id is not None
        assert provider_state_dir is not None
        exact_transcript_match = _opencode_exact_transcript_match(
            saved_exact_transcript_match=saved_exact_transcript_match,
            provider_session_id=provider_session_id,
            state_dir_session_id=state_dir_session_id,
        )
        _persist_opencode_session_id(provider_state_dir, provider_session_id)
    assert provider_session_id is not None
    result_text = invocation_result.output
    usage = invocation_result.usage
    if continuation_service == "claude":
        result_continuation = _build_claude_continuation(
            model=request.model,
            effort=request.effort,
            tool_access=request.tool_access,
            provider_session_id=provider_session_id,
        )
    else:
        result_continuation = _build_opencode_continuation(
            model=request.model,
            effort=request.effort,
            tool_access=request.tool_access,
            provider_session_id=provider_session_id,
            provider_state_dir=provider_state_dir,
            exact_transcript_match=exact_transcript_match,
        )
    cleanup_opencode_state_dir()
    return _completed_result(
        output=result_text,
        usage=usage,
        continuation=result_continuation,
        service=continuation_service,
        model=request.model,
        effort=request.effort,
    )
