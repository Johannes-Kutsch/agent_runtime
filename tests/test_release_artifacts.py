from __future__ import annotations

import shutil
import subprocess
import sys
import tarfile
import zipfile
from email import message_from_bytes
from pathlib import Path

from setuptools.build_meta import build_sdist, build_wheel  # type: ignore[import-untyped]


def _build_release_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    try:
        wheel_name = build_wheel(str(tmp_path))
        sdist_name = build_sdist(str(tmp_path))
        return tmp_path / wheel_name, tmp_path / sdist_name
    finally:
        shutil.rmtree(Path("build"), ignore_errors=True)


def _run_release_build(outdir: Path) -> subprocess.CompletedProcess[str]:
    repo_root = Path(__file__).resolve().parents[1]
    return subprocess.run(
        [
            sys.executable,
            "scripts/build_release_artifacts.py",
            "--outdir",
            str(outdir),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )


def test_release_artifacts_ship_typing_marker_without_package_build_metadata(
    tmp_path: Path,
) -> None:
    wheel_path, sdist_path = _build_release_artifacts(tmp_path)

    with zipfile.ZipFile(wheel_path) as wheel_archive:
        wheel_members = set(wheel_archive.namelist())

    assert "agent_runtime/py.typed" in wheel_members
    assert "agent_runtime/pyproject.toml" not in wheel_members

    with tarfile.open(sdist_path, "r:gz") as sdist_archive:
        sdist_members = {member.name for member in sdist_archive.getmembers()}

    package_root = f"{sdist_path.name.removesuffix('.tar.gz')}/src/agent_runtime"
    assert f"{package_root}/py.typed" in sdist_members
    assert f"{package_root}/pyproject.toml" not in sdist_members


def test_release_artifacts_omit_private_and_retired_modules(
    tmp_path: Path,
) -> None:
    wheel_path, sdist_path = _build_release_artifacts(tmp_path)

    with zipfile.ZipFile(wheel_path) as wheel_archive:
        wheel_members = set(wheel_archive.namelist())

    assert "agent_runtime/provider_session_adapter.py" not in wheel_members
    assert "agent_runtime/_provider_session_adapter.py" not in wheel_members
    assert "agent_runtime/execution_contracts.py" not in wheel_members
    assert "agent_runtime/service_registry.py" not in wheel_members
    assert "agent_runtime/session_planning.py" not in wheel_members

    with tarfile.open(sdist_path, "r:gz") as sdist_archive:
        sdist_members = {member.name for member in sdist_archive.getmembers()}

    package_root = f"{sdist_path.name.removesuffix('.tar.gz')}/src/agent_runtime"
    assert f"{package_root}/provider_session_adapter.py" not in sdist_members
    assert f"{package_root}/_provider_session_adapter.py" not in sdist_members
    assert f"{package_root}/execution_contracts.py" not in sdist_members
    assert f"{package_root}/service_registry.py" not in sdist_members


def test_release_wheel_metadata_matches_verified_python_support(
    tmp_path: Path,
) -> None:
    wheel_path, _ = _build_release_artifacts(tmp_path)

    with zipfile.ZipFile(wheel_path) as wheel_archive:
        metadata_name = next(
            name
            for name in wheel_archive.namelist()
            if name.endswith(".dist-info/METADATA")
        )
        metadata = message_from_bytes(wheel_archive.read(metadata_name))

    assert metadata["Requires-Python"] == ">=3.11"
    assert metadata.get_all("Classifier") == [
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ]
    assert all(
        'extra == "dev"' in requirement
        for requirement in metadata.get_all("Requires-Dist", [])
    )


def test_release_build_output_contains_only_fresh_runtime_artifacts(
    tmp_path: Path,
) -> None:
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    stale_artifact = dist_dir / "stale.txt"
    stale_artifact.write_text("stale", encoding="utf-8")
    stale_directory = dist_dir / "stale-dir"
    stale_directory.mkdir()
    (stale_directory / "nested.txt").write_text("stale", encoding="utf-8")

    result = _run_release_build(dist_dir)

    assert result.returncode == 0, result.stderr
    assert not stale_artifact.exists()
    assert not stale_directory.exists()


def test_release_artifacts_ignore_retired_facades_left_in_build_directory(
    tmp_path: Path,
) -> None:
    stale_build_root = Path("build/lib/agent_runtime")
    stale_build_root.mkdir(parents=True, exist_ok=True)
    for retired_module in (
        "execution_contracts.py",
        "provider_session_adapter.py",
        "service_registry.py",
        "session_planning.py",
        "_provider_session_adapter.py",
    ):
        (stale_build_root / retired_module).write_text(
            "# stale facade\n", encoding="utf-8"
        )

    wheel_path, sdist_path = _build_release_artifacts(tmp_path)

    with zipfile.ZipFile(wheel_path) as wheel_archive:
        wheel_members = set(wheel_archive.namelist())

    assert "agent_runtime/execution_contracts.py" not in wheel_members
    assert "agent_runtime/provider_session_adapter.py" not in wheel_members
    assert "agent_runtime/service_registry.py" not in wheel_members
    assert "agent_runtime/session_planning.py" not in wheel_members
    assert "agent_runtime/_provider_session_adapter.py" not in wheel_members

    with tarfile.open(sdist_path, "r:gz") as sdist_archive:
        sdist_members = {member.name for member in sdist_archive.getmembers()}

    package_root = f"{sdist_path.name.removesuffix('.tar.gz')}/src/agent_runtime"
    assert f"{package_root}/execution_contracts.py" not in sdist_members
    assert f"{package_root}/provider_session_adapter.py" not in sdist_members
    assert f"{package_root}/service_registry.py" not in sdist_members
    assert f"{package_root}/session_planning.py" not in sdist_members
    assert f"{package_root}/_provider_session_adapter.py" not in sdist_members


def test_release_build_output_uses_runtime_artifact_names(tmp_path: Path) -> None:
    dist_dir = tmp_path / "dist"

    result = _run_release_build(dist_dir)

    assert result.returncode == 0, result.stderr

    built_artifacts = sorted(path.name for path in dist_dir.iterdir())
    assert len(built_artifacts) == 2
    assert (
        sum(
            artifact.startswith("ruhken_agent_runtime-")
            and artifact.endswith(".tar.gz")
            for artifact in built_artifacts
        )
        == 1
    )
    assert (
        sum(
            artifact.startswith("ruhken_agent_runtime-") and artifact.endswith(".whl")
            for artifact in built_artifacts
        )
        == 1
    )
