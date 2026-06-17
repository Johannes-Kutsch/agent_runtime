from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from setuptools.build_meta import build_sdist, build_wheel  # type: ignore[import-untyped]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("dist"),
        help="Directory that receives the fresh release artifacts.",
    )
    return parser.parse_args()


def _clean_directory(path: Path) -> None:
    if path.exists():
        for child in path.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    else:
        path.mkdir(parents=True)


def main() -> int:
    args = _parse_args()
    outdir = args.outdir.resolve()

    _clean_directory(outdir)
    try:
        build_wheel(str(outdir))
        build_sdist(str(outdir))
    finally:
        shutil.rmtree(Path("build"), ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
