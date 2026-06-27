from __future__ import annotations

import shutil
from pathlib import Path

from setuptools import setup  # type: ignore[import-untyped]
from setuptools.command.build_py import build_py as _build_py  # type: ignore[import-untyped]


class build_py(_build_py):
    def run(self) -> None:
        # Reset the staged runtime package so retired modules cannot leak from prior builds.
        shutil.rmtree(Path(self.build_lib) / "agent_runtime", ignore_errors=True)
        super().run()


setup(cmdclass={"build_py": build_py})
