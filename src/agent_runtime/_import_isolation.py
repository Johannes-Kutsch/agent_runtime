from __future__ import annotations

import sys
from collections.abc import Iterable

_RUNTIME_IMPORT_SNAPSHOT = frozenset(sys.modules)

def assert_runtime_import_isolation(
    *,
    importer: str,
    newly_loaded_modules: Iterable[str] | None = None,
    forbidden_prefixes: Iterable[str] = (),
) -> None:
    if newly_loaded_modules is None:
        newly_loaded_modules = frozenset(sys.modules) - _RUNTIME_IMPORT_SNAPSHOT
    imported_modules = tuple(
        name
        for name in sorted(set(newly_loaded_modules))
        if any(
            name == prefix or name.startswith(f"{prefix}.")
            for prefix in forbidden_prefixes
        )
    )
    if not imported_modules:
        return
    imported = ", ".join(imported_modules)
    raise ImportError(
        f"{importer} imported forbidden modules during runtime package "
        f"initialization: {imported}. This violates the agent_runtime package "
        "boundary."
    )

