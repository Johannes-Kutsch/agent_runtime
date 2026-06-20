from __future__ import annotations

__all__: list[str] = []


def __getattr__(name: str) -> object:
    raise AttributeError(
        f"{name} is not part of the Runtime Consumer Surface; "
        "provider-session adapter seams are internal runtime details."
    )
