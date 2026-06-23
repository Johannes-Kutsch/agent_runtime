from __future__ import annotations

import ast
import sys
import tokenize
from pathlib import Path

TARGET_NAMES = {
    "InternalStageSelection",
    "ProviderSelection",
    "stage_selection_factory",
    "runtime.ProviderSelection",
    "prompt_runtime.ProviderSelection",
}


def _call_func_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_func_name(node.value)
        if prefix is None:
            return None
        return f"{prefix}.{node.attr}"
    return None


def _find_fallback_regions(
    source: str, path: Path
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    tree = ast.parse(source, filename=str(path))
    regions: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func_name = _call_func_name(node.func)
        if func_name not in TARGET_NAMES:
            continue
        for kw in node.keywords:
            if kw.arg != "fallback":
                continue
            if kw.end_lineno is None or kw.end_col_offset is None:
                continue
            start = (kw.lineno, kw.col_offset)
            end = (kw.end_lineno, kw.end_col_offset)
            regions.append((start, end))
    return regions


def _remove_regions(
    source: str, regions: list[tuple[tuple[int, int], tuple[int, int]]]
) -> str:
    lines = source.splitlines(keepends=True)
    regions = sorted(regions, key=lambda r: (r[0][0], r[0][1]))
    line_starts: list[int] = [0]
    for line in lines:
        line_starts.append(line_starts[-1] + len(line))

    def to_offset(pos: tuple[int, int]) -> int:
        return line_starts[pos[0] - 1] + pos[1]

    tokens = list(tokenize.generate_tokens(iter(source.splitlines()).__next__))

    removals: list[tuple[int, int]] = []
    for (s_line, s_col), (e_line, e_col) in regions:
        start_offset = to_offset((s_line, s_col))
        end_offset = to_offset((e_line, e_col))
        preceding_comma_idx = None
        for i, tok in enumerate(tokens):
            tok_end = to_offset((tok.end[0], tok.end[1]))
            if tok_end <= start_offset:
                if tok.type == tokenize.OP and tok.string == ",":
                    preceding_comma_idx = i
            else:
                break
        if preceding_comma_idx is not None:
            comma_start = to_offset(
                (
                    tokens[preceding_comma_idx].start[0],
                    tokens[preceding_comma_idx].start[1],
                )
            )
            removals.append((comma_start, end_offset))
        else:
            removals.append((start_offset, end_offset))
    result = source
    for start, end in sorted(removals, reverse=True):
        result = result[:start] + result[end:]
    return result


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]]
    for path in paths:
        source = path.read_text(encoding="utf-8")
        regions = _find_fallback_regions(source, path)
        if not regions:
            continue
        new_source = _remove_regions(source, regions)
        path.write_text(new_source, encoding="utf-8")
        print(f"Removed {len(regions)} fallback kwargs from {path}")


if __name__ == "__main__":
    main()
