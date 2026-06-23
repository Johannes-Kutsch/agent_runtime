import ast
import sys
from pathlib import Path


def delete_functions(source: str, names: set[str]) -> str:
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    removals: list[tuple[int, int]] = []
    top_level = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    for i, node in enumerate(top_level):
        if node.name not in names:
            continue
        start = node.lineno - 1
        if i + 1 < len(top_level):
            end = top_level[i + 1].lineno - 2
        else:
            end = len(lines) - 1
        # remove trailing blank lines before next def by trimming end downward
        while end >= start and lines[end].strip() == "":
            end -= 1
        removals.append((start, end))
    # remove in reverse
    for start, end in sorted(removals, reverse=True):
        del lines[start : end + 1]
    return "".join(lines)


def main() -> None:
    path = Path(sys.argv[1])
    names = set(sys.argv[2:])
    source = path.read_text(encoding="utf-8")
    new_source = delete_functions(source, names)
    path.write_text(new_source, encoding="utf-8")
    print(f"Deleted {len(names)} functions from {path}")


if __name__ == "__main__":
    main()
