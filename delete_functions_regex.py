import re
import sys
from pathlib import Path


def main() -> None:
    path = Path(sys.argv[1])
    target_names = set(sys.argv[2:])
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines(keepends=True)

    defs: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        m = re.match(r"^def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", line)
        if m:
            defs.append((i, m.group(1)))

    to_delete: set[int] = set()
    for idx, (start_line, name) in enumerate(defs):
        if name not in target_names:
            continue
        end_line = defs[idx + 1][0] - 1 if idx + 1 < len(defs) else len(lines)
        # trim trailing blank lines
        while end_line > start_line and lines[end_line - 1].strip() == "":
            end_line -= 1
        to_delete.update(range(start_line, end_line))

    new_lines = [line for i, line in enumerate(lines) if i not in to_delete]
    path.write_text("".join(new_lines), encoding="utf-8")
    print(f"Deleted {len(target_names)} functions from {path}")


if __name__ == "__main__":
    main()
