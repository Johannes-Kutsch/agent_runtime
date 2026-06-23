from pathlib import Path
import sys

for path_str in sys.argv[1:]:
    path = Path(path_str)
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    cleaned = [line for line in lines if not line.strip() == ","]
    path.write_text("".join(cleaned), encoding="utf-8")
    print(f"Cleaned {path}")
