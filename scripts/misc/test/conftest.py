import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
for path in (str(ROOT), str(ROOT / "scripts" / "misc"), str(ROOT / "scripts" / "misc" / "tooling")):
    if path not in sys.path:
        sys.path.insert(0, path)
