import sys
import platform
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError

ROOT = Path(__file__).resolve().parents[2]          # корень репо
OUT = Path(__file__).resolve().parents[1] / "results" / "versions.txt"

lines = [f"Python {sys.version}", f"Platform {platform.platform()}", ""]

req = (ROOT / "requirements.txt").read_text(encoding="utf-8-sig")
for line in req.splitlines():
    name = line.split("==")[0].strip()
    if not name:
        continue
    try:
        v = version(name)
    except PackageNotFoundError:
        v = "NOT INSTALLED"
    lines.append(f"{name}=={v}")

text = "\n".join(lines)
print(text)
OUT.write_text(text + "\n", encoding="utf-8")
print(f"\nSaved to {OUT}")