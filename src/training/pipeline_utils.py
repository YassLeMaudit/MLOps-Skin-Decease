from __future__ import annotations

from pathlib import Path
from typing import Iterable, List


def dockerized_python_command(project_root: Path | str, script: str, args: Iterable[str] | None = None) -> str:
    """Return a shell-safe command that cd's into project root then runs the python script."""
    root = Path(project_root)
    parts: List[str] = ["cd", str(root), "&&", "python", script]
    if args:
        parts.extend(str(arg) for arg in args)
    return " ".join(parts)
