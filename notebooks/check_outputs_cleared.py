"""Asserts a committed Kaggle notebook has every code cell's outputs cleared.

`AGENTS.md`/the brief: exam text must never be "left in a committed notebook's
outputs -- commit the notebook with outputs cleared, and add a test or a check
that asserts the committed notebook has empty outputs." This is that check.

    micromamba run -n lifespan python notebooks\\check_outputs_cleared.py [notebook.ipynb ...]

With no arguments, checks every `*.ipynb` under `notebooks/`. Exits non-zero
and names every offending cell if any code cell has a non-empty `outputs` list
or a non-null `execution_count`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent


def offending_cells(notebook_path: Path) -> list[str]:
    data = json.loads(notebook_path.read_text(encoding="utf-8"))
    problems: list[str] = []
    for i, cell in enumerate(data.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        outputs = cell.get("outputs", [])
        if outputs:
            problems.append(f"cell {i}: outputs is non-empty ({len(outputs)} output(s))")
        exec_count = cell.get("execution_count")
        if exec_count is not None:
            problems.append(f"cell {i}: execution_count is {exec_count!r}, expected null")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="*", type=Path)
    args = parser.parse_args(argv)

    targets = args.notebooks or sorted(NOTEBOOKS_DIR.glob("*.ipynb"))
    if not targets:
        print("no notebooks found")
        return 0

    any_bad = False
    for nb in targets:
        problems = offending_cells(nb)
        if problems:
            any_bad = True
            print(f"DIRTY OUTPUTS  {nb}")
            for p in problems:
                print(f"  - {p}")
        else:
            print(f"clean  {nb}")
    return 1 if any_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
