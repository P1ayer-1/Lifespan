"""The GPU-hour ledger: one row per run attempt, arm and seed, under
`results/`. `CLAUDE.md`/`PLAN.md`: "H2 is stated in compute as well as
accuracy," and a failed attempt is part of what an arm costs to operate even
though it is reported separately from the forgetting ratio
(`.claude/agents/run-operator.md`).

The ledger is plain JSON (`results/gpu_hours_ledger.json`): a list of records,
one per attempt, never rewritten in place after the fact -- a rerun of the
same (arm, seed) is a new record, not an edit of the old one, so a failed
attempt stays on the books. Read with `load()`, appended to with
`record_attempt()`, summarised per arm with `summarize()`.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_LEDGER_PATH = Path(__file__).resolve().parent.parent / "results" / "gpu_hours_ledger.json"


def load(ledger_path: Path = DEFAULT_LEDGER_PATH) -> list[dict[str, Any]]:
    path = Path(ledger_path)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def save(records: list[dict[str, Any]], ledger_path: Path = DEFAULT_LEDGER_PATH) -> None:
    _atomic_write(Path(ledger_path), json.dumps(records, indent=2) + "\n")


def record_attempt(
    *,
    arm: str,
    seed: int,
    run_dir: Path,
    status: str,
    gpu_hours: float | None,
    gpu_name: str | None,
    commit7: str | None = None,
    note: str = "",
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> dict[str, Any]:
    """Appends one record. `status` is "complete", "invalid" (ran but the
    result folder does not validate) or "failed" (the subprocess itself
    errored). `gpu_hours`/`gpu_name` come from timings.json when it exists,
    even for a failed or invalid attempt -- a dead session still burned time.
    """
    record = {
        "arm": arm,
        "seed": seed,
        "run_dir": str(run_dir),
        "status": status,
        "gpu_hours": float(gpu_hours) if gpu_hours is not None else None,
        "gpu_name": gpu_name,
        "commit7": commit7,
        "note": note,
        "recorded_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    records = load(ledger_path)
    records.append(record)
    save(records, ledger_path)
    return record


def gpu_hours_from_timings(run_dir: Path) -> tuple[float | None, str | None]:
    """Reads `gpu_hours`/`gpu_name` straight from `timings.json` if it exists.
    No import from `training/`: this is the same shape `results/validate.py`
    reads, by the frozen contract, not by trusting the writer's code."""
    path = Path(run_dir) / "timings.json"
    if not path.exists():
        return None, None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None, None
    hours = data.get("gpu_hours")
    name = data.get("gpu_name")
    hours = float(hours) if isinstance(hours, (int, float)) else None
    name = name if isinstance(name, str) and name.strip() else None
    return hours, name


def summarize(records: list[dict[str, Any]] | None = None, ledger_path: Path = DEFAULT_LEDGER_PATH) -> dict[str, dict[str, Any]]:
    """Per-arm totals: gpu_hours summed over every attempt (complete or not),
    a count of attempts, and the set of GPU names seen -- so a mixed-GPU arm
    is visible before anyone compares its numbers to another arm's."""
    records = records if records is not None else load(ledger_path)
    out: dict[str, dict[str, Any]] = {}
    for r in records:
        bucket = out.setdefault(
            r["arm"], {"attempts": 0, "gpu_hours_total": 0.0, "gpu_names": set(), "statuses": {}}
        )
        bucket["attempts"] += 1
        if r.get("gpu_hours") is not None:
            bucket["gpu_hours_total"] += float(r["gpu_hours"])
        if r.get("gpu_name"):
            bucket["gpu_names"].add(r["gpu_name"])
        bucket["statuses"][r["status"]] = bucket["statuses"].get(r["status"], 0) + 1
    for bucket in out.values():
        bucket["gpu_names"] = sorted(bucket["gpu_names"])
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m runbook.ledger", description="Inspect the GPU-hour ledger.")
    p.add_argument("--ledger-path", type=Path, default=DEFAULT_LEDGER_PATH)
    p.add_argument("--summarize", action="store_true", help="per-arm totals instead of every row")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.summarize:
        for arm, bucket in sorted(summarize(ledger_path=args.ledger_path).items()):
            print(
                f"{arm:8s} attempts={bucket['attempts']:2d} gpu_hours={bucket['gpu_hours_total']:.3f} "
                f"gpus={bucket['gpu_names']} statuses={bucket['statuses']}"
            )
    else:
        for r in load(args.ledger_path):
            print(
                f"{r['recorded_at']} {r['arm']:8s} seed={r['seed']} {r['status']:8s} "
                f"gpu_hours={r['gpu_hours']} gpu={r['gpu_name']} commit={r['commit7']} {r['run_dir']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
