"""Validator for a `results/<run_id>/` folder.

`run-operator` owns this file. It never edits a number inside a result file --
it only reads and reports. A folder either has every property below or it is
not a result (`CLAUDE.md`: "A run that cannot be reproduced did not happen.").

Checked, each one named on failure so a caller sees every problem at once, not
just the first (`AGENTS.md`/brief, wave 2):

1. All four files present: matrix.json, timings.json, config.json, commit.txt.
2. matrix.json: every exam type in `training.config`-style EXAM_TYPES
   ("perplexity", "cloze", "continuation", "continuation_summed" -- the fourth
   added 2026-09-22, stored and never headlined) has an `untrained` row of
   exactly 7 finite numbers and an `M` block of exactly 7x7 finite numbers.
   `null` (an unfinished phase) counts as non-finite: an incomplete matrix is
   not a result.
3. commit.txt: a `dirty` line exists, parses as exactly "true" or "false"
   (case-insensitive), and is "false". Missing or unparseable is a failure in
   its own right -- a dirty tree is never assumed clean.
4. timings.json: `gpu_name` and `gpu_hours` both present, `gpu_name` a
   non-empty string, `gpu_hours` a finite number >= 0.
5. The folder's own name, parsed as `{arm}_s{seed}_{commit7}_{utc}`, agrees
   with config.json's own `arm`, `seed` and the first 7 hex characters of
   `config.json["git"]["commit"]`. (The `{utc}` suffix is allowed to differ
   from config.json's recorded run_id -- a `--resume`d run keeps its original
   folder name but train.py stamps a fresh timestamp into config.json each
   time it writes.)

6. Subset runs (`train.py --phases 0,3,6`): config.json's `"phases"` must be a
   valid declared list (ints in 0..6, strictly ascending, starting at 0), and
   then only the declared rows of `M`, and within every row and the
   `untrained` row only the declared exam columns, must be finite. Rows of
   undeclared phases may be null and undeclared columns NaN; phase ids are
   never renumbered. No `"phases"` key means all seven (every run before the
   flag), so a full 0..6 folder is checked exactly as before.

This module does not import anything from `training/` (three other agents are
mid-edit on it as this is written); it reads only the four JSON/text files a
result folder contracts to produce.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

REQUIRED_FILES = ("matrix.json", "timings.json", "config.json", "commit.txt")

#: AGENTS.md, Amendments 2026-09-22: matrix.json gains a fourth key, stored and
#: never headlined. Kept local (not imported from training.config) on purpose:
#: this validator must work even while training/ is broken mid-edit.
EXAM_TYPES = ("perplexity", "cloze", "continuation", "continuation_summed")
N_PHASES = 7

RUN_ID_RE = re.compile(
    r"^(?P<arm>[A-Za-z0-9-]+)_s(?P<seed>\d+)_(?P<commit7>[0-9a-fA-F]{7})_(?P<utc>\d{8}T\d{6}Z)$"
)


class Failures(list):
    """A list of human-readable failure strings. Truthy iff non-empty."""

    def add(self, msg: str) -> None:
        self.append(msg)


def _is_finite_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def _load_json(path: Path, failures: Failures, label: str) -> dict | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        failures.add(f"{label}: could not read {path.name}: {exc}")
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        failures.add(f"{label}: {path.name} is not valid JSON: {exc}")
        return None


#: `--arm phase0` trains phase 0 only and stops (training/train.py:
#: "arm phase0 trains phase 0 only; A, B, C, D and D-nr take it from here").
#: Its matrix.json legitimately carries a full `untrained` row (evaluate_all
#: scores all seven phases before any training starts, for every arm) but
#: only row 0 of `M` -- rows 1..6 stay `null` forever, by design, because no
#: later phase is ever trained in that run. Every other arm trains and scores
#: all seven phases, so an incomplete row there still means a dead session.
PHASE0_ARM_NAME = "phase0"


def _trained_phases_for(arm: str | None, phases: tuple[int, ...] = tuple(range(N_PHASES))) -> set[int]:
    if arm == PHASE0_ARM_NAME:
        return {0}
    return set(phases)


def parse_phases(value: Any) -> tuple[int, ...] | str:
    """config.json's `"phases"` (the run's declared phases, `train.py
    --phases`), validated by the rules of `training.config.resolve_phases`,
    restated here because this module imports nothing from `training/`:
    a non-empty list of ints in 0..6, strictly ascending, starting at 0.
    Returns the tuple, or a failure message."""
    if not isinstance(value, list) or not value:
        return f"config.json 'phases' must be a non-empty list, got {value!r}"
    if any(isinstance(k, bool) or not isinstance(k, int) or not 0 <= k < N_PHASES for k in value):
        return f"config.json 'phases' {value!r} holds something that is not an int in 0..{N_PHASES - 1}"
    if any(b <= a for a, b in zip(value, value[1:])):
        return f"config.json 'phases' {value!r} is not strictly ascending"
    if value[0] != 0:
        return f"config.json 'phases' {value!r} does not start at phase 0"
    return tuple(value)


def _check_matrix(
    run_dir: Path,
    failures: Failures,
    *,
    arm: str | None,
    phases: tuple[int, ...] = tuple(range(N_PHASES)),
) -> None:
    path = run_dir / "matrix.json"
    if not path.exists():
        return  # already reported by the missing-files check
    data = _load_json(path, failures, "matrix.json")
    if not isinstance(data, dict):
        return
    if "untrained" not in data or "M" not in data:
        failures.add("matrix.json: missing 'untrained' or 'M' top-level key")
        return
    untrained = data["untrained"]
    m = data["M"]
    trained_phases = _trained_phases_for(arm, phases)
    #: Exam columns that must be scored: the declared phases. An undeclared
    #: column is NaN in a subset run by design (never examined).
    scored_columns = set(phases)
    for exam_type in EXAM_TYPES:
        # untrained row
        if not isinstance(untrained, dict) or exam_type not in untrained:
            failures.add(f"matrix.json: untrained row missing exam type {exam_type!r}")
        else:
            row = untrained[exam_type]
            if not isinstance(row, list) or len(row) != N_PHASES:
                failures.add(
                    f"matrix.json: untrained[{exam_type!r}] has "
                    f"{len(row) if isinstance(row, list) else type(row).__name__} entries, expected {N_PHASES}"
                )
            else:
                bad = [j for j, v in enumerate(row) if j in scored_columns and not _is_finite_number(v)]
                if bad:
                    failures.add(
                        f"matrix.json: untrained[{exam_type!r}] has non-finite/missing cell(s) at index {bad}"
                    )
        # M block
        if not isinstance(m, dict) or exam_type not in m:
            failures.add(f"matrix.json: M missing exam type {exam_type!r}")
            continue
        block = m[exam_type]
        if not isinstance(block, list) or len(block) != N_PHASES:
            failures.add(
                f"matrix.json: M[{exam_type!r}] has "
                f"{len(block) if isinstance(block, list) else type(block).__name__} rows, expected {N_PHASES}x{N_PHASES}"
            )
            continue
        for i, row in enumerate(block):
            if not isinstance(row, list) or len(row) != N_PHASES:
                failures.add(
                    f"matrix.json: M[{exam_type!r}][{i}] has "
                    f"{len(row) if isinstance(row, list) else type(row).__name__} entries, expected {N_PHASES}"
                )
                continue
            if i not in trained_phases:
                # phase0's untrained-forever rows (see PHASE0_ARM_NAME above), or
                # a subset run's undeclared phase: null by design.
                continue
            bad = [j for j, v in enumerate(row) if j in scored_columns and not _is_finite_number(v)]
            if bad:
                failures.add(
                    f"matrix.json: M[{exam_type!r}][{i}] has non-finite/missing cell(s) at column {bad}"
                )


def _check_commit(run_dir: Path, failures: Failures) -> None:
    path = run_dir / "commit.txt"
    if not path.exists():
        return
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        failures.add(f"commit.txt: could not read: {exc}")
        return
    dirty_value: str | None = None
    for line in text.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        if key.strip().lower() == "dirty":
            dirty_value = value.strip()
            break
    if dirty_value is None:
        failures.add("commit.txt: no 'dirty' line -- a dirty tree is never assumed clean")
        return
    lowered = dirty_value.lower()
    if lowered not in ("true", "false"):
        failures.add(f"commit.txt: 'dirty' value {dirty_value!r} is not 'true' or 'false'")
        return
    if lowered == "true":
        failures.add("commit.txt: dirty is true -- a dirty tree is not a result")


def _check_timings(run_dir: Path, failures: Failures) -> None:
    path = run_dir / "timings.json"
    if not path.exists():
        return
    data = _load_json(path, failures, "timings.json")
    if not isinstance(data, dict):
        return
    if "gpu_name" not in data or not isinstance(data["gpu_name"], str) or not data["gpu_name"].strip():
        failures.add("timings.json: missing or empty 'gpu_name'")
    if "gpu_hours" not in data or not _is_finite_number(data["gpu_hours"]) or float(data["gpu_hours"]) < 0:
        failures.add("timings.json: missing, non-finite, or negative 'gpu_hours'")


def _check_run_id(run_dir: Path, failures: Failures) -> None:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return
    config = _load_json(config_path, failures, "config.json")
    if not isinstance(config, dict):
        return
    name = run_dir.name
    match = RUN_ID_RE.match(name)
    if not match:
        failures.add(
            f"folder name {name!r} does not match run_id shape {{arm}}_s{{seed}}_{{commit7}}_{{utc}}"
        )
        return
    cfg_arm = config.get("arm")
    if cfg_arm != match.group("arm"):
        failures.add(f"run_id arm {match.group('arm')!r} disagrees with config.json arm {cfg_arm!r}")
    try:
        cfg_seed = int(config.get("seed"))
    except (TypeError, ValueError):
        failures.add(f"config.json seed {config.get('seed')!r} is not an integer")
        cfg_seed = None
    if cfg_seed is not None and cfg_seed != int(match.group("seed")):
        failures.add(f"run_id seed {match.group('seed')} disagrees with config.json seed {cfg_seed}")
    git = config.get("git")
    cfg_commit = git.get("commit") if isinstance(git, dict) else None
    if not isinstance(cfg_commit, str) or not cfg_commit:
        failures.add("config.json missing git.commit to check the run_id's commit7 against")
    elif cfg_commit[:7].lower() != match.group("commit7").lower():
        failures.add(
            f"run_id commit7 {match.group('commit7')!r} disagrees with "
            f"config.json git.commit {cfg_commit!r} (first 7: {cfg_commit[:7]!r})"
        )


def _config_for_matrix_check(run_dir: Path) -> dict:
    """config.json, read silently -- its own validity is `_check_run_id`'s
    job and is not re-reported here. Used to tell a legitimately-partial
    matrix (a `phase0` folder, a subset run) from an incomplete one."""
    path = run_dir / "config.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _arm_for_matrix_check(run_dir: Path) -> str | None:
    return _config_for_matrix_check(run_dir).get("arm")


def validate(run_dir: Path) -> Failures:
    """Validate one `results/<run_id>/` folder. Returns every failure found;
    an empty list means the folder is a valid result."""
    run_dir = Path(run_dir)
    failures = Failures()

    if not run_dir.is_dir():
        failures.add(f"{run_dir}: not a directory")
        return failures

    missing = [f for f in REQUIRED_FILES if not (run_dir / f).exists()]
    if missing:
        failures.add(f"missing required file(s): {', '.join(missing)}")
        # Still run the other checks against whatever files DO exist, so a
        # partial folder gets a full report in one pass.

    config = _config_for_matrix_check(run_dir)
    phases: tuple[int, ...] = tuple(range(N_PHASES))
    if "phases" in config:
        parsed = parse_phases(config["phases"])
        if isinstance(parsed, str):
            failures.add(parsed)
        else:
            phases = parsed
    _check_matrix(run_dir, failures, arm=config.get("arm"), phases=phases)
    _check_commit(run_dir, failures)
    _check_timings(run_dir, failures)
    _check_run_id(run_dir, failures)

    return failures


def iter_result_dirs(results_root: Path) -> list[Path]:
    """Every immediate subdirectory of `results/` that looks like a run folder
    (skips `_invalid`, `_phase0`, and anything starting with `.` or `_`, and
    anything that is not a directory)."""
    root = Path(results_root)
    if not root.is_dir():
        return []
    return sorted(
        p
        for p in root.iterdir()
        if p.is_dir() and not p.name.startswith((".", "_"))
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m results.validate",
        description="Validate one or more results/<run_id>/ folders. Exits non-zero if any fail.",
    )
    p.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="run folders to check; default: every folder directly under results/",
    )
    p.add_argument(
        "--results-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="used only when no explicit paths are given (default: this file's directory)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    targets = args.paths if args.paths else iter_result_dirs(args.results_root)
    if not targets:
        print("no result folders to check")
        return 0

    any_failed = False
    for run_dir in targets:
        failures = validate(run_dir)
        if failures:
            any_failed = True
            print(f"INVALID  {run_dir}")
            for msg in failures:
                print(f"  - {msg}")
        else:
            print(f"valid    {run_dir}")
    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
