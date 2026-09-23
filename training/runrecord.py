"""The run record: `config.json`, `commit.txt`, `timings.json`, `matrix.json`.

`CLAUDE.md`: "A result counts when reproduced. Record commit, seed, data
hashes, hardware and driver versions with every run." The frozen layout
(`AGENTS.md`):

    results/<run_id>/   run_id = {arm}_s{seed}_{commit7}_{utc}
    matrix.json   {untrained: {exam_type: [7]}, M: {exam_type: [[7]x7]}}
    timings.json  {per_phase: [{phase, train_s, consolidate_s, eval_s}], gpu_hours, gpu_name}
    config.json   full resolved config incl. arm, seed, token budget per phase,
                  data + manifest hashes
    commit.txt    commit sha, dirty flag, torch/CUDA/driver versions

Every file is written atomically and rewritten after every phase, so a session
that dies still leaves a readable partial record. Never wrap a write here in
`try`/`except` to keep a run going.

Timing uses `torch.cuda.synchronize()` at both ends of every span, because a
CUDA call returns before the work is done and an unsynchronised timer would
quietly attribute Arm D's distillation cost to whatever ran next -- and H2 is
stated in compute.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import torch

from training.checkpoint import atomic_write_text
from training.config import EXAM_TYPES, N_PHASES

TIMER_SPANS = ("train_s", "consolidate_s", "eval_s")

#: matrix.json's keys: the three frozen EXAM_TYPES plus the summed continuation
#: score, stored and never headlined. All four are REQUIRED of `evaluate_all`
#: (lead, 2026-09-22): `report.py` and `results/validate.py` both require the
#: fourth and `evaluate_all_detailed` always produces it, so a run that cannot
#: write it should fail here rather than produce a matrix the validator rejects
#: hours later.
MATRIX_KEYS: tuple[str, ...] = EXAM_TYPES + ("continuation_summed",)

class RunRecordError(RuntimeError):
    """The record cannot be written, so the run is not a result."""


# ---------------------------------------------------------------------------
# environment


def git_info(repo_root: Path) -> dict:
    """Commit sha and dirty flag. A dirty tree is recorded, loudly, not hidden:
    `run-operator` refuses such a folder as a result."""
    root = Path(repo_root)

    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", "-C", str(root), *args],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    branch = run("rev-parse", "--abbrev-ref", "HEAD")
    if commit is None:
        return {"commit": "unknown", "dirty": True, "branch": "unknown"}
    return {"commit": commit, "dirty": bool(status), "branch": branch or "unknown"}


def gpu_name() -> str:
    if torch.cuda.is_available():  # pragma: no cover - no CUDA on the dev box
        return torch.cuda.get_device_name(0)
    return "cpu"


def driver_version() -> str:
    try:  # pragma: no cover - no CUDA on the dev box
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError):
        pass
    return "none"


def env_info() -> dict:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda": torch.version.cuda or "none",
        "cudnn": str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else "none",
        "driver": driver_version(),
        "gpu_name": gpu_name(),
        "n_gpus": torch.cuda.device_count(),
    }


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def make_run_id(arm: str, seed: int, commit: str, stamp: str | None = None) -> str:
    """`{arm}_s{seed}_{commit7}_{utc}` (AGENTS.md)."""
    return f"{arm}_s{seed}_{(commit or 'unknown')[:7]}_{stamp or utc_stamp()}"


# ---------------------------------------------------------------------------
# timings


class Timings:
    """Per-phase wall clock, split train / consolidate / eval."""

    def __init__(self, per_phase: list[dict] | None = None) -> None:
        self.per_phase: list[dict] = list(per_phase or [])
        self._current: dict | None = None
        #: (phase, span) pairs currently being timed. A hook is handed
        #: `ctx.timer` and may open the span the loop already opened around it;
        #: the inner one must not add the same seconds a second time.
        self._active: set[tuple[int, str]] = set()

    def _bucket(self, phase: int) -> dict:
        for row in self.per_phase:
            if row["phase"] == phase:
                return row
        row = {"phase": phase, **{k: 0.0 for k in TIMER_SPANS}}
        self.per_phase.append(row)
        self.per_phase.sort(key=lambda r: r["phase"])
        return row

    def set_phase(self, phase: int) -> None:
        self._current = self._bucket(phase)

    def adopt(self, row: dict) -> None:
        """Copy another run's phase row in whole -- how the shared phase-0 wall
        clock reaches each arm's timings (`PLAN.md`, "Training arms")."""
        bucket = self._bucket(int(row["phase"]))
        for key in TIMER_SPANS:
            bucket[key] = float(row.get(key, 0.0))
        bucket["shared_from_phase0_run"] = True

    @contextmanager
    def timer(self, span: str) -> Iterator[None]:
        if span not in TIMER_SPANS:
            raise RunRecordError(f"unknown timing span {span!r}; expected one of {TIMER_SPANS}")
        if self._current is None:
            raise RunRecordError("timer used before set_phase()")
        bucket = self._current
        key = (int(bucket["phase"]), span)
        if key in self._active:
            yield  # already being timed by an enclosing span
            return
        self._active.add(key)
        if torch.cuda.is_available():  # pragma: no cover
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if torch.cuda.is_available():  # pragma: no cover
                torch.cuda.synchronize()
            bucket[span] += time.perf_counter() - t0
            self._active.discard(key)

    def total_seconds(self) -> float:
        return sum(float(row.get(k, 0.0)) for row in self.per_phase for k in TIMER_SPANS)

    def to_dict(self) -> dict:
        return {
            "per_phase": self.per_phase,
            "gpu_hours": self.total_seconds() / 3600.0,
            "gpu_name": gpu_name(),
        }


# ---------------------------------------------------------------------------
# matrix


def empty_matrix(n_phases: int = N_PHASES, exam_types: tuple[str, ...] = MATRIX_KEYS) -> dict:
    """`{untrained: {exam: [7]}, M: {exam: [[7]x7]}}` with `null` where nothing
    has been scored yet, so a partial record is still valid JSON."""
    return {
        "untrained": {t: [None] * n_phases for t in exam_types},
        "M": {t: [[None] * n_phases for _ in range(n_phases)] for t in exam_types},
    }


def _check_scores(scores: dict, n_phases: int, where: str) -> tuple[str, ...]:
    """Validate and return the keys to store. All four MATRIX_KEYS are required."""
    missing = set(MATRIX_KEYS) - set(scores)
    if missing:
        raise RunRecordError(
            f"{where}: evaluate_all returned no {sorted(missing)}; all of {list(MATRIX_KEYS)} "
            "are required, and report.py and results/validate.py both reject a matrix without them"
        )
    unknown = set(scores) - set(MATRIX_KEYS)
    if unknown:
        raise RunRecordError(f"{where}: evaluate_all returned unknown keys {sorted(unknown)}")
    for exam_type in MATRIX_KEYS:
        row = scores[exam_type]
        if not isinstance(row, (list, tuple)) or len(row) != n_phases:
            raise RunRecordError(
                f"{where}: evaluate_all returned {len(row) if hasattr(row, '__len__') else '?'} "
                f"{exam_type} scores, expected exactly {n_phases}"
            )
        for j, v in enumerate(row):
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                raise RunRecordError(f"{where}: {exam_type} score for exam phase {j} is {v!r}, not a float")
    return MATRIX_KEYS


def set_untrained(matrix: dict, scores: dict, n_phases: int = N_PHASES) -> None:
    for exam_type in _check_scores(scores, n_phases, "untrained row"):
        matrix["untrained"][exam_type] = [float(v) for v in scores[exam_type]]


def set_row(matrix: dict, phase: int, scores: dict, n_phases: int = N_PHASES) -> None:
    """`M[phase][j] = score on exam j after training phase `phase`."""
    for exam_type in _check_scores(scores, n_phases, f"matrix row {phase}"):
        matrix["M"][exam_type][phase] = [float(v) for v in scores[exam_type]]


def get_row(matrix: dict, phase: int) -> dict:
    """The stored row, including `continuation_summed` when it was scored. Used
    to copy phase 0's row from the shared checkpoint into each arm's matrix."""
    return {t: list(matrix["M"][t][phase]) for t in MATRIX_KEYS}


# ---------------------------------------------------------------------------
# the record


class RunRecord:
    """The four files of `results/<run_id>/`, rewritten after every phase."""

    def __init__(self, out_dir: Path, config: dict, git: dict, env: dict) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.git = git
        self.env = env

    # -- writers ------------------------------------------------------------

    def _dump(self, obj: Any, name: str) -> None:
        atomic_write_text(json.dumps(obj, indent=2, sort_keys=False) + "\n", self.out_dir / name)

    def write_config(self) -> None:
        self._dump(self.config, "config.json")

    def write_commit(self) -> None:
        lines = [
            f"commit: {self.git['commit']}",
            f"branch: {self.git['branch']}",
            f"dirty: {str(self.git['dirty']).lower()}",
            "",
            *(f"{k}: {v}" for k, v in self.env.items()),
        ]
        if self.git["dirty"]:
            lines.insert(
                3,
                "WARNING: the working tree was dirty. A dirty tree is not a result (CLAUDE.md).",
            )
        atomic_write_text("\n".join(lines) + "\n", self.out_dir / "commit.txt")

    def write_matrix(self, matrix: dict) -> None:
        self._dump(matrix, "matrix.json")

    def write_timings(self, timings: Timings) -> None:
        self._dump(timings.to_dict(), "timings.json")

    def write_all(self, matrix: dict, timings: Timings) -> None:
        self.write_config()
        self.write_commit()
        self.write_matrix(matrix)
        self.write_timings(timings)

    def log(self, message: str) -> None:
        """Appends to `log.txt` and echoes to stdout. `PhaseContext.log`."""
        line = f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        with (self.out_dir / "log.txt").open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
