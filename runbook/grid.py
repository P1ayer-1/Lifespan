"""The rented-A100 grid runbook: 3 phase0 runs + 6 arms x 3 seeds = 21
invocations, in one Python entry point, no shell scripts.

Order (`AGENTS.md`/brief: "the runbook must order them so every phase0
completes before the arms that load it"; `.claude/agents/run-operator.md`:
"for each seed, the shared phase-0 run first, then all six arms, then the
next seed"):

    seed 0: phase0, A, B, C, D, D-nr, E
    seed 1: phase0, A, B, C, D, D-nr, E
    seed 2: phase0, A, B, C, D, D-nr, E

so a session cut short still leaves complete arms-by-seed rather than half of
everything. `A, B, C, D, D-nr` all load seed s's `phase0` checkpoint (shared,
one `--phase0-dir` for the whole session); `E` trains from the random init and
does not wait on phase0, but still runs after it in this ordering, per the
seed-major schedule above.

This module never trains anything itself: every invocation is
`python -m training.train ...`, the frozen CLI contract, run as a subprocess.
It refuses to start on a dirty tree, skips any (arm, seed) whose result
folder already exists and validates, is resumable (a state file survives being
killed and re-run), and records every attempt -- complete, invalid or failed
-- to the GPU-hour ledger (`runbook.ledger`).

`--toy --cpu` runs the whole 21-invocation order on the toy config, on CPU,
for a local dry run (`run-operator.md`: "Before the session: a dry run of the
whole runbook on the toy config"). Without those flags this dispatches real
training and must only run on the rented A100, on the owner's go-ahead
(`CLAUDE.md`: no training on the laptop) -- this module never checks for a GPU
itself; that boundary is enforced by who runs it and with what flags, same as
every other entry point in this repo.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training.config import ARMS, SEEDS

from runbook import ledger

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results"
DEFAULT_STATE_PATH = DEFAULT_RESULTS_ROOT / "_runbook_state.json"

#: The frozen grid order (AGENTS.md / the brief): phase0 first per seed, then
#: the six arms in PLAN.md's table order. Not derived from `dict(ARMS)`'s
#: iteration order so this file states the contract in its own text rather
#: than inheriting it silently from another module.
SEQUENTIAL_ARMS = ("A", "B", "C", "D", "D-nr")
JOINT_ARMS = ("E",)
ARM_ORDER_PER_SEED = ("phase0",) + SEQUENTIAL_ARMS + JOINT_ARMS


def _load_validate():
    """`results/validate.py` loaded by file path -- it is not a conventional
    package (results/ is mostly run folders), same technique
    `tests/test_validate.py` uses."""
    spec = importlib.util.spec_from_file_location("results_validate", DEFAULT_RESULTS_ROOT / "validate.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


VALIDATE = _load_validate()


def grid_invocations(seeds: tuple[int, ...] = SEEDS) -> list[tuple[str, int]]:
    """The 21 (arm, seed) pairs in run order."""
    return [(arm, seed) for seed in seeds for arm in ARM_ORDER_PER_SEED]


# ---------------------------------------------------------------------------
# the working tree must be clean -- no import from training/ for this, a
# result's own commit.txt already records dirty; the *runbook* refuses before
# spending an hour finding out the hard way.


def git_is_dirty(repo_root: Path = REPO_ROOT) -> bool:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"could not check git status: {exc}") from exc
    if out.returncode != 0:
        raise RuntimeError(f"git status failed: {out.stderr.strip()}")
    return bool(out.stdout.strip())


def git_head_commit7(repo_root: Path = REPO_ROOT) -> str:
    out = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30, check=False
    )
    if out.returncode != 0 or not out.stdout.strip():
        raise RuntimeError(f"git rev-parse HEAD failed: {out.stderr.strip()}")
    return out.stdout.strip()[:7]


# ---------------------------------------------------------------------------
# resumable state: (arm, seed) -> the out_dir this session already picked for
# it, so re-running after a kill reuses the same folder and passes --resume
# instead of starting a 22nd, orphaned attempt.


def load_state(state_path: Path) -> dict[str, str]:
    if not state_path.exists():
        return {}
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def save_state(state: dict[str, str], state_path: Path) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_path.with_suffix(state_path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(state_path)


def _key(arm: str, seed: int) -> str:
    return f"{arm}_s{seed}"


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def find_existing_valid_run(results_root: Path, arm: str, seed: int, commit7: str):
    """Any folder already matching `{arm}_s{seed}_{commit7}_*` that validates
    -- so the runbook also recognises a result it did not itself launch this
    session (e.g. left over from a previous one)."""
    if not results_root.is_dir():
        return None
    prefix = f"{arm}_s{seed}_{commit7}_"
    for candidate in sorted(results_root.glob(f"{prefix}*")):
        if candidate.is_dir() and not VALIDATE.validate(candidate):
            return candidate
    return None


# ---------------------------------------------------------------------------
# one invocation


def build_train_argv(
    *,
    arm: str,
    seed: int,
    train_dir: Path,
    exam_dir: Path,
    out_dir: Path,
    phase0_dir: Path,
    resume: bool,
    manifest: Path | None,
    toy: bool,
    cpu: bool,
) -> list[str]:
    argv = [
        sys.executable,
        "-m",
        "training.train",
        "--arm",
        arm,
        "--seed",
        str(seed),
        "--train-dir",
        str(train_dir),
        "--exam-dir",
        str(exam_dir),
        "--out",
        str(out_dir),
        "--phase0-dir",
        str(phase0_dir),
    ]
    if manifest is not None:
        argv += ["--manifest", str(manifest)]
    if resume:
        argv.append("--resume")
    if toy:
        argv.append("--toy")
    if cpu:
        argv.append("--cpu")
    return argv


def run_grid(
    *,
    train_dir: Path,
    exam_dir: Path,
    results_root: Path = DEFAULT_RESULTS_ROOT,
    phase0_dir: Path | None = None,
    manifest: Path | None = None,
    state_path: Path = DEFAULT_STATE_PATH,
    ledger_path: Path = ledger.DEFAULT_LEDGER_PATH,
    seeds: tuple[int, ...] = SEEDS,
    toy: bool = False,
    cpu: bool = False,
    dry_run: bool = False,
    log=print,
) -> list[dict[str, Any]]:
    """Runs (or dry-runs) the 21 invocations in order. Returns one report dict
    per invocation: {arm, seed, action: skip|run, status, run_dir}.

    Skips an (arm, seed) whose result folder already exists and validates.
    Resumable: state_path remembers which out_dir an in-progress (arm, seed)
    is using, so re-running this function after it (or the process) was
    killed continues from there with `--resume` rather than starting over.
    """
    if git_is_dirty():
        raise RuntimeError("working tree is dirty; refusing to start the grid (CLAUDE.md: a dirty tree is not a result)")

    commit7 = git_head_commit7()
    phase0_dir = Path(phase0_dir) if phase0_dir else results_root / "_phase0"
    state = load_state(state_path)
    blocked_seeds: set[int] = set()
    reports: list[dict[str, Any]] = []

    for arm, seed in grid_invocations(seeds):
        requires_phase0 = ARMS[arm].shares_phase0 and arm != "phase0"
        if requires_phase0 and seed in blocked_seeds:
            log(f"SKIP  {arm} seed {seed}: phase0 for this seed did not complete; nothing to load")
            reports.append({"arm": arm, "seed": seed, "action": "skip", "status": "blocked_on_phase0", "run_dir": None})
            continue

        existing = find_existing_valid_run(results_root, arm, seed, commit7)
        if existing is not None:
            log(f"SKIP  {arm} seed {seed}: already valid at {existing}")
            reports.append({"arm": arm, "seed": seed, "action": "skip", "status": "already_complete", "run_dir": str(existing)})
            continue

        key = _key(arm, seed)
        resume = key in state
        out_dir = Path(state[key]) if resume else results_root / f"{arm}_s{seed}_{commit7}_{_utc_stamp()}"
        state[key] = str(out_dir)
        save_state(state, state_path)

        argv = build_train_argv(
            arm=arm,
            seed=seed,
            train_dir=Path(train_dir),
            exam_dir=Path(exam_dir),
            out_dir=out_dir,
            phase0_dir=phase0_dir,
            resume=resume,
            manifest=manifest,
            toy=toy,
            cpu=cpu,
        )

        if dry_run:
            log(f"DRY   {' '.join(argv)}")
            reports.append({"arm": arm, "seed": seed, "action": "dry_run", "status": "n/a", "run_dir": str(out_dir)})
            continue

        log(f"RUN   {'(resuming) ' if resume else ''}{arm} seed {seed} -> {out_dir}")
        proc = subprocess.run(argv, cwd=str(REPO_ROOT))
        gpu_hours, gpu_name = ledger.gpu_hours_from_timings(out_dir)

        if proc.returncode != 0:
            status = "failed"
        else:
            failures = VALIDATE.validate(out_dir) if out_dir.exists() else ["did not produce a result folder"]
            status = "complete" if not failures else "invalid"
            if failures:
                log(f"      {arm} seed {seed}: ran but did not validate: {failures}")

        ledger.record_attempt(
            arm=arm,
            seed=seed,
            run_dir=out_dir,
            status=status,
            gpu_hours=gpu_hours,
            gpu_name=gpu_name,
            commit7=commit7,
            note="" if status == "complete" else f"exit code {proc.returncode}",
            ledger_path=ledger_path,
        )
        reports.append({"arm": arm, "seed": seed, "action": "run", "status": status, "run_dir": str(out_dir)})

        if status != "complete" and arm == "phase0":
            blocked_seeds.add(seed)
            log(f"      phase0 seed {seed} did not complete; A/B/C/D/D-nr for seed {seed} will be skipped")

    return reports


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m runbook.grid",
        description="Run the 21-invocation Lifespan grid (3 phase0 + 6 arms x 3 seeds) in order.",
    )
    p.add_argument("--train-dir", type=Path, required=True)
    p.add_argument("--exam-dir", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    p.add_argument("--phase0-dir", type=Path, default=None)
    p.add_argument("--state-path", type=Path, default=DEFAULT_STATE_PATH)
    p.add_argument("--ledger-path", type=Path, default=ledger.DEFAULT_LEDGER_PATH)
    p.add_argument("--toy", action="store_true", help="a local dry run on the toy config, CPU only")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="print the commands without running any of them")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reports = run_grid(
        train_dir=args.train_dir,
        exam_dir=args.exam_dir,
        results_root=args.results_root,
        phase0_dir=args.phase0_dir,
        manifest=args.manifest,
        state_path=args.state_path,
        ledger_path=args.ledger_path,
        toy=args.toy,
        cpu=args.cpu,
        dry_run=args.dry_run,
    )
    n_run = sum(1 for r in reports if r["action"] == "run")
    n_skip = sum(1 for r in reports if r["action"] == "skip")
    n_failed = sum(1 for r in reports if r["status"] in ("failed", "invalid", "blocked_on_phase0"))
    print(f"\n{len(reports)} invocations: {n_run} run, {n_skip} skipped, {n_failed} not complete")
    return 1 if n_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
