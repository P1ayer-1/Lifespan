"""The operational resume round trip, proved on the toy config, on CPU only.

`trainer-core` has a resume test *inside* `training/` (state.pt round-trips
in-process). This is the operational version: it goes through the same hop a
Kaggle session survival does -- a live run is killed, its result folder is
copied to a staging directory (standing in for "uploaded to a Kaggle dataset
version") and copied back out to a fresh directory (standing in for "the next
session pulls the latest version"), and *that* copy is resumed. It is the one
that would catch a checkpoint the notebook forgot to upload, because it never
touches the live run's original directory again after the kill.

Every invocation of `training.train` goes through `python -m training.train`
as a subprocess -- the frozen CLI contract, not the module's internals -- so
this script is unaffected by `training/`'s in-flight edits beyond needing the
contract itself to run. `--toy --cpu` only: this never trains a real model and
never touches a GPU (`CLAUDE.md`: a CPU smoke test is not training).

Run it directly:

    micromamba run -n lifespan python -m runbook.resume_roundtrip

or import `prove_resume_roundtrip` and call it from a test.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from training.config import EXAM_TYPES, N_PHASES

REPO_ROOT = Path(__file__).resolve().parent.parent


class ResumeProofError(RuntimeError):
    """The round trip could not be proved -- either a race in the harness
    (the run finished before the kill point, or never reached it) or a real
    disagreement between the resumed matrix and the uninterrupted one."""


# ---------------------------------------------------------------------------
# fixture (the toy train/exam dirs every agent's CPU smoke test uses)


def _synthetic_exam_dir(root: Path, n_phases: int = N_PHASES, items_per_type: int = 4) -> Path:
    """A *real*, scoreable toy exam directory -- synthetic content, the frozen
    probe shapes (`AGENTS.md`, Probe items; `training/evaluate.py`'s
    `_stories_path` / `_probes_path`).

    `tests/conftest.py`'s `write_exam_dir` builds only a placeholder file,
    because every other agent's test stubs `evaluate_all` and never reads the
    exam directory for real (AGENTS.md's toy-config rule: "no agent's test
    reads the exam directory" -- meaning no test reads *real* exam text). This
    script runs the genuine, unstubbed `training.evaluate.evaluate_all`
    end-to-end as a real Kaggle/A100 session would, so it needs exam files
    that actually parse and score, built from nothing but synthetic text.

    STALE since amendment 2 (no `option_sources`, no story ids); no longer
    used by `build_fixture`. Kept only so an old import does not break.
    """
    stories_dir = root / "stories"
    probes_dir = root / "probes"
    stories_dir.mkdir(parents=True, exist_ok=True)
    probes_dir.mkdir(parents=True, exist_ok=True)
    for k in range(n_phases):
        letter = chr(ord("a") + k)
        stories = [{"story": f"{letter * 3} {letter * 4} {letter * 2} exam story {i}.\n"} for i in range(items_per_type)]
        (stories_dir / f"exam_phase_{k}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in stories) + "\n", encoding="utf-8"
        )

        cloze_rows = []
        for i in range(items_per_type):
            answer = f"{letter * 2}"
            candidates = [answer] + [f"{letter}{d}" for d in "xyzw"]
            cloze_rows.append(
                {
                    "id": f"cloze_{k}_{i}",
                    "phase": k,
                    "text_with_mask": f"the {letter * 3} went to [MASK] school today.",
                    "answer": answer,
                    "candidates": candidates,
                }
            )
        (probes_dir / f"cloze_phase_{k}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in cloze_rows) + "\n", encoding="utf-8"
        )

        cont_rows = []
        for i in range(items_per_type):
            options = [f"{letter * 2} true continuation {i}.", "zz distractor one.", "zz distractor two.", "zz distractor three."]
            cont_rows.append(
                {
                    "id": f"cont_{k}_{i}",
                    "phase": k,
                    "prefix": f"once upon a time in phase {letter}, ",
                    "options": options,
                    "answer_index": 0,
                    "distractor_phases": [(k + 1) % n_phases, (k + 2) % n_phases, (k + 3) % n_phases],
                }
            )
        (probes_dir / f"continuation_phase_{k}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in cont_rows) + "\n", encoding="utf-8"
        )
    return root


def build_fixture(root: Path) -> tuple[Path, Path, Path]:
    """Reuses `tests/conftest.py`'s synthetic train-dir and manifest builders
    -- "one definition, used by every agent's tests" (AGENTS.md) -- including
    the scoreable exam directory, because this script exercises the real,
    unstubbed evaluator end-to-end.

    Until 2026-09-30 this used `_synthetic_exam_dir` below, which predates
    amendment 2 (no `option_sources`) and the per-story manifest (no story
    `id`), so the proof could no longer build its fixture. conftest's
    `write_scoreable_exam_dir` is kept current with the contracts by the test
    suite that depends on it."""
    sys.path.insert(0, str(REPO_ROOT))
    # local import: optional dep on tests/
    from tests.conftest import write_manifest, write_scoreable_exam_dir, write_train_dir

    train_dir = write_train_dir(root / "train")
    exam_dir = write_scoreable_exam_dir(root / "exam")
    manifest = write_manifest(root / "manifest.json", exam_dir)
    return train_dir, exam_dir, manifest


# ---------------------------------------------------------------------------
# matrix inspection (JSON only -- no import from training/ beyond config)


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _phase_row_complete(matrix: dict, phase: int) -> bool:
    for exam_type in EXAM_TYPES:
        block = matrix.get("M", {}).get(exam_type)
        if not isinstance(block, list) or len(block) <= phase:
            return False
        row = block[phase]
        if not isinstance(row, list) or len(row) != N_PHASES or any(v is None for v in row):
            return False
    return True


def _matrix_has_any_incomplete_row(matrix: dict) -> bool:
    for exam_type in EXAM_TYPES:
        for row in matrix.get("M", {}).get(exam_type, []):
            if not isinstance(row, list) or any(v is None for v in row):
                return True
    return False


def wait_for_phase(matrix_path: Path, phase: int, *, timeout: float, poll: float = 0.005) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        matrix = _load_json(matrix_path)
        if matrix is not None and _phase_row_complete(matrix, phase):
            return True
        time.sleep(poll)
    return False


# ---------------------------------------------------------------------------
# subprocess plumbing -- the CLI contract, nothing else


def _train_argv(*, arm: str, seed: int, train_dir: Path, exam_dir: Path, manifest: Path, out: Path, resume: bool) -> list[str]:
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
        "--manifest",
        str(manifest),
        # The fixture manifest's own id: the guard refuses a run whose
        # experiment_id is not the manifest's (2026-09-30), and this fixture is
        # not the main experiment's.
        "--experiment-id",
        str(json.loads(Path(manifest).read_text(encoding="utf-8"))["experiment_id"]),
        "--out",
        str(out),
        "--toy",
        "--cpu",
    ]
    if resume:
        argv.append("--resume")
    return argv


def _run_to_completion(argv: list[str], *, timeout: float) -> None:
    proc = subprocess.run(argv, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise ResumeProofError(
            f"`{' '.join(argv[2:])}` exited {proc.returncode}\n--- stdout (tail) ---\n"
            + "\n".join(proc.stdout.splitlines()[-40:])
            + "\n--- stderr (tail) ---\n"
            + "\n".join(proc.stderr.splitlines()[-40:])
        )


# ---------------------------------------------------------------------------
# the proof


def prove_resume_roundtrip(
    work_root: Path,
    *,
    arm: str = "E",
    seed: int = 0,
    kill_after_phase: int = 2,
    kill_wait_timeout: float = 90.0,
    run_timeout: float = 180.0,
) -> dict[str, Any]:
    """Runs the operational round trip once and returns a small report, or
    raises `ResumeProofError` / `AssertionError` on any disagreement.

    Arm E is used because it needs no shared phase-0 checkpoint (one process,
    seven phases, checkpointed after each) -- the simplest single-run shape to
    kill mid-flight. `--phase0-dir` is left at its default
    (`<out>.parent/_phase0`); the reference, live and restored directories are
    all placed directly under `work_root` so they share one `_phase0/`, which
    is what makes the three runs' random inits identical in the first place.
    """
    work_root = Path(work_root)
    train_dir, exam_dir, manifest = build_fixture(work_root / "fixture")

    def argv_for(out: Path, resume: bool) -> list[str]:
        return _train_argv(arm=arm, seed=seed, train_dir=train_dir, exam_dir=exam_dir, manifest=manifest, out=out, resume=resume)

    # 1. The uninterrupted reference: what a session that never died would produce.
    ref_dir = work_root / "ref"
    _run_to_completion(argv_for(ref_dir, resume=False), timeout=run_timeout)
    ref_matrix = _load_json(ref_dir / "matrix.json")
    if ref_matrix is None or _matrix_has_any_incomplete_row(ref_matrix):
        raise ResumeProofError("reference run's matrix.json is missing or incomplete; cannot be the ground truth")

    # 2. A live session, killed after `kill_after_phase` completes.
    live_dir = work_root / "live"
    live_argv = argv_for(live_dir, resume=False)
    proc = subprocess.Popen(live_argv, cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        reached = wait_for_phase(live_dir / "matrix.json", kill_after_phase, timeout=kill_wait_timeout)
    finally:
        proc.kill()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:  # pragma: no cover - defensive
            pass
    if not reached:
        raise ResumeProofError(
            f"the live run never reached a complete phase-{kill_after_phase} row within "
            f"{kill_wait_timeout}s; lower --kill-after-phase or raise the timeout"
        )
    live_matrix = _load_json(live_dir / "matrix.json")
    if live_matrix is None or not _matrix_has_any_incomplete_row(live_matrix):
        raise ResumeProofError(
            "the killed run's matrix.json is already fully populated -- the kill landed too "
            "late to prove anything; lower --kill-after-phase"
        )

    # 3. The Kaggle-dataset round trip: session 1's folder -> a dataset version -> session 2's local copy.
    staging_dir = work_root / "staging_dataset_version"
    shutil.copytree(live_dir, staging_dir)
    restored_dir = work_root / "restored_session2"
    shutil.copytree(staging_dir, restored_dir)

    # 4. Session 2 resumes from the restored copy, never touching `live_dir` again.
    _run_to_completion(argv_for(restored_dir, resume=True), timeout=run_timeout)
    resumed_matrix = _load_json(restored_dir / "matrix.json")
    if resumed_matrix is None or _matrix_has_any_incomplete_row(resumed_matrix):
        raise ResumeProofError("the resumed run's matrix.json is missing or incomplete after --resume")

    # 5. The assertion the whole proof exists for.
    mismatches: list[tuple[str, int, int, float, float]] = []
    for exam_type in EXAM_TYPES:
        ref_block = ref_matrix["M"][exam_type]
        resumed_block = resumed_matrix["M"][exam_type]
        for i in range(N_PHASES):
            for j in range(N_PHASES):
                a, b = ref_block[i][j], resumed_block[i][j]
                if not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9):
                    mismatches.append((exam_type, i, j, a, b))
    if mismatches:
        raise AssertionError(
            f"resumed matrix disagrees with the uninterrupted reference at "
            f"{mismatches[:5]}{' (+more)' if len(mismatches) > 5 else ''}"
        )

    return {
        "arm": arm,
        "seed": seed,
        "kill_after_phase": kill_after_phase,
        "ref_dir": str(ref_dir),
        "live_dir_killed_partial": str(live_dir),
        "staging_dataset_version": str(staging_dir),
        "restored_session2_dir": str(restored_dir),
        "cells_compared": len(EXAM_TYPES) * N_PHASES * N_PHASES,
        "mismatches": 0,
        "result": "resumed matrix == uninterrupted reference matrix, cell for cell",
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m runbook.resume_roundtrip",
        description="Prove the Kaggle-dataset resume round trip on the toy config, CPU only.",
    )
    p.add_argument("--work-dir", type=Path, default=None, help="default: a fresh temp directory")
    p.add_argument("--arm", default="E", choices=["E"], help="E needs no phase-0 checkpoint; the simplest single run to kill mid-flight")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--kill-after-phase", type=int, default=2)
    p.add_argument("--keep", action="store_true", help="do not delete a freshly created --work-dir on success")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    made_temp = args.work_dir is None
    work_root = args.work_dir or Path(tempfile.mkdtemp(prefix="lifespan_resume_proof_"))
    work_root.mkdir(parents=True, exist_ok=True)
    try:
        report = prove_resume_roundtrip(
            work_root, arm=args.arm, seed=args.seed, kill_after_phase=args.kill_after_phase
        )
    except (ResumeProofError, AssertionError) as exc:
        print(f"RESUME ROUND TRIP FAILED: {exc}")
        return 1
    print("RESUME ROUND TRIP PROVED:")
    for k, v in report.items():
        print(f"  {k}: {v}")
    if made_temp and not args.keep:
        shutil.rmtree(work_root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
