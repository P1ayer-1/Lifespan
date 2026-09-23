"""The leak guard. It runs first, before a model exists, and it fails closed.

`CLAUDE.md`: "The training script refuses to start if any training file's hash
appears in `exams/manifest.json`." `AGENTS.md` adds the manifest shape and the
rule that the exam directory is somewhere the training script cannot read.

What `check()` refuses, every one of them by raising `GuardError`:

- a missing, unparseable, or empty `exams/manifest.json`;
- `--train-dir` and `--exam-dir` being the same directory, or one containing
  the other;
- a training directory that does not exist, or a file under it that cannot be
  read;
- any file under `--train-dir` whose SHA-256 appears in the manifest.

Nothing in this module reads the exam directory. It takes the exam path only to
compare it with the training path; the hashes it compares against come from the
committed manifest. `data.py` is never given the exam path at all.

Never wrap a call to `check()` in `try`/`except` to keep a run going
(`.claude/agents/trainer-core.md`, "What you never do"). A guard that can be
survived is not a guard.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

CHUNK = 1 << 20


class GuardError(RuntimeError):
    """The run must not start. Not to be caught."""


@dataclass(frozen=True)
class GuardReport:
    """What the guard learned, for the run record."""

    train_dir: str
    exam_dir: str
    manifest_path: str
    manifest_sha256: str
    experiment_id: str
    n_train_files: int
    n_manifest_files: int
    #: relative posix path -> sha256, every file under the training dir.
    train_file_hashes: dict[str, str] = field(default_factory=dict)

    @property
    def data_hash(self) -> str:
        """One hash standing for the whole training corpus, for config.json."""
        return _digest_of_pairs(self.train_file_hashes)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            while chunk := fh.read(CHUNK):
                h.update(chunk)
    except OSError as exc:
        raise GuardError(f"cannot read {path}: {exc}") from exc
    return h.hexdigest()


def _digest_of_pairs(mapping: dict[str, str]) -> str:
    h = hashlib.sha256()
    for key in sorted(mapping):
        h.update(key.encode("utf-8"))
        h.update(b"\0")
        h.update(mapping[key].encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def load_manifest(manifest_path: Path) -> dict:
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise GuardError(
            f"exam manifest missing: {manifest_path}. Exams are frozen before the first "
            "training run; without the manifest there is nothing to check training data against."
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GuardError(f"exam manifest unreadable: {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise GuardError(f"exam manifest is not a JSON object: {manifest_path}")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise GuardError(
            f"exam manifest lists no files: {manifest_path}. An empty manifest would let "
            "every training file through."
        )
    for i, entry in enumerate(files):
        if not isinstance(entry, dict) or not entry.get("sha256"):
            raise GuardError(f"exam manifest entry {i} has no sha256: {manifest_path}")
    return manifest


def _check_disjoint_paths(train_dir: Path, exam_dir: Path) -> None:
    if train_dir == exam_dir:
        raise GuardError(f"--train-dir and --exam-dir are the same directory: {train_dir}")
    if train_dir.is_relative_to(exam_dir):
        raise GuardError(f"--train-dir {train_dir} is inside --exam-dir {exam_dir}")
    if exam_dir.is_relative_to(train_dir):
        raise GuardError(f"--exam-dir {exam_dir} is inside --train-dir {train_dir}")


def check(train_dir: Path, exam_dir: Path, manifest_path: Path) -> GuardReport:
    """Hash every training file and refuse the run if anything is wrong.

    Returns the hashes so `runrecord.py` can put them in `config.json`; raises
    `GuardError` otherwise.
    """
    train_dir = Path(train_dir).resolve()
    exam_dir = Path(exam_dir).resolve()
    manifest_path = Path(manifest_path)

    if not train_dir.is_dir():
        raise GuardError(f"--train-dir does not exist or is not a directory: {train_dir}")
    _check_disjoint_paths(train_dir, exam_dir)
    if not exam_dir.is_dir():
        raise GuardError(f"--exam-dir does not exist or is not a directory: {exam_dir}")

    manifest = load_manifest(manifest_path)
    exam_hashes: dict[str, str] = {}
    for entry in manifest["files"]:
        exam_hashes.setdefault(entry["sha256"].lower(), entry.get("path_relative_to_exam_dir", "?"))

    train_hashes: dict[str, str] = {}
    leaks: list[str] = []
    for path in sorted(p for p in train_dir.rglob("*") if p.is_file()):
        rel = path.relative_to(train_dir).as_posix()
        digest = sha256_file(path)
        train_hashes[rel] = digest
        if digest in exam_hashes:
            leaks.append(f"{rel} == exam file {exam_hashes[digest]} (sha256 {digest})")
    if not train_hashes:
        raise GuardError(f"--train-dir has no files: {train_dir}")
    if leaks:
        raise GuardError(
            "exam text is in the training directory; this run would be worthless:\n  "
            + "\n  ".join(leaks)
        )

    return GuardReport(
        train_dir=str(train_dir),
        exam_dir=str(exam_dir),
        manifest_path=str(manifest_path.resolve()),
        manifest_sha256=sha256_file(manifest_path),
        experiment_id=str(manifest.get("experiment_id", "")),
        n_train_files=len(train_hashes),
        n_manifest_files=len(manifest["files"]),
        train_file_hashes=train_hashes,
    )


def main(argv: list[str] | None = None) -> int:
    """`python -m training.guard --train-dir ... --exam-dir ...` -- the same
    check `train.py` runs, usable on its own before booking GPU time."""
    import argparse

    parser = argparse.ArgumentParser(description="Refuse a run whose training data touches the exams.")
    parser.add_argument("--train-dir", required=True, type=Path)
    parser.add_argument("--exam-dir", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=Path("exams/manifest.json"))
    args = parser.parse_args(argv)
    report = check(args.train_dir, args.exam_dir, args.manifest)
    print(
        f"guard ok: {report.n_train_files} training files, "
        f"{report.n_manifest_files} exam files in the manifest, data_hash {report.data_hash[:12]}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
