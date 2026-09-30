"""The leak guard. It runs first, before a model exists, and it fails closed.

`CLAUDE.md`: "The training script refuses to start if any training file's hash
appears in `exams/manifest.json`." `AGENTS.md` adds the manifest shape and the
rule that the exam directory is somewhere the training script cannot read.

What `check()` refuses, every one of them by raising `GuardError`:

- a missing, unparseable, or empty `exams/manifest.json`, or one without the
  per-story hashes, the generator model or the experiment id (below);
- `--train-dir` and `--exam-dir` being the same directory, or one containing
  the other;
- a training directory that does not exist, or a file under it that cannot be
  read (a `.jsonl` / `.json` / `.txt` file that is not UTF-8 counts);
- any file under `--train-dir` whose SHA-256 appears in the manifest;
- any *line* of any training file that carries an exam story: any string in it
  (its `prompt_hash`, a replay file's `prompt_hashes`, ...) equals an exam
  `prompt_hash`, or the normalised hash of any string in it (the `story` field
  included, under whatever key) equals an exam `story_sha256`. This does not
  depend on file bytes, so a CRLF copy, a re-serialised line, or exam lines
  mixed into a file of genuine training lines are all refused (leakage audit
  2026-09-25, blocking finding 3). A plain-text file that *is* one exam story
  is caught by the same hash over the whole file;
- training story lines without a `model`, with more than one `model`, or whose
  one model differs from the manifest's `generator_model` (the exam
  generator's; audit 2026-09-25, blocking finding 2, training side);
- a manifest `experiment_id` different from the one the caller expects, and a
  caller that expects none: `experiment_id` is required, so a run that does
  not name its experiment is refused rather than checked against nothing
  (audit note 2026-09-25, "recorded but never checked"; wired 2026-09-30).

Manifest fields required beyond the 2026-09-21 frozen shape -- additions,
nothing renamed (`REQUIRED_MANIFEST_ADDITIONS`; the lead freezes them):

    experiment_id:   non-empty string (frozen key; was recorded, never checked)
    generator_model: the exam stories' generator model id
    stories: [{story_id, prompt_hash, story_sha256, phase}]   one per exam story

`story_sha256` is `story_sha256(text)`: SHA-256 of the UTF-8 bytes of
`normalise_story(text)`. The normalisation is fixed here, and every producer
and consumer (the probe builder, the near-duplicate check, the evaluator's
option_sources check) imports it from here:

    1. Unicode NFC;
    2. every run of whitespace (space, tab, CR, LF, CRLF, NBSP, ...) becomes one
       ASCII space;
    3. leading and trailing whitespace is removed.

Case and punctuation are kept: stories that differ in either are different
stories to the hash, and are the near-duplicate check's business
(`training/neardup.py`), which runs at freeze time and catches edited copies.

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
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

CHUNK = 1 << 20

#: Top-level manifest keys added 2026-09-30 (guard rewrite after the leakage
#: audit). The frozen 2026-09-21 keys are all still read under their names.
REQUIRED_MANIFEST_ADDITIONS = ("generator_model", "stories")

#: Required fields of one `manifest["stories"]` entry. `story_id` is written by
#: `manifest_story_entry` too, but a reader does not need it.
MANIFEST_STORY_FIELDS = ("prompt_hash", "story_sha256", "phase")

#: Files that must decode as UTF-8. Any other file is still whole-file hashed,
#: and line-checked too if it happens to decode.
TEXT_SUFFIXES = (".jsonl", ".json", ".txt")

#: How many leak hits an error message lists before summarising.
MAX_LISTED = 50

_HEX = frozenset("0123456789abcdef")


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
    #: The one generator model of the training corpus, equal to the manifest's.
    generator_model: str = ""
    #: Non-empty training lines checked against the manifest's stories.
    n_train_lines_checked: int = 0
    n_manifest_stories: int = 0
    #: normalised story sha256 -> phase for every exam story in the manifest,
    #: for the evaluator's option_sources check. Hashes only; not for config.json.
    exam_story_phases: dict[str, int] = field(default_factory=dict, repr=False)

    @property
    def data_hash(self) -> str:
        """One hash standing for the whole training corpus, for config.json."""
        return _digest_of_pairs(self.train_file_hashes)


# --------------------------------------------------------------------------- #
# the story hash -- one definition for every producer and consumer
# --------------------------------------------------------------------------- #


def normalise_story(text: str) -> str:
    """NFC, every whitespace run to one space, stripped (module docstring).
    Line endings and indentation therefore never change a story's hash."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def story_sha256(text: str) -> str:
    """SHA-256 hex of the normalised story -- the manifest's `story_sha256`."""
    return hashlib.sha256(normalise_story(text).encode("utf-8")).hexdigest()


def manifest_story_entry(row: dict) -> dict:
    """The manifest `stories` entry for one exam story line.

    `row` is an exam story line (the training-line shape: prompt_hash, phase,
    story, model, ...). `story_id` is the row's `id` if it has one, else its
    `prompt_hash`. Hashes and ids only, never text.
    """
    prompt_hash = row.get("prompt_hash") or row.get("id")
    if not prompt_hash:
        raise GuardError("an exam story line has neither prompt_hash nor id")
    return {
        "story_id": str(row.get("id") or prompt_hash),
        "prompt_hash": str(prompt_hash),
        "story_sha256": story_sha256(row["story"]),
        "phase": int(row["phase"]),
    }


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


# --------------------------------------------------------------------------- #
# the manifest
# --------------------------------------------------------------------------- #


def load_manifest(manifest_path: Path) -> dict:
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise GuardError(
            f"exam manifest missing: {manifest_path}. Exams are frozen before the first "
            "training run; without the manifest there is nothing to check training data against."
        )
    try:
        manifest = json.loads(manifest_path.read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
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

    experiment_id = manifest.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        raise GuardError(f"exam manifest has no experiment_id: {manifest_path}")
    model = manifest.get("generator_model")
    if not isinstance(model, str) or not model.strip():
        raise GuardError(
            f"exam manifest has no generator_model: {manifest_path}. Without the exam "
            "generator's model id a training corpus from a different model cannot be refused."
        )
    stories = manifest.get("stories")
    if not isinstance(stories, list) or not stories:
        raise GuardError(
            f"exam manifest lists no stories: {manifest_path}. Per-story prompt_hash and "
            "story_sha256 are required: a whole-file hash alone misses a CRLF copy, or exam "
            "lines mixed into a training file."
        )
    for i, entry in enumerate(stories):
        if not isinstance(entry, dict):
            raise GuardError(f"exam manifest story {i} is not an object: {manifest_path}")
        missing = [f for f in MANIFEST_STORY_FIELDS if f not in entry]
        if missing:
            raise GuardError(
                f"exam manifest story {i} is missing {', '.join(missing)}: {manifest_path}"
            )
        sha = entry["story_sha256"]
        if not isinstance(sha, str) or len(sha) != 64 or not set(sha.lower()) <= _HEX:
            raise GuardError(f"exam manifest story {i} has a malformed story_sha256: {manifest_path}")
        if not isinstance(entry["prompt_hash"], str) or not entry["prompt_hash"]:
            raise GuardError(f"exam manifest story {i} has an empty prompt_hash: {manifest_path}")
        if isinstance(entry["phase"], bool) or not isinstance(entry["phase"], int):
            raise GuardError(f"exam manifest story {i} has a non-integer phase: {manifest_path}")
    return manifest


def manifest_story_phases(manifest: dict) -> dict[str, int]:
    """normalised story sha256 -> phase, every exam story in a loaded manifest."""
    return {e["story_sha256"].lower(): int(e["phase"]) for e in manifest["stories"]}


# --------------------------------------------------------------------------- #
# line checks
# --------------------------------------------------------------------------- #


def _iter_strings(value: Any) -> Iterator[tuple[str | None, str]]:
    """Every string inside a JSON value (keys excluded), with the key it sits
    under -- None for the top level."""
    stack: list[tuple[str | None, Any]] = [(None, value)]
    while stack:
        key, v = stack.pop()
        if isinstance(v, str):
            yield key, v
        elif isinstance(v, dict):
            stack.extend((str(k), x) for k, x in v.items())
        elif isinstance(v, list):
            stack.extend((key, x) for x in v)


class _LineChecker:
    """Checks training text, line by line, against the manifest's stories, and
    collects the generator model of every story line."""

    def __init__(self, manifest: dict) -> None:
        self.prompt_hashes = {str(e["prompt_hash"]) for e in manifest["stories"]}
        self.story_hashes = {e["story_sha256"].lower() for e in manifest["stories"]}
        self.models: dict[str, list[str]] = {}
        self.n_lines = 0
        self.leaks: list[str] = []
        self.n_leaks = 0

    def _hit(self, where: str, why: str) -> None:
        self.n_leaks += 1
        if len(self.leaks) < MAX_LISTED:
            self.leaks.append(f"{where}: {why}")

    def check_value(self, where: str, value: Any) -> None:
        for key, text in _iter_strings(value):
            if text in self.prompt_hashes:
                self._hit(where, f"exam prompt_hash {text} (under key {key!r})")
                continue
            digest = story_sha256(text)
            if digest in self.story_hashes:
                self._hit(where, f"an exam story (story_sha256 {digest}, under key {key!r})")

    def check_text(self, rel: str, text: str) -> None:
        # The whole file as one story: a plain-text copy of a single exam story.
        whole = story_sha256(text)
        if whole in self.story_hashes:
            self._hit(rel, f"the whole file is an exam story (story_sha256 {whole})")
        # A JSON document (a replay buffer): every string in it.
        if rel.lower().endswith(".json"):
            try:
                self.check_value(rel, json.loads(text))
            except json.JSONDecodeError:
                pass
        # Every line, parsed as JSON where it parses and as text where not.
        for n, raw in enumerate(text.splitlines(), start=1):
            line = raw.strip()
            if not line:
                continue
            self.n_lines += 1
            where = f"{rel}:{n}"
            try:
                obj: Any = json.loads(line)
            except json.JSONDecodeError:
                obj = line
            self.check_value(where, obj)
            if isinstance(obj, dict) and "story" in obj:
                model = obj.get("model")
                key = model if isinstance(model, str) and model.strip() else "<missing>"
                self.models.setdefault(key, []).append(where)

    def leak_lines(self) -> list[str]:
        if self.n_leaks > len(self.leaks):
            return self.leaks + [f"... and {self.n_leaks - len(self.leaks)} more"]
        return list(self.leaks)


def _read_text(path: Path, rel: str) -> str | None:
    """The file as text, or None for a binary file without a text suffix.

    Bytes are read and decoded (never a text-mode read), so a CR survives to
    the normaliser and the result is the same on Windows and on Kaggle. A
    `.jsonl` / `.json` / `.txt` file that is not UTF-8 is refused, not skipped:
    it is the kind of file data.py reads, and an unreadable one is unchecked.
    """
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise GuardError(f"cannot read {path}: {exc}") from exc
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        if path.suffix.lower() in TEXT_SUFFIXES:
            raise GuardError(f"training file {rel} is not UTF-8 and cannot be checked: {exc}") from exc
        return None


def _check_generator_model(models: dict[str, list[str]], exam_model: str, train_dir: Path) -> str:
    """One model across every training story line, and it is the exam's."""
    if not models:
        raise GuardError(
            f"no training story lines found under {train_dir}; the generator model cannot "
            "be checked against the exam's"
        )
    if "<missing>" in models:
        where = models["<missing>"][:3]
        raise GuardError(
            f"{len(models['<missing>'])} training story lines have no `model` field, "
            f"e.g. {', '.join(where)}; the generator model cannot be checked"
        )
    if len(models) > 1:
        summary = ", ".join(f"{m!r} ({len(w)} lines, e.g. {w[0]})" for m, w in sorted(models.items()))
        raise GuardError(
            f"the training corpus has {len(models)} different generator models: {summary}. "
            "One model generates every story, train and exam."
        )
    (model,) = models
    if model != exam_model:
        raise GuardError(
            f"generator model mismatch: the training stories are from {model!r} but the exam "
            f"manifest's generator_model is {exam_model!r}. A model change between train and "
            "exam is a confound the forgetting curve cannot separate from forgetting."
        )
    return model


def _check_disjoint_paths(train_dir: Path, exam_dir: Path) -> None:
    if train_dir == exam_dir:
        raise GuardError(f"--train-dir and --exam-dir are the same directory: {train_dir}")
    if train_dir.is_relative_to(exam_dir):
        raise GuardError(f"--train-dir {train_dir} is inside --exam-dir {exam_dir}")
    if exam_dir.is_relative_to(train_dir):
        raise GuardError(f"--exam-dir {exam_dir} is inside --train-dir {train_dir}")


# --------------------------------------------------------------------------- #
# the check
# --------------------------------------------------------------------------- #


def check(
    train_dir: Path,
    exam_dir: Path,
    manifest_path: Path,
    *,
    experiment_id: str | None = None,
) -> GuardReport:
    """Hash every training file, check every training line against the
    manifest's stories and generator model, and refuse the run on any hit.

    `experiment_id` is required and must equal the manifest's; `None` or a
    blank string is refused (before 2026-09-30 it skipped the check, which the
    leakage re-audit called "recorded but never checked"). Returns the hashes
    so `runrecord.py` can put them in `config.json`; raises `GuardError`
    otherwise.
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
    if experiment_id is None or not str(experiment_id).strip():
        raise GuardError(
            f"no experiment_id given, but the manifest {manifest_path} is for "
            f"{manifest['experiment_id']!r}. A run must name the experiment it belongs to "
            "(training.config.EXPERIMENT_ID, or train.py --experiment-id); an unchecked id "
            "is how a run gets scored against another experiment's exams."
        )
    if experiment_id != manifest["experiment_id"]:
        raise GuardError(
            f"experiment_id mismatch: the run expects {experiment_id!r} but the manifest "
            f"{manifest_path} is for {manifest['experiment_id']!r}"
        )

    exam_hashes: dict[str, str] = {}
    for entry in manifest["files"]:
        exam_hashes.setdefault(entry["sha256"].lower(), entry.get("path_relative_to_exam_dir", "?"))

    lines = _LineChecker(manifest)
    train_hashes: dict[str, str] = {}
    leaks: list[str] = []
    for path in sorted(p for p in train_dir.rglob("*") if p.is_file()):
        rel = path.relative_to(train_dir).as_posix()
        digest = sha256_file(path)
        train_hashes[rel] = digest
        if digest in exam_hashes:
            leaks.append(f"{rel} == exam file {exam_hashes[digest]} (sha256 {digest})")
        text = _read_text(path, rel)
        if text is not None:
            lines.check_text(rel, text)
    if not train_hashes:
        raise GuardError(f"--train-dir has no files: {train_dir}")
    leaks.extend(lines.leak_lines())
    if leaks:
        raise GuardError(
            "exam text is in the training directory; this run would be worthless:\n  "
            + "\n  ".join(leaks)
        )

    generator_model = _check_generator_model(lines.models, manifest["generator_model"], train_dir)

    return GuardReport(
        train_dir=str(train_dir),
        exam_dir=str(exam_dir),
        manifest_path=str(manifest_path.resolve()),
        manifest_sha256=sha256_file(manifest_path),
        experiment_id=str(manifest["experiment_id"]),
        n_train_files=len(train_hashes),
        n_manifest_files=len(manifest["files"]),
        train_file_hashes=train_hashes,
        generator_model=generator_model,
        n_train_lines_checked=lines.n_lines,
        n_manifest_stories=len(manifest["stories"]),
        exam_story_phases=manifest_story_phases(manifest),
    )


def main(argv: list[str] | None = None) -> int:
    """`python -m training.guard --train-dir ... --exam-dir ...` -- the same
    check `train.py` runs, usable on its own before booking GPU time."""
    import argparse

    parser = argparse.ArgumentParser(description="Refuse a run whose training data touches the exams.")
    parser.add_argument("--train-dir", required=True, type=Path)
    parser.add_argument("--exam-dir", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=Path("exams/manifest.json"))
    from training.config import EXPERIMENT_ID

    parser.add_argument(
        "--experiment-id",
        default=EXPERIMENT_ID,
        help=(
            "refuse unless the manifest's experiment_id is this "
            f"(default: training.config.EXPERIMENT_ID, {EXPERIMENT_ID!r}, as train.py)"
        ),
    )
    args = parser.parse_args(argv)
    report = check(args.train_dir, args.exam_dir, args.manifest, experiment_id=args.experiment_id)
    print(
        f"guard ok: {report.n_train_files} training files, {report.n_train_lines_checked} lines "
        f"checked against {report.n_manifest_stories} exam stories and "
        f"{report.n_manifest_files} exam files, generator {report.generator_model!r}, "
        f"data_hash {report.data_hash[:12]}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
