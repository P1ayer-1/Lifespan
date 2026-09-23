"""The leak guard. Acceptance check 7.

Every case here is a run that must not start: a planted duplicate, a nested
train/exam pair, a missing manifest, an empty one. The last test runs the real
entry point in a subprocess and asserts a non-zero exit with nothing written --
the guard has to fire before a model is constructed, not after.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import write_exam_dir, write_manifest, write_train_dir
from training.guard import GuardError, check, load_manifest, sha256_file

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_a_clean_pair_passes_and_reports_the_hashes(toy_dirs):
    report = check(toy_dirs["train"], toy_dirs["exam"], toy_dirs["manifest"])
    assert report.n_train_files == 7 + 7  # story files plus replay buffers
    assert len(report.train_file_hashes) == report.n_train_files
    assert len(report.data_hash) == 64
    assert report.experiment_id == "toy"


def test_the_data_hash_changes_when_the_data_does(toy_dirs):
    before = check(toy_dirs["train"], toy_dirs["exam"], toy_dirs["manifest"]).data_hash
    path = toy_dirs["train"] / "train_phase_3.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    after = check(toy_dirs["train"], toy_dirs["exam"], toy_dirs["manifest"]).data_hash
    assert before != after


def test_a_planted_duplicate_is_refused(toy_dirs):
    """Acceptance check 7: a training file whose sha256 is in the manifest."""
    planted = toy_dirs["train"] / "leaked.jsonl"
    planted.write_text('{"story": "this came from the exam set"}\n', encoding="utf-8")
    write_manifest(toy_dirs["manifest"], toy_dirs["exam"], extra_hashes=[sha256_file(planted)])
    with pytest.raises(GuardError, match="exam text is in the training directory"):
        check(toy_dirs["train"], toy_dirs["exam"], toy_dirs["manifest"])


def test_a_duplicate_nested_deep_in_the_train_dir_is_refused(toy_dirs):
    nested = toy_dirs["train"] / "extra" / "deep" / "leak.jsonl"
    nested.parent.mkdir(parents=True)
    nested.write_text("leak\n", encoding="utf-8")
    write_manifest(toy_dirs["manifest"], toy_dirs["exam"], extra_hashes=[sha256_file(nested)])
    with pytest.raises(GuardError, match="leak.jsonl"):
        check(toy_dirs["train"], toy_dirs["exam"], toy_dirs["manifest"])


def test_the_same_directory_for_both_is_refused(toy_dirs):
    with pytest.raises(GuardError, match="same directory"):
        check(toy_dirs["train"], toy_dirs["train"], toy_dirs["manifest"])


def test_an_exam_dir_inside_the_train_dir_is_refused(tmp_path):
    train = write_train_dir(tmp_path / "train")
    exam = write_exam_dir(train / "exams")
    manifest = write_manifest(tmp_path / "m.json", exam)
    with pytest.raises(GuardError, match="is inside"):
        check(train, exam, manifest)


def test_a_train_dir_inside_the_exam_dir_is_refused(tmp_path):
    exam = write_exam_dir(tmp_path / "exam")
    train = write_train_dir(exam / "train")
    manifest = write_manifest(tmp_path / "m.json", tmp_path / "exam" / "stories")
    with pytest.raises(GuardError, match="is inside"):
        check(train, exam, manifest)


def test_a_missing_manifest_is_refused(toy_dirs):
    toy_dirs["manifest"].unlink()
    with pytest.raises(GuardError, match="manifest missing"):
        check(toy_dirs["train"], toy_dirs["exam"], toy_dirs["manifest"])


def test_an_empty_manifest_is_refused(toy_dirs):
    toy_dirs["manifest"].write_text(json.dumps({"experiment_id": "x", "files": []}), encoding="utf-8")
    with pytest.raises(GuardError, match="lists no files"):
        check(toy_dirs["train"], toy_dirs["exam"], toy_dirs["manifest"])


def test_a_manifest_entry_without_a_hash_is_refused(toy_dirs):
    toy_dirs["manifest"].write_text(
        json.dumps({"files": [{"path_relative_to_exam_dir": "a", "phase": 0}]}), encoding="utf-8"
    )
    with pytest.raises(GuardError, match="no sha256"):
        check(toy_dirs["train"], toy_dirs["exam"], toy_dirs["manifest"])


def test_unparseable_manifest_is_refused(toy_dirs):
    toy_dirs["manifest"].write_text("{not json", encoding="utf-8")
    with pytest.raises(GuardError, match="unreadable"):
        load_manifest(toy_dirs["manifest"])


def test_a_missing_train_dir_is_refused(toy_dirs):
    with pytest.raises(GuardError, match="--train-dir does not exist"):
        check(toy_dirs["tmp"] / "nope", toy_dirs["exam"], toy_dirs["manifest"])


def test_an_empty_train_dir_is_refused(toy_dirs):
    empty = toy_dirs["tmp"] / "empty"
    empty.mkdir()
    with pytest.raises(GuardError, match="has no files"):
        check(empty, toy_dirs["exam"], toy_dirs["manifest"])


# ---------------------------------------------------------------------------
# the real entry point


def run_train(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "training.train", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )


@pytest.mark.parametrize(
    "break_it,expect",
    [
        ("duplicate", "exam text is in the training directory"),
        ("nested", "is inside"),
        ("no_manifest", "manifest missing"),
    ],
)
def test_the_entry_point_exits_non_zero_before_building_a_model(toy_dirs, break_it, expect):
    """Acceptance check 7 end to end: non-zero exit, the guard's message, and
    an output folder that was never created -- nothing ran."""
    train, exam = toy_dirs["train"], toy_dirs["exam"]
    if break_it == "duplicate":
        planted = train / "leaked.jsonl"
        planted.write_text("exam text\n", encoding="utf-8")
        write_manifest(toy_dirs["manifest"], exam, extra_hashes=[sha256_file(planted)])
    elif break_it == "nested":
        exam = write_exam_dir(train / "exams")
    elif break_it == "no_manifest":
        toy_dirs["manifest"].unlink()

    out = toy_dirs["results"] / "guard_run"
    proc = run_train(
        [
            "--arm", "A", "--seed", "0",
            "--train-dir", str(train),
            "--exam-dir", str(exam),
            "--out", str(out),
            "--manifest", str(toy_dirs["manifest"]),
            "--phase0-dir", str(toy_dirs["shared"]),
            "--toy", "--cpu",
        ]
    )
    assert proc.returncode != 0, proc.stdout
    assert "GuardError" in proc.stderr
    assert expect in proc.stderr
    assert not out.exists(), "the run folder was created, so the guard did not run first"
    assert not toy_dirs["shared"].exists(), "a checkpoint dir was created before the guard fired"
