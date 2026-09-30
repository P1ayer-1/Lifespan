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
from training.guard import GuardError, load_manifest, sha256_file
from training.guard import check as guard_check

REPO_ROOT = Path(__file__).resolve().parent.parent


def check(train_dir, exam_dir, manifest, *, experiment_id="toy"):
    """`guard.check` with the fixture manifest's experiment id. The id is
    required since 2026-09-30; the tests of that call `guard_check` directly."""
    return guard_check(train_dir, exam_dir, manifest, experiment_id=experiment_id)


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
            "--experiment-id", "toy",
            "--toy", "--cpu",
        ]
    )
    assert proc.returncode != 0, proc.stdout
    assert "GuardError" in proc.stderr
    assert expect in proc.stderr
    assert not out.exists(), "the run folder was created, so the guard did not run first"
    assert not toy_dirs["shared"].exists(), "a checkpoint dir was created before the guard fired"


# ---------------------------------------------------------------------------
# per-story checks (leakage audit 2026-09-25, blocking finding 3) -- the
# audit's synthetic attacks, each on synthetic exam lines invented here.

from tests.conftest import FIXTURE_MODEL  # noqa: E402
from training.guard import (  # noqa: E402
    main as guard_main,
    manifest_story_entry,
    normalise_story,
    story_sha256,
)


def exam_rows(n: int = 3, phase: int = 2, model: str = FIXTURE_MODEL) -> list[dict]:
    """Synthetic exam story lines in the training-line shape. Invented text."""
    return [
        {
            "prompt_hash": f"exam{phase}{i:012d}",
            "phase": phase,
            "tier": f"tier_{phase}",
            "story": f"Exam fixture {i}.\nThe zebra counted {i} quiet lanterns.\n\nThen it slept.\n",
            "model": model,
            "timestamp": "2026-09-30T00:00:00Z",
        }
        for i in range(n)
    ]


def lines_of(rows: list[dict], newline: str = "\n") -> str:
    return newline.join(json.dumps(r) for r in rows) + newline


@pytest.fixture
def guarded(toy_dirs):
    """toy_dirs whose manifest carries three synthetic exam stories."""
    rows = exam_rows()
    write_manifest(
        toy_dirs["manifest"], toy_dirs["exam"], extra_stories=[manifest_story_entry(r) for r in rows]
    )
    return {**toy_dirs, "rows": rows}


def test_the_story_normalisation_ignores_line_endings_and_whitespace_but_not_case():
    base = "One line.\nTwo  line.\n\n Three."
    assert normalise_story(base) == "One line. Two line. Three."
    assert story_sha256(base) == story_sha256(base.replace("\n", "\r\n"))
    assert story_sha256(base) == story_sha256("\t" + base.replace(" ", " ") + "  \n")
    assert story_sha256(base) == story_sha256(base.replace("\n", "\r"))
    # NFC: a decomposed e-acute hashes as the composed one
    assert story_sha256("café") == story_sha256("café")
    assert story_sha256(base) != story_sha256(base.lower())


def test_a_clean_corpus_with_per_story_manifest_passes(guarded):
    report = check(guarded["train"], guarded["exam"], guarded["manifest"])
    assert report.generator_model == FIXTURE_MODEL
    assert report.n_train_lines_checked == 7 * 20 + 7  # story lines + one line per replay file
    assert report.n_manifest_stories == 7 + 3  # one placeholder per phase + three synthetic
    assert story_sha256(guarded["rows"][0]["story"]) in report.exam_story_phases


def test_a_crlf_copy_of_the_exam_lines_is_refused(guarded):
    """Audit attack 1: same lines, CRLF endings -- a different file sha256."""
    leaked = guarded["train"] / "extra_phase_2.jsonl"
    leaked.write_bytes(lines_of(guarded["rows"], "\r\n").encode("utf-8"))
    with pytest.raises(GuardError, match="exam prompt_hash"):
        check(guarded["train"], guarded["exam"], guarded["manifest"])


def test_exam_lines_plus_one_train_line_are_refused(guarded):
    """Audit attack 2: all exam lines plus one genuine training line."""
    path = guarded["train"] / "train_phase_2.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + lines_of(guarded["rows"]), encoding="utf-8")
    with pytest.raises(GuardError, match="train_phase_2.jsonl:21"):
        check(guarded["train"], guarded["exam"], guarded["manifest"])


def test_an_exam_story_under_a_new_prompt_hash_and_new_whitespace_is_refused(guarded):
    row = dict(guarded["rows"][1])
    row["prompt_hash"] = "0123456789abcdef"
    row["story"] = "  " + row["story"].replace("\n", "\r\n   ") + "\t"
    path = guarded["train"] / "train_phase_2.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(GuardError, match="an exam story"):
        check(guarded["train"], guarded["exam"], guarded["manifest"])


def test_an_exam_story_under_another_key_is_refused(guarded):
    row = {"prompt_hash": "feedfacefeedface", "phase": 2, "text": guarded["rows"][0]["story"]}
    (guarded["train"] / "notes.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(GuardError, match="under key 'text'"):
        check(guarded["train"], guarded["exam"], guarded["manifest"])


def test_an_exam_prompt_hash_with_a_different_story_is_refused(guarded):
    row = dict(guarded["rows"][2], story="an entirely different story")
    path = guarded["train"] / "train_phase_2.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(GuardError, match="exam prompt_hash"):
        check(guarded["train"], guarded["exam"], guarded["manifest"])


def test_a_replay_buffer_naming_an_exam_prompt_hash_is_refused(guarded):
    replay = guarded["train"] / "replay" / "phase_2.json"
    buf = json.loads(replay.read_text(encoding="utf-8"))
    buf["prompt_hashes"].append(guarded["rows"][0]["prompt_hash"])
    replay.write_text(json.dumps(buf), encoding="utf-8")
    with pytest.raises(GuardError, match="replay/phase_2.json"):
        check(guarded["train"], guarded["exam"], guarded["manifest"])


def test_a_plain_text_copy_of_one_exam_story_is_refused(guarded):
    story = guarded["rows"][0]["story"].replace("\n", "\r\n")
    (guarded["train"] / "story.txt").write_bytes(story.encode("utf-8"))
    with pytest.raises(GuardError, match="whole file is an exam story"):
        check(guarded["train"], guarded["exam"], guarded["manifest"])


def test_a_training_file_that_is_not_utf8_is_refused(guarded):
    (guarded["train"] / "bad.jsonl").write_bytes(b'{"story": "\xff\xfe"}\n')
    with pytest.raises(GuardError, match="not UTF-8"):
        check(guarded["train"], guarded["exam"], guarded["manifest"])


def test_the_line_check_does_not_depend_on_the_whole_file_hash(guarded):
    """The manifest's `files` never mention the leaked file: only the per-story
    entries can catch it."""
    leaked = guarded["train"] / "leak.jsonl"
    leaked.write_text(lines_of(guarded["rows"][:1]), encoding="utf-8")
    manifest = json.loads(guarded["manifest"].read_text(encoding="utf-8"))
    assert sha256_file(leaked) not in {f["sha256"] for f in manifest["files"]}
    with pytest.raises(GuardError, match="leak.jsonl:1"):
        check(guarded["train"], guarded["exam"], guarded["manifest"])


# ---------------------------------------------------------------------------
# the manifest's new required fields


@pytest.mark.parametrize(
    "key,value,expect",
    [
        ("stories", None, "lists no stories"),
        ("stories", [], "lists no stories"),
        ("stories", [{"prompt_hash": "x", "phase": 0}], "missing story_sha256"),
        ("stories", [{"prompt_hash": "x", "story_sha256": "zz", "phase": 0}], "malformed story_sha256"),
        ("stories", [{"prompt_hash": "", "story_sha256": "a" * 64, "phase": 0}], "empty prompt_hash"),
        ("stories", [{"prompt_hash": "x", "story_sha256": "a" * 64, "phase": "0"}], "non-integer phase"),
        ("generator_model", None, "no generator_model"),
        ("generator_model", "", "no generator_model"),
        ("experiment_id", None, "no experiment_id"),
        ("experiment_id", " ", "no experiment_id"),
    ],
)
def test_a_manifest_without_the_new_fields_is_refused(toy_dirs, key, value, expect):
    manifest = json.loads(toy_dirs["manifest"].read_text(encoding="utf-8"))
    if value is None:
        del manifest[key]
    else:
        manifest[key] = value
    toy_dirs["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(GuardError, match=expect):
        check(toy_dirs["train"], toy_dirs["exam"], toy_dirs["manifest"])


# ---------------------------------------------------------------------------
# generator model (audit 2026-09-25, blocking finding 2, training side)


def test_a_training_corpus_from_another_model_than_the_exams_is_refused(toy_dirs):
    write_manifest(toy_dirs["manifest"], toy_dirs["exam"], generator_model="exam-model-v2")
    with pytest.raises(GuardError, match="generator model mismatch"):
        check(toy_dirs["train"], toy_dirs["exam"], toy_dirs["manifest"])


def test_a_mixed_model_training_corpus_is_refused_by_the_guard(tmp_path):
    train = write_train_dir(tmp_path / "train", model_by_phase={3: "another-model"})
    exam = write_exam_dir(tmp_path / "exam")
    manifest = write_manifest(tmp_path / "m.json", exam)
    with pytest.raises(GuardError, match="2 different generator models"):
        check(train, exam, manifest)


def test_a_story_line_without_a_model_is_refused(toy_dirs):
    path = toy_dirs["train"] / "train_phase_0.jsonl"
    row = {"prompt_hash": "abababababababab", "phase": 0, "story": "a story with no model"}
    path.write_text(path.read_text(encoding="utf-8") + json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(GuardError, match="no `model` field"):
        check(toy_dirs["train"], toy_dirs["exam"], toy_dirs["manifest"])


def test_a_train_dir_with_no_story_lines_is_refused(tmp_path):
    train = tmp_path / "train"
    train.mkdir()
    (train / "readme.txt").write_text("nothing here\n", encoding="utf-8")
    exam = write_exam_dir(tmp_path / "exam")
    with pytest.raises(GuardError, match="no training story lines"):
        check(train, exam, write_manifest(tmp_path / "m.json", exam))


# ---------------------------------------------------------------------------
# experiment_id (audit note 2026-09-25: recorded but never checked)


def test_an_experiment_id_mismatch_is_refused_and_a_match_passes(toy_dirs):
    assert check(toy_dirs["train"], toy_dirs["exam"], toy_dirs["manifest"], experiment_id="toy")
    with pytest.raises(GuardError, match="experiment_id mismatch"):
        check(toy_dirs["train"], toy_dirs["exam"], toy_dirs["manifest"], experiment_id="pre-pilot")


def test_the_cli_checks_the_experiment_id(toy_dirs, capsys):
    args = ["--train-dir", str(toy_dirs["train"]), "--exam-dir", str(toy_dirs["exam"]),
            "--manifest", str(toy_dirs["manifest"])]
    assert guard_main(args + ["--experiment-id", "toy"]) == 0
    assert "guard ok" in capsys.readouterr().out
    with pytest.raises(GuardError, match="experiment_id mismatch"):
        guard_main(args + ["--experiment-id", "other"])


@pytest.mark.parametrize("missing", [None, "", "   "])
def test_a_missing_experiment_id_is_refused_not_skipped(toy_dirs, missing):
    """The manifest always has an experiment_id (load_manifest requires one),
    so a caller that names none must be refused: before 2026-09-30 `None`
    skipped the comparison, which is the re-audit's "recorded but never
    checked" in code."""
    with pytest.raises(GuardError, match="no experiment_id given.*'toy'"):
        guard_check(toy_dirs["train"], toy_dirs["exam"], toy_dirs["manifest"], experiment_id=missing)
    with pytest.raises(GuardError, match="no experiment_id given"):
        guard_check(toy_dirs["train"], toy_dirs["exam"], toy_dirs["manifest"])


def test_the_cli_defaults_to_the_run_configs_experiment_id(toy_dirs):
    """No --experiment-id means training.config.EXPERIMENT_ID, exactly as
    train.py: the fixture manifest is for 'toy', so the default is refused."""
    from training.config import EXPERIMENT_ID

    args = ["--train-dir", str(toy_dirs["train"]), "--exam-dir", str(toy_dirs["exam"]),
            "--manifest", str(toy_dirs["manifest"])]
    with pytest.raises(GuardError, match="experiment_id mismatch: the run expects " + repr(EXPERIMENT_ID)):
        guard_main(args)
    write_manifest(toy_dirs["manifest"], toy_dirs["exam"], experiment_id=EXPERIMENT_ID)
    assert guard_main(args) == 0


@pytest.mark.parametrize("sep", [" ", " ", "\x85", "\x0b", "\x0c", "\x1e"])
def test_an_exam_line_with_a_unicode_line_separator_in_a_side_field_is_refused(guarded, sep):
    """Audit re-run 3: str.splitlines() broke the line on U+2028/U+2029/NEL
    etc., so the guard saw fragments while data.py trained on the whole line."""
    row = dict(guarded["rows"][0], split="train", note=f"a{sep}b")
    path = guarded["train"] / "train_phase_2.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(GuardError, match="exam prompt_hash"):
        check(guarded["train"], guarded["exam"], guarded["manifest"])


def test_an_exam_story_containing_a_paragraph_separator_under_a_new_hash_is_refused(guarded):
    row = dict(guarded["rows"][1], split="train", prompt_hash="fedcba9876543210")
    row["story"] = row["story"].replace("\n\n", " ")
    path = guarded["train"] / "train_phase_2.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(GuardError, match="an exam story"):
        check(guarded["train"], guarded["exam"], guarded["manifest"])


def test_a_jsonl_line_that_does_not_parse_is_refused(guarded):
    path = guarded["train"] / "train_phase_2.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + '{"story": "half a line\n', encoding="utf-8")
    with pytest.raises(GuardError, match="does not parse"):
        check(guarded["train"], guarded["exam"], guarded["manifest"])


def test_loader_lines_matches_how_the_data_loader_reads_a_file(tmp_path):
    from training.guard import loader_lines

    text = "a b\r\nc\x85d\re\nf g\x0bh\n"
    p = tmp_path / "f.jsonl"
    p.write_bytes(text.encode("utf-8"))
    with p.open("r", encoding="utf-8") as fh:
        via_loader = [ln.rstrip("\n") for ln in fh]
    assert [ln for ln in loader_lines(text) if ln] == via_loader
