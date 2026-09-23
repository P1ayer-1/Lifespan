"""The run record: the matrix shape, the timings, and the four files."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.config import EXAM_TYPES, N_PHASES
from training.runrecord import (
    MATRIX_KEYS,
    RunRecord,
    RunRecordError,
    Timings,
    empty_matrix,
    env_info,
    get_row,
    git_info,
    make_run_id,
    set_row,
    set_untrained,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def scores(value: float = 0.5) -> dict:
    """A complete evaluate_all return: all four MATRIX_KEYS."""
    return {t: [value + i / 100 for i in range(N_PHASES)] for t in MATRIX_KEYS}


def test_empty_matrix_has_the_frozen_shape():
    """Four keys since 2026-09-22: the three EXAM_TYPES plus
    `continuation_summed`, stored and never headlined."""
    m = empty_matrix()
    assert set(m) == {"untrained", "M"}
    assert set(m["M"]) == set(MATRIX_KEYS) == set(EXAM_TYPES) | {"continuation_summed"}
    assert set(m["untrained"]) == set(MATRIX_KEYS)
    for t in MATRIX_KEYS:
        assert len(m["untrained"][t]) == N_PHASES
        assert len(m["M"][t]) == N_PHASES
        assert all(len(row) == N_PHASES for row in m["M"][t])


def test_the_summed_continuation_column_is_required():
    """Mandatory since 2026-09-22: report.py and results/validate.py both need
    it, so a run that cannot write it fails here rather than producing a matrix
    the validator rejects hours later."""
    assert "continuation_summed" in MATRIX_KEYS and len(MATRIX_KEYS) == 4

    complete = scores()
    complete["continuation_summed"] = [9.0 + i for i in range(N_PHASES)]
    m = empty_matrix()
    set_row(m, 1, complete)
    set_untrained(m, complete)
    assert m["M"]["continuation_summed"][1][0] == pytest.approx(9.0)
    assert m["untrained"]["continuation_summed"][0] == pytest.approx(9.0)

    three_only = {t: [0.5] * N_PHASES for t in EXAM_TYPES}
    with pytest.raises(RunRecordError, match="continuation_summed"):
        set_row(empty_matrix(), 0, three_only)
    with pytest.raises(RunRecordError, match="continuation_summed"):
        set_untrained(empty_matrix(), three_only)


def test_an_unknown_exam_key_is_refused():
    m = empty_matrix()
    bad = scores()
    bad["vibes"] = [0.0] * N_PHASES
    with pytest.raises(RunRecordError, match="unknown keys"):
        set_row(m, 0, bad)


def test_a_row_is_stored_where_the_contract_says():
    """M[i][j]: after phase i, exam j."""
    m = empty_matrix()
    set_row(m, 3, scores(0.7))
    assert m["M"]["continuation"][3][0] == pytest.approx(0.7)
    assert m["M"]["continuation"][2] == [None] * N_PHASES
    assert get_row(m, 3)["cloze"][1] == pytest.approx(0.71)


def test_a_missing_exam_type_is_refused():
    m = empty_matrix()
    bad = scores()
    del bad["cloze"]
    with pytest.raises(RunRecordError, match="cloze"):
        set_row(m, 0, bad)


def test_a_row_of_the_wrong_length_is_refused():
    m = empty_matrix()
    bad = scores()
    bad["perplexity"] = [0.1, 0.2]
    with pytest.raises(RunRecordError, match="expected exactly 7"):
        set_row(m, 0, bad)


def test_a_non_numeric_score_is_refused():
    m = empty_matrix()
    bad = scores()
    bad["cloze"][2] = None
    with pytest.raises(RunRecordError, match="not a float"):
        set_untrained(m, bad)


def test_timings_split_the_wall_clock_three_ways():
    t = Timings()
    t.set_phase(1)
    with t.timer("train_s"):
        pass
    with t.timer("eval_s"):
        pass
    row = t.per_phase[0]
    assert set(row) == {"phase", "train_s", "consolidate_s", "eval_s"}
    assert row["train_s"] >= 0 and row["consolidate_s"] == 0.0
    assert t.to_dict()["gpu_hours"] == pytest.approx(t.total_seconds() / 3600)


def test_a_nested_timer_on_the_same_span_is_not_double_counted():
    """A hook may use ctx.timer('consolidate_s') inside the loop's own
    consolidate span; the inner one must not add the time twice."""
    t = Timings()
    t.set_phase(0)
    with t.timer("consolidate_s"):
        with t.timer("consolidate_s"):
            pass
        outer = t.per_phase[0]["consolidate_s"]
    assert outer == 0.0  # nothing was committed by the inner span
    assert t.per_phase[0]["consolidate_s"] > 0


def test_an_unknown_span_is_refused():
    t = Timings()
    t.set_phase(0)
    with pytest.raises(RunRecordError, match="unknown timing span"):
        with t.timer("sleep_s"):
            pass


def test_a_timer_before_set_phase_is_refused():
    with pytest.raises(RunRecordError, match="set_phase"):
        with Timings().timer("train_s"):
            pass


def test_adopt_copies_another_runs_phase_row():
    """How the shared phase-0 wall clock reaches each arm's timings."""
    t = Timings()
    t.adopt({"phase": 0, "train_s": 12.5, "consolidate_s": 0.0, "eval_s": 3.5})
    assert t.per_phase[0]["train_s"] == 12.5
    assert t.per_phase[0]["shared_from_phase0_run"] is True
    assert t.total_seconds() == pytest.approx(16.0)


def test_run_id_matches_the_frozen_pattern():
    rid = make_run_id("D-nr", 2, "abcdef1234567890", "20260921T120000Z")
    assert rid == "D-nr_s2_abcdef1_20260921T120000Z"


def test_the_record_writes_all_four_files(tmp_path):
    git, env = git_info(REPO_ROOT), env_info()
    rec = RunRecord(tmp_path / "run", {"arm": "A", "pilot": False}, git, env)
    m = empty_matrix()
    set_row(m, 0, scores())
    t = Timings()
    t.set_phase(0)
    rec.write_all(m, t)
    names = {p.name for p in (tmp_path / "run").iterdir()}
    assert {"config.json", "commit.txt", "matrix.json", "timings.json"} <= names
    written = json.loads((tmp_path / "run" / "matrix.json").read_text(encoding="utf-8"))
    assert written["M"]["cloze"][0][0] == pytest.approx(0.5)
    commit = (tmp_path / "run" / "commit.txt").read_text(encoding="utf-8")
    pairs = dict(
        line.split(": ", 1) for line in commit.splitlines() if line and not line.startswith("#")
    )
    # AGENTS.md, Amendments 2026-09-22: one "key: value" per line, dirty required.
    assert {"commit", "dirty", "torch", "cuda", "driver"} <= set(pairs)
    assert pairs["dirty"] in ("true", "false")


def test_a_dirty_tree_is_flagged_in_commit_txt(tmp_path):
    rec = RunRecord(tmp_path / "run", {}, {"commit": "x" * 40, "dirty": True, "branch": "m"}, env_info())
    rec.write_commit()
    text = (tmp_path / "run" / "commit.txt").read_text(encoding="utf-8")
    assert "dirty: true" in text
    assert "not a result" in text
    # A missing or unparseable dirty flag must never be readable as clean, so
    # the line is unconditional and bare.
    assert "dirty: True" not in text and "dirty: 1" not in text
