"""Tests for `results/validate.py`, one fabricated folder per failure mode.

`run-operator` owns this file (and `results/validate.py`). No import from
`training/` -- the validator and its tests must work even while `training/` is
mid-edit by other agents.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

# `results/validate.py` is not inside a conventional package import path
# (results/ holds run folders, not python modules other agents own), so it is
# loaded directly from its file path.
_SPEC = importlib.util.spec_from_file_location(
    "results_validate", Path(__file__).resolve().parent.parent / "results" / "validate.py"
)
validate_module = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = validate_module
_SPEC.loader.exec_module(validate_module)

validate = validate_module.validate
main = validate_module.main
EXAM_TYPES = validate_module.EXAM_TYPES
N_PHASES = validate_module.N_PHASES

GOOD_ARM = "D"
GOOD_SEED = 1
GOOD_COMMIT = "0123456789abcdef0123456789abcdef01234567"
GOOD_COMMIT7 = GOOD_COMMIT[:7]
GOOD_UTC = "20260922T101500Z"
GOOD_NAME = f"{GOOD_ARM}_s{GOOD_SEED}_{GOOD_COMMIT7}_{GOOD_UTC}"


def _matrix() -> dict:
    return {
        "untrained": {t: [0.1 * i for i in range(N_PHASES)] for t in EXAM_TYPES},
        "M": {t: [[0.1 * (i + j) for j in range(N_PHASES)] for i in range(N_PHASES)] for t in EXAM_TYPES},
    }


def _config(**overrides) -> dict:
    cfg = {
        "run_id": GOOD_NAME,
        "arm": GOOD_ARM,
        "seed": GOOD_SEED,
        "git": {"commit": GOOD_COMMIT, "dirty": False, "branch": "master"},
    }
    cfg.update(overrides)
    return cfg


def _commit_txt(dirty: str | None = "false") -> str:
    lines = [
        f"commit: {GOOD_COMMIT}",
        f"branch: master",
    ]
    if dirty is not None:
        lines.append(f"dirty: {dirty}")
    lines += ["torch: 2.14.0+cpu", "cuda: none", "driver: none"]
    return "\n".join(lines) + "\n"


def _timings(*, gpu_name: str | None = "T4", gpu_hours: float | None = 1.25) -> dict:
    d: dict = {"per_phase": [{"phase": i, "train_s": 1.0, "consolidate_s": 0.0, "eval_s": 0.5} for i in range(N_PHASES)]}
    if gpu_name is not None:
        d["gpu_name"] = gpu_name
    if gpu_hours is not None:
        d["gpu_hours"] = gpu_hours
    return d


def make_good_folder(tmp_path: Path, name: str = GOOD_NAME) -> Path:
    run_dir = tmp_path / name
    run_dir.mkdir(parents=True)
    (run_dir / "matrix.json").write_text(json.dumps(_matrix()), encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps(_config()), encoding="utf-8")
    (run_dir / "commit.txt").write_text(_commit_txt(), encoding="utf-8")
    (run_dir / "timings.json").write_text(json.dumps(_timings()), encoding="utf-8")
    return run_dir


# ---------------------------------------------------------------------------
# the happy path


def test_a_well_formed_folder_is_valid(tmp_path):
    run_dir = make_good_folder(tmp_path)
    assert validate(run_dir) == []


# ---------------------------------------------------------------------------
# missing files


@pytest.mark.parametrize("missing_file", ["matrix.json", "timings.json", "config.json", "commit.txt"])
def test_missing_file_fails(tmp_path, missing_file):
    run_dir = make_good_folder(tmp_path)
    (run_dir / missing_file).unlink()
    failures = validate(run_dir)
    assert any(missing_file in f for f in failures)


def test_missing_all_four_files_reports_them_together(tmp_path):
    run_dir = tmp_path / GOOD_NAME
    run_dir.mkdir()
    failures = validate(run_dir)
    joined = " ".join(failures)
    for f in ("matrix.json", "timings.json", "config.json", "commit.txt"):
        assert f in joined


# ---------------------------------------------------------------------------
# matrix shape and finiteness


def test_matrix_wrong_row_count_fails(tmp_path):
    run_dir = make_good_folder(tmp_path)
    m = _matrix()
    m["M"]["cloze"] = m["M"]["cloze"][:6]  # 6x7 instead of 7x7
    (run_dir / "matrix.json").write_text(json.dumps(m), encoding="utf-8")
    failures = validate(run_dir)
    assert any("cloze" in f and "7" in f for f in failures)


def test_matrix_wrong_column_count_fails(tmp_path):
    run_dir = make_good_folder(tmp_path)
    m = _matrix()
    m["M"]["perplexity"][2] = m["M"]["perplexity"][2][:5]
    (run_dir / "matrix.json").write_text(json.dumps(m), encoding="utf-8")
    failures = validate(run_dir)
    assert any("perplexity" in f for f in failures)


def test_matrix_nan_cell_fails(tmp_path):
    run_dir = make_good_folder(tmp_path)
    m = _matrix()
    m["M"]["continuation"][3][4] = float("nan")
    (run_dir / "matrix.json").write_text(json.dumps(m), encoding="utf-8")
    failures = validate(run_dir)
    assert any("continuation" in f and "non-finite" in f for f in failures)


def test_matrix_infinite_cell_fails(tmp_path):
    run_dir = make_good_folder(tmp_path)
    m = _matrix()
    m["untrained"]["continuation_summed"][0] = float("inf")
    (run_dir / "matrix.json").write_text(json.dumps(m), encoding="utf-8")
    failures = validate(run_dir)
    assert any("continuation_summed" in f and "non-finite" in f for f in failures)


def test_matrix_null_cell_fails(tmp_path):
    """An unfinished phase (null) is not a complete result."""
    run_dir = make_good_folder(tmp_path)
    m = _matrix()
    m["M"]["cloze"][6][6] = None
    (run_dir / "matrix.json").write_text(json.dumps(m), encoding="utf-8")
    failures = validate(run_dir)
    assert any("cloze" in f and "non-finite" in f for f in failures)


def test_matrix_missing_exam_type_fails(tmp_path):
    run_dir = make_good_folder(tmp_path)
    m = _matrix()
    del m["M"]["continuation_summed"]
    (run_dir / "matrix.json").write_text(json.dumps(m), encoding="utf-8")
    failures = validate(run_dir)
    assert any("continuation_summed" in f for f in failures)


def test_matrix_invalid_json_fails(tmp_path):
    run_dir = make_good_folder(tmp_path)
    (run_dir / "matrix.json").write_text("{not json", encoding="utf-8")
    failures = validate(run_dir)
    assert any("matrix.json" in f for f in failures)


# ---------------------------------------------------------------------------
# commit.txt dirty flag


def test_dirty_true_fails(tmp_path):
    run_dir = make_good_folder(tmp_path)
    (run_dir / "commit.txt").write_text(_commit_txt(dirty="true"), encoding="utf-8")
    failures = validate(run_dir)
    assert any("dirty is true" in f for f in failures)


def test_dirty_missing_fails(tmp_path):
    run_dir = make_good_folder(tmp_path)
    (run_dir / "commit.txt").write_text(_commit_txt(dirty=None), encoding="utf-8")
    failures = validate(run_dir)
    assert any("no 'dirty' line" in f for f in failures)


def test_dirty_unparseable_fails(tmp_path):
    run_dir = make_good_folder(tmp_path)
    (run_dir / "commit.txt").write_text(_commit_txt(dirty="maybe"), encoding="utf-8")
    failures = validate(run_dir)
    assert any("not 'true' or 'false'" in f for f in failures)


def test_dirty_is_never_assumed_clean_on_empty_file(tmp_path):
    run_dir = make_good_folder(tmp_path)
    (run_dir / "commit.txt").write_text("", encoding="utf-8")
    failures = validate(run_dir)
    assert any("dirty" in f for f in failures)


# ---------------------------------------------------------------------------
# timings.json gpu fields


def test_missing_gpu_name_fails(tmp_path):
    run_dir = make_good_folder(tmp_path)
    (run_dir / "timings.json").write_text(json.dumps(_timings(gpu_name=None)), encoding="utf-8")
    failures = validate(run_dir)
    assert any("gpu_name" in f for f in failures)


def test_empty_gpu_name_fails(tmp_path):
    run_dir = make_good_folder(tmp_path)
    (run_dir / "timings.json").write_text(json.dumps(_timings(gpu_name="   ")), encoding="utf-8")
    failures = validate(run_dir)
    assert any("gpu_name" in f for f in failures)


def test_missing_gpu_hours_fails(tmp_path):
    run_dir = make_good_folder(tmp_path)
    (run_dir / "timings.json").write_text(json.dumps(_timings(gpu_hours=None)), encoding="utf-8")
    failures = validate(run_dir)
    assert any("gpu_hours" in f for f in failures)


def test_negative_gpu_hours_fails(tmp_path):
    run_dir = make_good_folder(tmp_path)
    (run_dir / "timings.json").write_text(json.dumps(_timings(gpu_hours=-1.0)), encoding="utf-8")
    failures = validate(run_dir)
    assert any("gpu_hours" in f for f in failures)


# ---------------------------------------------------------------------------
# run_id / config.json agreement


def test_run_id_arm_mismatch_fails(tmp_path):
    run_dir = make_good_folder(tmp_path)
    cfg = _config(arm="C")
    (run_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    failures = validate(run_dir)
    assert any("arm" in f and "disagrees" in f for f in failures)


def test_run_id_seed_mismatch_fails(tmp_path):
    run_dir = make_good_folder(tmp_path)
    cfg = _config(seed=2)
    (run_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    failures = validate(run_dir)
    assert any("seed" in f and "disagrees" in f for f in failures)


def test_run_id_commit_mismatch_fails(tmp_path):
    run_dir = make_good_folder(tmp_path)
    other_commit = "fedcba9876543210fedcba9876543210fedcba9"
    cfg = _config(git={"commit": other_commit, "dirty": False, "branch": "master"})
    (run_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    failures = validate(run_dir)
    assert any("commit7" in f and "disagrees" in f for f in failures)


def test_folder_name_not_matching_run_id_shape_fails(tmp_path):
    run_dir = make_good_folder(tmp_path, name="not_a_run_id")
    failures = validate(run_dir)
    assert any("does not match run_id shape" in f for f in failures)


def test_resumed_run_may_have_a_different_utc_suffix(tmp_path):
    """A --resume'd run keeps its original folder name while config.json's
    run_id gets a fresh timestamp; only arm/seed/commit7 are checked."""
    run_dir = make_good_folder(tmp_path)
    cfg = _config(run_id=f"{GOOD_ARM}_s{GOOD_SEED}_{GOOD_COMMIT7}_20260923T000000Z")
    (run_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    assert validate(run_dir) == []


# ---------------------------------------------------------------------------
# multiple failures reported together, not just the first


def test_multiple_failures_all_named(tmp_path):
    run_dir = make_good_folder(tmp_path)
    (run_dir / "commit.txt").write_text(_commit_txt(dirty="true"), encoding="utf-8")
    (run_dir / "timings.json").write_text(json.dumps(_timings(gpu_name=None)), encoding="utf-8")
    m = _matrix()
    m["M"]["perplexity"][0][0] = float("nan")
    (run_dir / "matrix.json").write_text(json.dumps(m), encoding="utf-8")
    failures = validate(run_dir)
    assert any("dirty is true" in f for f in failures)
    assert any("gpu_name" in f for f in failures)
    assert any("perplexity" in f for f in failures)
    assert len(failures) >= 3


# ---------------------------------------------------------------------------
# arm phase0's legitimately-partial matrix (row 0 only, forever)


def _phase0_matrix() -> dict:
    """Only row/col 0 of M is ever populated for `--arm phase0`
    (training/train.py: "arm phase0 trains phase 0 only"); the untrained row
    is always full (evaluate_all scores all seven phases up front)."""
    m = _matrix()
    for t in EXAM_TYPES:
        for i in range(1, N_PHASES):
            m["M"][t][i] = [None] * N_PHASES
    return m


def make_phase0_folder(tmp_path: Path, *, seed: int = 0) -> Path:
    name = f"phase0_s{seed}_{GOOD_COMMIT7}_{GOOD_UTC}"
    run_dir = tmp_path / name
    run_dir.mkdir(parents=True)
    (run_dir / "matrix.json").write_text(json.dumps(_phase0_matrix()), encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps(_config(arm="phase0", seed=seed)), encoding="utf-8")
    (run_dir / "commit.txt").write_text(_commit_txt(), encoding="utf-8")
    (run_dir / "timings.json").write_text(json.dumps(_timings()), encoding="utf-8")
    return run_dir


def test_phase0_with_only_row_zero_populated_is_valid(tmp_path):
    run_dir = make_phase0_folder(tmp_path)
    assert validate(run_dir) == []


def test_phase0_missing_even_row_zero_still_fails(tmp_path):
    run_dir = make_phase0_folder(tmp_path)
    m = _phase0_matrix()
    m["M"]["cloze"][0] = [None] * N_PHASES
    (run_dir / "matrix.json").write_text(json.dumps(m), encoding="utf-8")
    failures = validate(run_dir)
    assert any("cloze" in f and "non-finite" in f for f in failures)


def test_a_non_phase0_arm_with_only_row_zero_is_still_incomplete(tmp_path):
    """The phase0 exemption is keyed on config.json's arm, not on the shape
    alone: the same partial matrix under arm 'D' (GOOD_ARM) is not complete."""
    run_dir = make_good_folder(tmp_path)
    (run_dir / "matrix.json").write_text(json.dumps(_phase0_matrix()), encoding="utf-8")
    failures = validate(run_dir)
    assert any("non-finite" in f for f in failures)


# ---------------------------------------------------------------------------
# CLI


def test_main_returns_zero_for_all_valid(tmp_path, capsys):
    run_dir = make_good_folder(tmp_path)
    rc = main([str(run_dir)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "valid" in out


def test_main_returns_nonzero_and_lists_reasons(tmp_path, capsys):
    run_dir = make_good_folder(tmp_path)
    (run_dir / "commit.txt").write_text(_commit_txt(dirty="true"), encoding="utf-8")
    rc = main([str(run_dir)])
    assert rc == 1
    out = capsys.readouterr().out
    assert "INVALID" in out
    assert "dirty is true" in out


def test_main_with_no_args_scans_results_root(tmp_path, capsys):
    good = make_good_folder(tmp_path)
    bad_dir = tmp_path / f"A_s0_{GOOD_COMMIT7}_20260922T000000Z"
    bad_dir.mkdir()
    rc = main(["--results-root", str(tmp_path)])
    assert rc == 1
    out = capsys.readouterr().out
    assert good.name in out
    assert bad_dir.name in out


# ---------------------------------------------------------------------------
# subset runs (train.py --phases 0,3,6): config.json "phases", real ids

SUBSET = [0, 3, 6]
NAN = float("nan")


def _subset_matrix(phases=SUBSET) -> dict:
    """What train.py writes for a 0/3/6 run: undeclared rows null, undeclared
    exam columns NaN (json writes the NaN literal), untrained likewise."""
    m = _matrix()
    for t in EXAM_TYPES:
        m["untrained"][t] = [v if j in phases else NAN for j, v in enumerate(m["untrained"][t])]
        m["M"][t] = [
            [v if j in phases else NAN for j, v in enumerate(row)] if i in phases else [None] * N_PHASES
            for i, row in enumerate(m["M"][t])
        ]
    return m


def make_subset_folder(tmp_path: Path, *, arm: str = GOOD_ARM, phases=SUBSET) -> Path:
    run_dir = make_good_folder(tmp_path, name=f"{arm}_s{GOOD_SEED}_{GOOD_COMMIT7}_{GOOD_UTC}")
    (run_dir / "matrix.json").write_text(json.dumps(_subset_matrix()), encoding="utf-8")
    (run_dir / "config.json").write_text(
        json.dumps(_config(arm=arm, phases=phases, subset_phases=True)), encoding="utf-8"
    )
    return run_dir


def test_a_subset_run_with_null_rows_and_nan_columns_is_valid(tmp_path):
    assert validate(make_subset_folder(tmp_path)) == []


def test_a_subset_run_missing_a_declared_cell_fails(tmp_path):
    run_dir = make_subset_folder(tmp_path)
    m = _subset_matrix()
    m["M"]["cloze"][3][6] = None
    m["untrained"]["perplexity"][3] = NAN
    (run_dir / "matrix.json").write_text(json.dumps(m), encoding="utf-8")
    failures = validate(run_dir)
    assert any("M['cloze'][3]" in f and "[6]" in f for f in failures)
    assert any("untrained['perplexity']" in f and "[3]" in f for f in failures)


def test_a_subset_run_missing_a_whole_declared_row_fails(tmp_path):
    run_dir = make_subset_folder(tmp_path)
    m = _subset_matrix()
    m["M"]["continuation"][6] = [None] * N_PHASES
    (run_dir / "matrix.json").write_text(json.dumps(m), encoding="utf-8")
    assert any("M['continuation'][6]" in f for f in validate(run_dir))


def test_the_same_partial_matrix_without_a_phases_declaration_is_incomplete(tmp_path):
    """A full run is checked exactly as before: null rows mean a dead session."""
    run_dir = make_good_folder(tmp_path)
    (run_dir / "matrix.json").write_text(json.dumps(_subset_matrix()), encoding="utf-8")
    failures = validate(run_dir)
    assert any("non-finite" in f for f in failures)


def test_a_full_run_that_declares_all_seven_is_checked_as_a_full_run(tmp_path):
    run_dir = make_good_folder(tmp_path)
    (run_dir / "config.json").write_text(json.dumps(_config(phases=list(range(N_PHASES)))), encoding="utf-8")
    assert validate(run_dir) == []
    (run_dir / "matrix.json").write_text(json.dumps(_subset_matrix()), encoding="utf-8")
    assert any("non-finite" in f for f in validate(run_dir))


@pytest.mark.parametrize("bad", [[3, 6], [0, 6, 3], [0, 3, 3], [0, 7], [], "0,3,6", [0, True]])
def test_an_invalid_phases_declaration_fails(tmp_path, bad):
    run_dir = make_good_folder(tmp_path)
    (run_dir / "config.json").write_text(json.dumps(_config(phases=bad)), encoding="utf-8")
    assert any("'phases'" in f for f in validate(run_dir))


def test_a_subset_phase0_run_needs_row_zero_on_declared_columns_only(tmp_path):
    run_dir = make_subset_folder(tmp_path, arm="phase0")
    m = _subset_matrix()
    for t in EXAM_TYPES:
        for i in (3, 6):
            m["M"][t][i] = [None] * N_PHASES
    (run_dir / "matrix.json").write_text(json.dumps(m), encoding="utf-8")
    assert validate(run_dir) == []
