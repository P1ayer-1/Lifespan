"""Aggregation, exclusions and the H1-H4 table, on fabricated result folders.

Every matrix here is invented by this file, built from two numbers per arm
(final average accuracy and average forgetting) so the expected verdicts can be
read straight off the thresholds in PLAN.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training import report
from training.config import EXAM_TYPES, N_PHASES, SEEDS

GPU = "NVIDIA A100-SXM4-40GB"

# config.json's one frozen hash block (AGENTS.md, amendments 2026-09-22).
HASHES = {"hashes": {"manifest": "m" * 64,
                     "train_files": {"train_phase_0.jsonl": "t" * 64},
                     "config": "c" * 64}}

# Settled numbers, hard-coded here as checks rather than recomputed.
REAL_N_PARAMETERS = 29_901_824
GRID_STEPS_PER_PHASE = 367
GRID_NEW_TOKENS_PER_PHASE = 12_025_856
GRID_NEW_TOKENS_TOTAL = 84_180_992


# --------------------------------------------------------------------------- #
# fabricated runs
# --------------------------------------------------------------------------- #


def build_matrix(final_acc: float, forgetting: float) -> list[list[float]]:
    """A 7x7 matrix with exactly the stated average accuracy and forgetting.

    Every cell is `final_acc` except the diagonal for j < 6, which is the
    column's peak at `final_acc + forgetting`. So:
      - the last row is all `final_acc`      -> average accuracy = final_acc
      - forgetting of j < 6 = peak - final   = forgetting
      - forgetting of j = 6 = 0 (its peak is its final score), and j=6 is
        excluded from the average anyway.
    """
    M = [[final_acc] * N_PHASES for _ in range(N_PHASES)]
    for j in range(N_PHASES - 1):
        M[j][j] = final_acc + forgetting
    return M


def write_run(
    results_dir: Path,
    arm: str,
    seed: int,
    *,
    final_acc: float,
    forgetting: float,
    gpu_hours: float = 1.0,
    cloze_final_acc: float | None = None,
    cloze_forgetting: float | None = None,
    gpu_name: str = GPU,
    pilot: bool = False,
    dirty: bool = False,
    hashes: dict | None = None,
    omit: tuple[str, ...] = (),
    commit_txt: str | None = None,
    token_budget: dict | None = None,
    phases: tuple[int, ...] | list | None = None,
) -> Path:
    """One `results/<run_id>/` folder, fabricated.

    `phases` makes it a subset run the way train.py writes one: config.json
    carries "phases", undeclared rows of M are null and undeclared exam
    columns (and untrained entries) NaN."""
    run_id = arm + "_s" + str(seed) + "_abc1234_20260921T000000Z"
    path = results_dir / run_id
    path.mkdir(parents=True, exist_ok=True)
    cont = build_matrix(final_acc, forgetting)
    cloze = build_matrix(
        final_acc if cloze_final_acc is None else cloze_final_acc,
        forgetting if cloze_forgetting is None else cloze_forgetting,
    )
    # perplexity is stored as a loss; a mirror of the accuracy matrix would be
    # nonsense, so it is its own gently rising thing.
    ppl = [[2.0 - 0.1 * i + 0.05 * j for j in range(N_PHASES)] for i in range(N_PHASES)]
    # The summed continuation score is stored beside the three exam types and
    # never headlined; a fabricated run carries it like a real one would.
    summed = build_matrix(final_acc * 0.8, forgetting * 0.8)
    matrix = {
        "untrained": {"perplexity": [3.0] * N_PHASES, "cloze": [0.05] * N_PHASES,
                      "continuation": [0.25] * N_PHASES,
                      "continuation_summed": [0.25] * N_PHASES},
        "M": {"perplexity": ppl, "cloze": cloze, "continuation": cont,
              "continuation_summed": summed},
    }
    config_extra: dict = {}
    if phases is not None:
        config_extra["phases"] = list(phases)
        declared = set(phases) if all(isinstance(p, int) for p in phases) else set()
        nan = float("nan")
        for key, block in matrix["M"].items():
            matrix["M"][key] = [
                [v if j in declared else nan for j, v in enumerate(row)] if i in declared else [None] * N_PHASES
                for i, row in enumerate(block)
            ]
        for key, row in matrix["untrained"].items():
            matrix["untrained"][key] = [v if j in declared else nan for j, v in enumerate(row)]
    files = {
        "matrix.json": json.dumps(matrix),
        "timings.json": json.dumps({
            "per_phase": [{"phase": k, "train_s": 10.0, "consolidate_s": 0.0, "eval_s": 1.0}
                          for k in range(N_PHASES)],
            "gpu_hours": gpu_hours,
            "gpu_name": gpu_name,
        }),
        "config.json": json.dumps(dict(
            {"arm": arm, "seed": seed, "pilot": pilot,
             "token_budget": token_budget if token_budget is not None else {
                 "total_new_phase_tokens": GRID_NEW_TOKENS_TOTAL,
                 "total_replay_tokens": 0,
             }},
            **(HASHES if hashes is None else hashes),
            **config_extra,
        )),
        "commit.txt": commit_txt if commit_txt is not None else (
            "commit: " + "a" * 40 + "\n"
            "branch: main\n"
            "dirty: " + ("true" if dirty else "false") + "\n"
            "torch: 2.14.0+cu124\n"
            "cuda: 12.4\n"
            "driver: 550.54\n"
        ),
    }
    for name, body in files.items():
        if name in omit:
            continue
        (path / name).write_text(body, encoding="utf-8")
    return path


def write_arm(results_dir: Path, arm: str, *, final_acc: float, forgetting: float, **kw) -> None:
    """Three seeds with a small per-seed jitter, so every aggregate has a real
    (small) spread: accuracy +/-0.005, forgetting +/-0.004."""
    for seed in SEEDS:
        d = (seed - 1)
        kwargs = dict(kw)
        if "cloze_final_acc" in kwargs and kwargs["cloze_final_acc"] is not None:
            kwargs["cloze_final_acc"] = kwargs["cloze_final_acc"] + 0.005 * d
        if "cloze_forgetting" in kwargs and kwargs["cloze_forgetting"] is not None:
            kwargs["cloze_forgetting"] = kwargs["cloze_forgetting"] + 0.004 * d
        write_run(
            results_dir, arm, seed,
            final_acc=final_acc + 0.005 * d,
            forgetting=forgetting + 0.004 * d,
            **kwargs,
        )


# --------------------------------------------------------------------------- #
# 7. aggregation: mean, sample sd, n
# --------------------------------------------------------------------------- #


def test_aggregate_over_three_seeds_gives_the_hand_computed_mean_and_sd(tmp_path):
    # final accuracies 0.595 / 0.600 / 0.605 -> mean 0.600,
    # sample sd = sqrt(((0.005)^2 + 0 + (0.005)^2) / 2) = 0.005
    write_arm(tmp_path, "D", final_acc=0.600, forgetting=0.100)
    runs, excluded = report.discover_runs(tmp_path)
    assert excluded == []
    assert len(runs) == 3
    agg = report.arm_average_accuracy(runs, "continuation")
    assert agg.n == 3
    assert agg.mean == pytest.approx(0.600)
    assert agg.sd == pytest.approx(0.005)
    assert agg.seeds == [0, 1, 2]
    assert agg.values == pytest.approx([0.595, 0.600, 0.605])
    text = str(agg)
    assert "n=3" in text and "0.6000" in text and "0.0050" in text


def test_a_difference_smaller_than_the_seed_spread_is_no_difference():
    a = report.Aggregate("a", [0.50, 0.60, 0.70], list(SEEDS))  # mean 0.60, sd 0.10
    b = report.Aggregate("b", [0.64, 0.65, 0.66], list(SEEDS))  # mean 0.65, sd 0.01
    small = report.compare(a, b)
    assert small.diff == pytest.approx(-0.05)
    assert small.spread == pytest.approx(0.10)  # max(0.10, 0.01)
    assert small.mde == pytest.approx(0.10)
    assert not small.decided
    assert "no difference" in str(small)

    c = report.Aggregate("c", [0.89, 0.90, 0.91], list(SEEDS))  # mean 0.90, sd 0.01
    big = report.compare(c, a)
    assert big.diff == pytest.approx(0.30)
    assert big.decided  # 0.30 > spread 0.10
    assert "no difference" not in str(big)


def test_a_single_seed_prints_n_beside_its_zero_spread():
    one = report.Aggregate("one", [0.5], [0])
    assert one.sd == 0.0
    assert "n=1" in str(one)


# --------------------------------------------------------------------------- #
# 7. exclusions, by name
# --------------------------------------------------------------------------- #


def test_a_pilot_run_is_excluded_from_a_grid_report(tmp_path):
    write_arm(tmp_path, "A", final_acc=0.30, forgetting=0.30)
    write_run(tmp_path, "Apilot", 0, final_acc=0.30, forgetting=0.30, pilot=True)
    runs, excluded = report.discover_runs(tmp_path, pilot=False)
    assert len(runs) == 3
    assert [e.run_id for e in excluded] == ["Apilot_s0_abc1234_20260921T000000Z"]
    assert "pilot" in excluded[0].reason
    assert "Apilot" in report.render_exclusions(excluded)
    # and the other way round: the grid runs are excluded from a pilot report
    pilot_runs, pilot_excluded = report.discover_runs(tmp_path, pilot=True)
    assert len(pilot_runs) == 1
    assert len(pilot_excluded) == 3


def test_a_dirty_tree_run_is_excluded_and_reported(tmp_path):
    write_arm(tmp_path, "A", final_acc=0.30, forgetting=0.30)
    write_run(tmp_path, "B", 0, final_acc=0.30, forgetting=0.30, dirty=True)
    runs, excluded = report.discover_runs(tmp_path)
    assert len(runs) == 3
    assert len(excluded) == 1
    assert excluded[0].run_id.startswith("B_s0")
    assert "dirty" in excluded[0].reason
    assert "B_s0" in report.render_exclusions(excluded)


def test_an_incomplete_run_is_excluded_by_name(tmp_path):
    write_run(tmp_path, "A", 0, final_acc=0.3, forgetting=0.3, omit=("timings.json",))
    write_run(tmp_path, "A", 1, final_acc=0.3, forgetting=0.3, omit=("matrix.json",))
    runs, excluded = report.discover_runs(tmp_path)
    assert runs == []
    reasons = {e.run_id: e.reason for e in excluded}
    assert "timings.json is missing" in reasons["A_s0_abc1234_20260921T000000Z"]
    assert "matrix.json is missing" in reasons["A_s1_abc1234_20260921T000000Z"]


def test_a_partial_run_from_a_dead_session_is_excluded_not_fatal(tmp_path):
    """The Kaggle failure mode: a session dies part-way and the later rows are
    still null. The report must name that run and carry on, not die on it."""
    write_arm(tmp_path, "A", final_acc=0.30, forgetting=0.30)
    path = write_run(tmp_path, "D", 0, final_acc=0.6, forgetting=0.1)
    doc = json.loads((path / "matrix.json").read_text(encoding="utf-8"))
    for key in doc["M"]:
        for i in (4, 5, 6):
            doc["M"][key][i] = [None] * N_PHASES
    (path / "matrix.json").write_text(json.dumps(doc), encoding="utf-8")

    loaded = report.load_run(path)  # must not raise
    assert isinstance(loaded, report.Exclusion)
    assert "row(s) 4, 5, 6" in loaded.reason
    assert "did not finish" in loaded.reason

    # and the surrounding report survives it
    runs, excluded = report.discover_runs(tmp_path)
    assert [r.arm for r in runs] == ["A", "A", "A"]
    assert len(excluded) == 1 and excluded[0].run_id.startswith("D_s0")
    assert "row(s) 4, 5, 6" in report.render_exclusions(excluded)
    out = tmp_path / "out"
    text = report.write_report(tmp_path, out, plots=False).read_text(encoding="utf-8")
    assert "Excluded runs (1)" in text and "row(s) 4, 5, 6" in text


def test_a_partially_written_row_is_excluded(tmp_path):
    """Nulls inside one row, not whole rows: the same treatment."""
    path = write_run(tmp_path, "D", 0, final_acc=0.6, forgetting=0.1)
    doc = json.loads((path / "matrix.json").read_text(encoding="utf-8"))
    doc["M"]["continuation"][3][5] = None
    (path / "matrix.json").write_text(json.dumps(doc), encoding="utf-8")
    loaded = report.load_run(path)
    assert isinstance(loaded, report.Exclusion)
    assert "row(s) 3" in loaded.reason


def test_a_null_untrained_vector_is_excluded(tmp_path):
    """Forward transfer is measured against the untrained scores and they
    cannot be recovered afterwards."""
    path = write_run(tmp_path, "D", 0, final_acc=0.6, forgetting=0.1)
    doc = json.loads((path / "matrix.json").read_text(encoding="utf-8"))
    doc["untrained"]["continuation"][2] = None
    (path / "matrix.json").write_text(json.dumps(doc), encoding="utf-8")
    loaded = report.load_run(path)
    assert isinstance(loaded, report.Exclusion)
    assert "phase(s) 2" in loaded.reason
    assert "forward transfer" in loaded.reason


def test_an_entirely_unscored_matrix_is_excluded(tmp_path):
    """`runrecord.empty_matrix` before anything has been scored."""
    path = write_run(tmp_path, "D", 0, final_acc=0.6, forgetting=0.1)
    doc = json.loads((path / "matrix.json").read_text(encoding="utf-8"))
    for key in doc["M"]:
        doc["M"][key] = [[None] * N_PHASES for _ in range(N_PHASES)]
        doc["untrained"][key] = [None] * N_PHASES
    (path / "matrix.json").write_text(json.dumps(doc), encoding="utf-8")
    loaded = report.load_run(path)
    assert isinstance(loaded, report.Exclusion)
    assert "row(s) 0, 1, 2, 3, 4, 5, 6" in loaded.reason


def test_a_non_finite_matrix_is_excluded(tmp_path):
    path = write_run(tmp_path, "A", 0, final_acc=0.3, forgetting=0.3)
    doc = json.loads((path / "matrix.json").read_text(encoding="utf-8"))
    doc["M"]["continuation"][3][4] = None
    (path / "matrix.json").write_text(json.dumps(doc).replace("null", "NaN"), encoding="utf-8")
    loaded = report.load_run(path)
    assert isinstance(loaded, report.Exclusion)
    assert "non-finite" in loaded.reason


def test_a_run_with_different_hashes_is_excluded_from_its_comparison(tmp_path):
    write_arm(tmp_path, "D", final_acc=0.60, forgetting=0.10)
    write_run(tmp_path, "D", 3, final_acc=0.60, forgetting=0.10,
              hashes={"hashes": {"manifest": "x" * 64,
                                 "train_files": {"train_phase_0.jsonl": "y" * 64},
                                 "config": "z" * 64}})
    runs, excluded = report.discover_runs(tmp_path)
    assert len(runs) == 4 and excluded == []
    kept, refused = report.filter_consistent(runs)
    assert len(kept) == 3
    assert [r.seed for r in kept] == [0, 1, 2]
    assert len(refused) == 1
    assert refused[0].run_id.startswith("D_s3")
    assert "hashes differ" in refused[0].reason


def _hashes(fingerprint: str, scoring: str = "s" * 64) -> dict:
    return {"hashes": {**HASHES["hashes"], "scoring": scoring, "fingerprint": fingerprint}}


def test_the_per_run_fingerprint_does_not_split_a_comparison(tmp_path):
    """train.py writes the checkpoint fingerprint's digest (seed and commit
    included) into `hashes`; it differs between every two runs and must not
    count (2026-10-01: it would have kept one run per comparison)."""
    for seed in (0, 1, 2):
        write_run(tmp_path, "A", seed, final_acc=0.5, forgetting=0.2, hashes=_hashes(str(seed) * 64))
    runs, _ = report.discover_runs(tmp_path)
    kept, refused = report.filter_consistent(runs)
    assert len(kept) == 3 and refused == []


def test_runs_scored_under_another_rule_are_refused(tmp_path):
    """AGENTS.md, Amendment 4: a matrix scored before the PMI change is never
    pooled with one scored after it."""
    for seed in (0, 1, 2):
        write_run(tmp_path, "A", seed, final_acc=0.5, forgetting=0.2, hashes=_hashes(str(seed) * 64))
    write_run(tmp_path, "A", 3, final_acc=0.5, forgetting=0.2,
              hashes=_hashes("3" * 64, scoring="o" * 64))
    runs, _ = report.discover_runs(tmp_path)
    kept, refused = report.filter_consistent(runs)
    assert [r.seed for r in kept] == [0, 1, 2]
    assert len(refused) == 1 and refused[0].run_id.startswith("A_s3")


GOOD_COMMIT = (
    "commit: " + "a" * 40 + "\ndirty: false\ntorch: 2.14.0\ncuda: 12.4\ndriver: 550.54\n"
)
NO_DIRTY_KEY = "commit: " + "a" * 40 + "\ntorch: 2.14.0\ncuda: 12.4\ndriver: 550.54\n"


def test_commit_txt_is_one_key_value_per_line():
    fields = report.parse_commit_txt(GOOD_COMMIT)
    assert fields["dirty"] == "false"
    assert fields["commit"] == "a" * 40
    assert report.check_commit_txt(fields) == (False, None)
    # trainer-core's extra lines (branch, a blank line, the dirty WARNING) are
    # ignored rather than fatal
    fields = report.parse_commit_txt(
        GOOD_COMMIT.replace("dirty: false", "dirty: true")
        + "branch: main\n\nWARNING: the working tree was dirty.\n"
    )
    assert report.check_commit_txt(fields) == (True, None)


def test_an_unparseable_dirty_flag_is_never_assumed_clean():
    """The one default that would let a dirty tree become a result."""
    dirty, why = report.check_commit_txt(report.parse_commit_txt(NO_DIRTY_KEY))
    assert dirty is True
    assert "missing required key" in why and "dirty" in why

    dirty, why = report.check_commit_txt(
        report.parse_commit_txt(GOOD_COMMIT.replace("dirty: false", "dirty: maybe"))
    )
    assert dirty is True
    assert "unparseable dirty flag" in why

    # the old fail-open forms are gone: one format, and anything else is refused
    dirty, why = report.check_commit_txt(report.parse_commit_txt("commit=abc\ndirty=false\n"))
    assert dirty is True and why is not None
    dirty, why = report.check_commit_txt(report.parse_commit_txt("abc123\ndirty\n"))
    assert dirty is True and why is not None


def test_a_run_whose_commit_txt_is_unparseable_is_excluded_and_named(tmp_path):
    write_arm(tmp_path, "A", final_acc=0.30, forgetting=0.30)
    write_run(tmp_path, "B", 0, final_acc=0.3, forgetting=0.3, commit_txt=NO_DIRTY_KEY)
    write_run(tmp_path, "C", 0, final_acc=0.3, forgetting=0.3,
              commit_txt=GOOD_COMMIT.replace("dirty: false", "dirty: probably"))
    write_run(tmp_path, "E", 0, final_acc=0.3, forgetting=0.3, commit_txt="who knows\n")
    runs, excluded = report.discover_runs(tmp_path)
    assert [r.arm for r in runs] == ["A", "A", "A"]  # only the provably clean ones survive
    reasons = {e.run_id.split("_")[0]: e.reason for e in excluded}
    assert set(reasons) == {"B", "C", "E"}
    assert "missing required key" in reasons["B"] and "dirty" in reasons["B"]
    assert "unparseable dirty flag" in reasons["C"]
    assert "never assumed" in reasons["B"]
    text = report.render_exclusions(excluded)
    assert "B_s0" in text and "C_s0" in text and "E_s0" in text


def test_a_run_without_a_hashes_block_is_excluded(tmp_path):
    """The pre-amendment shape (loose data_hash / manifest_hash keys) no longer
    counts: the hashes live under one top-level key or the run is not
    comparable."""
    write_run(tmp_path, "A", 0, final_acc=0.3, forgetting=0.3, hashes={"data_hash": "d" * 64})
    runs, excluded = report.discover_runs(tmp_path)
    assert runs == []
    assert 'no top-level "hashes" block' in excluded[0].reason


def test_a_matrix_without_the_stored_summed_key_is_excluded(tmp_path):
    path = write_run(tmp_path, "A", 0, final_acc=0.3, forgetting=0.3)
    doc = json.loads((path / "matrix.json").read_text(encoding="utf-8"))
    del doc["M"]["continuation_summed"]
    (path / "matrix.json").write_text(json.dumps(doc), encoding="utf-8")
    loaded = report.load_run(path)
    assert isinstance(loaded, report.Exclusion)
    assert "continuation_summed" in loaded.reason


def test_continuation_summed_is_loaded_but_never_a_verdict_metric(tmp_path):
    write_arm(tmp_path, "A", final_acc=0.30, forgetting=0.30)
    runs, _ = report.discover_runs(tmp_path)
    assert len(runs[0].matrix["continuation_summed"]) == N_PHASES
    # it aggregates like any stored key ...
    agg = report.arm_average_accuracy(runs, "continuation_summed")
    assert agg.mean == pytest.approx(0.24)  # 0.8 * 0.30
    # ... but it can never decide a hypothesis
    with pytest.raises(ValueError):
        report.hypothesis_table(report.group_by_arm(runs), "continuation_summed")


def test_the_settled_grid_numbers_are_what_the_report_reads(tmp_path):
    """REAL's parameter count and the grid's token budget, hard-coded as checks.

    The B/A token ratio is 1.375, not PLAN.md's "about 1.43x", because phase 0
    carries no replay - which is why report.py reads it out of config.json
    instead of deriving it from the replay fraction.
    """
    from training.config import REAL

    assert sum(REAL.n_params()) == REAL_N_PARAMETERS
    assert GRID_STEPS_PER_PHASE * 32 * 1024 == GRID_NEW_TOKENS_PER_PHASE
    assert GRID_NEW_TOKENS_PER_PHASE * N_PHASES == GRID_NEW_TOKENS_TOTAL

    # Arm B: 14 replay sequences per batch, on phases 1..6 only.
    replay_total = GRID_STEPS_PER_PHASE * 14 * 1024 * 6
    path = write_run(tmp_path, "B", 0, final_acc=0.4, forgetting=0.3, token_budget={
        "total_new_phase_tokens": GRID_NEW_TOKENS_TOTAL,
        "total_replay_tokens": replay_total,
    })
    assert report.load_run(path).token_ratio == pytest.approx(1.375, abs=5e-4)

    a_path = write_run(tmp_path, "A", 0, final_acc=0.3, forgetting=0.3)
    assert report.load_run(a_path).token_ratio == pytest.approx(1.0)


def test_gpu_hours_across_two_gpu_models_is_an_error(tmp_path):
    write_run(tmp_path, "B", 0, final_acc=0.4, forgetting=0.3, gpu_hours=1.0, gpu_name="Tesla T4")
    write_run(tmp_path, "B", 1, final_acc=0.4, forgetting=0.3, gpu_hours=1.0)
    runs, _ = report.discover_runs(tmp_path)
    with pytest.raises(ValueError, match="different GPU models"):
        report.arm_gpu_hours(runs)


# --------------------------------------------------------------------------- #
# 8. the hypothesis table, both directions
# --------------------------------------------------------------------------- #


def confirming_grid(tmp_path: Path) -> dict[str, list[report.RunRecord]]:
    """Every one of H1-H4 confirmed.

    H1:  Arm A peak phase-0 0.60, final 0.30 -> lost 0.30/0.60 = 50% >= 15%.
    H2a: D forgetting 0.10 < B's 0.30, and E 0.62 - D 0.60 = 2 points <= 3.
    H2b: D 1.4 gpu-h / B 1.0 gpu-h = 1.4x. Reported, never a verdict.
    H3:  D 0.60 - C 0.50 = 10 points >= 5, and 0.10 clears the ~0.005 spread.
    H4:  D-nr forgetting 0.25 - D 0.10 = 0.15, clears the ~0.004 spread.
    """
    write_arm(tmp_path, "A", final_acc=0.30, forgetting=0.30, gpu_hours=1.0)
    write_arm(tmp_path, "B", final_acc=0.40, forgetting=0.30, gpu_hours=1.0)
    write_arm(tmp_path, "C", final_acc=0.50, forgetting=0.20, gpu_hours=1.1)
    write_arm(tmp_path, "D", final_acc=0.60, forgetting=0.10, gpu_hours=1.4)
    write_arm(tmp_path, "D-nr", final_acc=0.45, forgetting=0.25, gpu_hours=1.4)
    write_arm(tmp_path, "E", final_acc=0.62, forgetting=0.02, gpu_hours=1.0)
    runs, excluded = report.discover_runs(tmp_path)
    assert excluded == []
    return report.group_by_arm(runs)


def killing_grid(tmp_path: Path) -> dict[str, list[report.RunRecord]]:
    """Every one of H1-H4 killed.

    H1:  Arm A peak 0.50, final 0.48 -> lost 0.02/0.50 = 4% < 5%.
    H2a: D forgetting 0.30 > B's 0.10 and the gap clears the spread.
    H3:  C and D have the same average accuracy -> no difference -> "C matches D".
    H4:  D-nr and D forget the same amount -> no difference.

    H2b has no kill clause, so it is absent here by construction.
    """
    write_arm(tmp_path, "A", final_acc=0.48, forgetting=0.02, gpu_hours=1.0)
    write_arm(tmp_path, "B", final_acc=0.55, forgetting=0.10, gpu_hours=1.0)
    write_arm(tmp_path, "C", final_acc=0.55, forgetting=0.30, gpu_hours=1.1)
    write_arm(tmp_path, "D", final_acc=0.55, forgetting=0.30, gpu_hours=1.4)
    write_arm(tmp_path, "D-nr", final_acc=0.55, forgetting=0.30, gpu_hours=1.4)
    write_arm(tmp_path, "E", final_acc=0.60, forgetting=0.05, gpu_hours=1.0)
    runs, excluded = report.discover_runs(tmp_path)
    assert excluded == []
    return report.group_by_arm(runs)


def test_every_hypothesis_confirms_on_the_confirming_grid(tmp_path):
    arms = confirming_grid(tmp_path)
    verdicts = {v.name: v for v in report.hypothesis_table(arms, "continuation")}
    assert list(verdicts) == list(report.VERDICT_HYPOTHESES) == ["H1", "H2a", "H3", "H4"]
    assert verdicts["H1"].verdict == report.CONFIRMED
    assert verdicts["H2a"].verdict == report.CONFIRMED
    assert verdicts["H3"].verdict == report.CONFIRMED
    assert verdicts["H4"].verdict == report.CONFIRMED
    assert "50.0%" in verdicts["H1"].measured
    assert "10.00 points" in verdicts["H3"].measured
    # the threshold is quoted from PLAN.md, not restated
    assert verdicts["H1"].confirmed_if == report.H_THRESHOLDS["H1"]["confirmed_if"]
    # H2a's confirm clause no longer mentions GPU time
    assert "GPU time" not in report.H_THRESHOLDS["H2a"]["confirmed_if"]
    assert "GPU time" not in verdicts["H2a"].measured


def test_every_hypothesis_is_killed_on_the_killing_grid(tmp_path):
    arms = killing_grid(tmp_path)
    verdicts = {v.name: v for v in report.hypothesis_table(arms, "continuation")}
    assert verdicts["H1"].verdict == report.KILLED
    assert verdicts["H2a"].verdict == report.KILLED
    assert verdicts["H3"].verdict == report.KILLED
    assert verdicts["H4"].verdict == report.KILLED
    assert "4.0%" in verdicts["H1"].measured
    assert "no difference" in verdicts["H3"].measured
    assert "no difference" in verdicts["H4"].measured


def test_h2a_ignores_gpu_time_entirely(tmp_path):
    """The old H2 would have been killed by a 3x cost; H2a is a retention
    claim and a cost cannot decide it either way."""
    write_arm(tmp_path, "B", final_acc=0.40, forgetting=0.30, gpu_hours=1.0)
    write_arm(tmp_path, "D", final_acc=0.60, forgetting=0.10, gpu_hours=3.0)
    write_arm(tmp_path, "E", final_acc=0.62, forgetting=0.02, gpu_hours=1.0)
    runs, _ = report.discover_runs(tmp_path)
    arms = report.group_by_arm(runs)
    verdict = {v.name: v for v in report.hypothesis_table(arms, "continuation")}["H2a"]
    assert verdict.verdict == report.CONFIRMED
    # ... and the cost is still reported, as a number
    assert report.h2b(arms).value.mean == pytest.approx(3.0)


def test_h2a_says_plainly_when_a_confirm_rests_on_the_spread_rule(tmp_path):
    """H2a can confirm while Arm D's raw forgetting mean is the *higher* of the
    two, when the gap is inside the seed spread. The rule is working as written
    - but the printed line has to say so, not leave the reader to reconcile
    "CONFIRMED" against a number that looks like D losing."""
    # D forgets 0.002 more than B on the mean, with a ~0.004 seed spread.
    write_arm(tmp_path, "B", final_acc=0.40, forgetting=0.300, gpu_hours=1.0)
    write_arm(tmp_path, "D", final_acc=0.60, forgetting=0.302, gpu_hours=1.4)
    write_arm(tmp_path, "E", final_acc=0.62, forgetting=0.02, gpu_hours=1.0)
    runs, _ = report.discover_runs(tmp_path)
    arms = report.group_by_arm(runs)
    verdict = {v.name: v for v in report.hypothesis_table(arms, "continuation")}["H2a"]

    assert verdict.verdict == report.CONFIRMED
    assert "+0.0020" in verdict.measured  # D's raw mean IS the higher one
    assert "within the seed spread" in verdict.measured
    assert "NOT a difference" in verdict.measured
    assert "Arm D's raw mean is the higher of the two" in verdict.measured
    # it reads the same way in the rendered table
    assert "NOT a difference" in report.render_hypothesis_table(arms)


def test_h2a_states_a_real_difference_without_the_spread_caveat(tmp_path):
    arms = confirming_grid(tmp_path)
    verdict = {v.name: v for v in report.hypothesis_table(arms, "continuation")}["H2a"]
    assert "-0.2000" in verdict.measured
    assert "within the seed spread" not in verdict.measured


def test_h2b_reports_a_number_and_a_band_and_never_a_verdict(tmp_path):
    arms = confirming_grid(tmp_path)
    m = report.h2b(arms)
    assert isinstance(m, report.Measurement)
    assert m.name == "H2b"
    # structurally unable to emit a verdict
    assert not hasattr(m, "verdict")
    assert "verdict" not in report.Measurement.__dataclass_fields__
    # H2b carries no confirm/kill thresholds at all
    assert "confirmed_if" not in report.H_THRESHOLDS["H2b"]
    assert "killed_if" not in report.H_THRESHOLDS["H2b"]
    # it is not in the verdict table, and not in the split table
    assert "H2b" not in {v.name for v in report.hypothesis_table(arms, "continuation")}
    assert "H2b" not in {row[0] for row in report.split_hypothesis_table(arms)}
    # the number, with its seed spread, and the band
    assert m.value.mean == pytest.approx(1.4)
    assert m.value.n == 3
    assert m.value.seeds == [0, 1, 2]
    assert "1.40x" in m.measured and "n=3" in m.measured
    assert m.band == "within the <= 1.5x band the original H2 named"
    for word in (report.CONFIRMED, report.KILLED, report.NEITHER, report.SPLIT):
        assert word not in m.band


def test_h2b_band_labels_cover_the_three_ranges(tmp_path):
    assert "within the <= 1.5x" in report._h2b_band(1.0)
    assert "within the <= 1.5x" in report._h2b_band(1.5)
    assert "between the 1.5x and 2x" in report._h2b_band(1.8)
    assert "between the 1.5x and 2x" in report._h2b_band(2.0)
    assert "above the 2x" in report._h2b_band(2.1)

    # the ratio is formed per seed, so it carries a spread rather than being a
    # ratio of two means: B 1.0/1.0/1.0 and D 2.0/2.1/2.2 -> 2.1x +/- 0.1
    for seed, hours in zip(SEEDS, (2.0, 2.1, 2.2)):
        write_run(tmp_path, "D", seed, final_acc=0.6, forgetting=0.1, gpu_hours=hours)
        write_run(tmp_path, "B", seed, final_acc=0.4, forgetting=0.3, gpu_hours=1.0)
    runs, _ = report.discover_runs(tmp_path)
    m = report.h2b(report.group_by_arm(runs))
    assert m.value.mean == pytest.approx(2.1)
    assert m.value.sd == pytest.approx(0.1)
    assert m.band == "above the 2x band the original H2 named"


def test_h2b_still_refuses_to_mix_two_gpu_models(tmp_path):
    write_arm(tmp_path, "B", final_acc=0.40, forgetting=0.30, gpu_hours=1.0)
    write_arm(tmp_path, "D", final_acc=0.60, forgetting=0.10, gpu_hours=1.4,
              gpu_name="Tesla T4")
    runs, _ = report.discover_runs(tmp_path)
    with pytest.raises(ValueError, match="different GPU models"):
        report.h2b(report.group_by_arm(runs))


def test_a_missing_arm_is_undetermined_not_guessed(tmp_path):
    write_arm(tmp_path, "A", final_acc=0.30, forgetting=0.30)
    runs, _ = report.discover_runs(tmp_path)
    arms = report.group_by_arm(runs)
    verdicts = {v.name: v for v in report.hypothesis_table(arms, "continuation")}
    assert verdicts["H1"].verdict == report.CONFIRMED
    for name in ("H2a", "H3", "H4"):
        assert verdicts[name].verdict == report.UNDETERMINED
    # H2b says so rather than inventing a ratio
    m = report.h2b(arms)
    assert m.value is None
    assert m.band == "not measurable"
    assert "no runs for arm(s) B, D" in m.measured


def test_perplexity_is_refused_as_a_verdict_metric(tmp_path):
    arms = confirming_grid(tmp_path)
    for stored_only in ("perplexity", "continuation_summed"):
        with pytest.raises(ValueError):
            report.hypothesis_table(arms, stored_only)


def test_a_disagreement_between_continuation_and_cloze_is_reported_as_split(tmp_path):
    """Arm A's continuation column loses 50% of its peak (H1 confirmed) while
    its cloze column loses only 0.02/0.50 = 4% (H1 killed). Neither is dropped
    and neither is a tiebreaker: the verdict is "split"."""
    write_arm(tmp_path, "A", final_acc=0.30, forgetting=0.30,
              cloze_final_acc=0.48, cloze_forgetting=0.02)
    write_arm(tmp_path, "B", final_acc=0.40, forgetting=0.30, gpu_hours=1.0)
    write_arm(tmp_path, "C", final_acc=0.50, forgetting=0.20, gpu_hours=1.1)
    write_arm(tmp_path, "D", final_acc=0.60, forgetting=0.10, gpu_hours=1.4)
    write_arm(tmp_path, "D-nr", final_acc=0.45, forgetting=0.25, gpu_hours=1.4)
    write_arm(tmp_path, "E", final_acc=0.62, forgetting=0.02, gpu_hours=1.0)
    runs, _ = report.discover_runs(tmp_path)
    arms = report.group_by_arm(runs)

    rows = {name: (combined, head, other) for name, combined, head, other in
            report.split_hypothesis_table(arms)}
    combined, head, other = rows["H1"]
    assert head.exam_type == "continuation" and head.verdict == report.CONFIRMED
    assert other.exam_type == "cloze" and other.verdict == report.KILLED
    assert combined == report.SPLIT
    # the other three agree, so they are not split
    for name in ("H2a", "H3", "H4"):
        assert rows[name][0] != report.SPLIT
    # H2b is not read on an exam type, so it has nothing to disagree about
    assert "H2b" not in rows

    text = report.render_hypothesis_table(arms)
    assert "H1" in text and "SPLIT" in text
    assert "continuation: confirmed" in text
    assert "cloze:        killed" in text
    assert report.NO_DIFFERENCE_RULE in text
    # the thresholds appear verbatim
    assert report.H_THRESHOLDS["H1"]["confirmed_if"] in text
    # H2b renders as a measured row with a band and no verdict column
    assert "H2b What that retention costs -> 1.40x" in text
    assert "not a pass/fail test: no confirmed / killed / split verdict" in text
    h2b_line = [ln for ln in text.splitlines() if ln.startswith("H2b ")][0]
    for word in ("CONFIRMED", "KILLED", "NEITHER", "SPLIT", "UNDETERMINED"):
        assert word not in h2b_line


# --------------------------------------------------------------------------- #
# plots and the end-to-end report
# --------------------------------------------------------------------------- #


def test_plots_are_written_headless(tmp_path):
    arms = confirming_grid(tmp_path / "results")
    out = tmp_path / "out"
    heat = report.plot_heatmaps(arms, "continuation", out)
    curve = report.plot_forgetting_curve(arms, "continuation", out)
    assert heat.is_file() and heat.stat().st_size > 0
    assert curve.is_file() and curve.stat().st_size > 0
    assert heat.name == "heatmap_continuation.png"
    assert curve.name == "forgetting_continuation.png"


def test_write_report_end_to_end(tmp_path):
    results = tmp_path / "results"
    confirming_grid(results)
    write_run(results, "A", 9, final_acc=0.3, forgetting=0.3, dirty=True)
    write_run(results, "Apilot", 0, final_acc=0.3, forgetting=0.3, pilot=True)
    out = tmp_path / "out"
    path = report.write_report(results, out)
    text = path.read_text(encoding="utf-8")
    assert "Excluded runs (2)" in text
    assert "A_s9" in text and "Apilot_s0" in text
    assert "Arm D (3 runs)" in text
    assert "chance: continuation 0.2500, cloze 0.0500" in text
    assert "n=3" in text
    assert "[stored, never headlined]" in text  # perplexity and continuation_summed
    assert "token ratio vs Arm A (recorded in config.json, not recomputed)" in text
    for exam_type in EXAM_TYPES:
        assert (out / ("heatmap_" + exam_type + ".png")).is_file()
        assert (out / ("forgetting_" + exam_type + ".png")).is_file()


# --------------------------------------------------------------------------- #
# subset runs (train.py --phases 0,3,6)
# --------------------------------------------------------------------------- #

SUBSET = (0, 3, 6)


def test_a_subset_run_loads_with_real_ids_and_nan_outside_its_phases(tmp_path):
    import math

    path = write_run(tmp_path, "A", 0, final_acc=0.5, forgetting=0.2, phases=SUBSET)
    run = report.load_run(path)
    assert isinstance(run, report.RunRecord), run
    assert run.phases == SUBSET and run.subset and run.metric_phases == SUBSET
    M = run.matrix["continuation"]
    assert len(M) == N_PHASES and all(len(r) == N_PHASES for r in M)
    assert all(math.isnan(M[i][j]) for i in range(N_PHASES) for j in range(N_PHASES)
               if i not in SUBSET or j not in SUBSET)
    assert M[3][3] == pytest.approx(0.7) and M[6][3] == pytest.approx(0.5)
    assert math.isnan(run.untrained["cloze"][4]) and run.untrained["cloze"][3] == pytest.approx(0.05)


def test_subset_metrics_are_over_the_declared_phases(tmp_path):
    """build_matrix puts final_acc everywhere and final_acc + forgetting on the
    diagonal for j < 6. Over rows/columns 0, 3, 6: last declared row all 0.5
    -> accuracy 0.5; forgetting of 0 and of 3 is 0.2 each -> average 0.2;
    phase 0 lost 0.2 of a 0.7 peak = 0.285714..."""
    write_arm(tmp_path, "A", final_acc=0.5, forgetting=0.2, phases=SUBSET)
    runs, excluded = report.discover_runs(tmp_path)
    assert excluded == [] and len(runs) == 3
    acc = report.arm_average_accuracy(runs, "continuation")
    forg = report.arm_average_forgetting(runs, "continuation")
    lost = report.arm_fraction_of_peak_lost(runs, "continuation", 0)
    # seed jitter: final_acc +/-0.005, forgetting +/-0.004 (write_arm)
    assert acc.values == pytest.approx([0.495, 0.5, 0.505])
    assert forg.values == pytest.approx([0.196, 0.2, 0.204])
    assert lost.values[1] == pytest.approx(0.2 / 0.7)


def test_a_full_run_that_declares_all_seven_reads_exactly_like_one_that_does_not(tmp_path):
    a = report.load_run(write_run(tmp_path / "x", "A", 0, final_acc=0.5, forgetting=0.2))
    b = report.load_run(write_run(tmp_path / "y", "A", 0, final_acc=0.5, forgetting=0.2, phases=range(N_PHASES)))
    assert a.phases == b.phases and not b.subset and b.metric_phases is None
    assert a.matrix == b.matrix and a.untrained == b.untrained
    for t in ("continuation", "cloze"):
        assert report.arm_average_accuracy([a], t).values == report.arm_average_accuracy([b], t).values
        assert report.arm_average_forgetting([a], t).values == report.arm_average_forgetting([b], t).values


def test_a_subset_run_missing_a_declared_cell_is_excluded_by_name(tmp_path):
    path = write_run(tmp_path, "A", 0, final_acc=0.5, forgetting=0.2, phases=SUBSET)
    doc = json.loads((path / "matrix.json").read_text(encoding="utf-8"))
    doc["M"]["cloze"][6][3] = None
    (path / "matrix.json").write_text(json.dumps(doc), encoding="utf-8")
    got = report.load_run(path)
    assert isinstance(got, report.Exclusion)
    assert "row(s) 6 of the declared phases [0, 3, 6]" in got.reason


def test_an_undeclared_row_left_null_is_not_an_incomplete_run(tmp_path):
    """The same null rows that exclude a full run as a dead session are the
    design of a subset run."""
    full = write_run(tmp_path / "full", "A", 0, final_acc=0.5, forgetting=0.2)
    doc = json.loads((full / "matrix.json").read_text(encoding="utf-8"))
    for key in doc["M"]:
        doc["M"][key][1] = [None] * N_PHASES
    (full / "matrix.json").write_text(json.dumps(doc), encoding="utf-8")
    assert isinstance(report.load_run(full), report.Exclusion)
    assert isinstance(
        report.load_run(write_run(tmp_path / "sub", "A", 0, final_acc=0.5, forgetting=0.2, phases=SUBSET)),
        report.RunRecord,
    )


@pytest.mark.parametrize("bad", [[3, 6], [0, 6, 3], [0, 7], "0,3,6"])
def test_an_invalid_phase_list_is_excluded(tmp_path, bad):
    path = write_run(tmp_path, "A", 0, final_acc=0.5, forgetting=0.2)
    cfg = json.loads((path / "config.json").read_text(encoding="utf-8"))
    cfg["phases"] = bad
    (path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    got = report.load_run(path)
    assert isinstance(got, report.Exclusion) and '"phases"' in got.reason


def test_runs_over_different_phase_lists_are_never_compared(tmp_path):
    results = tmp_path / "results"
    write_arm(results, "A", final_acc=0.5, forgetting=0.2, phases=SUBSET)
    write_run(results, "E", 0, final_acc=0.6, forgetting=0.0)  # a full run
    path = report.write_report(results, tmp_path / "out", plots=False)
    text = path.read_text(encoding="utf-8")
    assert "E_s0_abc1234_20260921T000000Z: declares phases [0, 1, 2, 3, 4, 5, 6]" in text
    assert "phases: [0, 3, 6] (subset run" in text
    assert "Arm A (3 runs)" in text and "Arm E (1 runs)" not in text


def test_a_subset_report_writes_its_plots_on_real_phase_ids(tmp_path):
    results = tmp_path / "results"
    write_arm(results, "A", final_acc=0.5, forgetting=0.2, phases=SUBSET)
    write_arm(results, "E", final_acc=0.6, forgetting=0.0, phases=SUBSET)
    out = tmp_path / "out"
    text = report.write_report(results, out).read_text(encoding="utf-8")
    assert "Excluded runs: none." in text
    for exam_type in EXAM_TYPES:
        assert (out / ("heatmap_" + exam_type + ".png")).is_file()
        assert (out / ("forgetting_" + exam_type + ".png")).is_file()
