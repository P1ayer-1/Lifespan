"""Tests for `runbook/grid.py`'s ordering, dirty-tree refusal, skip and
resume logic. Not under `tests/` (that directory's `conftest.py` is other
agents' in-flight work); run directly:

    micromamba run -n lifespan python -m pytest runbook/test_grid.py -q

No GPU, no real training: every `subprocess.run` call is monkeypatched out, so
this never invokes `training.train` at all.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from runbook import grid, ledger


def test_grid_invocations_are_the_frozen_21_in_order():
    invocations = grid.grid_invocations((0, 1, 2))
    assert len(invocations) == 21
    assert invocations == [
        ("phase0", 0), ("A", 0), ("B", 0), ("C", 0), ("D", 0), ("D-nr", 0), ("E", 0),
        ("phase0", 1), ("A", 1), ("B", 1), ("C", 1), ("D", 1), ("D-nr", 1), ("E", 1),
        ("phase0", 2), ("A", 2), ("B", 2), ("C", 2), ("D", 2), ("D-nr", 2), ("E", 2),
    ]


def test_phase0_precedes_every_arm_that_shares_it_within_its_seed():
    invocations = grid.grid_invocations((0, 1, 2))
    for seed in (0, 1, 2):
        idx = {arm: i for i, (arm, s) in enumerate(invocations) if s == seed}
        for arm in ("A", "B", "C", "D", "D-nr"):
            assert idx["phase0"] < idx[arm]


def test_refuses_a_dirty_tree(monkeypatch, tmp_path):
    monkeypatch.setattr(grid, "git_is_dirty", lambda repo_root=grid.REPO_ROOT: True)
    with pytest.raises(RuntimeError, match="dirty"):
        grid.run_grid(train_dir=tmp_path / "train", exam_dir=tmp_path / "exam", results_root=tmp_path / "results")


@pytest.fixture
def clean_tree(monkeypatch):
    monkeypatch.setattr(grid, "git_is_dirty", lambda repo_root=grid.REPO_ROOT: False)
    monkeypatch.setattr(grid, "git_head_commit7", lambda repo_root=grid.REPO_ROOT: "abc1234")


def test_skips_an_already_valid_folder(monkeypatch, tmp_path, clean_tree):
    results_root = tmp_path / "results"
    results_root.mkdir(parents=True)
    existing = results_root / "phase0_s0_abc1234_20260101T000000Z"
    existing.mkdir()

    monkeypatch.setattr(grid.VALIDATE, "validate", lambda path: [] if path == existing else ["fabricated"])
    calls = []
    monkeypatch.setattr(grid.subprocess, "run", lambda *a, **k: calls.append(a) or pytest.fail("must not run a subprocess for a valid existing folder"))

    reports = grid.run_grid(
        train_dir=tmp_path / "train",
        exam_dir=tmp_path / "exam",
        results_root=results_root,
        state_path=tmp_path / "state.json",
        ledger_path=tmp_path / "ledger.json",
        dry_run=True,  # the other 20 invocations are dry-run so this test is about the ONE skip
    )
    first = reports[0]
    assert first == {"arm": "phase0", "seed": 0, "action": "skip", "status": "already_complete", "run_dir": str(existing)}
    assert not calls


def test_dry_run_prints_and_records_no_ledger_rows(monkeypatch, tmp_path, clean_tree):
    monkeypatch.setattr(grid.VALIDATE, "validate", lambda path: ["no such folder"])
    reports = grid.run_grid(
        train_dir=tmp_path / "train",
        exam_dir=tmp_path / "exam",
        results_root=tmp_path / "results",
        state_path=tmp_path / "state.json",
        ledger_path=tmp_path / "ledger.json",
        dry_run=True,
    )
    assert len(reports) == 21
    assert all(r["action"] == "dry_run" for r in reports)
    assert ledger.load(tmp_path / "ledger.json") == []


def _fake_run_factory(outcomes: dict):
    """outcomes: {(arm, seed): "complete"|"failed"|"invalid"} -- writes a
    minimal timings.json for "complete"/"invalid" so gpu_hours can be read,
    and returns the exit code `run_grid` expects."""

    def _fake_run(argv, cwd=None):
        arm = argv[argv.index("--arm") + 1]
        seed = int(argv[argv.index("--seed") + 1])
        out_dir = Path(argv[argv.index("--out") + 1])
        outcome = outcomes.get((arm, seed), "complete")
        if outcome == "failed":
            return type("R", (), {"returncode": 1})()
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "timings.json").write_text(json.dumps({"gpu_hours": 0.01, "gpu_name": "cpu"}), encoding="utf-8")
        return type("R", (), {"returncode": 0})()

    return _fake_run


def test_a_failed_phase0_blocks_its_seeds_dependent_arms_but_not_e(monkeypatch, tmp_path, clean_tree):
    monkeypatch.setattr(grid.subprocess, "run", _fake_run_factory({("phase0", 0): "failed"}))
    monkeypatch.setattr(grid.VALIDATE, "validate", lambda path: [] if path.exists() else ["missing"])
    reports = grid.run_grid(
        train_dir=tmp_path / "train",
        exam_dir=tmp_path / "exam",
        results_root=tmp_path / "results",
        state_path=tmp_path / "state.json",
        ledger_path=tmp_path / "ledger.json",
        seeds=(0,),
    )
    by_arm = {r["arm"]: r for r in reports}
    assert by_arm["phase0"]["status"] == "failed"
    for arm in ("A", "B", "C", "D", "D-nr"):
        assert by_arm[arm]["status"] == "blocked_on_phase0"
        assert by_arm[arm]["action"] == "skip"
    assert by_arm["E"]["status"] == "complete"  # E does not depend on phase0


def test_resumable_after_being_killed_halfway(monkeypatch, tmp_path, clean_tree):
    """First call completes only phase0 and A for seed 0 (simulating a kill);
    a second call with the same state/ledger paths must not redo them and
    must complete the rest."""
    state_path = tmp_path / "state.json"
    ledger_path = tmp_path / "ledger.json"
    results_root = tmp_path / "results"

    ran_first = []
    def fake_run_1(argv, cwd=None):
        arm = argv[argv.index("--arm") + 1]
        ran_first.append(arm)
        if arm in ("phase0", "A"):
            out_dir = Path(argv[argv.index("--out") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "timings.json").write_text(json.dumps({"gpu_hours": 0.01, "gpu_name": "cpu"}), encoding="utf-8")
            return type("R", (), {"returncode": 0})()
        raise KeyboardInterrupt("simulated session death")

    monkeypatch.setattr(grid.VALIDATE, "validate", lambda path: [] if path.exists() else ["missing"])
    monkeypatch.setattr(grid.subprocess, "run", fake_run_1)
    with pytest.raises(KeyboardInterrupt):
        grid.run_grid(
            train_dir=tmp_path / "train", exam_dir=tmp_path / "exam", results_root=results_root,
            state_path=state_path, ledger_path=ledger_path, seeds=(0,),
        )
    assert ran_first == ["phase0", "A", "B"]  # B is attempted and raises before completing

    ran_second = []
    def fake_run_2(argv, cwd=None):
        arm = argv[argv.index("--arm") + 1]
        ran_second.append(arm)
        out_dir = Path(argv[argv.index("--out") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "timings.json").write_text(json.dumps({"gpu_hours": 0.01, "gpu_name": "cpu"}), encoding="utf-8")
        return type("R", (), {"returncode": 0})()

    monkeypatch.setattr(grid.subprocess, "run", fake_run_2)
    reports = grid.run_grid(
        train_dir=tmp_path / "train", exam_dir=tmp_path / "exam", results_root=results_root,
        state_path=state_path, ledger_path=ledger_path, seeds=(0,),
    )
    # phase0 and A were already valid on disk from the first (interrupted) call, so
    # the second call must skip them, not rerun them.
    assert "phase0" not in ran_second
    assert "A" not in ran_second
    assert set(ran_second) == {"B", "C", "D", "D-nr", "E"}
    statuses = {r["arm"]: r["status"] for r in reports}
    assert all(s == "complete" or s == "already_complete" for s in statuses.values())


def test_the_experiment_id_is_passed_through_only_when_given(tmp_path):
    """Omitted, train.py falls back to training.config.EXPERIMENT_ID (the
    grid's); given, it reaches the frozen CLI verbatim (2026-09-30)."""
    common = dict(
        arm="A", seed=0, train_dir=tmp_path / "t", exam_dir=tmp_path / "e", out_dir=tmp_path / "o",
        phase0_dir=tmp_path / "p", resume=False, manifest=None, toy=True, cpu=True,
    )
    assert "--experiment-id" not in grid.build_train_argv(**common)
    argv = grid.build_train_argv(**common, experiment_id="pre-pilot")
    assert argv[argv.index("--experiment-id") + 1] == "pre-pilot"
    assert grid.build_parser().parse_args(
        ["--train-dir", "t", "--exam-dir", "e", "--experiment-id", "x"]
    ).experiment_id == "x"
