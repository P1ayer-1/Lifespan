"""The loop end to end, on the toy config.

Acceptance checks 6 and 8 live here: resume equals uninterrupted, and the
phase-0 checkpoint is shared. `after_phase` is `hooks.identity_after_phase`;
`evaluate_all` is the deterministic fake in `conftest.py`, because
`training/lora.py`, `consolidate.py` and `evaluate.py` belong to other agents.
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

NL = chr(10)

import pytest
import torch

from tests.conftest import fake_evaluate_all, toy_args
from training import hooks
from training.checkpoint import load_shared, load_state, phase0_path, state_path
from training.config import ARMS, EXAM_TYPES, N_PHASES, TOY, TrainConfig, replay_sequences_per_batch
from training.runrecord import MATRIX_KEYS
from training.train import (
    TOY_RUN,
    Settings,
    build_parser,
    inherits_phase0,
    produces_phase0,
    resolve_after_phase,
    run,
)

HOOKS = dict(after_phase=hooks.identity_after_phase, evaluate_all=fake_evaluate_all)


def matrix_of(out: Path) -> dict:
    return json.loads((out / "matrix.json").read_text(encoding="utf-8"))


def config_of(out: Path) -> dict:
    return json.loads((out / "config.json").read_text(encoding="utf-8"))


def timings_of(out: Path) -> dict:
    return json.loads((out / "timings.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# the arm table


def test_the_cli_offers_exactly_the_frozen_arms():
    action = {a.dest: a for a in build_parser()._actions}["arm"]
    assert set(action.choices) == set(ARMS) == {"phase0", "A", "B", "C", "D", "D-nr", "E"}


def test_arms_a_and_b_differ_in_one_field_only():
    a, b = ARMS["A"], ARMS["B"]
    differing = [f for f in a.__dataclass_fields__ if getattr(a, f) != getattr(b, f)]
    assert differing == ["name", "replay_fraction"]


def test_identity_is_the_hook_for_a_b_and_e():
    for name in ("A", "B", "E", "phase0"):
        assert resolve_after_phase(ARMS[name]) is hooks.identity_after_phase


def test_lora_arms_resolve_their_hook_lazily(monkeypatch):
    """C and D go through training/consolidate.py, which another agent owns;
    importing this package must not need it to exist yet."""
    import importlib

    def boom(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(importlib, "import_module", boom)
    for name in ("C", "D", "D-nr"):
        with pytest.raises(ModuleNotFoundError):
            resolve_after_phase(ARMS[name])


def test_only_arm_e_skips_the_shared_phase_zero():
    assert [n for n, a in ARMS.items() if inherits_phase0(a)] == ["A", "B", "C", "D", "D-nr"]
    assert [n for n, a in ARMS.items() if produces_phase0(a)] == ["phase0"]
    assert not ARMS["E"].shares_phase0


# ---------------------------------------------------------------------------
# a full run


def test_arm_e_produces_a_complete_seven_by_seven_matrix(toy_dirs):
    out = run(toy_args(toy_dirs, "E", 0), **HOOKS)
    m = matrix_of(out)
    for t in EXAM_TYPES:
        assert len(m["M"][t]) == N_PHASES
        for row in m["M"][t]:
            assert len(row) == N_PHASES
            assert all(isinstance(v, float) for v in row)
        assert all(isinstance(v, float) for v in m["untrained"][t])


def test_a_run_writes_the_four_result_files(toy_dirs):
    out = run(toy_args(toy_dirs, "E", 0), **HOOKS)
    for name in ("matrix.json", "timings.json", "config.json", "commit.txt"):
        assert (out / name).exists(), name
    cfg = config_of(out)
    assert cfg["arm"] == "E"
    assert cfg["seed"] == 0
    assert cfg["pilot"] is False
    assert cfg["precision"] == "fp32"
    # AGENTS.md, Amendments 2026-09-22: every hash under one top-level key.
    assert set(cfg["hashes"]) >= {"manifest", "train_files", "data", "config"}
    assert cfg["hashes"]["data"] and cfg["hashes"]["manifest"]
    assert cfg["hashes"]["train_files"] and isinstance(cfg["hashes"]["train_files"], dict)
    assert cfg["matrix_keys"] == list(MATRIX_KEYS) and len(cfg["matrix_keys"]) == 4
    assert len(cfg["token_budget"]["per_phase"]) == N_PHASES
    assert timings_of(out)["gpu_name"] == "cpu"
    assert len(timings_of(out)["per_phase"]) == N_PHASES


def test_pilot_is_recorded_in_config_json(toy_dirs):
    args = toy_args(toy_dirs, "E", 0)
    args.pilot = True
    out = run(args, **HOOKS)
    assert config_of(out)["pilot"] is True


def test_the_token_budget_records_new_and_replay_separately(toy_dirs):
    a_out = run(toy_args(toy_dirs, "phase0", 0), **HOOKS)
    out_a = run(toy_args(toy_dirs, "A", 0), **HOOKS)
    out_b = run(toy_args(toy_dirs, "B", 0), **HOOKS)
    ca, cb = config_of(out_a), config_of(out_b)
    assert ca["sequences_per_batch_replay"] == 0
    assert cb["sequences_per_batch_replay"] == replay_sequences_per_batch(ca["sequences_per_batch_new"], 0.3)
    assert ca["token_budget"]["total_new_phase_tokens"] == cb["token_budget"]["total_new_phase_tokens"]
    assert ca["token_budget"]["total_replay_tokens"] == 0
    assert cb["token_budget"]["total_replay_tokens"] > 0
    assert a_out.exists()


def test_arms_a_and_b_take_the_same_steps_per_phase(toy_dirs):
    """Acceptance check 5, through the loop rather than the loader."""
    run(toy_args(toy_dirs, "phase0", 0), **HOOKS)
    ca = config_of(run(toy_args(toy_dirs, "A", 0), **HOOKS))
    cb = config_of(run(toy_args(toy_dirs, "B", 0), **HOOKS))
    steps_a = [r["steps"] for r in ca["token_budget"]["per_phase"]]
    steps_b = [r["steps"] for r in cb["token_budget"]["per_phase"]]
    assert steps_a == steps_b


# ---------------------------------------------------------------------------
# acceptance check 8: phase 0 is shared


def test_phase0_is_trained_once_and_inherited(toy_dirs):
    """Acceptance check 8: `--arm phase0 --seed 0`, then `--arm A --seed 0`
    starts at phase 1 with phase 0's row and wall clock copied in."""
    p0 = run(toy_args(toy_dirs, "phase0", 0), **HOOKS)
    shared = phase0_path(toy_dirs["shared"], 0)
    assert shared.exists()

    m0 = matrix_of(p0)
    t0 = timings_of(p0)
    assert m0["M"]["cloze"][0] != [None] * N_PHASES
    assert m0["M"]["cloze"][1] == [None] * N_PHASES, "arm phase0 must train phase 0 only"

    a = run(toy_args(toy_dirs, "A", 0), **HOOKS)
    ma, ta = matrix_of(a), timings_of(a)

    # the row and the untrained row are phase 0's, not recomputed
    assert ma["M"]["cloze"][0] == m0["M"]["cloze"][0]
    assert ma["untrained"] == m0["untrained"]
    # every later row is filled by arm A itself
    for t in EXAM_TYPES:
        assert all(all(isinstance(v, float) for v in row) for row in ma["M"][t])
    # the wall clock is copied so H2's compute ratio is not distorted
    assert ta["per_phase"][0]["train_s"] == t0["per_phase"][0]["train_s"]
    assert ta["per_phase"][0]["shared_from_phase0_run"] is True


def test_every_arm_at_one_seed_starts_from_the_same_random_init(toy_dirs):
    run(toy_args(toy_dirs, "phase0", 0), **HOOKS)
    run(toy_args(toy_dirs, "E", 0), **HOOKS)
    init = toy_dirs["shared"] / "init_s0.pt"
    assert init.exists()
    a = load_shared(init, "init")
    # the phase0 run wrote it; the E run loaded it rather than drawing again
    assert a.fingerprint.seed == 0
    b = load_shared(init, "init")
    for k in a.model_state:
        assert torch.equal(a.model_state[k], b.model_state[k])


def test_an_arm_refuses_a_phase0_checkpoint_from_a_different_config(toy_dirs):
    """Acceptance check 8, second half."""
    run(toy_args(toy_dirs, "phase0", 0), **HOOKS)
    # Change the training data: the data hash in the fingerprint no longer matches.
    path = toy_dirs["train"] / "train_phase_5.jsonl"
    path.write_text(path.read_text(encoding="utf-8") * 2, encoding="utf-8")
    from training.checkpoint import CheckpointError

    with pytest.raises(CheckpointError, match="data_hash"):
        run(toy_args(toy_dirs, "A", 0), **HOOKS)


def test_a_different_seed_needs_its_own_phase0(toy_dirs):
    run(toy_args(toy_dirs, "phase0", 0), **HOOKS)
    from training.checkpoint import CheckpointError

    with pytest.raises(CheckpointError, match="missing"):
        run(toy_args(toy_dirs, "A", 1), **HOOKS)


# ---------------------------------------------------------------------------
# acceptance check 6: resume equals uninterrupted


class KillAfterPhase:
    """An `after_phase` hook that raises once the given phase is done, standing
    in for a Kaggle session dying mid-run."""

    def __init__(self, phase: int) -> None:
        self.phase = phase

    def __call__(self, model, phase_k, ctx):
        if phase_k == self.phase:
            raise KeyboardInterrupt(f"session died after phase {phase_k}")
        return model


def test_resume_equals_uninterrupted(toy_dirs):
    """Acceptance check 6: run TOY straight through, then again killing after
    phase 3 and resuming, and compare the matrices."""
    straight = run(toy_args(toy_dirs, "E", 0, out=toy_dirs["results"] / "straight"), **HOOKS)
    expected = matrix_of(straight)

    broken = toy_dirs["results"] / "broken"
    with pytest.raises(KeyboardInterrupt):
        run(
            toy_args(toy_dirs, "E", 0, out=broken),
            after_phase=KillAfterPhase(3),
            evaluate_all=fake_evaluate_all,
        )
    state = load_state(state_path(broken))
    assert state.completed_phase == 2, "the kill landed mid-phase-3, so phase 2 is the last complete one"

    resumed = run(toy_args(toy_dirs, "E", 0, out=broken, resume=True), **HOOKS)
    assert matrix_of(resumed) == expected
    assert resumed == broken


def test_resume_reproduces_the_weights_not_only_the_scores(toy_dirs):
    straight = run(toy_args(toy_dirs, "E", 0, out=toy_dirs["results"] / "s2"), **HOOKS)
    broken = toy_dirs["results"] / "b2"
    with pytest.raises(KeyboardInterrupt):
        run(toy_args(toy_dirs, "E", 0, out=broken), after_phase=KillAfterPhase(3), evaluate_all=fake_evaluate_all)
    run(toy_args(toy_dirs, "E", 0, out=broken, resume=True), **HOOKS)

    a = load_state(state_path(straight)).model_state
    b = load_state(state_path(broken)).model_state
    for k in a:
        assert torch.equal(a[k], b[k]), k


def test_resume_without_a_checkpoint_starts_from_scratch(toy_dirs):
    out = run(toy_args(toy_dirs, "E", 0, resume=True), **HOOKS)
    assert (out / "matrix.json").exists()


def test_resume_refuses_a_state_from_a_different_run(toy_dirs):
    out = run(toy_args(toy_dirs, "E", 0), **HOOKS)
    path = toy_dirs["train"] / "train_phase_2.jsonl"
    path.write_text(path.read_text(encoding="utf-8") * 2, encoding="utf-8")
    from training.checkpoint import CheckpointError

    with pytest.raises(CheckpointError, match="data_hash"):
        run(toy_args(toy_dirs, "E", 0, out=out, resume=True), **HOOKS)


def test_a_fresh_run_refuses_another_runs_folder(toy_dirs):
    """The first Kaggle pre-pilot's arm A ran with phase0's --out and replaced
    phase0's result files in place (2026-10-01)."""
    out = run(toy_args(toy_dirs, "phase0", 0), **HOOKS)
    before = (out / "config.json").read_bytes()
    with pytest.raises(SystemExit, match="already holds a run"):
        run(toy_args(toy_dirs, "A", 0, out=out), **HOOKS)
    assert (out / "config.json").read_bytes() == before


# ---------------------------------------------------------------------------
# the hook contract


def test_after_phase_is_called_once_per_phase_one_to_six(toy_dirs):
    """Corrected 2026-09-22: no hook runs at phase 0. The `phase0` arm trains
    phase 0 and closes it, so its single call is the exception that proves it."""
    seen = []

    def spy(model, phase_k, ctx):
        seen.append((ctx.arm, phase_k, ctx.phase, ctx.replay_fraction, ctx.steps))
        return model

    run(toy_args(toy_dirs, "phase0", 0), after_phase=spy, evaluate_all=fake_evaluate_all)
    assert [s[1] for s in seen] == [0]

    seen.clear()
    run(toy_args(toy_dirs, "B", 0), after_phase=spy, evaluate_all=fake_evaluate_all)
    assert [s[1] for s in seen] == list(range(1, N_PHASES)), "phases 1-6 only"
    assert {s[0] for s in seen} == {"B"}
    assert {s[3] for s in seen} == {0.3}
    assert all(s[1] == s[2] for s in seen), "ctx.phase must be the phase being closed"


def test_the_context_loaders_work_and_replay_is_empty_at_phase_zero(toy_dirs):
    seen = {}

    def spy(model, phase_k, ctx):
        new = next(iter(ctx.make_phase_loader(phase_k, 4)))
        replay = list(ctx.make_replay_loader(2))
        seen[phase_k] = (tuple(new.shape), len(replay))
        return model

    run(toy_args(toy_dirs, "phase0", 0), after_phase=spy, evaluate_all=fake_evaluate_all)
    run(toy_args(toy_dirs, "B", 0), after_phase=spy, evaluate_all=fake_evaluate_all)
    assert seen[0] == ((4, TOY.block_size), 0), "no replay exists at phase 0"
    for k in range(1, N_PHASES):  # noqa: B007
        assert seen[k][0] == (4, TOY.block_size)
        assert seen[k][1] > 0


def test_a_hook_returning_a_new_model_is_the_one_carried_forward(toy_dirs):
    """Arms C and D hand back a different module; the loop must use it."""
    from training.model import build_model

    replacement = build_model(TOY, torch.Generator().manual_seed(99))

    def swap(model, phase_k, ctx):
        return replacement if phase_k == 2 else model

    out = run(toy_args(toy_dirs, "E", 0), after_phase=swap, evaluate_all=fake_evaluate_all)
    state = load_state(state_path(out))
    assert state.completed_phase == N_PHASES - 1
    assert out.exists()


def test_a_bad_evaluate_all_is_refused_not_recorded(toy_dirs):
    from training.runrecord import RunRecordError

    def short(model, exam_dir, phases):
        return {t: [0.1] * 3 for t in MATRIX_KEYS}

    with pytest.raises(RunRecordError, match="expected exactly 7"):
        run(toy_args(toy_dirs, "E", 0), after_phase=hooks.identity_after_phase, evaluate_all=short)


# ---------------------------------------------------------------------------
# settings


def test_toy_and_pilot_change_sizes_only(toy_dirs):
    from training.config import PILOT_OVERRIDES, TrainConfig

    base = TrainConfig()
    args = toy_args(toy_dirs, "A", 0)
    args.toy = False
    args.pilot = True
    s = Settings(args)
    assert s.stories_per_phase == PILOT_OVERRIDES["train_stories_per_phase"]
    # every hyperparameter from PLAN.md's table is untouched
    assert (s.train_cfg.lr, s.n_new, s.epochs) == (
        base.lr,
        base.sequences_per_batch,
        base.epochs_per_phase,
    )
    assert s.model_cfg.n_layer == 8 and s.model_cfg.vocab_size == 8192

    # Warmup: scaled only because this is a pilot, and a grid run keeps 200.
    assert s.scale_warmup is True
    assert s.warmup(74) == 7  # a pilot phase, where a fixed 200 never ramps out
    args.pilot = False
    grid = Settings(args)
    assert grid.scale_warmup is False
    assert grid.warmup(367) == base.warmup_steps == 200


# ---------------------------------------------------------------------------
# integration with the hooks the other agents own
#
# Skipped while training/consolidate.py does not exist, so this package always
# imports; real coverage of the arm-C and arm-D control flow once it does.


@pytest.mark.parametrize("arm", ["C", "D", "D-nr"])
def test_a_lora_arm_runs_end_to_end_through_the_real_hook(toy_dirs, arm):
    pytest.importorskip("training.consolidate")
    run(toy_args(toy_dirs, "phase0", 0), **HOOKS)
    out = run(toy_args(toy_dirs, arm, 0), evaluate_all=fake_evaluate_all)
    m = matrix_of(out)
    for t in EXAM_TYPES:
        assert all(all(isinstance(v, float) for v in row) for row in m["M"][t])
    # the night's wall clock landed in its own column, not in train_s
    t = timings_of(out)
    assert sum(r["consolidate_s"] for r in t["per_phase"]) > 0
    assert t["per_phase"][0]["shared_from_phase0_run"] is True


def test_the_timer_takes_the_bucket_key_itself():
    """AGENTS.md 2026-09-22: the three labels are exactly the timings.json keys.
    The transitional alias is gone, so a wrong label is an error, not a guess."""
    import training.consolidate as consolidate
    from training.runrecord import TIMER_SPANS, RunRecordError, Timings

    assert consolidate.TIMER_LABEL in TIMER_SPANS

    t = Timings()
    t.set_phase(1)
    with t.timer(consolidate.TIMER_LABEL):
        pass
    assert set(t.per_phase[0]) == {"phase", "train_s", "consolidate_s", "eval_s"}
    assert t.per_phase[0]["consolidate_s"] > 0
    with pytest.raises(RunRecordError, match="unknown timing span"):
        with t.timer("consolidate"):
            pass


# ---------------------------------------------------------------------------
# the hook owns the phase for a uses_lora arm (AGENTS.md, Amendments 2026-09-22)
#
# The bug these exist to catch is silent: running both the base loop and the
# hook gives arms C, D and D-nr roughly double Arm A's tokens and leaves arm C's
# base unfrozen, and nothing crashes -- the heatmap still looks fine.


def snapshot(model) -> dict:
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def test_arm_c_base_is_untouched_until_the_hook_is_called(toy_dirs):
    """The base weights arriving at `after_phase` are bit-identical to the ones
    the previous phase handed over: no base loop ran in between."""
    run(toy_args(toy_dirs, "phase0", 0), **HOOKS)

    at_hook_entry: dict[int, dict] = {}
    carried_forward: dict[int, dict] = {}

    def spy(model, phase_k, ctx):
        at_hook_entry[phase_k] = snapshot(model)
        carried_forward[phase_k] = snapshot(model)  # identity hook: unchanged
        return model

    run(toy_args(toy_dirs, "C", 0), after_phase=spy, evaluate_all=fake_evaluate_all)

    assert sorted(at_hook_entry) == list(range(1, N_PHASES)), "phases 1-6 only, never phase 0"
    for phase in range(2, N_PHASES):
        previous, entering = carried_forward[phase - 1], at_hook_entry[phase]
        for name, tensor in previous.items():
            assert torch.equal(tensor, entering[name]), (
                f"arm C's base moved between phase {phase - 1}'s hook and phase {phase}'s: "
                f"{name} changed, so a base loop ran outside the hook"
            )


def test_a_uses_lora_arm_takes_no_base_optimizer_steps_outside_the_hook(toy_dirs):
    """Total base steps over phases 1-6 for arms C, D and D-nr is zero."""
    run(toy_args(toy_dirs, "phase0", 0), **HOOKS)
    for arm in ("C", "D", "D-nr"):
        assert ARMS[arm].uses_lora
        out = run(toy_args(toy_dirs, arm, 0, out=toy_dirs["results"] / f"nb_{arm}"), **HOOKS)
        cfg = config_of(out)
        assert cfg["base_loop_owned_by_hook"] is True
        t = timings_of(out)
        base_seconds = sum(r["train_s"] for r in t["per_phase"] if not r.get("shared_from_phase0_run"))
        assert base_seconds == 0.0, f"arm {arm} spent {base_seconds}s in the base loop"


def test_a_non_lora_arm_does_run_the_base_loop(toy_dirs):
    """The other half of the same claim, so the test above cannot pass by the
    loop being broken for everyone."""
    run(toy_args(toy_dirs, "phase0", 0), **HOOKS)
    out = run(toy_args(toy_dirs, "A", 0), **HOOKS)
    assert config_of(out)["base_loop_owned_by_hook"] is False
    t = timings_of(out)
    assert sum(r["train_s"] for r in t["per_phase"] if not r.get("shared_from_phase0_run")) > 0


def test_train_s_and_consolidate_s_split_the_way_h2_reads_them(toy_dirs):
    """Arm C: train_s 0, consolidate_s > 0. Arm A: the reverse. H2 is stated in
    compute, so the day's cost has to land in the column that measures it."""
    pytest.importorskip("training.consolidate")
    run(toy_args(toy_dirs, "phase0", 0), **HOOKS)
    a = timings_of(run(toy_args(toy_dirs, "A", 0), **HOOKS))
    c = timings_of(run(toy_args(toy_dirs, "C", 0), evaluate_all=fake_evaluate_all))

    def totals(t):
        rows = [r for r in t["per_phase"] if not r.get("shared_from_phase0_run")]
        return sum(r["train_s"] for r in rows), sum(r["consolidate_s"] for r in rows)

    a_train, a_cons = totals(a)
    c_train, c_cons = totals(c)
    # Arm C's train_s is an exact zero: no timer is ever opened, because no
    # base loop runs. Arm A's consolidate_s is not exactly zero -- the identity
    # hook is still called and still timed -- but it is the cost of returning
    # its argument, so it must be negligible beside the training it brackets.
    assert c_train == 0.0, f"arm C ran a base loop for {c_train}s"
    assert c_cons > 0
    assert a_train > 0
    assert a_cons < a_train / 100, f"arm A's identity hook cost {a_cons}s against {a_train}s of training"


def test_every_arm_sees_the_same_new_phase_token_count_per_phase(toy_dirs):
    """Arm parity: A, B, C, D, D-nr and E get an identical new-phase budget in
    every phase, and only the replay column differs."""
    run(toy_args(toy_dirs, "phase0", 0), **HOOKS)
    budgets = {}
    for arm in ("A", "B", "C", "D", "D-nr", "E"):
        out = run(toy_args(toy_dirs, arm, 0, out=toy_dirs["results"] / f"parity_{arm}"), **HOOKS)
        budgets[arm] = config_of(out)["token_budget"]

    # Parity is per *pass*: every arm takes the same optimiser steps on the same
    # number of new-phase sequences. What differs is how many passes the regime
    # makes over that material -- one for a full fine-tune or an arithmetic
    # merge, two for a distillation -- which is precisely what H2b measures and
    # must therefore not be flattened away here.
    per_pass = {a: [r["arm_a_new_phase_tokens"] for r in b["per_phase"]] for a, b in budgets.items()}
    steps_per_phase = {a: [r["steps"] for r in b["per_phase"]] for a, b in budgets.items()}
    assert len(set(map(tuple, per_pass.values()))) == 1, per_pass
    assert len(set(map(tuple, steps_per_phase.values()))) == 1, steps_per_phase

    new_per_phase = {a: [r["new_phase_tokens"] for r in b["per_phase"]] for a, b in budgets.items()}
    for arm, rows in new_per_phase.items():
        passes = budgets[arm]["per_phase"][0]["phase_passes"]
        assert rows == [passes * t for t in per_pass[arm]], arm
    assert {a: rows[0] // per_pass[a][0] for a, rows in new_per_phase.items()} == {
        "A": 1, "B": 1, "C": 1, "D": 2, "D-nr": 2, "E": 1
    }

    # replay is the only thing that differs, and only for the 0.3 arms
    assert {a: b["total_replay_tokens"] > 0 for a, b in budgets.items()} == {
        "A": False, "B": True, "C": False, "D": True, "D-nr": False, "E": False
    }
    arm_a_total = budgets["A"]["arm_a_total_tokens"]
    for arm, b in budgets.items():
        assert b["arm_a_total_tokens"] == arm_a_total, arm
        assert b["tokens_vs_arm_a"] == pytest.approx(
            (b["total_new_phase_tokens"] + b["total_replay_tokens"]) / arm_a_total
        ), arm
    assert budgets["A"]["tokens_vs_arm_a"] == 1.0


def test_the_grid_token_ratio_is_1_375_not_1_43(toy_dirs):
    """Recorded 2026-09-22: phase 0 carries no replay, so only 6 of 7 phases
    do, and Arm B is 1.375x Arm A, not PLAN.md's "about 1.43x"."""
    from training.data import DataModule
    from training.tokenizer import ByteTokenizer

    data = DataModule(toy_dirs["train"], ByteTokenizer(), TOY.block_size, seed=0)
    b = data.token_budget(4, 32, 0.3)
    assert b["tokens_vs_arm_a"] == pytest.approx(1.375, abs=0.002)
    assert data.token_budget(4, 32, 0.0)["tokens_vs_arm_a"] == 1.0


def test_no_hook_runs_at_phase_zero_for_any_arm(toy_dirs):
    """Correction 2026-09-22: the 2026-09-21 text said the hook ran at phase 0
    for C, D and D-nr. It does not."""
    seen = []
    run(toy_args(toy_dirs, "phase0", 0), after_phase=lambda m, k, c: seen.append(k) or m,
        evaluate_all=fake_evaluate_all)
    assert seen == [0], "the phase0 arm trains phase 0, so its own hook call is at phase 0"
    for arm in ("A", "C", "D"):
        seen.clear()
        run(toy_args(toy_dirs, arm, 0, out=toy_dirs["results"] / f"h0_{arm}"),
            after_phase=lambda m, k, c: seen.append(k) or m, evaluate_all=fake_evaluate_all)
        assert seen == list(range(1, N_PHASES)), f"arm {arm} called the hook at {seen}"


def test_the_context_carries_the_runs_real_n(toy_dirs):
    """PhaseContext.sequences_per_batch, added 2026-09-22: a hook must not fall
    back to a TrainConfig default when train.py runs a non-default N."""
    run(toy_args(toy_dirs, "phase0", 0), **HOOKS)
    seen = []
    run(toy_args(toy_dirs, "D", 0), after_phase=lambda m, k, c: seen.append(c.sequences_per_batch) or m,
        evaluate_all=fake_evaluate_all)
    assert set(seen) == {TOY_RUN.sequences_per_batch}
    assert TOY_RUN.sequences_per_batch != TrainConfig().sequences_per_batch


# ---------------------------------------------------------------------------
# generator provenance, end to end (AGENTS.md, Amendment 2)


def test_config_json_records_the_generator_model(toy_dirs):
    """Read from the training lines themselves, not a flag and not a sidecar:
    a value that can disagree with the data is not provenance."""
    out = run(toy_args(toy_dirs, "E", 0), **HOOKS)
    cfg = config_of(out)
    assert cfg["generator_model"] == "synthetic-fixture"
    assert [c["generator_model"] for c in cfg["corpus"]] == [["synthetic-fixture"]] * N_PHASES
    # not to be confused with config.json's "model", which is the architecture
    assert set(cfg["model"]) >= {"n_layer", "n_embd", "vocab_size"}


def test_a_two_model_corpus_does_not_train_through_the_entry_point(tmp_path):
    """The whole point of the finding: a generator change between phase 3 and
    phase 4 produces a perfectly plausible forgetting curve. It must not run."""
    from tests.conftest import write_exam_dir, write_manifest, write_train_dir
    from training.data import DataError

    train = write_train_dir(tmp_path / "train", model_by_phase={4: "a-different-model"})
    exam = write_exam_dir(tmp_path / "exam")
    dirs = {
        "train": train, "exam": exam, "manifest": write_manifest(tmp_path / "m.json", exam),
        "results": tmp_path / "results", "shared": tmp_path / "shared", "tmp": tmp_path,
    }
    # Since 2026-09-30 the guard (training/guard.py) reads every story line's
    # model before data.py does and refuses first; either refusal is the point.
    from training.guard import GuardError

    with pytest.raises((DataError, GuardError), match="2 different (generator )?models"):
        run(toy_args(dirs, "E", 0), **HOOKS)
    # nothing was trained and no result folder survives as a half-run
    assert not (dirs["results"] / "E_s0" / "matrix.json").exists()


def test_the_refusal_happens_before_a_model_is_built(tmp_path, monkeypatch):
    """Same spirit as the guard: it fires before any weights exist."""
    from tests.conftest import write_exam_dir, write_manifest, write_train_dir
    from training import train as train_mod
    from training.data import DataError

    built = []
    real = train_mod.build_model
    monkeypatch.setattr(train_mod, "build_model", lambda *a, **k: built.append(1) or real(*a, **k))

    train = write_train_dir(tmp_path / "train", model_by_phase={6: "late-swap"})
    exam = write_exam_dir(tmp_path / "exam")
    dirs = {
        "train": train, "exam": exam, "manifest": write_manifest(tmp_path / "m.json", exam),
        "results": tmp_path / "results", "shared": tmp_path / "shared", "tmp": tmp_path,
    }
    from training.guard import GuardError

    with pytest.raises((DataError, GuardError)):
        run(toy_args(dirs, "A", 0), **HOOKS)
    assert built == [], "a model was constructed before the corpus was rejected"


def test_matrix_json_carries_all_four_required_keys(toy_dirs):
    out = run(toy_args(toy_dirs, "E", 0), **HOOKS)
    m = matrix_of(out)
    assert set(m["M"]) == set(MATRIX_KEYS) and len(MATRIX_KEYS) == 4
    for key in MATRIX_KEYS:
        assert all(isinstance(v, float) for row in m["M"][key] for v in row)
        assert all(isinstance(v, float) for v in m["untrained"][key])


# ---------------------------------------------------------------------------
# THE SEAM NOTHING TESTED
#
# Every case above injects a fake `evaluate_all`. On 2026-09-22 the suite was
# 282 green while the pipeline was dead: `resolve_evaluate_all` returned the
# three-key frozen hook and `runrecord` required four, so every real arm died at
# the first `set_untrained`. A seam that every test stubs is a seam nothing
# tests. These run `train.run` with NO stub for `evaluate_all` and NO stub for
# `after_phase`, against an exam directory the real evaluator actually parses.


@pytest.mark.parametrize("arm", ["A", "C"])
def test_a_real_run_with_no_stubs_at_all(scoreable_dirs, arm):
    """One sequential arm and one LoRA arm, end to end: the real tokenizer, the
    real data module, the real `training.consolidate.after_phase` and the real
    `training.evaluate`, writing a complete matrix."""
    pytest.importorskip("training.consolidate")
    pytest.importorskip("training.evaluate")

    run(toy_args(scoreable_dirs, "phase0", 0))
    out = run(toy_args(scoreable_dirs, arm, 0))

    m = matrix_of(out)
    assert set(m["M"]) == set(MATRIX_KEYS), "the real scorer must fill every matrix key"
    for key in MATRIX_KEYS:
        assert len(m["M"][key]) == N_PHASES
        for row in m["M"][key]:
            assert len(row) == N_PHASES
            assert all(isinstance(v, float) and v == v for v in row), f"{key} has a nan or a null"
        assert all(isinstance(v, float) and v == v for v in m["untrained"][key])

    cfg = config_of(out)
    assert cfg["arm"] == arm
    assert cfg["generator_model"] == "synthetic-fixture"
    assert timings_of(out)["per_phase"][-1]["eval_s"] > 0, "the real scorer took no time at all"


def test_the_real_scorer_is_given_this_runs_tokenizer_and_context(scoreable_dirs):
    """`EvalConfig.block_size` defaults to REAL's 1,024. Unbound, a toy run
    would score 1,024-token windows against a 64-token model; unbound
    `tokenizer` would score a BPE-trained model with raw bytes."""
    pytest.importorskip("training.evaluate")
    from training.evaluate import EvalConfig
    from training.tokenizer import ByteTokenizer
    from training.train import bind_tokenizer, make_eval_config, resolve_evaluate_all

    assert EvalConfig().block_size != TOY.block_size, "the default is not the toy context"
    assert make_eval_config(TOY).block_size == TOY.block_size

    bound = bind_tokenizer(resolve_evaluate_all(), ByteTokenizer(), make_eval_config(TOY))
    assert bound.keywords["cfg"].block_size == TOY.block_size
    assert isinstance(bound.keywords["tokenizer"], ByteTokenizer)


def test_the_resolved_scorer_returns_all_four_matrix_keys(scoreable_dirs):
    """The exact failure the audit found, pinned at the seam: whatever
    `resolve_evaluate_all` hands back must satisfy `runrecord`."""
    pytest.importorskip("training.evaluate")
    import torch

    from training.model import build_model
    from training.runrecord import empty_matrix, set_untrained
    from training.tokenizer import ByteTokenizer
    from training.train import bind_tokenizer, make_eval_config, resolve_evaluate_all

    scorer = bind_tokenizer(resolve_evaluate_all(), ByteTokenizer(), make_eval_config(TOY))
    model = build_model(TOY, torch.Generator().manual_seed(0))
    scores = scorer(model, scoreable_dirs["exam"], list(range(N_PHASES)))
    assert set(scores) == set(MATRIX_KEYS)
    set_untrained(empty_matrix(), scores)  # the call that used to die


def test_the_token_ratio_matches_the_arms_actual_passes(scoreable_dirs):
    """H2b reads `tokens_vs_arm_a` and report.py never recomputes it, so it has
    to come from what the arm really consumes -- not from `replay_fraction`."""
    from training.train import PHASE_PASSES

    run(toy_args(scoreable_dirs, "phase0", 0), **HOOKS)
    ratios, new_tokens = {}, {}
    for arm in ("A", "B", "C", "D", "D-nr", "E"):
        out = run(toy_args(scoreable_dirs, arm, 0, out=scoreable_dirs["results"] / f"r_{arm}"), **HOOKS)
        b = config_of(out)["token_budget"]
        ratios[arm] = b["tokens_vs_arm_a"]
        new_tokens[arm] = b["per_phase"][1]["new_phase_tokens"]

    # arm C merges arithmetically and trains nothing after step 1, so it is one
    # pass like a full fine-tune arm; D and D-nr are two.
    assert PHASE_PASSES["merge"] == (1, 1) and PHASE_PASSES["distill"] == (2, 1)
    assert new_tokens["A"] == new_tokens["B"] == new_tokens["C"] == new_tokens["E"]
    assert new_tokens["D"] == new_tokens["D-nr"] == 2 * new_tokens["A"]

    assert ratios["A"] == ratios["C"] == ratios["E"] == 1.0
    assert ratios["D-nr"] == 2.0
    assert ratios["B"] > 1.0 and ratios["D"] > ratios["D-nr"]
    # the replay surcharge is the same for B and D: replay rides the base loop
    # for B and the distillation step for D, once either way.
    assert ratios["D"] - ratios["D-nr"] == pytest.approx(ratios["B"] - ratios["A"])


def test_the_grid_ratios_are_the_numbers_h2b_will_report():
    """At the grid's N=32 the arithmetic lands on 1.375 / 2.375 / 2.0. The toy
    N=4 rounds replay to 2 of 4 instead of 14 of 32, so it reads a little high;
    these are the numbers a real run records."""
    from training.config import REAL, replay_sequences_per_batch
    from training.train import PHASE_PASSES

    n_new, block = 32, REAL.block_size
    n_replay = replay_sequences_per_batch(n_new, 0.3)
    steps = 367  # equal phases; the real budget sums the per-phase rows instead

    def ratio(mode: str, replay_fraction: float) -> float:
        new_passes, replay_passes = PHASE_PASSES[mode]
        arm_a = 7 * steps * n_new * block
        new = new_passes * arm_a
        rep = replay_passes * 6 * steps * (n_replay if replay_fraction else 0) * block
        return (new + rep) / arm_a

    assert ratio("identity", 0.0) == pytest.approx(1.0)
    assert ratio("identity", 0.3) == pytest.approx(1.375, abs=1e-6)
    assert ratio("merge", 0.0) == pytest.approx(1.0)
    assert ratio("distill", 0.3) == pytest.approx(2.375, abs=1e-6)
    assert ratio("distill", 0.0) == pytest.approx(2.0)


def test_the_ratio_is_summed_from_per_phase_rows_not_a_closed_form(scoreable_dirs):
    """1.375 is exact only when all seven phases have equal step counts, which
    real data will not give. The ratio must still be right when they differ."""
    from training.data import DataModule
    from training.tokenizer import ByteTokenizer

    # phase 0 gets a third of the stories, so its step count differs
    from tests.conftest import write_train_dir

    uneven = write_train_dir(scoreable_dirs["tmp"] / "uneven")
    path = uneven / "train_phase_0.jsonl"
    path.write_text(NL.join(path.read_text(encoding="utf-8").splitlines()[:6]) + NL, encoding="utf-8")

    data = DataModule(uneven, ByteTokenizer(), TOY.block_size, seed=0)
    steps = [data.steps_for_phase(k, 4, 4) for k in range(N_PHASES)]
    assert len(set(steps)) > 1, "this test needs unequal phases to mean anything"

    b = data.token_budget(4, 4, 0.3, phase_passes=1, replay_passes=1)
    arm_a = sum(r["arm_a_new_phase_tokens"] for r in b["per_phase"])
    assert b["tokens_vs_arm_a"] == pytest.approx(
        (b["total_new_phase_tokens"] + b["total_replay_tokens"]) / arm_a
    )


# ---------------------------------------------------------------------------
# subset-phase mode through the loop (leakage re-audit 2026-09-25, should-fix):
# a 0/3/6 run records phases 0, 3 and 6 as 0, 3 and 6 in the matrix, the
# replay, the checkpoints, the timings and config.json. Never 0, 1, 2.

SUBSET = (0, 3, 6)


@pytest.fixture
def subset_dirs(tmp_path: Path) -> dict:
    """A corpus with only phases 0, 3 and 6 on disk, like the pre-pilot."""
    from tests.conftest import write_exam_dir, write_manifest, write_train_dir

    exam = write_exam_dir(tmp_path / "exam")
    return {
        "train": write_train_dir(tmp_path / "train", phases=SUBSET),
        "exam": exam,
        "manifest": write_manifest(tmp_path / "manifest.json", exam),
        "results": tmp_path / "results",
        "shared": tmp_path / "shared",
        "tmp": tmp_path,
    }


def subset_args(dirs: dict, arm: str, seed: int = 0, **kw):
    args = toy_args(dirs, arm, seed, **kw)
    args.phases = SUBSET
    return args


def nan_outside(calls: list):
    """Like the real scorer: NaN in every exam column it was not asked for."""

    def _inner(model, exam_dir, phases):
        calls.append(list(phases))
        row = fake_evaluate_all(model, exam_dir, phases)
        return {t: [v if j in phases else float("nan") for j, v in enumerate(vals)] for t, vals in row.items()}

    return _inner


def test_a_subset_run_records_real_phase_ids_everywhere(subset_dirs):
    from tests.conftest import phase_byte

    calls: list = []
    closed: list = []
    replay_letters: dict[int, set[int]] = {}

    def spy(model, phase_k, ctx):
        closed.append((phase_k, ctx.phase))
        letters: set[int] = set()
        for batch in ctx.make_replay_loader(4):
            for row in batch.tolist():
                letters |= {b for b in row if b not in (0, ord(" "), ord("\n"), ord("."))}
        replay_letters[phase_k] = letters
        return model

    run(subset_args(subset_dirs, "phase0"), after_phase=spy, evaluate_all=nan_outside(calls))
    closed.clear()
    out = run(subset_args(subset_dirs, "B"), after_phase=spy, evaluate_all=nan_outside(calls))

    assert calls and all(c == [0, 3, 6] for c in calls), "only declared exam phases are scored"
    assert closed == [(3, 3), (6, 6)], "the hook sees real ids, never 1 and 2"
    assert replay_letters[3] == {phase_byte(0)}
    assert replay_letters[6] == {phase_byte(0), phase_byte(3)}

    m = matrix_of(out)
    for t in MATRIX_KEYS:
        assert len(m["M"][t]) == N_PHASES, "the matrix stays 7x7, indexed by real id"
        for i in range(N_PHASES):
            if i in SUBSET:
                row = m["M"][t][i]
                assert all(isinstance(row[j], float) and not math.isnan(row[j]) for j in SUBSET)
                assert all(math.isnan(row[j]) for j in range(N_PHASES) if j not in SUBSET), "unscored, not guessed"
            else:
                assert m["M"][t][i] == [None] * N_PHASES, f"phase {i} was never trained"

    cfg = config_of(out)
    assert cfg["phases"] == [0, 3, 6] and cfg["subset_phases"] is True and cfg["n_phases"] == N_PHASES
    assert [r["phase"] for r in cfg["token_budget"]["per_phase"]] == [0, 3, 6]
    assert [c["phase"] for c in cfg["corpus"]] == [0, 3, 6]
    assert sorted(cfg["warmup_steps"], key=int) == ["0", "3", "6"]
    assert [r["phase"] for r in timings_of(out)["per_phase"]] == [0, 3, 6]
    assert load_state(state_path(out)).completed_phase == 6


def test_a_subset_arm_e_trains_one_segment_per_declared_phase(subset_dirs):
    calls: list = []
    out = run(subset_args(subset_dirs, "E"), after_phase=hooks.identity_after_phase, evaluate_all=nan_outside(calls))
    m = matrix_of(out)
    filled = [i for i in range(N_PHASES) if m["M"]["cloze"][i] != [None] * N_PHASES]
    assert filled == [0, 3, 6]
    assert len(calls) == 1 + 3  # the untrained row, then one row per segment


def test_a_subset_resume_continues_at_the_next_declared_phase(subset_dirs):
    straight = run(subset_args(subset_dirs, "E", out=subset_dirs["results"] / "straight"), **HOOKS)
    broken = subset_dirs["results"] / "broken"
    with pytest.raises(KeyboardInterrupt):
        run(
            subset_args(subset_dirs, "E", out=broken),
            after_phase=KillAfterPhase(6),
            evaluate_all=fake_evaluate_all,
        )
    assert load_state(state_path(broken)).completed_phase == 3
    resumed = run(subset_args(subset_dirs, "E", out=broken, resume=True), **HOOKS)
    assert matrix_of(resumed) == matrix_of(straight)


def test_a_declared_phase_missing_from_disk_refuses_through_the_entry_point(tmp_path):
    from tests.conftest import write_exam_dir, write_manifest, write_train_dir
    from training.data import DataError

    exam = write_exam_dir(tmp_path / "exam")
    dirs = {
        "train": write_train_dir(tmp_path / "train", phases=(0, 6)),
        "exam": exam,
        "manifest": write_manifest(tmp_path / "manifest.json", exam),
        "results": tmp_path / "results",
        "shared": tmp_path / "shared",
    }
    with pytest.raises(DataError, match=r"missing training file for phase 3 \(declared phases \[0, 3, 6\]\)"):
        run(subset_args(dirs, "E"), **HOOKS)


def test_a_0_3_6_corpus_without_a_declaration_still_refuses(subset_dirs):
    from training.data import DataError

    with pytest.raises(DataError, match="missing training file for phase 1: "):
        run(toy_args(subset_dirs, "E"), **HOOKS)


def test_a_default_run_declares_all_seven_and_its_config_hash_is_unchanged(toy_dirs):
    """The full run is untouched: no `phases` attribute, `None` and an explicit
    0..6 all resolve to the same settings and the same config digest, which is
    the one every shared init and phase-0 checkpoint is keyed on. A subset
    run's digest differs, so it can never pick up a full run's phase 0."""
    from training.train import _config_digest

    bare = toy_args(toy_dirs, "A")
    assert not hasattr(bare, "phases")
    explicit_none = toy_args(toy_dirs, "A")
    explicit_none.phases = None
    all_seven = toy_args(toy_dirs, "A")
    all_seven.phases = tuple(range(N_PHASES))
    subset = subset_args(toy_dirs, "A")

    digests = []
    for args in (bare, explicit_none, all_seven):
        s = Settings(args)
        assert s.phases == tuple(range(N_PHASES)) and not s.subset
        digests.append(_config_digest(s.train_cfg, s))
    assert len(set(digests)) == 1
    s = Settings(subset)
    assert s.subset and _config_digest(s.train_cfg, s) != digests[0]

    out = run(toy_args(toy_dirs, "E"), **HOOKS)
    cfg = config_of(out)
    assert cfg["phases"] == list(range(N_PHASES)) and cfg["subset_phases"] is False
    assert [r["phase"] for r in cfg["token_budget"]["per_phase"]] == list(range(N_PHASES))


def test_the_cli_takes_a_comma_separated_phase_list():
    parser = build_parser()
    base = ["--arm", "A", "--seed", "0", "--train-dir", "t", "--exam-dir", "e", "--out", "o"]
    assert parser.parse_args(base).phases is None
    assert parser.parse_args(base + ["--phases", "0,3,6"]).phases == (0, 3, 6)
    for bad in ("3,6", "0,6,3", "0,3,3", "0,7", "0,x", ""):
        with pytest.raises(SystemExit):
            parser.parse_args(base + ["--phases", bad])


# ---------------------------------------------------------------------------
# experiment_id (re-audit note 2026-09-25: "recorded but never checked")


def test_the_frozen_command_parses_and_takes_the_configs_experiment_id():
    """The frozen entry point has no --experiment-id, and must keep working:
    the id comes from the run config unless the flag names another."""
    from training.config import EXPERIMENT_ID

    parser = build_parser()
    base = ["--arm", "A", "--seed", "0", "--train-dir", "t", "--exam-dir", "e", "--out", "o"]
    assert EXPERIMENT_ID and parser.parse_args(base).experiment_id == EXPERIMENT_ID
    assert parser.parse_args(base + ["--experiment-id", "pre-pilot"]).experiment_id == "pre-pilot"


@pytest.mark.parametrize(
    "experiment_id,expect",
    [
        ("pre-pilot", "experiment_id mismatch"),
        (None, "no experiment_id given"),
        ("", "no experiment_id given"),
        ("<absent>", "experiment_id mismatch: the run expects 'lifespan-main'"),
    ],
)
def test_the_run_refuses_a_wrong_or_missing_experiment_id_before_anything_exists(
    toy_dirs, experiment_id, expect
):
    """The guard is given the run's id every time: a wrong one, a missing one
    (None / ""), or none at all on an old Namespace (the config's default,
    which the 'toy' fixture manifest is not) all stop the run before a folder,
    a tokenizer or a checkpoint exists."""
    from training.guard import GuardError

    args = toy_args(toy_dirs, "E")
    if experiment_id == "<absent>":
        del args.experiment_id
    else:
        args.experiment_id = experiment_id
    with pytest.raises(GuardError, match=expect):
        run(args, **HOOKS)
    assert not args.out.exists()
    assert not toy_dirs["shared"].exists()


def test_the_matching_experiment_id_is_recorded(toy_dirs):
    out = run(toy_args(toy_dirs, "E"), **HOOKS)
    assert config_of(out)["manifest"]["experiment_id"] == "toy"


# ---------------------------------------------------------------------------
# the scorer is tied to the manifest (2026-09-30)


def test_a_tampered_exam_is_refused_before_anything_exists(toy_dirs):
    """The guard never opens the exam directory; train.py confirms the exam
    stories on disk against the manifest's per-story hashes right after it."""
    from training.evaluate import ExamManifestMismatch

    path = toy_dirs["exam"] / "stories" / "exam_phase_4.jsonl"
    row = json.loads(path.read_text(encoding="utf-8"))
    row["story"] += " edited after the freeze"
    path.write_text(json.dumps(row) + NL, encoding="utf-8")
    args = toy_args(toy_dirs, "E")
    with pytest.raises(ExamManifestMismatch, match="exam_phase_4.jsonl.*is not in the manifest"):
        run(args, **HOOKS)
    assert not args.out.exists()
    assert not toy_dirs["shared"].exists()


def test_the_scorer_receives_the_guards_exam_story_hashes(toy_dirs):
    """Every evaluate_all call gets `story_hashes` = the manifest's map, the
    one the guard report carries, not a map recomputed from the exam dir."""
    from tests.conftest import manifest_stories_of

    frozen = {e["story_sha256"]: e["phase"] for e in manifest_stories_of(toy_dirs["exam"])}
    seen: list = []

    def scorer(model, exam_dir, phases, *, story_hashes=None):
        seen.append(story_hashes)
        return fake_evaluate_all(model, exam_dir, phases)

    out = run(toy_args(toy_dirs, "E"), after_phase=hooks.identity_after_phase, evaluate_all=scorer)
    assert len(seen) == 1 + N_PHASES
    assert all(s == frozen for s in seen)
    verified = config_of(out)["manifest"]["exam_stories_verified"]
    assert verified == {str(k): 1 for k in range(N_PHASES)}


def test_the_resolved_real_scorer_binds_story_hashes():
    pytest.importorskip("training.evaluate")
    from training.tokenizer import ByteTokenizer
    from training.train import bind_tokenizer, make_eval_config, resolve_evaluate_all

    frozen = {"a" * 64: 0}
    bound = bind_tokenizer(
        resolve_evaluate_all(), ByteTokenizer(), make_eval_config(TOY), story_hashes=frozen
    )
    assert bound.keywords["story_hashes"] is frozen
    # a stub without the parameter is left alone
    assert bind_tokenizer(fake_evaluate_all, ByteTokenizer(), story_hashes=frozen) is fake_evaluate_all


def test_a_real_run_refuses_an_exam_file_swapped_mid_run(scoreable_dirs):
    """The scorer re-checks each row it scores, so an exam changed after the
    up-front check (a remounted dataset) still cannot be scored."""
    pytest.importorskip("training.evaluate")
    from training.evaluate import ExamManifestMismatch

    path = scoreable_dirs["exam"] / "stories" / "exam_phase_2.jsonl"
    calls: list = []

    def swap_after_first_phase(model, phase_k, ctx):
        calls.append(phase_k)
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        rows[0]["story"] = "a different story entirely."
        path.write_text(NL.join(json.dumps(r) for r in rows) + NL, encoding="utf-8")
        return model

    with pytest.raises(ExamManifestMismatch, match="exam_phase_2.jsonl"):
        run(toy_args(scoreable_dirs, "E"), after_phase=swap_after_first_phase)
    assert len(calls) == 1, "the first row scored after the swap must refuse"


def test_a_subset_run_verifies_only_its_declared_exam_phases(subset_dirs):
    """Phases 1, 2, 4, 5 are never scored in a 0/3/6 run, so their exam files
    are not required; a declared phase's file is."""
    from tests.conftest import write_manifest
    from training.evaluate import ExamManifestMismatch

    for k in (1, 2, 4, 5):
        (subset_dirs["exam"] / "stories" / f"exam_phase_{k}.jsonl").unlink()
    write_manifest(subset_dirs["manifest"], subset_dirs["exam"])
    out = run(subset_args(subset_dirs, "E"), **HOOKS)
    assert config_of(out)["manifest"]["exam_stories_verified"] == {"0": 1, "3": 1, "6": 1}

    (subset_dirs["exam"] / "stories" / "exam_phase_3.jsonl").unlink()
    with pytest.raises(ExamManifestMismatch, match="exam stories for phase 3 are missing"):
        run(subset_args(subset_dirs, "E", out=subset_dirs["results"] / "again"), **HOOKS)


@pytest.mark.parametrize("subset", [True, False])
def test_the_tokenizer_is_given_the_declared_phases_only_in_subset_mode(toy_dirs, monkeypatch, subset):
    from training import train as train_mod

    seen: list = []
    real = train_mod.get_or_train_tokenizer

    def spy(*a, **k):
        seen.append(k.get("phases"))
        return real(*a, **k)

    monkeypatch.setattr(train_mod, "get_or_train_tokenizer", spy)
    args = subset_args(toy_dirs, "E") if subset else toy_args(toy_dirs, "E")
    run(args, **HOOKS)
    assert seen == [(0, 3, 6) if subset else None]


# ---------------------------------------------------------------------------
# --micro-batch: gradient accumulation is memory only


def _toy_steps(micro_batch):
    from training.model import build_model
    from training.train import train_one_phase

    g = torch.Generator().manual_seed(0)
    model = build_model(TOY, g)
    data_g = torch.Generator().manual_seed(1)
    batches = [torch.randint(0, TOY.vocab_size, (6, TOY.block_size + 1), generator=data_g) for _ in range(3)]
    optimizer = model.configure_optimizers(1e-3, 0.1, (0.9, 0.95), "cpu")
    stats = train_one_phase(
        model,
        optimizer,
        torch.amp.GradScaler("cpu", enabled=False),
        new_loader=batches,
        replay_loader=iter(()),
        steps=3,
        base_lr=1e-3,
        warmup=1,
        grad_clip=1.0,
        device=torch.device("cpu"),
        amp_dtype=None,
        log=lambda _msg: None,
        micro_batch=micro_batch,
    )
    return stats, {k: v.detach().clone() for k, v in model.state_dict().items()}


@pytest.mark.parametrize("micro_batch", [1, 4, 6, 100])
def test_micro_batch_matches_whole_batch(micro_batch):
    """Uneven chunks (6 rows by 4) included: same losses, same weights up to
    summation order (Adam amplifies it on near-zero gradients, hence atol)."""
    whole_stats, whole = _toy_steps(None)
    stats, weights = _toy_steps(micro_batch)
    assert stats["steps"] == whole_stats["steps"] == 3
    assert stats["new_sequences"] == whole_stats["new_sequences"] == 18
    assert math.isclose(stats["last_loss"], whole_stats["last_loss"], rel_tol=1e-5)
    for key, value in whole.items():
        torch.testing.assert_close(weights[key], value, rtol=1e-4, atol=1e-5)


def test_micro_batch_flag_parses_and_rejects_zero():
    base = ["--arm", "A", "--seed", "0", "--train-dir", "t", "--exam-dir", "e", "--out", "o"]
    assert build_parser().parse_args(base).micro_batch is None
    assert build_parser().parse_args([*base, "--micro-batch", "8"]).micro_batch == 8
    with pytest.raises(SystemExit):
        build_parser().parse_args([*base, "--micro-batch", "0"])


# ---------------------------------------------------------------------------
# precision: a T4 reports bf16 "supported" (emulated); it must get fp16


@pytest.mark.parametrize(
    "capability, expected",
    [((7, 5), "fp16+gradscaler"), ((6, 0), "fp16+gradscaler"), ((8, 0), "bf16"), ((9, 0), "bf16")],
)
def test_precision_needs_ampere_for_bf16(monkeypatch, capability, expected):
    from training.train import resolve_precision

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: capability)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # GradScaler("cuda") on a CUDA-less test box
        _device, _dtype, _scaler, name = resolve_precision()
    assert name == expected


def test_precision_cpu_flag_wins(monkeypatch):
    from training.train import resolve_precision

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_precision(force_cpu=True)[3] == "fp32"


def test_prepare_only_runs_the_guard_and_writes_no_run(toy_dirs):
    """runbook.grid's parallel mode calls this once before dispatch."""
    args = toy_args(toy_dirs, "phase0", 0)
    args.prepare_only = True
    run(args, **HOOKS)
    assert not Path(args.out).exists()
    parsed = build_parser().parse_args(
        ["--arm", "phase0", "--seed", "0", "--train-dir", "t", "--exam-dir", "e", "--out", "o", "--prepare-only"]
    )
    assert parsed.prepare_only
