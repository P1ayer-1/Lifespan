"""Packing, the shared data order, and the replay mix.

Acceptance checks 3, 4 and 5 live here: arms A and B share their batches, a
0.3-replay arm's batch is 32 new + 14 replay drawn only from earlier phases,
and the two arms take the same number of optimiser steps per phase.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from pathlib import Path

from tests.conftest import phase_byte, write_train_dir
from training.config import ARMS, TOY, TrainConfig, replay_sequences_per_batch
from training.data import DataError, DataModule, concat_batches, derive_seed, shift_for_lm
from training.tokenizer import ByteTokenizer

BLOCK = TOY.block_size


def module(train_dir, seed=0, **kw) -> DataModule:
    return DataModule(train_dir, ByteTokenizer(), BLOCK, seed, **kw)


# ---------------------------------------------------------------------------
# packing


def test_every_phase_packs_to_fixed_length_sequences(toy_dirs):
    data = module(toy_dirs["train"])
    assert len(data.phases) == 7
    for k, packed in enumerate(data.phases):
        assert packed.phase == k
        assert packed.n_stories == 20
        assert packed.sequences.shape[1] == BLOCK
        assert len(packed) > 0


def test_the_packed_set_does_not_depend_on_the_seed(toy_dirs):
    """Seed controls order, not the data (PLAN.md: "the data itself is fixed")."""
    a, b = module(toy_dirs["train"], seed=0), module(toy_dirs["train"], seed=2)
    for pa, pb in zip(a.phases, b.phases):
        assert np.array_equal(pa.sequences, pb.sequences)


def test_shift_for_lm_is_the_ordinary_next_token_shift():
    batch = torch.arange(12).reshape(2, 6)
    x, y = shift_for_lm(batch)
    assert x.shape == y.shape == (2, 5)
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_missing_phase_file_is_an_error(tmp_path):
    (tmp_path / "train_phase_0.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(DataError, match="missing training file"):
        module(tmp_path)


def test_derived_seeds_do_not_alias_across_streams_or_seeds():
    tags = ["init", "data.new", "data.replay", "data.joint"]
    values = [derive_seed(s, t) for s in (0, 1, 2) for t in tags]
    assert len(set(values)) == len(values)


# ---------------------------------------------------------------------------
# acceptance check 3: A and B share the new-phase stream


def first_new_batch(train_dir, arm_name: str, seed: int, phase: int, n_new: int) -> torch.Tensor:
    arm = ARMS[arm_name]
    data = module(train_dir, seed=seed)
    steps = data.steps_for_phase(phase, 4, n_new)
    n_replay = replay_sequences_per_batch(n_new, arm.replay_fraction)
    new_loader = data.make_phase_loader(phase, n_new, steps=steps)
    replay_loader = data.make_replay_loader(n_replay, phase=phase, steps=steps)
    return next(iter(new_loader)), next(iter(replay_loader), None)


def test_arms_a_and_b_have_an_identical_first_batch_of_phase_0(toy_dirs):
    """Acceptance check 3. At phase 0 there is no replay in any arm, so the
    batches are identical tensors, not merely the same shape."""
    a_new, a_replay = first_new_batch(toy_dirs["train"], "A", 0, 0, 8)
    b_new, b_replay = first_new_batch(toy_dirs["train"], "B", 0, 0, 8)
    assert torch.equal(a_new, b_new)
    assert a_replay is None and b_replay is None


@pytest.mark.parametrize("phase", [0, 1, 3, 6])
def test_arms_a_and_b_share_the_new_phase_stream_at_every_phase(toy_dirs, phase):
    """Stronger than the acceptance check asks: because replay draws come from
    their own generator, adding replay never shifts the new-phase batches."""
    a_new, _ = first_new_batch(toy_dirs["train"], "A", 1, phase, 8)
    b_new, _ = first_new_batch(toy_dirs["train"], "B", 1, phase, 8)
    assert torch.equal(a_new, b_new)


def test_a_different_seed_changes_the_order(toy_dirs):
    a, _ = first_new_batch(toy_dirs["train"], "A", 0, 0, 8)
    b, _ = first_new_batch(toy_dirs["train"], "A", 1, 0, 8)
    assert not torch.equal(a, b)


# ---------------------------------------------------------------------------
# acceptance check 4: the replay mix


def test_replay_fraction_gives_32_new_plus_14_replay():
    """Acceptance check 4, the arithmetic: round(32 * 0.3 / 0.7) == 14, and
    14 / 46 is 30% of the batch."""
    n_new = TrainConfig().sequences_per_batch
    assert n_new == 32
    n_replay = replay_sequences_per_batch(n_new, ARMS["B"].replay_fraction)
    assert n_replay == 14
    assert n_replay / (n_new + n_replay) == pytest.approx(0.30, abs=0.005)
    assert replay_sequences_per_batch(n_new, ARMS["A"].replay_fraction) == 0


def test_a_replay_batch_is_32_new_and_14_replay_from_earlier_phases_only(toy_dirs):
    """Acceptance check 4, the loader. Provenance is checked from the token
    content, not from the loader's own bookkeeping: phase k's stories are
    written entirely in the letter chr(ord('a') + k)."""
    data = module(toy_dirs["train"])
    n_new, current = 32, 4
    n_replay = replay_sequences_per_batch(n_new, 0.3)
    steps = 3
    new_loader = data.make_phase_loader(current, n_new, steps=steps)
    replay_loader = data.make_replay_loader(n_replay, phase=current, steps=steps, with_provenance=True)

    earlier_bytes = {phase_byte(k) for k in range(current)}
    for new_batch, (replay_batch, provenance) in zip(new_loader, replay_loader):
        assert new_batch.shape == (32, BLOCK)
        assert replay_batch.shape == (14, BLOCK)
        assert concat_batches(new_batch, replay_batch).shape == (46, BLOCK)
        # every replay sequence comes from a phase strictly earlier
        assert provenance.max() < current
        for row in replay_batch.tolist():
            letters = {b for b in row if b not in (0, ord(" "), ord("\n"), ord("."))}
            assert letters and letters <= earlier_bytes
        # and the new rows are this phase's
        for row in new_batch.tolist():
            letters = {b for b in row if b not in (0, ord(" "), ord("\n"), ord("."))}
            assert letters <= {phase_byte(current)}


def test_phase_0_has_no_replay_in_any_arm(toy_dirs):
    """Acceptance check 4, last clause."""
    data = module(toy_dirs["train"])
    for arm in ARMS.values():
        n_replay = replay_sequences_per_batch(32, arm.replay_fraction)
        batches = list(data.make_replay_loader(n_replay, phase=0, steps=5))
        assert batches == []


def test_replay_pool_never_contains_the_current_phase(toy_dirs):
    data = module(toy_dirs["train"])
    for phase in range(1, 7):
        pool = data.replay_pool(phase)
        assert pool.phases.max() == phase - 1
        assert pool.phases.min() == 0


def test_replay_buffer_is_the_selected_stories_only(toy_dirs):
    data = module(toy_dirs["train"])
    full = len(data.phases[0])
    buffered = len(data._replay_pack(0))
    assert 0 < buffered < full, (buffered, full)


def test_a_missing_replay_file_is_an_error_not_a_silent_skip(toy_dirs):
    (toy_dirs["train"] / "replay" / "phase_0.json").unlink()
    data = module(toy_dirs["train"])
    with pytest.raises(DataError, match="replay buffer missing"):
        list(data.make_replay_loader(14, phase=1, steps=1))


def test_a_replay_hash_that_is_not_in_the_training_file_is_an_error(toy_dirs):
    path = toy_dirs["train"] / "replay" / "phase_0.json"
    path.write_text(json.dumps({"seed": 0, "prompt_hashes": ["deadbeef"]}), encoding="utf-8")
    data = module(toy_dirs["train"])
    with pytest.raises(DataError, match="not in train_phase_0.jsonl"):
        list(data.make_replay_loader(14, phase=1, steps=1))


# ---------------------------------------------------------------------------
# acceptance check 5: the same budget


@pytest.mark.parametrize("phase", range(7))
def test_every_arm_takes_the_same_number_of_steps_per_phase(toy_dirs, phase):
    """Acceptance check 5. Steps depend on the data and the new-phase batch
    size only, so no arm can get more of them by turning replay on."""
    data = module(toy_dirs["train"])
    steps = {name: data.steps_for_phase(phase, 4, 32) for name in ARMS}
    assert len(set(steps.values())) == 1


def test_token_budget_records_new_and_replay_separately(toy_dirs):
    data = module(toy_dirs["train"])
    a = data.token_budget(4, 32, ARMS["A"].replay_fraction)
    b = data.token_budget(4, 32, ARMS["B"].replay_fraction)
    assert a["total_replay_tokens"] == 0
    assert b["total_replay_tokens"] > 0
    # Replay is on top: the new-phase budget is identical.
    assert a["total_new_phase_tokens"] == b["total_new_phase_tokens"]
    assert [r["steps"] for r in a["per_phase"]] == [r["steps"] for r in b["per_phase"]]
    # Phase 0 has no replay even in arm B.
    assert b["per_phase"][0]["replay_tokens"] == 0
    assert b["per_phase"][1]["replay_sequences_per_batch"] == 14
    # PLAN.md: "Arm B therefore processes about 1.43x Arm A's tokens".
    ratio = (b["total_new_phase_tokens"] + b["total_replay_tokens"]) / a["total_new_phase_tokens"]
    assert 1.30 < ratio < 1.45


def test_arm_e_sees_arm_a_total(toy_dirs):
    """PLAN.md: "Arm E's 'same total tokens' means Arm A's total"."""
    data = module(toy_dirs["train"])
    a_steps = sum(data.steps_for_phase(k, 4, 32) for k in range(7))
    assert data.total_steps_all_phases(4, 32) == a_steps


def test_joint_loader_mixes_every_phase(toy_dirs):
    data = module(toy_dirs["train"])
    seen = set()
    for batch in data.make_joint_loader(32, steps=6):
        for row in batch.tolist():
            for b in row:
                if ord("a") <= b <= ord("g"):
                    seen.add(b - ord("a"))
    assert seen == set(range(7))


def test_pilot_sizing_takes_a_fixed_prefix_not_a_sample(toy_dirs):
    a = module(toy_dirs["train"], seed=0, stories_per_phase=5)
    b = module(toy_dirs["train"], seed=2, stories_per_phase=5)
    assert a.phases[0].n_stories == 5
    assert np.array_equal(a.phases[0].sequences, b.phases[0].sequences)


# ---------------------------------------------------------------------------
# generator provenance (AGENTS.md, Amendment 2 — the leakage audit)
#
# "One model generates every story, train and exam, all seven phases: a model
# change between phases is a confound the forgetting curve cannot separate from
# forgetting." The generator's own check only ever compared a model id against
# lines in the same output file, and each phase is its own file, so a change
# between phase 3 and phase 4 was caught by nothing and left no trace anywhere.


def test_a_single_model_corpus_records_its_generator(toy_dirs):
    data = module(toy_dirs["train"])
    assert data.generator_model == "synthetic-fixture"
    assert all(p.generator_models == frozenset({"synthetic-fixture"}) for p in data.phases)


def test_a_two_model_corpus_refuses_to_train(tmp_path):
    """The headline case: phases 0-3 from one model, 4-6 from another. Nothing
    crashes downstream and the forgetting curve looks perfectly plausible, so
    this has to be an assertion, not a convention."""
    train = write_train_dir(
        tmp_path / "train", model_by_phase={4: "other-model", 5: "other-model", 6: "other-model"}
    )
    with pytest.raises(DataError) as exc:
        module(train)
    message = str(exc.value)
    assert "generated by 2 different models" in message
    assert "other-model" in message and "synthetic-fixture" in message
    assert "cannot separate from forgetting" in message
    assert "not a flag to override" in message


def test_one_odd_phase_out_of_seven_is_caught(tmp_path):
    """Never a sample: a single changed phase is exactly the case that must be
    caught, and it is the one a spot check would miss."""
    for odd in range(7):
        train = write_train_dir(tmp_path / f"t{odd}", model_by_phase={odd: "sneaky-model"})
        with pytest.raises(DataError, match="2 different models"):
            module(train)


def test_a_story_with_no_model_field_refuses_to_train(tmp_path):
    """Unrecorded provenance is not provenance: it is the exact hole the audit
    found, so it fails closed rather than defaulting to 'probably the same'."""
    train = write_train_dir(tmp_path / "train")
    path = train / "train_phase_2.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[3])
    del rec["model"]
    lines[3] = json.dumps(rec)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(DataError, match="has no 'model' field"):
        module(train)


def test_the_check_costs_no_extra_pass_over_the_corpus(toy_dirs, monkeypatch):
    """Provenance is collected on the pass that encodes the stories, so each
    phase file is opened exactly once."""
    opened: list[str] = []
    real_open = Path.open

    def counting_open(self, *a, **kw):
        if self.name.startswith("train_phase_"):
            opened.append(self.name)
        return real_open(self, *a, **kw)

    monkeypatch.setattr(Path, "open", counting_open)
    data = module(toy_dirs["train"])
    assert data.generator_model
    assert sorted(opened) == sorted(f"train_phase_{k}.jsonl" for k in range(7))


# ---------------------------------------------------------------------------
# split provenance (leakage re-audit 2026-09-25, BLOCKING): a story line that is
# not declared `split: "train"` never reaches the model.


def _rewrite_line(path: Path, index: int, edit) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[index])
    edit(rec)
    lines[index] = json.dumps(rec)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_every_fixture_line_says_train_and_loads(toy_dirs):
    for k in range(7):
        for line in (toy_dirs["train"] / f"train_phase_{k}.jsonl").read_text(encoding="utf-8").splitlines():
            assert json.loads(line)["split"] == "train"
    module(toy_dirs["train"])


@pytest.mark.parametrize("split", ["exam", "test", "", "TRAIN", " train", None])
def test_a_line_whose_split_is_not_train_refuses(tmp_path, split):
    train = write_train_dir(tmp_path / "train")
    path = train / "train_phase_4.jsonl"
    _rewrite_line(path, 7, lambda rec: rec.__setitem__("split", split))
    with pytest.raises(DataError) as exc:
        module(train)
    message = str(exc.value)
    assert f"{path}:8" in message  # the file and the 1-based line
    assert f"has split {split!r}" in message
    assert "not a flag to override" in message


def test_a_line_with_no_split_field_refuses(tmp_path):
    train = write_train_dir(tmp_path / "train")
    path = train / "train_phase_1.jsonl"
    _rewrite_line(path, 0, lambda rec: rec.pop("split"))
    with pytest.raises(DataError) as exc:
        module(train)
    assert f"{path}:1 has no 'split' field" in str(exc.value)


def test_a_mixed_file_refuses_even_when_the_bad_line_is_last(tmp_path):
    """One exam line appended to an otherwise clean train file -- the audit's
    exact attack -- stops the run and names the line."""
    train = write_train_dir(tmp_path / "train")
    path = train / "train_phase_2.jsonl"
    rec = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    rec.update(prompt_hash="appended-exam", split="exam")
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    with pytest.raises(DataError, match=r"train_phase_2\.jsonl:21 has split 'exam'"):
        module(train)


def test_the_split_check_runs_before_the_pilot_prefix_filter(tmp_path):
    """--pilot trains on the first N stories only; an exam line after the
    prefix still means the file is not a training file."""
    train = write_train_dir(tmp_path / "train")
    _rewrite_line(train / "train_phase_0.jsonl", 19, lambda rec: rec.__setitem__("split", "exam"))
    with pytest.raises(DataError, match=r"train_phase_0\.jsonl:20 has split 'exam'"):
        module(train, stories_per_phase=5)
