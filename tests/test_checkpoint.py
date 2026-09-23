"""Checkpoints: atomic writes, RNG restore, and refusing a foreign fingerprint."""

from __future__ import annotations

import torch

import pytest

from training.checkpoint import (
    CheckpointError,
    Fingerprint,
    init_path,
    load_shared,
    load_state,
    phase0_path,
    require_compatible,
    restore_rng_state,
    rng_state,
    save_shared,
    save_state,
    seed_everything,
    state_path,
)
from training.config import TOY
from training.model import build_model
from training.runrecord import Timings, empty_matrix


def fingerprint(**kw) -> Fingerprint:
    base = dict(
        commit="a" * 40,
        dirty=False,
        seed=0,
        model_config={"n_layer": 2},
        data_hash="d" * 64,
        manifest_sha256="m" * 64,
        tokenizer_sha256="t" * 64,
        train_config_hash="c" * 64,
        pilot=False,
    )
    base.update(kw)
    return Fingerprint(**base)


def test_state_round_trips(tmp_path):
    model = build_model(TOY, torch.Generator().manual_seed(0))
    opt = model.configure_optimizers(3e-4, 0.1, (0.9, 0.95))
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    gens = {"new": torch.Generator().manual_seed(7)}
    matrix = empty_matrix()
    timings = Timings()
    timings.set_phase(0)

    path = state_path(tmp_path)
    save_state(
        path,
        model=model,
        optimizer=opt,
        scaler=scaler,
        generators=gens,
        completed_phase=2,
        matrix=matrix,
        timings=timings.to_dict(),
        fingerprint=fingerprint(),
    )
    state = load_state(path)
    assert state.completed_phase == 2
    assert state.fingerprint == fingerprint()
    assert set(state.model_state) == set(model.state_dict())


def test_the_write_is_atomic_and_leaves_no_temp_file(tmp_path):
    model = build_model(TOY, torch.Generator().manual_seed(0))
    path = tmp_path / "sub" / "state.pt"
    save_state(
        path,
        model=model,
        optimizer=None,
        scaler=None,
        generators={},
        completed_phase=0,
        matrix=empty_matrix(),
        timings={},
        fingerprint=fingerprint(),
    )
    assert path.exists()
    assert [p.name for p in path.parent.iterdir()] == ["state.pt"]


def test_rng_state_round_trips_exactly():
    seed_everything(3)
    g = torch.Generator().manual_seed(11)
    state = rng_state({"new": g})
    first = [torch.rand(4, generator=g) for _ in range(3)]
    restore_rng_state(state, {"new": g})
    second = [torch.rand(4, generator=g) for _ in range(3)]
    assert all(torch.equal(a, b) for a, b in zip(first, second))


def test_restoring_without_a_generator_the_checkpoint_knows_is_an_error():
    g = torch.Generator().manual_seed(1)
    state = rng_state({"new": g})
    with pytest.raises(CheckpointError, match="no RNG state"):
        restore_rng_state(state, {"new": g, "replay": torch.Generator()})


def test_a_shared_checkpoint_with_a_different_commit_is_refused(tmp_path):
    """Acceptance check 8, second half: an arm refuses a phase-0 checkpoint
    whose commit, config or data hashes differ from its own."""
    model = build_model(TOY, torch.Generator().manual_seed(0))
    path = phase0_path(tmp_path, 0)
    save_shared(path, kind="phase0", model=model, fingerprint=fingerprint(), payload={})
    load_shared(path, "phase0", expect=fingerprint())  # same run: fine
    for field in ("commit", "data_hash", "manifest_sha256", "train_config_hash", "seed", "pilot"):
        other = fingerprint(**{field: ("z" * 40 if isinstance(getattr(fingerprint(), field), str) else 9)})
        with pytest.raises(CheckpointError, match=field):
            load_shared(path, "phase0", expect=other)


def test_a_shared_checkpoint_of_the_wrong_kind_is_refused(tmp_path):
    model = build_model(TOY, torch.Generator().manual_seed(0))
    path = init_path(tmp_path, 0)
    save_shared(path, kind="init", model=model, fingerprint=fingerprint())
    with pytest.raises(CheckpointError, match="expected 'phase0'"):
        load_shared(path, "phase0")


def test_require_compatible_names_every_difference():
    with pytest.raises(CheckpointError) as exc:
        require_compatible(fingerprint(), fingerprint(commit="b" * 40, seed=1), "x")
    assert "commit" in str(exc.value) and "seed" in str(exc.value)


def test_a_missing_checkpoint_is_an_error_not_a_silent_fresh_start(tmp_path):
    with pytest.raises(CheckpointError, match="missing"):
        load_state(tmp_path / "nothing.pt")
