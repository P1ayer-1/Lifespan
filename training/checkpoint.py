"""Per-phase checkpoints, written atomically, resumable.

`CLAUDE.md`: "Code must survive a Kaggle session dying: checkpoint after every
phase." A checkpoint holds everything a resumed run needs to produce *the same
matrix* an uninterrupted run would: model and optimiser state, the GradScaler,
every RNG stream, the partial matrix and the partial timings.

Two kinds of file:

- `state.pt` in the run folder -- the resume point, rewritten after every phase.
- `init_s{seed}.pt` and `phase0_s{seed}.pt` in the shared folder -- written
  once per seed and read by every arm, so all arms provably start from the same
  weights (`PLAN.md`, "Training arms"). Both carry a `Fingerprint`, and an arm
  refuses one whose commit, config or data hashes differ from its own.

Every write goes to a temp file in the destination directory and is then
`os.replace`d, which is atomic on Windows and on Linux. Never wrap a write in
`try`/`except` to keep a run going: a run that lost its checkpoint has lost the
only thing that makes it reproducible.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

#: Bumped when the payload shape changes; a mismatch is refused, not migrated.
CHECKPOINT_FORMAT = 1


class CheckpointError(RuntimeError):
    """A checkpoint is missing, corrupt, or does not belong to this run."""


# ---------------------------------------------------------------------------
# fingerprint


@dataclass(frozen=True)
class Fingerprint:
    """What must match before one run may inherit another run's weights."""

    commit: str
    dirty: bool
    seed: int
    model_config: dict
    data_hash: str
    manifest_sha256: str
    tokenizer_sha256: str
    train_config_hash: str
    pilot: bool

    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def mismatches(self, other: "Fingerprint") -> list[str]:
        out = []
        for key, mine in asdict(self).items():
            theirs = asdict(other)[key]
            if mine != theirs:
                out.append(f"{key}: this run has {mine!r}, the checkpoint has {theirs!r}")
        return out


def require_compatible(mine: Fingerprint, theirs: Fingerprint, what: str) -> None:
    """Refuse a shared checkpoint that does not belong to this run."""
    bad = mine.mismatches(theirs)
    if bad:
        raise CheckpointError(
            f"{what} was produced by a different run and cannot be shared:\n  " + "\n  ".join(bad)
        )


# ---------------------------------------------------------------------------
# rng


def seed_everything(seed: int) -> None:
    """One call, at the top of a run, before anything draws a number."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - no CUDA on the dev box
        torch.cuda.manual_seed_all(seed)


def rng_state(generators: dict[str, torch.Generator]) -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "generators": {name: g.get_state() for name, g in generators.items()},
    }
    if torch.cuda.is_available():  # pragma: no cover
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any], generators: dict[str, torch.Generator]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    for name, g in generators.items():
        if name not in state["generators"]:
            raise CheckpointError(f"checkpoint has no RNG state for generator {name!r}")
        g.set_state(state["generators"][name])
    if "torch_cuda" in state and torch.cuda.is_available():  # pragma: no cover
        torch.cuda.set_rng_state_all(state["torch_cuda"])


# ---------------------------------------------------------------------------
# atomic write


def atomic_save(payload: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def atomic_write_text(text: str, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def _load(path: Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise CheckpointError(f"checkpoint missing: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise CheckpointError(
            f"{path} is checkpoint format {payload.get('format')}, this code writes {CHECKPOINT_FORMAT}"
        )
    return payload


# ---------------------------------------------------------------------------
# run state


def save_state(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scaler: Any | None,
    generators: dict[str, torch.Generator],
    completed_phase: int,
    matrix: dict,
    timings: dict,
    fingerprint: Fingerprint,
    extra: dict | None = None,
) -> None:
    """The resume point after finishing `completed_phase`."""
    atomic_save(
        {
            "format": CHECKPOINT_FORMAT,
            "kind": "state",
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "rng": rng_state(generators),
            "completed_phase": completed_phase,
            "matrix": matrix,
            "timings": timings,
            "fingerprint": asdict(fingerprint),
            "extra": extra or {},
        },
        path,
    )


@dataclass
class RunState:
    model_state: dict
    optimizer_state: dict | None
    scaler_state: dict | None
    rng: dict
    completed_phase: int
    matrix: dict
    timings: dict
    fingerprint: Fingerprint
    extra: dict


def load_state(path: Path) -> RunState:
    p = _load(path)
    if p.get("kind") != "state":
        raise CheckpointError(f"{path} is a {p.get('kind')!r} checkpoint, not a run state")
    return RunState(
        model_state=p["model"],
        optimizer_state=p["optimizer"],
        scaler_state=p["scaler"],
        rng=p["rng"],
        completed_phase=p["completed_phase"],
        matrix=p["matrix"],
        timings=p["timings"],
        fingerprint=Fingerprint(**p["fingerprint"]),
        extra=p.get("extra", {}),
    )


# ---------------------------------------------------------------------------
# shared weights: the per-seed init and the per-seed phase 0


def save_shared(
    path: Path,
    *,
    kind: str,
    model: torch.nn.Module,
    fingerprint: Fingerprint,
    payload: dict | None = None,
) -> None:
    """`kind` is "init" or "phase0". `payload` carries phase 0's matrix row and
    wall clock, which every sequential arm copies into its own records."""
    if kind not in ("init", "phase0"):
        raise ValueError(f"shared checkpoint kind must be 'init' or 'phase0', got {kind!r}")
    atomic_save(
        {
            "format": CHECKPOINT_FORMAT,
            "kind": kind,
            "model": model.state_dict(),
            "fingerprint": asdict(fingerprint),
            "payload": payload or {},
        },
        path,
    )


@dataclass
class SharedCheckpoint:
    kind: str
    model_state: dict
    fingerprint: Fingerprint
    payload: dict


def load_shared(path: Path, kind: str, expect: Fingerprint | None = None) -> SharedCheckpoint:
    p = _load(path)
    if p.get("kind") != kind:
        raise CheckpointError(f"{path} is a {p.get('kind')!r} checkpoint, expected {kind!r}")
    got = Fingerprint(**p["fingerprint"])
    if expect is not None:
        require_compatible(expect, got, f"{kind} checkpoint {path}")
    return SharedCheckpoint(kind=kind, model_state=p["model"], fingerprint=got, payload=p.get("payload", {}))


def init_path(shared_dir: Path, seed: int) -> Path:
    return Path(shared_dir) / f"init_s{seed}.pt"


def phase0_path(shared_dir: Path, seed: int) -> Path:
    return Path(shared_dir) / f"phase0_s{seed}.pt"


def state_path(out_dir: Path) -> Path:
    return Path(out_dir) / "state.pt"
