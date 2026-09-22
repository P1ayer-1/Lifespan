"""The two hooks `train.py` calls, so `lora.py`, `consolidate.py` and
`evaluate.py` never edit the training loop.

Frozen contract (AGENTS.md, Contracts, 2026-09-21). `trainer-core` builds the
`PhaseContext` and calls the hooks; `consolidation` and `evaluator` implement
them. Nobody edits this file without a lead decision.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Callable, Iterable, Protocol

import torch


@dataclass(frozen=True)
class PhaseContext:
    """Everything a hook may use. Hooks must not mutate it.

    `make_phase_loader(phase, n_sequences)` yields batches of token ids
    (LongTensor, shape [n_sequences, ctx]) drawn from that phase's training
    stories. `make_replay_loader(n_sequences)` yields batches drawn from the
    replay buffers of phases strictly earlier than `phase`; it is an empty
    iterable when `replay_fraction == 0.0` or `phase == 0`.
    """

    arm: str
    seed: int
    phase: int
    replay_fraction: float
    device: torch.device
    amp_dtype: torch.dtype | None
    make_phase_loader: Callable[[int, int], Iterable[torch.Tensor]]
    make_replay_loader: Callable[[int], Iterable[torch.Tensor]]
    steps: int
    lr: float
    warmup: int
    log: Callable[[str], None]
    timer: Callable[[str], AbstractContextManager[None]]


class AfterPhase(Protocol):
    """Called after every phase, including phase 0 for arms C, D and D-nr.

    Returns the model to carry into the next phase. Identity for A, B and E.
    """

    def __call__(self, model: torch.nn.Module, phase_k: int, ctx: PhaseContext) -> torch.nn.Module: ...


def identity_after_phase(model: torch.nn.Module, phase_k: int, ctx: PhaseContext) -> torch.nn.Module:
    """The hook for arms A, B, E and phase0: nothing happens at night."""
    return model
