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
    #: N, the new-phase sequences per batch. The hook must use this and not a
    #: default from TrainConfig, or the night desyncs from the day when train.py
    #: runs a non-default N (added 2026-09-22 at consolidation's request).
    sequences_per_batch: int
    lr: float
    warmup: int
    log: Callable[[str], None]
    timer: Callable[[str], AbstractContextManager[None]]


class AfterPhase(Protocol):
    """Called after phases 1..6. Returns the model to carry forward.

    Identity for A, B and E. Phase 0 is ordinary full training for every
    sequential arm and is shared between them (docs/DECISIONS.md, 2026-09-21),
    so no hook runs at phase 0 and the LoRA regime starts at phase 1, which is
    what PLAN.md means by "one LoRA per phase 1-6".

    For an arm with `uses_lora`, this hook owns the whole phase: train.py does
    NOT run its ordinary base phase loop, because the day (training the LoRA on
    phase k) happens in here. Running both would double the arm's token budget
    and would leave arm C's base unfrozen, which is the opposite of what arm C
    is for (corrected 2026-09-22).
    """

    def __call__(self, model: torch.nn.Module, phase_k: int, ctx: PhaseContext) -> torch.nn.Module: ...


def identity_after_phase(model: torch.nn.Module, phase_k: int, ctx: PhaseContext) -> torch.nn.Module:
    """The hook for arms A, B, E and phase0: nothing happens at night."""
    return model
