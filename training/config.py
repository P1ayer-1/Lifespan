"""Resolved configuration: the model, the arm table, and the toy config every
agent's CPU smoke tests use.

The numbers here come from PLAN.md, "Model and compute" and "Training arms",
and are frozen for the grid. Changing one is a lead decision recorded in
docs/DECISIONS.md *before* the grid, never after a matrix comes back.
"""

from __future__ import annotations

from dataclasses import dataclass, field

EXAM_TYPES: tuple[str, ...] = ("perplexity", "cloze", "continuation")
N_PHASES = 7
SEEDS = (0, 1, 2)

#: The metric H1-H4 are read on (AGENTS.md, frozen 2026-09-21). "cloze" is
#: reported beside it; a disagreement is reported as split, not resolved.
HEADLINE_EXAM_TYPE = "continuation"


@dataclass(frozen=True)
class ModelConfig:
    n_layer: int
    n_head: int
    n_embd: int
    block_size: int
    vocab_size: int
    dropout: float = 0.0

    def n_params(self) -> tuple[int, int]:
        """(non-embedding, embedding) parameter counts, tied output head.

        Linear layers carry no bias (nanoGPT's convention, and one fewer thing
        for LoRA to miss). LayerNorm is affine: two vectors per norm, two norms
        per block plus the final one. Corrected 2026-09-22 - the first count
        omitted them, and the fix is to the arithmetic, not to the model.
        """
        per_layer = 4 * self.n_embd**2 + 2 * self.n_embd * 4 * self.n_embd
        norms = (2 * self.n_layer + 1) * 2 * self.n_embd
        non_emb = self.n_layer * per_layer + norms
        emb = self.vocab_size * self.n_embd + self.block_size * self.n_embd
        return non_emb, emb


#: PLAN.md: 8 layers, 512 hidden, 8 heads, 1,024-token context. vocab 8,192 is
#: the bottom of PLAN.md's 8k-16k band and lands the total at ~30M:
#: 25,183,232 non-embedding + 4,718,592 embedding = 29,901,824. A 16k vocab
#: would make it ~34M and put a seventh of the model in the embedding table.
REAL = ModelConfig(n_layer=8, n_head=8, n_embd=512, block_size=1024, vocab_size=8192)

#: CPU smoke tests only (AGENTS.md: toy config, <=1M parameters, <=50 steps).
#: Byte-level vocab so no tokenizer has to be trained in a test.
TOY = ModelConfig(n_layer=2, n_head=2, n_embd=64, block_size=64, vocab_size=256)


@dataclass(frozen=True)
class Arm:
    """A row of the arm table. Arms are data; there is no per-arm script."""

    name: str
    replay_fraction: float
    uses_lora: bool
    after_phase: str  # "identity" | "merge" | "distill"
    joint: bool = False
    #: A, B, C, D and D-nr all start from the shared phase-0 checkpoint.
    shares_phase0: bool = True


ARMS: dict[str, Arm] = {
    "phase0": Arm("phase0", 0.0, False, "identity"),
    "A": Arm("A", 0.0, False, "identity"),
    "B": Arm("B", 0.3, False, "identity"),
    "C": Arm("C", 0.0, True, "merge"),
    "D": Arm("D", 0.3, True, "distill"),
    "D-nr": Arm("D-nr", 0.0, True, "distill"),
    "E": Arm("E", 0.0, False, "identity", joint=True, shares_phase0=False),
}


@dataclass(frozen=True)
class TrainConfig:
    """Per-phase budget and optimiser settings. Identical for every arm:
    replay is added on top of the batch, it does not change steps or schedule.
    """

    epochs_per_phase: int = 4
    lr: float = 3e-4
    warmup_steps: int = 200
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    #: N new-phase sequences per batch. An arm with replay fraction f adds
    #: round(N * f / (1 - f)) replay sequences, so replay is f of the batch.
    sequences_per_batch: int = 32
    #: LoRA, arms C and D (PLAN.md, Model and compute).
    lora_rank: int = 16
    lora_alpha: int = 32
    #: The consolidation loss (PLAN.md, The consolidation step).
    distill_lambda: float = 1.0
    #: Sizes, full grid.
    train_stories_per_phase: int = 5_000
    exam_items_per_phase: int = 500
    replay_buffer_per_phase: int = 500


#: --pilot changes sizes and nothing else (AGENTS.md, frozen 2026-09-21).
PILOT_OVERRIDES: dict[str, int] = {
    "train_stories_per_phase": 1_000,
    "exam_items_per_phase": 100,
}


def warmup_for(steps_per_phase: int, base_warmup: int = 200, *, scaled: bool = False) -> int:
    """Warmup steps. The grid gets PLAN.md's fixed 200 and nothing else.

    `scaled=True` is passed ONLY by a --pilot or --toy run. Those phases are
    shorter than 200 steps (pilot: 74, toy: <=50), so at the fixed warmup they
    would never leave the linear ramp and would report a peak lr they never
    reached, which would make the pilot a bad instrument for finding the bugs it
    exists to find. Warmup is counted in steps, and steps is a size, so scaling
    it at pilot sizes is inside what --pilot may change (decided 2026-09-22). A
    grid run never passes `scaled` and its warmup is exactly 200.

    Note for the owner, not acted on: a grid phase is 367 steps, so the fixed
    200 is 54% of the phase. PLAN.md fixes it and the plan is frozen; changing
    it is a decision to make before the grid, never after a matrix comes back.
    """
    if not scaled:
        return base_warmup
    return min(base_warmup, max(1, steps_per_phase // 10))


@dataclass(frozen=True)
class ToyOverrides:
    """The sizes a CPU smoke test runs at; the model is `TOY`.

    Frozen in AGENTS.md as `training.config.TOY_RUN`, which is where three
    agents' tests look for it. It briefly lived in `train.py` after the lead
    truncated this file on 2026-09-22; restored here, and `train.py` re-exports
    the name so existing imports keep working.
    """

    stories_per_phase: int = 20
    exam_items_per_phase: int = 4
    max_steps: int = 50
    sequences_per_batch: int = 4
    warmup_steps: int = 2


TOY_RUN = ToyOverrides()


def replay_sequences_per_batch(n_new: int, replay_fraction: float) -> int:
    """round(N * f / (1 - f)): replay is `replay_fraction` of the final batch,
    and the new-phase count N is the same in every arm (AGENTS.md, frozen)."""
    if not 0.0 <= replay_fraction < 1.0:
        raise ValueError(f"replay_fraction must be in [0, 1), got {replay_fraction}")
    if replay_fraction == 0.0:
        return 0
    return round(n_new * replay_fraction / (1.0 - replay_fraction))
