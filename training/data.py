"""Training data: packing, ordering, and the replay mix.

This module only ever sees the training directory. It has no parameter for an
exam path and `train.py` does not have one to give it (`guard.py` holds that
line). Its inputs are the frozen contract, plus `split`, which the 2026-09-25
leakage re-audit made required: every line must say `"split": "train"`, and a
line that says anything else, or nothing, stops the run (`_require_train_split`):

    train_phase_{k}.jsonl   {prompt_hash, phase, tier, story, model, timestamp, split}
    replay/phase_{k}.json   {seed, prompt_hashes: [500]}   # same file every arm

Packing: each phase's stories are encoded in file order, joined with the
tokenizer's end-of-text id, and cut into fixed `block_size` sequences; the tail
remainder is dropped. Because file order is fixed and the cut is deterministic,
every arm and every seed sees the identical *set* of sequences -- only the order
differs, and only through the data generator.

Ordering and the arm-parity invariant: there are **two** data generators, one
for new-phase order and one for replay draws (plus a third, held in `train.py`,
for weight init). An arm with replay therefore draws its replay sequences from a
stream that no other arm touches, so arms A and B see byte-identical new-phase
batches at every phase, not only at phase 0.

Replay accounting (`PLAN.md`, "Training arms", decided 2026-09-21): replay is
**on top of** the budget. A batch is `n_new` new-phase sequences -- the same
number in every arm -- plus `replay_sequences_per_batch(n_new, f)` sequences
drawn from phases strictly earlier than the current one. At phase 0 there is no
earlier phase and therefore no replay in any arm.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from training.config import N_PHASES, replay_sequences_per_batch
from training.tokenizer import Tokenizer

REPLAY_SUBDIR = "replay"


class DataError(RuntimeError):
    """The training data is not what the contract says it is."""


# ---------------------------------------------------------------------------
# seeds


def derive_seed(seed: int, tag: str) -> int:
    """A stable sub-seed per stream, so streams cannot alias each other.

    Not `seed + 1`: two arms with adjacent seeds would then share a stream.
    """
    h = hashlib.sha256(f"lifespan:{tag}:{seed}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") % (2**63 - 1)


def make_generator(seed: int, tag: str) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(derive_seed(seed, tag))
    return g


# ---------------------------------------------------------------------------
# packing


@dataclass(frozen=True)
class PackedPhase:
    """One phase's stories as `[n_sequences, block_size]` token ids."""

    phase: int
    sequences: np.ndarray  # int32 [n, block_size]
    n_stories: int
    n_tokens_raw: int
    #: Every distinct `model` value seen in this phase's file. Collected on the
    #: pass that encodes the stories, so proving provenance costs no extra read.
    generator_models: frozenset[str] = frozenset()

    def __len__(self) -> int:
        return int(self.sequences.shape[0])


def _pack(ids: list[int], block_size: int) -> np.ndarray:
    n_seq = len(ids) // block_size
    if n_seq == 0:
        return np.zeros((0, block_size), dtype=np.int32)
    arr = np.asarray(ids[: n_seq * block_size], dtype=np.int32)
    return arr.reshape(n_seq, block_size)


#: The only `split` value a line this module trains on may carry.
TRAIN_SPLIT = "train"


def _require_train_split(rec: dict, path: Path, lineno: int) -> None:
    """Refuse a story line that is not declared `split: "train"`.

    The 2026-09-25 leakage re-audit appended exam-split stories to a train
    `stories.jsonl` without complaint: the response generator had no split
    check, and the story lines carried no `split` at all. The generator now
    writes `split` into every line; this is the training side of the same
    fix. A missing `split` is refused too -- unrecorded provenance is not
    provenance, and "probably train" is exactly the default that lets an exam
    story be trained on without a trace. Raised, never caught, so it fires
    before a model is built.
    """
    split = rec.get("split")
    if split != TRAIN_SPLIT:
        what = "has no 'split' field" if "split" not in rec else f"has split {split!r}"
        raise DataError(
            f"{path}:{lineno} {what}; only split {TRAIN_SPLIT!r} lines may be trained on. "
            "A story line that is not declared training data may be an exam story, and "
            "training on one makes every score on its phase meaningless. Fix the corpus; "
            "this is not a flag to override."
        )


def pack_phase(
    path: Path,
    phase: int,
    tokenizer: Tokenizer,
    block_size: int,
    keep_hashes: set[str] | None = None,
) -> PackedPhase:
    """Encode and pack one `train_phase_{k}.jsonl`.

    `keep_hashes` restricts to those `prompt_hash` values -- that is how the
    replay buffer becomes a packed pool without a second pass over the file.
    """
    ids: list[int] = []
    n_stories = 0
    seen: set[str] = set()
    models: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "story" not in rec:
                raise DataError(f"{path}:{lineno} has no 'story' field")
            # Every line, before the pilot/replay filter: one exam line anywhere
            # in the file means the file is not a training file.
            _require_train_split(rec, path, lineno)
            ph = rec.get("prompt_hash")
            if keep_hashes is not None:
                if ph not in keep_hashes:
                    continue
                seen.add(ph)
            model = rec.get("model")
            if not model or not isinstance(model, str):
                raise DataError(
                    f"{path}:{lineno} has no 'model' field. The generation model is the one "
                    "confound the forgetting curve cannot separate from forgetting, so a story "
                    "whose generator is unrecorded cannot be trained on."
                )
            models.add(model)
            ids.extend(tokenizer.encode(rec["story"]))
            ids.append(tokenizer.eot_id)
            n_stories += 1
    if keep_hashes is not None:
        missing = keep_hashes - seen
        if missing:
            raise DataError(
                f"replay buffer for phase {phase} names {len(missing)} prompt_hash values that are "
                f"not in {Path(path).name} (first: {sorted(missing)[:3]}). The buffer is chosen once "
                "from the training data and must match it exactly."
            )
    return PackedPhase(
        phase=phase,
        sequences=_pack(ids, block_size),
        n_stories=n_stories,
        n_tokens_raw=len(ids),
        generator_models=frozenset(models),
    )


# ---------------------------------------------------------------------------
# replay pool


@dataclass(frozen=True)
class ReplayPool:
    """Sequences from phases strictly earlier than `before_phase`.

    `phases[i]` is the phase `sequences[i]` came from; tests assert on it, and
    it is the only record of provenance the loop keeps.
    """

    before_phase: int
    sequences: np.ndarray  # int32 [n, block_size]
    phases: np.ndarray  # int16 [n]

    def __len__(self) -> int:
        return int(self.sequences.shape[0])


# ---------------------------------------------------------------------------
# the module


class DataModule:
    """Every loader the training loop and the hooks use.

    Built once per run from the training directory; holds the packed phases,
    the replay pools, and the two data generators.
    """

    def __init__(
        self,
        train_dir: Path,
        tokenizer: Tokenizer,
        block_size: int,
        seed: int,
        *,
        n_phases: int = N_PHASES,
        replay_dir: Path | None = None,
        stories_per_phase: int | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.train_dir = Path(train_dir).resolve()
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.seed = seed
        self.n_phases = n_phases
        self.device = device or torch.device("cpu")
        self.replay_dir = Path(replay_dir) if replay_dir is not None else self.train_dir / REPLAY_SUBDIR
        self.stories_per_phase = stories_per_phase

        self.g_new = make_generator(seed, "data.new")
        self.g_replay = make_generator(seed, "data.replay")
        self.g_joint = make_generator(seed, "data.joint")

        self.phases: list[PackedPhase] = [self._load_phase(k) for k in range(n_phases)]
        self.generator_model: str = self._check_one_generator()
        self._replay_packs: dict[int, PackedPhase] = {}
        self._replay_pools: dict[int, ReplayPool] = {}

    # -- provenance ---------------------------------------------------------

    def _check_one_generator(self) -> str:
        """The single generation model behind the whole corpus, or refuse.

        AGENTS.md: "One model generates every story, train and exam, all seven
        phases: a model change between phases is a confound the forgetting curve
        cannot separate from forgetting." Nothing used to enforce that -- the
        generator compared a model id only against lines already in the same
        output file, and each phase is its own file, so a change between phase 3
        and phase 4 was caught by nothing and left no trace in any artefact.

        The check is over every phase present, never a sample: a model change in
        one phase out of seven is exactly the case that must be caught. Raised
        from `__init__`, so it fires before a model is built, and never caught
        to keep a run going.
        """
        by_phase = {p.phase: sorted(p.generator_models) for p in self.phases}
        distinct = sorted({m for models in by_phase.values() for m in models})
        if not distinct:
            raise DataError("no generator model id found in any training file")
        if len(distinct) > 1:
            detail = ", ".join(f"phase {k}: {v}" for k, v in sorted(by_phase.items()))
            raise DataError(
                f"the training corpus was generated by {len(distinct)} different models "
                f"({distinct}). One model generates every story, train and exam, all seven "
                "phases; a model change between phases is a confound the forgetting curve "
                f"cannot separate from forgetting. Per phase -- {detail}. "
                "Regenerate the odd phases with the corpus's model; this is not a flag to "
                "override."
            )
        return distinct[0]

    # -- loading ------------------------------------------------------------

    def phase_path(self, phase: int) -> Path:
        path = self.train_dir / f"train_phase_{phase}.jsonl"
        if not path.is_file():
            raise DataError(f"missing training file for phase {phase}: {path}")
        return path

    def _load_phase(self, phase: int) -> PackedPhase:
        path = self.phase_path(phase)
        if self.stories_per_phase is None:
            return pack_phase(path, phase, self.tokenizer, self.block_size)
        keep = self._first_n_hashes(path, self.stories_per_phase)
        return pack_phase(path, phase, self.tokenizer, self.block_size, keep_hashes=keep)

    @staticmethod
    def _first_n_hashes(path: Path, n: int) -> set[str]:
        """`--pilot` takes the first N stories of each phase, not a sample: the
        subset must be the same file-order prefix in every arm and seed."""
        out: set[str] = set()
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                out.add(rec["prompt_hash"])
                if len(out) >= n:
                    break
        return out

    def replay_hashes(self, phase: int) -> set[str]:
        path = self.replay_dir / f"phase_{phase}.json"
        if not path.is_file():
            raise DataError(
                f"replay buffer missing: {path}. An arm with replay > 0 cannot run without it, and "
                "the same file must be used by every arm and seed."
            )
        rec = json.loads(path.read_text(encoding="utf-8"))
        hashes = rec.get("prompt_hashes")
        if not isinstance(hashes, list) or not hashes:
            raise DataError(f"{path} has no prompt_hashes")
        return set(hashes)

    def _replay_pack(self, phase: int) -> PackedPhase:
        if phase not in self._replay_packs:
            self._replay_packs[phase] = pack_phase(
                self.phase_path(phase),
                phase,
                self.tokenizer,
                self.block_size,
                keep_hashes=self.replay_hashes(phase),
            )
        return self._replay_packs[phase]

    def replay_pool(self, before_phase: int) -> ReplayPool:
        """The pool for a phase: every earlier phase's replay buffer, packed."""
        if before_phase not in self._replay_pools:
            seqs, phs = [], []
            for k in range(before_phase):
                pack = self._replay_pack(k)
                if len(pack) == 0:
                    raise DataError(f"replay buffer for phase {k} packs to zero sequences")
                seqs.append(pack.sequences)
                phs.append(np.full(len(pack), k, dtype=np.int16))
            if seqs:
                pool = ReplayPool(before_phase, np.concatenate(seqs), np.concatenate(phs))
            else:
                pool = ReplayPool(
                    before_phase,
                    np.zeros((0, self.block_size), dtype=np.int32),
                    np.zeros((0,), dtype=np.int16),
                )
            self._replay_pools[before_phase] = pool
        return self._replay_pools[before_phase]

    # -- budget -------------------------------------------------------------

    def steps_for_phase(self, phase: int, epochs: int, n_new: int) -> int:
        """Optimiser steps for one phase: `ceil(epochs * sequences / n_new)`.

        It depends on the data and the batch size only, so it is identical in
        every arm -- which is the point (`PLAN.md`: "Every arm takes the same
        number of optimizer steps per phase").
        """
        n_seq = len(self.phases[phase])
        if n_seq == 0:
            raise DataError(f"phase {phase} packs to zero sequences; block_size {self.block_size} too large?")
        return max(1, -(-epochs * n_seq // n_new))

    def total_steps_all_phases(self, epochs: int, n_new: int) -> int:
        """Arm A's total, which is what Arm E's "same total tokens" means."""
        return sum(self.steps_for_phase(k, epochs, n_new) for k in range(self.n_phases))

    # -- loaders ------------------------------------------------------------

    def _epoch_order(self, n: int, g: torch.Generator) -> torch.Tensor:
        return torch.randperm(n, generator=g)

    def _batches(self, sequences: np.ndarray, n_per_batch: int, steps: int, g: torch.Generator) -> Iterator[torch.Tensor]:
        """`steps` batches of `n_per_batch` rows, reshuffling each epoch."""
        n = sequences.shape[0]
        order = self._epoch_order(n, g)
        cursor = 0
        for _ in range(steps):
            picks: list[torch.Tensor] = []
            have = 0
            while have < n_per_batch:
                if cursor >= n:
                    order = self._epoch_order(n, g)
                    cursor = 0
                take = min(n_per_batch - have, n - cursor)
                picks.append(order[cursor : cursor + take])
                cursor += take
                have += take
            idx = torch.cat(picks).numpy()
            yield torch.from_numpy(sequences[idx].astype(np.int64)).to(self.device)

    def make_phase_loader(self, phase: int, n_sequences: int, *, steps: int | None = None, epochs: int = 4) -> Iterator[torch.Tensor]:
        """`PhaseContext.make_phase_loader`. Yields `steps` batches of shape
        `[n_sequences, block_size]` drawn from phase `phase`."""
        if steps is None:
            steps = self.steps_for_phase(phase, epochs, n_sequences)
        return self._batches(self.phases[phase].sequences, n_sequences, steps, self.g_new)

    def make_replay_loader(
        self,
        n_sequences: int,
        *,
        phase: int,
        steps: int,
        with_provenance: bool = False,
    ) -> Iterator[torch.Tensor] | Iterator[tuple[torch.Tensor, np.ndarray]]:
        """`PhaseContext.make_replay_loader`, bound to the current phase.

        Empty when `n_sequences == 0` or `phase == 0`: at phase 0 there is no
        earlier phase, so no arm replays anything. Draws are with replacement
        from the union of earlier phases' buffers, on the replay generator.
        """
        if n_sequences <= 0 or phase <= 0:
            return iter(())
        pool = self.replay_pool(phase)
        if len(pool) == 0:
            raise DataError(f"replay pool for phase {phase} is empty")
        return self._replay_batches(pool, n_sequences, steps, with_provenance)

    def _replay_batches(self, pool: ReplayPool, n: int, steps: int, with_provenance: bool):
        for _ in range(steps):
            idx = torch.randint(0, len(pool), (n,), generator=self.g_replay).numpy()
            batch = torch.from_numpy(pool.sequences[idx].astype(np.int64)).to(self.device)
            yield (batch, pool.phases[idx]) if with_provenance else batch

    def make_joint_loader(self, n_sequences: int, steps: int) -> Iterator[torch.Tensor]:
        """Arm E: all seven phases shuffled together (`PLAN.md`, arm E)."""
        pool = np.concatenate([p.sequences for p in self.phases])
        return self._batches(pool, n_sequences, steps, self.g_joint)

    # -- accounting ---------------------------------------------------------

    def token_budget(
        self,
        epochs: int,
        n_new: int,
        replay_fraction: float,
        *,
        phase_passes: int = 1,
        replay_passes: int = 1,
    ) -> dict:
        """New-phase and replay tokens per phase and in total, recorded
        separately in `config.json` so H2's compute ratio is readable.

        `phase_passes` is how many times the arm consumes the phase's material
        and `replay_passes` how many of those carry replay. They are not the
        same number, and neither is implied by `replay_fraction`:

          - a full fine-tune arm (A, B, E) makes 1 pass, with replay if it has
            any -- 1 and 1;
          - arm C trains the LoRA and then merges arithmetically, and the merge
            trains nothing -- 1 and 1, and its replay fraction is 0 anyway;
          - arms D and D-nr train the LoRA (phase data only, no replay) and
            then distil (phase data plus replay) -- 2 and 1.

        Deriving the ratio from `replay_fraction` alone, which is what this did
        until the 2026-09-22 audit, recorded 1.375 for arm D when it spends
        2.375x Arm A. `report.recorded_token_ratio` reads this field and never
        recomputes it, and H2b is the compute claim, so the error was about the
        size of the thing being claimed.
        """
        n_replay = replay_sequences_per_batch(n_new, replay_fraction)
        per_phase = []
        for k in range(self.n_phases):
            steps = self.steps_for_phase(k, epochs, n_new)
            replay_steps = steps if k > 0 else 0
            per_phase.append(
                {
                    "phase": k,
                    "sequences_in_phase": len(self.phases[k]),
                    "steps": steps,
                    "phase_passes": phase_passes,
                    "replay_passes": replay_passes,
                    "new_sequences_per_batch": n_new,
                    "replay_sequences_per_batch": n_replay if k > 0 else 0,
                    "new_phase_tokens": phase_passes * steps * n_new * self.block_size,
                    "replay_tokens": replay_passes * replay_steps * n_replay * self.block_size,
                    #: What a replay-free full fine-tune arm would spend on this
                    #: phase. The denominator of the ratio below.
                    "arm_a_new_phase_tokens": steps * n_new * self.block_size,
                }
            )
        total_new = sum(p["new_phase_tokens"] for p in per_phase)
        total_replay = sum(p["replay_tokens"] for p in per_phase)
        # Arm A's total, summed from the same per-phase rows rather than from a
        # closed form: 1.375 is exact only when all seven phases have equal step
        # counts, which real data will not give (the audit flagged the
        # assumption). Summing the rows is right for any phase-length mix.
        arm_a_total = sum(p["arm_a_new_phase_tokens"] for p in per_phase)
        return {
            "block_size": self.block_size,
            "epochs_per_phase": epochs,
            "replay_fraction": replay_fraction,
            "per_phase": per_phase,
            "total_new_phase_tokens": total_new,
            "total_replay_tokens": total_replay,
            "total_tokens": total_new + total_replay,
            "arm_a_total_tokens": arm_a_total,
            #: This arm's tokens as a multiple of Arm A's, summed from the rows
            #: above. Arm A is 1.0; replay 0.3 adds 0.375 (phase 0 carries no
            #: replay, so only 6 of 7 phases do, which is why it is 1.375 and
            #: not PLAN.md's "about 1.43x"); a uses_lora arm's two passes over
            #: phase-k material add another 1.0. H2b reads this field and
            #: `report.recorded_token_ratio` never recomputes it.
            "tokens_vs_arm_a": (total_new + total_replay) / arm_a_total if arm_a_total else 1.0,
        }

    def corpus_summary(self) -> list[dict]:
        return [
            {
                "phase": p.phase,
                "stories": p.n_stories,
                "tokens": p.n_tokens_raw,
                "sequences": len(p),
                "generator_model": sorted(p.generator_models),
            }
            for p in self.phases
        ]


def shift_for_lm(batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`[n, ctx]` token ids -> `(inputs, targets)`, the ordinary next-token
    shift. Kept here so the loop, the distillation step and the evaluator all
    slice a batch the same way."""
    return batch[:, :-1].contiguous(), batch[:, 1:].contiguous()


def concat_batches(new: torch.Tensor, replay: torch.Tensor | None) -> torch.Tensor:
    """The mixed batch: new-phase rows first, replay rows appended. Replay is
    on top, so the new-phase rows are exactly the rows a replay-free arm would
    have seen."""
    if replay is None or replay.numel() == 0:
        return new
    return torch.cat([new, replay], dim=0)


def sequences_per_batch_for(n_new: int, replay_fraction: float) -> tuple[int, int]:
    """`(n_new, n_replay)` -- the table lookup, in one place."""
    return n_new, replay_sequences_per_batch(n_new, replay_fraction)


__all__ = [
    "DataError",
    "DataModule",
    "PackedPhase",
    "ReplayPool",
    "concat_batches",
    "derive_seed",
    "make_generator",
    "pack_phase",
    "sequences_per_batch_for",
    "shift_for_lm",
]
