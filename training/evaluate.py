"""Exam scoring: log-likelihood only, never generation.

`evaluate_all(model, exam_dir, phases)` is the frozen hook (AGENTS.md,
"Contracts - frozen 2026-09-21"). It returns every key of
`training.config.EXAM_TYPES`, each a list of exactly 7 floats indexed by exam
phase. Phases that have not been trained yet are still scored - that is the
matrix's upper triangle.

Scoring definitions, fixed here and not changed after a matrix comes back:

* **perplexity** - mean per-token loss over the phase's exam stories, *per
  token over the phase*, never a mean of per-story means. Stored as a loss, so
  lower is better; `metrics.py` negates it wherever a "score" is required.
* **cloze** - each of the 20 candidates is substituted for the single literal
  ``[MASK]`` substring with ``str.replace`` and the **whole filled sequence**
  is scored: the summed log-probability of every predicted token of the filled
  text (positions 1..n-1; the first token has no context and is not scored).
  Not the candidate span alone, and never the first sub-token alone. Rank-1
  accuracy over the 20 candidates; a tie for the maximum counts as wrong.
* **continuation** - each option is scored by the log-likelihood of the
  option's tokens given the prefix. The headline score is
  **length-normalised**: the mean per-token log-likelihood of the option. The
  summed score is computed and stored beside it (`evaluate_all_detailed`) and
  is never headlined. A tie for the maximum counts as wrong. Every item's
  provenance is asserted first (`validate_continuation_item`) and the chance
  level is derived from `len(options)`, never assumed to be 1/4.

Perplexity windowing rule (one fixed rule, stated in `EvalConfig` so it lands
in config.json verbatim): each exam story is tokenised on its own, then cut
into consecutive **non-overlapping** windows of `block_size` tokens
(stride == block_size). Every window is scored on its positions 1..n-1. A
trailing window with fewer than 2 tokens carries no prediction and is dropped.
Windows are not padded, stories are never concatenated, and the phase's loss is
(total NLL) / (total predicted tokens).

Everything runs under `model.eval()` and `torch.no_grad()`. There is no
sampling and no judge model. The log-softmax is computed in float32 even when
the model runs in fp16/bf16.

The summed continuation score is `matrix.json`'s fourth key,
`continuation_summed` (AGENTS.md, amendments 2026-09-22). It is not a member of
`EXAM_TYPES`, `evaluate_all` does not return it, and `metrics.py` refuses it as
a verdict metric; `evaluate_all_detailed(...).as_matrix_row()` is what the run
record writes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from training.config import EXAM_TYPES, N_PHASES, REAL

__all__ = [
    "ByteTokenizer",
    "CONTINUATION_SUMMED",
    "STORED_MATRIX_KEYS",
    "EvalConfig",
    "EvalResult",
    "evaluate_all",
    "evaluate_all_detailed",
    "score_cloze_phase",
    "score_continuation_phase",
    "score_perplexity_phase",
    "validate_continuation_item",
    "exam_story_phases",
    "ExamManifestMismatch",
    "verify_exam_stories",
    "load_jsonl",
]

MASK = "[MASK]"

#: The frozen option count. It is the *expected* number, not an assumption: the
#: chance level returned for a phase is always 1 / len(options), derived per
#: item, and a phase whose items disagree on the count is refused.
CONTINUATION_N_OPTIONS = 4
CONTINUATION_CHANCE = 1.0 / CONTINUATION_N_OPTIONS

#: The fields of a continuation item's `option_sources` entry (AGENTS.md,
#: amendment 2, 2026-09-22).
OPTION_SOURCE_FIELDS = ("story_id", "story_sha256", "phase")

#: The fourth key of matrix.json (AGENTS.md, amendments 2026-09-22): stored
#: beside the three exam types and never headlined. It is not an EXAM_TYPE.
CONTINUATION_SUMMED = "continuation_summed"

#: What matrix.json carries: the three exam types plus the stored sibling.
STORED_MATRIX_KEYS: tuple[str, ...] = tuple(EXAM_TYPES) + (CONTINUATION_SUMMED,)

#: AGENTS.md: a toy config is <= 1M parameters. A model bigger than that was
#: trained with a real tokenizer, so scoring it with bytes is meaningless and
#: `evaluate_all` refuses rather than falling back.
TOY_PARAM_BUDGET = 1_000_000


# --------------------------------------------------------------------------- #
# tokenizer
# --------------------------------------------------------------------------- #


class ByteTokenizer:
    """UTF-8 byte tokenizer, vocab 256.

    The fallback when the model carries no tokenizer, and the only tokenizer
    the evaluator's tests use: it matches `training.config.TOY.vocab_size` and
    needs no training, so a smoke test never touches a trained tokenizer.
    """

    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: Sequence[int]) -> str:
        return bytes(ids).decode("utf-8", errors="replace")


def _resolve_tokenizer(model: Any, tokenizer: Any | None) -> Any:
    """The run's tokenizer, or a byte tokenizer only at toy scale.

    `train.py` passes the real tokenizer explicitly (frozen 2026-09-22); it is
    not hung on the model, because arm D deep-copies the model. A missing
    tokenizer on a real model would silently make every grid score byte-level
    nonsense, so it is a guard, not a default: anything above the toy parameter
    budget raises.
    """
    if tokenizer is not None:
        return tokenizer
    attached = getattr(model, "tokenizer", None)
    if attached is not None:
        return attached
    n_params = sum(p.numel() for p in model.parameters())
    if n_params > TOY_PARAM_BUDGET:
        raise ValueError(
            "evaluate_all was called without a tokenizer on a model with "
            + str(n_params)
            + " parameters (toy budget is "
            + str(TOY_PARAM_BUDGET)
            + "). Pass the run's real tokenizer: a byte fallback would make "
            "every score in the matrix meaningless."
        )
    return ByteTokenizer()


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EvalConfig:
    """Everything the scorer needs that is not the model or the exam files.

    `block_size` is the context length. Sequences longer than it are
    **left-truncated** (the tail is kept) for cloze and continuation, so the
    scored tokens are always present; perplexity instead cuts a story into
    non-overlapping `block_size` windows and scores all of them.
    """

    block_size: int = REAL.block_size
    #: Sequences per forward pass. Cloze is 20 candidates per item, so this is
    #: the knob that trades memory for wall clock.
    batch_size: int = 16
    #: Positions per log-softmax chunk; caps the float32 [chunk, vocab] buffer.
    logit_chunk: int = 256
    device: str | None = None
    #: The one perplexity windowing rule, stated for config.json.
    perplexity_window_rule: str = (
        "per story: non-overlapping windows of block_size tokens (stride == "
        "block_size); each window scored on positions 1..n-1; a trailing window "
        "of fewer than 2 tokens is dropped; loss = total NLL / total predicted "
        "tokens over the phase"
    )
    cloze_scoring_rule: str = "whole filled sequence"
    continuation_headline: str = "length-normalised (mean per-token log-likelihood)"


@dataclass
class EvalResult:
    """What `evaluate_all_detailed` returns.

    `scores` holds exactly the `EXAM_TYPES` keys - what the frozen hook
    returns. The rest is stored beside the matrix and never headlined.
    """

    scores: dict[str, list[float]]
    continuation_summed: list[float]
    n_items: dict[str, list[int]]
    chance: dict[str, list[float | None]]

    def as_hook_dict(self) -> dict[str, list[float]]:
        """Exactly the three EXAM_TYPES keys - what the frozen hook returns."""
        return {k: list(self.scores[k]) for k in EXAM_TYPES}

    def as_matrix_row(self) -> dict[str, list[float]]:
        """The four keys matrix.json stores: the three exam types plus
        `continuation_summed`, which is stored and never headlined."""
        out = self.as_hook_dict()
        out[CONTINUATION_SUMMED] = list(self.continuation_summed)
        return out


# --------------------------------------------------------------------------- #
# model plumbing
# --------------------------------------------------------------------------- #


def _unwrap_logits(out: Any) -> torch.Tensor:
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)) and out:
        return out[0]
    logits = getattr(out, "logits", None)
    if isinstance(logits, torch.Tensor):
        return logits
    raise TypeError("cannot find logits in model output of type " + repr(type(out)))


def _forward_logits(model: torch.nn.Module, idx: torch.Tensor) -> torch.Tensor:
    """Full-sequence logits [B, T, V] from whatever the model returns.

    Accepts a bare tensor, a (logits, loss) tuple (nanoGPT's signature) or an
    object with `.logits`. nanoGPT also has an inference shortcut where
    `forward(idx)` returns only the last position; when the time dimension
    comes back short we retry with `targets=idx`, which forces the full set.
    """
    out = model(idx)
    logits = _unwrap_logits(out)
    if logits.dim() == 3 and logits.shape[1] != idx.shape[1]:
        logits = _unwrap_logits(model(idx, targets=idx))
    if logits.dim() != 3 or tuple(logits.shape[:2]) != tuple(idx.shape[:2]):
        raise ValueError(
            "model returned logits of shape "
            + str(tuple(logits.shape))
            + " for input "
            + str(tuple(idx.shape))
            + "; the scorer needs full-sequence [B, T, V] logits"
        )
    return logits


def _token_logprobs(logits: torch.Tensor, targets: torch.Tensor, chunk: int) -> torch.Tensor:
    """log P(target_t) per position, float32, computed in chunks over time.

    `logits[b, t]` predicts `targets[b, t]`. The log-softmax is taken in
    float32 whatever the model's dtype: an fp16 logsumexp over an 8k vocab
    loses real accuracy, and these numbers are the experiment's output.
    """
    pieces = []
    total = logits.shape[1]
    step = max(1, chunk)
    for start in range(0, total, step):
        part = logits[:, start : start + step].float()
        tgt = targets[:, start : start + step]
        gathered = part.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        pieces.append(gathered - torch.logsumexp(part, dim=-1))
    return torch.cat(pieces, dim=1)


@dataclass
class _Scored:
    """A token sequence and the target span to score, after truncation."""

    ids: list[int]
    start: int  # first *target* index, >= 1
    end: int  # one past the last target index


def _truncate_left(ids: list[int], start: int, end: int, block_size: int) -> _Scored:
    """Keep the last `block_size` tokens so the scored span survives."""
    if len(ids) > block_size:
        drop = len(ids) - block_size
        ids = ids[drop:]
        start = start - drop
        end = end - drop
    start = max(1, start)
    return _Scored(ids=ids, start=start, end=max(start, end))


@torch.no_grad()
def _score_spans(
    model: torch.nn.Module,
    items: Sequence[_Scored],
    cfg: EvalConfig,
    device: torch.device,
) -> list[tuple[float, int]]:
    """(summed log-likelihood, n scored tokens) for each item.

    Items are padded to the longest in the batch; padding contributes nothing
    because only the requested span is summed. Sums are accumulated in float64
    so a batch-size change cannot move a score.
    """
    results: list[tuple[float, int]] = []
    for lo in range(0, len(items), cfg.batch_size):
        batch = items[lo : lo + cfg.batch_size]
        width = max(len(it.ids) for it in batch)
        idx = torch.zeros((len(batch), width), dtype=torch.long)
        mask = torch.zeros((len(batch), width - 1), dtype=torch.bool)
        for row, it in enumerate(batch):
            idx[row, : len(it.ids)] = torch.tensor(it.ids, dtype=torch.long)
            # target index t is predicted by the logits at t-1
            mask[row, it.start - 1 : it.end - 1] = True
        idx = idx.to(device)
        mask = mask.to(device)
        logits = _forward_logits(model, idx)
        logp = _token_logprobs(logits[:, :-1, :], idx[:, 1:], cfg.logit_chunk)
        logp = logp.double() * mask.double()
        sums = logp.sum(dim=1).cpu().tolist()
        counts = mask.sum(dim=1).cpu().tolist()
        results.extend((float(s), int(c)) for s, c in zip(sums, counts))
    return results


# --------------------------------------------------------------------------- #
# exam files
# --------------------------------------------------------------------------- #


def load_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _stories_path(exam_dir: Path, phase: int) -> Path:
    return exam_dir / "stories" / ("exam_phase_" + str(phase) + ".jsonl")


def _probes_path(exam_dir: Path, phase: int, kind: str) -> Path:
    return exam_dir / "probes" / (kind + "_phase_" + str(phase) + ".jsonl")


# --------------------------------------------------------------------------- #
# the three exams
# --------------------------------------------------------------------------- #


def score_perplexity_phase(
    model: torch.nn.Module,
    stories: Sequence[dict],
    tokenizer: Any,
    cfg: EvalConfig,
    device: torch.device,
) -> tuple[float, int]:
    """Mean per-token loss over the phase. Returns (loss, n_predicted_tokens).

    Per token over the whole phase: the NLL of every predicted token in every
    window of every story is summed and divided by the total count. A mean of
    per-story means would weight a 40-token story like a 900-token one.
    """
    windows: list[_Scored] = []
    for row in stories:
        ids = tokenizer.encode(row["story"])
        for lo in range(0, len(ids), cfg.block_size):
            chunk = ids[lo : lo + cfg.block_size]
            if len(chunk) < 2:
                continue  # carries no prediction
            windows.append(_Scored(ids=chunk, start=1, end=len(chunk)))
    if not windows:
        return float("nan"), 0
    scored = _score_spans(model, windows, cfg, device)
    total_lp = sum(s for s, _ in scored)
    total_n = sum(n for _, n in scored)
    if total_n == 0:
        return float("nan"), 0
    return -total_lp / total_n, total_n


def score_cloze_phase(
    model: torch.nn.Module,
    items: Sequence[dict],
    tokenizer: Any,
    cfg: EvalConfig,
    device: torch.device,
) -> tuple[float, int, float | None]:
    """Rank-1 accuracy over the candidate set. Returns (acc, n_items, chance).

    The whole filled sequence is scored (see the module docstring). A tie for
    the maximum counts as wrong. `chance` is the mean of 1/len(candidates)
    over the phase's items - 0.05 under the frozen 20-candidate contract.
    """
    if not items:
        return float("nan"), 0, None
    flat: list[_Scored] = []
    bounds: list[tuple[int, int]] = []
    answer_at: list[int] = []
    sizes: list[int] = []
    for row in items:
        text = row["text_with_mask"]
        if text.count(MASK) != 1:
            raise ValueError(
                "cloze item " + repr(row.get("id")) + " does not contain " + MASK + " exactly once"
            )
        candidates = list(row["candidates"])
        if row["answer"] not in candidates:
            raise ValueError(
                "cloze item " + repr(row.get("id")) + ": answer is not among its candidates"
            )
        start = len(flat)
        for cand in candidates:
            ids = tokenizer.encode(text.replace(MASK, cand))
            if len(ids) < 2:
                ids = list(ids) + [0]
            flat.append(_truncate_left(list(ids), 1, len(ids), cfg.block_size))
        bounds.append((start, len(flat)))
        answer_at.append(candidates.index(row["answer"]))
        sizes.append(len(candidates))
    scored = _score_spans(model, flat, cfg, device)
    correct = 0
    for (lo, hi), gold in zip(bounds, answer_at):
        totals = [scored[i][0] for i in range(lo, hi)]
        if _argmax_unique(totals) == gold:
            correct += 1
    chance = sum(1.0 / s for s in sizes) / len(sizes)
    return correct / len(items), len(items), chance


def exam_story_phases(exam_dir: str | Path, phases: Iterable[int] | None = None) -> dict[str, int]:
    """normalised story sha256 -> phase for every exam story on disk.

    The hash is `training.guard.story_sha256`, the one the manifest's
    per-story entries carry. Every phase's stories file is read (distractors
    come from *other* phases), and a phase whose file is missing is skipped;
    an exam dir with no stories at all raises.
    """
    from training.guard import story_sha256

    exam_dir = Path(exam_dir)
    out: dict[str, int] = {}
    for phase in range(N_PHASES) if phases is None else phases:
        path = _stories_path(exam_dir, phase)
        if not path.is_file():
            continue
        for row in load_jsonl(path):
            if "story" in row:
                out[story_sha256(row["story"])] = int(row.get("phase", phase))
    if not out:
        raise ValueError(
            "no exam stories under " + str(exam_dir / "stories")
            + "; option_sources cannot be checked against anything"
        )
    return out


class ExamManifestMismatch(ValueError):
    """The exam stories on disk are not the ones the manifest froze."""


def _verify_phase_stories(
    rows: Sequence[dict], phase: int, story_hashes: Mapping[str, int], where: str
) -> int:
    """Check one phase's exam story rows against the manifest's per-story
    hashes; return the number of stories. See `verify_exam_stories`."""
    from training.guard import story_sha256

    story_hashes = {str(h).lower(): p for h, p in story_hashes.items()}
    expected = {h for h, p in story_hashes.items() if p == phase}
    seen: dict[str, str] = {}
    for n, row in enumerate(rows, start=1):
        rid = repr(row.get("id", "line " + str(n)))
        if "story" not in row:
            raise ExamManifestMismatch(where + ": story " + rid + " has no 'story' field")
        if "phase" in row and row["phase"] != phase:
            raise ExamManifestMismatch(
                where + ": story " + rid + " says phase " + repr(row["phase"])
                + " but sits in the phase-" + str(phase) + " file"
            )
        sha = story_sha256(row["story"])
        if sha not in story_hashes:
            raise ExamManifestMismatch(
                where + ": story " + rid + " (story_sha256 " + sha + ") is not in the manifest; "
                "the exam on disk is not the one that was frozen"
            )
        if story_hashes[sha] != phase:
            raise ExamManifestMismatch(
                where + ": story " + rid + " (story_sha256 " + sha + ") is a phase-"
                + str(story_hashes[sha]) + " exam story in the manifest, not phase " + str(phase)
            )
        if sha in seen:
            raise ExamManifestMismatch(
                where + ": stories " + seen[sha] + " and " + rid + " are the same story ("
                + sha + "); the manifest froze each exam story once"
            )
        seen[sha] = rid
    missing = expected - set(seen)
    if missing:
        raise ExamManifestMismatch(
            where + ": " + str(len(missing)) + " phase-" + str(phase) + " exam stories in the "
            "manifest are not on disk (first: " + sorted(missing)[0] + ")"
        )
    return len(seen)


def verify_exam_stories(
    exam_dir: str | Path, story_hashes: Mapping[str, int], phases: Iterable[int]
) -> dict[int, int]:
    """Refuse unless the exam stories on disk are exactly the manifest's.

    `story_hashes` is the manifest's normalised story sha256 -> phase (the
    guard report's `exam_story_phases`). For each phase in `phases`, the
    `stories/exam_phase_{k}.jsonl` file must exist, and its stories -- hashed
    with `training.guard.story_sha256`, the manifest's definition -- must be
    exactly the manifest's stories of phase k: none missing, none extra, none
    duplicated, none filed under another phase. Returns {phase: n_stories}.

    Before 2026-09-30 nothing tied the scorer to the frozen exam: the guard
    checks the training side against the manifest's hashes and never opens the
    exam directory, so an exam directory edited or swapped after the freeze
    would have been scored without complaint. Messages carry ids, phases and
    hashes only, never exam text.
    """
    exam_dir = Path(exam_dir)
    if not story_hashes:
        raise ExamManifestMismatch("the manifest's story hashes are empty; nothing to verify against")
    counts: dict[int, int] = {}
    for phase in phases:
        path = _stories_path(exam_dir, phase)
        if not path.is_file():
            raise ExamManifestMismatch(
                "exam stories for phase " + str(phase) + " are missing: " + str(path)
            )
        counts[phase] = _verify_phase_stories(load_jsonl(path), phase, story_hashes, path.name)
    return counts


def validate_continuation_item(row: dict, known_story_hashes: Any = None) -> int:
    """Check one continuation item and return its option count.

    Raises on anything that would make the score meaningless. Cloze items have
    always been validated; continuation ones were not, and the asymmetry hid a
    permanent hole - `distractor_phases` was written by the contract and read
    by nothing, so a distractor accidentally drawn from a *training* story
    would have been undetectable, and would have inflated the headline metric
    on exactly the phases the arms are supposed to have forgotten.

    That cannot be repaired after the freeze: a manifest entry hashes a whole
    exam file, while an option is a paragraph excerpted from one, so an
    option's sha256 will never match a manifest hash. `exam-keeper` checks the
    hashes against the manifest at freeze time; this assertion is the one that
    runs on every evaluation, and after the freeze it is the only one left.

    The checks (AGENTS.md, amendment 2, 2026-09-22):
      - `options` is a list of at least two paragraphs;
      - `answer_index` is an in-range integer;
      - `option_sources` has one entry per option, each carrying story_id,
        story_sha256 and phase;
      - the answer option's source phase equals the item's own phase;
      - each of the other options' sources has neither the item's phase nor the
        answer's story id;
      - when `known_story_hashes` is given (a set of exam story hashes, or a
        mapping hash -> phase), every source's `story_sha256` is a member, and
        with a mapping its claimed `phase` is the phase the exam records. The
        hash was self-attested until the 2026-09-25 audit: a distractor whose
        hash was in no manifest was accepted. `evaluate_all_detailed` always
        passes one.

    Error messages carry ids, phases and hashes only - never a prefix, an
    option or any other exam text, because a traceback ends up in a Kaggle
    notebook's output.
    """
    where = "continuation item " + repr(row.get("id"))
    options = row.get("options")
    if not isinstance(options, (list, tuple)) or len(options) < 2:
        raise ValueError(where + ": 'options' must be a list of at least 2 paragraphs")
    n = len(options)

    answer_index = row.get("answer_index")
    if isinstance(answer_index, bool) or not isinstance(answer_index, int):
        raise ValueError(where + ": 'answer_index' must be an int, got " + type(answer_index).__name__)
    if not 0 <= answer_index < n:
        raise ValueError(
            where + ": answer_index " + str(answer_index) + " is out of range for "
            + str(n) + " options"
        )

    sources = row.get("option_sources")
    if not isinstance(sources, (list, tuple)):
        raise ValueError(
            where + ": 'option_sources' is required (AGENTS.md amendment 2); a probe "
            "set that cannot prove its own provenance is not an exam"
        )
    if len(sources) != n:
        raise ValueError(
            where + ": 'option_sources' has " + str(len(sources)) + " entries for "
            + str(n) + " options; one per option is required"
        )
    for i, source in enumerate(sources):
        if not isinstance(source, dict):
            raise ValueError(where + ": option_sources[" + str(i) + "] is not an object")
        missing = [f for f in OPTION_SOURCE_FIELDS if f not in source]
        if missing:
            raise ValueError(
                where + ": option_sources[" + str(i) + "] is missing " + ", ".join(missing)
            )
        if known_story_hashes is not None:
            sha = str(source["story_sha256"]).lower()
            if sha not in known_story_hashes:
                raise ValueError(
                    where + ": option_sources[" + str(i) + "].story_sha256 " + sha
                    + " is not the hash of any exam story; an option's source must be "
                    "an exam story recorded in the manifest"
                )
            if isinstance(known_story_hashes, Mapping) and known_story_hashes[sha] != source["phase"]:
                raise ValueError(
                    where + ": option_sources[" + str(i) + "] claims phase "
                    + repr(source["phase"]) + " but story " + sha + " is an exam story of phase "
                    + repr(known_story_hashes[sha])
                )

    item_phase = row.get("phase")
    answer_source = sources[answer_index]
    if answer_source["phase"] != item_phase:
        raise ValueError(
            where + ": the answer option's source phase is " + repr(answer_source["phase"])
            + " but the item's phase is " + repr(item_phase)
            + "; the true continuation must come from this phase"
        )
    answer_story = answer_source["story_id"]
    for i, source in enumerate(sources):
        if i == answer_index:
            continue
        if source["phase"] == item_phase:
            raise ValueError(
                where + ": distractor option " + str(i) + " was drawn from the item's own "
                "phase " + repr(item_phase) + "; a distractor must come from another phase"
            )
        if source["story_id"] == answer_story:
            raise ValueError(
                where + ": distractor option " + str(i) + " shares the answer's source story "
                + repr(answer_story)
            )
    return n


def score_continuation_phase(
    model: torch.nn.Module,
    items: Sequence[dict],
    tokenizer: Any,
    cfg: EvalConfig,
    device: torch.device,
    known_story_hashes: Any = None,
) -> tuple[float, float, int, float | None]:
    """Returns (normalised accuracy, summed accuracy, n_items, chance).

    `known_story_hashes` is passed to `validate_continuation_item` for every
    item (a set of exam story hashes, or a mapping hash -> phase).

    The headline is the length-normalised accuracy - the mean per-token
    log-likelihood of the option given the prefix. The summed accuracy is
    computed and stored beside it and is never headlined. Ties count as wrong
    under both.

    Every item is validated first (`validate_continuation_item`). `chance` is
    derived as 1 / len(options), never assumed to be 1/4, and a phase whose
    items do not agree on the option count is refused rather than scored
    against a chance level that would be wrong for some of them.
    """
    if not items:
        return float("nan"), float("nan"), 0, None
    n_options = validate_continuation_item(items[0], known_story_hashes)
    for row in items[1:]:
        k = validate_continuation_item(row, known_story_hashes)
        if k != n_options:
            raise ValueError(
                "continuation item " + repr(row.get("id")) + " has " + str(k)
                + " options but this phase's earlier items have " + str(n_options)
                + "; one chance level per phase or the accuracies are not comparable"
            )
    flat: list[_Scored] = []
    bounds: list[tuple[int, int]] = []
    answer_at: list[int] = []
    for row in items:
        prefix_ids = tokenizer.encode(row["prefix"])
        start = len(flat)
        for option in row["options"]:
            # Re-tokenise the join rather than concatenating two token lists, so
            # the boundary token is what the model would really see. The span
            # start is the length of the prefix's own encoding: exact for a byte
            # tokenizer and the standard approximation for BPE.
            full = list(tokenizer.encode(row["prefix"] + option))
            if len(full) < 2:
                full = full + [0]
            span_start = max(1, min(len(prefix_ids), len(full) - 1))
            flat.append(_truncate_left(full, span_start, len(full), cfg.block_size))
        bounds.append((start, len(flat)))
        answer_at.append(int(row["answer_index"]))
    scored = _score_spans(model, flat, cfg, device)
    correct_norm = 0
    correct_sum = 0
    for (lo, hi), gold in zip(bounds, answer_at):
        sums = [scored[i][0] for i in range(lo, hi)]
        norms = [scored[i][0] / max(1, scored[i][1]) for i in range(lo, hi)]
        if _argmax_unique(norms) == gold:
            correct_norm += 1
        if _argmax_unique(sums) == gold:
            correct_sum += 1
    n = len(items)
    return correct_norm / n, correct_sum / n, n, 1.0 / n_options


def _argmax_unique(values: Sequence[float]) -> int:
    """Index of the strict maximum, or -1 when it is tied. Ties count wrong."""
    best = max(values)
    if list(values).count(best) != 1:
        return -1
    return list(values).index(best)


# --------------------------------------------------------------------------- #
# the hook
# --------------------------------------------------------------------------- #


def evaluate_all_detailed(
    model: torch.nn.Module,
    exam_dir: str | Path,
    phases: Iterable[int] | None = None,
    *,
    tokenizer: Any | None = None,
    cfg: EvalConfig | None = None,
    story_hashes: Mapping[str, int] | None = None,
) -> EvalResult:
    """Score every exam type on every phase in `phases`.

    A phase not listed in `phases` gets `nan` in its slot, so the returned
    lists are always length `N_PHASES`. The grid always passes all seven: a
    phase not yet trained is still scored (the matrix's upper triangle).

    `story_hashes` (normalised story sha256 -> phase, e.g. the guard report's
    `exam_story_phases`, i.e. the manifest's) is what every continuation
    option's `option_sources[*].story_sha256` must be a member of. Without it
    the set is computed from the exam stories on disk (`exam_story_phases`),
    so the check never silently turns off.

    When `story_hashes` is given it is also the frozen exam: each scored
    phase's stories on disk must be exactly the manifest's stories of that
    phase (`verify_exam_stories`'s rule), or `ExamManifestMismatch` is raised
    before anything is scored. `train.py` always passes the guard's map.
    """
    cfg = cfg or EvalConfig()
    exam_dir = Path(exam_dir)
    tied_to_manifest = story_hashes is not None
    if story_hashes is None:
        story_hashes = exam_story_phases(exam_dir)
    elif not story_hashes:
        raise ExamManifestMismatch("story_hashes is empty; nothing to verify the exam against")
    wanted = sorted(set(range(N_PHASES) if phases is None else phases))
    for phase in wanted:
        if not 0 <= phase < N_PHASES:
            raise ValueError("phase " + str(phase) + " out of range 0.." + str(N_PHASES - 1))
    stories_by_phase = {phase: load_jsonl(_stories_path(exam_dir, phase)) for phase in wanted}
    if tied_to_manifest:
        # Every scored phase first, so a mismatch refuses before any scoring.
        for phase in wanted:
            _verify_phase_stories(
                stories_by_phase[phase], phase, story_hashes, _stories_path(exam_dir, phase).name
            )
    tok = _resolve_tokenizer(model, tokenizer)
    device = torch.device(cfg.device) if cfg.device is not None else _model_device(model)

    was_training = model.training
    model.eval()
    try:
        scores = {k: [float("nan")] * N_PHASES for k in EXAM_TYPES}
        summed = [float("nan")] * N_PHASES
        n_items = {k: [0] * N_PHASES for k in EXAM_TYPES}
        chance: dict[str, list[float | None]] = {k: [None] * N_PHASES for k in EXAM_TYPES}
        for phase in wanted:
            stories = stories_by_phase[phase]
            loss, n_tok = score_perplexity_phase(model, stories, tok, cfg, device)
            scores["perplexity"][phase] = loss
            n_items["perplexity"][phase] = n_tok

            cloze_items = load_jsonl(_probes_path(exam_dir, phase, "cloze"))
            acc, n, ch = score_cloze_phase(model, cloze_items, tok, cfg, device)
            scores["cloze"][phase] = acc
            n_items["cloze"][phase] = n
            chance["cloze"][phase] = ch

            cont_items = load_jsonl(_probes_path(exam_dir, phase, "continuation"))
            acc_n, acc_s, n, ch2 = score_continuation_phase(
                model, cont_items, tok, cfg, device, known_story_hashes=story_hashes
            )
            scores["continuation"][phase] = acc_n
            summed[phase] = acc_s
            n_items["continuation"][phase] = n
            chance["continuation"][phase] = ch2
    finally:
        if was_training:
            model.train()

    return EvalResult(scores=scores, continuation_summed=summed, n_items=n_items, chance=chance)


def evaluate_all(
    model: torch.nn.Module,
    exam_dir: str | Path,
    phases: Iterable[int] | None = None,
    *,
    tokenizer: Any | None = None,
    cfg: EvalConfig | None = None,
    story_hashes: Mapping[str, int] | None = None,
) -> dict[str, list[float]]:
    """The frozen hook: {exam_type: [7 floats]}, exactly the EXAM_TYPES keys.

    `tokenizer` and `cfg` are keyword-only (frozen 2026-09-22); the
    three-positional-argument call still works. `train.py` passes the run's
    real tokenizer explicitly. With no tokenizer at all this falls back to
    bytes **only** at toy scale and raises on anything larger - see
    `_resolve_tokenizer`.

    The summed continuation score is not a fourth key here; it is returned by
    `evaluate_all_detailed(...).as_matrix_row()` for `matrix.json`.
    """
    return evaluate_all_detailed(
        model, exam_dir, phases, tokenizer=tokenizer, cfg=cfg, story_hashes=story_hashes
    ).as_hook_dict()


def _model_device(model: torch.nn.Module) -> torch.device:
    for p in model.parameters():
        return p.device
    for b in model.buffers():
        return b.device
    return torch.device("cpu")
