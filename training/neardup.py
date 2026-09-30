"""Near-duplicate check and verbatim-overlap report, exam vs training.

Owned by `exam-keeper` (AGENTS.md, Ownership: "the near-duplicate check"). It
runs once, at freeze time, over the exam stories and *every* training phase,
and its output is ids, hashes, counts and fractions -- never story text, so the
report can go in a log, a brief or `docs/DECISIONS.md`.

Two jobs (leakage audit 2026-09-25, blocking finding 3 and its note):

1. **Near duplicates** -- an exam story that is a near copy of a training
   story. An exam story is a near duplicate of a training story when either

   * **prefix rule** (`.claude/agents/exam-keeper.md`): their first
     `PREFIX_CHARS` = 200 characters match after `prefix_key` (NFC, casefold,
     whitespace runs to one space, stripped); or
   * **shingle rule**: `CONTAINMENT_THRESHOLD` = 0.5 or more of the exam
     story's word 5-gram shingles (`SHINGLE_N`) occur in that one training
     story. Containment (|E & T| / |E|) rather than Jaccard, so an exam story
     pasted inside a longer training story is caught too; Jaccard is reported
     beside it.

   Why 5-grams and 0.5: one edited word destroys at most 5 shingles, so an
   exam story with an edit every 20 words keeps ~75% of its shingles and an
   edit every 10 words ~50%; both are copies. Two independently generated
   stories from the same template and fact bank share stock phrases and at
   most a fact sentence or two -- a shared 20-word sentence is ~16 shingles of
   a ~400-shingle story, containment ~0.04. 0.5 sits an order of magnitude
   above that and still catches a copy edited every ten words. The measured
   distribution (`NearDupResult.max_containment_by_phase`) is reported so the
   margin can be checked on the real corpus rather than assumed.

   Candidates are found through an inverted index of the exam shingles that
   occur in at most `MAX_POSTINGS` exam stories (a stock phrase posted to
   thousands of stories finds nothing a rarer shingle does not), and every
   candidate pair is then scored exactly. A pair can be missed only if every
   shingle it shares is posted to more than `MAX_POSTINGS` exam stories.

2. **Verbatim overlap per phase** -- train and exam stories share one
   per-activity fact bank by design, so a verbatim fact sentence in an exam
   story rewards memorised strings that the 200-character rule cannot see.
   `verbatim_overlap` reports, for every (exam phase, training phase) pair:
   the fraction of the exam phase's word `OVERLAP_N`-grams (8) found anywhere
   in that training phase, and the number and fraction of exam sentences of
   at least `MIN_SENTENCE_WORDS` (6) words found verbatim (normalised) in that
   training phase. exam-keeper compares tier 0's same-phase numbers with
   phases 3 and 6 (different fact banks) before freezing.

Text is tokenised by `words`: NFC, casefold, `\\w+` runs (apostrophes and
punctuation split words). Sentences split on `.`, `!`, `?` and newlines.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from training.guard import loader_lines

PREFIX_CHARS = 200
SHINGLE_N = 5
CONTAINMENT_THRESHOLD = 0.5
MAX_POSTINGS = 64
OVERLAP_N = 8
MIN_SENTENCE_WORDS = 6

_WORD = re.compile(r"\w+")
_SENTENCE_END = re.compile(r"[.!?\n]+")


# --------------------------------------------------------------------------- #
# tokenisation
# --------------------------------------------------------------------------- #


def words(text: str) -> list[str]:
    """NFC, casefold, `\\w+` runs."""
    return _WORD.findall(unicodedata.normalize("NFC", text).casefold())


def prefix_key(text: str, n_chars: int = PREFIX_CHARS) -> str:
    """The first `n_chars` of the text after NFC, casefold and whitespace
    normalisation -- the exam-keeper prefix rule's key."""
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())[:n_chars]


def shingles(text: str, n: int = SHINGLE_N) -> frozenset[tuple[str, ...]]:
    """Word n-gram shingles. A text shorter than n words is one shingle of
    all its words, so a short story is still comparable."""
    w = words(text)
    if not w:
        return frozenset()
    if len(w) < n:
        return frozenset([tuple(w)])
    return frozenset(tuple(w[i : i + n]) for i in range(len(w) - n + 1))


def sentences(text: str, min_words: int = MIN_SENTENCE_WORDS) -> list[tuple[str, ...]]:
    """Sentences as word tuples, those of at least `min_words` words."""
    out = []
    for part in _SENTENCE_END.split(text):
        w = tuple(words(part))
        if len(w) >= min_words:
            out.append(w)
    return out


def _row_id(row: Mapping) -> str:
    return str(row.get("prompt_hash") or row.get("id") or "?")


def _by_phase(rows: Iterable[Mapping]) -> dict[int, list[Mapping]]:
    out: dict[int, list[Mapping]] = defaultdict(list)
    for row in rows:
        out[int(row["phase"])].append(row)
    return dict(out)


# --------------------------------------------------------------------------- #
# 1. near duplicates
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class NearDuplicate:
    """One exam story that is a near copy of one training story. Ids only."""

    exam_id: str
    exam_phase: int
    train_id: str
    train_phase: int
    rule: str  # "prefix" | "shingle" | "prefix+shingle"
    containment: float
    jaccard: float


@dataclass
class NearDupResult:
    pairs: list[NearDuplicate] = field(default_factory=list)
    #: exam phase -> number of distinct exam stories flagged (the manifest's
    #: `near_duplicates_dropped`).
    dropped_by_phase: dict[int, int] = field(default_factory=dict)
    #: exam phase -> the largest containment any of its stories reached
    #: against any training story (flagged or not): the margin to the threshold.
    max_containment_by_phase: dict[int, float] = field(default_factory=dict)
    n_exam: int = 0
    n_train: int = 0

    @property
    def dropped_ids(self) -> set[str]:
        return {p.exam_id for p in self.pairs}

    def summary(self) -> dict:
        """Counts and numbers only, for the log and the report."""
        return {
            "n_exam": self.n_exam,
            "n_train": self.n_train,
            "rule": {
                "prefix_chars": PREFIX_CHARS,
                "shingle_n": SHINGLE_N,
                "containment_threshold": CONTAINMENT_THRESHOLD,
                "max_postings": MAX_POSTINGS,
            },
            "dropped_by_phase": {str(k): v for k, v in sorted(self.dropped_by_phase.items())},
            "max_containment_by_phase": {
                str(k): round(v, 4) for k, v in sorted(self.max_containment_by_phase.items())
            },
            "pairs": [p.__dict__ for p in self.pairs],
        }


def find_near_duplicates(
    exam_rows: Sequence[Mapping],
    train_rows: Sequence[Mapping],
    *,
    n: int = SHINGLE_N,
    threshold: float = CONTAINMENT_THRESHOLD,
    prefix_chars: int = PREFIX_CHARS,
    max_postings: int = MAX_POSTINGS,
) -> NearDupResult:
    """Every (exam, training) pair that is a near duplicate under either rule.

    `exam_rows` and `train_rows` are story lines (`prompt_hash` or `id`,
    `phase`, `story`). Training rows of every phase are compared with exam
    rows of every phase.
    """
    exam_sh = [shingles(r["story"], n) for r in exam_rows]
    exam_prefix: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(exam_rows):
        key = prefix_key(r["story"], prefix_chars)
        if key:
            exam_prefix[key].append(i)

    postings: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for i, sh in enumerate(exam_sh):
        for s in sh:
            postings[s].append(i)
    rare = {s: ids for s, ids in postings.items() if len(ids) <= max_postings}

    result = NearDupResult(n_exam=len(exam_rows), n_train=len(train_rows))
    best: dict[int, float] = {}
    flagged: dict[tuple[int, int], NearDuplicate] = {}
    for t_row in train_rows:
        t_sh = shingles(t_row["story"], n)
        candidates: set[int] = set()
        for s in t_sh:
            ids = rare.get(s)
            if ids:
                candidates.update(ids)
        prefix_hits = set(exam_prefix.get(prefix_key(t_row["story"], prefix_chars), ()))
        for i in candidates | prefix_hits:
            e_sh = exam_sh[i]
            inter = len(e_sh & t_sh)
            containment = inter / len(e_sh) if e_sh else 0.0
            union = len(e_sh | t_sh)
            jaccard = inter / union if union else 0.0
            best[i] = max(best.get(i, 0.0), containment)
            by_prefix = i in prefix_hits
            by_shingle = containment >= threshold
            if not (by_prefix or by_shingle):
                continue
            rule = "prefix+shingle" if by_prefix and by_shingle else ("prefix" if by_prefix else "shingle")
            e_row = exam_rows[i]
            pair = NearDuplicate(
                exam_id=_row_id(e_row),
                exam_phase=int(e_row["phase"]),
                train_id=_row_id(t_row),
                train_phase=int(t_row["phase"]),
                rule=rule,
                containment=round(containment, 4),
                jaccard=round(jaccard, 4),
            )
            flagged[(i, id(t_row))] = pair
    result.pairs = sorted(flagged.values(), key=lambda p: (p.exam_phase, p.exam_id, p.train_phase, p.train_id))

    phases = sorted({int(r["phase"]) for r in exam_rows})
    dropped: dict[int, set[str]] = {k: set() for k in phases}
    for p in result.pairs:
        dropped[p.exam_phase].add(p.exam_id)
    result.dropped_by_phase = {k: len(v) for k, v in dropped.items()}
    maxc: dict[int, float] = {k: 0.0 for k in phases}
    for i, c in best.items():
        k = int(exam_rows[i]["phase"])
        maxc[k] = max(maxc[k], c)
    result.max_containment_by_phase = maxc
    return result


def drop_near_duplicates(
    exam_rows: Sequence[Mapping], train_rows: Sequence[Mapping], **kwargs
) -> tuple[list[Mapping], NearDupResult]:
    """The exam rows with every near duplicate removed, and the result whose
    `dropped_by_phase` goes into the manifest's `near_duplicates_dropped`."""
    result = find_near_duplicates(exam_rows, train_rows, **kwargs)
    gone = result.dropped_ids
    return [r for r in exam_rows if _row_id(r) not in gone], result


# --------------------------------------------------------------------------- #
# 2. verbatim overlap per phase
# --------------------------------------------------------------------------- #


def _ngrams(w: Sequence[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(w[i : i + n]) for i in range(len(w) - n + 1)}


def verbatim_overlap(
    exam_rows: Sequence[Mapping],
    train_rows: Sequence[Mapping],
    *,
    n: int = OVERLAP_N,
    min_sentence_words: int = MIN_SENTENCE_WORDS,
) -> dict:
    """Verbatim overlap of each exam phase with each training phase.

    Returns `{"n": n, "min_sentence_words": m, "pairs": {exam_phase:
    {train_phase: {...}}}, "same_phase": {phase: {...}}}` where each cell is

      ngram_fraction      distinct exam n-grams found in that training phase
                          / distinct exam n-grams of the exam phase
      sentences_shared    exam sentences (>= m words) found verbatim in that
                          training phase, counted with multiplicity
      sentence_fraction   sentences_shared / exam sentences of the phase
      stories_with_shared_sentence   exam stories with at least one such
                          sentence, and `story_fraction` = that / n stories

    Counts and fractions only.
    """
    exam_by = _by_phase(exam_rows)
    train_by = _by_phase(train_rows)
    train_ngrams: dict[int, set] = {}
    train_sents: dict[int, set] = {}
    for k, rows in train_by.items():
        grams: set = set()
        sents: set = set()
        for r in rows:
            grams |= _ngrams(words(r["story"]), n)
            sents.update(sentences(r["story"], min_sentence_words))
        train_ngrams[k] = grams
        train_sents[k] = sents

    pairs: dict[int, dict[int, dict]] = {}
    for e, rows in sorted(exam_by.items()):
        e_grams: set = set()
        e_story_sents = []
        for r in rows:
            e_grams |= _ngrams(words(r["story"]), n)
            e_story_sents.append(sentences(r["story"], min_sentence_words))
        n_sents = sum(len(s) for s in e_story_sents)
        pairs[e] = {}
        for t in sorted(train_by):
            shared_grams = len(e_grams & train_ngrams[t])
            shared_sents = 0
            stories_hit = 0
            for ss in e_story_sents:
                hits = sum(1 for s in ss if s in train_sents[t])
                shared_sents += hits
                stories_hit += 1 if hits else 0
            pairs[e][t] = {
                "ngram_fraction": round(shared_grams / len(e_grams), 6) if e_grams else 0.0,
                "sentences_shared": shared_sents,
                "sentence_fraction": round(shared_sents / n_sents, 6) if n_sents else 0.0,
                "stories_with_shared_sentence": stories_hit,
                "story_fraction": round(stories_hit / len(rows), 6) if rows else 0.0,
                "n_exam_stories": len(rows),
                "n_exam_sentences": n_sents,
            }
    same = {e: pairs[e][e] for e in pairs if e in pairs[e]}
    return {"n": n, "min_sentence_words": min_sentence_words, "pairs": pairs, "same_phase": same}


# --------------------------------------------------------------------------- #
# CLI -- numbers only
# --------------------------------------------------------------------------- #


def _read_rows(paths: Iterable[Path]) -> list[dict]:
    rows = []
    for path in paths:
        for line in loader_lines(path.read_bytes().decode("utf-8")):
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main(argv: list[str] | None = None) -> int:
    """`python -m training.neardup --exam-glob ... --train-glob ...`

    Prints a JSON report of counts, ids and fractions (never text)."""
    p = argparse.ArgumentParser(description="Exam vs training near-duplicate and verbatim-overlap report.")
    p.add_argument("--exam-dir", type=Path, required=True, help="dir holding exam_phase_*.jsonl (searched recursively)")
    p.add_argument("--train-dir", type=Path, required=True, help="dir holding train_phase_*.jsonl")
    p.add_argument("--out", type=Path, default=None, help="write the JSON report here as well")
    args = p.parse_args(argv)
    exam = _read_rows(sorted(args.exam_dir.rglob("exam_phase_*.jsonl")))
    train = _read_rows(sorted(args.train_dir.glob("train_phase_*.jsonl")))
    report = {
        "near_duplicates": find_near_duplicates(exam, train).summary(),
        "verbatim_overlap": verbatim_overlap(exam, train),
    }
    text = json.dumps(report, indent=1, sort_keys=True, default=str)
    if args.out is not None:
        args.out.write_bytes((text + "\n").encode("utf-8"))
    print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
