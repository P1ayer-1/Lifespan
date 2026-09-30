"""The probe builder: cloze and continuation items to the frozen contract.

Owned by `exam-keeper` (AGENTS.md, Ownership: "the probe builder"). Built once,
from the exam stories (after near-duplicates are dropped, `training/neardup.py`)
and the phase lexicons, with the exam seed. Same stories, lexicons and seed:
same probes, byte for byte -- input order does not matter, because stories and
lexicons are sorted before anything is drawn.

Frozen contract (AGENTS.md, 2026-09-21 and amendment 2, 2026-09-22):

    cloze:        {id, phase, text_with_mask, answer, candidates: [20]}
      text_with_mask contains "[MASK]" exactly once; `answer` is the surface
      string and a member of `candidates`; the true answer plus 19 drawn from
      the same phase's lexicon with the exam seed.
      + story_id, story_sha256 (ADDED 2026-09-30, for the lead to freeze): the
      source story, so a cloze item's provenance is checkable by hash too.
      The evaluator ignores keys it does not read.
    continuation: {id, phase, prefix, options: [4], answer_index,
                   distractor_phases: [3],
                   option_sources: [{story_id, story_sha256, phase} x4]}
      distractors from other phases' exam stories only, never the same story.

How items are made (fixed here, the same for every phase):

* **cloze** -- the answer is a lexicon entry that occurs in the story exactly
  once as a whole word (case-insensitively; the occurrence must match the
  lexicon form exactly), so the masked word is never visible elsewhere in the
  text. `text_with_mask` is the whole story with that occurrence replaced.
  The 19 other candidates are drawn from the phase lexicon, excluding the
  answer, case variants and inflections of it (`related`), and duplicates.
  Candidate order is shuffled with the seed.
* **continuation** -- a story is split into paragraphs (blank lines; single
  newlines if it has no blank line). A split point i in 1..len-1 is drawn;
  `prefix` is paragraphs[:i] joined and followed by the separator, the answer
  is paragraph i. Three distractor phases are drawn from the other phases
  (with fewer than three other phases, as in a 0/3/6 exam, the three are
  spread over them as evenly as possible, from different stories where
  possible; owner, 2026-09-30); from each, one non-opening paragraph of one
  exam story, preferring
  paragraphs whose length is within `LENGTH_BAND` (0.5x-2x) of the answer's,
  so length is not a giveaway. `answer_index` is balanced over the four slots
  (a seeded shuffle of 0,1,2,3,0,1,2,3,...).

`option_sources[*].story_sha256` is `training.guard.story_sha256` of the
source story -- the hash the manifest's `stories` entries carry -- and
`story_id` is the manifest's `story_id` for that story.

`verify_probes` is the freeze-time assertion amendment 2 requires: every
option's source hash is an exam story in the manifest (with that phase) and in
no training file, no distractor shares the item's phase or story, and
`distractor_phases` is non-empty and matches the sources. `giveaway_report`
gives the numbers exam-keeper reports (length ratio, answer slot histogram,
masked word visible elsewhere). Neither prints text.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Collection, Iterable, Mapping, Sequence

from training.guard import manifest_story_entry, story_sha256

MASK = "[MASK]"
N_CANDIDATES = 20
N_OPTIONS = 4
N_DISTRACTORS = N_OPTIONS - 1
LENGTH_BAND = (0.5, 2.0)


class ProbeError(ValueError):
    """The probes cannot be built, or fail the freeze-time checks."""


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def derive_rng(seed: int, *labels: object) -> random.Random:
    """A `random.Random` seeded from the exam seed and a label path, via
    SHA-256, so it does not depend on Python's hash randomisation."""
    key = "|".join([str(seed), *map(str, labels)]).encode("utf-8")
    return random.Random(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))


def story_id_of(row: Mapping) -> str:
    """The manifest's `story_id` for an exam story line."""
    return manifest_story_entry(dict(row))["story_id"]


def _sorted_stories(rows: Iterable[Mapping]) -> list[Mapping]:
    return sorted(rows, key=story_id_of)


def split_paragraphs(story: str) -> tuple[list[str], str]:
    """(paragraphs, separator). Blank-line paragraphs if the story has any
    blank line, else one paragraph per non-empty line."""
    text = story.replace("\r\n", "\n").replace("\r", "\n").strip()
    if re.search(r"\n\s*\n", text):
        paras, sep = re.split(r"\n\s*\n", text), "\n\n"
    else:
        paras, sep = text.split("\n"), "\n"
    return [p.strip() for p in paras if p.strip()], sep


def _stem(word: str) -> str:
    w = word.casefold()
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    for suffix in ("ing", "est", "ed", "er", "es", "ly", "s"):
        if w.endswith(suffix) and len(w) - len(suffix) >= 3 and not (suffix == "s" and w.endswith("ss")):
            w = w[: -len(suffix)]
            if len(w) >= 3 and w[-1] == w[-2]:  # running -> runn -> run
                w = w[:-1]
            break
    if len(w) > 3 and w.endswith("e"):  # make / making
        w = w[:-1]
    return w


def related(a: str, b: str) -> bool:
    """Same word up to case or a common English inflection (conservative: a
    false 'related' only removes a candidate)."""
    return a.casefold() == b.casefold() or _stem(a) == _stem(b)


def _word_re(word: str, flags: int = 0) -> re.Pattern:
    return re.compile(r"(?<!\w)" + re.escape(word) + r"(?!\w)", flags)


def _dedupe_lexicon(lexicon: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for w in sorted({str(w).strip() for w in lexicon if str(w).strip()}):
        if w.casefold() not in seen:
            seen.add(w.casefold())
            out.append(w)
    return out


def _check_phase_rows(stories_by_phase: Mapping[int, Sequence[Mapping]]) -> None:
    for phase, rows in stories_by_phase.items():
        for r in rows:
            if int(r["phase"]) != int(phase):
                raise ProbeError(
                    f"story {story_id_of(r)} has phase {r['phase']} but was given as phase {phase}"
                )
            if MASK in r["story"]:
                raise ProbeError(f"story {story_id_of(r)} contains the literal {MASK}")


# --------------------------------------------------------------------------- #
# cloze
# --------------------------------------------------------------------------- #


def build_cloze(
    stories_by_phase: Mapping[int, Sequence[Mapping]],
    lexicons: Mapping[int, Iterable[str]],
    seed: int,
    n_per_phase: int | None = None,
) -> dict[int, list[dict]]:
    """Cloze items per phase. `n_per_phase=None` makes one per usable story."""
    _check_phase_rows(stories_by_phase)
    out: dict[int, list[dict]] = {}
    for phase in sorted(stories_by_phase):
        lexicon = _dedupe_lexicon(lexicons.get(phase, ()))
        if len(lexicon) < N_CANDIDATES:
            raise ProbeError(
                f"phase {phase} lexicon has {len(lexicon)} distinct entries; {N_CANDIDATES} are needed"
            )
        rng = derive_rng(seed, "cloze", phase)
        loose = {w: _word_re(w, re.IGNORECASE) for w in lexicon}
        exact = {w: _word_re(w) for w in lexicon}
        pools: dict[str, list[str]] = {}

        def pool_of(word: str) -> list[str]:
            if word not in pools:
                pools[word] = [w for w in lexicon if not related(w, word)]
            return pools[word]

        rows = _sorted_stories(stories_by_phase[phase])
        rng.shuffle(rows)
        items: list[dict] = []
        for row in rows:
            if n_per_phase is not None and len(items) >= n_per_phase:
                break
            story = row["story"]
            eligible = [
                w for w in lexicon
                if len(loose[w].findall(story)) == 1 and exact[w].search(story)
                and len(pool_of(w)) >= N_CANDIDATES - 1
            ]
            if not eligible:
                continue
            answer = eligible[rng.randrange(len(eligible))]
            candidates = [answer] + rng.sample(pool_of(answer), N_CANDIDATES - 1)
            rng.shuffle(candidates)
            match = exact[answer].search(story)
            text = story[: match.start()] + MASK + story[match.end() :]
            items.append(
                {
                    "id": f"cloze-p{phase}-{len(items):04d}",
                    "phase": phase,
                    "text_with_mask": text,
                    "answer": answer,
                    "candidates": candidates,
                    "story_id": story_id_of(row),
                    "story_sha256": story_sha256(story),
                }
            )
        if n_per_phase is not None and len(items) < n_per_phase:
            raise ProbeError(f"phase {phase}: only {len(items)} cloze items possible, {n_per_phase} asked")
        out[phase] = items
    return out


# --------------------------------------------------------------------------- #
# continuation
# --------------------------------------------------------------------------- #


def _distractor_phases(others: Sequence[int], rng: random.Random) -> list[int]:
    """The phases the three distractors come from, never the item's own.

    With at least three other phases: three distinct ones, drawn exactly as
    before (so a full 0..6 exam is byte-identical). With fewer (the 0/3/6
    pre-pilot; owner, 2026-09-30): the three are spread over the other phases
    as evenly as possible (2+1 for two, 3 for one) in a seeded order, so the
    item keeps four options and chance stays 1/4.
    """
    if len(others) >= N_DISTRACTORS:
        return rng.sample(list(others), N_DISTRACTORS)
    k, extra = divmod(N_DISTRACTORS, len(others))
    phases = list(others) * k + rng.sample(list(others), extra)
    rng.shuffle(phases)
    return phases


def build_continuation(
    stories_by_phase: Mapping[int, Sequence[Mapping]],
    seed: int,
    n_per_phase: int | None = None,
) -> dict[int, list[dict]]:
    """Continuation items per phase. `n_per_phase=None` makes one per story
    with at least two paragraphs."""
    _check_phase_rows(stories_by_phase)
    # every phase's non-opening paragraphs: the distractor pool
    pool: dict[int, list[tuple[Mapping, str]]] = {}
    for phase in sorted(stories_by_phase):
        entries = []
        for row in _sorted_stories(stories_by_phase[phase]):
            paras, _ = split_paragraphs(row["story"])
            entries.extend((row, p) for p in paras[1:])
        pool[phase] = entries

    out: dict[int, list[dict]] = {}
    for phase in sorted(stories_by_phase):
        others = [p for p in sorted(pool) if p != phase and pool[p]]
        if not others:
            raise ProbeError(
                f"phase {phase}: distractors need at least one other phase with multi-paragraph "
                f"exam stories, found none"
            )
        rng = derive_rng(seed, "continuation", phase)
        rows = [r for r in _sorted_stories(stories_by_phase[phase]) if len(split_paragraphs(r["story"])[0]) >= 2]
        rng.shuffle(rows)
        if n_per_phase is not None:
            if len(rows) < n_per_phase:
                raise ProbeError(
                    f"phase {phase}: only {len(rows)} multi-paragraph stories, {n_per_phase} items asked"
                )
            rows = rows[:n_per_phase]
        slots = [i % N_OPTIONS for i in range(len(rows))]
        rng.shuffle(slots)

        items: list[dict] = []
        for row, answer_index in zip(rows, slots):
            paras, sep = split_paragraphs(row["story"])
            cut = rng.randrange(1, len(paras))
            prefix = sep.join(paras[:cut]) + sep
            answer = paras[cut]
            dphases = _distractor_phases(others, rng)
            distractors: list[tuple[Mapping, str]] = []
            for dp in dphases:
                lo, hi = LENGTH_BAND[0] * len(answer), LENGTH_BAND[1] * len(answer)
                usable = [(r, p) for r, p in pool[dp] if p != answer and all(p != d for _, d in distractors)]
                # Two distractors from one phase (a subset exam) come from two
                # stories where possible; with one per phase this is a no-op.
                used = {story_id_of(r) for r, _ in distractors}
                fresh = [(r, p) for r, p in usable if story_id_of(r) not in used] or usable
                banded = [(r, p) for r, p in fresh if lo <= len(p) <= hi]
                choice_from = banded or fresh
                if not choice_from:
                    raise ProbeError(f"phase {phase}: no distractor paragraph left in phase {dp}")
                distractors.append(choice_from[rng.randrange(len(choice_from))])

            options: list[str] = []
            sources: list[dict] = []
            it = iter(distractors)
            for slot in range(N_OPTIONS):
                src_row, text = (row, answer) if slot == answer_index else next(it)
                options.append(text)
                sources.append(
                    {
                        "story_id": story_id_of(src_row),
                        "story_sha256": story_sha256(src_row["story"]),
                        "phase": int(src_row["phase"]),
                    }
                )
            items.append(
                {
                    "id": f"continuation-p{phase}-{len(items):04d}",
                    "phase": phase,
                    "prefix": prefix,
                    "options": options,
                    "answer_index": answer_index,
                    "distractor_phases": [s["phase"] for i, s in enumerate(sources) if i != answer_index],
                    "option_sources": sources,
                }
            )
        out[phase] = items
    return out


# --------------------------------------------------------------------------- #
# freeze-time checks and giveaway numbers
# --------------------------------------------------------------------------- #


def verify_probes(
    cloze: Mapping[int, Sequence[dict]],
    continuation: Mapping[int, Sequence[dict]],
    *,
    exam_story_phases: Mapping[str, int],
    train_story_hashes: Collection[str],
) -> None:
    """Raise `ProbeError` on the first violation of the contract or of the
    provenance rules (AGENTS.md amendment 2). Messages carry ids and hashes,
    never text.

    `exam_story_phases` is the manifest's per-story map (normalised story
    sha256 -> phase, `training.guard.manifest_story_phases`);
    `train_story_hashes` the normalised hashes of every training story.
    """
    train_story_hashes = set(train_story_hashes)
    ids: set[str] = set()

    def new_id(item: dict) -> str:
        iid = str(item.get("id"))
        if iid in ids:
            raise ProbeError(f"duplicate item id {iid}")
        ids.add(iid)
        return iid

    for phase, items in cloze.items():
        for item in items:
            iid = new_id(item)
            if item["phase"] != phase:
                raise ProbeError(f"cloze {iid}: phase {item['phase']} filed under {phase}")
            text, answer, cands = item["text_with_mask"], item["answer"], list(item["candidates"])
            if text.count(MASK) != 1:
                raise ProbeError(f"cloze {iid}: {MASK} occurs {text.count(MASK)} times, not once")
            if len(cands) != N_CANDIDATES:
                raise ProbeError(f"cloze {iid}: {len(cands)} candidates, not {N_CANDIDATES}")
            if answer not in cands:
                raise ProbeError(f"cloze {iid}: answer is not among the candidates")
            if len({c.casefold() for c in cands}) != len(cands):
                raise ProbeError(f"cloze {iid}: duplicate candidates")
            if any(c != answer and related(c, answer) for c in cands):
                raise ProbeError(f"cloze {iid}: a candidate is an inflection of the answer")
            if _word_re(answer, re.IGNORECASE).search(text.replace(MASK, " ")):
                raise ProbeError(f"cloze {iid}: the answer is visible elsewhere in the text")
            sha = item.get("story_sha256")
            if sha is not None:
                if exam_story_phases.get(sha) != phase:
                    raise ProbeError(f"cloze {iid}: source story {sha} is not a phase-{phase} exam story")
                if sha in train_story_hashes:
                    raise ProbeError(f"cloze {iid}: source story {sha} is a training story")

    for phase, items in continuation.items():
        for item in items:
            iid = new_id(item)
            if item["phase"] != phase:
                raise ProbeError(f"continuation {iid}: phase {item['phase']} filed under {phase}")
            options, sources = item["options"], item.get("option_sources")
            k = item["answer_index"]
            if len(options) != N_OPTIONS:
                raise ProbeError(f"continuation {iid}: {len(options)} options, not {N_OPTIONS}")
            if isinstance(k, bool) or not isinstance(k, int) or not 0 <= k < len(options):
                raise ProbeError(f"continuation {iid}: answer_index {k!r} out of range")
            if not isinstance(sources, list) or len(sources) != len(options):
                raise ProbeError(f"continuation {iid}: option_sources must have one entry per option")
            for i, s in enumerate(sources):
                for f in ("story_id", "story_sha256", "phase"):
                    if f not in s:
                        raise ProbeError(f"continuation {iid}: option_sources[{i}] missing {f}")
                sha = s["story_sha256"]
                if sha not in exam_story_phases:
                    raise ProbeError(
                        f"continuation {iid}: option_sources[{i}].story_sha256 {sha} is in no exam manifest"
                    )
                if exam_story_phases[sha] != s["phase"]:
                    raise ProbeError(
                        f"continuation {iid}: option_sources[{i}] claims phase {s['phase']}, the manifest "
                        f"says {exam_story_phases[sha]}"
                    )
                if sha in train_story_hashes:
                    raise ProbeError(f"continuation {iid}: option_sources[{i}] {sha} is a training story")
            ans = sources[k]
            if ans["phase"] != phase:
                raise ProbeError(f"continuation {iid}: the answer's source is phase {ans['phase']}, not {phase}")
            dist = [s for i, s in enumerate(sources) if i != k]
            for s in dist:
                if s["phase"] == phase:
                    raise ProbeError(f"continuation {iid}: a distractor comes from the item's own phase")
                if s["story_id"] == ans["story_id"]:
                    raise ProbeError(f"continuation {iid}: a distractor comes from the answer's story")
            dp = item.get("distractor_phases")
            if not dp:
                raise ProbeError(f"continuation {iid}: distractor_phases is empty")
            if list(dp) != [s["phase"] for s in dist]:
                raise ProbeError(f"continuation {iid}: distractor_phases do not match option_sources")
            if len({o for o in options}) != len(options):
                raise ProbeError(f"continuation {iid}: duplicate options")


def giveaway_report(
    cloze: Mapping[int, Sequence[dict]], continuation: Mapping[int, Sequence[dict]]
) -> dict[int, dict]:
    """Per phase, the numbers exam-keeper checks before freezing: counts, the
    answer-slot histogram, the answer / mean-distractor length ratio, how often
    the answer is the longest option, and cloze answers visible elsewhere."""
    out: dict[int, dict] = {}
    for phase in sorted(set(cloze) | set(continuation)):
        c_items = list(cloze.get(phase, ()))
        k_items = list(continuation.get(phase, ()))
        ratios, longest = [], 0
        slots: Counter = Counter()
        for it in k_items:
            k = it["answer_index"]
            slots[k] += 1
            lens = [len(o) for o in it["options"]]
            others = [n for i, n in enumerate(lens) if i != k]
            ratios.append(lens[k] / (sum(others) / len(others)) if sum(others) else float("inf"))
            longest += 1 if lens[k] > max(others) else 0
        visible = sum(
            1 for it in c_items if _word_re(it["answer"], re.IGNORECASE).search(it["text_with_mask"].replace(MASK, " "))
        )
        out[phase] = {
            "n_cloze": len(c_items),
            "n_continuation": len(k_items),
            "answer_slot_counts": [slots.get(i, 0) for i in range(N_OPTIONS)],
            "mean_length_ratio": round(sum(ratios) / len(ratios), 4) if ratios else None,
            "answer_longest_fraction": round(longest / len(k_items), 4) if k_items else None,
            "cloze_answer_visible_elsewhere": visible,
        }
    return out


def write_probes(
    exam_dir: Path, cloze: Mapping[int, Sequence[dict]], continuation: Mapping[int, Sequence[dict]]
) -> list[Path]:
    """`probes/{cloze,continuation}_phase_{k}.jsonl` under the exam dir (the
    layout `training/evaluate.py` reads): UTF-8, `\\n` newlines, one object per
    line, so the manifest hash is the same on Windows and on Kaggle."""
    written = []
    for kind, by_phase in (("cloze", cloze), ("continuation", continuation)):
        for phase, items in sorted(by_phase.items()):
            path = Path(exam_dir) / "probes" / f"{kind}_phase_{phase}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            data = "".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items)
            path.write_bytes(data.encode("utf-8"))
            written.append(path)
    return written
