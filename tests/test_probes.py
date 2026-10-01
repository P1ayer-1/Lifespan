"""The probe builder (training/probes.py) on synthetic fixtures only.

Every story and lexicon here is invented by a seeded generator in this file.
No test reads the exam directory.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path

import pytest

from training.guard import story_sha256
from training.probes import (
    MASK,
    N_CANDIDATES,
    ProbeError,
    build_cloze,
    build_continuation,
    giveaway_report,
    related,
    split_paragraphs,
    story_id_of,
    verify_probes,
    write_probes,
)

N_PHASES = 7
SEED = 20260930


def lexicon(phase: int) -> list[str]:
    return [f"lex{phase}q{j}" for j in range(30)]


def make_story(phase: int, i: int, n_paras: int = 4) -> str:
    rng = random.Random(f"{phase}:{i}")
    lex = lexicon(phase)
    once = rng.sample(lex, 3)
    twice = [w for w in lex if w not in once][0]
    paras = []
    for p in range(n_paras):
        sent_words = [f"p{phase}w{rng.randrange(200)}" for _ in range(rng.randrange(12, 30))]
        if p < len(once):
            sent_words.insert(rng.randrange(len(sent_words)), once[p])
        if p in (1, 2):
            sent_words.append(twice)  # a word seen twice is never a cloze answer
        paras.append(" ".join(sent_words).capitalize() + ".")
    return "\n\n".join(paras) + "\n"


def stories(n: int = 10, phases: int = N_PHASES) -> dict[int, list[dict]]:
    return {
        k: [
            {"prompt_hash": f"h{k}{i:05d}", "phase": k, "story": make_story(k, i), "model": "m"}
            for i in range(n)
        ]
        for k in range(phases)
    }


def words_of(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.casefold()))


def manifest_map(by_phase: dict[int, list[dict]]) -> dict[str, int]:
    return {story_sha256(r["story"]): k for k, rows in by_phase.items() for r in rows}


@pytest.fixture(scope="module")
def built():
    s = stories()
    lex = {k: lexicon(k) for k in s}
    return s, build_cloze(s, lex, SEED), build_continuation(s, SEED)


# --------------------------------------------------------------------------- #
# the frozen contract


def test_cloze_items_meet_the_frozen_contract(built):
    s, cloze, _ = built
    by_id = {story_id_of(r): r for rows in s.values() for r in rows}
    for phase, items in cloze.items():
        assert len(items) == 10
        for it in items:
            assert set(it) >= {"id", "phase", "text_with_mask", "answer", "candidates"}
            assert it["phase"] == phase
            assert it["text_with_mask"].count(MASK) == 1
            assert len(it["candidates"]) == N_CANDIDATES == 20
            assert it["answer"] in it["candidates"]
            assert set(it["candidates"]) <= set(lexicon(phase))
            assert len(set(it["candidates"])) == 20
            src = by_id[it["story_id"]]
            assert it["text_with_mask"].replace(MASK, it["answer"]) == src["story"]
            assert it["story_sha256"] == story_sha256(src["story"])
            # the masked word is not visible anywhere else in the text
            assert it["answer"].casefold() not in words_of(it["text_with_mask"])


def test_continuation_items_meet_the_frozen_contract(built):
    s, _, cont = built
    by_id = {story_id_of(r): r for rows in s.values() for r in rows}
    for phase, items in cont.items():
        assert len(items) == 10
        for it in items:
            assert set(it) == {
                "id", "phase", "prefix", "options", "answer_index", "distractor_phases", "option_sources"
            }
            k = it["answer_index"]
            assert len(it["options"]) == 4 and 0 <= k < 4
            src = it["option_sources"]
            assert len(src) == 4 and src[k]["phase"] == phase
            # the true continuation is the story's next paragraph
            answer_story = by_id[src[k]["story_id"]]["story"]
            assert answer_story.startswith(it["prefix"] + it["options"][k])
            dist = [x for i, x in enumerate(src) if i != k]
            assert it["distractor_phases"] == [x["phase"] for x in dist]
            # Amendment 5: three other stories of the item's own phase
            assert it["distractor_phases"] == [phase] * 3
            assert len({x["story_id"] for x in dist} | {src[k]["story_id"]}) == 4
            for i, x in enumerate(src):
                story = by_id[x["story_id"]]
                assert x["story_sha256"] == story_sha256(story["story"])
                assert x["phase"] == story["phase"]
                assert it["options"][i] in split_paragraphs(story["story"])[0]
                if i != k:  # a distractor is never an opening paragraph
                    assert it["options"][i] != split_paragraphs(story["story"])[0][0]


def test_the_evaluator_accepts_the_built_items_against_the_manifest_hashes(built):
    from training.evaluate import validate_continuation_item

    s, _, cont = built
    known = manifest_map(s)
    for items in cont.values():
        for it in items:
            assert validate_continuation_item(it, known) == 4


def test_answer_slots_are_balanced(built):
    _, _, cont = built
    for items in cont.values():
        counts = Counter(it["answer_index"] for it in items)
        assert max(counts.values()) - min(counts.get(i, 0) for i in range(4)) <= 1


# --------------------------------------------------------------------------- #
# determinism


def _bytes(tmp: Path, cloze, cont) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in write_probes(tmp, cloze, cont)}


def test_same_stories_and_seed_give_the_same_bytes_in_any_input_order(tmp_path, built):
    s, cloze, cont = built
    shuffled = {k: list(reversed(v)) for k, v in reversed(list(s.items()))}
    lex = {k: list(reversed(lexicon(k))) for k in s}
    a = _bytes(tmp_path / "a", cloze, cont)
    b = _bytes(tmp_path / "b", build_cloze(shuffled, lex, SEED), build_continuation(shuffled, SEED))
    assert a == b and len(a) == 14
    assert all(b"\r\n" not in v for v in a.values())
    c = _bytes(tmp_path / "c", build_cloze(s, lex, SEED + 1), build_continuation(s, SEED + 1))
    assert c != a


# --------------------------------------------------------------------------- #
# giveaways


def test_inflections_of_the_answer_are_never_candidates():
    assert related("jump", "Jumping") and related("jumped", "jumps") and related("baby", "babies")
    assert related("run", "running") and related("make", "making")
    assert not related("jump", "junk")
    story = {"prompt_hash": "only", "phase": 0, "story": "The frog was jumping all day.\n\nThen it slept.\n"}
    lex = ["jump", "jumps", "jumped", "jumping"] + [f"filler{j}" for j in range(20)]
    (item,) = build_cloze({0: [story]}, {0: lex}, SEED)[0]
    assert item["answer"] == "jumping"
    assert not {"jump", "jumps", "jumped"} & set(item["candidates"])


def test_a_word_that_occurs_twice_is_never_the_answer():
    story = {"prompt_hash": "x", "phase": 0, "story": "Alpha saw beta. Alpha left.\n"}
    lex = ["Alpha", "beta"] + [f"filler{j}" for j in range(20)]
    (item,) = build_cloze({0: [story]}, {0: lex}, SEED)[0]
    assert item["answer"] == "beta"


def test_giveaway_report_numbers(built):
    _, cloze, cont = built
    report = giveaway_report(cloze, cont)
    for phase, r in report.items():
        assert r["n_cloze"] == r["n_continuation"] == 10
        assert sum(r["answer_slot_counts"]) == 10
        assert r["cloze_answer_visible_elsewhere"] == 0
        assert 0.3 < r["mean_length_ratio"] < 3.0


# --------------------------------------------------------------------------- #
# freeze-time verification: clean passes, every attack refused


def test_verify_passes_clean_probes(built):
    s, cloze, cont = built
    verify_probes(cloze, cont, exam_story_phases=manifest_map(s), train_story_hashes=set())


def _tampered(built, change):
    s, cloze, cont = built
    cloze = json.loads(json.dumps(cloze))
    cont = json.loads(json.dumps(cont))
    cloze = {int(k): v for k, v in cloze.items()}
    cont = {int(k): v for k, v in cont.items()}
    change(cloze, cont)
    return s, cloze, cont


@pytest.mark.parametrize(
    "change,match",
    [
        (lambda c, k: k[2][0]["option_sources"][1 if k[2][0]["answer_index"] != 1 else 2].update(
            story_sha256="0" * 64), "is in no exam manifest"),
        (lambda c, k: k[2][0]["option_sources"][k[2][0]["answer_index"]].update(phase=3), "claims phase 3"),
        (lambda c, k: k[1][0].update(distractor_phases=[]), "distractor_phases is empty"),
        (lambda c, k: k[1][0].update(answer_index=4), "out of range"),
        (lambda c, k: c[0][0].update(text_with_mask=c[0][0]["text_with_mask"] + " " + MASK), "occurs 2 times"),
        (lambda c, k: c[0][0].update(candidates=c[0][0]["candidates"][:19] if c[0][0]["candidates"][19] != c[0][0]["answer"] else c[0][0]["candidates"][1:]), "19 candidates"),
        (lambda c, k: c[3][1].update(id=c[3][0]["id"]), "duplicate item id"),
    ],
)
def test_verify_refuses_tampered_probes(built, change, match):
    s, cloze, cont = _tampered(built, change)
    with pytest.raises(ProbeError, match=match):
        verify_probes(cloze, cont, exam_story_phases=manifest_map(s), train_story_hashes=set())


def test_verify_refuses_a_source_that_is_a_training_story(built):
    s, cloze, cont = built
    item = cont[4][0]
    distractor = item["option_sources"][(item["answer_index"] + 1) % 4]["story_sha256"]
    with pytest.raises(ProbeError, match="is a training story"):
        verify_probes(cloze, cont, exam_story_phases=manifest_map(s), train_story_hashes={distractor})


def test_verify_refuses_a_distractor_from_another_phase(built):
    """Amendment 5 (2026-10-01): a distractor from another phase lets the item
    be answered by register alone."""
    s, cloze, cont = _tampered(built, lambda c, k: None)
    item = cont[5][0]
    k = item["answer_index"]
    d = (k + 1) % 4
    other = s[4][0]
    item["option_sources"][d] = {"story_id": story_id_of(other), "story_sha256": story_sha256(other["story"]), "phase": 4}
    item["distractor_phases"] = [x["phase"] for i, x in enumerate(item["option_sources"]) if i != k]
    with pytest.raises(ProbeError, match="not the item's own phase"):
        verify_probes(cloze, cont, exam_story_phases=manifest_map(s), train_story_hashes=set())


def test_verify_refuses_two_distractors_from_one_story(built):
    s, cloze, cont = _tampered(built, lambda c, k: None)
    item = cont[5][0]
    k = item["answer_index"]
    a, b = [i for i in range(4) if i != k][:2]
    item["option_sources"][b] = dict(item["option_sources"][a])
    with pytest.raises(ProbeError, match="two distractors come from one story"):
        verify_probes(cloze, cont, exam_story_phases=manifest_map(s), train_story_hashes=set())


# --------------------------------------------------------------------------- #
# refusals at build time


def test_a_lexicon_under_twenty_words_is_refused():
    with pytest.raises(ProbeError, match="20 are needed"):
        build_cloze(stories(2, 1), {0: lexicon(0)[:19]}, SEED)


def test_continuation_needs_four_stories_in_the_phase():
    """The answer's story plus three distinct distractor stories."""
    with pytest.raises(ProbeError, match="fewer than 4 exam stories"):
        build_continuation(stories(3, 1), SEED)
    assert build_continuation(stories(4, 1), SEED)[0]


def _subset(phases: tuple[int, ...], n: int = 10) -> dict[int, list[dict]]:
    full = stories(n, max(phases) + 1)
    return {k: full[k] for k in phases}


def test_a_subset_exam_draws_every_distractor_from_the_items_own_phase():
    """The 0/3/6 pre-pilot: the other phases are never used (Amendment 5), so a
    subset exam's items are built exactly as a full exam's."""
    s = _subset((0, 3, 6))
    cont = build_continuation(s, SEED)
    for phase, items in cont.items():
        assert items
        for item in items:
            assert len(item["options"]) == 4 and len(set(item["options"])) == 4
            assert item["distractor_phases"] == [phase] * 3
            dist = [src for i, src in enumerate(item["option_sources"]) if i != item["answer_index"]]
            assert len({src["story_id"] for src in dist}) == 3
    full = build_continuation(stories(10, 7), SEED)
    assert cont[3] == full[3]  # same phase stories, same items
    lex = {k: lexicon(k) for k in s}
    verify_probes(build_cloze(s, lex, SEED), cont, exam_story_phases=manifest_map(s), train_story_hashes=set())


def test_a_subset_exam_is_deterministic():
    s = _subset((0, 3, 6))
    assert build_continuation(s, SEED) == build_continuation(s, SEED)


def test_a_story_filed_under_the_wrong_phase_is_refused():
    s = stories(2, 4)
    s[1].append(dict(s[2][0]))
    with pytest.raises(ProbeError, match="was given as phase 1"):
        build_continuation(s, SEED)


def test_asking_for_more_items_than_stories_is_refused():
    with pytest.raises(ProbeError, match="items asked"):
        build_continuation(stories(3, 4), SEED, n_per_phase=5)


# --------------------------------------------------------------------------- #
# end to end: the evaluator scores what the builder wrote


def test_the_evaluator_scores_the_built_probes(tmp_path):
    from tests.conftest import make_model
    from training.config import TOY
    from training.evaluate import EvalConfig, evaluate_all_detailed

    s = stories(4)
    exam = tmp_path / "exam"
    (exam / "stories").mkdir(parents=True)
    for k, rows in s.items():
        (exam / "stories" / f"exam_phase_{k}.jsonl").write_bytes(
            "".join(json.dumps(r) + "\n" for r in rows).encode("utf-8"))
    write_probes(exam, build_cloze(s, {k: lexicon(k) for k in s}, SEED), build_continuation(s, SEED))
    result = evaluate_all_detailed(
        make_model(), exam, [0, 3], cfg=EvalConfig(block_size=TOY.block_size, batch_size=64)
    )
    assert result.n_items["cloze"][0] == 4 and result.n_items["continuation"][3] == 4
    assert result.chance["cloze"][0] == pytest.approx(0.05)
    assert result.chance["continuation"][3] == pytest.approx(0.25)
