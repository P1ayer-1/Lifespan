"""The near-duplicate check and the verbatim-overlap report (training/neardup.py).

Synthetic stories only: seeded draws from an invented vocabulary. No test here
reads the exam directory.
"""

from __future__ import annotations

import json
import random

import pytest

from training.neardup import (
    CONTAINMENT_THRESHOLD,
    drop_near_duplicates,
    find_near_duplicates,
    main,
    prefix_key,
    sentences,
    shingles,
    verbatim_overlap,
    words,
)

VOCAB = [f"w{i}" for i in range(400)]


def random_story(seed: int, n_sentences: int = 30, words_per: int = 10) -> str:
    rng = random.Random(seed)
    sents = [" ".join(rng.choice(VOCAB) for _ in range(words_per)).capitalize() + "." for _ in range(n_sentences)]
    return "\n".join(sents) + "\n"


def row(pid: str, phase: int, story: str) -> dict:
    return {"prompt_hash": pid, "phase": phase, "story": story, "model": "m"}


def edit_every(story: str, k: int, seed: int = 0) -> str:
    """Replace every k-th word with a word no story uses."""
    rng = random.Random(seed)
    out, n = [], 0
    for token in story.split(" "):
        n += 1
        out.append(f"edit{rng.randrange(10**6)}" if n % k == 0 else token)
    return " ".join(out)


@pytest.fixture
def corpus():
    exam = [row(f"e{p}{i}", p, random_story(1000 * p + i)) for p in range(3) for i in range(5)]
    train = [row(f"t{p}{i}", p, random_story(50_000 + 1000 * p + i)) for p in range(3) for i in range(20)]
    return exam, train


def test_tokenisation_is_case_and_whitespace_blind():
    assert words("The  Cat\r\nsat.") == ["the", "cat", "sat"]
    assert prefix_key("  Hello\r\n  World ") == "hello world"
    assert shingles("a b", n=5) == frozenset([("a", "b")])
    assert len(shingles("a b c d e f", n=5)) == 2
    assert sentences("One two three four five six. Short one.\nseven eight nine ten eleven twelve!") == [
        ("one", "two", "three", "four", "five", "six"),
        ("seven", "eight", "nine", "ten", "eleven", "twelve"),
    ]


def test_independent_stories_are_not_near_duplicates(corpus):
    exam, train = corpus
    result = find_near_duplicates(exam, train)
    assert result.pairs == []
    assert result.dropped_by_phase == {0: 0, 1: 0, 2: 0}
    assert all(c < 0.1 for c in result.max_containment_by_phase.values())


def test_an_exact_copy_is_caught_by_both_rules(corpus):
    exam, train = corpus
    train = train + [row("tcopy", 1, exam[7]["story"])]
    result = find_near_duplicates(exam, train)
    assert [(p.exam_id, p.train_id, p.rule, p.containment) for p in result.pairs] == [
        (exam[7]["prompt_hash"], "tcopy", "prefix+shingle", 1.0)
    ]


def test_a_near_copy_with_small_edits_is_caught(corpus):
    """The audit's attack: small edits defeat any exact hash. An edit every
    20th word -- including the first sentence, so the prefix rule misses --
    keeps ~75% of the shingles."""
    exam, train = corpus
    edited = "Changed opening. " + edit_every(exam[3]["story"], 20)
    assert prefix_key(edited) != prefix_key(exam[3]["story"])
    result = find_near_duplicates(exam, train + [row("tedit", 0, edited)])
    (pair,) = result.pairs
    assert pair.exam_id == exam[3]["prompt_hash"] and pair.rule == "shingle"
    assert 0.6 < pair.containment < 0.9
    assert result.dropped_by_phase[exam[3]["phase"]] == 1


def test_a_copy_edited_every_ten_words_is_still_caught(corpus):
    exam, train = corpus
    edited = edit_every(exam[0]["story"], 10, seed=1)
    result = find_near_duplicates(exam, train + [row("tedit", 2, edited)])
    assert [p.exam_id for p in result.pairs] == [exam[0]["prompt_hash"]]
    assert result.pairs[0].containment >= CONTAINMENT_THRESHOLD


def test_an_exam_story_inside_a_longer_training_story_is_caught(corpus):
    exam, train = corpus
    long = random_story(9) + exam[11]["story"] + random_story(10)
    result = find_near_duplicates(exam, train + [row("tlong", 2, long)])
    (pair,) = result.pairs
    assert pair.containment == 1.0 and pair.jaccard < 0.5  # containment, not Jaccard, catches it


def test_the_prefix_rule_ignores_case_and_whitespace(corpus):
    exam, train = corpus
    story = exam[4]["story"]
    variant = story[:250].upper().replace(" ", "\t  ") + random_story(77)
    result = find_near_duplicates(exam, train + [row("tpre", 1, variant)])
    assert [(p.exam_id, p.rule) for p in result.pairs] == [(exam[4]["prompt_hash"], "prefix")]


def test_a_shared_fact_sentence_is_not_a_near_duplicate_but_is_reported(corpus):
    exam, train = corpus
    fact = "The heart pumps blood through the body every single day."
    exam = [dict(r) for r in exam]
    exam[0]["story"] = exam[0]["story"] + fact + "\n"
    train = [dict(r) for r in train]
    train[0]["story"] = fact + "\n" + train[0]["story"]
    assert find_near_duplicates(exam, train).pairs == []
    report = verbatim_overlap(exam, train)
    cell = report["pairs"][0][0]
    assert cell["sentences_shared"] == 1
    assert cell["stories_with_shared_sentence"] == 1
    assert cell["story_fraction"] == pytest.approx(1 / 5)
    assert report["same_phase"][0] == cell
    assert report["pairs"][0][1]["sentences_shared"] == 0


def test_verbatim_overlap_fractions_are_the_hand_arithmetic():
    exam = [row("e", 0, "a b c d e f g h i j. zz yy xx ww vv uu.")]
    train = [row("t", 0, "a b c d e f g h i j k."), row("u", 1, "nothing in common here at all.")]
    report = verbatim_overlap(exam, train, n=8, min_sentence_words=6)
    cell = report["pairs"][0][0]
    # exam 8-grams over "a..j zz yy xx ww vv uu" (16 words): 9 distinct; 3 lie inside "a..j"
    assert cell["ngram_fraction"] == pytest.approx(3 / 9)
    # sentences >= 6 words: "a..j" (10) and "zz..uu" (6); train phase 0 holds "a..k", not "a..j"
    assert cell["n_exam_sentences"] == 2 and cell["sentences_shared"] == 0
    assert report["pairs"][0][1]["ngram_fraction"] == 0.0


def test_drop_near_duplicates_counts_per_phase_across_all_training_phases(corpus):
    exam, train = corpus
    train = train + [row("x1", 2, exam[0]["story"]), row("x2", 0, exam[0]["story"]), row("x3", 1, exam[12]["story"])]
    kept, result = drop_near_duplicates(exam, train)
    assert len(kept) == len(exam) - 2
    assert result.dropped_by_phase == {0: 1, 1: 0, 2: 1}
    assert {p.train_phase for p in result.pairs if p.exam_id == exam[0]["prompt_hash"]} == {0, 2}


def test_the_cli_reports_numbers_and_never_text(corpus, tmp_path, capsys):
    exam, train = corpus
    train = train + [row("tcopy", 1, exam[2]["story"])]
    (tmp_path / "exam").mkdir()
    (tmp_path / "train").mkdir()
    for p in range(3):
        (tmp_path / "exam" / f"exam_phase_{p}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in exam if r["phase"] == p), encoding="utf-8")
        (tmp_path / "train" / f"train_phase_{p}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in train if r["phase"] == p), encoding="utf-8")
    out = tmp_path / "report.json"
    assert main(["--exam-dir", str(tmp_path / "exam"), "--train-dir", str(tmp_path / "train"), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["near_duplicates"]["dropped_by_phase"] == {"0": 1, "1": 0, "2": 0}
    for r in exam + train:
        assert r["story"].splitlines()[0] not in printed
