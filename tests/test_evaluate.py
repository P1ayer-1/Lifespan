"""Scoring tests. Synthetic fixtures only - no real exam text ever reaches a
test file (AGENTS.md: one agent reads exam text, and it is not this one).

The stub models below return logits this file controls, so every expected
number is worked out on paper in a comment and written in as a literal.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
from torch import nn

from tests.conftest import (
    _write_probe_jsonl,
    option_sources as fake_sources,
    write_scoreable_exam_dir,
)
from training.config import EXAM_TYPES, N_PHASES
from training.evaluate import (
    CONTINUATION_CHANCE,
    CONTINUATION_SUMMED,
    STORED_MATRIX_KEYS,
    TOY_PARAM_BUDGET,
    ByteTokenizer,
    EvalConfig,
    _Scored,
    _score_spans,
    evaluate_all,
    evaluate_all_detailed,
    exam_story_phases,
    load_jsonl,
    score_cloze_phase,
    score_continuation_phase,
    score_perplexity_phase,
    validate_continuation_item,
)

# --------------------------------------------------------------------------- #
# stub models
# --------------------------------------------------------------------------- #


class ConstantLogitsLM(nn.Module):
    """logits[b, t, v] = pref[v], whatever the context.

    Context-free on purpose: the log-likelihood of a scored span of n tokens is
    then exactly  sum(pref[tok]) - n * C  with  C = logsumexp(pref), which can
    be done on paper.
    """

    def __init__(self, pref: torch.Tensor, block_size: int = 1024):
        super().__init__()
        self.register_buffer("pref", pref)
        self.block_size = block_size

    def forward(self, idx, targets=None):
        B, T = idx.shape
        return self.pref.view(1, 1, -1).expand(B, T, -1).contiguous()


class TinyBigramLM(nn.Module):
    """A real (tiny) module with learned-shaped weights, seeded.

    Used where the point is determinism or shape rather than hand arithmetic.
    Causal by construction: position t's logits depend only on token t, so
    right-padding a batch cannot leak into an earlier position.
    """

    def __init__(self, vocab_size: int = 256, n_embd: int = 16, block_size: int = 64, seed: int = 7):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.emb = nn.Embedding(vocab_size, n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        with torch.no_grad():
            self.emb.weight.copy_(torch.randn(vocab_size, n_embd, generator=g))
            self.head.weight.copy_(torch.randn(vocab_size, n_embd, generator=g))
        self.block_size = block_size

    def forward(self, idx, targets=None):
        logits = self.head(self.emb(idx))
        if targets is None:
            return logits
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1)
        )
        return logits, loss


class NanoGPTStyleLM(TinyBigramLM):
    """nanoGPT's inference shortcut: forward(idx) returns only the last
    position unless targets are supplied. The scorer must notice and retry."""

    def forward(self, idx, targets=None):
        logits = self.head(self.emb(idx))
        if targets is None:
            return logits[:, [-1], :], None
        return logits, None


# --------------------------------------------------------------------------- #
# the hand-worked constants
# --------------------------------------------------------------------------- #

X = ord("x")  # 120
Y = ord("y")  # 121
Z = ord("z")  # 122
Q = ord("q")  # 113


# `fake_sources` is `tests.conftest.option_sources`, imported above.


def make_pref() -> torch.Tensor:
    """pref[x] = 3, pref[y] = 1, every other byte 0."""
    pref = torch.zeros(256, dtype=torch.float32)
    pref[X] = 3.0
    pref[Y] = 1.0
    return pref


# C = log( 254 * e^0 + e^3 + e^1 )
#   = log( 254 + 20.085536923187668 + 2.718281828459045 )
#   = log( 276.80381875164673 )
#   = 5.6233090197164370
C = math.log(254.0 + math.exp(3.0) + math.exp(1.0))
TOK = ByteTokenizer()
CFG = EvalConfig(block_size=64, batch_size=4, logit_chunk=8)
DEV = torch.device("cpu")


def test_constant_C_matches_the_paper_value():
    assert C == pytest.approx(5.6233090197, abs=1e-9)


def test_span_scoring_is_the_hand_arithmetic():
    """sum log P over a scored span = sum(pref[tok]) - n * C."""
    model = ConstantLogitsLM(make_pref())
    # "qx"  -> ids [113, 120]; one predicted token (120): 3.0 - C = -2.6233090
    # "qzz" -> ids [113, 122, 122]; two predicted (0 each): 0 - 2C = -11.2466180
    items = [
        _Scored(ids=[Q, X], start=1, end=2),
        _Scored(ids=[Q, Z, Z], start=1, end=3),
    ]
    got = _score_spans(model, items, CFG, DEV)
    assert got[0][0] == pytest.approx(3.0 - C, abs=1e-5)
    assert got[0][1] == 1
    assert got[1][0] == pytest.approx(-2.0 * C, abs=1e-5)
    assert got[1][1] == 2


# --------------------------------------------------------------------------- #
# 1. hand-worked cloze
# --------------------------------------------------------------------------- #


def test_cloze_picks_the_hand_computed_candidate():
    """Whole-filled-sequence scoring, worked out on paper.

    text_with_mask = "q[MASK]", candidates ["x", "y", "zz"]:
      "qx"  -> predicted tokens [x]     -> 3.0 - C  = -2.6233090
      "qy"  -> predicted tokens [y]     -> 1.0 - C  = -4.6233090
      "qzz" -> predicted tokens [z, z]  -> 0 - 2C   = -11.2466180
    so the scorer must rank "x" first. Item 1's answer is "x" (correct);
    item 2 is the same text with answer "y" (wrong). Accuracy = 1/2.
    """
    model = ConstantLogitsLM(make_pref())
    items = [
        {"id": "c1", "phase": 0, "text_with_mask": "q[MASK]", "answer": "x",
         "candidates": ["x", "y", "zz"]},
        {"id": "c2", "phase": 0, "text_with_mask": "q[MASK]", "answer": "y",
         "candidates": ["x", "y", "zz"]},
    ]
    acc, n, chance = score_cloze_phase(model, items, TOK, CFG, DEV)
    assert acc == 0.5
    assert n == 2
    assert chance == pytest.approx(1.0 / 3.0)


def test_cloze_chance_is_five_percent_at_twenty_candidates():
    """The frozen contract: 20 candidates, so chance is 0.05."""
    model = ConstantLogitsLM(make_pref())
    cands = ["x"] + ["w" + str(i) for i in range(19)]
    items = [{"id": "c", "phase": 0, "text_with_mask": "q[MASK]", "answer": "x",
              "candidates": cands}]
    acc, n, chance = score_cloze_phase(model, items, TOK, CFG, DEV)
    assert chance == pytest.approx(0.05)
    assert acc == 1.0  # "x" is the only favoured byte and the shortest candidate


def test_cloze_rejects_a_malformed_item():
    model = ConstantLogitsLM(make_pref())
    with pytest.raises(ValueError):
        score_cloze_phase(
            model,
            [{"id": "c", "phase": 0, "text_with_mask": "no mask here",
              "answer": "x", "candidates": ["x"]}],
            TOK, CFG, DEV,
        )
    with pytest.raises(ValueError):
        score_cloze_phase(
            model,
            [{"id": "c", "phase": 0, "text_with_mask": "q[MASK]",
              "answer": "x", "candidates": ["y", "z"]}],
            TOK, CFG, DEV,
        )


# --------------------------------------------------------------------------- #
# 1b + 4. hand-worked continuation, and the length bias
# --------------------------------------------------------------------------- #


class RegisterLM(nn.Module):
    """A bigram stub with a register: every position's logits are `make_pref()`
    (x = 3, y = 1, else 0) except right after a "q", where z is raised to 4.

    So the model prefers x everywhere (its "register"), and the prefix "q" is
    evidence for z. Position t depends only on token t, so the context-free
    baseline (<|endoftext|> = byte 0, then the option) differs from the
    conditional score only at the option's first token.
    """

    def __init__(self, block_size: int = 1024):
        super().__init__()
        table = make_pref().repeat(256, 1)
        table[Q, Z] = 4.0
        self.register_buffer("table", table)
        self.block_size = block_size

    def forward(self, idx, targets=None):
        return self.table[idx]


# Cq = log( 253 + e^3 + e^1 + e^4 ), the normaliser after "q"
#    = log( 253 + 20.085536923 + 2.718281828 + 54.598150033 )
#    = log( 330.401968784 ) = 5.800310...
CQ = math.log(253.0 + math.exp(3.0) + math.exp(1.0) + math.exp(4.0))
PMI_ITEM = {
    "id": "k1", "phase": 0, "prefix": "q",
    "options": ["xx", "zz", "yy", "ww"],
    "answer_index": 1, "distractor_phases": [1, 3, 4],
}


def test_continuation_headline_is_pmi_and_disagrees_with_the_sum():
    """The prefix supports "zz"; the model's register prefers x.

    prefix "q"; option = 2 bytes; first predicted after "q", second after
    the first option byte (logits = pref there).
      summed:  xx  (3 - Cq) + (3 - C) = 6 - Cq - C    <- argmax (register)
               zz  (4 - Cq) + (0 - C) = 4 - Cq - C
               yy  (1 - Cq) + (1 - C) = 2 - Cq - C
               ww  (0 - Cq) + (0 - C) = 0 - Cq - C
      baseline (after byte 0, logits = pref): first byte pref[o1] - C, second
      as above, so PMI = (row_q[o1] - Cq) - (pref[o1] - C):
               xx  3 - 3 + C - Cq = C - Cq
               zz  4 - 0 + C - Cq = 4 + C - Cq            <- argmax (true)
               yy  1 - 1 + C - Cq = C - Cq
               ww  0 - 0 + C - Cq = C - Cq
    PMI accuracy 1.0, summed accuracy 0.0.
    """
    assert 6 - CQ - C > 4 - CQ - C and 4 + C - CQ > C - CQ  # the paper above
    item = dict(PMI_ITEM, option_sources=fake_sources(0, 1, 4, 1))
    acc_pmi, acc_sum, n, chance = score_continuation_phase(RegisterLM(), [item], TOK, CFG, DEV)
    assert acc_pmi == 1.0
    assert acc_sum == 0.0
    assert n == 1
    assert chance == 0.25


def test_pmi_cancels_a_context_free_model():
    """A model that ignores context scores every option's PMI exactly 0: a
    four-way tie, which counts as wrong, whatever its register prefers."""
    item = dict(PMI_ITEM, option_sources=fake_sources(0, 1, 4, 1))
    acc_pmi, acc_sum, _, _ = score_continuation_phase(ConstantLogitsLM(make_pref()), [item], TOK, CFG, DEV)
    assert acc_pmi == 0.0
    assert acc_sum == 0.0  # summed picks "xx" (index 0), not the answer


def test_pmi_needs_an_eot_token():
    class NoEot:
        def encode(self, text):
            return list(text.encode("utf-8"))

    item = dict(PMI_ITEM, option_sources=fake_sources(0, 1, 4, 1))
    with pytest.raises(ValueError, match="eot_id"):
        score_continuation_phase(RegisterLM(), [item], NoEot(), CFG, DEV)


def test_continuation_headline_in_evaluate_all_is_the_pmi_score(tmp_path):
    """evaluate_all()["continuation"] is the PMI accuracy; the summed one is
    stored beside it and never headlined."""
    exam_dir = _write_pmi_exam_dir(tmp_path)
    detailed = evaluate_all_detailed(RegisterLM(), exam_dir, range(N_PHASES), tokenizer=TOK, cfg=CFG)
    assert detailed.scores["continuation"][0] == 1.0
    assert detailed.continuation_summed[0] == 0.0
    hook = evaluate_all(RegisterLM(), exam_dir, range(N_PHASES), tokenizer=TOK, cfg=CFG)
    assert hook["continuation"][0] == detailed.scores["continuation"][0]
    assert set(hook) == set(EXAM_TYPES)  # the summed score is not a fourth key


# --------------------------------------------------------------------------- #
# leakage audit (AGENTS.md, amendment 2): provenance, bounds, chance
# --------------------------------------------------------------------------- #


def a_continuation_item(**overrides) -> dict:
    """A valid synthetic item; `overrides` break exactly one thing."""
    item = {
        "id": "k0", "phase": 2, "prefix": "q",
        "options": ["xxxx", "y", "yy", "zz"],
        "answer_index": 0,
        "distractor_phases": [2, 2, 2],
        "option_sources": [
            {"story_id": "s_true", "story_sha256": "a" * 64, "phase": 2},
            {"story_id": "s_d1", "story_sha256": "b" * 64, "phase": 2},
            {"story_id": "s_d2", "story_sha256": "c" * 64, "phase": 2},
            {"story_id": "s_d3", "story_sha256": "d" * 64, "phase": 2},
        ],
    }
    item.update(overrides)
    return item


def test_a_valid_item_passes_provenance_and_returns_its_option_count():
    assert validate_continuation_item(a_continuation_item()) == 4


def test_an_item_missing_option_sources_is_refused(tmp_path):
    """An item that cannot prove its own provenance is not an exam. After the
    freeze this is the only check left: an option is a paragraph excerpted from
    a story, so its sha256 can never be matched against a manifest entry, which
    hashes a whole file."""
    item = a_continuation_item()
    del item["option_sources"]
    with pytest.raises(ValueError, match="'option_sources' is required"):
        validate_continuation_item(item)
    # and it is refused through the scorer and the hook, not only in isolation
    with pytest.raises(ValueError, match="'option_sources' is required"):
        score_continuation_phase(ConstantLogitsLM(make_pref()), [item], TOK, CFG, DEV)
    exam_dir = write_synthetic_exam_dir(tmp_path / "exams", n_stories=1, n_probes=1)
    rows = [json.loads(line) for line in
            (exam_dir / "probes" / "continuation_phase_0.jsonl").read_text(
                encoding="utf-8").splitlines()]
    for row in rows:
        del row["option_sources"]
    _write_jsonl(exam_dir / "probes" / "continuation_phase_0.jsonl", rows)
    with pytest.raises(ValueError, match="'option_sources' is required"):
        evaluate_all(TinyBigramLM(), exam_dir, [0], tokenizer=TOK, cfg=CFG)


@pytest.mark.parametrize(
    "overrides, match",
    [
        # one source short of one per option
        ({"option_sources": [
            {"story_id": "s_true", "story_sha256": "a" * 64, "phase": 2},
            {"story_id": "s_d1", "story_sha256": "b" * 64, "phase": 2},
            {"story_id": "s_d2", "story_sha256": "c" * 64, "phase": 2},
        ]}, "one per option is required"),
        # an entry missing a required field
        ({"option_sources": [
            {"story_id": "s_true", "phase": 2},
            {"story_id": "s_d1", "story_sha256": "b" * 64, "phase": 2},
            {"story_id": "s_d2", "story_sha256": "c" * 64, "phase": 2},
            {"story_id": "s_d3", "story_sha256": "d" * 64, "phase": 2},
        ]}, "missing story_sha256"),
        # the true continuation did not come from this phase
        ({"option_sources": [
            {"story_id": "s_true", "story_sha256": "a" * 64, "phase": 6},
            {"story_id": "s_d1", "story_sha256": "b" * 64, "phase": 2},
            {"story_id": "s_d2", "story_sha256": "c" * 64, "phase": 2},
            {"story_id": "s_d3", "story_sha256": "d" * 64, "phase": 2},
        ]}, "the answer option's source phase"),
        # Amendment 5: a distractor drawn from another phase (until 2026-10-01
        # the rule was the reverse; then the item measured register)
        ({"option_sources": [
            {"story_id": "s_true", "story_sha256": "a" * 64, "phase": 2},
            {"story_id": "s_d1", "story_sha256": "b" * 64, "phase": 3},
            {"story_id": "s_d2", "story_sha256": "c" * 64, "phase": 2},
            {"story_id": "s_d3", "story_sha256": "d" * 64, "phase": 2},
        ]}, "must come from the item's own phase"),
        # a distractor excerpted from the answer's own story
        ({"option_sources": [
            {"story_id": "s_true", "story_sha256": "a" * 64, "phase": 2},
            {"story_id": "s_true", "story_sha256": "b" * 64, "phase": 2},
            {"story_id": "s_d2", "story_sha256": "c" * 64, "phase": 2},
            {"story_id": "s_d3", "story_sha256": "d" * 64, "phase": 2},
        ]}, "shares the answer's source story"),
    ],
)
def test_bad_provenance_fails_loudly(overrides, match):
    with pytest.raises(ValueError, match=match):
        validate_continuation_item(a_continuation_item(**overrides))


def test_provenance_errors_never_carry_exam_text():
    """A traceback ends up in a Kaggle notebook's output, so an error message
    carries ids, phases and hashes and never a prefix or an option."""
    item = a_continuation_item(prefix="SECRET PREFIX", options=["SECRET OPTION A"] * 4)
    item["option_sources"][1]["phase"] = 3
    with pytest.raises(ValueError) as exc:
        validate_continuation_item(item)
    assert "SECRET" not in str(exc.value)
    assert "k0" in str(exc.value)


# option_sources hashes are checked against the exam's per-story hashes
# (leakage audit 2026-09-25, should-fix: they were self-attested).

KNOWN = {"a" * 64: 2, "b" * 64: 2, "c" * 64: 2, "d" * 64: 2}


def test_option_sources_whose_hashes_are_exam_stories_pass():
    assert validate_continuation_item(a_continuation_item(), KNOWN) == 4
    assert validate_continuation_item(a_continuation_item(), set(KNOWN)) == 4


def test_a_distractor_whose_hash_is_in_no_manifest_is_refused():
    item = a_continuation_item()
    item["option_sources"][2]["story_sha256"] = "e" * 64
    with pytest.raises(ValueError, match=r"option_sources\[2\].story_sha256 e{64} is not the hash"):
        validate_continuation_item(item, KNOWN)
    with pytest.raises(ValueError, match="is not the hash of any exam story"):
        score_continuation_phase(
            ConstantLogitsLM(make_pref()), [item], TOK, CFG, DEV, known_story_hashes=set(KNOWN)
        )


def test_a_source_claiming_the_wrong_phase_for_its_story_is_refused():
    item = a_continuation_item()
    item["option_sources"][3]["story_sha256"] = "e" * 64  # a phase-4 story, claimed as 2
    with pytest.raises(ValueError, match="claims phase 2 but story e{64} is an exam story of phase 4"):
        validate_continuation_item(item, {**KNOWN, "e" * 64: 4})


def test_the_hook_checks_option_sources_against_the_exam_stories_by_default(tmp_path):
    """No hashes passed: the hook derives them from the exam stories on disk,
    so the check cannot be switched off by forgetting an argument."""
    exam_dir = write_synthetic_exam_dir(tmp_path / "exams", n_stories=2, n_probes=2)
    known = exam_story_phases(exam_dir)
    assert len(known) == 4 * N_PHASES  # at least 4 stories a phase (Amendment 5)
    evaluate_all(TinyBigramLM(), exam_dir, [0], tokenizer=TOK, cfg=CFG)  # clean passes

    path = exam_dir / "probes" / "continuation_phase_0.jsonl"
    rows = load_jsonl(path)
    rows[1]["option_sources"][1]["story_sha256"] = "f" * 64  # a distractor from nowhere
    _write_jsonl(path, rows)
    with pytest.raises(ValueError, match="is not the hash of any exam story"):
        evaluate_all(TinyBigramLM(), exam_dir, [0], tokenizer=TOK, cfg=CFG)
    # an explicit hash map (the manifest's, via the guard report) is used as given
    with pytest.raises(ValueError, match="is not the hash of any exam story"):
        evaluate_all(TinyBigramLM(), exam_dir, [0], tokenizer=TOK, cfg=CFG, story_hashes=known)


def test_an_exam_dir_without_stories_cannot_vouch_for_option_sources(tmp_path):
    (tmp_path / "stories").mkdir()
    with pytest.raises(ValueError, match="no exam stories"):
        exam_story_phases(tmp_path)


# --------------------------------------------------------------------------- #
# the exam on disk is the manifest's (2026-09-30): verify_exam_stories and
# the same check inside evaluate_all when story_hashes is given
# --------------------------------------------------------------------------- #


def _manifest_map(exam_dir: Path) -> dict[str, int]:
    """What the guard report hands over: normalised hash -> phase, computed
    here the manifest's way (conftest.manifest_stories_of)."""
    from tests.conftest import manifest_stories_of

    return {e["story_sha256"]: e["phase"] for e in manifest_stories_of(exam_dir)}


def test_an_exam_matching_the_manifest_verifies_and_scores(tmp_path):
    from training.evaluate import verify_exam_stories

    exam_dir = write_synthetic_exam_dir(tmp_path / "exams", n_stories=2, n_probes=2)
    frozen = _manifest_map(exam_dir)
    assert verify_exam_stories(exam_dir, frozen, range(N_PHASES)) == {k: 4 for k in range(N_PHASES)}  # >= 4 stories, Amendment 5
    evaluate_all(TinyBigramLM(), exam_dir, [0, 3], tokenizer=TOK, cfg=CFG, story_hashes=frozen)


def test_a_crlf_or_reindented_copy_still_matches(tmp_path):
    """The manifest hash is of the normalised story, so line endings on the
    Kaggle side cannot turn a frozen exam into a mismatch."""
    from training.evaluate import verify_exam_stories

    exam_dir = write_synthetic_exam_dir(tmp_path / "exams", n_stories=2, n_probes=2)
    frozen = _manifest_map(exam_dir)
    path = exam_dir / "stories" / "exam_phase_2.jsonl"
    rows = load_jsonl(path)
    for r in rows:
        r["story"] = "  " + r["story"].replace("\n", "\r\n") + "\t"
    _write_jsonl(path, rows)
    assert verify_exam_stories(exam_dir, frozen, [2]) == {2: 4}


@pytest.mark.parametrize(
    "tamper,match",
    [
        ("edit", "is not in the manifest"),
        ("drop", "exam stories in the manifest are not on disk"),
        ("extra", "is not in the manifest"),
        ("duplicate", "are the same story"),
        ("wrong_phase_field", "says phase 5 but sits in the phase-1 file"),
        ("moved", "is a phase-2 exam story in the manifest, not phase 1"),
        ("no_story", "has no 'story' field"),
        ("missing_file", "exam stories for phase 1 are missing"),
    ],
)
def test_an_exam_that_differs_from_the_manifest_is_refused_before_scoring(tmp_path, tamper, match):
    from training.evaluate import ExamManifestMismatch, verify_exam_stories

    exam_dir = write_synthetic_exam_dir(tmp_path / "exams", n_stories=3, n_probes=2)
    frozen = _manifest_map(exam_dir)
    path = exam_dir / "stories" / "exam_phase_1.jsonl"
    rows = load_jsonl(path)
    if tamper == "edit":
        rows[1]["story"] = rows[1]["story"] + " one edited word"
    elif tamper == "drop":
        rows = rows[:-1]
    elif tamper == "extra":
        rows.append({"id": "planted", "phase": 1, "story": "a story nobody froze."})
    elif tamper == "duplicate":
        rows.append(dict(rows[0], id="copy"))
    elif tamper == "wrong_phase_field":
        rows[0]["phase"] = 5
    elif tamper == "moved":
        rows[0] = dict(load_jsonl(exam_dir / "stories" / "exam_phase_2.jsonl")[0], phase=1)
    elif tamper == "no_story":
        del rows[0]["story"]
    if tamper == "missing_file":
        path.unlink()
    else:
        _write_jsonl(path, rows)

    with pytest.raises(ExamManifestMismatch, match=match):
        verify_exam_stories(exam_dir, frozen, range(N_PHASES))
    scored: list = []

    class Spy(TinyBigramLM):
        def forward(self, *a, **k):
            scored.append(1)
            return super().forward(*a, **k)

    with pytest.raises(ExamManifestMismatch if tamper != "missing_file" else (ExamManifestMismatch, OSError)):
        evaluate_all(Spy(), exam_dir, [0, 1], tokenizer=TOK, cfg=CFG, story_hashes=frozen)
    assert scored == [], "a phase was scored before the mismatch was refused"
    # an unscored phase's file is not the evaluator's business in this call
    verify_exam_stories(exam_dir, frozen, [0, 2])


def test_mismatch_messages_carry_hashes_and_ids_never_exam_text(tmp_path):
    from training.evaluate import ExamManifestMismatch, verify_exam_stories

    exam_dir = write_synthetic_exam_dir(tmp_path / "exams", n_stories=2, n_probes=2)
    frozen = _manifest_map(exam_dir)
    path = exam_dir / "stories" / "exam_phase_0.jsonl"
    rows = load_jsonl(path)
    rows[0]["story"] = "SECRETEXAMTEXT " + rows[0]["story"]
    _write_jsonl(path, rows)
    with pytest.raises(ExamManifestMismatch) as info:
        verify_exam_stories(exam_dir, frozen, [0])
    assert "SECRETEXAMTEXT" not in str(info.value)


def test_an_empty_manifest_map_is_refused_not_treated_as_no_check(tmp_path):
    from training.evaluate import ExamManifestMismatch, verify_exam_stories

    exam_dir = write_synthetic_exam_dir(tmp_path / "exams", n_stories=1, n_probes=1)
    with pytest.raises(ExamManifestMismatch, match="empty"):
        verify_exam_stories(exam_dir, {}, [0])
    with pytest.raises(ExamManifestMismatch, match="empty"):
        evaluate_all(TinyBigramLM(), exam_dir, [0], tokenizer=TOK, cfg=CFG, story_hashes={})


def test_answer_index_is_bounds_checked():
    model = ConstantLogitsLM(make_pref())
    for bad in (4, -1, 99):
        with pytest.raises(ValueError, match="out of range"):
            validate_continuation_item(a_continuation_item(answer_index=bad))
    with pytest.raises(ValueError, match="must be an int"):
        validate_continuation_item(a_continuation_item(answer_index="0"))
    # and it never reaches the scorer as an index into something
    with pytest.raises(ValueError, match="out of range"):
        score_continuation_phase(model, [a_continuation_item(answer_index=4)], TOK, CFG, DEV)


def test_chance_is_derived_from_the_option_count_not_hardcoded():
    """A 5-option item must report chance 0.20, not 0.25."""
    model = ConstantLogitsLM(make_pref())
    five = a_continuation_item(
        options=["xxxx", "y", "yy", "zz", "zzz"],
        answer_index=0,
        option_sources=[
            {"story_id": "s_true", "story_sha256": "a" * 64, "phase": 2},
            {"story_id": "s_d1", "story_sha256": "b" * 64, "phase": 2},
            {"story_id": "s_d2", "story_sha256": "c" * 64, "phase": 2},
            {"story_id": "s_d3", "story_sha256": "d" * 64, "phase": 2},
            {"story_id": "s_d4", "story_sha256": "e" * 64, "phase": 2},
        ],
    )
    _, _, _, chance = score_continuation_phase(model, [five], TOK, CFG, DEV)
    assert chance == pytest.approx(0.20)
    assert chance != pytest.approx(CONTINUATION_CHANCE)
    # the frozen 4-option set still reports 0.25
    _, _, _, chance4 = score_continuation_phase(
        model, [a_continuation_item()], TOK, CFG, DEV
    )
    assert chance4 == pytest.approx(0.25) == pytest.approx(CONTINUATION_CHANCE)


def test_a_phase_with_mixed_option_counts_is_refused():
    """One chance level per phase, or the accuracies are not comparable."""
    model = ConstantLogitsLM(make_pref())
    four = a_continuation_item(id="k4")
    five = a_continuation_item(
        id="k5",
        options=["xxxx", "y", "yy", "zz", "zzz"],
        option_sources=a_continuation_item()["option_sources"]
        + [{"story_id": "s_d4", "story_sha256": "e" * 64, "phase": 2}],
    )
    with pytest.raises(ValueError, match="one chance level per phase"):
        score_continuation_phase(model, [four, five], TOK, CFG, DEV)


def test_chance_reaches_the_detailed_result(tmp_path):
    model = TinyBigramLM()
    exam_dir = write_synthetic_exam_dir(tmp_path / "exams", n_stories=1, n_probes=2)
    detailed = evaluate_all_detailed(model, exam_dir, [0, 1], tokenizer=TOK, cfg=CFG)
    assert detailed.chance["continuation"][0] == pytest.approx(0.25)
    assert detailed.chance["cloze"][0] == pytest.approx(0.05)
    assert detailed.chance["continuation"][5] is None  # not scored


# --------------------------------------------------------------------------- #
# 6. ties count as wrong
# --------------------------------------------------------------------------- #


def test_continuation_tie_counts_as_wrong():
    """Options 0 and 1 are permutations, so both scores are 4 - 2C exactly:
    normalised (2 - C) and summed (4 - 2C) both tie, and a tie is wrong."""
    model = ConstantLogitsLM(make_pref())
    items = [{
        "id": "k2", "phase": 0, "prefix": "q",
        "options": ["xy", "yx", "zz", "zzz"],
        "answer_index": 0, "distractor_phases": [1, 2, 3],
        "option_sources": fake_sources(0, 2, 4, 0),
    }]
    acc_norm, acc_sum, _, _ = score_continuation_phase(model, items, TOK, CFG, DEV)
    assert acc_norm == 0.0
    assert acc_sum == 0.0


def test_cloze_tie_counts_as_wrong():
    """Candidates "xy" and "yx" fill to sequences with identical token
    multisets, so their whole-sequence scores are equal: no rank-1 winner."""
    model = ConstantLogitsLM(make_pref())
    items = [{"id": "c3", "phase": 0, "text_with_mask": "q[MASK]", "answer": "xy",
              "candidates": ["xy", "yx"]}]
    acc, _, _ = score_cloze_phase(model, items, TOK, CFG, DEV)
    assert acc == 0.0


# --------------------------------------------------------------------------- #
# perplexity: per token over the phase, and the windowing rule
# --------------------------------------------------------------------------- #


def test_perplexity_is_per_token_over_the_phase_not_a_mean_of_story_means():
    """Two stories, one token predicted in the first and two in the second:

      "xy"  -> predicted [y]    : lp = 1 - C          , n = 1
      "zzz" -> predicted [z, z] : lp = 0 - 2C         , n = 2
      total lp = 1 - 3C over 3 tokens -> loss = C - 1/3 = 5.2899757
      a mean of per-story means would be ((C-1) + C)/2 = C - 1/2 = 5.1233090
    """
    model = ConstantLogitsLM(make_pref())
    stories = [{"story": "xy"}, {"story": "zzz"}]
    loss, n_tok = score_perplexity_phase(model, stories, TOK, CFG, DEV)
    assert n_tok == 3
    assert loss == pytest.approx(C - 1.0 / 3.0, abs=1e-5)
    assert loss == pytest.approx(5.2899757, abs=1e-5)
    assert loss != pytest.approx(C - 0.5, abs=1e-4)  # not the mean of story means


def test_perplexity_windowing_rule_is_non_overlapping_and_drops_a_1_token_tail():
    """block_size 4: a 10-byte story gives windows 4/4/2 -> 3+3+1 = 7 predicted
    tokens; a 5-byte story gives windows 4/1 and the 1-token tail is dropped,
    so 3 predicted tokens. Total 10."""
    model = ConstantLogitsLM(make_pref())
    cfg = EvalConfig(block_size=4, batch_size=4, logit_chunk=8)
    stories = [{"story": "zzzzzzzzzz"}, {"story": "zzzzz"}]
    _, n_tok = score_perplexity_phase(model, stories, TOK, cfg, DEV)
    assert n_tok == 10


# --------------------------------------------------------------------------- #
# synthetic exam directories
# --------------------------------------------------------------------------- #


# One exam-directory builder in the repo, and it is `trainer-core`'s in the
# shared conftest, so its end-to-end test and these tests score the same
# fixture. This is only a thin adapter to the argument names used below; it
# builds nothing of its own.
_write_jsonl = _write_probe_jsonl


def write_synthetic_exam_dir(
    root: Path, *, n_stories: int = 3, n_probes: int = 2, n_phases: int = N_PHASES
) -> Path:
    """`write_scoreable_exam_dir` with one knob per call site.

    The shared builder ties the story count and the probe count together, so a
    caller asking for different numbers gets the larger of the two; these tests
    only ever want "small".
    """
    return write_scoreable_exam_dir(
        Path(root), n_phases=n_phases, items_per_type=max(n_stories, n_probes)
    )


def _write_pmi_exam_dir(tmp_path: Path) -> Path:
    """The hand-worked PMI item (`PMI_ITEM`) in phase 0, filler elsewhere."""
    root = write_synthetic_exam_dir(tmp_path / "exams")
    # option_sources must name real exam stories (audit 2026-09-25), so borrow
    # the provenance of the builder's own first phase-0 item.
    path = root / "probes" / "continuation_phase_0.jsonl"
    sources = load_jsonl(path)[0]["option_sources"]
    _write_jsonl(
        path,
        # PMI_ITEM's answer is option 1: move the borrowed answer source there.
        [dict(PMI_ITEM, distractor_phases=[s["phase"] for s in sources[1:]],
              option_sources=[sources[1], sources[0], *sources[2:]])],
    )
    return root


# --------------------------------------------------------------------------- #
# 3. the hook's shape
# --------------------------------------------------------------------------- #


def test_evaluate_all_returns_three_keys_of_seven_floats(tmp_path):
    model = TinyBigramLM()
    exam_dir = write_synthetic_exam_dir(tmp_path / "exams")
    out = evaluate_all(model, exam_dir, range(N_PHASES), tokenizer=TOK, cfg=CFG)
    assert set(out) == set(EXAM_TYPES)
    assert list(out) == list(EXAM_TYPES)
    for exam_type in EXAM_TYPES:
        assert len(out[exam_type]) == N_PHASES == 7
        assert all(isinstance(v, float) for v in out[exam_type])
        assert all(math.isfinite(v) for v in out[exam_type])
    for exam_type in ("cloze", "continuation"):
        assert all(0.0 <= v <= 1.0 for v in out[exam_type])


def test_evaluate_all_scores_untrained_phases_too(tmp_path):
    """The matrix's upper triangle: a phase not yet trained is still scored."""
    model = TinyBigramLM()
    exam_dir = write_synthetic_exam_dir(tmp_path / "exams")
    out = evaluate_all(model, exam_dir, [0, 6], tokenizer=TOK, cfg=CFG)
    assert math.isfinite(out["continuation"][6])
    assert math.isnan(out["continuation"][3])  # not requested
    assert len(out["continuation"]) == N_PHASES


def test_evaluate_all_leaves_the_model_in_eval_mode_and_restores_training(tmp_path):
    model = TinyBigramLM()
    exam_dir = write_synthetic_exam_dir(tmp_path / "exams", n_stories=1, n_probes=1)
    model.train()
    evaluate_all(model, exam_dir, [0], tokenizer=TOK, cfg=CFG)
    assert model.training  # restored
    model.eval()
    evaluate_all(model, exam_dir, [0], tokenizer=TOK, cfg=CFG)
    assert not model.training


# --------------------------------------------------------------------------- #
# 2. determinism
# --------------------------------------------------------------------------- #


def test_evaluate_all_is_deterministic(tmp_path):
    """Same model, same fixtures, identical floats - not merely close."""
    model = TinyBigramLM()
    exam_dir = write_synthetic_exam_dir(tmp_path / "exams")
    first = evaluate_all(model, exam_dir, range(N_PHASES), tokenizer=TOK, cfg=CFG)
    second = evaluate_all(model, exam_dir, range(N_PHASES), tokenizer=TOK, cfg=CFG)
    assert first == second
    for exam_type in EXAM_TYPES:
        for a, b in zip(first[exam_type], second[exam_type]):
            assert a == b


def test_scores_do_not_depend_on_batch_size(tmp_path):
    """Right-padding must not leak: a causal model scored one item at a time
    gives the same answer as a padded batch of sixteen."""
    model = TinyBigramLM()
    exam_dir = write_synthetic_exam_dir(tmp_path / "exams")
    one = evaluate_all(
        model, exam_dir, range(N_PHASES), tokenizer=TOK,
        cfg=EvalConfig(block_size=64, batch_size=1, logit_chunk=8),
    )
    many = evaluate_all(
        model, exam_dir, range(N_PHASES), tokenizer=TOK,
        cfg=EvalConfig(block_size=64, batch_size=16, logit_chunk=256),
    )
    for exam_type in ("cloze", "continuation"):
        assert one[exam_type] == many[exam_type]
    for a, b in zip(one["perplexity"], many["perplexity"]):
        assert a == pytest.approx(b, rel=1e-9)


# --------------------------------------------------------------------------- #
# model plumbing
# --------------------------------------------------------------------------- #


def test_handles_a_nanogpt_style_last_position_shortcut(tmp_path):
    """forward(idx) returning only the last position must be detected and
    retried with targets, not silently scored on one token."""
    exam_dir = write_synthetic_exam_dir(tmp_path / "exams", n_stories=1, n_probes=1)
    plain = evaluate_all(TinyBigramLM(), exam_dir, [0], tokenizer=TOK, cfg=CFG)
    shortcut = evaluate_all(NanoGPTStyleLM(), exam_dir, [0], tokenizer=TOK, cfg=CFG)
    assert plain["cloze"][0] == shortcut["cloze"][0]
    assert plain["perplexity"][0] == pytest.approx(shortcut["perplexity"][0], rel=1e-9)


def test_log_softmax_is_taken_in_float32_for_a_half_precision_model():
    """A half-precision model must not be log-softmaxed in half: the returned
    log-likelihood matches the float32 computation to better than fp16's
    resolution over a 256-way vocab."""
    pref = make_pref()
    model = ConstantLogitsLM(pref).half()
    items = [_Scored(ids=[Q, X], start=1, end=2)]
    got = _score_spans(model, items, CFG, DEV)[0][0]
    assert got == pytest.approx(3.0 - C, abs=1e-2)


def test_falls_back_to_the_byte_tokenizer_only_at_toy_scale(tmp_path):
    exam_dir = write_synthetic_exam_dir(tmp_path / "exams", n_stories=1, n_probes=1)
    toy = TinyBigramLM()  # 256*16 + 256*16 = 8,192 parameters
    assert sum(p.numel() for p in toy.parameters()) <= TOY_PARAM_BUDGET
    out = evaluate_all(toy, exam_dir, [0], cfg=CFG)
    assert math.isfinite(out["perplexity"][0])


def test_a_missing_tokenizer_on_a_real_model_raises_rather_than_scoring_bytes(tmp_path):
    """A byte fallback on a model trained with a real tokenizer would make every
    number in the matrix meaningless, so it is a guard, not a default."""
    exam_dir = write_synthetic_exam_dir(tmp_path / "exams", n_stories=1, n_probes=1)
    big = TinyBigramLM(vocab_size=8192, n_embd=64)  # ~1.05M parameters, over the toy budget
    assert sum(p.numel() for p in big.parameters()) > TOY_PARAM_BUDGET
    with pytest.raises(ValueError, match="without a tokenizer"):
        evaluate_all(big, exam_dir, [0], cfg=CFG)
    # an explicit tokenizer is all it wants
    out = evaluate_all(big, exam_dir, [0], tokenizer=TOK, cfg=CFG)
    assert math.isfinite(out["perplexity"][0])


def test_the_matrix_row_carries_the_stored_summed_key(tmp_path):
    """matrix.json's fourth key: stored beside the three exam types, never
    headlined, and not a member of EXAM_TYPES."""
    exam_dir = _write_pmi_exam_dir(tmp_path)
    detailed = evaluate_all_detailed(RegisterLM(), exam_dir, range(N_PHASES), tokenizer=TOK, cfg=CFG)
    row = detailed.as_matrix_row()
    assert list(row) == list(STORED_MATRIX_KEYS)
    assert CONTINUATION_SUMMED not in EXAM_TYPES
    assert len(row[CONTINUATION_SUMMED]) == N_PHASES
    assert row[CONTINUATION_SUMMED][0] == 0.0  # the PMI item, summed
    assert row["continuation"][0] == 1.0  # ... and by PMI, which is the headline


def test_bad_phase_index_is_refused(tmp_path):
    exam_dir = write_synthetic_exam_dir(tmp_path / "exams", n_stories=1, n_probes=1)
    with pytest.raises(ValueError):
        evaluate_all(TinyBigramLM(), exam_dir, [7], tokenizer=TOK, cfg=CFG)
