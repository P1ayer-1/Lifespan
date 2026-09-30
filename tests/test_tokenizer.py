"""The tokenizer, and the rule that it never sees exam text."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.tokenizer import (
    END_OF_TEXT,
    BPETokenizer,
    ByteTokenizer,
    corpus_files,
    corpus_meta_path,
    get_or_train_tokenizer,
    iter_corpus,
    iter_stories,
    load_tokenizer,
    train_bpe,
)


def test_byte_tokenizer_round_trips():
    tok = ByteTokenizer()
    assert tok.vocab_size == 256
    assert tok.eot_id == 0
    text = "aaa bbb.\nccc"
    assert tok.decode(tok.encode(text)) == text
    assert all(0 <= i < 256 for i in tok.encode("héllo"))


def test_byte_tokenizer_refuses_a_non_byte_vocab():
    with pytest.raises(ValueError):
        ByteTokenizer(vocab_size=8192)


def test_corpus_is_only_training_story_files(toy_dirs, tmp_path):
    files = corpus_files(toy_dirs["train"])
    assert [f.name for f in files] == [f"train_phase_{k}.jsonl" for k in range(7)]
    # The replay json and anything else under the dir are not corpus.
    assert all(f.parent == Path(toy_dirs["train"]).resolve() for f in files)


def test_corpus_refuses_a_directory_with_no_training_files(toy_dirs):
    """Pointing the tokenizer at the exam directory raises, it does not quietly
    train on whatever it finds."""
    with pytest.raises(FileNotFoundError):
        corpus_files(toy_dirs["exam"])


def test_iter_corpus_yields_only_the_story_field(toy_dirs):
    stories = list(iter_corpus(toy_dirs["train"]))
    assert len(stories) == 7 * 20
    assert all(isinstance(s, str) and s for s in stories)


def test_iter_stories_rejects_a_line_without_a_story(tmp_path):
    bad = tmp_path / "train_phase_0.jsonl"
    bad.write_text(json.dumps({"prompt_hash": "x", "phase": 0}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="story"):
        list(iter_stories(bad))


def test_bpe_trains_to_the_requested_vocab_and_round_trips(toy_dirs, tmp_path):
    """The real tokenizer path, at a size a laptop can do in a second. The
    corpus is `iter_corpus(train_dir)` -- no exam path is reachable from here."""
    out = tmp_path / "tok.json"
    tok = train_bpe(iter_corpus(toy_dirs["train"]), vocab_size=400, out_path=out)
    # The synthetic corpus runs out of merges before 400; a BPE may end up
    # smaller than asked, never larger.
    assert 256 < tok.vocab_size <= 400
    assert tok.eot_id == BPETokenizer.load(out).eot_id
    text = "aaa aaaa bbb"
    assert tok.decode(tok.encode(text)) == text
    assert out.exists()


def test_get_or_train_tokenizer_returns_bytes_at_vocab_256(toy_dirs, tmp_path):
    tok = get_or_train_tokenizer(toy_dirs["train"], 256, tmp_path / "unused.json")
    assert isinstance(tok, ByteTokenizer)
    assert not (tmp_path / "unused.json").exists()


def test_get_or_train_tokenizer_caches_so_every_arm_shares_a_vocabulary(toy_dirs, tmp_path):
    cache = tmp_path / "tok.json"
    a = get_or_train_tokenizer(toy_dirs["train"], 400, cache)
    b = get_or_train_tokenizer(toy_dirs["train"], 400, cache)
    probe = "aaa bbb ccc"
    assert a.encode(probe) == b.encode(probe)
    assert isinstance(load_tokenizer(cache), BPETokenizer)


def test_get_or_train_tokenizer_refuses_a_tokenizer_bigger_than_the_model(toy_dirs, tmp_path):
    """Ids past the embedding table would be an index error thousands of steps
    in; it is refused before the model is built instead."""
    cache = tmp_path / "tok.json"
    tok = get_or_train_tokenizer(toy_dirs["train"], 400, cache)
    with pytest.raises(ValueError, match="larger than the model"):
        get_or_train_tokenizer(toy_dirs["train"], tok.vocab_size - 1, cache)


def test_end_of_text_is_a_real_token():
    tok = train_bpe(["aaa bbb", "ccc ddd"], vocab_size=300)
    assert tok.encode(END_OF_TEXT) != []
    assert tok.eot_id >= 0


# ---------------------------------------------------------------------------
# cache staleness (leakage audit, 2026-09-22)


def test_a_cached_tokenizer_from_another_corpus_is_refused(toy_dirs, tmp_path):
    """A vocabulary trained on other data changes every token id and every exam
    score. Caught here, not several phases later as a fingerprint mismatch."""
    cache = tmp_path / "tok.json"
    get_or_train_tokenizer(toy_dirs["train"], 400, cache, corpus_hash="a" * 64)
    get_or_train_tokenizer(toy_dirs["train"], 400, cache, corpus_hash="a" * 64)  # same corpus: fine
    with pytest.raises(ValueError, match="was trained on corpus"):
        get_or_train_tokenizer(toy_dirs["train"], 400, cache, corpus_hash="b" * 64)


def test_a_cache_with_no_corpus_record_is_refused(toy_dirs, tmp_path):
    cache = tmp_path / "tok.json"
    get_or_train_tokenizer(toy_dirs["train"], 400, cache, corpus_hash="a" * 64)
    corpus_meta_path(cache).unlink()
    with pytest.raises(ValueError, match="cannot be established"):
        get_or_train_tokenizer(toy_dirs["train"], 400, cache, corpus_hash="a" * 64)


# ---------------------------------------------------------------------------
# the split check (leakage re-audit 2026-09-25): the tokenizer reads every
# line data.py reads, and refuses the same lines


def _rewrite_line(path: Path, index: int, **changes) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[index])
    for k, v in changes.items():
        if v is None:
            rec.pop(k, None)
        else:
            rec[k] = v
    lines[index] = json.dumps(rec)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.mark.parametrize("split,match", [("exam", "has split 'exam'"), (None, "has no 'split' field")])
def test_the_tokenizer_refuses_a_line_that_is_not_declared_train(toy_dirs, tmp_path, split, match):
    """An exam story appended to a train file must not shape the vocabulary:
    the same refusal, by the same function, as the data loader's."""
    from training.data import DataError

    path = Path(toy_dirs["train"]) / "train_phase_5.jsonl"
    _rewrite_line(path, 13, split=split)
    with pytest.raises(DataError, match=r"train_phase_5\.jsonl:14 " + match):
        list(iter_corpus(toy_dirs["train"]))
    with pytest.raises(DataError, match=match):
        get_or_train_tokenizer(toy_dirs["train"], 400, tmp_path / "tok.json")
    assert not (tmp_path / "tok.json").exists(), "a tokenizer was written from a refused corpus"


def test_the_split_check_covers_every_line_not_only_the_first(toy_dirs):
    from training.data import DataError

    path = Path(toy_dirs["train"]) / "train_phase_0.jsonl"
    last = len(path.read_text(encoding="utf-8").splitlines()) - 1
    _rewrite_line(path, last, split="exam")
    with pytest.raises(DataError, match=rf":{last + 1} has split 'exam'"):
        list(iter_stories(path))


# ---------------------------------------------------------------------------
# subset mode: the BPE trains on the declared phases' files only


def test_subset_corpus_is_the_declared_phases_only(toy_dirs):
    files = corpus_files(toy_dirs["train"], (0, 3, 6))
    assert [f.name for f in files] == ["train_phase_0.jsonl", "train_phase_3.jsonl", "train_phase_6.jsonl"]
    stories = list(iter_corpus(toy_dirs["train"], (0, 3, 6)))
    assert len(stories) == 3 * 20
    # phase k's stories are written in letter 'a'+k only (conftest)
    letters = {c for s in stories for c in s if c.isalpha()}
    assert letters == {"a", "d", "g"}


def test_an_undeclared_phase_is_never_read_even_if_it_is_broken(toy_dirs):
    """A 0/3/6 run must not care what else sits in the directory: phase 1's
    file with an exam line in it is ignored, exactly as DataModule ignores it."""
    _rewrite_line(Path(toy_dirs["train"]) / "train_phase_1.jsonl", 0, split="exam")
    assert len(list(iter_corpus(toy_dirs["train"], (0, 3, 6)))) == 3 * 20


def test_a_declared_phase_without_a_file_is_refused(tmp_path):
    from tests.conftest import write_train_dir

    train = write_train_dir(tmp_path / "train", phases=(0, 6))
    with pytest.raises(FileNotFoundError, match="declared phase 3"):
        corpus_files(train, (0, 3, 6))
    with pytest.raises(ValueError):
        corpus_files(train, (3, 6))  # phase 0 is required, as config.resolve_phases says


def test_a_subset_bpe_is_fitted_to_the_declared_phases_and_cached_separately(toy_dirs, tmp_path):
    cache = tmp_path / "tok.json"
    sub = get_or_train_tokenizer(toy_dirs["train"], 400, cache, corpus_hash="a" * 64, phases=(0, 3, 6))
    assert json.loads(corpus_meta_path(cache).read_text(encoding="utf-8")) == {
        "corpus_hash": "a" * 64,
        "phases": [0, 3, 6],
    }
    # merges exist for declared letters; no merge was learned for an undeclared one
    assert len(sub.encode("bbbbbb")) == 6
    assert len(sub.encode("dddddd")) < 6
    # the same subset reuses the cache; a full run from the same dir does not
    get_or_train_tokenizer(toy_dirs["train"], 400, cache, corpus_hash="a" * 64, phases=(0, 3, 6))
    with pytest.raises(ValueError, match="trained on phases \\[0, 3, 6\\] of this corpus, this run declares all phases"):
        get_or_train_tokenizer(toy_dirs["train"], 400, cache, corpus_hash="a" * 64)
    full_cache = tmp_path / "full.json"
    get_or_train_tokenizer(toy_dirs["train"], 400, full_cache, corpus_hash="a" * 64)
    assert json.loads(corpus_meta_path(full_cache).read_text(encoding="utf-8")) == {"corpus_hash": "a" * 64}
    with pytest.raises(ValueError, match="trained on all phases of this corpus, this run declares phases"):
        get_or_train_tokenizer(toy_dirs["train"], 400, full_cache, corpus_hash="a" * 64, phases=(0, 3, 6))
