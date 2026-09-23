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
