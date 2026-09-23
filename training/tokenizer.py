"""The BPE tokenizer, trained on the training stories and nothing else.

`PLAN.md`, "Model and compute": "a small BPE tokenizer trained on the story
corpus so the embedding matrix does not dominate the parameter count", 8k-16k
vocab; the lead froze 8,192 (`config.REAL.vocab_size`).

**A BPE trained on exam text is a leak.** The corpus for `train_bpe` is the
`train_phase_{k}.jsonl` files under `--train-dir` and only those; this module
never takes an exam path, and `corpus_files` refuses anything that is not a
training-story file directly under the training directory.

Two implementations share one interface (`encode`, `decode`, `vocab_size`,
`eot_id`, `save`, `load`):

- `BPETokenizer` -- the real one, `tokenizers` ByteLevel BPE.
- `ByteTokenizer` -- the toy one, identity over the 256 byte values, so a CPU
  smoke test never trains a tokenizer (`AGENTS.md`, toy config: "vocab 256
  byte-level (no tokenizer training in a test)").
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, Protocol, runtime_checkable

TRAIN_FILE_GLOB = "train_phase_*.jsonl"
END_OF_TEXT = "<|endoftext|>"


@runtime_checkable
class Tokenizer(Protocol):
    vocab_size: int
    eot_id: int

    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...
    def save(self, path: Path) -> None: ...


# ---------------------------------------------------------------------------
# corpus


def corpus_files(train_dir: Path) -> list[Path]:
    """The training-story files, sorted by phase. Fails closed.

    Anything that is not `train_phase_{k}.jsonl` sitting directly in
    `train_dir` is not corpus. The tokenizer never sees the exam directory, and
    this is where that is enforced rather than assumed.
    """
    train_dir = Path(train_dir).resolve()
    if not train_dir.is_dir():
        raise FileNotFoundError(f"train dir does not exist: {train_dir}")
    files = sorted(train_dir.glob(TRAIN_FILE_GLOB), key=lambda p: _phase_of(p))
    if not files:
        raise FileNotFoundError(f"no {TRAIN_FILE_GLOB} under {train_dir}")
    for f in files:
        if f.resolve().parent != train_dir:
            raise ValueError(f"training story file outside the training dir: {f}")
    return files


def _phase_of(path: Path) -> int:
    stem = path.stem  # train_phase_3
    try:
        return int(stem.rsplit("_", 1)[1])
    except (IndexError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError(f"cannot read a phase number from {path.name}") from exc


def iter_stories(path: Path) -> Iterator[str]:
    """Yield the `story` field of every line of a training jsonl.

    Line shape is the frozen contract:
    `{prompt_hash, phase, tier, story, model, timestamp}`.
    """
    with Path(path).open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno} is not JSON") from exc
            if "story" not in rec:
                raise ValueError(f"{path}:{lineno} has no 'story' field")
            yield rec["story"]


def iter_corpus(train_dir: Path) -> Iterator[str]:
    for f in corpus_files(train_dir):
        yield from iter_stories(f)


# ---------------------------------------------------------------------------
# real tokenizer


class BPETokenizer:
    """ByteLevel BPE over the training stories."""

    def __init__(self, backend) -> None:  # tokenizers.Tokenizer
        self._tok = backend
        self.vocab_size = backend.get_vocab_size()
        eot = backend.token_to_id(END_OF_TEXT)
        if eot is None:
            raise ValueError(f"tokenizer has no {END_OF_TEXT} token")
        self.eot_id = eot

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False).ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [e.ids for e in self._tok.encode_batch(texts, add_special_tokens=False)]

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=False)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._tok.save(str(path))

    @classmethod
    def load(cls, path: Path) -> "BPETokenizer":
        from tokenizers import Tokenizer as _T

        return cls(_T.from_file(str(Path(path))))


def train_bpe(corpus: Iterable[str], vocab_size: int, out_path: Path | None = None) -> BPETokenizer:
    """Train a ByteLevel BPE of exactly `vocab_size` tokens on `corpus`.

    `corpus` is an iterable of training stories -- see `iter_corpus`. Nothing
    from the exam directory may reach it.
    """
    from tokenizers import Tokenizer as _T
    from tokenizers import decoders, models, pre_tokenizers, trainers

    tok = _T(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[END_OF_TEXT],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(corpus, trainer=trainer)
    wrapped = BPETokenizer(tok)
    if out_path is not None:
        wrapped.save(Path(out_path))
    return wrapped


def train_bpe_from_train_dir(train_dir: Path, vocab_size: int, out_path: Path | None = None) -> BPETokenizer:
    return train_bpe(iter_corpus(Path(train_dir)), vocab_size, out_path)


# ---------------------------------------------------------------------------
# toy tokenizer


class ByteTokenizer:
    """UTF-8 bytes as ids. Vocab 256, `eot_id` 0 (NUL never occurs in text).

    Used by every CPU smoke test so a test never trains a tokenizer, and by the
    toy config whose `vocab_size` is 256.
    """

    def __init__(self, vocab_size: int = 256) -> None:
        if vocab_size != 256:
            raise ValueError("ByteTokenizer is byte-level; vocab_size must be 256")
        self.vocab_size = 256
        self.eot_id = 0

    def encode(self, text: str) -> list[int]:
        return [b for b in text.encode("utf-8") if b != 0]

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [self.encode(t) for t in texts]

    def decode(self, ids: list[int]) -> str:
        return bytes(i for i in ids if i != 0).decode("utf-8", errors="replace")

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"kind": "byte", "vocab_size": 256}), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ByteTokenizer":
        rec = json.loads(Path(path).read_text(encoding="utf-8"))
        if rec.get("kind") != "byte":
            raise ValueError(f"{path} is not a byte tokenizer")
        return cls()


def load_tokenizer(path: Path) -> Tokenizer:
    """Load whichever kind was saved to `path`."""
    path = Path(path)
    rec = json.loads(path.read_text(encoding="utf-8"))
    if rec.get("kind") == "byte":
        return ByteTokenizer()
    return BPETokenizer.load(path)


def get_or_train_tokenizer(
    train_dir: Path, vocab_size: int, cache_path: Path, corpus_hash: str | None = None
) -> Tokenizer:
    """The tokenizer a run uses: byte-level at vocab 256, otherwise a BPE
    trained once on the training stories and cached at `cache_path`.

    Caching keeps every arm and seed on the same vocabulary, which they must be
    for the exam scores to be comparable at all. `corpus_hash` is the guard's
    hash of the training corpus; when given it is written beside the cache and
    checked on reload, so a tokenizer trained on a *different* corpus is refused
    here rather than showing up indirectly, phases later, as a fingerprint
    mismatch on a shared checkpoint (leakage audit, 2026-09-22).
    """
    cache_path = Path(cache_path)
    if vocab_size == 256:
        return ByteTokenizer()
    if cache_path.exists():
        tok = load_tokenizer(cache_path)
        _check_fits(tok.vocab_size, vocab_size, f"cached tokenizer at {cache_path}")
        _check_corpus(cache_path, corpus_hash)
        return tok
    tok = train_bpe_from_train_dir(Path(train_dir), vocab_size, cache_path)
    _check_fits(tok.vocab_size, vocab_size, "the tokenizer just trained")
    _write_corpus_hash(cache_path, corpus_hash)
    return tok


def corpus_meta_path(cache_path: Path) -> Path:
    return Path(cache_path).with_name(Path(cache_path).name + ".corpus.json")


def _write_corpus_hash(cache_path: Path, corpus_hash: str | None) -> None:
    if corpus_hash is None:
        return
    corpus_meta_path(cache_path).write_text(
        json.dumps({"corpus_hash": corpus_hash}), encoding="utf-8"
    )


def _check_corpus(cache_path: Path, corpus_hash: str | None) -> None:
    """Refuse a cached tokenizer that was trained on a different corpus."""
    if corpus_hash is None:
        return
    meta = corpus_meta_path(cache_path)
    if not meta.is_file():
        raise ValueError(
            f"cached tokenizer at {cache_path} has no {meta.name}, so the corpus it was trained "
            "on cannot be established. Delete it and let this run retrain the tokenizer."
        )
    cached = json.loads(meta.read_text(encoding="utf-8")).get("corpus_hash")
    if cached != corpus_hash:
        raise ValueError(
            f"cached tokenizer at {cache_path} was trained on corpus {cached}, this run's corpus "
            f"is {corpus_hash}. A vocabulary from other data silently changes every token id and "
            "every exam score. Delete the cache to retrain, or point --phase0-dir at this "
            "corpus's own directory."
        )


def _check_fits(trained: int, model_vocab: int, what: str) -> None:
    """A tokenizer may end up *smaller* than the requested vocab -- BPE stops
    when the corpus has no merges left, which happens on a small pilot corpus.
    That is harmless: the embedding table is simply larger than it needs to be,
    and the frozen parameter count is unchanged. A tokenizer *larger* than the
    model's vocab is not harmless; its ids would index past the table.
    """
    if trained > model_vocab:
        raise ValueError(f"{what} has vocab {trained}, larger than the model's {model_vocab}")
