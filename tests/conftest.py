"""Synthetic fixtures for the CPU smoke tests.

`AGENTS.md`: a CPU smoke test uses the toy config (<= 1M parameters), synthetic
or 20-story fixture data and <= 50 steps, and asserts mechanics. **No test in
this package reads the exam directory.** The "exam" directory built here holds
one placeholder file whose only job is to exist so `guard.py` can check that it
is not the training directory and that its hashes are disjoint.

The synthetic corpus is built so provenance is checkable from content: phase
`k`'s stories are written entirely in the letter `chr(ord('a') + k)`, so a byte
tokenizer turns a phase-`k` sequence into ids drawn from `{97+k, 32, 10}`. A
test can therefore assert that every replay sequence came from a phase strictly
earlier than the current one without trusting the loader's own bookkeeping.
"""

from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from training.config import EXAM_TYPES, N_PHASES, TOY, TrainConfig, warmup_for
from training.runrecord import MATRIX_KEYS
from training.train import TOY_RUN

NL = chr(10)
STORIES_PER_PHASE = 20
SENTENCES_PER_STORY = 12
WORDS_PER_SENTENCE = 8
REPLAY_STORIES = 8


def phase_letter(phase: int) -> str:
    return chr(ord("a") + phase)


def phase_byte(phase: int) -> int:
    return ord(phase_letter(phase))


def make_story(phase: int, index: int) -> str:
    """A story in phase `phase`'s single letter, with a deterministic shape."""
    letter = phase_letter(phase)
    lines = []
    for s in range(SENTENCES_PER_STORY):
        words = [letter * (2 + ((index + s + w) % 5)) for w in range(WORDS_PER_SENTENCE)]
        lines.append(" ".join(words) + ".")
    return "\n".join(lines) + "\n"


def prompt_hash(phase: int, index: int) -> str:
    return hashlib.sha256(f"toy:{phase}:{index}".encode("utf-8")).hexdigest()[:16]


def write_train_dir(
    root: Path,
    n_phases: int = N_PHASES,
    stories: int = STORIES_PER_PHASE,
    model: str = "synthetic-fixture",
    model_by_phase: dict[int, str] | None = None,
) -> Path:
    """`train_phase_{k}.jsonl` plus `replay/phase_{k}.json`, the frozen shapes.

    `model_by_phase` overrides the generator id for particular phases, which is
    how a test builds the corpus that must refuse to train.
    """
    root.mkdir(parents=True, exist_ok=True)
    replay_dir = root / "replay"
    replay_dir.mkdir(exist_ok=True)
    for k in range(n_phases):
        lines = []
        for i in range(stories):
            lines.append(
                json.dumps(
                    {
                        "prompt_hash": prompt_hash(k, i),
                        "phase": k,
                        "tier": f"tier_{k}",
                        "story": make_story(k, i),
                        "model": (model_by_phase or {}).get(k, model),
                        "timestamp": "2026-09-21T00:00:00Z",
                    }
                )
            )
        (root / f"train_phase_{k}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (replay_dir / f"phase_{k}.json").write_text(
            json.dumps({"seed": 0, "prompt_hashes": [prompt_hash(k, i) for i in range(REPLAY_STORIES)]}),
            encoding="utf-8",
        )
    return root


def write_exam_dir(root: Path) -> Path:
    """A placeholder exam directory. Nothing in the test suite reads it; it
    exists so the guard has a real path to refuse to overlap with."""
    (root / "stories").mkdir(parents=True, exist_ok=True)
    (root / "stories" / "exam_phase_0.jsonl").write_text(
        json.dumps({"id": "placeholder", "note": "not read by any test"}) + "\n", encoding="utf-8"
    )
    return root


def _jsonl(rows: list[dict]) -> str:
    """One JSON object per line, UTF-8 -- the frozen line format."""
    return NL.join(json.dumps(r) for r in rows) + NL


def write_scoreable_exam_dir(root: Path, n_phases: int = N_PHASES, items_per_type: int = 4) -> Path:
    """A *real*, scoreable toy exam directory: synthetic content, frozen shapes.

    `write_exam_dir` above is a placeholder that the real evaluator never reads;
    it exists only so the guard has a path to refuse to overlap with. This one
    parses and scores, so a test can run `training.evaluate` unstubbed through
    `train.run` -- the seam that every other test replaces with a fake, and that
    therefore nothing tested until the 2026-09-22 audit found the pipeline dead.

    Layout is `training/evaluate.py`'s (`stories/exam_phase_{k}.jsonl`,
    `probes/{cloze,continuation}_phase_{k}.jsonl`) and the probe schema is
    AGENTS.md's, including `option_sources`, which amendment 2 made required and
    which the evaluator asserts at load. Every byte here is synthetic: no test
    reads real exam text.
    """
    stories_dir, probes_dir = root / "stories", root / "probes"
    stories_dir.mkdir(parents=True, exist_ok=True)
    probes_dir.mkdir(parents=True, exist_ok=True)

    def story_id(phase: int, i: int) -> str:
        return f"exam_{phase}_{i}"

    def story_text(phase: int, i: int) -> str:
        letter = phase_letter(phase)
        return f"{letter * 3} {letter * 4} {letter * 2} story {i}." + NL

    def story_sha(phase: int, i: int) -> str:
        # the manifest's normalised story hash of the story each option cites,
        # which the evaluator checks option_sources against (audit 2026-09-25)
        from training.guard import story_sha256

        return story_sha256(story_text(phase, i))

    for k in range(n_phases):
        letter = phase_letter(k)
        other = (k + 1) % n_phases
        rows = [
            {"id": story_id(k, i), "phase": k, "story": story_text(k, i)}
            for i in range(items_per_type)
        ]
        (stories_dir / f"exam_phase_{k}.jsonl").write_text(
            _jsonl(rows), encoding="utf-8"
        )

        cloze = [
            {
                "id": f"cloze_{k}_{i}",
                "phase": k,
                "text_with_mask": f"the {letter * 3} went to [MASK] school today.",
                "answer": letter * 2,
                "candidates": [letter * 2] + [f"{letter}{d}" for d in "xyzwvutsrqponmlkjih"],
            }
            for i in range(items_per_type)
        ]
        (probes_dir / f"cloze_phase_{k}.jsonl").write_text(
            _jsonl(cloze), encoding="utf-8"
        )

        cont = []
        for i in range(items_per_type):
            distractor_phases = [(k + 1 + d) % n_phases for d in range(3)]
            # never the item's own phase, so the probe stays a phase test
            distractor_phases = [p if p != k else other for p in distractor_phases]
            cont.append(
                {
                    "id": f"cont_{k}_{i}",
                    "phase": k,
                    "prefix": f"once upon a time in phase {letter}, ",
                    "options": [f"{letter * 2} the true continuation {i}."]
                    + [f"{phase_letter(p)}{phase_letter(p)} a continuation from elsewhere." for p in distractor_phases],
                    "answer_index": 0,
                    "distractor_phases": distractor_phases,
                    # amendment 2: the answer's source is this phase; no
                    # distractor shares the item's phase or the answer's story.
                    "option_sources": [{"story_id": story_id(k, i), "story_sha256": story_sha(k, i), "phase": k}]
                    + [
                        {"story_id": story_id(p, i), "story_sha256": story_sha(p, i), "phase": p}
                        for p in distractor_phases
                    ],
                }
            )
        (probes_dir / f"continuation_phase_{k}.jsonl").write_text(
            _jsonl(cont), encoding="utf-8"
        )
    return root


FIXTURE_MODEL = "synthetic-fixture"


def manifest_stories_of(exam_dir: Path) -> list[dict]:
    """The manifest's per-story entries for every story line under
    `exam_dir/stories/`, or one invented entry for the placeholder exam dir
    (whose only line is not a story), so the guard's required list is never
    empty."""
    from training.guard import manifest_story_entry, story_sha256

    out = []
    stories_dir = Path(exam_dir) / "stories"
    for f in sorted(stories_dir.glob("*.jsonl")) if stories_dir.is_dir() else []:
        for line in f.read_text(encoding="utf-8").splitlines():
            row = json.loads(line) if line.strip() else {}
            if "story" in row:
                out.append(manifest_story_entry(row))
    if not out:
        out.append(
            {
                "story_id": "exam-placeholder",
                "prompt_hash": "exam-placeholder",
                "story_sha256": story_sha256("a placeholder exam story no fixture writes"),
                "phase": 0,
            }
        )
    return out


def write_manifest(
    path: Path,
    exam_dir: Path,
    extra_hashes: list[str] | None = None,
    *,
    stories: list[dict] | None = None,
    extra_stories: list[dict] | None = None,
    generator_model: str = FIXTURE_MODEL,
    experiment_id: str = "toy",
) -> Path:
    """A manifest of the exam directory, plus any planted hashes.

    `stories` / `extra_stories` are the per-story entries the guard checks every
    training line against (default: derived from `exam_dir`); `generator_model`
    must equal the training corpus's model (`write_train_dir`'s default)."""
    files = []
    for f in sorted(p for p in Path(exam_dir).rglob("*") if p.is_file()):
        files.append(
            {
                "path_relative_to_exam_dir": f.relative_to(exam_dir).as_posix(),
                "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
                "phase": 0,
                "kind": "story",
            }
        )
    for i, h in enumerate(extra_hashes or []):
        files.append({"path_relative_to_exam_dir": f"planted_{i}", "sha256": h, "phase": 0, "kind": "story"})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "experiment_id": experiment_id,
                "frozen_at": "2026-09-21T00:00:00Z",
                "generator_commit": "0" * 40,
                "seed": 42,
                "config_hashes": {},
                "files": files,
                "near_duplicates_dropped": {str(k): 0 for k in range(N_PHASES)},
                "generator_model": generator_model,
                "stories": (manifest_stories_of(exam_dir) if stories is None else list(stories))
                + list(extra_stories or []),
            }
        ),
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# fixtures


@pytest.fixture
def toy_dirs(tmp_path: Path) -> dict:
    """train / exam / manifest / out / shared, all disjoint."""
    train = write_train_dir(tmp_path / "train")
    exam = write_exam_dir(tmp_path / "exam")
    manifest = write_manifest(tmp_path / "manifest.json", exam)
    return {
        "train": train,
        "exam": exam,
        "manifest": manifest,
        "results": tmp_path / "results",
        "shared": tmp_path / "shared",
        "tmp": tmp_path,
    }


@pytest.fixture
def scoreable_dirs(tmp_path: Path) -> dict:
    """`toy_dirs`, but with an exam directory the real evaluator can score."""
    train = write_train_dir(tmp_path / "train")
    exam = write_scoreable_exam_dir(tmp_path / "exam")
    return {
        "train": train,
        "exam": exam,
        "manifest": write_manifest(tmp_path / "manifest.json", exam),
        "results": tmp_path / "results",
        "shared": tmp_path / "shared",
        "tmp": tmp_path,
    }


@pytest.fixture
def toy_cfg() -> TrainConfig:
    """`TrainConfig` at toy batch size, for the hook-side tests.

    Only the two sizes a toy run fixes are changed. Every hyperparameter from
    `PLAN.md`'s table -- lr, betas, weight decay, grad clip, epochs, LoRA rank
    and alpha, lambda -- is left exactly as the grid will run it, so a test
    cannot pass because a fixture softened it. Warmup comes from
    `config.warmup_for(..., scaled=True)`, the same call a --pilot run makes.
    """
    return replace(
        TrainConfig(),
        sequences_per_batch=TOY_RUN.sequences_per_batch,
        warmup_steps=warmup_for(TOY_RUN.max_steps, scaled=True),
    )


def toy_args(dirs: dict, arm: str, seed: int = 0, out: Path | None = None, resume: bool = False) -> Namespace:
    """The Namespace `training.train.run` takes, at toy size."""
    return Namespace(
        arm=arm,
        seed=seed,
        train_dir=dirs["train"],
        exam_dir=dirs["exam"],
        out=out if out is not None else dirs["results"] / f"{arm}_s{seed}",
        resume=resume,
        pilot=False,
        manifest=dirs["manifest"],
        replay_dir=None,
        phase0_dir=dirs["shared"],
        toy=True,
        cpu=True,
    )


# ---------------------------------------------------------------------------
# the stubbed hooks
#
# `training/lora.py`, `consolidate.py` and `evaluate.py` belong to other agents
# and are stubbed here, per the brief. `identity_after_phase` is the real hook
# from `training/hooks.py`.


def fake_evaluate_all(model: torch.nn.Module, exam_dir, phases) -> dict:
    """A deterministic stand-in for `training.evaluate.evaluate_all`.

    Returns every key of `MATRIX_KEYS` -- the three EXAM_TYPES plus the
    required `continuation_summed` -- each a list of exactly 7 floats. The
    scores are a cheap deterministic function of the model's weights, which is
    what makes "resume equals uninterrupted" a real test: two runs agree on the
    matrix only if they agree on every weight. It never opens `exam_dir`.
    """
    with torch.no_grad():
        signal = float(model.wte.weight.detach().float().sum().item())
        drift = float(model.blocks[0].mlp.c_fc.weight.detach().float().abs().sum().item())
    out = {}
    for t_index, exam_type in enumerate(MATRIX_KEYS):
        out[exam_type] = [
            round(((signal * (j + 1) + drift * (t_index + 1)) % 1.0), 12) for j in range(N_PHASES)
        ]
    return out


def counting_evaluate_all(calls: list) -> callable:
    def _inner(model, exam_dir, phases):
        calls.append(int(getattr(model, "_phase_marker", -1)))
        return fake_evaluate_all(model, exam_dir, phases)

    return _inner


# ---------------------------------------------------------------------------
# shared helpers for the hook-side tests (training/lora.py, consolidate.py)
#
# `trainer-core` owns this file; these three exist so `consolidation`'s tests
# can build a toy model, a batch of ids and a `PhaseContext` without depending
# on the training loop. Keep them boring: a test that fails here should fail
# because the code under test is wrong, not because a fixture is clever.


def make_model(seed: int = 0):
    """A toy `GPT` with deterministic weights (<= 1M parameters)."""
    from training.model import build_model

    return build_model(TOY, torch.Generator().manual_seed(seed))


def token_batch(n_sequences: int, seed: int = 0, block_size: int | None = None) -> torch.Tensor:
    """`[n_sequences, block_size]` token ids in the toy vocabulary."""
    g = torch.Generator().manual_seed(seed)
    size = (n_sequences, block_size if block_size is not None else TOY.block_size)
    return torch.randint(0, TOY.vocab_size, size, generator=g)


class RecordingContext:
    """A `PhaseContext` whose loaders, logger and timer record what a hook did.

    `sequences_per_batch` is N, the field `PhaseContext` gained on 2026-09-22;
    a hook must read it from the context rather than from a `TrainConfig`
    default, or the night desyncs from the day when train.py runs a non-default
    N. `phase_calls` is the `(phase, n_sequences)` of every `make_phase_loader`
    call, `replay_calls` the `n_sequences` of every `make_replay_loader` call,
    `timers` the span names entered and `logs` the lines logged. Each loader
    hands back exactly one batch, so a hook that wants more calls the loader
    again -- which is what makes the call counts assertable.
    """

    def __init__(
        self,
        arm: str,
        phase: int,
        replay_fraction: float,
        steps: int,
        *,
        seed: int = 0,
        sequences_per_batch: int | None = None,
        lr: float = 3e-4,
        warmup: int = 2,
        device: torch.device | None = None,
        amp_dtype: torch.dtype | None = None,
    ) -> None:
        from contextlib import contextmanager

        from training.hooks import PhaseContext

        self.phase_calls: list[tuple[int, int]] = []
        self.replay_calls: list[int] = []
        self.timers: list[str] = []
        self.logs: list[str] = []
        self._draw = 0

        def phase_loader(p: int, n: int):
            self.phase_calls.append((p, n))
            self._draw += 1
            return [token_batch(n, seed=1000 + self._draw)]

        def replay_loader(n: int):
            if replay_fraction == 0.0 or phase == 0:
                return iter(())
            self.replay_calls.append(n)
            self._draw += 1
            return [token_batch(n, seed=2000 + self._draw)]

        @contextmanager
        def timer(span: str):
            self.timers.append(span)
            yield

        self.ctx = PhaseContext(
            arm=arm,
            seed=seed,
            phase=phase,
            replay_fraction=replay_fraction,
            device=device or torch.device("cpu"),
            amp_dtype=amp_dtype,
            make_phase_loader=phase_loader,
            make_replay_loader=replay_loader,
            steps=steps,
            sequences_per_batch=(
                sequences_per_batch if sequences_per_batch is not None else TOY_RUN.sequences_per_batch
            ),
            lr=lr,
            warmup=warmup,
            log=self.logs.append,
            timer=timer,
        )


# ---------------------------------------------------------------------------
# Probe helpers (added by `evaluator`, 2026-09-22)
#
# For tests that build one probe item at a time rather than a whole directory -
# the exam-directory builder is `write_scoreable_exam_dir` above, and there is
# only one of it. Every string these produce is invented here; nothing derives
# from the real exam directory, which no test reads.
# ---------------------------------------------------------------------------

CLOZE_CANDIDATES = 20
CONTINUATION_OPTIONS = 4


def _write_probe_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def option_sources(phase: int, item: int, n_options: int, answer_index: int) -> list[dict]:
    """`option_sources` for a synthetic continuation item: the answer drawn from
    this phase, every distractor from another one, ids and hashes invented."""
    out = []
    for i in range(n_options):
        if i == answer_index:
            src_phase = phase
        else:
            src_phase = (phase + 1 + i) % N_PHASES
            if src_phase == phase:
                src_phase = (phase + 1) % N_PHASES
        out.append(
            {
                "story_id": f"story_{src_phase}_{item}_{i}",
                "story_sha256": f"{(phase * 37 + item * 11 + i) % 256:02x}" * 32,
                "phase": src_phase,
            }
        )
    return out


# There is deliberately no second `write_scoreable_exam_dir` here. `evaluator`
# appended one on 2026-09-22 within minutes of `trainer-core` adding the one at
# the top of this file, and the later definition silently shadowed the earlier:
# the end-to-end test and the `toy_dirs` fixture were both getting a builder
# their authors had never seen. One builder, the one above; the helpers in this
# section are for probes built item by item inside a test.
