# Lifespan — agent protocol

Read `CLAUDE.md` first (the rules), then `PLAN.md` (the experiment and its fixed
hypotheses). This file says who builds what, which contracts are shared, and in
what order. Start a build session with `claude --agent lifespan-lead`.

## The thing that makes this repo different

The deliverable is a measurement, and the ways it goes wrong are silent. An exam
story that leaks into training, an arm that quietly got more tokens, a threshold
nudged after the first matrix came back: nothing crashes, the heatmap looks
fine, and the result is worth nothing. So:

- **H1–H4 and their thresholds are frozen.** No agent edits the hypothesis table
  in `PLAN.md`. After the first training run starts, nobody does.
- **One agent reads exam text.** `exam-keeper` builds and freezes the exams.
  `evaluator` writes the code that scores them, and tests it on synthetic
  fixtures. Everyone else sees hashes and scores, never the text.
- **Nothing is tuned against an exam score.** The pilot exists to find bugs and
  to check H1's gate. Hyperparameters in `PLAN.md` are fixed for the grid; a
  change is a dated entry in `docs/DECISIONS.md` made before the grid, not after.
- **No training on the laptop.** A CPU smoke test is a test, not training: toy
  config (<= 1M parameters), synthetic or 20-story fixture data, <= 50 steps,
  asserting on mechanics (loss falls, checkpoint resumes, matrix has the right
  shape). Anything that produces a number someone might quote runs on Kaggle or
  the rented A100.
- **Money and quota need the owner's yes.** Story generation beyond a 20-story
  smoke batch, a Kaggle GPU session, an A100 rental.
- **The story generator is not Gemini.** `PLAN.md` was written around Gemini
  Flash; on 2026-09-21 the owner moved generation off it, probably to Claude.
  The provider and exact model id are the owner's call, recorded in
  `docs/DECISIONS.md` before the first paid batch. One model generates every
  story, train and exam, all seven phases: a model change between phases is a
  confound the forgetting curve cannot separate from forgetting.

## Ownership

One owner per file at a time. Anything not listed belongs to `lifespan-lead`.

| Agent | Owns | Never touches |
| --- | --- | --- |
| `lifespan-lead` | `PLAN.md`, `CLAUDE.md`, `AGENTS.md`, `docs/`, `pyproject.toml` / requirements, commits, spend gates, the H1–H4 verdicts | — |
| `curriculum-data` | In `D:\Tiny Models\curriculum-learning`: `generate_prompts.py` and the `--split` flag, the batch response generator, lexicons for phases 3–6, the age-check gate, the egg-info cleanup | exam output directories; this repo's `training/` |
| `exam-keeper` | The exam directory (outside both repos), the probe builder, the near-duplicate check, `exams/manifest.json`, the replay-buffer selection file | training code; regenerating anything after the freeze |
| `trainer-core` | `training/tokenizer.py`, `model.py`, `data.py`, `train.py`, `checkpoint.py`, `guard.py`, `runrecord.py` and their tests | `lora.py`, `consolidate.py`, `evaluate.py`; the exam directory |
| `consolidation` | `training/lora.py`, `training/consolidate.py` and their tests | the training loop's control flow beyond the agreed hook; the exam directory |
| `evaluator` | `training/evaluate.py`, `training/metrics.py`, `training/report.py` (aggregation, heatmaps, forgetting curves) and their tests | how a model is trained; real exam text in tests |
| `run-operator` | `notebooks/`, the contents and validity of `results/<run_id>/`, GPU-hour accounting, the reproduction runs | `training/` source; any number inside a result file |
| `lifespan-auditor` | nothing — read-only, runs tests | any file |

`train.py` is the shared file. `trainer-core` owns it; `consolidation` and
`evaluator` plug in through the two hooks below and ask for changes to the loop
in their report instead of editing it.

**Only the lead adds a dependency.** Kaggle's image pins torch; a package that
is not already there costs a pip install in every session that can die.

## Contracts — frozen 2026-09-21 for waves 2 and 3

Frozen by the lead before dispatching waves 2 and 3, and pasted in full into
every brief that holds one. Changing a frozen contract is a lead decision
announced to every holder. Where `PLAN.md` left something underspecified, the
resolution is marked `[frozen 2026-09-21]` and has a matching entry in
`docs/DECISIONS.md`; none of these touch the H1–H4 table.

```text
# Training story line (curriculum-learning writes, training/data.py reads)
{prompt_hash, phase, tier, story, model, timestamp}          # PLAN.md, repo checklist
# one JSON object per line, UTF-8, file name train_phase_{k}.jsonl, k in 0..6

# exams/manifest.json (exam-keeper writes, training/guard.py reads)
{experiment_id, frozen_at, generator_commit, seed, config_hashes: {...},
 files: [{path_relative_to_exam_dir, sha256, phase, kind: story|cloze|continuation}],
 near_duplicates_dropped: {phase: n}}

# Probe items (exam-keeper writes, training/evaluate.py reads)
cloze:        {id, phase, text_with_mask, answer, candidates: [phase lexicon]}
continuation: {id, phase, prefix, options: [4 paragraphs], answer_index,
               distractor_phases: [3]}                        # distractors from exam stories only
# [frozen 2026-09-21] text_with_mask contains the literal substring "[MASK]"
#   exactly once; the scorer forms a candidate sequence by str.replace of that
#   substring and scores the whole filled text. `answer` is the surface string,
#   and is always a member of `candidates`.
# [frozen 2026-09-21] len(candidates) == 20: the true answer plus 19 drawn from
#   the same phase's lexicon with the exam seed. Chance accuracy is 5%.

# Exam type keys — these three strings, everywhere (config, matrix.json, plots)
EXAM_TYPES = ("perplexity", "cloze", "continuation")
# [frozen 2026-09-21] "continuation" accuracy is the headline metric for H1-H4;
#   "cloze" accuracy is reported beside it at every step. Where the two give
#   different verdicts the result is reported as split and neither is dropped.
#   "perplexity" is the continuous forgetting curve and is never a verdict
#   metric (a 15% change in a loss does not mean what H1's threshold means).
# [frozen 2026-09-21] continuation is scored length-normalised (mean per-token
#   log-likelihood of the option given the prefix); the summed score is computed
#   and stored too, headlined never. Ties count as wrong.

# Replay buffer (exam-keeper selects once, all arms read)
replay/phase_{k}.json: {seed, prompt_hashes: [500]}           # same file for every arm and seed

# Entry point — one script, arms are data
python -m training.train --arm {phase0,A,B,C,D,D-nr,E} --seed {0,1,2}
    --train-dir <dir> --exam-dir <dir> --out results/<run_id> [--resume] [--pilot]
ARMS = {A: full-ft, replay 0.0 | B: full-ft, replay 0.3 | C: lora+merge |
        D: lora+distill, replay 0.3 | D-nr: lora+distill, replay 0.0 | E: joint}
# Decided 2026-09-21 (PLAN.md, Training arms; docs/DECISIONS.md):
#  - Phase 0 is full training, run once per seed (`--arm phase0`), and its checkpoint,
#    matrix row and wall clock are shared by A, B, C, D, D-nr. E trains from the random init.
#  - Replay is on top of the budget: same steps per phase in every arm, N new-phase
#    sequences per batch, plus round(N*0.3/0.7) replay sequences where replay = 0.3.
# [frozen 2026-09-21] --pilot is the only knob that changes sizes: 1,000 training
#   stories and 100 items per exam type per phase, same model, same 4 epochs, same
#   arms. It never changes a hyperparameter in PLAN.md's table. A pilot run's
#   result folder carries "pilot": true in config.json and is never mixed with grid
#   runs in report.py.

# Hooks train.py calls (so lora/consolidate/evaluate never edit the loop)
after_phase(model, phase_k, ctx) -> model        # C: merge; D: distill; A/B/E: identity
evaluate_all(model, exam_dir, phases) -> {exam_type: [scores for phases 0..6]}
# [frozen 2026-09-21] ctx is a dataclass training.hooks.PhaseContext with fields:
#   arm: str; seed: int; phase: int; replay_fraction: float; device: torch.device;
#   amp_dtype: torch.dtype | None; make_phase_loader: Callable[[int, int], Iterable]
#     -> batches of token ids for one phase, args (phase, n_sequences_per_batch);
#   make_replay_loader: Callable[[int], Iterable] -> batches drawn from the replay
#     buffers of phases < phase (empty iterable when replay_fraction == 0.0);
#   steps: int; lr: float; warmup: int; log: Callable[[str], None];
#   timer: contextmanager(str) that accumulates into timings.json.
#   after_phase returns the model to carry forward and must not mutate ctx.
#   train.py calls it after every phase including phase 0 for arms C, D, D-nr.
# [frozen 2026-09-21] evaluate_all returns every key of EXAM_TYPES, each a list of
#   exactly 7 floats indexed by exam phase; a phase not yet trained is still scored
#   (that is the matrix's upper triangle). Higher is better for cloze and
#   continuation; for perplexity the stored value is mean per-token loss (lower is
#   better) and metrics.py negates it where a "score" is required.

# results/<run_id>/   run_id = {arm}_s{seed}_{commit7}_{utc}
matrix.json   {untrained: {exam_type: [7]}, M: {exam_type: [[7]x7]}}   # M[i][j]: after phase i, exam j
timings.json  {per_phase: [{phase, train_s, consolidate_s, eval_s}], gpu_hours, gpu_name}
config.json   full resolved config incl. arm, seed, token budget per phase, data + manifest hashes
commit.txt    commit sha, dirty flag (a dirty tree is not a result), torch/CUDA/driver versions

# Toy config for CPU smoke tests — one definition, used by every agent's tests
# [frozen 2026-09-21] training.config.TOY: 2 layers, 64 hidden, 2 heads, ctx 64,
#   vocab 256 (byte-level, no tokenizer training), <= 1M parameters, 7 phases of
#   20 synthetic stories, 4 exam items per type per phase, <= 50 steps. Synthetic
#   fixtures only; no agent's test reads the exam directory.
```

`untrained` is in the matrix file because forward transfer (PLAN.md,
Evaluation) is measured against it and it cannot be recovered afterwards.

## Waves

Follow `PLAN.md`'s schedule; a wave is green when its gate is answered in
writing in `docs/DECISIONS.md` with the measurement.

| Wave | Agents (parallel within a wave) | Gate |
| --- | --- | --- |
| 1 — data and exams | `curriculum-data` (split flag, response generator, age gate), then `exam-keeper` (exam run, probes, near-dup check, manifest), `lifespan-auditor` on the freeze | One phase generated and inspected; age-check pass rate recorded; exam dir provably disjoint from training; manifest committed |
| 2 — baseline pipeline | `curriculum-data` (phases 1–6, lexicons 3–6) ‖ `trainer-core` ‖ `evaluator`; then `run-operator` (Kaggle pilot, arms A and E) | Arm A forgets on the pilot (H1), or the curriculum goes back |
| 3 — the mechanism | `consolidation`; `lifespan-auditor` on arm parity; `run-operator` (pilot C, D, D-nr, then the 18-run grid) | 18 runs with valid result folders and timings |
| 4 — the claim | `run-operator` (reproduction of the headline arms), `evaluator` (report), `lifespan-auditor` on the verdicts | Two runs reproduce within seed spread; H1–H4 marked in `PLAN.md` |

## Working across two repos

`D:\Tiny Models\curriculum-learning` has a space in its path: quote it, and use
`git -C "<path>"` rather than `cd`. It had uncommitted work on 2026-09-21
(`engine/arcs.py` modified, `response/scratch.py` staged); check `git status`
there before editing and do not fold the owner's changes into yours. `D:\Reflex`
and `D:\WORK` are read-only from here.
