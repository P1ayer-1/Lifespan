# Decisions

Append-only. Date, the measurement, the decision it drove.

## 2026-09-21

- Plan written before any code; H1-H4 thresholds fixed (PLAN.md). Training code lives in this repo; data generation stays in D:\Tiny Models\curriculum-learning.
- Agent team and AGENTS.md added (.claude/agents/, lead is `lifespan-lead`). Contract shapes in AGENTS.md are drafts until the lead freezes them per wave.
- CPU smoke tests are not "training on the laptop": toy config (<= 1M parameters), synthetic or 20-story fixture data, <= 50 steps, asserting mechanics only. No measurement; without it the pipeline cannot be tested before a GPU session. Any quotable number still comes from Kaggle or the A100.
- Replay is on top of the per-phase token budget, implemented as a larger batch (N new-phase sequences + ~0.43 N replay), same step count and schedule in every arm. No measurement; reasoning: inside the budget Arm B would see 2.8 epochs of new material, lowering its peak and flattering its forgetting number (peak minus final). Cost lands in GPU-hours, which H2 already counts: Arm B ~1.43x Arm A's tokens. Same rule for Arm D's distillation step; Arm E matches Arm A's total.
- Phase 0 is full training for all sequential arms, trained once per seed and shared by A, B, C, D, D-nr; its wall clock is copied into each arm's timings. No measurement; reasoning: the consolidation recipe taken literally at k=0 trains a rank-16 LoRA on a frozen random base (embeddings included), which cannot learn, so Arm D's first teacher would be noise. PLAN.md already had this for Arm C only.
- Story generation moves off Gemini, probably to Claude (owner, 2026-09-21). Provider and model id not final; to be recorded here, with the re-costed ~23M output tokens, before the first paid batch. PLAN.md's Data section still says Gemini Flash until then.

## 2026-09-21 (later) — contracts frozen for waves 2 and 3

Freezing the drafts in `AGENTS.md` before dispatching `trainer-core`,
`evaluator` and `consolidation`, so two agents cannot guess differently. None of
these edits the H1-H4 table; each resolves something `PLAN.md` left open, and
each is made before any run, not after a matrix came back.

- **Headline exam type is continuation-choice accuracy; cloze is reported beside
  it at every step; perplexity is never a verdict metric.** No measurement;
  reasoning: H1-H4 are stated in "accuracy" and in percentages of a score, and
  perplexity is a loss, so "loses >= 15% of its peak" is not defined on it.
  Continuation choice is the probe that tests what the phases actually differ in
  (reasoning level and tone) and has a known chance floor. Cloze is kept as a
  second, independent read rather than a tiebreaker: where the two disagree the
  result is reported as split. Choosing one after seeing the matrix would be the
  exact failure `AGENTS.md` warns about.
- **Continuation options are scored length-normalised (mean per-token
  log-likelihood given the prefix); the summed score is stored too.** No
  measurement; reasoning: the four options are paragraphs of different lengths
  and the summed log-likelihood is dominated by length, which would make the
  probe a length detector. Both are stored so the choice can be checked later.
- **Cloze candidate sets are 20 surface strings: the true answer plus 19 drawn
  from the same phase's lexicon with the exam seed.** No measurement; reasoning:
  a whole-lexicon candidate set makes the exam cost scale with lexicon size and
  differ per phase, which would make phases incomparable. 20 fixes a 5% chance
  floor identical in every phase. Cost at 20: ~490k forward passes per grid run,
  minutes on an A100.
- **`[MASK]` is the literal mask substring, replaced by `str.replace`, and the
  whole filled text is scored** (PLAN.md's Evaluation already requires scoring
  the filled sequence, not the first sub-token).
- **`--pilot` changes sizes only** (1,000 stories and 100 exam items per phase);
  it never changes a hyperparameter, and pilot results are never mixed with grid
  results in `report.py`.
- **`training/hooks.py` and `training/config.py` are lead-owned**, written now
  rather than by whichever agent needed them first: they are the contract as
  code, and `PhaseContext`, `ARMS`, `EXAM_TYPES` and `TOY` are held by three
  agents at once.
- **Model vocab is 8,192**, the bottom of PLAN.md's 8k-16k band. Measurement:
  the parameter arithmetic - 8L/512/8H/1024ctx gives 25,165,824 non-embedding
  parameters, and vocab 8,192 adds 4,718,592 for a total of 29,884,416, which is
  the "~30M" PLAN.md specifies; a 16k vocab would make it 34.1M and put a
  seventh of the model in the embedding table.
- **Dev environment**: `.venv` with CPU-only torch, numpy, tokenizers, pytest,
  matplotlib, pinned by floor in `requirements.txt` and `pyproject.toml` so
  Kaggle's image satisfies them without a reinstall. CPU torch on the laptop is
  deliberate: it makes a smoke test possible and training impossible.
