# Decisions

Append-only. Date, the measurement, the decision it drove.

## 2026-09-21

- Plan written before any code; H1-H4 thresholds fixed (PLAN.md). Training code lives in this repo; data generation stays in D:\Lifespan\curriculum-learning.
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

## 2026-09-22 — generation provider, and the wave 2/3 review

- **Story generation is Claude Haiku 4.5, model id `claude-haiku-4-5`** (owner,
  2026-09-22), replacing PLAN.md's Gemini Flash and closing the 2026-09-21 entry.
  The id carries no date suffix; a suffixed variant is not a valid model string.
  One model generates every story, train and exam, all seven phases.
  Re-costed at the current published rates ($1.00 input / $5.00 output per 1M
  tokens, halved by the Batch API): 38,500 stories at ~600 output tokens is
  ~23.1M output tokens, so ~$116 output + ~$23 input at standard rates, or
  **~$58 + ~$12 = ~$70 through the Batch API**. That is PLAN.md's "tens of
  dollars" and the batch path is the default. Haiku 4.5 takes no `effort`
  parameter and needs no thinking for this task.
- **Arms C, D and D-nr were getting the base phase loop AND the consolidation
  hook.** Measurement: `uses_lora` appeared nowhere in `train.py`; the phase loop
  called `train_one_phase` for every arm and then `after_phase`, which trains a
  LoRA on the same phase. Consequences had it run: arm C's base fine-tuned every
  phase, so it was not "base frozen after phase 0" and H3 would have measured
  nothing; and C/D/D-nr at roughly double A's token budget, so H2's compute ratio
  and every accuracy comparison would have been invalid. Nothing crashed. Decision:
  for an arm with `uses_lora`, `train.py` does not run its ordinary base phase
  loop — the hook owns the phase, because the day (training the LoRA on phase k)
  happens inside it. Found by the consolidation agent as a question about the
  contract, not by a test; an arm-parity test now covers it.
- **`after_phase` does not run at phase 0.** The 2026-09-21 contract said it did.
  That was wrong: phase 0 is ordinary full training shared by A, B, C, D and D-nr,
  and PLAN.md gives C and D "one LoRA per phase 1-6".
- **`REAL` is 29,901,824 parameters, not 29,884,416.** Measurement: the first
  count omitted the affine LayerNorm vectors ((2*n_layer + 1) * 2 * n_embd =
  17,408). `trainer-core` had correctly deduced a parameter-free LayerNorm from
  the wrong number; the fix is to the arithmetic, not to the model. LayerNorm
  stays affine, linears stay bias-free, the head stays tied.
- **Warmup scales only at `--pilot` and `--toy` sizes** (`config.warmup_for`).
  Measurement: a grid phase is 367 steps and a pilot phase 74, so at PLAN.md's
  fixed 200-step warmup a pilot run never leaves the linear ramp and peaks at
  ~1.1e-4 instead of 3e-4. Warmup is counted in steps and steps is a size, so
  this is inside what `--pilot` may change. A grid run's warmup is exactly 200.
  Noted and not acted on: 200 is 54% of a 367-step grid phase. PLAN.md fixes it;
  changing it is a decision to make before the grid, never after a matrix.
- **A run whose `commit.txt` dirty flag is missing or unparseable is excluded
  from a report, not assumed clean.** The evaluator's first pass defaulted to
  clean, which is the one default that lets a dirty tree become a result.
  `commit.txt` is now one `key: value` per line with a mandatory `dirty`.
- **The Arm B / Arm A token ratio is 1.375, not PLAN.md's "about 1.43x"**,
  because phase 0 carries no replay and so 6 of 7 phases carry it. Measured from
  the budget: 12,025,856 new-phase tokens per phase, 5,261,312 replay tokens on
  each of phases 1-6. `config.json` records the computed ratio so the H2 reader
  does not re-derive it.
- **Kaggle notebook is built now and run on the owner's go** (owner, 2026-09-22).
  No GPU quota is spent until then; the resume round trip is proven on the toy
  config first.

## 2026-09-22 (later) — H2 split into H2a and H2b, before any run

Measurement: the consolidation agent's re-measured cost on the toy config (one
full phase per arm, 30 steps x 4 sequences, CPU, best of 5) — arm A 0.262s,
arm B 0.308s (1.18x A), arm C 0.346s (1.32x A), arm D 0.773s (2.96x A, 2.51x B),
arm D-nr 0.755s (2.89x A). Decomposed against PLAN.md's units: step 1 = 1.21
units, step 3 with replay = 1.68, step 3 without = 1.10. Toy CPU wall clock is
not A100 economics — at N=4 arm B's replay costs 18% more time rather than the
~43% more tokens it will cost on GPU, which flatters the denominator — but the
shape of the algorithm does not change with scale: a full 1-unit day plus a
night of two frozen teacher forwards and a student step. PLAN.md's own *Budget*
paragraph independently estimates arm D at 2.5-3x arm A, and the measured
arm B / arm A token ratio is 1.375, so arm D is 1.8-2.2x arm B by construction.

Decision (owner, 2026-09-22): H2's confirm clause of "<= 1.5x Arm B's GPU time"
was unreachable by any implementation of the recipe in PLAN.md, so the
hypothesis was split — H2a keeps the retention claim and its thresholds exactly
as frozen; H2b reports the measured GPU-time ratio against the original <= 1.5x
and > 2x bands rather than as pass/fail. No threshold moved. This was done
before the first training run, with no matrix in hand; the same change made
after a result would be the failure AGENTS.md exists to prevent, and the reason
it is recorded here with the arithmetic that forced it is so a reader can check
that ordering. PLAN.md already anticipated the outcome: "If H2 holds only at the
3x cost, the honest conclusion is that consolidation buys retention for compute,
and the product question becomes whether that trade is worth it."

Resolved the same day (owner, 2026-09-22): the 20-story smoke batch runs
first and is inspected before any decision about generating a full phase. The
batch is spread across all seven phases rather than drawn from one, because the
failure it exists to catch is a generator that ignores the age in the prompt,
and that is only visible as a contrast between a junior-kindergarten story and a
grade-12 one. It also measures the real output-token count per story, which the
~$70 estimate for the full run assumes to be ~600. The full run is not approved,
and no GPU session is approved.

Blocked as of 2026-09-22: no ANTHROPIC_API_KEY exists in the shell, the Windows
User scope or the Machine scope, and the `ant` CLI is not installed, so the
smoke batch cannot call the API yet. The generator, its dry-run cost estimate
and its tests are built so the batch is a single command once a credential
exists.

## 2026-09-22 — the data side, and two bugs that would not have announced themselves

- **Lexicons for phases 4, 5 and 6 were near-duplicates of each other** (91-100%
  word overlap; phase 5 was a literal subset of phase 4). PLAN.md's checklist
  said these files were *missing*; they existed and were worse than missing,
  because nothing would have failed. The phases would have shared vocabulary,
  the exams would not have discriminated between them, and H1 would have been
  killed by a curriculum bug rather than by a fact about forgetting.
  Rebuilt from public-domain texts matched to each phase's `dominant_grades`
  (Tom Sawyer, Pride and Prejudice, Moby-Dick, Mill's On Liberty), lemma seen
  >= 2 times in source, proper nouns excluded, ~1,100 words each.
  Verified by the lead independently of the agent's report: no lexicon is a
  subset of any other, and pairwise overlap is now 25-53% across the rebuilt
  phases (0-6 at 25.1%, the most distant pair).
  Open and NOT fixed: phases 0, 1 and 2 still overlap each other 70-76%. They
  were not rebuilt. For adjacent early-reading phases that is arguably correct —
  a junior-kindergarten and a senior-kindergarten vocabulary really do overlap —
  but it means less measurable forgetting at the early end, and H1 is read on
  phase 0. The pilot's H1 gate is the designed catch for this: if Arm A does not
  forget, the curriculum goes back before any more is spent.
- **`names.py::get_bio()` used the unseeded global `random.choice` instead of
  `self.rng.choice`.** Found by a determinism test, not by inspection. Every
  rule in CLAUDE.md about seeds controlling data order rests on the generator
  being seeded; this silently broke it, and the damage would have been prompts
  that differ run to run while the manifest claimed a seed.
- **Full-run cost re-estimated: $121.77 standard, $60.89 through the Batch API**
  (38,498 prompts, ~6.28M input + 23.1M output tokens), against the ~$70 in the
  entry above. The difference is the input side: real prompts average ~163
  tokens, not the ~600 that estimate assumed symmetrically with output.
  Caveat on the number: it comes from a chars/4 heuristic, because
  `messages.count_tokens` needs a credential and none exists yet. Confirm it
  against the real tokenizer before approving the full run.
- **`response/scratch.py` was left in place, not replaced.** The brief said to
  replace it; the standing rule says never touch the owner's uncommitted work,
  and it is staged. The agent followed the standing rule and built the
  replacement (`response/providers.py`, `response/generate_responses.py`)
  alongside it, which was the right call. The owner removes `scratch.py`
  themselves, and with it the GCP project id hardcoded in its source.
- Flagged, not fixed, not ours: mojibake in some prompt template strings
  (`children\u00e2\u20ac\u2122s` for `children's`), inconsistent across
  templates, probably a stray encoding in a content_type or tone yaml. It would
  end up in the training text.

## 2026-09-22 — leakage audit: BLOCK on two findings

Audit 1 of 4 (read-only, ran the suite at 259 passed, opened no exam text).
Areas 1, 2, 3, 5 and 6 passed, several of them against constructed attacks:
the guard is the first statement of `run()` and refused a missing, empty,
unparseable and sha-less manifest, train==exam, either directory nested in the
other, relative-vs-absolute and case-variant spellings, a read-locked file, and
a directory junction from the training tree into the exam tree; the tokenizer's
corpus glob returned only real training files even through that junction, which
is the leak that would otherwise never have been found, because a BPE trained on
exam text fails no test.

- **BLOCKING — generation-model provenance is recorded nowhere.** The generator's
  consistency check compares a model id only against lines already in the same
  output file, and each phase is its own file, so a change between phases is
  caught by nothing; its refusal message even suggested routing around it with a
  new `--out`. On the training side, `config.json`'s `"model"` key is the
  architecture config and there is no field for the generator at all; `data.py`
  and `tokenizer.py` read only `prompt_hash` and `story`. A generator change
  between phase 3 and phase 4 would produce a plausible forgetting curve with no
  trace in any artefact. Fixed in two independent places, deliberately: the
  generator checks the whole corpus directory, and `config.json` carries a
  `generator_model` read from the training lines and asserted single-valued, so
  a two-model corpus does not train.
- **BLOCKING as a gate on exam-keeper — continuation-probe provenance cannot be
  verified after the freeze.** `distractor_phases` is written by the contract and
  read by nothing, and it is not recoverable later: a manifest entry hashes a
  whole exam file, while an option is a paragraph excerpted from one, so an
  option's sha256 will never match a manifest hash. A distractor accidentally
  drawn from a training story would be permanently undetectable and would inflate
  the headline metric on exactly the phases the arms are supposed to have
  forgotten. The probe line gains a required `option_sources` field recording each
  option's story id, sha256 and phase; exam-keeper asserts membership against the
  manifest at freeze time, and `evaluate.py` asserts it at every load, because
  after the freeze that assertion is the only check left.
- Two asymmetries with cloze, which is validated: `answer_index` was not
  bounds-checked, and `CONTINUATION_CHANCE` was hardcoded 1/4 regardless of
  `len(options)`, so a 5-option item would have silently reported chance 0.25.
- `.gitignore` now carries `exams/**` with `!exams/manifest.json`, so git refuses
  exam text rather than the repo merely discouraging it. Exams live outside the
  repo by design; this makes the wrong `git add` impossible, not just unlikely.
- Ruled, on the divergence the evaluator and trainer-core disagreed on:
  `continuation_summed` is REQUIRED. The evaluator kept the strict side and was
  right; `OPTIONAL_MATRIX_KEYS` goes.

## 2026-09-22 — arm-parity and mechanism audit: BLOCK on two findings

Audit 2 of 4, read-only, independent of the agents' own tests. It re-derived the
things worth re-deriving rather than reading them: arm C's base hashed at every
hook entry and exit (bit-identical across phases 1-6, 0 optimiser steps outside
the hook and 300 inside, with the converse measured for arm A so the check
cannot pass by the loop being broken for every arm); arms A and B byte-identical
new-phase batches at all seven phases; resume == uninterrupted for all six arms
including full weight hashes, which nothing in the repo had covered for C and D;
replay provenance checked by batch *content*, not by configuration; the KL worked
by hand (p_T=(0.8,0.2), p_S=(0.5,0.5) -> 0.1927447570 expected, 0.1927447617
returned, reverse KL not returned); and the parameter count checked as a split
and a LayerNorm tensor count so 29,901,824 cannot be two cancelling mistakes.

- **BLOCKING — no run could complete, and the suite was green anyway.**
  `evaluate_all` returns `as_hook_dict()`, three keys; `runrecord._check_scores`
  demands four after the `continuation_summed` ruling; `train.py` wires exactly
  that three-key callable. Every arm would have died at the first
  `set_untrained`. The 282-test suite passed because every `test_train.py` case
  injects a four-key fake and **no test wires the real `evaluate_all` into
  `train.run`**. The bug is mine — the ruling landed with nothing behind it — but
  the hole is structural: a seam that every test stubs is a seam nothing tests.
  Fixed by calling `evaluate_all_detailed(...).as_matrix_row()`, and by an
  end-to-end toy run with no stub for `evaluate_all` or `after_phase`, against a
  synthetic exam directory in the real layout with the full probe schema.
- **BLOCKING — `tokens_vs_arm_a` is wrong for the LoRA arms, and H2b reads it.**
  `data.py::token_budget` derives the ratio from `replay_fraction` alone and
  ignores that a `uses_lora` arm makes two passes over phase-k material, step 1
  training the LoRA and step 3 distilling. Measured per-phase new-phase tokens at
  toy scale: 12,800 for A/B/C/E, 25,600 for D and D-nr. `config.json` recorded
  1.375 for D and 1.0 for D-nr against actual spends near 2.375x and 2.0x, and
  `report.recorded_token_ratio` reads that field with a docstring promising it is
  never recomputed. H2b is the compute claim; it would have reported a number
  wrong by roughly the factor the hypothesis is about. Now derived from the arm's
  actual passes and from the per-phase budget rows, since the closed form is
  exact only when all seven phases have equal step counts, which real data will
  not give.
- `report.load_run` float-casts every matrix cell, so a session that died
  mid-run - the expected Kaggle outcome - took the whole report down with a
  TypeError instead of being named as an exclusion. Fixed: a null-bearing matrix
  is an exclusion with a reason naming the incomplete rows.
- Noted, not defects: arms D and D-nr legitimately spend 2x the new-phase tokens
  per phase, because PLAN.md's step 3 is a second pass over phase-k material -
  the *day* budget is equal across all six arms, the total is not, and H2b is
  where that shows up. `PILOT_OVERRIDES["exam_items_per_phase"]` is never read,
  so `--pilot` does not cut the exam set (quota, not correctness).
  `apply_lora(seed=ctx.seed)` is not phase-dependent, so the same LoRA A is drawn
  at every phase and across C, D and D-nr - deterministic, undocumented.
  `consolidate.py`'s `AFTER_PHASE` dict is a second arm table keyed by name that
  `train.py` never reaches: dead, and free to drift.
  H2a can confirm while Arm D forgets more than Arm B if the gap is inside the
  seed spread. That is the seed-spread rule working as written; the report must
  say so on the line rather than leave a reader to reconcile it.

Both audits must be re-run once these land: audit 2 read a tree that was moving
under it, and a clean verdict on a moving tree is not a verdict.

## 2026-09-23 — curriculum-learning moved beside this repo

- **`D:\Tiny Models\curriculum-learning` is now `D:\Lifespan\curriculum-learning`**
  (owner, 2026-09-23). Whole repo moved with its git history, uncommitted work,
  smoke-batch outputs and `.venv`; the split of responsibilities is unchanged
  (data generation there, training here). Its venv's editable-install pointer,
  `pyvenv.cfg` and activation scripts were repointed; the `.exe` shims in its
  `Scripts` folder embed the old path, so use `python -m pip` / `python -m pytest`
  from that venv or recreate it. The old folder could not be deleted at move time
  because a process held its root open; it is an empty husk to remove by hand.
  Every path reference in `CLAUDE.md`, `AGENTS.md`, `docs/RESOURCES.md` and the
  agent files was updated.

## 2026-09-23 — dev environment is one micromamba env, `lifespan`

- **Both repos now run from a single micromamba env named `lifespan`**
  (owner, 2026-09-23), replacing the two per-repo `.venv`s. Python 3.13,
  CPU-only torch 2.14 from the PyTorch CPU index, then each repo's
  `requirements.txt` and an editable install of both packages. Recreate with
  `micromamba create -f environment.yml` from the Lifespan repo; run with
  `micromamba run -n lifespan python -m ...`. Measurement: Lifespan's 295 tests
  and curriculum-learning's 22 pass in the new env. Reason: the moved venv's
  `.exe` shims were dead, and the machine's other projects already live in
  `D:\Micromamba`. Both `.venv` folders deleted. No dependency was added.

## 2026-09-23 — prompt template chosen by A/B measurement; lexicons found unusable for phases 3-6

Harness: `curriculum-learning/tools/prompt_lab` (README there). Same seeded
configurations rendered through the old template and each candidate, stories
generated with Haiku 4.5 (the production generator), scored by text metrics
and a blind Opus 5 judge that first guesses the grade band from the text alone
and only then is told the target. Two rounds, seeds 1 and 2, 3 prompts per
phase, 21 stories per variant per round; small n, so only large effects count.

- **Template is now `engine/story_prompt.py` (candidate "v2"), one template for
  all three content types.** Round 2, old vs v2: title/heading on the story
  20/21 -> 0/21; judge band-within-one 16/21 -> 19/21, band-exact 8/21 ->
  11/21; reading-level fit 3.05 -> 3.81 (of 5); educational value 2.10 -> 2.67;
  naturalness 2.38 -> 3.33; "lecture-like" 16/21 -> 5/21; coherence 3.76 ->
  3.71 (within noise). Mean output 669 -> 525 tokens per story, so the corpus
  gets cheaper. What changed: reply rules first (no title, no markdown, end on
  an action); a per-phase "thinking move" written from phases.yaml's
  developmental descriptions; every story turns on one real-world idea about
  a sampled knowledge domain; mentor speeches and narrator explanations
  banned; a reader-assumption line and a scene rule for older phases. The
  ablation without the thinking move ("v3") lost the phase-6 separation
  (judge guessed band 4 for all three phase-6 stories vs 4,5,5 with it).
- **Facts come from a verified bank, not from the generator.** The judge found
  about a third of the facts Haiku invented for v2 wrong or muddled (hot tap on
  the right, a base-rate sum that did not add up, wrong genetics). So
  `engine/facts.py` samples a fact from `config/facts/phase_{id}.json` (built
  by subagents: one per phase writes ~15 facts per domain, an independent one
  fact-checks them blind, `tools/prompt_lab/merge_builds.py` keeps only
  "true") matched to the story's
  activity, and the prompt says to keep it accurate as stated. Without a bank
  the prompt falls back to naming the domain only. The fact and domain are
  recorded in every prompt's metadata, which makes a per-phase fact-recall
  exam possible later. Not yet measured against v2 (the bank was being built
  when the API credit balance ran out; measure with `gen_ab.py --variants
  v2,v4` before generating a phase).
- **Title stripping is a deterministic backstop** in `generate_responses.py`
  (`clean_story`): the old template produced a title on 41 of 42 stories
  despite "Only use plain text"; the new one produces none, and a title is a
  phase-uninformative artefact that would land in training and exam text alike.
- **The phase 3-6 lexicons are not usable and phases 0-2 need cleaning**
  (`tools/prompt_lab/lex_audit/report.txt`, Haiku 4.5 rating 8,000 words for
  part of speech, fitness and earliest band). Of 400 words per part of speech,
  the number that are the stated part of speech, fit for the corpus and within
  one band of the phase: phase 5 verbs 3, nouns 2, adjectives 14; phase 6
  verbs 3, nouns 1, adjectives 3; phase 4 verbs 70, nouns 53, adjectives 99.
  The 2026-09-22 rebuild from 19th-century novels achieved low overlap between
  phases by drawing on different books, but the words themselves are 'say',
  'go', 'time' plus POS errors ('lightning' as a verb, 'cried' as an
  adjective) and archaic or offensive entries ('thou', 'negro'). Phase 0 has
  143 verbs, 265 nouns and 113 adjectives two or more bands above JK ('wealthy',
  'missile'). Replacement: one subagent per phase writes the words a reader first
  learns in that band across seven topic areas; `merge_builds.py lexicons`
  deduplicates so a word belongs to its earliest phase. Install over
  `config/lexicons/` before the exam freeze; this changes every prompt hash.
- Also fixed: curly apostrophes in the templates and `config_loader` reading
  YAML without `encoding="utf-8"` (the source of the mojibake flagged
  2026-09-22); `ASSUMED_OUTPUT_TOKENS_PER_STORY` 663 -> 550 from measurement.
- Spend: ~$7 of API credit on 2026-09-23 before the balance ran out, mostly
  the Opus 5 judge over 126 stories and the first (Opus) fact and lexicon
  builds; owner refilled $20 and ruled that LLM labor (judging, fact and lexicon
  building, verification) runs as Claude Code subagents on the Max plan, not
  through the API; API credit is for the production Haiku generator only.

## 2026-09-23 (later) — fact bank and lexicons installed; round 3; two bugs

- **Fact bank and graded lexicons are in `config/facts/` and `config/lexicons/`**,
  built by subagents (one writer and one independent fact-checker per phase):
  682 of 720 facts kept (verdicts per phase in `tools/prompt_lab/facts/_verify`);
  376-620 words per phase, at most 7 shared between any two phases.
- **Round 3 (seed 3, 21 stories per variant, subagent judges):** the repo
  template with the injected fact vs the same template naming only a domain.
  Blind band guess exact 17/21 vs 12/21, within one band 21/21 vs 20/21;
  educational value 3.14 vs 3.05; coherence 3.38 vs 3.24; naturalness 2.71 vs
  2.67; lecture-like 3 vs 0; one "#" title slipped through (the pipeline's
  `clean_story` removes it). The fact-injected template stays. Two judges
  (one per variant), so cross-variant differences carry judge variance.
- Judges' recurring complaints, both variants: the harder graded-lexicon words
  are sometimes misused ("the mesmerize quality", "excavate the problem"), and
  Haiku occasionally garbles an injected number or direction (a Titanic
  lifeboat count, "shorter in the morning"). Template now says to keep every
  number, direction and comparison exactly as given, and to leave a target
  word out rather than misuse it. Not yet re-measured.
- **Bug found in the old generator: `AdvancedLearningArc` had no `build_prompt`,
  so every advanced_learning prompt (10% weight in tiers 2-13) rendered the
  base-class placeholder ("Generate a <Tone object> story about ...").** No
  test covered it and nothing crashed. The shared template fixes it because
  `Arc.build_prompt` now serves both arc kinds; `test_every_prompt_uses_the_shared_template`
  covers it.
- **Harness bug (rounds 1-3):** `gen_ab.py` hooked only `BasicLearningArc`, so
  within a phase every row after an advanced_learning draw had its metadata
  (age, grade, target words) shifted by one; phase ids were right, so the
  band-guess numbers stand, but per-story age fit for those rows was judged
  against a neighbour's age. Fixed by hooking `Arc`.

## 2026-09-23 (evening) — template frozen after rounds 4 and 5

Both rounds: 14 stories (2 per phase), the repo template with injected facts
and the new lexicons, one subagent judge each, blind band guess first.

| Round | Change measured | Band exact | Level fit | Edu value | Coherence | Natural | Words misused | False statements |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3 (n=21) | facts injected | 17/21 | 3.86 | 3.14 | 3.38 | 2.71 | many | 2 |
| 4 | exact-number rule, correct-or-omit words | 14/14 | 4.57 | 3.64 | 3.36 | 2.64 | 8/14 | 3 |
| 5 | no explaining beyond the fact; words only where they fit | 14/14 | 4.50 | 3.64 | 3.07 | 3.07 | 4/14 | 2 |

Decision: the round-5 wording is the template for the corpus. The blind band
guess is now exact for every story in two consecutive rounds (the old
template managed 8/21), which is the property H1-H4 depend on. Open and
accepted: Haiku still adds a wrong mechanism or number of its own in roughly
2 of 14 stories (the injected facts themselves were correct in every flagged
case), and about a quarter of stories misuse one of the harder lexicon
words. Both are generator limits, not template limits; the age-check gate
and, if wanted later, a fact-consistency pass over the corpus are the place
to catch them. Output length is ~545-590 tokens per story, so the corpus
cost estimate stands.

Rule applied throughout: story generation on the API (rounds 3-5 cost
$0.24 in total), judging and data building by subagents on the Max plan.

## 2026-09-23 (night) — generator: GLM 5.3 (Baidu, fp8) replaces Haiku 4.5

Measurement: 42 identical prompts per generator (seeds 6 and 7, 3 per phase),
stories shuffled under opaque ids and split between two blind subagent
judges so neither judge knew which generator wrote what
(`tools/prompt_lab/mix_blind.py`, runs r7*). Same rubric as rounds 1-6.

| Generator | Band exact | Within 1 | Level fit | Edu value | Coherence | Natural | Lecture-like | Words misused | False statements | Tokens out/story |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Haiku 4.5 | 30/42 | 42/42 | 4.10 | 3.07 | 3.36 | 2.90 | 9 | 15 | 12 | 566 |
| GLM 5.3, Baidu fp8, low reasoning | 27/42 | 41/42 | 4.52 | 3.69 | 3.98 | 4.00 | 1 | 3 | 7 | 704 (incl. ~80 reasoning) |

GLM is better on every quality measure except a three-story deficit in
exact band placement (both are at or near perfect within one band). The
gap is largest where it matters most, phases 3-6: naturalness 4.42 vs 3.00,
coherence 4.25 vs 3.33, misused words 3 vs 13. A 14-story pilot of Sonnet 5
(run r6_sonnet) scored like GLM on coherence and naturalness but not on
false statements (3/14) at three times GLM's cost, so it was not carried
into the head-to-head.

Cost at the standard OpenRouter Baidu endpoint ($0.56 in / $1.76 out per
MTok, no batch discount available): ~$62 for the 38,500-story corpus
versus ~$66 for Haiku through the Anthropic Batch API. The `:batch`
listing at $0.45/$2.00 is an fp4 quantization and is not used.

Decisions (owner, 2026-09-23):
- The corpus generator is `z-ai/glm-5.3` through OpenRouter, upstream
  pinned to Baidu (fp8), fallbacks disabled, reasoning effort low and
  excluded from the response. Reasoning cannot be disabled for this model on
  any of the eight OpenRouter providers probed; low effort costs ~80 output
  tokens a story. The recorded model string in every story line carries the
  upstream and quantization so a routing change cannot pass unnoticed.
- False statements are a generator-independent ~15-30% (they are the
  model's own added mechanisms, arithmetic and analogies, never the injected
  fact), so the corpus gets a fact-consistency gate run by subagents on the
  Max plan (`tools/prompt_lab/fact_check_export.py`): every story is checked
  against its fact, failures are queued for regeneration. This gate runs
  before the exam freeze, on train and exam stories alike.
- The 2026-09-22 Haiku decision is superseded; `generate_responses.py`'s
  default model changes with the OpenRouter provider (next entry).

## 2026-09-23 (night) — OpenRouter provider in the pipeline

- `response/openrouter_provider.py` presents OpenRouter's chat endpoint to
  `generate_responses.py` as a batch provider: requests run on a thread
  pool, every finished story lands in `<out>.openrouter_cache.jsonl` as it
  arrives, and batch membership is persisted there too, so a killed run
  resumes without re-requesting anything paid for. Upstream is pinned
  (`provider.order=[Baidu]`, fallbacks off); a story served by any other
  upstream, a truncated story, or an empty one is a retryable failure.
  Reasoning effort low, excluded from the response, 1,500 tokens of
  max_tokens headroom for it. The model string recorded on every story line
  and checked by the one-model rule is
  `z-ai/glm-5.3@openrouter/Baidu/fp8/reasoning=low`.
- `generate_responses.py --provider openrouter` is the default;
  `--provider anthropic` keeps the Claude Message Batches path with
  `claude-haiku-4-5`. Dry-run prices per provider; assumed output 700
  tokens a story for GLM (measured 690-720), 550 for Haiku.
- Measured on the real prompt set (34,996 train prompts, seed 42): dry-run
  estimate $58 for train on GLM/Baidu, so ~$64 with the 3,500 exam stories;
  Haiku through the Batch API would be $61 + $6. Six new provider tests, 41
  passing in the repo.
- `OPEN_ROUTER_API_KEY` joins `ANTHROPIC_API_KEY` in the environment rule
  in CLAUDE.md; `.env` is git-ignored and was checked to be.

## 2026-09-23 (night) — generator configuration rebuilt

Measurement (`curriculum-learning/tools/config_audit.py`, run before the
rewrite): at grades 9-12 the only basic-learning arc was "learning to code"
and it carried 30-50% of the draw; ten tiers shared one placeholder weight
row; all 20 arc goals were "more complex X" placeholders; exposures for
grades 3-12 numbered three per tier; tones were ten early-childhood
registers with no tier gating; a feature asked for a stated moral that the
template forbids; locations included "at the kitchen" and "online". 66
problems in all.

Decision (owner: "a lot of config data is placeholder or incomplete"):
- Schema: an arc's `basic_learning` / `advanced_learning` is now a LIST of
  grade-band levels (`{min_tier, max_tier, goal}`), so one subject carries a
  concrete goal per band; tones take `min_tier` / `max_tier`; the old
  nested shape still loads. Tier weights set per band (exposure-heavy in
  kindergarten, learning-heavy from grade 3, advanced share rising to 25%
  by grade 11-12); age ranges tightened to two years around the grade norm.
- Content, authored by four subagents against explicit coverage rules:
  24 arcs / 149 basic and 135 advanced levels (math, reading, writing,
  science, physics, chemistry, biology, earth and space, history, geography,
  civics, economics and money, second language, music, art, drama and
  speaking, sports, health, cooking, coding, technology tools, practical
  life, religious and cultural practice, social skills); 48 experiences;
  65 exposures; 18 tones (wry, suspenseful, reflective, tense, bittersweet,
  matter-of-fact, earnest, hopeful added for older tiers); 21 gated craft
  features with the moral one removed.
- After: every tier has 12-19 exposures, 15-33 experiences, 13-23 basic and
  0-23 advanced arcs, at least 12 tones and 7 features; the audit reports 0
  problems and `tests/test_config_coverage.py` (9 tests) enforces the
  minimums, unique non-placeholder goals, prepositional locations,
  contiguous arc bands and a full generator run at every tier.
- Every prompt hash changes with this; nothing had been generated, so
  nothing is invalidated. Fact-domain matching gained a keyword fallback for
  the ~90 new content keys.
