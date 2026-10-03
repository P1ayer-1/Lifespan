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

## 2026-09-25 — pre-pilot approved: phases 0, 3 and 6, arms A and E

Measurement: tier-0 stories average 171 words (~230 tokens) against the ~600
PLAN.md assumed, so a 1,000-story pilot phase 0 is under 1M token-passes.
GLM story length rises steeply with tier (prompt_lab runs: 474 words at
phase 3, 842 at phase 6), so 1,000 stories per phase is ~0.23M tokens at
phase 0 and ~1.1M at phase 6.

Decisions (owner, 2026-09-25):
- A pre-pilot runs before PLAN.md's week-2 pilot. It is a sanity check and
  never an H1 verdict: does a 30M model learn phase 0 at pilot data size, and
  do the exams separate phases? Phases 0, 3, 6; arms phase0 -> A, and E;
  1,000 training stories per phase (phase 0 = the 995 accepted tier-0
  stories); 100 exam stories per phase; Kaggle free quota on the owner's go.
- Its exams are a throwaway experiment: own experiment_id, seed 4243, own
  manifest, never reused by the pilot or the grid, so the real exam seed and
  freeze stay unseen. Phase-0 exams are tier 0 only, because the phase-0
  training corpus is tier 0 only (phase 0 spans tiers 0-1).
- Generation cap $10 on OpenRouter (raised from $4 after the lead's estimate
  of ~$5 at list rates for 2,200 phase 3/6 stories, 300 exam stories and
  regeneration). Stop at the cap.
- Quality gate: every exam story through both Opus reviewers; a seeded
  200-story sample of the phase 3 and phase 6 training stories through both;
  failures regenerated as for tier 0 (separate --out per round). The review
  wave runs from a session started in D:\Lifespan, where the reviewers live.
- Audit re-runs 1 and 2 launched from the D:\Lifespan session on 2026-09-25
  did not finish (that session exited while they ran; they were
  general-purpose agents acting from lifespan-auditor.md, not the auditor
  itself). No findings exist from them. Both re-run with the real
  lifespan-auditor from this repo's session, same day.
- **Phases 3 and 6 get per-activity fact banks before any story is
  generated** (owner, 2026-09-25). Measurement (curriculum-learning ce15c91,
  pre-pilot train prompts, seed 42): an injected verified fact reaches 17 of
  1,000 phase-3 prompts and 59 of 1,000 phase-6 prompts, against 1,000 of
  1,000 at phase 0, because the phase-level bank matches only on a >= 3-term
  overlap. The rest fall back to domain-only, the mode measured on
  2026-09-23 at ~15-30% false statements. Generating now would have mixed two
  recipes across phases and measured a recipe the pilot would not keep. Same
  build as phase 0 (commit ad97bc6): per-activity lists, Opus writer
  subagents at 12 facts per activity, independent Opus reviewers, only
  "true" ships; spec adapted to grades 6-8 and grade 12. Costs no API credit.
- Dry-run cost for the pre-pilot at measured story lengths (tier 0's 1.46
  output tokens per word): phase 3 $1.59, phase 6 $2.57, 300 exam stories
  $0.51, one regeneration round ~$0.17; ~$4.84 at list rates. The
  generator's flat 700-token assumption undercounts phase 6 by ~43%.
- Phase 3 and 6 generation runs pass `--corpus-registry` pointing at
  `data/tier0/_generation_model.json`. Without it each new corpus directory
  starts its own registry and the one-model rule never compares against
  tier 0. (Superseded the same day by audit 1's finding 2: a convention is
  not enough; the tool enforces it.)

## 2026-09-25 — leakage audit re-run: BLOCK on three findings

Audit 1 re-run by `lifespan-auditor` (read-only, synthetic attacks under its
scratchpad, no exam text opened; Lifespan HEAD d7a7a88, curriculum-learning
5f3a7e2). Held from 2026-09-22, each re-attacked: `answer_index` bounds and
chance = 1/len(options) (`evaluate.py:459-466, 572`); `option_sources`
required with phase/story checks (`evaluate.py:468-509`); `generator_model`
in `config.json` (`train.py:477`) with every training line read and a mixed
file refused (`data.py:126-133, 217-246`). Clean: nothing under `exams/` in
git history, no weights or `.env` tracked, tier-0 corpus 995 lines / 995
hashes / one model / LF. curriculum-learning `test_generate_responses.py`
21 passed; the Lifespan suite did not finish inside the audit.

- **BLOCKING — the response generator has no split check and story lines
  carry no `split`.** Attack: exam-split prompts with `--out` at an existing
  train `stories.jsonl` were appended without complaint
  (`generate_responses.py:444-461`). `generate_prompts.py` refuses a split
  mismatch; the story side, which is what training reads, did not. Fix:
  refuse a split different from the output file's or registry's, write
  `split` into every story line, and `data.py` refuses any line whose split
  is not `train`.
- **BLOCKING — the one-model rule is still per directory, not per corpus.**
  A new corpus dir with a different model and no `--corpus-registry` was
  accepted and silently started its own registry;
  `existing_hashes_and_model` keeps only the last line's model, so a file
  of M2 then M1 passed for M1 (`generate_responses.py:216, 252-300`). A stale
  `data/_generation_model.json` still names `claude-haiku-4-5`. And the
  training side never sees the exam stories' model. Fix: the corpus
  registry is required (a new registry only with an explicit new-experiment
  flag), every line's model is collected, the manifest carries the exam
  generator model, and `train.py` refuses when it differs from the training
  corpus's.
- **BLOCKING before any exam freeze — the only leak check is whole-file
  SHA-256, and no near-duplicate check or probe builder exists as code.**
  Synthetic test: an identical copy is refused; the same exam lines with
  CRLF endings, or all exam lines plus one train line, pass
  (`guard.py:137-142`). Fix: the manifest carries per-story `prompt_hash`
  and `story_sha256`; the guard checks every training line against them;
  the near-duplicate check and the probe builder are code with tests before
  the pre-pilot exams are frozen.
- Should-fix: `option_sources[*].story_sha256` is self-attested; a
  distractor whose hash is in no manifest was accepted. Check it against the
  manifest's per-story hashes.
- Should-fix: `N_PHASES = 7` makes `DataModule` refuse a 0/3/6 corpus
  ("missing training file for phase 1"). Safe as it stands; the danger is
  the workaround (renaming 3 -> 1 and 6 -> 2 mislabels the matrix and
  replay). The subset-phase mode is the fix; no renaming, ever.
- Notes: the pre-pilot manifest is passed with `--manifest` and never
  committed as `exams/manifest.json`, which would read as the main
  experiment's freeze; `experiment_id` is recorded but never checked.
  Train and exam stories sharing facts from one per-activity bank is by
  design, but a verbatim fact sentence in an exam rewards memorised
  strings that a 200-character opening check cannot see, so exam-keeper
  reports verbatim n-gram overlap with training per phase, and checks it is
  comparable between tier 0's bank and the phase 3/6 banks.

## 2026-09-29 — phase 3 and 6 activity fact banks verified and merged

- **Phase 3 and 6 per-activity banks are merged: phase 3 keeps 1,384 of
  1,620 facts (66/66 activities), phase 6 keeps 790 of 936 (55/55); no
  activity under 6 facts** (curriculum-learning f3d9015). Verdicts from 22
  independent Opus fact-bank-verifier agents, one per writer part; only
  "true" ships, as for phase 0.
- **First-pass verdicts were not trusted as sourced, and every part was
  re-checked.** Measurement: on re-questioning, verifiers reported that
  most passes naming a person, date, number or quotation had been judged
  from memory, including many reasons labelled "checked:" (e.g. phase 3
  part 1: 109 of 111 passes; phase 6 part 5: all 108). The spec says a
  fact that cannot be confirmed fails, so each part was sent back to fetch
  a source for every such pass. The re-check moved 1,438 -> 1,384 (phase 3)
  and 835 -> 790 (phase 6), 99 facts in all, mostly on rule 2
  (unconfirmed or misworded). Phase 6 part 5 alone fell 108 -> 82.
  Some fails are "unconfirmed with the pages reachable", not "known false":
  the session's 200-call WebSearch limit ran out and later checks used
  WebFetch only, with several government pages blocked (403/404). So the
  banks err toward dropping true facts, which is the safe direction.
  Future verifier runs: require a fetched source in the reason for every
  pass with a specific claim, not a "checked:" label, and budget web
  search across parts.
- The brief's domain-bank path `config/facts/phase_{3,6}.json` does not
  exist in curriculum-learning; verifiers de-duplicated against
  `tools/prompt_lab/facts/phase_{3,6}.json` and
  `src/lifespan_learning/dataset_generation/prompt/config/facts/`. Fix the
  path in the agent brief.
- Kept passes worth a human look: phase 3 part 0 fact 2 (Rogers and
  Farson 1957, sourced only to a bookseller listing); phase 6 part 4 facts
  66 (Ontario rent deposit, landlord-blog source) and 67 (England deposit
  caps, stated before the 1 May 2026 tenancy change). One likely wrongful
  fail: phase 3 part 10 fact 83 ("Hakon" vs "Håkon").
- **Installed and measured (2026-09-30): an injected verified fact now
  reaches 1,000 of 1,000 phase-3 and 1,000 of 1,000 phase-6 pre-pilot train
  prompts, against 17 and 59 before** (curriculum-learning fbfaf69; same
  seed 42, tiers and paths as ce15c91). Every fact comes from the activity
  bank; 877 and 741 distinct facts used, none more than 3 times. The only
  other metadata change is `domain` on the prompts that had drawn a
  domain-level fact. All prompt hashes changed; no phase 3/6 stories or
  exams existed, so nothing downstream is invalidated. Story generation is
  no longer blocked on facts.
- **Facts must match the prompt's goal, not only its activity
  (2026-09-30, curriculum-learning a7e2d46).** Measurement: phases 3 and 6
  split one content_key into several goals (learning_math: fractions,
  congruent triangles), but the merged bank kept only content_key, so
  380/1000 phase-3 and 280/1000 phase-6 prompts got a fact written for a
  sibling goal. The seeded 200-story review (07c981c) traced phase-3
  failures to it: either gate failed on 34.9% of mismatched prompts vs
  17.5% of matched ones (phase 6: 20.3% vs 18.4%). Fix: merge tags each
  fact with its row's activity; FactBank serves a goal only its own facts.
  Tier 0 prompts regenerate byte-identical. 466 + 484 prompts changed;
  their stories were archived and regenerated.
- **Full review of all 2,000 phase 3/6 stories, not the planned 200-story
  sample** (owner, 2026-09-30). Every story through both Opus gates
  (sample, wave 1 26c399a, wave 2 37633c6 / 881280f). Before
  regeneration, pass both: phase 3 811/1000, phase 6 822/1000. Goal-matched
  facts lifted phase-3 age fit (91.8% -> 94.8%) but not fact accuracy
  (~85% both), because fact fails are the generator's own added claims,
  not the injected fact, which was stated correctly almost everywhere.
- **Two regeneration rounds, then drop** (tier 0 rule; 865e660, 719300a).
  Round 1: 250/374 now pass both; round 2: 72/125. Final pre-pilot train
  corpus: **phase 3 972/1000, phase 6 974/1000**, every story passing both
  gates on its latest version; the 54 drops are in train/dropped.jsonl.
  Drops cluster where the generator must get exact technical detail right
  (coding and JUnit behaviour, Canadian/US payroll tax, stoichiometry,
  codon tables, bar-level Mozart analysis). Generation cost about $3 in all.
- Reviewers pass some real problems; seven stories were queued by hand
  (uncorrected unsafe acts, stray CJK/Cyrillic characters from the
  generator, an invented book quote). One (a grease-fire lid lifted again)
  was passed by both reviewers in all three versions and dropped by hand.
  A stray-script scan now runs on every generated batch.
- Open, before the main run: side-character names repeat heavily ("Priya"
  in up to half of a phase-6 part); the template fixes only the main
  character's name. A fix changes every prompt hash, so it belongs with
  the next template change, not mid-pilot.

## 2026-09-30 — leakage audit re-run 3: PASS

The three BLOCKING findings of 2026-09-25 are fixed, merged and re-attacked:
audit-data-split (79b9f87: data.py refuses any line whose split is not
"train"; `--phases` subset mode with real phase ids), audit-guard (per-story
`prompt_hash`/`story_sha256` in the manifest, every training line checked,
manifest `generator_model` must equal the corpus model, option_sources
checked against manifest hashes, `training/neardup.py`, `training/probes.py`)
and audit-wiring (experiment_id required, the scorer tied to the manifest,
tokenizer split check, subset-aware metrics/report/validate, AGENTS.md
amendment 3). curriculum-learning's generator side: 767c6fd, d98ae34.

- Re-run 3 (lifespan-auditor, read-only, synthetic attacks, no exam text;
  Lifespan fe39adf) confirmed all three old findings closed and **blocked on
  one new one**: guard.py split lines with `str.splitlines()` (also breaking
  on U+2028/U+2029/NEL/VT/FF/x1c-x1e) while data.py and tokenizer.py split on
  CRLF/CR/LF only. An exam line carrying U+2028 in any field passed the guard
  and was trained on. Realistic: the generator writes `ensure_ascii=False`.
  Fixed in 7e1ebd3 (`guard.loader_lines`, unparseable .jsonl lines refused,
  neardup uses it); re-attacked with all separators in side fields, stories
  and replay .json, lone-CR and mixed endings: refused. **Verdict at e7620bc:
  PASS.** Suite 472 passed, 0 failed.
- Should-fix from the same run: a 0/3/6 exam could not build continuation
  probes (three distinct other phases required). **Owner decision:** keep 4
  options and chance 1/4; with fewer than three other phases the three
  distractors are spread over them as evenly as possible (2+1), from
  different stories where possible (e7620bc, AGENTS.md). A full 0..6 exam's
  probes are byte-identical; 3,600 subset items over 40 seeds had 0
  violations and verify_probes/the evaluator refuse own-phase, own-story and
  mislabelled distractors.
- Guard cost on the real corpora (pre-pilot, 9 MB): 0.5 s per run; ~5 s at
  grid size. Near-dup check at freeze: 2.3 s for 300 exam vs 2,641 train.
- Open, not blocking: the guard is exact-hash by design, so near-copies are
  left to neardup at exam freeze; neardup misses a story split in halves with
  a new opening, the middle 40% of a story, and zero-width spaces inside
  words; nothing ties the freeze-time near-dup result to stories added
  later (phases 1, 2, 4, 5, regeneration rounds). The main experiment's
  EXPERIMENT_ID defaults to "lifespan-main" (agent's choice, awaiting the
  lead). The Kaggle notebook does not yet pass --manifest, --phases or
  --experiment-id, so it cannot launch the pre-pilot as is.

## 2026-09-30 — pre-pilot exams generated, reviewed and frozen

- **Experiment ids:** the 0/3/6 pre-pilot is `lifespan-prepilot` (its own
  manifest and seed 4243, never committed as exams/manifest.json); the main
  experiment keeps `training.config.EXPERIMENT_ID = "lifespan-main"` (owner,
  2026-09-30: "do all the things"). The Kaggle notebook passes `--manifest`,
  `--phases`, `--experiment-id` when set (499b3ad); the pre-pilot values are
  in its parameter cell.
- **Exams:** 120 exam prompts per phase (split exam, seed 4243, same tiers as
  training: 0 / 7-9 / 13), generated with the training model, every story
  through both Opus gates (reviewers report counts only; no exam text left
  D:\Lifespan\exams_prepilot). Pass both: phase 0 110/120, phase 3 99/120 ->
  114 after one regeneration round of its 20 failures, phase 6 101/120 (one
  excluded for stray CJK text). 100 per phase selected by a seeded shuffle
  (4243+phase), the rest kept as an ordered reserve (unused).
- **Freeze** (exam-keeper, 2026-09-30T23:58Z; manifest sha256 415aa98a...,
  generator commit c936e968): near-duplicate check of all exam candidates
  against all 2,939 training stories flagged 0 (max containment 0.106 / 0.048
  / 0.028 vs threshold 0.5). Probes, seed 4243: cloze 100/99/99 and
  continuation 96/100/100 (phase 0 has 4 single-paragraph stories);
  verify_probes passes; answer slots uniform; masked word never visible;
  length ratio true/mean-distractor 0.94 / 1.12 / 1.13. Replay buffers 100
  per phase (10%, the grid's ratio; unused by phase0/A/E). guard.check and
  evaluate.verify_exam_stories pass on D:\Lifespan\prepilot_train and the
  exam dir; "lifespan-main" and a missing id are refused.
- **Accepted for the pre-pilot only** (a sanity check, never an H1 verdict;
  each is to be revisited before the main exam freeze): (1) a mild length
  giveaway in continuation (longest option right 0.30 / 0.35 at phases 3/6,
  shortest 0.18 at phase 0, vs 0.25 chance); (2) item counts 1-4 under 100
  in three probe files; (3) cloze candidates mix parts of speech (~34% share
  the answer's); (4) replay size 100/phase; (5) tier 0's same-phase verbatim
  overlap with its training set (8-gram 0.0098, 35% of exam stories share a
  sentence, mostly ~7-word ones) is higher than phases 3 and 6 (0.0053 /
  0.0056), so DECISIONS 2026-09-25's "comparable" check is not met; only 8 of
  the 38 shared sentences match a fact-bank 6-gram.
- **Found on the way:** the phase-3 fact "Ted Williams ... last MLB player
  above .400" is outdated since MLB added Negro league statistics (May 2024;
  Josh Gibson .466, 1943); the bank verifier had passed it. Retired via
  revisions3.json, its 2 training stories dropped (phase 3 corpus 970;
  curriculum-learning 3c4b395); the installed prompt config keeps the old
  bank until the next template change so committed prompt hashes stay valid.
  And the corpus registry lost entries when generation runs ran in parallel
  (last save won); it now merges under a lock (f546ccd), entries restored.
- **To launch:** upload D:\Lifespan\prepilot_train and the exam dir as two
  private Kaggle datasets keeping their subfolders (replay/, stories/,
  probes/), set the notebook's three pre-pilot variables, owner's go.

## 2026-09-30 — `--micro-batch`: gradient accumulation after a T4 OOM

- **Measured:** the first Kaggle pre-pilot launch (arm phase0, T4, 14.56 GiB)
  ran out of CUDA memory at 13.38 GiB allocated, failing a 1 GiB request: the
  float32 logits of one 32 x 1,024 x 8,192 batch. The model is ~25M
  parameters; the activations of the full 32-sequence batch are the cost.
  The run used one GPU; the session's second T4 was idle.
- **Decided:** an additive flag, `--micro-batch N`, splits each batch into
  chunks of N rows and accumulates gradients before the single optimiser step.
  Each chunk's mean loss is weighted by its row share; rows have equal token
  counts and dropout is 0, so the step equals the whole-batch step up to
  float summation order. Batch, steps, lr schedule and replay share are
  unchanged: memory only, never a hyperparameter. Recorded in config.json as
  `micro_batch`. tests/test_train.py checks losses and weights against the
  whole batch on the toy model (chunks 1, 4 of 6, 6, 100). The notebook sets
  MICRO_BATCH = 8 and PYTORCH_ALLOC_CONF=expandable_segments:True.
- **Not covered:** consolidate.py's LoRA/distillation loops (arms C, D, D-nr)
  still run whole batches; they are not in the pre-pilot and hold teacher and
  student logits, so they need the same change before the grid. Multi-GPU
  (DDP) is not used; a cheaper use of a second GPU is running two
  independent arms at once (e.g. E beside phase0).

## 2026-10-01 — first Kaggle phase0 run: T4 precision bug; run discarded

- **Measured:** arm phase0, seed 0, pre-pilot (commit 0f43922, MICRO_BATCH 8)
  finished on a T4: phase 0 loss 9.09 -> 5.67 in 29 steps (928 sequences),
  ~3.4 s/step, each full exam scoring ~1m45s, shared phase-0 checkpoint
  written. But the log said "bf16 on cuda": `torch.cuda.is_bf16_supported()`
  counts emulation and is True on a T4 (compute 7.5), against PLAN.md's
  "fp16 with loss scaling on T4".
- **Decided:** native bf16 now also needs compute capability 8.0+; a T4 gets
  fp16 + GradScaler (tests/test_train.py fakes 6.0/7.5/8.0/9.0). The run is
  discarded and phase0 re-run at the fixed commit; the commit is in the
  shared-checkpoint fingerprint, so A and E could not inherit it anyway.
- **Notebook:** the checkpoint dataset's `_phase0/` is now restored whenever
  the dataset is attached, not only on RESUME (a fresh arm A needs it), and
  each push carries the other arms' `out/` folders forward from the attached
  version, since a dataset version replaces the whole dataset.

## 2026-10-01 — pre-pilot arm A: continuation measures register, not retention

- **Measured** (arm A, seed 0, commit 2c183f8, T4 fp16, pilot sizes, phases
  0/3/6; matrix pulled from the lifespan-checkpoints dataset). Phase-0 exam
  row by row (untrained / after 0 / after 3 / after 6):
  per-token loss 9.10 / 5.71 / 5.37 / 5.37; cloze 0.05 / 0.26 / 0.30 / 0.35
  (chance 0.05); continuation 0.24 / 0.885 / 0.29 / 0.10 (chance 0.25).
  After phase 0 alone (29 steps, loss 5.67) continuation is 0.885 on phase 0
  and 0.010 / 0.000 on phases 3 / 6; after phase 6 it is 0.10 / 0.26 / 0.45.
  Train time 17 + 41 + 77 s; each exam pass ~108 s; 0.156 GPU-hours.
- **Reading:** loss and cloze say arm A does not forget phase 0 at all; it
  keeps improving. Continuation says it loses 88% of its peak. Scores far
  below chance (0.010, 0.000) are impossible for a retention measure and
  expected for a preference one: the distractors are other phases' stories
  (PLAN.md, Evaluation), so the item asks which phase's register the model
  currently prefers, and the prefix barely matters to a model this weak.
  H1 on the headline metric would "hold" for a reason that is not forgetting.
- **Open, owner's decision** (H1-H4 and the continuation scoring are frozen):
  (1) score continuation by PMI, log p(option | prefix) - log p(option),
  which cancels the register prior and needs no exam rebuild; (2) same-phase
  distractors, which needs new probes and a new freeze; (3) keep it and
  report it as register preference, with H1 read on loss and cloze.
- **Run hygiene:** arm A ran with a stale --out (the notebook's run-folder
  cell was not re-run after ARM changed) and replaced phase0's result files;
  A's matrix carries the phase-0 row, so no number is lost. train.py now
  refuses a fresh run over a folder holding state.pt, and the launch cell
  asserts the run folder matches ARM/SEED. Arm E's result is not in the
  dataset.

## 2026-10-01 — continuation headline changed to PMI (owner); report hash bug

- **Decided (owner):** keep the exams, change the scoring. Continuation's
  headline is now summed PMI: log p(option | prefix) - log p(the same option
  tokens | <|endoftext|>). AGENTS.md Amendment 4; tests/test_evaluate.py has a
  hand-worked case where the raw sum picks the model's register and PMI picks
  the option the prefix supports.
- **Measured before freezing it**, arm A's pre-pilot checkpoints re-scored on
  CPU against the frozen 0/3/6 exam (accuracy, chance 0.25; n 96/100/100):

  | model / exam | raw sum | old length-norm | PMI sum | PMI per-token |
  | --- | --- | --- | --- | --- |
  | after p0 / p0 | 0.250 | 0.885 | 0.167 | 0.167 |
  | after p0 / p3 | 0.040 | 0.010 | 0.240 | 0.220 |
  | after p0 / p6 | 0.060 | 0.000 | 0.290 | 0.280 |
  | after p6 / p0 | 0.146 | 0.104 | 0.146 | 0.135 |
  | after p6 / p3 | 0.070 | 0.260 | 0.270 | 0.180 |
  | after p6 / p6 | 0.190 | 0.450 | 0.420 | 0.340 |

  Rate of picking the longest option (the answer is longest 0.16-0.31):
  raw sum 0.00, old length-norm 0.13-0.24, PMI sum 0.20-0.23, PMI per-token
  0.07-0.18. PMI sum removes the far-below-chance register artefact, has no
  length bias, and keeps the one real signal (after p6 / p6, 0.42, ~4 sd
  above chance); per-token PMI is short-biased and weaker, so it is not used.
- **Open:** phase-0 PMI sits ~2 sd below chance (16/96, 14/96), not from
  length. A candidate is the baseline context: every option is a mid-story
  paragraph but <|endoftext|> means "a story starts here". At pilot scale
  the model barely uses the prefix at all (after p0 / p0 is at chance), so
  continuation is near chance except where training was longest; cloze and
  loss carry the pilot's signal. Revisit on the re-run's numbers.
- **Found on the way:** report.filter_consistent compared every key of
  config.json's `hashes`, including the checkpoint fingerprint's digest,
  which contains the seed and commit, so any real grid would have kept one
  run per comparison and refused the rest. The fingerprint key is now
  skipped (test fails without the fix); the scoring rules go into
  config.json ("evaluation") and their digest into `hashes.scoring`, so runs
  scored under different rules are refused rather than pooled.
- **Next:** re-run phase0, A and E at the new commit (the commit is in the
  shared-checkpoint fingerprint, and E's result was never pushed).

## 2026-10-01 — pre-pilot complete (phase0, A, E at 4f769a2): continuation with cross-phase distractors is register discrimination under any scoring

- **Runs** (seed 0, T4 fp16, micro-batch 8, PMI headline): phase0
  phase0_s0_4f769a2_20261001T173103Z, A A_s0_4f769a2_20261001T174050Z
  (0.194 GPU-h), E E_s0_4f769a2_20261001T175434Z (0.195 GPU-h). All three
  pass results/validate.py; commit.txt dirty false. A's loss and cloze
  reproduce the 2c183f8 run to the third decimal across two Kaggle sessions,
  and its PMI continuation equals the local CPU re-score exactly.
- **Measured, phase-0 exam after phase 6, A vs E:** loss 5.37 vs 4.34; cloze
  0.35 vs 0.55; continuation (PMI, cross-phase distractors) 0.146 vs 0.917.
  A right after phase 0 scores 0.167 on the same item, before anything
  could be forgotten.
- **Test:** the same items re-scored locally with distractors drawn from
  other stories of the SAME phase (seeded, same length band; built in memory,
  never written; accuracies only). PMI accuracy p0 / p3 / p6 (chance 0.25):
  untrained 0.208 / 0.250 / 0.150; after p0 0.208 / 0.260 / 0.250; A after
  p6 0.281 / 0.250 / 0.370; E after p6 0.312 / 0.380 / 0.330. Raw summed
  scoring is 0.10-0.15 throughout (useless).
- **Reading:** E's 0.917 falls to 0.312: with other phases' paragraphs as
  distractors, PMI rewards matching the prefix's register, which needs a
  model that knows several registers. A sequential arm at row i knows one,
  so its diagonal sits at chance however well it learned the phase, and
  "forgetting" (peak minus final) cannot be read from it. Amendment 4 fixed
  the below-chance artefact but not this. With same-phase distractors the
  item asks only whether the option follows this story; at pilot scale every
  model is near chance there (best 0.37-0.38, ~2.8 sd).
- **And:** by every measure that can register learning (loss, cloze,
  same-phase continuation), arm A does not forget phase 0 at pilot scale; its
  phase-0 cloze rises 0.26 -> 0.30 -> 0.35 and its loss improves 5.71 ->
  5.37 while it trains on phases 3 and 6. The model is far from fitting any
  phase (final train loss ~4.96), so later phases still teach general
  English that helps phase 0. A pre-pilot is never an H1 verdict, but this
  is H1's kill direction.
- **Open, owner's decision:** (1) continuation distractors from the same
  phase (probe rebuild and a new freeze; stories unchanged), keeping PMI;
  (2) what the pilot needs for forgetting to be measurable at all: more
  steps per phase, the 5,000-story sizes, or phases that differ more.

## 2026-10-01 — Amendment 5 (owner): same-phase continuation distractors; pre-pilot exam re-frozen as lifespan-prepilot-v2

- **Decided (owner):** continuation distractors come from three other exam
  stories of the item's own phase (AGENTS.md Amendment 5, af21ee0), PMI
  scoring kept. Builder, verify_probes and the evaluator's item check all
  enforce it; 487 tests pass.
- **Re-frozen** (exam-keeper script outside the repo,
  exams_prepilot/scratch/keeper/reprobe_v2.py): the 300 exam stories and the
  cloze files are byte-identical to the 2026-09-30 freeze; only the three
  continuation files changed. New dir exam_dir_v2, experiment_id
  lifespan-prepilot-v2, manifest sha256 27bc091c...edf0, which records
  `supersedes` (lifespan-prepilot, 415aa98a...) and the probe-builder
  commit. Items 96 / 100 / 100; answer slots balanced; answer / mean
  distractor length median 1.04 / 0.95 / 0.93 (was 0.94 / 1.12 / 1.13);
  answer is the longest option 0.27 / 0.16 / 0.19 (chance 0.25). guard.check
  passes on prepilot_train with the new id and refuses the old id and none.
- Matrices scored on the old probes are never pooled with these.

## 2026-10-01 — grid036: phases 0/3/6 at full size on a rented 8x3090 node (owner)

- **Decided (owner):** the next run is the grid (arms phase0, A, B, C, D,
  D-nr, E; seeds 0-2) on phases 0/3/6 at PLAN.md's full sizes, 5,000 training
  stories and 500 exam items per phase, on a rented 8x RTX 3090 node
  (~$1/h). Corpus rebuilt on one template: side-character names merged
  (curriculum-learning 37eac30) and the revised phase-3 fact bank installed
  (282901c). Train prompts seed 42 (5,000/phase), exam prompts seed 4242
  (600/phase: 500 + reserve), tiers as the pre-pilot (0; 7-9; 13). No hash
  or name/location/noun/verb overlap between train and exam prompts.
- **Review (owner): a 10% audit of the training stories**, both gates on
  Opus; the exam stories get the full review. **Fixed before any result:**
  the audit draws 500 stories per phase uniformly at random (seed 42). A
  phase passes when the sample's fact-check pass rate AND age-fit pass rate
  are each >= 90% (the pre-pilot's first passes were 93-96%); a phase below
  either gets the full review of all 5,000. Sampled stories that fail are
  dropped from the corpus; the unsampled stories' estimated defect rate is
  recorded, not corrected.
- **Smoke:** 20 phase-3 stories, all 20 use an assigned side-character
  name; mean 398 words.

## 2026-10-01 — grid036 training audit: phases 3 and 6 miss the fact bar; full fact check (owner)

- **Measured** (the pre-registered 10% audit, 500 stories per phase drawn with
  seed 42, both gates on Opus; every verdict file checked to cover exactly
  its part): fact-check pass / age-fit pass / both, with 95% intervals on the
  fact rate —
  phase 0: 96.4% [94.8, 98.0] / 95.2% / 462 of 500;
  phase 3: 88.6% [85.8, 91.4] / 93.6% / 422;
  phase 6: 88.2% [85.4, 91.0] / 96.6% / 425.
  Phase 3/6 fact failures are mostly worked arithmetic and wrong details of
  real works, rules and places, not register.
- **Rule outcome:** phase 0 passes; phases 3 and 6 fall below the fixed 90%
  fact bar, so they get the full review.
- **Decided (owner):** the full review is the gate that failed: a fact check
  of every unaudited phase 3/6 training story; every failure is dropped, none
  regenerated (corpora end near 4,450). Age fit passed in all three phases
  and is not re-run. Phase 0 keeps all stories except its audited failures.
  The session's web-search budget is spent, so checkers verify by fetching
  pages directly.
- **Also dropped everywhere:** the mechanical defects (CJK characters, a
  "Continue reading" stub, duplicated paragraphs): 1 / 2 / 4 training stories
  with duplicated paragraphs and 0 / 6 / 13 with CJK characters in phases
  0 / 3 / 6.

## 2026-10-01 — grid036 full fact check done; train dir built

- **Measured:** all 226 full-fact parts verified complete (the build refuses
  on any story without a verdict). Kept / source after dropping fact
  failures, audited age < 3 and mechanical defects: phase 0 4,961 / 5,000;
  phase 3 4,373 / 4,997; phase 6 4,364 / 5,000. Phase 3/6 full-check fact
  failures ~12%, matching the audit estimate.
- **Built:** `D:\Lifespan\grid036_train` (train_phase_{0,3,6}.jsonl, replay
  500 per phase, seed 4243). Report: `exams_grid036/scratch/build_train_report.json`.
- **Guard:** `training.guard` passes against the frozen exam (manifest sha
  25cee546…, experiment_id lifespan-grid036): 13,701 lines vs 1,500 exam
  stories, data_hash 37861c52b23f.

## 2026-10-02 — grid036 result: arm B memorised its replay buffer; arms C/D could not learn new phases (owner)

- **Measured (grid036, 21 valid runs, 2× RTX 5090, about 1.7 GPU-hours, ~$1):**
  continuation forgetting was 0 for nearly every arm, so the headline exam
  could not separate them. On cloze and exam loss (seed 0):
  - arm A (no replay) kept phase 0 well: phase-0 cloze 0.58 → 0.54; exam loss 4.21 → 4.48.
  - arm B (replay) did worse than A: phase-0 cloze 0.58 → 0.35 → 0.28.
  - arms C, D and D-nr barely learned new phases: phase-6 exam loss about 5.5,
    against arm A's 3.9.
  - H2b: D took 1.26× B's GPU time.
- **Arm B diagnosis:** the 500-story buffer packs to 121 phase-0 sequences.
  B replayed 4,816 sequences in phase 3, about 40 per sequence. Final arm-B
  loss was 1.15 on the replayed sequences and 5.0 on other phase-0 training
  stories and on the phase-0 exam. Arm A, with no replay, was about 4.5 on all
  three. The replay code is correct; the buffer is too small.
- **Arms C/D diagnosis** (laptop RTX 3080 probes, seed-0 phase-0 checkpoint,
  grid-faithful phase 3: 344 steps × 32 × 1024 tokens, warmup 200, cosine).
  Phase-3 exam loss / phase-0 exam loss:
  - start (phase-0 checkpoint): 6.29 / 4.23
  - full fine-tuning, lr 3e-4: 4.59 / 4.55 (the grid's arm A: 4.60)
  - LoRA rank 16, lr 3e-4: 5.65 / 4.67 (the grid's arm C: 5.66)
  - LoRA rank 16, lr 1e-3: 5.59 / 4.65; lr 3e-3: 5.81 / 4.89
  - LoRA rank 64: 5.52 / 4.64
  - full fine-tuning with embeddings frozen: 5.34 / 4.60
  - LoRA rank 16 + low-rank embedding adapter, rank 16: 5.56 / 4.71; rank 64: 5.27 / 4.80
  - **LoRA rank 16 + trainable embeddings: 4.80 / 4.66**

  The bottleneck is the frozen embeddings, not rank or learning rate. LoRA did
  not protect phase 0 better than full fine-tuning in any configuration.
- **Decided (owner):**
  - AGENTS.md Amendment 7: replay draws from all earlier-phase training data.
    Built as `D:\Lifespan\grid036_train_v2`, with training files byte-identical
    to v1; guard ok, data_hash cb2217bbcba5.
  - AGENTS.md Amendment 6: the adapter phase (step 1) also trains the
    embeddings, `TrainConfig.lora_train_embeddings = True`. The cost: about 17%
    of the model is trainable in step 1, against about 6% with LoRA alone.
  - Re-run the grid with the same exams and experiment id. Its results are not
    pooled with the first grid's.
- **Kept, not re-run:** the first grid's results, in
  `D:\Lifespan\grid036_out` (report under `report/`).
