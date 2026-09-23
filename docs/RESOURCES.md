# Resources

Everything this project draws on: the four links, the three local projects, the
three original ideas, and how they fit together. Written 2026-09-21 from the
session that produced `PLAN.md`; the live copy of the plan is
https://claude.ai/code/artifact/a0c89fc1-111f-48ab-ae5e-a42f1df1d3a1.

## Links

### Building a harness with Jev — LangChain blog
https://www.langchain.com/blog/building-a-harness-with-jev

Jev (TypeSafe AI) is a "System One" model: it does not generate text, it answers
typed questions about a state — `choice`, `score`, `noul` (yes/no) — with
calibrated probabilities, trained with reinforcement learning for calibrated
decisions (RLCD). Claimed up to 200x faster and 400x cheaper than an LLM for
classification. Used inside agent loops for model routing and pre-execution
safety gating. Core thesis: "every decision requires another model call" is what
makes agent loops slow and expensive.

What it gives this project: the System-1 half of the dual-process design, and
the market proof that people pay for it.

### Laya — NandhaKishorM/laya (GitHub)
https://github.com/NandhaKishorM/laya

Open-source Jev-class decision engine. ModernBERT-large (421M) English checkpoint,
mmBERT-base multilingual checkpoint, and a `typed-decisions` checkpoint; a
sub-millisecond router picks the checkpoint by script/language. Same three
question types as Jev; trained with policy gradients against strictly proper
scoring rules (RLCD). Claims 0.766 vs Jev's 0.727 on the typed-decisions
benchmark, ECE 0.081 vs 0.246 after temperature fitting, ~7 ms per batched
question on GPU. README states domain fine-tuning lifts accuracy substantially
from near-baseline zero-shot.

What it gives this project: the age-appropriateness gate on generated stories
(a `score` question per story), and the open model for the System-1 layer.
Reflex's `probe/laya_probe.py` measured it on the RTX 3080 laptop: the
`typed-decisions` checkpoint with a dict state scored 8/8 on the inline-vs-
background decision it owns.

### Dream-RSI — recursive self-improvement through evolving worlds
https://dream-rsi.com/

An agent's exploration policy improves by replaying candidate policies over the
recorded tree of past attempts ("history is the world it dreams in — not a learned
world model"). Loop: explore online → the finished discovery tree becomes a replay
simulator → thousands of alternative policies are scored offline at zero
executions → the winner redeploys and grows the pool of worlds. The current policy
is always a candidate, so quality cannot degrade between rounds. Reported: 162x
fewer discovery calls on a Lasso algorithm-engineering task, 2.4x fewer generations
on VGG16 GPU kernels. Exact only inside the explored space.

What it gives this project: the replay targets in the consolidation step are the
previous base model's own logits — the old model as the exact record of what it
knew. Also the offline-replay answer to Reflex's measurement problem (see below).

### Human-Timescale Adaptation in an Open-Ended Task Space — DeepMind, arXiv 2301.07608
https://arxiv.org/abs/2301.07608

The Adaptive Agent (AdA). Large-scale meta-RL over a vast, smooth, diverse task
distribution + an attention-based memory + an automated curriculum that
prioritises tasks at the agent's skill frontier produces an in-context learning
algorithm that adapts to held-out 3D tasks as fast as humans, with no weight
updates at test time. Scales with network size, memory length and task diversity.

What it gives this project: the frontier curriculum is Vygotsky's zone of
proximal development made computational — the argument for exam-gated promotion
between phases (extension in `PLAN.md`) rather than a fixed schedule, and for
prioritising the worst-calibrated region when retraining a decision model.

## Local projects

### `D:\Reflex` — non-blocking dispatch layer for agent harnesses

Python, ~5k lines, five Claude Code hooks wired, a Laya probe, a redacted
training-corpus writer (`src/reflex/corpus.py`: the deterministic rules layer as a
free labelling oracle), and `bench/` — a harness that measures on/off with seeds,
t-tests and minimum detectable effect. Plan and measurements in `REFLEX_PLAN.md`.

Findings that shape this project:
- Hook overhead was ~0.5% of wall clock; the cost of a System-1 layer is never
  latency, it is LLM turns. Reflex added a turn (the model checking its job) and
  ran 49.8% slower; the best case measured was break-even.
- `blocked_at_use_ms = 0` in every run: lazy blocking never fired because the
  model waits eagerly. Design assumptions must be measured in a live session.
- Resolving a 25% effect needed n≈10 per mode (~67 min) and the variance was the
  model's verbosity — the problem Dream-RSI's offline replay solves.
- Its statistical discipline (3 seeds minimum, report MDE, discard instrument
  defects found by reading transcripts) is the standard for `PLAN.md`.

### `D:\WORK` — Holdout Labs (AI Work Protocol MVP)

FastAPI + Next.js + Postgres/Redis + Docker worker. Milestones 1–8 done: bounties
with private datasets, source submissions rebuilt from a named commit (`verified`
is reserved for those; pre-built images are `unverified_image`), per-task
reputation, agents as first-class principals over REST and MCP, simulated escrow
and ledger, disputes. `docs/vision.md`: "the work itself becomes a verified,
persistent asset"; verification ladder Level 4 (continuous evaluation) is unbuilt.
Falsifiers listed there: nobody funds problems; verification costs more than it is
worth; objective evaluation does not generalise; verified history does not
transfer.

What it gives this project: the reproduction rule applied to our own results, and
the destination — the exam sets plus evaluation code become a continual-learning
bounty type whose metric (average forgetting on held-out phases, reproduced from
commit) is code.

### `D:\Lifespan\curriculum-learning` — package `lifespan_learning`

A prompt generator for a developmental curriculum. Seven neo-Piagetian phases
(`config/phases.yaml`: Emergent Symbolic Thought → Stable Rule-Based Reasoning →
Relational System Coordination → Abstract Variable Reasoning → Systematic
Hypothetical Construction → Meta-System Integration → Epistemic Synthesis),
fourteen grade tiers (JK–12), content types (exposures, experiences, basic and
advanced learning arcs), tones, features, per-phase lexicons mined from books,
gendered name lists. `PromptDatasetGenerator` writes prompts + metadata to jsonl;
response generation is a scratch Gemini call (`response/scratch.py`); no model has
been trained. Known gaps: lexicon files only for phases 0–2 (one book processed),
no exam/holdout side, no batch response pipeline — all on the checklist in
`PLAN.md`.

What it gives this project: the data generator and the developmental structure
the whole experiment is built on.

### `Meridian` (not in this session's folders)

Blender terrain addon over open terrain-diffusion weights; first of several
planned design tools. Not connected to this project; listed so it is not
forgotten as a separate line of work.

## The three original ideas, and what became of them

1. **Train a model to update the weights of another model, for continual
   learning.** Exists in several forms: Text-to-LoRA (Sakana AI, 2025 — a
   hypernetwork emits a LoRA from a task description), SEAL (MIT, 2025 — the model
   writes its own fine-tuning data and update directives; reports catastrophic
   forgetting), test-time-training layers and Titans (Google, 2025 — the hidden
   state is a small model updated at test time). The mechanism is not the
   differentiator; verified non-regression around the update is. Cheapest
   sellable form: a decision model retrained from accumulated traces with an
   independent held-out gate (see product framing below).
2. **A model that works like a brain, psychiatric (functional) view.** Became the
   architecture diagram in the connection map below. Kahneman's System 1/2 is
   Jev/Laya + LLM; Piaget's stages are the curriculum; complementary learning
   systems (hippocampus fast, cortex slow, consolidated in sleep — McClelland,
   McNaughton & O'Reilly 1995) is ideas 1 and 3 combined; ACT-R's production
   compilation is deliberate decisions becoming habits.
3. **Reverse LoRA: freeze a task LoRA, unfreeze the base, train the base to absorb
   it.** As literally stated it is a merge (`W += BA`, no training). The version
   worth building is when merging is not enough — many LoRAs that interfere when
   summed (the TIES/DARE problem), a LoRA from an older base, absorbing into a
   subset of layers — where the frozen LoRA is a teacher and the base is distilled
   toward it with replay of prior tasks. That is the consolidation step in
   `PLAN.md`, and hypothesis H3 is exactly "is distillation needed, or is merging
   enough".

## Connection map

| Function (psychiatric view) | Computational component | Source | Where it exists |
| --- | --- | --- | --- |
| Reflex arc | Deterministic deny/background/inline rules | Reflex | `.reflex.toml`, done |
| Habit / System 1 | Calibrated typed-decision encoder | Jev, Laya | Laya probe in Reflex |
| Metacognition ("feeling of knowing") | Calibrated probability; low confidence escalates | Laya's ECE; Reflex's `ask_human` tier | Done |
| Automaticity (deliberate → habitual) | Repeated System-2 decisions become System-1 training data | ACT-R; Reflex `corpus.py` | Write side done |
| Inhibition / veto | Deny list beneath the classifier; classifier can only downgrade | Reflex | Done |
| Deliberation / System 2 | The LLM | — | Orchestrator tier |
| Working memory | Context window; AdA's attention memory | AdA | Free |
| Episodic memory | Recorded trace store | Dream-RSI trees; Reflex `metrics.jsonl` | Needs a first-class store |
| Semantic memory | Base weights | — | Free |
| Procedural memory | Rules + decision model | Reflex | Done / next |
| Fast memory (hippocampus) | Per-phase LoRA | Idea 3 | **This project** |
| Sleep / consolidation | Replay traces to score policies; distil fast weights into slow with interleaved replay | Dream-RSI; CLS theory; idea 3 | **This project** — missing everywhere else |
| Development | Staged curriculum, promotion by exam | curriculum-learning; AdA frontier | Data side exists; exam gating is an extension |
| Affect / arousal as global gain | Stakes scalar shifting the escalation threshold | Speculative | Reflex's destructive-command asymmetry is the first instance |
| The psychiatrist | Independent, repeated, held-out assessment | Holdout Level 4 | Unbuilt |

## Self-improvement loops, in build order

1. **Harness improves, model frozen.** Dream-RSI applied to Reflex: record
   sessions, replay candidate dispatch/routing policies offline, promote the
   winner. No training.
2. **System-1 model improves.** Retrain the Laya-class decision model from
   accumulated labelled traces, prioritising the worst-calibrated region (AdA),
   gated by Holdout on a held-out slice. The sellable one.
3. **Base model improves.** SEAL-style self-generated data → LoRA → consolidation
   with replay → held-out exam on old phases. Research; this repo is its testbed.

Common hazard: Goodhart. Defences: a vast task distribution (AdA), exact history
(Dream-RSI), held-out plus reproduction (Holdout), and a held-out *generator*
rather than a held-out file.

## Product framing (from the session, for reference)

"Habit formation for agents": record the decision points a harness makes with
the LLM, label them from rules and outcomes, fine-tune a Laya-class model, serve
it behind a daemon, and score it in LLM turns removed and calibration on a
held-out slice of the customer's own traces, reproduced from a named commit.
Jev proves the demand, Laya makes the model a commodity, neither sells evidence
that the model is safe to trust in a given loop. Continual learning and
consolidation are the roadmap, not the MVP. Fast falsifier: fine-tune Laya on
Reflex's traces and measure turns removed on the bench task; if the completion
digest alone removes the extra turn, the classifier is a feature, not a product.
