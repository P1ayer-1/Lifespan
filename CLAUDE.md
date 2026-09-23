# Lifespan — continual-learning experiment

Read this file first in every session. `PLAN.md` is the source of truth for what
gets built and what counts as a result; `docs/RESOURCES.md` records every
project, paper and link this work draws on and why.

## What this is

A ~30M-parameter model learns seven developmental phases (junior kindergarten to
grade 12) in order, and we measure how much it forgets earlier phases under five
training regimes. The claim under test: a frozen per-phase LoRA distilled into the
base weights with replay of earlier phases ("sleep consolidation") retains earlier
phases at least as well as interleaved-replay fine-tuning, at no more than 1.5x its
compute. Hypotheses H1–H4 and their kill criteria are fixed in `PLAN.md` and are
not changed after the first run starts.

This is the research testbed for a larger design (see `docs/RESOURCES.md`,
"Connection map"): a brain-shaped agent architecture — reflex rules, a calibrated
System-1 decision model, an LLM for deliberation, fast weights, a sleep phase, a
developmental curriculum, and an independent evaluator. This repo tests the sleep
phase at tiny scale.

## Related repos on this machine

| Path | Role here | Touch it? |
| --- | --- | --- |
| `D:\Lifespan\curriculum-learning` (package `lifespan_learning`) | Generates the training prompts; owns the phases/tiers/arcs/tones config; needs the exam side added (checklist in `PLAN.md`) | Yes — the repo-change checklist targets it. Keep data generation there; keep training code here. Moved beside this repo on 2026-09-23. |
| `D:\Reflex` | Dispatch/classifier layer for agent harnesses. Reuse: the bench harness's statistics discipline (seeds, MDE, t-test), the rules-as-labelling-oracle pattern, the Laya probe | Read-only from this project |
| `D:\WORK` (Holdout Labs) | Verified evaluation platform. Its rules apply to our own results: a score counts only when reproduced from a named commit on held-out data | Read-only from this project |

## Layout (target)

```
Lifespan/
  CLAUDE.md            this file
  AGENTS.md            who builds what: ownership, shared contracts, waves
  .claude/agents/      the agent team; start with `claude --agent lifespan-lead`
  PLAN.md              the experiment plan (mirror of the live doc)
  docs/RESOURCES.md    projects, links, papers, ideas, connection map
  docs/DECISIONS.md    append-only decision log (create on first decision)
  training/            tokenizer, model, arm-flagged training loop, LoRA,
                       consolidation, evaluation
  exams/               holdout manifest (hashes) — never the exam text itself
  notebooks/           Kaggle notebook that calls training/ and checkpoints per phase
  results/             one folder per run: matrix.json, timings.json, config, commit
```

Exam stories and probes live outside this repo and outside any directory the
training script can read. Only their hashes are committed (`exams/manifest.json`).

## Rules

- **No training on the laptop.** Pilot on Kaggle (free quota), final 3-seed grid
  on one rented A100 session. Code must survive a Kaggle session dying: checkpoint
  after every phase.
- **Exams are frozen before the first training run.** The training script refuses
  to start if any training file's hash appears in `exams/manifest.json`. Changing
  the exams is a new experiment with a new manifest.
- **Every arm is one script with a flag.** Arms A and B differ only by replay
  fraction; C and D share LoRA code. Do not fork scripts per arm.
- **Seeds 0, 1, 2 for everything.** A difference smaller than seed spread is
  reported as no difference. Reflex's bench showed what an underpowered n costs.
- **A result counts when reproduced.** Record commit, seed, data hashes, hardware
  and driver versions with every run. The headline arms get a second run from the
  commit before anything is called confirmed.
- **Log GPU-hours per arm.** H2 is stated in compute as well as accuracy.
- **Windows-native dev machine.** Python paths and scripts must run on native
  Windows (not WSL) as well as on Kaggle's Linux. No shell scripts; Python entry
  points only.
- **No secrets in the repo.** Generation-model credentials (`OPEN_ROUTER_API_KEY`
  for the GLM 5.3 corpus generator since 2026-09-23; `ANTHROPIC_API_KEY` for the
  Claude fallback) and Kaggle credentials come from the environment (`.env` is
  git-ignored). The generation scripts must never print or log them.
- **LLM labor runs as subagents, not API calls** (owner, 2026-09-23): judging
  stories, writing fact banks and lexicons, fact-checking and audits are done by
  Claude Code subagents on the Max plan. API credit is spent only on the corpus
  generator itself.
- **Decisions go in `docs/DECISIONS.md`**, dated, with the measurement that drove
  them, in the style of `D:\Reflex\REFLEX_PLAN.md`: measure, then decide, then
  record both.

## Status

2026-09-21 — plan written; no code yet. Week 1 of the schedule starts with the
curriculum-learning repo changes and one generated phase.
