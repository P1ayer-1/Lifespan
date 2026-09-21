# Decisions

Append-only. Date, the measurement, the decision it drove.

## 2026-09-21

- Plan written before any code; H1-H4 thresholds fixed (PLAN.md). Training code lives in this repo; data generation stays in D:\Tiny Models\curriculum-learning.
- Agent team and AGENTS.md added (.claude/agents/, lead is `lifespan-lead`). Contract shapes in AGENTS.md are drafts until the lead freezes them per wave.
- CPU smoke tests are not "training on the laptop": toy config (<= 1M parameters), synthetic or 20-story fixture data, <= 50 steps, asserting mechanics only. No measurement; without it the pipeline cannot be tested before a GPU session. Any quotable number still comes from Kaggle or the A100.
- Replay is on top of the per-phase token budget, implemented as a larger batch (N new-phase sequences + ~0.43 N replay), same step count and schedule in every arm. No measurement; reasoning: inside the budget Arm B would see 2.8 epochs of new material, lowering its peak and flattering its forgetting number (peak minus final). Cost lands in GPU-hours, which H2 already counts: Arm B ~1.43x Arm A's tokens. Same rule for Arm D's distillation step; Arm E matches Arm A's total.
- Phase 0 is full training for all sequential arms, trained once per seed and shared by A, B, C, D, D-nr; its wall clock is copied into each arm's timings. No measurement; reasoning: the consolidation recipe taken literally at k=0 trains a rank-16 LoRA on a frozen random base (embeddings included), which cannot learn, so Arm D's first teacher would be noise. PLAN.md already had this for Arm C only.
- Story generation moves off Gemini, probably to Claude (owner, 2026-09-21). Provider and model id not final; to be recorded here, with the re-costed ~23M output tokens, before the first paid batch. PLAN.md's Data section still says Gemini Flash until then.
