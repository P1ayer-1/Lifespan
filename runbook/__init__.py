"""Operational tooling owned by `run-operator`: the rented-A100 grid runbook,
the GPU-hour ledger, and the operational resume round-trip proof.

Nothing here trains anything. `grid.py` and `resume_roundtrip.py` both call
`python -m training.train` as a subprocess -- the frozen CLI contract
(`AGENTS.md`) -- rather than importing `training.train`'s internals, so this
package keeps working while `trainer-core`, `consolidation` and `evaluator`
are mid-edit on `training/`. The only import from `training/` anywhere in this
package is `training.config` (arm names, EXAM_TYPES, N_PHASES), per the brief.
"""
