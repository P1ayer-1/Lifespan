"""Matrix arithmetic: the three continual-learning numbers, exactly as
PLAN.md's *Evaluation* section defines them.

M is the 7x7 matrix from `matrix.json`: `M[i][j]` is phase j's exam score after
training phase i. `untrained[j]` is phase j's score before any training, scored
once per seed and stored in the same file because forward transfer cannot be
recovered afterwards.

    Average accuracy   = mean of the last row, M[6][:].
    Forgetting of j    = max over i of M[i][j]  -  M[6][j].
    Average forgetting = mean of that over j < 6.
    Forward transfer j = M[j-1][j] - untrained[j]      (undefined for j = 0).

**Sign, defined once.** Higher is better for `cloze` and `continuation`. For
`perplexity` the stored value is a mean per-token loss, where lower is better,
so every function here first maps the stored value to a score with
`to_score` = `SIGN[exam_type] * value`, i.e. the loss is negated. The
consequence, stated so nobody has to re-derive it: on perplexity, *forgetting
is positive when the loss rose*, and it is measured in nats per token.

**Stored keys vs verdict keys.** `matrix.json` carries four keys: the three
`EXAM_TYPES` plus `continuation_summed`, the length-biased score kept only so
the normalisation choice stays checkable (AGENTS.md, amendments 2026-09-22).
The arithmetic here accepts all four (`STORED_EXAM_KEYS`), but
`fraction_of_peak_lost` accepts only `VERDICT_EXAM_TYPES` - a fraction of a
negated loss is not a quantity, and a score the contract says is never
headlined must not be able to decide a hypothesis by accident.

**Units.** Scores are fractions in [0, 1]; a "point" in H2 and H3 is a
percentage point, so a difference in points is `100 * (a - b)`. Use
`to_points`; do not restate a threshold in friendlier units.
"""

from __future__ import annotations

from typing import Sequence

from training.config import EXAM_TYPES, N_PHASES

__all__ = [
    "SIGN",
    "STORED_EXAM_KEYS",
    "VERDICT_EXAM_TYPES",
    "NEVER_HEADLINE",
    "POINTS_PER_UNIT",
    "to_points",
    "to_score",
    "as_score_matrix",
    "average_accuracy",
    "per_phase_forgetting",
    "average_forgetting",
    "forward_transfer",
    "fraction_of_peak_lost",
    "phase_column",
    "validate_matrix",
]

#: The one place the direction of each stored key is written down.
SIGN: dict[str, float] = {
    "perplexity": -1.0,  # stored as a loss: lower is better
    "cloze": 1.0,
    "continuation": 1.0,
    "continuation_summed": 1.0,  # an accuracy, stored and never headlined
}

#: What matrix.json carries and what the arithmetic here accepts: the three
#: EXAM_TYPES plus the stored sibling (AGENTS.md, amendments 2026-09-22).
STORED_EXAM_KEYS: tuple[str, ...] = tuple(EXAM_TYPES) + ("continuation_summed",)

#: The only keys a verdict may be read on. `perplexity` is a loss and a
#: percentage of one means nothing; `continuation_summed` is the length-biased
#: score kept only so the normalisation choice stays checkable. Both are
#: computed, stored and plotted - neither ever decides H1-H4.
VERDICT_EXAM_TYPES: tuple[str, ...] = ("cloze", "continuation")
NEVER_HEADLINE: tuple[str, ...] = tuple(k for k in STORED_EXAM_KEYS if k not in VERDICT_EXAM_TYPES)

#: Scores are fractions; H2/H3 thresholds are in percentage points.
POINTS_PER_UNIT = 100.0

Matrix = Sequence[Sequence[float]]


def to_points(x: float) -> float:
    """A score difference in percentage points (H2's 3, H3's 5)."""
    return x * POINTS_PER_UNIT


def to_score(value: float, exam_type: str) -> float:
    """Stored value -> a number where higher is better."""
    _check_exam_type(exam_type)
    return SIGN[exam_type] * float(value)


def _check_exam_type(exam_type: str) -> None:
    if exam_type not in STORED_EXAM_KEYS:
        raise ValueError(
            "unknown exam key " + repr(exam_type) + "; expected one of " + repr(STORED_EXAM_KEYS)
        )


def validate_matrix(M: Matrix) -> None:
    """A matrix is 7 rows of 7 floats or it is not a result."""
    rows = list(M)
    if len(rows) != N_PHASES:
        raise ValueError("matrix has " + str(len(rows)) + " rows, expected " + str(N_PHASES))
    for i, row in enumerate(rows):
        if len(list(row)) != N_PHASES:
            raise ValueError(
                "matrix row " + str(i) + " has " + str(len(list(row))) + " entries, expected "
                + str(N_PHASES)
            )


def as_score_matrix(M: Matrix, exam_type: str) -> list[list[float]]:
    """The matrix mapped so that higher is better (perplexity negated)."""
    validate_matrix(M)
    _check_exam_type(exam_type)
    s = SIGN[exam_type]
    return [[s * float(v) for v in row] for row in M]


def phase_column(M: Matrix, phase: int, exam_type: str) -> list[float]:
    """Phase `phase`'s score after each of the seven phases, as a score."""
    S = as_score_matrix(M, exam_type)
    return [row[phase] for row in S]


def average_accuracy(M: Matrix, exam_type: str = "continuation") -> float:
    """Mean of the last row: after phase 6, across all seven phases."""
    S = as_score_matrix(M, exam_type)
    last = S[N_PHASES - 1]
    return sum(last) / len(last)


def per_phase_forgetting(M: Matrix, exam_type: str = "continuation") -> list[float]:
    """max over i of M[i][j] minus M[6][j], for every j. Length 7."""
    S = as_score_matrix(M, exam_type)
    final = S[N_PHASES - 1]
    out = []
    for j in range(N_PHASES):
        peak = max(S[i][j] for i in range(N_PHASES))
        out.append(peak - final[j])
    return out


def average_forgetting(M: Matrix, exam_type: str = "continuation") -> float:
    """Mean of per-phase forgetting over j < 6.

    Phase 6 is excluded because it was trained last: its peak *is* its final
    score, so it contributes a structural zero. This is the number H2 and H4
    are stated in.
    """
    per = per_phase_forgetting(M, exam_type)[: N_PHASES - 1]
    return sum(per) / len(per)


def forward_transfer(
    M: Matrix, untrained: Sequence[float], exam_type: str = "continuation"
) -> list[float | None]:
    """M[j-1][j] - untrained[j] for j >= 1; None for j = 0.

    Phase 0 has no earlier phase, so its forward transfer is undefined rather
    than zero: M[-1][0] is not a row of the matrix.
    """
    S = as_score_matrix(M, exam_type)
    if len(list(untrained)) != N_PHASES:
        raise ValueError("untrained vector must have " + str(N_PHASES) + " entries")
    base = [SIGN[exam_type] * float(v) for v in untrained]
    out: list[float | None] = [None]
    for j in range(1, N_PHASES):
        out.append(S[j - 1][j] - base[j])
    return out


def fraction_of_peak_lost(M: Matrix, exam_type: str = "continuation", phase: int = 0) -> float:
    """H1's number: (peak - final) / peak on one phase's column.

    Stated as a *fraction of the peak score lost*, which is how PLAN.md's H1
    row is written ("loses >= 15% of its peak phase-0 exam score"). Never
    restated as an absolute point difference.

    Refuses every key in `NEVER_HEADLINE`: `perplexity` is a loss and a
    percentage of a negated loss means nothing, and `continuation_summed` is
    the length-biased score kept only so the normalisation choice stays
    checkable. Both are stored; neither decides a hypothesis.
    """
    _check_exam_type(exam_type)
    if exam_type not in VERDICT_EXAM_TYPES:
        raise ValueError(
            repr(exam_type) + " is never a verdict metric (frozen contract); "
            "a verdict is read on one of " + repr(VERDICT_EXAM_TYPES)
        )
    col = phase_column(M, phase, exam_type)
    peak = max(col)
    final = col[N_PHASES - 1]
    if peak <= 0.0:
        raise ValueError(
            "peak phase-" + str(phase) + " score is " + str(peak)
            + "; a fraction of a non-positive peak is undefined"
        )
    return (peak - final) / peak
