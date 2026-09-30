"""Aggregation across runs: seed spread, plots, and the H1-H4 table.

Three rules run through everything here, and they are written down once:

1. **Every comparison is over seeds 0, 1, 2.** A mean is never printed without
   its sample standard deviation, its n, and the three values themselves.
2. **A difference smaller than the seed spread is "no difference"**
   (`NO_DIFFERENCE_RULE`). Seed spread is `max(sd(a), sd(b))`, sample sd with
   ddof=1. The same quantity is the grid's minimum detectable difference and is
   printed beside every comparison.
3. **The report refuses runs, by name, rather than averaging what exists.** A
   run is excluded when a file is missing or unreadable; when `matrix.json`
   lacks one of the four stored keys, is not 7x7, or holds a non-finite cell;
   when `timings.json` has no `gpu_name` or `gpu_hours`; when `commit.txt` says
   the tree was dirty **or fails to say** (a missing or unparseable `dirty`
   flag is excluded exactly like a dirty one - it is never assumed clean);
   when `config.json` has no top-level `hashes` block; when it is a pilot run
   in a grid report or a grid run in a pilot report; or when its hashes differ
   from the others it would be compared with. Every exclusion is printed with
   its reason.

**Subset runs** (`train.py --phases 0,3,6`). A run's trained phases are read
from `config.json["phases"]` (absent: all seven, as every run before the flag
existed). The matrix stays 7x7 by real phase id and is never renumbered: rows
of undeclared phases may be null and undeclared exam columns NaN, and only the
declared rows and columns must be complete. Every metric is computed over the
declared phases only (`metrics.py`, "Subset runs"), and a report never mixes
runs that declare different phase lists -- the minority is excluded by name.
A full 0..6 run takes exactly the path it always took.

This module computes the hypothesis verdicts; the lead decides what is written
into `PLAN.md`. Nothing here edits a threshold - they are quoted from PLAN.md
as strings in `H_THRESHOLDS` and implemented in the units they are stated in.

H1, H2a, H3 and H4 have a confirm clause and a kill clause and are read on an
exam type, so the continuation/cloze split rule applies to them. **H2b is not a
pass/fail test**: it reports Arm D's measured GPU-time ratio to Arm B against
the <= 1.5x and > 2x bands the original H2 named, and it returns a
`Measurement`, which has no verdict field (see `MEASUREMENT_ONLY`).
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from training import metrics
from training.config import ALL_PHASES, EXAM_TYPES, HEADLINE_EXAM_TYPE, N_PHASES, SEEDS, resolve_phases

__all__ = [
    "NO_DIFFERENCE_RULE",
    "H_THRESHOLDS",
    "RunRecord",
    "Exclusion",
    "Aggregate",
    "Comparison",
    "Verdict",
    "Measurement",
    "VERDICT_HYPOTHESES",
    "MEASUREMENT_ONLY",
    "h2b",
    "load_run",
    "discover_runs",
    "group_by_arm",
    "aggregate",
    "compare",
    "arm_average_accuracy",
    "arm_average_forgetting",
    "arm_gpu_hours",
    "hypothesis_table",
    "split_hypothesis_table",
    "render_hypothesis_table",
    "render_exclusions",
    "plot_heatmaps",
    "plot_forgetting_curve",
    "write_report",
]

NO_DIFFERENCE_RULE = (
    "a difference counts only when |mean(a) - mean(b)| exceeds the seed spread, "
    "defined as max(sd(a), sd(b)) with sample sd (ddof=1) over seeds "
    + str(SEEDS)
    + "; otherwise it is reported as 'no difference'. The same quantity is the "
    "minimum difference this grid could have detected."
)

#: Quoted verbatim from PLAN.md's hypothesis table. Never edited here.
H_THRESHOLDS: dict[str, dict[str, str]] = {
    "H1": {
        "claim": "Forgetting is real at this scale",
        "confirmed_if": (
            "Arm A (sequential, no replay) loses >= 15% of its peak phase-0 exam "
            "score by the end of phase 6"
        ),
        "killed_if": (
            "Arm A loses < 5%: the phases are too similar or the exams too easy "
            "to test anything; fix the curriculum first"
        ),
    },
    "H2a": {
        "claim": "Consolidation beats plain replay on retention",
        "confirmed_if": (
            "Arm D's average forgetting <= Arm B's, and final average accuracy "
            "within 3 points of Arm E (joint oracle)"
        ),
        "killed_if": "Arm D forgets more than Arm B",
    },
    # H2b carries no confirmed_if / killed_if keys at all. That is deliberate:
    # any code that tried to read a verdict threshold for it raises KeyError
    # rather than inventing one.
    "H2b": {
        "claim": "What that retention costs",
        "reported_as": (
            "Not a pass/fail test. The measured GPU-time ratio of Arm D to Arm B, "
            "reported against the <= 1.5x and > 2x bands the original H2 named"
        ),
    },
    "H3": {
        "claim": "Distillation is needed, merging is not enough",
        "confirmed_if": "Arm D beats Arm C (arithmetic merge) by >= 5 points average accuracy",
        "killed_if": (
            "Arm C matches Arm D: reverse-LoRA is just W += BA and the "
            "distillation step is dropped"
        ),
    },
    "H4": {
        "claim": "Replay is the active ingredient",
        "confirmed_if": "Arm D without replay forgets clearly more than Arm D with replay",
        "killed_if": (
            "No difference: the consolidation alone preserves old phases, which "
            "would be a surprising and publishable result on its own"
        ),
    },
}

REQUIRED_FILES = ("matrix.json", "timings.json", "config.json", "commit.txt")


# --------------------------------------------------------------------------- #
# loading runs
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Exclusion:
    """A run the report refuses, and why. Printed by name, never dropped
    silently."""

    run_id: str
    reason: str

    def __str__(self) -> str:
        return self.run_id + ": " + self.reason


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    path: Path
    arm: str
    seed: int
    pilot: bool
    dirty: bool
    commit: str
    gpu_name: str
    gpu_hours: float
    matrix: dict[str, list[list[float]]]
    untrained: dict[str, list[float]]
    hashes: dict[str, str]
    #: This arm's tokens per Arm A's, as the run recorded it. Never recomputed.
    token_ratio: float | None
    config: dict
    #: The phases the run trained, by real id (config.json "phases"). All seven
    #: for a full run; an undeclared phase's cells are NaN in `matrix` and
    #: `untrained`, whatever the file held.
    phases: tuple[int, ...] = ALL_PHASES

    @property
    def subset(self) -> bool:
        return self.phases != ALL_PHASES

    @property
    def metric_phases(self) -> tuple[int, ...] | None:
        """What `metrics.*(phases=...)` takes: None for a full run, so it goes
        through the unchanged full-run code path."""
        return self.phases if self.subset else None


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


#: commit.txt's frozen keys (AGENTS.md, amendments 2026-09-22). All five must
#: be present; `dirty` must read exactly true or false.
COMMIT_REQUIRED_KEYS = ("commit", "dirty", "torch", "cuda", "driver")


def parse_commit_txt(text: str) -> dict[str, str]:
    """Parse `commit.txt`: one `key: value` per line, and nothing else.

    One format (frozen 2026-09-22). Blank lines and lines without a colon are
    ignored; every `key: value` line is kept, lower-cased on the key. This
    function does not judge the result - `check_commit_txt` does.
    """
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        out[key.strip().lower()] = value.strip()
    return out


def check_commit_txt(fields: Mapping[str, str]) -> tuple[bool, str | None]:
    """(dirty, reason_it_is_unusable).

    The dirty flag is never assumed. A `commit.txt` that is missing a required
    key, or whose `dirty` value is not exactly true or false, is unusable and
    its run is excluded and named - that is the one default that would let a
    dirty tree become a result (CLAUDE.md's reproduction rule).
    """
    missing = [k for k in COMMIT_REQUIRED_KEYS if not fields.get(k)]
    if missing:
        return True, "commit.txt is missing required key(s): " + ", ".join(missing)
    flag = str(fields["dirty"]).strip().lower()
    if flag not in ("true", "false"):
        return True, "commit.txt has an unparseable dirty flag " + repr(fields["dirty"])
    return flag == "true", None


def _run_id_parts(run_id: str) -> tuple[str | None, int | None]:
    """`{arm}_s{seed}_{commit7}_{utc}` -> (arm, seed), best effort."""
    parts = run_id.split("_")
    for i, part in enumerate(parts):
        if len(part) > 1 and part[0] == "s" and part[1:].isdigit():
            return "_".join(parts[:i]) or None, int(part[1:])
    return None, None


def _collect_hashes(config: Mapping) -> dict[str, str]:
    """config.json's one top-level `hashes` block, flattened.

    Frozen 2026-09-22: `{"manifest": ..., "train_files": {...}, "config": ...}`.
    Matched on that key alone - a substring search over arbitrary key names
    would one day match something that is not a hash and quietly split or merge
    a comparison.
    """
    block = config.get("hashes")
    if not isinstance(block, Mapping):
        return {}
    out: dict[str, str] = {}
    for key, value in sorted(block.items()):
        if isinstance(value, Mapping):
            for k2, v2 in sorted(value.items()):
                out[str(key) + "." + str(k2)] = str(v2)
        else:
            out[str(key)] = str(value)
    return out


def recorded_token_ratio(config: Mapping) -> float | None:
    """This arm's tokens per Arm A's, read from `config.json`, never recomputed.

    `token_budget` records the new-phase and replay totals separately, so the
    ratio is (new + replay) / new. It is 1.375 for Arm B, not PLAN.md's "about
    1.43x", because phase 0 carries no replay - which is exactly why this is
    read from the run rather than derived from the replay fraction.
    """
    budget = config.get("token_budget")
    if not isinstance(budget, Mapping):
        return None
    new = budget.get("total_new_phase_tokens")
    replay = budget.get("total_replay_tokens")
    if not isinstance(new, (int, float)) or not isinstance(replay, (int, float)) or not new:
        return None
    return (float(new) + float(replay)) / float(new)


def _is_number(v: object) -> bool:
    """A real, finite number. `None` (the run record's placeholder for "not
    scored yet") and bools are not."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return False
    return math.isfinite(v)


def _run_phases(config: Mapping) -> tuple[int, ...] | str:
    """The run's declared phases from config.json, or why they are unusable.
    Absent means all seven (every run written before `--phases` existed)."""
    if "phases" not in config:
        return ALL_PHASES
    raw = config["phases"]
    if not isinstance(raw, (list, tuple)):
        return 'config.json "phases" is not a list: ' + repr(raw)
    try:
        return resolve_phases(raw)
    except ValueError as exc:
        return 'config.json "phases" is invalid: ' + str(exc)


def _matrix_problem(rows: object, key: str, phases: tuple[int, ...] = ALL_PHASES) -> str | None:
    """Why this 7x7 block is unusable, or None.

    A Kaggle session dying part-way through is the *expected* failure, not an
    exotic one: `runrecord.empty_matrix` writes `null` in every cell and fills
    them in row by row, so a folder from a dead session has null rows at the
    bottom. Such a run is excluded and named - it never raises, because one
    unfinished run must not take the whole report down with it.

    In a subset run only the declared rows and, within them, the declared
    columns must be complete; an undeclared row is null by design and an
    undeclared column NaN.
    """
    where = "matrix.json[" + repr(key) + "]"
    if not isinstance(rows, (list, tuple)) or len(rows) != N_PHASES:
        return where + " is not 7 rows"
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) != N_PHASES:
            return where + " is not 7x7"
    incomplete = [
        i for i, row in enumerate(rows) if i in phases and not all(_is_number(row[j]) for j in phases)
    ]
    if incomplete:
        of = "0-6" if phases == ALL_PHASES else "the declared phases " + str(list(phases))
        return (
            where + " is incomplete: row(s) "
            + ", ".join(str(i) for i in incomplete)
            + " of " + of + " hold a null or non-finite cell (a run that did not finish - "
            "a dead session leaves the later rows unscored)"
        )
    return None


def _untrained_problem(values: object, key: str, phases: tuple[int, ...] = ALL_PHASES) -> str | None:
    where = "matrix.json untrained[" + repr(key) + "]"
    if not isinstance(values, (list, tuple)) or len(values) != N_PHASES:
        return where + " is not 7 long"
    missing = [j for j, v in enumerate(values) if j in phases and not _is_number(v)]
    if missing:
        return (
            where + " is incomplete: phase(s) "
            + ", ".join(str(j) for j in missing)
            + " hold a null or non-finite value; forward transfer cannot be "
            "computed without the untrained scores and they cannot be recovered later"
        )
    return None


def load_run(path: str | Path) -> RunRecord | Exclusion:
    """Load one `results/<run_id>/` folder, or say why it is refused."""
    path = Path(path)
    run_id = path.name
    for name in REQUIRED_FILES:
        if not (path / name).is_file():
            return Exclusion(run_id, "incomplete: " + name + " is missing")
    try:
        matrix_doc = _read_json(path / "matrix.json")
        timings = _read_json(path / "timings.json")
        config = _read_json(path / "config.json")
        commit_info = parse_commit_txt((path / "commit.txt").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return Exclusion(run_id, "unreadable: " + str(exc))

    dirty, commit_problem = check_commit_txt(commit_info)
    if commit_problem is not None:
        return Exclusion(run_id, commit_problem + " (the dirty flag is never assumed)")
    if dirty:
        return Exclusion(run_id, "dirty tree in commit.txt (a dirty tree is not a result)")

    phases = _run_phases(config)
    if isinstance(phases, str):
        return Exclusion(run_id, phases)

    M = matrix_doc.get("M")
    untrained = matrix_doc.get("untrained")
    if not isinstance(M, Mapping) or not isinstance(untrained, Mapping):
        return Exclusion(run_id, 'matrix.json lacks the "M" / "untrained" blocks')
    for exam_type in metrics.STORED_EXAM_KEYS:
        if exam_type not in M or exam_type not in untrained:
            return Exclusion(run_id, "matrix.json is missing key " + repr(exam_type))
        problem = _matrix_problem(M[exam_type], exam_type, phases)
        if problem is not None:
            return Exclusion(run_id, problem)
        problem = _untrained_problem(untrained[exam_type], exam_type, phases)
        if problem is not None:
            return Exclusion(run_id, problem)

    gpu_name = str(timings.get("gpu_name", "")).strip()
    if not gpu_name:
        return Exclusion(run_id, "timings.json has no gpu_name (H2 is stated in GPU time)")
    if "gpu_hours" not in timings:
        return Exclusion(run_id, "timings.json has no gpu_hours")

    hashes = _collect_hashes(config)
    if not hashes:
        return Exclusion(
            run_id,
            'config.json has no top-level "hashes" block; the run cannot be '
            "checked against the others in its comparison",
        )

    fallback_arm, fallback_seed = _run_id_parts(run_id)
    arm = str(config.get("arm", fallback_arm or "?"))
    seed = int(config.get("seed", fallback_seed if fallback_seed is not None else -1))

    return RunRecord(
        run_id=run_id,
        path=path,
        arm=arm,
        seed=seed,
        pilot=bool(config.get("pilot", False)),
        dirty=False,
        commit=str(commit_info.get("commit", commit_info.get("sha", ""))),
        gpu_name=gpu_name,
        gpu_hours=float(timings["gpu_hours"]),
        matrix={
            k: [
                [float(v) if (i in phases and j in phases) else math.nan for j, v in enumerate(row)]
                for i, row in enumerate(M[k])
            ]
            for k in metrics.STORED_EXAM_KEYS
        },
        untrained={
            k: [float(v) if j in phases else math.nan for j, v in enumerate(untrained[k])]
            for k in metrics.STORED_EXAM_KEYS
        },
        hashes=hashes,
        token_ratio=recorded_token_ratio(config),
        config=dict(config),
        phases=phases,
    )


def discover_runs(
    results_dir: str | Path, *, pilot: bool = False
) -> tuple[list[RunRecord], list[Exclusion]]:
    """Every run folder under `results_dir`, split into kept and refused.

    `pilot=False` (a grid report) excludes every folder whose config.json says
    `"pilot": true`, by name; `pilot=True` excludes the grid runs instead. The
    two are never mixed (frozen contract).
    """
    results_dir = Path(results_dir)
    kept: list[RunRecord] = []
    refused: list[Exclusion] = []
    for child in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        loaded = load_run(child)
        if isinstance(loaded, Exclusion):
            refused.append(loaded)
            continue
        if loaded.pilot != pilot:
            kind = "pilot" if loaded.pilot else "grid"
            want = "pilot" if pilot else "grid"
            refused.append(
                Exclusion(loaded.run_id, kind + " run excluded from a " + want + " report")
            )
            continue
        kept.append(loaded)
    return kept, refused


def group_by_arm(runs: Iterable[RunRecord]) -> dict[str, list[RunRecord]]:
    out: dict[str, list[RunRecord]] = {}
    for run in runs:
        out.setdefault(run.arm, []).append(run)
    for runs_for_arm in out.values():
        runs_for_arm.sort(key=lambda r: r.seed)
    return out


def filter_same_phases(
    runs: Sequence[RunRecord],
) -> tuple[list[RunRecord], list[Exclusion]]:
    """Keep the runs that declare the majority's phase list; refuse the rest
    by name. A 0/3/6 run's average accuracy is over three phases and a full
    run's over seven: comparing them would compare two different numbers."""
    if not runs:
        return [], []
    majority = statistics.mode([r.phases for r in runs])
    kept, refused = [], []
    for run in runs:
        if run.phases == majority:
            kept.append(run)
        else:
            refused.append(
                Exclusion(
                    run.run_id,
                    "declares phases " + str(list(run.phases)) + " but the rest of this report "
                    "declares " + str(list(majority)) + "; runs over different phases are not compared",
                )
            )
    return kept, refused


def filter_consistent(
    runs: Sequence[RunRecord],
) -> tuple[list[RunRecord], list[Exclusion]]:
    """Keep only runs whose hashes match the majority; refuse the rest by name.

    A run whose data or manifest hash differs is not comparable with the
    others, whatever else it has in common.
    """
    if not runs:
        return [], []
    keys = [json.dumps(r.hashes, sort_keys=True) for r in runs]
    majority = statistics.mode(keys)
    kept, refused = [], []
    for run, key in zip(runs, keys):
        if key == majority:
            kept.append(run)
        else:
            refused.append(
                Exclusion(run.run_id, "config/manifest hashes differ from the rest of its comparison")
            )
    return kept, refused


# --------------------------------------------------------------------------- #
# seed statistics
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Aggregate:
    """A number over seeds: never a mean without its spread, n and values."""

    label: str
    values: list[float]
    seeds: list[int]

    @property
    def n(self) -> int:
        return len(self.values)

    @property
    def mean(self) -> float:
        return sum(self.values) / len(self.values)

    @property
    def sd(self) -> float:
        """Sample standard deviation (ddof=1). 0.0 when n < 2, and n is printed
        beside it so a single-seed 0.0 cannot be mistaken for agreement."""
        if len(self.values) < 2:
            return 0.0
        return statistics.stdev(self.values)

    def __str__(self) -> str:
        vals = ", ".join(format(v, ".4f") for v in self.values)
        return (
            self.label
            + " = "
            + format(self.mean, ".4f")
            + " +/- "
            + format(self.sd, ".4f")
            + " (n="
            + str(self.n)
            + "; seeds "
            + ",".join(str(s) for s in self.seeds)
            + " -> "
            + vals
            + ")"
        )


def aggregate(label: str, runs: Sequence[RunRecord], fn) -> Aggregate:
    return Aggregate(
        label=label,
        values=[float(fn(r)) for r in runs],
        seeds=[r.seed for r in runs],
    )


@dataclass(frozen=True)
class Comparison:
    """a - b, with the seed-spread rule applied. See NO_DIFFERENCE_RULE."""

    a: Aggregate
    b: Aggregate

    @property
    def diff(self) -> float:
        return self.a.mean - self.b.mean

    @property
    def spread(self) -> float:
        """The seed spread: the larger of the two sample sds."""
        return max(self.a.sd, self.b.sd)

    @property
    def decided(self) -> bool:
        return abs(self.diff) > self.spread

    @property
    def mde(self) -> float:
        """Minimum difference this grid could have detected: the spread."""
        return self.spread

    def __str__(self) -> str:
        head = (
            self.a.label
            + " - "
            + self.b.label
            + " = "
            + format(self.diff, ".4f")
            + " (seed spread "
            + format(self.spread, ".4f")
            + ", MDE "
            + format(self.mde, ".4f")
            + ")"
        )
        return head + ("" if self.decided else "  -> no difference")


def compare(a: Aggregate, b: Aggregate) -> Comparison:
    return Comparison(a=a, b=b)


# --------------------------------------------------------------------------- #
# per-arm quantities
# --------------------------------------------------------------------------- #


def arm_average_accuracy(runs: Sequence[RunRecord], exam_type: str) -> Aggregate:
    return aggregate(
        "avg accuracy [" + exam_type + "]",
        runs,
        lambda r: metrics.average_accuracy(r.matrix[exam_type], exam_type, phases=r.metric_phases),
    )


def arm_average_forgetting(runs: Sequence[RunRecord], exam_type: str) -> Aggregate:
    return aggregate(
        "avg forgetting [" + exam_type + "]",
        runs,
        lambda r: metrics.average_forgetting(r.matrix[exam_type], exam_type, phases=r.metric_phases),
    )


def arm_fraction_of_peak_lost(
    runs: Sequence[RunRecord], exam_type: str, phase: int = 0
) -> Aggregate:
    return aggregate(
        "fraction of peak phase-" + str(phase) + " score lost [" + exam_type + "]",
        runs,
        lambda r: metrics.fraction_of_peak_lost(
            r.matrix[exam_type], exam_type, phase, phases=r.metric_phases
        ),
    )


def arm_gpu_hours(runs: Sequence[RunRecord]) -> Aggregate:
    """GPU hours over seeds. Raises if the runs are not all on one GPU model:
    mixing a T4 and an A100 is an error, not a footnote."""
    names = sorted({r.gpu_name for r in runs})
    if len(names) > 1:
        raise ValueError(
            "GPU hours compared across different GPU models: " + ", ".join(names)
        )
    return aggregate("gpu hours", runs, lambda r: r.gpu_hours)


# --------------------------------------------------------------------------- #
# the hypothesis table
# --------------------------------------------------------------------------- #

CONFIRMED = "confirmed"
KILLED = "killed"
NEITHER = "neither"
UNDETERMINED = "undetermined"
SPLIT = "split"

#: The hypotheses that have a confirm and a kill clause and are read on an exam
#: type. H2b is not one of them and is not in this tuple.
VERDICT_HYPOTHESES = ("H1", "H2a", "H3", "H4")

#: Reported as a measurement only. Split into H2a/H2b on 2026-09-22 (owner's
#: decision, before any run): the consolidation recipe is a 1-unit LoRA day plus
#: a night of two frozen teacher forwards and a student step, which PLAN.md's
#: Budget paragraph puts at 2.5-3x Arm A. With Arm B measured at 1.375x Arm A,
#: Arm D is 1.8-2.2x Arm B *by construction*, so the old "<= 1.5x" confirm
#: clause was unreachable by any faithful implementation. No threshold moved;
#: the compute clause stopped being a pass/fail gate on a retention claim it
#: could not fairly gate. The bands are still reported.
MEASUREMENT_ONLY = ("H2b",)

#: The bands the original H2 named, kept as reporting labels only.
H2B_CHEAP_BAND = 1.5
H2B_EXPENSIVE_BAND = 2.0


@dataclass
class Verdict:
    name: str
    exam_type: str
    verdict: str
    measured: str
    confirmed_if: str
    killed_if: str
    detail: list[str] = field(default_factory=list)


@dataclass
class Measurement:
    """A reported number with no verdict.

    This class has no `verdict` field and there is no code path that gives it
    one: H2b is not a pass/fail test, so it must be structurally unable to emit
    confirmed, killed, neither or split.
    """

    name: str
    claim: str
    reported_as: str
    measured: str
    band: str
    value: Aggregate | None
    detail: list[str] = field(default_factory=list)


def _missing(name: str, exam_type: str, arms: Sequence[str]) -> Verdict:
    return Verdict(
        name=name,
        exam_type=exam_type,
        verdict=UNDETERMINED,
        measured="no runs for arm(s) " + ", ".join(arms),
        confirmed_if=H_THRESHOLDS[name]["confirmed_if"],
        killed_if=H_THRESHOLDS[name]["killed_if"],
    )


def h1(arms: Mapping[str, Sequence[RunRecord]], exam_type: str) -> Verdict:
    """Arm A loses >= 15% of its peak phase-0 score -> confirmed; < 5% -> killed."""
    if not arms.get("A"):
        return _missing("H1", exam_type, ["A"])
    lost = arm_fraction_of_peak_lost(arms["A"], exam_type, phase=0)
    if lost.mean >= 0.15:
        verdict = CONFIRMED
    elif lost.mean < 0.05:
        verdict = KILLED
    else:
        verdict = NEITHER
    return Verdict(
        name="H1",
        exam_type=exam_type,
        verdict=verdict,
        measured="Arm A lost " + format(lost.mean * 100.0, ".1f") + "% of its peak phase-0 score",
        confirmed_if=H_THRESHOLDS["H1"]["confirmed_if"],
        killed_if=H_THRESHOLDS["H1"]["killed_if"],
        detail=[str(lost)],
    )


def h2a(arms: Mapping[str, Sequence[RunRecord]], exam_type: str) -> Verdict:
    """Retention only: D's forgetting <= B's, and D within 3 points of E.

    The compute clause moved to `h2b`, which is a measurement and not a gate.
    """
    need = [a for a in ("B", "D", "E") if not arms.get(a)]
    if need:
        return _missing("H2a", exam_type, need)
    f_d = arm_average_forgetting(arms["D"], exam_type)
    f_b = arm_average_forgetting(arms["B"], exam_type)
    forget = compare(f_d, f_b)
    acc_d = arm_average_accuracy(arms["D"], exam_type)
    acc_e = arm_average_accuracy(arms["E"], exam_type)
    gap_points = metrics.to_points(acc_e.mean - acc_d.mean)

    forgets_more = forget.diff > 0 and forget.decided
    within_3 = gap_points <= 3.0

    if (not forgets_more) and within_3:
        verdict = CONFIRMED
    elif forgets_more:
        verdict = KILLED
    else:
        verdict = NEITHER

    # A reader who sees "H2a CONFIRMED" beside a positive forgetting difference
    # is owed the reason in the same line: the rule is that a difference smaller
    # than the seed spread is not a difference, so a raw number that looks like
    # Arm D losing can still satisfy "D's forgetting <= B's". Spell it out rather
    # than leaving it to be reconciled against the rule at the top of the report.
    if forget.decided:
        forget_text = "D forgetting - B forgetting = " + format(forget.diff, "+.4f")
    else:
        forget_text = (
            "D forgetting - B forgetting = "
            + format(forget.diff, "+.4f")
            + ", which is within the seed spread of "
            + format(forget.spread, ".4f")
            + " and is therefore NOT a difference"
            + (
                " (Arm D's raw mean is the higher of the two, but not by enough "
                "for this grid to tell them apart)"
                if forget.diff > 0
                else ""
            )
        )
    return Verdict(
        name="H2a",
        exam_type=exam_type,
        verdict=verdict,
        measured=(
            forget_text
            + "; E - D final avg accuracy = "
            + format(gap_points, ".2f")
            + " points"
        ),
        confirmed_if=H_THRESHOLDS["H2a"]["confirmed_if"],
        killed_if=H_THRESHOLDS["H2a"]["killed_if"],
        detail=[str(f_d), str(f_b), str(forget), str(acc_d), str(acc_e)],
    )


def _h2b_band(ratio: float) -> str:
    """The band label. Reporting only - none of these is a verdict."""
    if ratio <= H2B_CHEAP_BAND:
        return (
            "within the <= " + format(H2B_CHEAP_BAND, ".1f")
            + "x band the original H2 named"
        )
    if ratio <= H2B_EXPENSIVE_BAND:
        return (
            "between the " + format(H2B_CHEAP_BAND, ".1f") + "x and "
            + format(H2B_EXPENSIVE_BAND, ".0f")
            + "x bands the original H2 named"
        )
    return (
        "above the " + format(H2B_EXPENSIVE_BAND, ".0f")
        + "x band the original H2 named"
    )


def h2b(arms: Mapping[str, Sequence[RunRecord]]) -> Measurement:
    """Arm D's GPU time as a multiple of Arm B's. A number, not a verdict.

    The ratio is formed per seed and then averaged, so it carries a real seed
    spread rather than a ratio of two means. `arm_gpu_hours` still raises if D
    and B were timed on different GPU models: mixing a T4 and an A100 is an
    error, not a footnote, and that applies to H2b exactly as it did to H2.

    This function returns a `Measurement`, which has no verdict field. H2b is
    not read on an exam type either, so the continuation/cloze split rule does
    not apply to it.
    """
    need = [a for a in ("B", "D") if not arms.get(a)]
    if need:
        return Measurement(
            name="H2b",
            claim=H_THRESHOLDS["H2b"]["claim"],
            reported_as=H_THRESHOLDS["H2b"]["reported_as"],
            measured="no runs for arm(s) " + ", ".join(need),
            band="not measurable",
            value=None,
        )
    # Raises if the two arms were not timed on the same GPU model.
    arm_gpu_hours(list(arms["D"]) + list(arms["B"]))
    by_seed_b = {r.seed: r.gpu_hours for r in arms["B"]}
    common = sorted(r.seed for r in arms["D"] if r.seed in by_seed_b and by_seed_b[r.seed])
    if not common:
        return Measurement(
            name="H2b",
            claim=H_THRESHOLDS["H2b"]["claim"],
            reported_as=H_THRESHOLDS["H2b"]["reported_as"],
            measured="arms D and B share no seed with a non-zero GPU time",
            band="not measurable",
            value=None,
        )
    by_seed_d = {r.seed: r.gpu_hours for r in arms["D"]}
    ratios = Aggregate(
        label="GPU time D/B",
        values=[by_seed_d[s] / by_seed_b[s] for s in common],
        seeds=list(common),
    )
    return Measurement(
        name="H2b",
        claim=H_THRESHOLDS["H2b"]["claim"],
        reported_as=H_THRESHOLDS["H2b"]["reported_as"],
        measured=format(ratios.mean, ".2f") + "x +/- " + format(ratios.sd, ".2f")
        + " (n=" + str(ratios.n) + ")",
        band=_h2b_band(ratios.mean),
        value=ratios,
        detail=[str(ratios), str(arm_gpu_hours(arms["D"])), str(arm_gpu_hours(arms["B"]))],
    )


def h3(arms: Mapping[str, Sequence[RunRecord]], exam_type: str) -> Verdict:
    """D beats C by >= 5 points average accuracy -> confirmed; C matches D -> killed."""
    need = [a for a in ("C", "D") if not arms.get(a)]
    if need:
        return _missing("H3", exam_type, need)
    acc_d = arm_average_accuracy(arms["D"], exam_type)
    acc_c = arm_average_accuracy(arms["C"], exam_type)
    cmp_ = compare(acc_d, acc_c)
    diff_points = metrics.to_points(cmp_.diff)
    if cmp_.decided and diff_points >= 5.0:
        verdict = CONFIRMED
    elif not cmp_.decided:
        verdict = KILLED  # "Arm C matches Arm D"
    else:
        verdict = NEITHER
    return Verdict(
        name="H3",
        exam_type=exam_type,
        verdict=verdict,
        measured=(
            "D - C avg accuracy = "
            + format(diff_points, ".2f")
            + " points"
            + ("" if cmp_.decided else " (no difference)")
        ),
        confirmed_if=H_THRESHOLDS["H3"]["confirmed_if"],
        killed_if=H_THRESHOLDS["H3"]["killed_if"],
        detail=[str(acc_d), str(acc_c), str(cmp_)],
    )


def h4(arms: Mapping[str, Sequence[RunRecord]], exam_type: str) -> Verdict:
    """D-nr forgets clearly more than D -> confirmed; no difference -> killed."""
    need = [a for a in ("D", "D-nr") if not arms.get(a)]
    if need:
        return _missing("H4", exam_type, need)
    f_nr = arm_average_forgetting(arms["D-nr"], exam_type)
    f_d = arm_average_forgetting(arms["D"], exam_type)
    cmp_ = compare(f_nr, f_d)
    if cmp_.decided and cmp_.diff > 0:
        verdict = CONFIRMED
    elif not cmp_.decided:
        verdict = KILLED  # "No difference"
    else:
        verdict = NEITHER
    return Verdict(
        name="H4",
        exam_type=exam_type,
        verdict=verdict,
        measured=(
            "D-nr - D avg forgetting = "
            + format(cmp_.diff, ".4f")
            + ("" if cmp_.decided else " (no difference)")
        ),
        confirmed_if=H_THRESHOLDS["H4"]["confirmed_if"],
        killed_if=H_THRESHOLDS["H4"]["killed_if"],
        detail=[str(f_nr), str(f_d), str(cmp_)],
    )


def hypothesis_table(
    arms: Mapping[str, Sequence[RunRecord]], exam_type: str = HEADLINE_EXAM_TYPE
) -> list[Verdict]:
    """H1-H4 on one exam type.

    `perplexity` and `continuation_summed` are refused: the frozen contract
    stores both and headlines neither, so neither may decide a hypothesis.
    """
    if exam_type not in metrics.VERDICT_EXAM_TYPES:
        raise ValueError(
            repr(exam_type) + " is never a verdict metric (frozen contract); "
            "a verdict is read on one of " + repr(metrics.VERDICT_EXAM_TYPES)
        )
    return [h1(arms, exam_type), h2a(arms, exam_type), h3(arms, exam_type), h4(arms, exam_type)]


def split_hypothesis_table(
    arms: Mapping[str, Sequence[RunRecord]],
) -> list[tuple[str, str, Verdict, Verdict]]:
    """The headline table beside the cloze table.

    Returns (name, combined_verdict, headline_verdict, cloze_verdict) for the
    four `VERDICT_HYPOTHESES`. The combined verdict is the shared one where the
    two agree and `"split"` where they do not - neither is dropped and neither
    is a tiebreaker. H2b is absent: it is not read on an exam type, so it has
    nothing to disagree about.
    """
    head = {v.name: v for v in hypothesis_table(arms, HEADLINE_EXAM_TYPE)}
    other = {v.name: v for v in hypothesis_table(arms, "cloze")}
    rows = []
    for name in VERDICT_HYPOTHESES:
        a, b = head[name], other[name]
        combined = a.verdict if a.verdict == b.verdict else SPLIT
        rows.append((name, combined, a, b))
    return rows


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #


def render_exclusions(exclusions: Sequence[Exclusion]) -> str:
    if not exclusions:
        return "Excluded runs: none."
    lines = ["Excluded runs (" + str(len(exclusions)) + "), by name:"]
    lines.extend("  - " + str(e) for e in exclusions)
    return "\n".join(lines)


def render_hypothesis_table(arms: Mapping[str, Sequence[RunRecord]]) -> str:
    """The H1-H4 table: threshold quoted from PLAN.md, measured number, verdict.

    Both exam types are printed at every step, and a disagreement is printed as
    `split` with both verdicts kept.
    """
    lines = [
        "Hypotheses (headline exam type: " + HEADLINE_EXAM_TYPE + "; cloze reported beside it)",
        "Rule: " + NO_DIFFERENCE_RULE,
        "",
    ]
    rows = {name: (combined, head, other) for name, combined, head, other
            in split_hypothesis_table(arms)}
    for name in ("H1", "H2a", "H2b", "H3", "H4"):
        if name in MEASUREMENT_ONLY:
            lines.extend(_render_measurement(h2b(arms)))
            continue
        combined, head, other = rows[name]
        lines.append(name + " " + H_THRESHOLDS[name]["claim"] + " -> " + combined.upper())
        lines.append("    confirmed if: " + H_THRESHOLDS[name]["confirmed_if"])
        lines.append("    killed if:    " + H_THRESHOLDS[name]["killed_if"])
        lines.append("    " + HEADLINE_EXAM_TYPE + ": " + head.verdict + " - " + head.measured)
        lines.append("    cloze:        " + other.verdict + " - " + other.measured)
        for d in head.detail:
            lines.append("      . " + d)
        lines.append("")
    return "\n".join(lines)


def _render_measurement(m: Measurement) -> list[str]:
    """A measured row: a number and a band, and no verdict column at all."""
    lines = [
        m.name + " " + m.claim + " -> " + m.measured + " (" + m.band + ")",
        "    reported as:  " + m.reported_as,
        "    not a pass/fail test: no confirmed / killed / split verdict",
    ]
    lines.extend("      . " + d for d in m.detail)
    lines.append("")
    return lines


# --------------------------------------------------------------------------- #
# plots
# --------------------------------------------------------------------------- #


def _pyplot():
    """matplotlib with the Agg backend, imported lazily so the aggregation
    side of this module works headless and without matplotlib installed."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _mean_matrix(runs: Sequence[RunRecord], exam_type: str) -> list[list[float]]:
    n = len(runs)
    return [
        [sum(r.matrix[exam_type][i][j] for r in runs) / n for j in range(N_PHASES)]
        for i in range(N_PHASES)
    ]


def plot_heatmaps(
    arms: Mapping[str, Sequence[RunRecord]], exam_type: str, out_dir: str | Path
) -> Path:
    """One 7x7 heatmap per arm, seed-averaged, on one shared colour scale."""
    plt = _pyplot()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = [a for a in sorted(arms) if arms[a]]
    mats = {a: _mean_matrix(arms[a], exam_type) for a in names}
    # A subset run's undeclared cells are NaN and draw as blank; they never
    # set the colour scale.
    flat = [v for m in mats.values() for row in m for v in row if math.isfinite(v)]
    vmin, vmax = (min(flat), max(flat)) if flat else (0.0, 1.0)
    fig, axes = plt.subplots(1, max(1, len(names)), figsize=(3.0 * max(1, len(names)), 3.4))
    axes = [axes] if len(names) <= 1 else list(axes)
    image = None
    for ax, name in zip(axes, names):
        image = ax.imshow(mats[name], vmin=vmin, vmax=vmax, cmap="viridis")
        ax.set_title(name + " (n=" + str(len(arms[name])) + ")")
        ax.set_xlabel("exam phase j")
        ax.set_ylabel("after phase i")
        ax.set_xticks(range(N_PHASES))
        ax.set_yticks(range(N_PHASES))
    fig.suptitle(exam_type + ": M[i][j], mean over seeds")
    if image is not None:
        fig.colorbar(image, ax=axes, shrink=0.8)
    path = out_dir / ("heatmap_" + exam_type + ".png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_forgetting_curve(
    arms: Mapping[str, Sequence[RunRecord]], exam_type: str, out_dir: str | Path
) -> Path:
    """Phase-0 exam score after each phase: one line per arm, seed spread as a
    band."""
    plt = _pyplot()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for name in sorted(arms):
        runs = arms[name]
        if not runs:
            continue
        # Real phase ids on the x axis: a 0/3/6 run plots at 0, 3 and 6.
        xs = list(runs[0].phases)
        cols = [[r.matrix[exam_type][i][0] for r in runs] for i in xs]
        means = [sum(c) / len(c) for c in cols]
        sds = [statistics.stdev(c) if len(c) > 1 else 0.0 for c in cols]
        ax.plot(xs, means, marker="o", label=name + " (n=" + str(len(runs)) + ")")
        ax.fill_between(
            xs,
            [m - s for m, s in zip(means, sds)],
            [m + s for m, s in zip(means, sds)],
            alpha=0.15,
        )
    if exam_type == "continuation":
        ax.axhline(0.25, linestyle="--", linewidth=1, color="grey")
    ax.set_xlabel("after phase i")
    ax.set_ylabel("phase-0 " + exam_type)
    ax.set_title("Forgetting curve: phase 0, " + exam_type + " (band = seed sd)")
    ax.legend(fontsize=8)
    path = out_dir / ("forgetting_" + exam_type + ".png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #


def write_report(
    results_dir: str | Path,
    out_dir: str | Path,
    *,
    pilot: bool = False,
    plots: bool = True,
) -> Path:
    """Discover runs, refuse the bad ones by name, aggregate, plot, write
    `report.txt`. Returns the path of the report."""
    runs, exclusions = discover_runs(results_dir, pilot=pilot)
    runs, refused_phases = filter_same_phases(runs)
    exclusions.extend(refused_phases)
    by_arm = group_by_arm(runs)
    consistent: dict[str, list[RunRecord]] = {}
    for arm, arm_runs in by_arm.items():
        kept, refused = filter_consistent(arm_runs)
        consistent[arm] = kept
        exclusions.extend(refused)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        ("Lifespan " + ("pilot" if pilot else "grid") + " report"),
        "results dir: " + str(Path(results_dir)),
        "",
    ]
    if runs and runs[0].subset:
        lines += [
            "phases: " + str(list(runs[0].phases)) + " (subset run: every number below is over "
            "these phases only, by real id; the others were never trained or examined)",
            "",
        ]
    lines += [
        render_exclusions(exclusions),
        "",
    ]
    for arm in sorted(consistent):
        arm_runs = consistent[arm]
        if not arm_runs:
            continue
        lines.append("Arm " + arm + " (" + str(len(arm_runs)) + " runs)")
        for exam_type in metrics.STORED_EXAM_KEYS:
            suffix = "  [stored, never headlined]" if exam_type in metrics.NEVER_HEADLINE else ""
            if exam_type == "perplexity":
                lines.append("    " + str(arm_average_forgetting(arm_runs, exam_type)) + suffix)
                continue
            lines.append("    " + str(arm_average_accuracy(arm_runs, exam_type)) + suffix)
            lines.append("    " + str(arm_average_forgetting(arm_runs, exam_type)) + suffix)
        lines.append("    chance: continuation 0.2500, cloze 0.0500 (20 candidates)")
        ratios = sorted({r.token_ratio for r in arm_runs if r.token_ratio is not None})
        if ratios:
            lines.append(
                "    token ratio vs Arm A (recorded in config.json, not recomputed): "
                + ", ".join(format(r, ".3f") for r in ratios)
            )
        else:
            lines.append("    token ratio vs Arm A: not recorded in config.json")
        lines.append("")
    lines.append(render_hypothesis_table(consistent))

    if plots:
        for exam_type in EXAM_TYPES:
            lines.append("plot: " + str(plot_heatmaps(consistent, exam_type, out_dir)))
            lines.append("plot: " + str(plot_forgetting_curve(consistent, exam_type, out_dir)))

    path = out_dir / "report.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Aggregate Lifespan runs into a report.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-dir", default="results/_report")
    parser.add_argument("--pilot", action="store_true", help="report the pilot runs instead")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    path = write_report(
        args.results_dir, args.out_dir, pilot=args.pilot, plots=not args.no_plots
    )
    print(path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
