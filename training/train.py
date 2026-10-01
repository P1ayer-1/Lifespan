"""One script; the arms are data.

    python -m training.train --arm {phase0,A,B,C,D,D-nr,E} --seed {0,1,2}
        --train-dir <dir> --exam-dir <dir> --out results/<run_id> [--resume] [--pilot]

Additive flags (none changes a hyperparameter): `--manifest`, `--phases`
(subset-phase mode), `--experiment-id` (default `config.EXPERIMENT_ID`; the
guard refuses a run whose id is not the manifest's, and a missing one).

`--arm` selects a row of `training.config.ARMS` and nothing else in this file
branches on an arm's name. What a row can say is: how much replay (`A` 0.0 vs
`B` 0.3), whether it trains LoRA, which `after_phase` hook runs, whether it
trains jointly, and whether it inherits the shared phase-0 checkpoint. Arms A
and B are therefore the same code path with a different number in one field,
which is what `PLAN.md` demands ("implement them as one script with a flag so
they cannot drift").

Order of operations, and the first one is not negotiable:

1. `guard.check(...)` -- before a tokenizer, a dataset or a model exists --
   with the run's experiment id; then the exam stories on disk are confirmed
   against the manifest's per-story hashes (`verify_exam_dir`), and the scorer
   is bound to the same hashes.
2. Resolve the config; seed; build or load the per-seed random init, so every
   arm at a given seed starts from byte-identical weights.
3. Sequential arms other than `phase0` load `phase0_s{seed}.pt` and adopt its
   matrix row, its untrained row and its wall clock, then start at phase 1.
   Arm E trains from the random init in seven equal segments of the shuffled
   corpus, totalling Arm A's tokens.
4. Per phase: train, `after_phase`, `evaluate_all`, write the row, checkpoint.

Two things `PLAN.md` left open and this file resolves, both reported to the lead:

- **A fresh AdamW per phase.** The schedule restarts at each phase by the
  plan's own table; `after_phase` may hand back a different module (arm C
  merges, arm D distils), so optimiser state cannot always survive a boundary;
  and the shared phase-0 checkpoint then needs to carry weights only, which
  makes "every forgetting curve starts from the same model" literally true.
- **Arm E is seven segments.** The matrix is 7x7 for every arm, so Arm E is
  evaluated seven times; segment `i` takes exactly the step count phase `i`
  would have taken, so E's total is Arm A's total, drawn from all phases
  shuffled together.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Callable, Iterable

import torch

from training import guard, hooks
from training.checkpoint import (
    Fingerprint,
    init_path,
    load_shared,
    load_state,
    phase0_path,
    require_compatible,
    save_shared,
    save_state,
    seed_everything,
    state_path,
    restore_rng_state,
)
from training.config import (
    ALL_PHASES,
    ARMS,
    EXAM_TYPES,
    EXPERIMENT_ID,
    N_PHASES,
    PILOT_OVERRIDES,
    REAL,
    TOY,
    TOY_RUN,
    Arm,
    ModelConfig,
    TrainConfig,
    replay_sequences_per_batch,
    resolve_phases,
    warmup_for,
)
from training.data import DataModule, concat_batches, make_generator, shift_for_lm
from training.model import build_model, cosine_lr
from training.runrecord import (
    MATRIX_KEYS,
    RunRecord,
    Timings,
    empty_matrix,
    env_info,
    get_row,
    git_info,
    make_run_id,
    set_row,
    set_untrained,
)
from training.tokenizer import get_or_train_tokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "exams" / "manifest.json"

#: The helper run that produces the shared phase-0 checkpoint. It is not an
#: experimental arm -- no hypothesis is stated about it, and it is the only
#: place in this file where a row of ARMS is identified by name, because
#: "which run writes the shared file" is not a property of a regime.
PHASE0_ARM = "phase0"

#: Re-exported from `training.config`, where AGENTS.md's contract puts it, so
#: `from training.train import TOY_RUN` keeps working for the fixtures. One
#: definition, in the file the contract names.
__all__ = ["TOY_RUN", "build_parser", "main", "parse_phases", "run"]


#: Passes over a phase's material, by `arm.after_phase` mode -- (new, replay).
#: Not derivable from `replay_fraction`, and not the same as `uses_lora`: arm C
#: trains a LoRA and then merges arithmetically, and the merge trains nothing,
#: so C makes one pass like a full fine-tune arm. Arms D and D-nr make two --
#: step 1 on phase data alone, step 3 on phase data plus replay -- which is the
#: whole of why they cost what H2b says they cost.
PHASE_PASSES: dict[str, tuple[int, int]] = {
    "identity": (1, 1),
    "merge": (1, 1),
    "distill": (2, 1),
}


def produces_phase0(arm: Arm) -> bool:
    return arm.name == PHASE0_ARM


def inherits_phase0(arm: Arm) -> bool:
    return arm.shares_phase0 and not produces_phase0(arm)


AfterPhaseFn = Callable[[torch.nn.Module, int, hooks.PhaseContext], torch.nn.Module]
EvaluateAllFn = Callable[[torch.nn.Module, Path, Iterable[int]], dict]


# ---------------------------------------------------------------------------
# hook resolution -- lazy, so this package imports while lora.py / consolidate.py
# / evaluate.py are still being written by the other agents.


def resolve_after_phase(arm: Arm) -> AfterPhaseFn:
    """Table lookup on `arm.after_phase`, never on `arm.name`."""
    mode = arm.after_phase
    if mode == "identity":
        return hooks.identity_after_phase
    mod = importlib.import_module("training.consolidate")
    factory = getattr(mod, "make_after_phase", None)
    if callable(factory):
        return factory(mode)
    fn = getattr(mod, f"{mode}_after_phase", None)
    if callable(fn):
        return fn
    raise ImportError(
        f"training.consolidate provides neither make_after_phase({mode!r}) nor {mode}_after_phase; "
        "the frozen hook is after_phase(model, phase_k, ctx) -> model"
    )


def resolve_evaluate_all() -> EvaluateAllFn:
    """The scorer the run record needs: all four `MATRIX_KEYS`.

    `evaluate.evaluate_all` is the *frozen hook*, and the frozen hook returns
    the three EXAM_TYPES. `matrix.json` requires a fourth,
    `continuation_summed`, which `EvalResult.as_matrix_row()` already produces
    -- so the run goes through `evaluate_all_detailed`. Wiring the three-key
    hook here would kill every arm at the first `set_untrained`, which is
    exactly what it did until the audit found it (2026-09-22).
    """
    mod = importlib.import_module("training.evaluate")
    detailed = getattr(mod, "evaluate_all_detailed", None)
    if not callable(detailed):
        raise ImportError(
            "training.evaluate has no evaluate_all_detailed(model, exam_dir, phases); "
            f"matrix.json needs all of {list(MATRIX_KEYS)} and the 3-key hook cannot supply them"
        )

    # `tokenizer` and `cfg` are named, not **kw: `bind_tokenizer` looks for a
    # parameter called `tokenizer`, and a **kw wrapper would hide it, leaving
    # every score computed with a byte tokenizer the model never saw.
    # `story_hashes` likewise: it is how the scorer is tied to the manifest.
    def evaluate_all_for_matrix(model, exam_dir, phases, *, tokenizer=None, cfg=None, story_hashes=None):
        return detailed(
            model, exam_dir, phases, tokenizer=tokenizer, cfg=cfg, story_hashes=story_hashes
        ).as_matrix_row()

    return evaluate_all_for_matrix


def verify_exam_dir(exam_dir: Path, story_hashes: dict[str, int], phases) -> dict[int, int]:
    """The exam stories on disk must be the manifest's, phase by phase.

    `evaluate.verify_exam_stories` does the checking (the evaluator owns the
    exam file format); `run` calls this once, right after the guard and before
    a tokenizer or a model exists, so a swapped or edited exam directory costs
    no GPU time. The scorer re-checks every row it scores.
    """
    mod = importlib.import_module("training.evaluate")
    return mod.verify_exam_stories(exam_dir, story_hashes, phases)


def bind_tokenizer(
    evaluate_all: EvaluateAllFn, tokenizer, cfg=None, story_hashes=None
) -> EvaluateAllFn:
    """Give the scorer the run's own tokenizer and eval config.

    The frozen hook is three positional arguments, so neither can be an
    argument of the call; `training/evaluate.py` accepts both as keyword-only
    extras. Binding here beats hanging the tokenizer off the model, which arm D
    would then have to `deepcopy`.

    `cfg` matters as much as the tokenizer: `EvalConfig.block_size` defaults to
    `REAL.block_size`, so without it a toy run would score 1,024-token windows
    against a 64-token model. A scorer without the parameter -- a test stub --
    is left unchanged, one keyword at a time, so a stub that takes only one of
    them still works.

    `story_hashes` is the manifest's normalised story hash -> phase (the guard
    report's `exam_story_phases`): the real scorer checks every continuation
    option's source against it and refuses an exam whose stories on disk are
    not the manifest's (2026-09-30).
    """
    try:
        params = inspect.signature(evaluate_all).parameters
    except (TypeError, ValueError):  # pragma: no cover - builtins
        return evaluate_all
    extras = {}
    if "tokenizer" in params:
        extras["tokenizer"] = tokenizer
    if cfg is not None and "cfg" in params:
        extras["cfg"] = cfg
    if story_hashes is not None and "story_hashes" in params:
        extras["story_hashes"] = story_hashes
    return partial(evaluate_all, **extras) if extras else evaluate_all


def make_eval_config(model_cfg: ModelConfig):
    """`EvalConfig` at this run's context length, or None if the evaluator is
    not importable yet (its tests stub the scorer anyway)."""
    try:
        from training.evaluate import EvalConfig
    except ImportError:  # pragma: no cover - evaluator not written yet
        return None
    return EvalConfig(block_size=model_cfg.block_size)


# ---------------------------------------------------------------------------
# precision


def resolve_precision(force_cpu: bool = False) -> tuple[torch.device, torch.dtype | None, torch.amp.GradScaler, str]:
    """bf16 where supported, fp16 + GradScaler on a T4, fp32 on CPU.

    Detected, never assumed (`PLAN.md`: "bf16 on A100, fp16 with loss scaling on
    T4 -- Kaggle's GPUs are T4s and lack bf16"); the answer goes in config.json.
    `is_bf16_supported()` alone says True on a T4 (it counts emulation), so
    native bf16 also needs compute capability 8.0+ (Ampere): the first Kaggle
    pre-pilot run reported "bf16" on a T4 (2026-10-01).
    """
    if not force_cpu and torch.cuda.is_available():
        device = torch.device("cuda")
        major, _minor = torch.cuda.get_device_capability(device)
        if major >= 8 and torch.cuda.is_bf16_supported():
            return device, torch.bfloat16, torch.amp.GradScaler("cuda", enabled=False), "bf16"
        return device, torch.float16, torch.amp.GradScaler("cuda", enabled=True), "fp16+gradscaler"
    return torch.device("cpu"), None, torch.amp.GradScaler("cpu", enabled=False), "fp32"


# ---------------------------------------------------------------------------
# resolved run settings


class Settings:
    """Everything `--arm`, `--pilot` and `--toy` resolve to. One object so
    `config.json` and the loop cannot disagree."""

    def __init__(self, args: argparse.Namespace) -> None:
        if args.arm not in ARMS:
            raise SystemExit(f"unknown arm {args.arm!r}; ARMS has {sorted(ARMS)}")
        self.arm: Arm = ARMS[args.arm]
        self.seed: int = args.seed
        self.pilot: bool = bool(args.pilot)
        self.toy: bool = bool(args.toy)
        self.model_cfg: ModelConfig = TOY if self.toy else REAL
        self.train_cfg = TrainConfig()
        self.stories_per_phase: int | None = None
        self.max_steps: int | None = None
        self.n_new: int = self.train_cfg.sequences_per_batch
        #: --micro-batch: memory only; the base loop and the hooks both use it.
        self.micro_batch: int | None = getattr(args, "micro_batch", None)
        #: The phases this run trains, by real id. All seven unless `--phases`
        #: declares a subset (e.g. 0,3,6 for the pre-pilot corpus). `getattr`
        #: because a Namespace built before the flag existed has no attribute.
        try:
            self.phases: tuple[int, ...] = resolve_phases(getattr(args, "phases", None))
        except ValueError as exc:
            raise SystemExit(f"--phases: {exc}") from None

        if self.pilot:
            # --pilot changes sizes and nothing else (AGENTS.md, frozen).
            self.stories_per_phase = PILOT_OVERRIDES["train_stories_per_phase"]
        if self.toy:
            # A CPU smoke test is not training (AGENTS.md). Never a result.
            self.stories_per_phase = TOY_RUN.stories_per_phase
            self.max_steps = TOY_RUN.max_steps
            self.n_new = TOY_RUN.sequences_per_batch

    @property
    def subset(self) -> bool:
        """True when the run declares fewer than all seven phases."""
        return self.phases != ALL_PHASES

    @property
    def scale_warmup(self) -> bool:
        """Only a pilot or a toy run may scale warmup (config.warmup_for)."""
        return self.pilot or self.toy

    def warmup(self, steps_per_phase: int) -> int:
        """PLAN.md's fixed 200 for a grid run; scaled only at pilot/toy sizes."""
        return warmup_for(
            steps_per_phase, self.train_cfg.warmup_steps, scaled=self.scale_warmup
        )

    @property
    def epochs(self) -> int:
        return self.train_cfg.epochs_per_phase

    @property
    def replay_fraction(self) -> float:
        return self.arm.replay_fraction


# ---------------------------------------------------------------------------
# one phase of training


def train_one_phase(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    *,
    new_loader: Iterable[torch.Tensor],
    replay_loader: Iterable[torch.Tensor],
    steps: int,
    base_lr: float,
    warmup: int,
    grad_clip: float,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    log: Callable[[str], None],
    micro_batch: int | None = None,
) -> dict:
    """The identical inner loop for every arm.

    Replay is *on top*: the new-phase rows of a batch are exactly the rows a
    replay-free arm would have seen at the same step, and the step count and the
    schedule do not know replay exists.

    `micro_batch` splits each batch into chunks of that many rows and
    accumulates their gradients before the one optimiser step: memory only.
    Each chunk's mean loss is weighted by its share of the rows, and every row
    has the same token count, so the summed gradient is the full batch's mean
    gradient (dropout is 0). None runs the batch in one piece.
    """
    model.train()
    replay_iter = iter(replay_loader)
    losses: list[float] = []
    n_new_seen = n_replay_seen = 0
    for step, new_batch in enumerate(new_loader):
        if step >= steps:
            break
        replay_batch = next(replay_iter, None)
        batch = concat_batches(new_batch, replay_batch)
        n_new_seen += int(new_batch.shape[0])
        n_replay_seen += 0 if replay_batch is None else int(replay_batch.shape[0])

        lr = cosine_lr(step, base_lr=base_lr, warmup=warmup, total_steps=steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = shift_for_lm(batch)
        n_rows = int(x.shape[0])
        chunk = micro_batch or n_rows
        step_loss = 0.0
        for lo in range(0, n_rows, chunk):
            xs, ys = x[lo : lo + chunk], y[lo : lo + chunk]
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                _, loss = model(xs, ys)
            part = loss * (xs.shape[0] / n_rows) if chunk < n_rows else loss
            scaler.scale(part).backward()
            step_loss += float(part.detach())
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad and p.grad is not None], grad_clip
            )
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        losses.append(step_loss)
        if step == 0 or (step + 1) % max(1, steps // 4) == 0:
            log(f"    step {step + 1}/{steps} lr {lr:.2e} loss {losses[-1]:.4f}")
    return {
        "steps": len(losses),
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        "mean_loss": sum(losses) / len(losses) if losses else None,
        "new_sequences": n_new_seen,
        "replay_sequences": n_replay_seen,
    }


# ---------------------------------------------------------------------------
# the run


def run(
    args: argparse.Namespace,
    *,
    after_phase: AfterPhaseFn | None = None,
    evaluate_all: EvaluateAllFn | None = None,
) -> Path:
    # -- 1. the guard, before anything else exists --------------------------
    # The experiment id comes from the run config (`config.EXPERIMENT_ID`)
    # unless `--experiment-id` names another (the pre-pilot's own manifest). A
    # Namespace built before the flag existed gets the config's id; one that
    # carries None or "" is refused by the guard, never skipped.
    guard_report = guard.check(
        args.train_dir,
        args.exam_dir,
        args.manifest,
        experiment_id=getattr(args, "experiment_id", EXPERIMENT_ID),
    )

    settings = Settings(args)
    # The exam this run will be scored on is the one the manifest froze: every
    # declared phase's stories on disk, by the manifest's per-story hashes.
    # Still before a tokenizer, a dataset or a model exists.
    exam_story_counts = verify_exam_dir(
        Path(args.exam_dir), guard_report.exam_story_phases, settings.phases
    )
    arm = settings.arm
    tcfg = settings.train_cfg
    n_new = settings.n_new
    n_replay = replay_sequences_per_batch(n_new, settings.replay_fraction)

    seed_everything(settings.seed)
    g_init = make_generator(settings.seed, "init")

    git = git_info(REPO_ROOT)
    env = env_info()
    out_dir = Path(args.out)
    run_id = make_run_id(arm.name, settings.seed, git["commit"])
    shared_dir = Path(args.phase0_dir) if args.phase0_dir else out_dir.parent / "_phase0"

    device, amp_dtype, scaler, precision = resolve_precision(force_cpu=args.cpu)

    # -- 2. data ------------------------------------------------------------
    tokenizer = get_or_train_tokenizer(
        args.train_dir,
        settings.model_cfg.vocab_size,
        shared_dir / "tokenizer.json",
        corpus_hash=guard_report.data_hash,
        # A subset run's BPE sees the declared phases' files only, the ones
        # DataModule trains on; a full run passes None and is unchanged.
        phases=settings.phases if settings.subset else None,
    )
    if getattr(args, "prepare_only", False):
        # The guard has passed and the shared tokenizer exists; nothing else is
        # written. runbook.grid calls this once before dispatching runs in
        # parallel, so no two runs race to train and write tokenizer.json.
        print(f"prepared: guard passed, tokenizer at {shared_dir / 'tokenizer.json'}")
        return shared_dir
    data = DataModule(
        args.train_dir,
        tokenizer,
        settings.model_cfg.block_size,
        settings.seed,
        phases=settings.phases,
        replay_dir=args.replay_dir,
        stories_per_phase=settings.stories_per_phase,
        device=device,
    )

    def steps_for(phase: int) -> int:
        s = data.steps_for_phase(phase, settings.epochs, n_new)
        return min(s, settings.max_steps) if settings.max_steps else s

    # How much phase-k material this arm actually consumes (AGENTS.md,
    # Amendment 2; the compute claim is H2b). A table lookup on the hook mode,
    # like every other per-arm difference in this file.
    phase_passes, replay_passes = PHASE_PASSES[arm.after_phase]
    budget = data.token_budget(
        settings.epochs,
        n_new,
        settings.replay_fraction,
        phase_passes=phase_passes,
        replay_passes=replay_passes,
    )
    if settings.max_steps:
        for row in budget["per_phase"]:
            row["steps"] = min(row["steps"], settings.max_steps)

    fingerprint = Fingerprint(
        commit=git["commit"],
        dirty=git["dirty"],
        seed=settings.seed,
        model_config=asdict(settings.model_cfg),
        data_hash=guard_report.data_hash,
        manifest_sha256=guard_report.manifest_sha256,
        tokenizer_sha256=_tokenizer_digest(tokenizer),
        train_config_hash=_config_digest(tcfg, settings),
        pilot=settings.pilot,
    )

    warmups = {k: settings.warmup(steps_for(k)) for k in settings.phases}
    for row in budget["per_phase"]:
        row["warmup_steps"] = warmups[row["phase"]]

    config = {
        "run_id": run_id,
        "arm": arm.name,
        "arm_row": asdict(arm),
        "seed": settings.seed,
        "pilot": settings.pilot,
        "toy": settings.toy,
        "precision": precision,
        "device": str(device),
        #: --micro-batch: gradient accumulation chunk, memory only (None = whole batch).
        "micro_batch": settings.micro_batch,
        "model": asdict(settings.model_cfg),
        "n_parameters": sum(settings.model_cfg.n_params()),
        "n_parameters_nonembedding": settings.model_cfg.n_params()[0],
        "n_parameters_embedding": settings.model_cfg.n_params()[1],
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "train": asdict(tcfg),
        "sequences_per_batch_new": n_new,
        "sequences_per_batch_replay": n_replay,
        #: PLAN.md's fixed 200 on the grid; scaled only at pilot/toy sizes
        #: (config.warmup_for). Per phase, because phases differ in length.
        "warmup_steps": warmups,
        "warmup_scaled": settings.scale_warmup,
        #: The frozen contract's pilot exam size. The training script cannot
        #: enforce it: `EvalConfig` has no item limit, so every item present in
        #: --exam-dir is scored. Recorded so a pilot's true exam size is in the
        #: record rather than assumed. See the report to the lead.
        "pilot_exam_items_per_phase": (
            PILOT_OVERRIDES["exam_items_per_phase"] if settings.pilot else None
        ),
        "exam_items_enforced_by_training_script": False,
        #: The day happens inside after_phase for arms C, D and D-nr, so this
        #: run's base loop takes no steps and its cost is in consolidate_s.
        "base_loop_owned_by_hook": arm.uses_lora,
        #: Passes over each phase's material: 1 for a full fine-tune arm, 2 for
        #: a LoRA arm (day + night). Feeds tokens_vs_arm_a, which H2b reads.
        "phase_passes": phase_passes,
        "replay_passes": replay_passes,
        #: The model that generated the training stories, read from the lines
        #: that will actually be trained on and asserted single-valued across
        #: every phase (AGENTS.md, Amendment 2). Not a flag, not a sidecar: a
        #: value that can disagree with the data is not provenance.
        "generator_model": data.generator_model,
        "token_budget": budget,
        "corpus": data.corpus_summary(),
        "exam_types": list(EXAM_TYPES),
        "matrix_keys": list(MATRIX_KEYS),
        #: The matrix is always N_PHASES x N_PHASES, indexed by real phase id.
        "n_phases": N_PHASES,
        #: The phases this run trained, by real id. A subset run (--phases)
        #: leaves the other rows of M null and their exam columns unscored
        #: (NaN): those phases were never trained or examined, and a reader
        #: must take the trained set from here, never infer it or renumber.
        "phases": list(settings.phases),
        "subset_phases": settings.subset,
        "train_dir": guard_report.train_dir,
        "exam_dir": guard_report.exam_dir,
        "replay_dir": str(data.replay_dir),
        "manifest": {
            "path": guard_report.manifest_path,
            "experiment_id": guard_report.experiment_id,
            "n_files": guard_report.n_manifest_files,
            #: Per declared phase, the exam stories confirmed on disk against
            #: the manifest's per-story hashes before the run started.
            "exam_stories_verified": {str(k): n for k, n in exam_story_counts.items()},
        },
        # AGENTS.md, Amendments 2026-09-22: every hash under one key.
        "hashes": {
            "manifest": guard_report.manifest_sha256,
            "train_files": guard_report.train_file_hashes,
            "data": guard_report.data_hash,
            "config": fingerprint.train_config_hash,
            "tokenizer": fingerprint.tokenizer_sha256,
            #: How the exams are scored (AGENTS.md, Amendment 4): runs scored
            #: under different rules are never compared.
            "scoring": _scoring_digest(make_eval_config(settings.model_cfg)),
            "fingerprint": fingerprint.digest(),
        },
        "evaluation": _scoring_rules(make_eval_config(settings.model_cfg)),
        "fingerprint": asdict(fingerprint),
        "shared_dir": str(shared_dir),
        "env": env,
        "git": git,
    }

    record = RunRecord(out_dir, config, git, env)
    record.log(f"run {run_id}: arm {arm.name}, seed {settings.seed}, {precision} on {device}")
    record.log(
        f"corpus generated by {data.generator_model!r}, single-valued across "
        + (f"declared phases {list(settings.phases)}" if settings.subset else f"all {N_PHASES} phases")
    )
    record.log(
        f"batch: {n_new} new + {n_replay} replay sequences "
        f"(replay fraction {settings.replay_fraction}) x {phase_passes} pass(es), "
        f"{budget['tokens_vs_arm_a']:.3f}x arm A's tokens, warmup {warmups[0]}"
        f"{' (scaled: pilot/toy sizes)' if settings.scale_warmup else ' (PLAN.md fixed)'}"
    )

    after_phase = after_phase if after_phase is not None else resolve_after_phase(arm)
    evaluate_all = bind_tokenizer(
        evaluate_all if evaluate_all is not None else resolve_evaluate_all(),
        tokenizer,
        make_eval_config(settings.model_cfg),
        story_hashes=guard_report.exam_story_phases,
    )
    exam_dir = Path(args.exam_dir)
    # The exam phases scored at every row: the declared ones. On the grid that
    # is all seven; in a subset run an undeclared phase has no exam to score,
    # and evaluate_all leaves its column NaN rather than guessing.
    all_phases = list(settings.phases)

    generators = {"init": g_init, "new": data.g_new, "replay": data.g_replay, "joint": data.g_joint}

    # -- 3. the shared random init, one per seed ----------------------------
    model = build_model(settings.model_cfg, g_init).to(device)
    ipath = init_path(shared_dir, settings.seed)
    if ipath.exists():
        shared = load_shared(ipath, "init", expect=fingerprint)
        model.load_state_dict(shared.model_state)
        record.log(f"loaded the shared random init {ipath.name}")
    else:
        save_shared(ipath, kind="init", model=model, fingerprint=fingerprint)
        record.log(f"wrote the shared random init {ipath.name}")
    model = model.to(device)

    matrix = empty_matrix(N_PHASES, MATRIX_KEYS)
    timings = Timings()
    start_phase = 0
    inherited_phase0 = False

    # -- 4. resume ----------------------------------------------------------
    spath = state_path(out_dir)
    if not args.resume and spath.exists():
        # A fresh run never writes over another run's folder: the first Kaggle
        # pre-pilot's arm A ran with a stale --out and replaced phase0's result
        # files in place (2026-10-01).
        raise SystemExit(
            f"{out_dir} already holds a run ({spath.name}). Pass --resume to continue "
            "that run, or point --out at this run's own folder."
        )
    if args.resume and spath.exists():
        state = load_state(spath)
        require_compatible(fingerprint, state.fingerprint, f"run state {spath}")
        model.load_state_dict(state.model_state)
        if state.scaler_state is not None:
            scaler.load_state_dict(state.scaler_state)
        restore_rng_state(state.rng, generators)
        matrix = state.matrix
        timings = Timings(state.timings.get("per_phase"))
        start_phase = state.completed_phase + 1
        inherited_phase0 = bool(state.extra.get("inherited_phase0", False))
        record.log(f"resumed from {spath.name}: phases 0..{state.completed_phase} are done")
    elif args.resume:
        record.log(f"--resume: no {spath.name} yet, starting from scratch")

    # -- 5. phase 0: train it, or inherit it --------------------------------
    if start_phase == 0:
        if inherits_phase0(arm):
            ppath = phase0_path(shared_dir, settings.seed)
            shared = load_shared(ppath, "phase0", expect=fingerprint)
            model.load_state_dict(shared.model_state)
            model = model.to(device)
            set_untrained(matrix, shared.payload["untrained"], N_PHASES)
            set_row(matrix, 0, shared.payload["row0"], N_PHASES)
            timings.adopt(shared.payload["timing_row"])
            timings.set_phase(0)
            inherited_phase0 = True
            start_phase = 1
            record.log(
                f"inherited phase 0 from {ppath.name}: row, untrained row and "
                f"{shared.payload['timing_row'].get('train_s', 0.0):.1f}s of wall clock"
            )
            # No hook runs at phase 0 (AGENTS.md, Amendments 2026-09-22). Phase 0
            # is ordinary full training shared by A, B, C, D and D-nr, and
            # PLAN.md gives C and D "one LoRA per phase 1-6", so the LoRA regime
            # starts at phase 1.
            _save(spath, model, None, scaler, generators, 0, matrix, timings, fingerprint, inherited_phase0)
            record.write_all(matrix, timings)
        else:
            timings.set_phase(0)
            with timings.timer("eval_s"):
                set_untrained(matrix, evaluate_all(model, exam_dir, all_phases), N_PHASES)
            record.log("scored the untrained model (forward transfer is measured against it)")

    # -- 6. the phases ------------------------------------------------------
    # Declared phases only, by real id. `start_phase` is "the first phase not
    # yet done" (1 after inheriting phase 0, completed + 1 after a resume), so
    # in a 0/3/6 run this continues at 3, never at a renumbered 1.
    for phase in (k for k in settings.phases if k >= start_phase):
        timings.set_phase(phase)
        steps = steps_for(phase)
        ctx = _make_ctx(settings, data, phase, steps, device, amp_dtype, record, timings)
        optimizer = None

        if arm.uses_lora:
            # The hook owns the phase (AGENTS.md, Amendments 2026-09-22). The
            # day -- training the LoRA on phase k -- happens inside
            # `after_phase`, so there is no ordinary base loop here. Running
            # both would give arms C, D and D-nr roughly double Arm A's tokens
            # and would fine-tune arm C's base on every phase, which is the
            # opposite of "base frozen after phase 0". Nothing would crash and
            # the heatmap would have looked fine, which is why this is asserted
            # in tests rather than left to review.
            stats = _no_base_training(arm.name, phase)
            record.log(f"  phase {phase}: arm {arm.name}'s day happens in the hook; no base loop, train_s stays 0")
        else:
            if arm.joint:
                new_loader = data.make_joint_loader(n_new, steps)
                replay_loader: Iterable[torch.Tensor] = iter(())
                label = f"segment {phase} (all phases shuffled)"
            else:
                new_loader = data.make_phase_loader(phase, n_new, steps=steps)
                replay_loader = data.make_replay_loader(n_replay, phase=phase, steps=steps)
                label = f"phase {phase}"
            record.log(f"  training {label}: {steps} steps x ({n_new} new + {n_replay if phase else 0} replay)")
            optimizer = model.configure_optimizers(tcfg.lr, tcfg.weight_decay, tcfg.betas, device.type)

            with timings.timer("train_s"):
                stats = train_one_phase(
                    model,
                    optimizer,
                    scaler,
                    new_loader=new_loader,
                    replay_loader=replay_loader,
                    steps=steps,
                    base_lr=tcfg.lr,
                    warmup=settings.warmup(steps),
                    grad_clip=tcfg.grad_clip,
                    device=device,
                    amp_dtype=amp_dtype,
                    log=record.log,
                    micro_batch=settings.micro_batch,
                )
            record.log(
                f"  {label} done: loss {stats['first_loss']:.4f} -> {stats['last_loss']:.4f}, "
                f"{stats['new_sequences']} new + {stats['replay_sequences']} replay sequences"
            )

        # Phases 1..6 only; nothing runs at phase 0.
        with timings.timer("consolidate_s"):
            model = after_phase(model, phase, ctx)

        with timings.timer("eval_s"):
            set_row(matrix, phase, evaluate_all(model, exam_dir, all_phases), N_PHASES)

        if produces_phase0(arm) and phase == 0:
            save_shared(
                phase0_path(shared_dir, settings.seed),
                kind="phase0",
                model=model,
                fingerprint=fingerprint,
                payload={
                    "untrained": {t: list(matrix["untrained"][t]) for t in MATRIX_KEYS},
                    "row0": get_row(matrix, 0),
                    "timing_row": dict(timings.per_phase[0]),
                    "run_id": run_id,
                    "stats": stats,
                },
            )
            record.log(f"wrote the shared phase-0 checkpoint {phase0_path(shared_dir, settings.seed).name}")

        _save(spath, model, optimizer, scaler, generators, phase, matrix, timings, fingerprint, inherited_phase0)
        record.write_all(matrix, timings)

        if produces_phase0(arm):
            record.log("arm phase0 trains phase 0 only; A, B, C, D and D-nr take it from here")
            break

    record.write_all(matrix, timings)
    record.log(f"done: {out_dir}")
    return out_dir


def _no_base_training(arm_name: str, phase: int) -> dict:
    """The `stats` row for a phase whose training happened inside the hook.

    Zeroes, not absences: the run record must say out loud that the base loop
    took no steps, so `timings.json` showing `train_s: 0` for arm C reads as a
    fact about the regime rather than a missing measurement.
    """
    return {
        "steps": 0,
        "first_loss": None,
        "last_loss": None,
        "mean_loss": None,
        "new_sequences": 0,
        "replay_sequences": 0,
        "owned_by_hook": True,
        "arm": arm_name,
        "phase": phase,
    }


def _save(path, model, optimizer, scaler, generators, phase, matrix, timings, fingerprint, inherited) -> None:
    save_state(
        path,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        generators=generators,
        completed_phase=phase,
        matrix=matrix,
        timings=timings.to_dict(),
        fingerprint=fingerprint,
        extra={"inherited_phase0": inherited},
    )


def _make_ctx(settings, data, phase, steps, device, amp_dtype, record, timings) -> hooks.PhaseContext:
    n_new = settings.n_new
    return hooks.PhaseContext(
        arm=settings.arm.name,
        seed=settings.seed,
        phase=phase,
        replay_fraction=settings.replay_fraction,
        device=device,
        amp_dtype=amp_dtype,
        make_phase_loader=partial(_phase_loader, data, steps),
        make_replay_loader=partial(data.make_replay_loader, phase=phase, steps=steps),
        steps=steps,
        sequences_per_batch=n_new,
        lr=settings.train_cfg.lr,
        warmup=settings.warmup(steps),
        log=record.log,
        timer=timings.timer,
        micro_batch=settings.micro_batch,
    )


def _phase_loader(data: DataModule, steps: int, phase: int, n_sequences: int):
    return data.make_phase_loader(phase, n_sequences, steps=steps)


def _tokenizer_digest(tokenizer) -> str:
    import hashlib

    probe = "the quick brown fox jumps over the lazy dog 0123456789"
    payload = json.dumps(
        {"vocab_size": tokenizer.vocab_size, "eot": tokenizer.eot_id, "probe": tokenizer.encode(probe)},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _scoring_rules(eval_cfg) -> dict | None:
    """The evaluator's scoring rules, for config.json. Memory knobs (batch
    size, logit chunk, device) are left out: they do not move a score."""
    if eval_cfg is None:
        return None
    return {
        "block_size": eval_cfg.block_size,
        "perplexity_window_rule": eval_cfg.perplexity_window_rule,
        "cloze_scoring_rule": eval_cfg.cloze_scoring_rule,
        "continuation_headline": eval_cfg.continuation_headline,
    }


def _scoring_digest(eval_cfg) -> str | None:
    import hashlib

    rules = _scoring_rules(eval_cfg)
    if rules is None:
        return None
    return hashlib.sha256(json.dumps(rules, sort_keys=True).encode("utf-8")).hexdigest()


def _config_digest(tcfg: TrainConfig, settings: "Settings") -> str:
    import hashlib

    payload = json.dumps(
        {
            "train": asdict(tcfg),
            "epochs": settings.epochs,
            "n_new": settings.n_new,
            # The warmup *policy*, not a resolved number: warmup now depends on
            # the phase's step count, and the step count already enters the
            # fingerprint through the data hash and the sizes below.
            "warmup_base": settings.train_cfg.warmup_steps,
            "warmup_scaled": settings.scale_warmup,
        #: The frozen contract's pilot exam size. The training script cannot
        #: enforce it: `EvalConfig` has no item limit, so every item present in
        #: --exam-dir is scored. Recorded so a pilot's true exam size is in the
        #: record rather than assumed. See the report to the lead.
        "pilot_exam_items_per_phase": (
            PILOT_OVERRIDES["exam_items_per_phase"] if settings.pilot else None
        ),
        "exam_items_enforced_by_training_script": False,
            "stories_per_phase": settings.stories_per_phase,
            "max_steps": settings.max_steps,
            "toy": settings.toy,
            # Only in subset mode, so a full run's digest -- and every shared
            # init and phase-0 checkpoint keyed on it -- is unchanged.
            **({"phases": list(settings.phases)} if settings.subset else {}),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# cli


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m training.train",
        description="Train one arm of the Lifespan experiment. Arms are rows of training.config.ARMS.",
    )
    p.add_argument("--arm", required=True, choices=sorted(ARMS))
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--train-dir", required=True, type=Path)
    p.add_argument("--exam-dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--pilot",
        action="store_true",
        help="sizes only: 1,000 stories and 100 exam items per phase. Never a hyperparameter.",
    )
    # Paths and the smoke-test switch. None of these is a hyperparameter.
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="exams/manifest.json")
    p.add_argument("--replay-dir", type=Path, default=None, help="default: <train-dir>/replay")
    p.add_argument("--phase0-dir", type=Path, default=None, help="shared init and phase-0 checkpoints")
    p.add_argument(
        "--toy",
        action="store_true",
        help="training.config.TOY: a CPU smoke test, never a result (AGENTS.md).",
    )
    p.add_argument("--cpu", action="store_true", help="force CPU even where CUDA exists")
    p.add_argument(
        "--prepare-only",
        action="store_true",
        help=(
            "run the guard and build the shared tokenizer, then exit without training or "
            "writing --out (runbook.grid runs this once before parallel dispatch)"
        ),
    )
    p.add_argument(
        "--micro-batch",
        type=_positive_int,
        default=None,
        help=(
            "rows per forward/backward; gradients accumulate to the full batch before each "
            "optimiser step. Memory only, never a hyperparameter: the batch, steps and lr "
            "schedule are unchanged. Default: the whole batch at once."
        ),
    )
    p.add_argument(
        "--phases",
        type=parse_phases,
        default=None,
        help=(
            "subset-phase mode: the phases this run trains, comma-separated and ascending, "
            "e.g. 0,3,6. Phases keep their real ids everywhere (matrix, replay, checkpoints, "
            "report); undeclared phases need no training file. Default: all seven."
        ),
    )
    p.add_argument(
        "--experiment-id",
        default=EXPERIMENT_ID,
        help=(
            "the experiment this run belongs to; the guard refuses unless the manifest's "
            f"experiment_id equals it. Default: training.config.EXPERIMENT_ID ({EXPERIMENT_ID!r}). "
            "Only a throwaway experiment with its own --manifest (the pre-pilot) passes another."
        ),
    )
    return p


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {value}")
    return value


def parse_phases(text: str) -> tuple[int, ...]:
    """`"0,3,6"` -> `(0, 3, 6)`, validated by `config.resolve_phases`."""
    try:
        return resolve_phases(int(part) for part in text.split(",") if part.strip() != "")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def main(
    argv: list[str] | None = None,
    *,
    after_phase: AfterPhaseFn | None = None,
    evaluate_all: EvaluateAllFn | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    run(args, after_phase=after_phase, evaluate_all=evaluate_all)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
