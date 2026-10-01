"""The night: the `after_phase` hook for arms C, D and D-nr.

PLAN.md, "The consolidation step", is the spec. One "day" is a phase; one
"night" is this file.

    1. Freeze base B_k. Train LoRA L_k on phase-k stories with the ordinary LM
       loss, same token budget as the other arms. Only LoRA parameters have
       requires_grad; assert it.
    2. Freeze L_k. The teacher for phase-k material is T = B_k + L_k.
    3. Unfreeze the base and train it toward

           L = KL(T || B_{k+1})            on phase-k stories
             + lambda * KL(B_k || B_{k+1}) on replay, phases < k

       lambda = 1, batches mixed 70% phase-k and 30% replay. The replay targets
       are the previous base's own logits, not the stored text's labels.
    4. Discard L_k and hand B_{k+1} back as the next frozen base.

Arm C stops after step 1 and merges arithmetically. Arm D-nr is arm D with the
replay term and the replay batches removed and nothing else changed: the same
function, entered with `replay_fraction == 0.0`.

For an arm with `uses_lora` this hook owns the whole phase (AGENTS.md,
Amendments 2026-09-22): `train.py` does not run its ordinary base phase loop,
because the day -- training the LoRA on phase k -- happens in step 1 below. So
step 1 spends the phase's entire budget, `ctx.steps` optimizer steps of
`ctx.sequences_per_batch` new-phase sequences, exactly what arm A spends on the
same phase and no more. Running both loops would double the arm's tokens and
leave arm C's base unfrozen, which is the opposite of what arm C is for.

Phase 0 is ordinary full training for every sequential arm and no hook runs
there (AGENTS.md, corrected 2026-09-22): a rank-16 LoRA on a frozen random base
cannot learn, so there is no L_0 and no night after day 0. The phase-0 guard
below stays anyway -- it costs nothing and it is the one case where being called
by mistake would silently corrupt the shared phase-0 checkpoint.

Adapters are found by name pattern, never by importing a model class (see
`training/lora.py`). The two things this file does import from its siblings are
`cosine_lr` and `shift_for_lm`, deliberately: the night must use the day's
schedule and the day's batch slicing exactly, and a second copy of either is
how arm D quietly stops being comparable to arm A.
"""

from __future__ import annotations

import copy
from typing import Callable, Iterable, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TrainConfig, replay_sequences_per_batch
from .data import shift_for_lm
from .hooks import PhaseContext
from .model import cosine_lr
from .lora import (
    apply_lora,
    assert_only_lora_trainable,
    forward_logits,
    freeze_all,
    has_lora,
    lora_parameters,
    merge_lora,
    strip_lora,
    unfreeze_all,
)

#: The timing bucket every consolidation second is accumulated under, so H2's
#: compute ratio is measured rather than estimated. One of the three frozen keys
#: (AGENTS.md, Amendments 2026-09-22); the bare "consolidate" alias is not used
#: here, so nothing load-bearing depends on it.
TIMER_LABEL = "consolidate_s"


# ---------------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------------


def kl_teacher_student(
    teacher_logits: torch.Tensor, student_logits: torch.Tensor
) -> torch.Tensor:
    """KL(teacher || student), mean over tokens, summed over the vocabulary.

    Direction, spelled out because it is silent when wrong:

        KL(T || S) = sum_v p_T(v) * (log p_T(v) - log p_S(v))

    `torch.nn.functional.kl_div(input, target)` takes the *student's*
    log-probabilities as `input` and the *teacher's* distribution as `target`,
    and computes `sum target * (log target - input)`. So the student goes first
    in the call and second in the name:

        F.kl_div(input=log p_S, target=log p_T, log_target=True) == KL(T || S)

    Both log-softmaxes are taken in float32 even under fp16/bf16 autocast: a KL
    over an 8k vocabulary underflows quietly in half precision.
    """
    t_logp = F.log_softmax(teacher_logits.float(), dim=-1)
    s_logp = F.log_softmax(student_logits.float(), dim=-1)
    per_token = F.kl_div(s_logp, t_logp, log_target=True, reduction="none").sum(-1)
    return per_token.mean()


def lm_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Ordinary next-token cross entropy in float32, over the same positions
    `train.py`'s day loop scores: inputs and targets come from `shift_for_lm`,
    so arm C's and arm D's day is the same loss on the same tokens as arm A's.
    """
    vocab = logits.shape[-1]
    return F.cross_entropy(logits.reshape(-1, vocab).float(), targets.reshape(-1))


# ---------------------------------------------------------------------------
# small training utilities (local: this file must not touch train.py's loop)
# ---------------------------------------------------------------------------


def _endless(make_loader: Callable[[], Iterable[torch.Tensor]]) -> Iterator[torch.Tensor]:
    """Batches forever, re-opening the loader when it is exhausted."""
    while True:
        empty = True
        for batch in make_loader():
            empty = False
            yield batch
        if empty:
            raise RuntimeError("loader yielded no batches")


def _lr_at(step: int, total: int, warmup: int, base_lr: float) -> float:
    """The day's schedule, not a second copy of it: `training.model.cosine_lr`,
    linear warmup then cosine decay, restarted for every phase and for each of
    step 1 and step 3 (optimizer state is fresh each time)."""
    return cosine_lr(step, base_lr=base_lr, warmup=warmup, total_steps=total)


def _make_optimizer(
    model: nn.Module, params: list[nn.Parameter], cfg: TrainConfig, lr: float, device_type: str
) -> torch.optim.Optimizer:
    """The same optimizer the day uses, built the same way.

    `train.py` builds arm A's optimizer with `model.configure_optimizers`, which
    puts 1-D tensors in a no-decay group; the night uses it too where the model
    offers it, so arm D's base does not get a subtly different AdamW from arm
    A's. It filters on `requires_grad`, which the caller has already set. The
    fallback keeps this file runnable against a bare `nn.Module` fixture.
    """
    configure = getattr(model, "configure_optimizers", None)
    if callable(configure) and params:
        return configure(lr, cfg.weight_decay, cfg.betas, device_type)
    return torch.optim.AdamW(params, lr=lr, betas=cfg.betas, weight_decay=cfg.weight_decay)


def check_batch(idx: torch.Tensor, n_sequences: int, what: str) -> torch.Tensor:
    """Loaders yield contiguous full blocks, never padded (AGENTS.md, frozen
    2026-09-22), so every position carries a prediction and the KL averages over
    all of them. Assert the shape rather than trusting it: a loader that hands
    back a different N would silently change this arm's token budget."""
    if idx.ndim != 2 or idx.shape[0] != n_sequences or idx.shape[1] < 2:
        raise AssertionError(
            f"{what} batch has shape {tuple(idx.shape)}; expected "
            f"[{n_sequences}, block_size] of contiguous unpadded ids"
        )
    if idx.dtype not in (torch.long, torch.int32, torch.int64):
        raise AssertionError(f"{what} batch has dtype {idx.dtype}; expected token ids")
    return idx


def _chunks(n_rows: int, micro_batch: int | None):
    """Row slices of one batch for gradient accumulation (`--micro-batch`).

    Each chunk's mean loss is weighted by its share of the rows: rows are
    contiguous unpadded blocks (`check_batch`), so every row has the same
    token count and the weighted sum is the whole batch's token mean.
    """
    size = micro_batch or n_rows
    for lo in range(0, n_rows, size):
        hi = min(n_rows, lo + size)
        yield slice(lo, hi), (hi - lo) / n_rows


def _autocast(ctx: PhaseContext):
    if ctx.amp_dtype is None:
        return torch.autocast(device_type=ctx.device.type, enabled=False)
    return torch.autocast(device_type=ctx.device.type, dtype=ctx.amp_dtype)


# ---------------------------------------------------------------------------
# step 1 -- shared byte for byte by arms C and D (and D-nr)
# ---------------------------------------------------------------------------


def train_phase_lora(
    model: nn.Module, phase_k: int, ctx: PhaseContext, cfg: TrainConfig
) -> nn.Module:
    """Step 1: freeze B_k, train L_k on phase-k stories with the ordinary LM
    loss. This is the arm's day, and since `train.py` runs no base phase loop
    for a `uses_lora` arm it carries the phase's whole budget: `ctx.steps`
    optimizer steps of `ctx.sequences_per_batch` new-phase sequences, the same
    step count, batch size, lr and schedule arm A gets on phase k. Returns the
    model with the adapters still attached and frozen.

    There is exactly one of these. Arms C and D both reach it through the module
    global `STEP1`, so they cannot drift apart.
    """
    n_new = ctx.sequences_per_batch
    apply_lora(model, rank=cfg.lora_rank, alpha=cfg.lora_alpha, seed=ctx.seed)
    trainable = assert_only_lora_trainable(model)  # invariant, not an assumption
    ctx.log(
        f"[consolidate] phase {phase_k} step 1: LoRA rank {cfg.lora_rank} "
        f"alpha {cfg.lora_alpha}, {len(trainable)} trainable tensors, "
        f"{ctx.steps} steps x {n_new} sequences"
    )

    model.to(ctx.device)
    model.train()
    params = lora_parameters(model)
    opt = _make_optimizer(model, params, cfg, ctx.lr, ctx.device.type)
    scaler = torch.amp.GradScaler(
        ctx.device.type, enabled=(ctx.amp_dtype == torch.float16 and ctx.device.type == "cuda")
    )
    batches = _endless(lambda: ctx.make_phase_loader(phase_k, n_new))

    for step in range(ctx.steps):
        lr = _lr_at(step, ctx.steps, ctx.warmup, ctx.lr)
        for group in opt.param_groups:
            group["lr"] = lr
        idx = check_batch(next(batches), n_new, f"phase-{phase_k}").to(ctx.device)
        x, y = shift_for_lm(idx)
        opt.zero_grad(set_to_none=True)
        for rows, share in _chunks(x.shape[0], getattr(ctx, "micro_batch", None)):
            with _autocast(ctx):
                logits = forward_logits(model, x[rows])
            loss = lm_loss(logits, y[rows]) * share
            scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        scaler.step(opt)
        scaler.update()

    # Step 2: freeze L_k.
    freeze_all(model)
    model.eval()
    return model


#: The single step-1 callable. Both `merge_after_phase` and
#: `distill_after_phase` call it through this name, so "arms C and D share step
#: 1 byte for byte" is a property of the code, not of a comment.
STEP1: Callable[[nn.Module, int, PhaseContext, TrainConfig], nn.Module] = train_phase_lora


# ---------------------------------------------------------------------------
# step 3 -- the distillation
# ---------------------------------------------------------------------------


def distill_into_base(
    student: nn.Module,
    teacher_T: nn.Module,
    prev_base: nn.Module | None,
    phase_k: int,
    ctx: PhaseContext,
    cfg: TrainConfig,
) -> nn.Module:
    """Step 3: unfreeze the base and train B_{k+1} toward

        KL(T || B_{k+1}) on phase-k stories
        + lambda * KL(B_k || B_{k+1}) on replay from phases < k

    `prev_base` is B_k, a genuine frozen copy taken before the student was
    unfrozen; it is None exactly when there is no replay term (arm D-nr, or
    phase 0). Each term is averaged over its own tokens before lambda is
    applied, so lambda = 1 means what it says.
    """
    n_new = ctx.sequences_per_batch
    n_replay = (
        replay_sequences_per_batch(n_new, ctx.replay_fraction) if prev_base is not None else 0
    )

    unfreeze_all(student)
    student.to(ctx.device)
    student.train()
    for teacher in (teacher_T, prev_base):
        if teacher is not None:
            freeze_all(teacher)
            teacher.to(ctx.device)
            teacher.eval()

    ctx.log(
        f"[consolidate] phase {phase_k} step 3: distilling into the base, "
        f"{ctx.steps} steps x ({n_new} phase-{phase_k} + {n_replay} replay) sequences, "
        f"lambda={cfg.distill_lambda}"
    )

    params = [p for p in student.parameters() if p.requires_grad]
    opt = _make_optimizer(student, params, cfg, ctx.lr, ctx.device.type)
    scaler = torch.amp.GradScaler(
        ctx.device.type, enabled=(ctx.amp_dtype == torch.float16 and ctx.device.type == "cuda")
    )
    phase_batches = _endless(lambda: ctx.make_phase_loader(phase_k, n_new))
    replay_batches = (
        _endless(lambda: ctx.make_replay_loader(n_replay)) if n_replay > 0 else None
    )

    for step in range(ctx.steps):
        lr = _lr_at(step, ctx.steps, ctx.warmup, ctx.lr)
        for group in opt.param_groups:
            group["lr"] = lr

        # The teachers and the student see exactly the inputs the day saw:
        # `shift_for_lm`'s x, so every position carries a next-token prediction
        # and no arm is scored on a position another arm never saw.
        idx = check_batch(next(phase_batches), n_new, f"phase-{phase_k}").to(ctx.device)
        x, _ = shift_for_lm(idx)
        micro = getattr(ctx, "micro_batch", None)
        opt.zero_grad(set_to_none=True)
        # Each term is still its own token mean (chunk shares sum to 1 within
        # a term), so lambda keeps its meaning under --micro-batch.
        for rows, share in _chunks(x.shape[0], micro):
            with torch.no_grad(), _autocast(ctx):
                t_logits = forward_logits(teacher_T, x[rows])
            with _autocast(ctx):
                s_logits = forward_logits(student, x[rows])
            scaler.scale(kl_teacher_student(t_logits, s_logits) * share).backward()

        if replay_batches is not None:
            r_idx = check_batch(next(replay_batches), n_replay, "replay").to(ctx.device)
            rx, _ = shift_for_lm(r_idx)
            for rows, share in _chunks(rx.shape[0], micro):
                with torch.no_grad(), _autocast(ctx):
                    b_logits = forward_logits(prev_base, rx[rows])
                with _autocast(ctx):
                    sr_logits = forward_logits(student, rx[rows])
                term = cfg.distill_lambda * kl_teacher_student(b_logits, sr_logits) * share
                scaler.scale(term).backward()

        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        scaler.step(opt)
        scaler.update()

    return student


# ---------------------------------------------------------------------------
# the hook
# ---------------------------------------------------------------------------


def merge_after_phase(
    model: nn.Module, phase_k: int, ctx: PhaseContext, cfg: TrainConfig | None = None
) -> nn.Module:
    """Arm C: step 1, then the arithmetic merge `W += (alpha/rank) B A`.

    The null hypothesis for reverse-LoRA (PLAN.md, H3): merging without
    training. It must be the *same* step 1 arm D runs, or H3 compares two
    different LoRAs.
    """
    # `cfg` carries the frozen hyperparameters (rank, alpha, lambda, betas,
    # weight decay, grad clip) only. N comes from ctx.sequences_per_batch, never
    # from here, so a non-default N in train.py cannot desync day from night.
    cfg = cfg or TrainConfig()
    if phase_k == 0:
        # train.py no longer calls the hook at phase 0 (AGENTS.md, corrected
        # 2026-09-22). Kept as a guard: phase 0 is the checkpoint every
        # sequential arm shares, and a LoRA night on it would corrupt all five.
        ctx.log("[consolidate] phase 0 is full training; after_phase is the identity")
        return model
    with ctx.timer(TIMER_LABEL):
        model = STEP1(model, phase_k, ctx, cfg)
        model = merge_lora(model)
        assert not has_lora(model), "LoRA survived the merge"
    # Handed back as an ordinary model: arm C's base is held frozen by the next
    # `apply_lora`, not by leaving requires_grad False here, so a checkpoint
    # round-trip or an eval cannot silently inherit a frozen flag.
    unfreeze_all(model)
    model.train()
    return model


def distill_after_phase(
    model: nn.Module, phase_k: int, ctx: PhaseContext, cfg: TrainConfig | None = None
) -> nn.Module:
    """Arms D and D-nr: step 1, then the distillation, then discard L_k.

    D-nr is this function with `ctx.replay_fraction == 0.0`; nothing else about
    the path differs.
    """
    # `cfg` carries the frozen hyperparameters (rank, alpha, lambda, betas,
    # weight decay, grad clip) only. N comes from ctx.sequences_per_batch, never
    # from here, so a non-default N in train.py cannot desync day from night.
    cfg = cfg or TrainConfig()
    if phase_k == 0:
        # train.py no longer calls the hook at phase 0 (AGENTS.md, corrected
        # 2026-09-22). Kept as a guard: phase 0 is the checkpoint every
        # sequential arm shares, and a LoRA night on it would corrupt all five.
        ctx.log("[consolidate] phase 0 is full training; after_phase is the identity")
        return model
    with ctx.timer(TIMER_LABEL):
        model = STEP1(model, phase_k, ctx, cfg)

        # Step 2: T = B_k + L_k, as a separate frozen module. The copy is taken
        # here, while the base is still frozen and before anything is unfrozen.
        teacher_T = merge_lora(copy.deepcopy(model))
        freeze_all(teacher_T)
        teacher_T.eval()

        # Step 4 (the "discard" half): the adapters come off the student, which
        # is therefore exactly B_k again.
        student = strip_lora(model, merge=False)
        assert not has_lora(student), "LoRA survived the discard"

        # B_k: a genuine copy, taken BEFORE the student is unfrozen, never an
        # alias of the module being trained. A distillation whose teacher is the
        # student is a no-op that looks like it works.
        use_replay = ctx.replay_fraction > 0.0 and phase_k > 0
        prev_base = copy.deepcopy(student) if use_replay else None
        if prev_base is not None:
            freeze_all(prev_base)
            prev_base.eval()

        student = distill_into_base(student, teacher_T, prev_base, phase_k, ctx, cfg)

    del teacher_T, prev_base  # discard L_k and both teachers
    student.train()
    return student


#: Arm -> night. D and D-nr are the *same object*: one code path, entered with a
#: different replay fraction.
AFTER_PHASE: dict[str, Callable[..., nn.Module]] = {
    "C": merge_after_phase,
    "D": distill_after_phase,
    "D-nr": distill_after_phase,
}


def after_phase(
    model: nn.Module, phase_k: int, ctx: PhaseContext, cfg: TrainConfig | None = None
) -> nn.Module:
    """The hook `train.py` calls after every phase (AGENTS.md, frozen contract).

    Identity for arms A, B, E and phase0. train.py calls it for phases 1..6
    only; the phase-0 guard inside each night is belt and braces.
    """
    fn = AFTER_PHASE.get(ctx.arm)
    if fn is None:
        return model
    return fn(model, phase_k, ctx, cfg=cfg)
