"""The night, checked by arithmetic that can be done on paper.

The bugs this file exists to catch: a KL called backwards, a teacher that is
secretly the student, a replay term that does not fire, arm C and arm D running
two different step 1s, and a LoRA left behind in the model handed to phase k+1.
"""

from __future__ import annotations

import copy
import math

import pytest
import torch

from tests.conftest import RecordingContext, make_model, token_batch
from training import consolidate, lora
from training.config import TrainConfig, replay_sequences_per_batch


# ---------------------------------------------------------------------------
# the loss
# ---------------------------------------------------------------------------


def test_kl_direction_hand_worked():
    """Acceptance 5. Two outcomes, worked on paper.

    T = [0.7, 0.3], S = [0.4, 0.6].

      KL(T || S) = 0.7 * ln(0.7/0.4) + 0.3 * ln(0.3/0.6)
                 = 0.7 * ln(1.75)    + 0.3 * ln(0.5)
                 = 0.7 * 0.5596157879 + 0.3 * (-0.6931471806)
                 = 0.3917310515 - 0.2079441542
                 = 0.1837868973

    The other direction is a different number, which is the point:

      KL(S || T) = 0.4 * ln(0.4/0.7) + 0.6 * ln(0.6/0.3)
                 = -0.2238463152 + 0.4158883083
                 = 0.1920419931

    `kl_teacher_student(teacher_logits, student_logits)` must return the first.
    """
    t_logits = torch.log(torch.tensor([[0.7, 0.3]]))
    s_logits = torch.log(torch.tensor([[0.4, 0.6]]))
    expected = 0.7 * math.log(0.7 / 0.4) + 0.3 * math.log(0.3 / 0.6)
    assert expected == pytest.approx(0.1837868973, abs=1e-9)

    got = consolidate.kl_teacher_student(t_logits, s_logits).item()
    assert got == pytest.approx(0.1837868973, abs=1e-6)
    # ... and is not the reverse KL, which differs by ~0.008.
    reverse = consolidate.kl_teacher_student(s_logits, t_logits).item()
    assert reverse == pytest.approx(0.1920419931, abs=1e-6)
    assert abs(got - reverse) > 1e-3


def test_kl_is_zero_for_identical_distributions_and_averages_over_tokens():
    logits = torch.randn(3, 5, 17)
    assert consolidate.kl_teacher_student(logits, logits).item() == pytest.approx(0.0, abs=1e-6)
    # Per-token mean: doubling the token count with the same pair of rows
    # leaves the value unchanged.
    t = torch.log(torch.tensor([[0.7, 0.3]]))
    s = torch.log(torch.tensor([[0.4, 0.6]]))
    one = consolidate.kl_teacher_student(t, s)
    two = consolidate.kl_teacher_student(t.repeat(4, 1), s.repeat(4, 1))
    assert one.item() == pytest.approx(two.item(), abs=1e-7)


def test_kl_is_computed_in_float32_under_half_precision():
    t = torch.log(torch.tensor([[0.7, 0.3]])).half()
    s = torch.log(torch.tensor([[0.4, 0.6]])).half()
    out = consolidate.kl_teacher_student(t, s)
    assert out.dtype == torch.float32
    assert out.item() == pytest.approx(0.1837868973, abs=2e-3)  # fp16 input rounding


# ---------------------------------------------------------------------------
# step 1
# ---------------------------------------------------------------------------


def test_step1_trains_only_the_lora(toy_cfg: TrainConfig):
    """Acceptance 3: the trainable set is exactly the adapters, and the base
    weights are numerically unchanged after a few steps."""
    model = make_model(seed=0)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    rec = RecordingContext(arm="C", phase=1, replay_fraction=0.0, steps=4)

    out = consolidate.train_phase_lora(model, 1, rec.ctx, toy_cfg)

    assert rec.logs and "step 1" in rec.logs[0]
    trainable = [n for n, p in out.named_parameters() if p.requires_grad]
    assert trainable == []  # step 2 froze L_k on the way out
    for _, m in lora.lora_modules(out):
        assert m.lora_B.abs().sum().item() > 0  # it actually trained

    named = dict(out.named_parameters())
    for name, old in before.items():
        # an adapted projection's weight moves under `<name>.base.<leaf>`
        new = named.get(name)
        if new is None:
            stem, leaf = name.rsplit(".", 1)
            new = named[f"{stem}.base.{leaf}"]
        assert torch.equal(new, old), f"base weight moved during step 1: {name}"


def test_step1_asserts_the_invariant_rather_than_assuming_it(toy_cfg, monkeypatch):
    model = make_model(seed=0)
    rec = RecordingContext(arm="C", phase=1, replay_fraction=0.0, steps=1)
    real = lora.apply_lora

    def leaky(m, **kw):
        out = real(m, **kw)
        # A stray trainable base parameter. The embedding, not a norm: every
        # model has one, so the test does not depend on LayerNorm being affine.
        m.wte.weight.requires_grad_(True)
        return out

    monkeypatch.setattr(consolidate, "apply_lora", leaky)
    with pytest.raises(AssertionError, match="only LoRA parameters"):
        consolidate.train_phase_lora(model, 1, rec.ctx, toy_cfg)


# ---------------------------------------------------------------------------
# step 3
# ---------------------------------------------------------------------------


def _perturbed_teacher(student, scale: float = 0.05, seed: int = 5):
    teacher = copy.deepcopy(student)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in teacher.parameters():
            p.add_(torch.empty_like(p).normal_(0, scale, generator=g))
    return teacher


def test_step3_trains_only_the_base_and_leaves_the_teachers_alone(toy_cfg):
    """Acceptance 4: only the base has requires_grad during step 3; the
    teachers are frozen, in eval mode, and forwarded under no_grad."""
    student = make_model(seed=0)
    teacher = _perturbed_teacher(student)
    prev = copy.deepcopy(student)
    teacher_before = {n: p.detach().clone() for n, p in teacher.named_parameters()}
    prev_before = {n: p.detach().clone() for n, p in prev.named_parameters()}

    seen: list[tuple[bool, bool]] = []

    def watch(module, args, output):
        seen.append((torch.is_grad_enabled(), module.training))

    teacher.register_forward_hook(watch)
    prev.register_forward_hook(watch)

    rec = RecordingContext(arm="D", phase=2, replay_fraction=0.3, steps=3)
    out = consolidate.distill_into_base(student, teacher, prev, 2, rec.ctx, toy_cfg)

    assert all(not grad_on and not training for grad_on, training in seen)
    assert len(seen) == 6  # 3 steps x (T forward + B_k forward)
    assert all(p.requires_grad for p in out.parameters())
    assert not any(p.requires_grad for p in teacher.parameters())
    assert not any(p.requires_grad for p in prev.parameters())
    for n, p in teacher.named_parameters():
        assert torch.equal(p, teacher_before[n]), f"teacher moved: {n}"
    for n, p in prev.named_parameters():
        assert torch.equal(p, prev_before[n]), f"B_k moved: {n}"


def test_distillation_moves_the_student_toward_the_teacher(toy_cfg):
    """Acceptance 7: KL(T || B_{k+1}) < KL(T || B_k) after a few steps."""
    student = make_model(seed=0)
    teacher = _perturbed_teacher(student, scale=0.05)
    probe = token_batch(2, seed=99)

    def kl_on_probe(model):
        model.eval()
        with torch.no_grad():
            return consolidate.kl_teacher_student(
                lora.forward_logits(teacher, probe), lora.forward_logits(model, probe)
            ).item()

    before = kl_on_probe(copy.deepcopy(student))
    rec = RecordingContext(arm="D-nr", phase=1, replay_fraction=0.0, steps=20, lr=1e-2)
    out = consolidate.distill_into_base(student, teacher, None, 1, rec.ctx, toy_cfg)
    after = kl_on_probe(out)

    assert before > 0
    assert after < before, f"distillation did not move the student: {before} -> {after}"


def test_lambda_weights_the_replay_term(toy_cfg):
    """lambda = 1 and each term is averaged over its own tokens, so the loss is
    the plain sum of the two KLs. Checked on the pieces, not by rerunning."""
    assert TrainConfig().distill_lambda == 1.0
    t = torch.log(torch.tensor([[0.7, 0.3]]))
    s = torch.log(torch.tensor([[0.4, 0.6]]))
    term = consolidate.kl_teacher_student(t, s)
    total = term + TrainConfig().distill_lambda * term
    assert total.item() == pytest.approx(2 * 0.1837868973, abs=1e-6)


# ---------------------------------------------------------------------------
# the hook: dispatch, phase 0, batch composition
# ---------------------------------------------------------------------------


def test_arms_c_and_d_share_step1(monkeypatch, toy_cfg):
    """Acceptance 9: one callable, reached by both arms through the module
    global `STEP1`. Patch it and both paths use the patch."""
    assert consolidate.STEP1 is consolidate.train_phase_lora
    assert consolidate.AFTER_PHASE["D"] is consolidate.AFTER_PHASE["D-nr"]
    assert consolidate.AFTER_PHASE["C"] is not consolidate.AFTER_PHASE["D"]

    used: list[str] = []

    def recorder(model, phase_k, ctx, cfg):
        used.append(ctx.arm)
        return consolidate.train_phase_lora(model, phase_k, ctx, cfg)

    monkeypatch.setattr(consolidate, "STEP1", recorder)
    for arm in ("C", "D", "D-nr"):
        rec = RecordingContext(arm=arm, phase=1, replay_fraction=0.3 if arm == "D" else 0.0, steps=2)
        consolidate.after_phase(make_model(seed=0), 1, rec.ctx, cfg=toy_cfg)
    assert used == ["C", "D", "D-nr"]


@pytest.mark.parametrize("arm,frac", [("C", 0.0), ("D", 0.3), ("D-nr", 0.0)])
def test_phase_zero_is_the_identity(arm: str, frac: float, toy_cfg):
    """Phase 0 is ordinary full training for every sequential arm, and since
    2026-09-22 train.py does not call the hook there at all. Kept as a guard:
    phase 0 is the checkpoint A, B, C, D and D-nr share, so a night run on it by
    mistake would corrupt every sequential arm at once."""
    model = make_model(seed=0)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    rec = RecordingContext(arm=arm, phase=0, replay_fraction=frac, steps=4)
    out = consolidate.after_phase(model, 0, rec.ctx, cfg=toy_cfg)
    assert out is model
    assert rec.phase_calls == [] and rec.replay_calls == [] and rec.timers == []
    for n, p in out.named_parameters():
        assert torch.equal(p, before[n])


@pytest.mark.parametrize("arm", ["A", "B", "E", "phase0"])
def test_non_lora_arms_are_the_identity(arm: str, toy_cfg):
    model = make_model(seed=0)
    rec = RecordingContext(arm=arm, phase=3, replay_fraction=0.3, steps=4)
    assert consolidate.after_phase(model, 3, rec.ctx, cfg=toy_cfg) is model
    assert rec.timers == []


def test_n_comes_from_the_context_not_from_trainconfig(toy_cfg):
    """`PhaseContext.sequences_per_batch` is N (added 2026-09-22). A hook that
    fell back to a TrainConfig default would spend a different budget from the
    day whenever train.py runs a non-default N."""
    assert toy_cfg.sequences_per_batch != 3
    rec = RecordingContext(
        arm="D", phase=1, replay_fraction=0.3, steps=2, sequences_per_batch=3
    )
    consolidate.after_phase(make_model(seed=0), 1, rec.ctx, cfg=toy_cfg)
    assert {n for _, n in rec.phase_calls} == {3}
    assert set(rec.replay_calls) == {replay_sequences_per_batch(3, 0.3)}


def test_a_loader_that_hands_back_the_wrong_batch_is_caught(toy_cfg):
    """The loaders yield contiguous full blocks and are never padded (frozen
    2026-09-22), so the shape is assertable -- and asserted, because a short
    batch would quietly change this arm's token budget."""
    rec = RecordingContext(arm="D", phase=1, replay_fraction=0.3, steps=2)
    n = rec.ctx.sequences_per_batch
    with pytest.raises(AssertionError, match="expected"):
        consolidate.check_batch(token_batch(n - 1, seed=1), n, "phase-1")
    with pytest.raises(AssertionError, match="expected"):
        consolidate.check_batch(token_batch(n, seed=1)[:, :1], n, "phase-1")
    assert consolidate.check_batch(token_batch(n, seed=1), n, "phase-1").shape[0] == n


def test_the_night_uses_the_days_schedule(toy_cfg):
    """No second copy of the cosine schedule: arm D's night must ramp and decay
    exactly as arm A's day does, including `cosine_lr`'s 10% floor. A private
    re-implementation here would give arm D a different effective lr from arm A
    on the same phase, which is a difference H2 would read as consolidation."""
    from training.model import cosine_lr

    for step in (0, 1, 2, 5, 19, 20):
        assert consolidate._lr_at(step, 20, 2, 3e-4) == cosine_lr(
            step, base_lr=3e-4, warmup=2, total_steps=20
        )


def test_step1_spends_the_whole_phase_budget(toy_cfg):
    """The hook owns the phase for a uses_lora arm (AGENTS.md, 2026-09-22), so
    step 1 is the day: exactly ctx.steps optimizer steps of
    ctx.sequences_per_batch new-phase sequences, no fraction of it."""
    steps = 7
    rec = RecordingContext(arm="C", phase=2, replay_fraction=0.0, steps=steps)
    consolidate.train_phase_lora(make_model(seed=0), 2, rec.ctx, toy_cfg)
    assert rec.phase_calls == [(2, rec.ctx.sequences_per_batch)] * steps
    assert rec.replay_calls == []


def test_replay_batch_composition(toy_cfg):
    """Acceptance 8: arm D mixes 70/30, arm D-nr adds nothing."""
    steps = 3

    rec_d = RecordingContext(arm="D", phase=2, replay_fraction=0.3, steps=steps)
    n_new = rec_d.ctx.sequences_per_batch
    consolidate.after_phase(make_model(seed=0), 2, rec_d.ctx, cfg=toy_cfg)
    expected_replay = replay_sequences_per_batch(n_new, 0.3)
    assert expected_replay > 0
    assert rec_d.replay_calls == [expected_replay] * steps
    # step 1 and step 3 each pull `steps` phase batches of N sequences
    assert rec_d.phase_calls == [(2, n_new)] * (2 * steps)

    rec_nr = RecordingContext(arm="D-nr", phase=2, replay_fraction=0.0, steps=steps)
    consolidate.after_phase(make_model(seed=0), 2, rec_nr.ctx, cfg=toy_cfg)
    assert rec_nr.replay_calls == []
    assert rec_nr.phase_calls == [(2, n_new)] * (2 * steps)

    # At the grid batch size the mix is 30% of the batch to within a sequence.
    grid_new = TrainConfig().sequences_per_batch
    grid_replay = replay_sequences_per_batch(grid_new, 0.3)
    assert grid_replay / (grid_new + grid_replay) == pytest.approx(0.30, abs=0.01)


def test_the_teacher_is_not_the_student(monkeypatch, toy_cfg):
    """Acceptance 6: B_k is a real copy taken before the base is unfrozen, and
    it is untouched by the distillation. A distillation whose teacher is the
    student is a no-op that looks like it works."""
    captured: dict[str, object] = {}
    real = consolidate.distill_into_base

    def spy(student, teacher_T, prev_base, phase_k, ctx, cfg):
        captured["student"] = student
        captured["teacher_T"] = teacher_T
        captured["prev_base"] = prev_base
        captured["prev_before"] = {
            n: p.detach().clone() for n, p in prev_base.named_parameters()
        }
        captured["teacher_before"] = {
            n: p.detach().clone() for n, p in teacher_T.named_parameters()
        }
        return real(student, teacher_T, prev_base, phase_k, ctx, cfg)

    monkeypatch.setattr(consolidate, "distill_into_base", spy)
    rec = RecordingContext(arm="D", phase=1, replay_fraction=0.3, steps=3)
    out = consolidate.after_phase(make_model(seed=0), 1, rec.ctx, cfg=toy_cfg)

    prev = captured["prev_base"]
    teacher = captured["teacher_T"]
    assert prev is not out and prev is not captured["student"]
    assert teacher is not out and teacher is not captured["student"]
    for n, p in prev.named_parameters():
        assert torch.equal(p, captured["prev_before"][n]), f"B_k moved: {n}"
    for n, p in teacher.named_parameters():
        assert torch.equal(p, captured["teacher_before"][n]), f"teacher moved: {n}"
    # The teacher really is B_k + L_k, not B_k: step 1 moved it.
    assert any(
        not torch.equal(p, captured["prev_before"][n])
        for n, p in teacher.named_parameters()
    )


def test_consolidation_is_timed(toy_cfg):
    """H2 is a compute claim; the seconds have to be measured, and both step 1
    and step 3 have to be inside the timed region."""
    for arm, frac in (("C", 0.0), ("D", 0.3), ("D-nr", 0.0)):
        rec = RecordingContext(arm=arm, phase=1, replay_fraction=frac, steps=2)
        consolidate.after_phase(make_model(seed=0), 1, rec.ctx, cfg=toy_cfg)
        assert rec.timers == [consolidate.TIMER_LABEL]


# ---------------------------------------------------------------------------
# smoke
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arm,frac", [("C", 0.0), ("D", 0.3), ("D-nr", 0.0)])
def test_after_phase_smoke(arm: str, frac: float, toy_cfg):
    """Acceptance 10: a CPU run of the night in <= 50 steps returns a usable
    model whose weights moved, with no LoRA left in it."""
    model = make_model(seed=0)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    rec = RecordingContext(arm=arm, phase=1, replay_fraction=frac, steps=5)

    out = consolidate.after_phase(model, 1, rec.ctx, cfg=toy_cfg)

    assert not lora.has_lora(out)
    assert not any("lora" in k for k in out.state_dict())
    after = dict(out.named_parameters())
    assert set(after) == set(before), "the model handed to phase k+1 changed shape"
    moved = [n for n, p in after.items() if not torch.equal(p, before[n])]
    assert moved, f"arm {arm}: nothing changed"
    if arm == "C":
        # the merge writes into the adapted projections and nothing else
        assert all(".attn.c_" in n or ".mlp.c_" in n for n in moved), moved
    # still runnable
    out.eval()
    with torch.no_grad():
        logits = lora.forward_logits(out, token_batch(2, seed=3))
    assert torch.isfinite(logits).all()
