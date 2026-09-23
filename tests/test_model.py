"""The model: the frozen parameter count, determinism, and that loss falls."""

from __future__ import annotations

import math

import pytest
import torch

from training.config import REAL, TOY, TrainConfig, warmup_for
from training.train import TOY_RUN
from training.model import LORA_TARGET_SUFFIXES, GPT, build_model, cosine_lr


def test_real_config_has_the_frozen_parameter_count():
    """Acceptance check 1: config.REAL built as a model is 29,901,824 params.

    Corrected 2026-09-22 (AGENTS.md, Amendments): 8 layers / 512 hidden /
    8 heads / 1,024 context gives 25,165,824 in the projections plus 17,408 in
    the affine LayerNorm vectors -- (2*8 + 1) norms x 2 vectors x 512 -- for
    25,183,232 non-embedding, and vocab 8,192 adds 4,718,592. The first frozen
    count, 29,884,416, omitted the norms.
    """
    model = GPT(REAL)
    assert model.num_parameters() == 29_901_824
    non_emb, emb = model.parameter_split()
    assert (non_emb, emb) == (25_183_232, 4_718_592)
    assert (non_emb, emb) == REAL.n_params()
    assert non_emb - 8 * (4 * 512**2 + 2 * 512 * 4 * 512) == 17_408  # the norms


def test_layer_norms_are_affine_and_linears_carry_no_bias():
    """The shape the corrected arithmetic assumes, asserted directly rather
    than inferred from a total that several mistakes could produce."""
    model = GPT(TOY)
    norms = [m for m in model.modules() if isinstance(torch.nn.LayerNorm(1), type(m)) or isinstance(m, torch.nn.LayerNorm)]
    assert len(norms) == 2 * TOY.n_layer + 1
    for n in norms:
        assert n.elementwise_affine and n.weight is not None and n.bias is not None
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            assert m.bias is None


def test_toy_config_is_the_frozen_toy_size():
    model = GPT(TOY)
    assert model.num_parameters() == 119_424
    assert model.num_parameters() < 1_000_000  # AGENTS.md's ceiling for a smoke test


def test_output_head_is_tied_to_the_embedding():
    model = GPT(TOY)
    assert model.lm_head.weight is model.wte.weight


def test_lora_targets_all_exist():
    """training/lora.py adapts these names; if they move, it breaks silently."""
    model = GPT(TOY)
    names = {n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)}
    for layer in range(TOY.n_layer):
        for suffix in LORA_TARGET_SUFFIXES:
            assert f"blocks.{layer}.{suffix}" in names


def test_forward_shapes_and_loss():
    model = GPT(TOY)
    x = torch.randint(0, TOY.vocab_size, (3, 16))
    logits, loss = model(x, x)
    assert logits.shape == (3, 16, TOY.vocab_size)
    assert loss.ndim == 0
    # A freshly initialised model is near-uniform: a bit under ln(vocab),
    # because the output head is tied to the (random) embedding table.
    assert 0.5 * math.log(TOY.vocab_size) < float(loss.detach()) <= math.log(TOY.vocab_size) + 0.1


def test_forward_refuses_a_sequence_longer_than_the_context():
    model = GPT(TOY)
    with pytest.raises(ValueError, match="block_size"):
        model(torch.zeros((1, TOY.block_size + 1), dtype=torch.long))


def test_init_is_a_pure_function_of_the_generator():
    """The same seed gives every arm the same initial weights."""
    g1 = torch.Generator().manual_seed(1234)
    g2 = torch.Generator().manual_seed(1234)
    g3 = torch.Generator().manual_seed(5678)
    a, b, c = build_model(TOY, g1), build_model(TOY, g2), build_model(TOY, g3)
    drawn = 0
    for (na, pa), (_, pb), (_, pc) in zip(a.named_parameters(), b.named_parameters(), c.named_parameters()):
        assert torch.equal(pa, pb), na
        if ".ln_" in na or na.startswith("ln_"):
            # LayerNorm gains and biases are constants (1 and 0), not draws:
            # they are identical at every seed by design, and drawing them
            # would consume init-generator state for nothing.
            assert torch.equal(pa, pc), na
            continue
        assert not torch.equal(pa, pc), na
        drawn += 1
    assert drawn == 2 + 4 * TOY.n_layer  # wte, wpe, and four projections a block


def test_cosine_schedule_warms_up_then_decays_and_restarts():
    kw = dict(base_lr=3e-4, warmup=2, total_steps=20)
    assert cosine_lr(0, **kw) == pytest.approx(1.5e-4)
    assert cosine_lr(1, **kw) == pytest.approx(3e-4)
    assert cosine_lr(2, **kw) == pytest.approx(3e-4)
    assert cosine_lr(19, **kw) < cosine_lr(10, **kw) < cosine_lr(2, **kw)
    # Restarting the schedule at each phase means step 0 of every phase is the
    # same lr (PLAN.md: "Restart the schedule at each phase").
    assert cosine_lr(0, **kw) == cosine_lr(0, **kw)


def test_loss_falls_on_the_toy_config(toy_dirs):
    """Acceptance check 2: TOY config, the 20-story synthetic fixture, <= 50
    steps, and the loss goes down."""
    from training.data import DataModule, shift_for_lm
    from training.tokenizer import ByteTokenizer

    data = DataModule(toy_dirs["train"], ByteTokenizer(), TOY.block_size, seed=0)
    model = build_model(TOY, torch.Generator().manual_seed(0))
    tcfg = TrainConfig()
    opt = model.configure_optimizers(tcfg.lr, tcfg.weight_decay, tcfg.betas)

    steps = TOY_RUN.max_steps
    assert steps <= 50
    losses = []
    for step, batch in enumerate(data.make_phase_loader(0, TOY_RUN.sequences_per_batch, steps=steps)):
        x, y = shift_for_lm(batch)
        _, loss = model(x, y)
        loss.backward()
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, base_lr=tcfg.lr, warmup=warmup_for(steps, scaled=True), total_steps=steps)
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(float(loss.detach()))

    assert len(losses) == steps
    first, last = sum(losses[:5]) / 5, sum(losses[-5:]) / 5
    assert last < first, f"loss did not fall: {first:.4f} -> {last:.4f}"
