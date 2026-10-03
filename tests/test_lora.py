"""LoRA: identity at init, and merge == forward.

These two are the premise of H3. If a fresh LoRA is not the identity, phase k
starts from a model nobody trained; if the merge does not equal the forward
pass, arm C ships a model that is not the one it trained, and arm D beats it for
a reason that has nothing to do with distillation.
"""

from __future__ import annotations

import pytest
import torch

from tests.conftest import make_model, token_batch
from training import lora
from training.config import TrainConfig


def test_targets_are_the_attention_and_mlp_projections():
    model = make_model()
    names = lora.find_target_modules(model)
    # 2 layers x 4 projections
    assert len(names) == 8
    assert set(n.split(".", 2)[-1] for n in names) == {
        "attn.c_attn",
        "attn.c_proj",
        "mlp.c_fc",
        "mlp.c_proj",
    }
    # Never the embeddings or the (tied) head: adapting those would change the
    # parameter count comparison between arms.
    assert not any("lm_head" in n or "wte" in n or "wpe" in n for n in names)


def test_no_targets_raises_rather_than_silently_adapting_nothing():
    model = torch.nn.Sequential(torch.nn.Linear(4, 4))
    with pytest.raises(ValueError, match="no LoRA target modules"):
        lora.apply_lora(model, rank=16, alpha=32, seed=0)


def test_fresh_lora_is_bitwise_identity():
    """Acceptance 1: B = 0, so the adapted model's logits are the base's."""
    cfg = TrainConfig()
    model = make_model(seed=0)
    x = token_batch(2, seed=11)
    model.eval()
    with torch.no_grad():
        before = lora.forward_logits(model, x).clone()
    lora.apply_lora(model, rank=cfg.lora_rank, alpha=cfg.lora_alpha, seed=0)
    with torch.no_grad():
        after = lora.forward_logits(model, x)
    assert torch.equal(before, after)
    # A is not trivially zero -- the identity comes from B, not from a dead init.
    a_norms = [m.lora_A.abs().sum().item() for _, m in lora.lora_modules(model)]
    assert all(n > 0 for n in a_norms)


def test_lora_a_init_is_seeded_and_reproducible():
    cfg = TrainConfig()
    m1 = lora.apply_lora(make_model(seed=0), cfg.lora_rank, cfg.lora_alpha, seed=3)
    m2 = lora.apply_lora(make_model(seed=0), cfg.lora_rank, cfg.lora_alpha, seed=3)
    m3 = lora.apply_lora(make_model(seed=0), cfg.lora_rank, cfg.lora_alpha, seed=4)
    a1 = dict(lora.lora_modules(m1))
    a2 = dict(lora.lora_modules(m2))
    a3 = dict(lora.lora_modules(m3))
    for name in a1:
        assert torch.equal(a1[name].lora_A, a2[name].lora_A)
        assert not torch.equal(a1[name].lora_A, a3[name].lora_A)


def _randomise_lora(model, seed: int = 7) -> None:
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for _, m in lora.lora_modules(model):
            m.lora_A.copy_(torch.empty_like(m.lora_A).normal_(0, 0.25, generator=g))
            m.lora_B.copy_(torch.empty_like(m.lora_B).normal_(0, 0.02, generator=g))


def test_merge_equals_forward():
    """Acceptance 2: the merge is exact.

    Randomise A and B, read the logits through the unmerged forward pass, merge
    with W += (alpha/rank) B A, read them again. A missing or doubled scaling
    factor shows up here and nowhere else.
    """
    cfg = TrainConfig()
    model = make_model(seed=0)
    lora.apply_lora(model, rank=cfg.lora_rank, alpha=cfg.lora_alpha, seed=0)
    _randomise_lora(model)
    model.eval()
    x = token_batch(2, seed=13)

    with torch.no_grad():
        unmerged = lora.forward_logits(model, x).clone()
    assert not torch.allclose(unmerged, lora.forward_logits(make_model(seed=0).eval(), x))

    lora.merge_lora(model)
    assert not lora.has_lora(model)
    with torch.no_grad():
        merged = lora.forward_logits(model, x)

    max_diff = (merged - unmerged).abs().max().item()
    scale = unmerged.abs().max().item()
    assert max_diff < 1e-4, f"merge != forward: max |diff| = {max_diff} (logits ~ {scale})"
    assert torch.allclose(merged, unmerged, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("wrong", [0.5, 2.0])
def test_wrong_scaling_breaks_the_merge(wrong: float):
    """The other half of acceptance 2: the test above can fail. Merging with
    alpha/rank scaled by 0.5 or 2 must NOT match the forward pass."""
    cfg = TrainConfig()
    model = make_model(seed=0)
    lora.apply_lora(model, rank=cfg.lora_rank, alpha=cfg.lora_alpha, seed=0)
    _randomise_lora(model)
    model.eval()
    x = token_batch(2, seed=13)
    with torch.no_grad():
        unmerged = lora.forward_logits(model, x).clone()
        for _, m in lora.lora_modules(model):
            m.base.weight.add_(wrong * m.merged_delta())
        merged = lora.forward_logits(model, x)
    assert not torch.allclose(merged, unmerged, atol=1e-4, rtol=1e-4)


def test_scaling_is_alpha_over_rank():
    cfg = TrainConfig()
    model = lora.apply_lora(make_model(), cfg.lora_rank, cfg.lora_alpha, seed=0)
    for _, m in lora.lora_modules(model):
        assert m.scaling == cfg.lora_alpha / cfg.lora_rank == 2.0
        delta = m.merged_delta()
        assert torch.allclose(delta, 2.0 * (m.lora_B @ m.lora_A))
        assert delta.shape == m.base.weight.shape


def test_apply_lora_freezes_everything_else():
    """Half of acceptance 3, at the module level."""
    cfg = TrainConfig()
    model = lora.apply_lora(make_model(), cfg.lora_rank, cfg.lora_alpha, seed=0)
    trainable = lora.assert_only_lora_trainable(model)
    assert len(trainable) == 16  # 8 adapted modules x (A, B)
    assert all(n.endswith(".lora_A") or n.endswith(".lora_B") for n in trainable)


def test_strip_without_merge_restores_the_base_exactly():
    """Arm D's step 3 starts from B_k, not from B_k plus a bit of L_k."""
    cfg = TrainConfig()
    base = make_model(seed=0)
    before = {n: p.detach().clone() for n, p in base.named_parameters()}
    lora.apply_lora(base, cfg.lora_rank, cfg.lora_alpha, seed=0)
    _randomise_lora(base)
    lora.strip_lora(base, merge=False)
    assert not lora.has_lora(base)
    after = dict(base.named_parameters())
    assert set(after) == set(before)
    for n, p in after.items():
        assert torch.equal(p, before[n]), n


def _train_embeddings_a_little(model):
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in lora.EMBEDDING_SUFFIXES:
                p.add_(0.01)


def test_trained_embeddings_join_the_adapter_set():
    """Amendment 6: with train_embeddings, step 1's trainable set is the LoRA
    parameters plus the token and position embeddings, and nothing else."""
    cfg = TrainConfig()
    model = lora.apply_lora(make_model(), cfg.lora_rank, cfg.lora_alpha, seed=0, train_embeddings=True)
    trainable = lora.assert_only_lora_trainable(model)
    assert set(trainable) - set(lora.lora_parameter_names(model)) == set(lora.EMBEDDING_SUFFIXES)
    # the snapshot is bookkeeping, never part of a checkpoint
    assert not any("snapshot" in k for k in model.state_dict())


def test_discard_restores_trained_embeddings_exactly():
    """Arm D: the student starts from B_k, embeddings included."""
    cfg = TrainConfig()
    base = make_model(seed=0)
    before = {n: p.detach().clone() for n, p in base.named_parameters()}
    lora.apply_lora(base, cfg.lora_rank, cfg.lora_alpha, seed=0, train_embeddings=True)
    _randomise_lora(base)
    _train_embeddings_a_little(base)
    lora.strip_lora(base, merge=False)
    after = dict(base.named_parameters())
    for n, p in after.items():
        assert torch.equal(p, before[n]), n


def test_merge_keeps_trained_embeddings():
    """Arm C and arm D's teacher: what step 1 learned in the embeddings stays."""
    cfg = TrainConfig()
    base = make_model(seed=0)
    before = {n: p.detach().clone() for n, p in base.named_parameters()}
    lora.apply_lora(base, cfg.lora_rank, cfg.lora_alpha, seed=0, train_embeddings=True)
    _train_embeddings_a_little(base)
    lora.merge_lora(base)
    after = dict(base.named_parameters())
    for n in lora.EMBEDDING_SUFFIXES:
        assert torch.allclose(after[n], before[n] + 0.01), n
    assert not hasattr(base, "_lora_embedding_snapshot")


def test_discard_means_gone():
    cfg = TrainConfig()
    model = lora.apply_lora(make_model(), cfg.lora_rank, cfg.lora_alpha, seed=0)
    lora.merge_lora(model)
    assert lora.lora_parameter_names(model) == []
    assert not any("lora" in k for k in model.state_dict())
