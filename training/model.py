"""The ~30M GPT-2-style decoder every arm trains.

Shape comes from `PLAN.md`, "Model and compute": 8 layers, 512 hidden, 8 heads,
1,024-token context, and a small BPE vocab so the embedding table does not
dominate. `training.config.REAL` holds the numbers and `ModelConfig.n_params()`
holds the arithmetic the lead corrected on 2026-09-22 (29,901,824 total). This
file realises that arithmetic exactly, which fixes three conventions:

- **No biases** in any Linear. The per-layer count is `4*d^2 + 2*d*4d` --
  attention qkv+proj and MLP fc+proj, and nothing else.
- **Affine LayerNorm**, the ordinary GPT-2 kind: two vectors per norm, two norms
  per block plus the final one, which is the 17,408 parameters the first frozen
  count omitted (AGENTS.md, Amendments 2026-09-22).
- **Tied embedding / output head**, as `n_params()` counts the vocab table once.

Module names are part of the surface `training/lora.py` adapts. The four
projections PLAN.md names ("attention and MLP projections") are reachable as
`blocks.{i}.attn.c_attn`, `blocks.{i}.attn.c_proj`, `blocks.{i}.mlp.c_fc`,
`blocks.{i}.mlp.c_proj` -- GPT-2's own names, which is what `training/lora.py`
looks for first -- and `LORA_TARGET_SUFFIXES` lists them.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.config import ModelConfig

#: The Linear module names `training/lora.py` wraps (PLAN.md: "rank 16 on
#: attention and MLP projections"). Suffixes of `named_modules()` keys.
LORA_TARGET_SUFFIXES: tuple[str, ...] = ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj")


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if cfg.n_embd % cfg.n_head != 0:
            raise ValueError(f"n_embd {cfg.n_embd} not divisible by n_head {cfg.n_head}")
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.dropout = cfg.dropout
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        head = c // self.n_head
        q = q.view(b, t, self.n_head, head).transpose(1, 2)
        k = k.view(b, t, self.n_head, head).transpose(1, 2)
        v = v.view(b, t, self.n_head, head).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.resid_drop(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.c_proj(F.gelu(self.c_fc(x), approximate="tanh")))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    """Decoder-only transformer. `forward(idx)` returns logits; `forward(idx,
    targets)` also returns the mean cross-entropy over the target positions."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # tied; counted once in parameters()
        # Torch's default Embedding init is N(0, 1), which makes a fresh model's
        # logits absurd. Init here from the global RNG so a directly-constructed
        # GPT is sane; `build_model` re-runs it with the run's init generator.
        self.init_weights(None)

    # -- init ---------------------------------------------------------------

    def init_weights(self, generator: torch.Generator | None = None) -> "GPT":
        """Deterministic GPT-2 init from an explicit generator.

        The generator is the *init* generator (`train.py` keeps a separate one
        for data order), so adding an arm's extra draw cannot shift another
        arm's weights. LayerNorm gains stay at 1 and biases at 0 -- torch's own
        defaults, and drawing them would consume generator state for nothing.
        """
        std = 0.02
        resid_std = 0.02 / math.sqrt(2 * self.cfg.n_layer)
        with torch.no_grad():
            self.wte.weight.normal_(mean=0.0, std=std, generator=generator)
            self.wpe.weight.normal_(mean=0.0, std=std, generator=generator)
            for block in self.blocks:
                block.attn.c_attn.weight.normal_(mean=0.0, std=std, generator=generator)
                block.attn.c_proj.weight.normal_(mean=0.0, std=resid_std, generator=generator)
                block.mlp.c_fc.weight.normal_(mean=0.0, std=std, generator=generator)
                block.mlp.c_proj.weight.normal_(mean=0.0, std=resid_std, generator=generator)
        return self

    # -- forward ------------------------------------------------------------

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        b, t = idx.shape
        if t > self.cfg.block_size:
            raise ValueError(f"sequence length {t} exceeds block_size {self.cfg.block_size}")
        pos = torch.arange(t, dtype=torch.long, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1), reduction="mean"
        )
        return logits, loss

    # -- counting -----------------------------------------------------------

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def parameter_split(self) -> tuple[int, int]:
        """(non-embedding, embedding), matching `ModelConfig.n_params()`."""
        emb = self.wte.weight.numel() + self.wpe.weight.numel()
        return self.num_parameters() - emb, emb

    # -- optimiser ----------------------------------------------------------

    def configure_optimizers(
        self,
        lr: float,
        weight_decay: float,
        betas: tuple[float, float],
        device_type: str = "cpu",
    ) -> torch.optim.AdamW:
        """AdamW (PLAN.md's table). Decay on matrices and embeddings, not on
        anything 1-D; there are no biases and no affine norms here, so the
        no-decay group is normally empty and kept only so a LoRA-wrapped or
        otherwise extended model still gets the right grouping."""
        decay, no_decay = [], []
        for _, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        fused_ok = device_type == "cuda" and "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
        kwargs = {"fused": True} if fused_ok else {}
        return torch.optim.AdamW(groups, lr=lr, betas=betas, **kwargs)


def build_model(cfg: ModelConfig, generator: torch.Generator | None = None) -> GPT:
    """A freshly initialised model. `generator` makes the init reproducible."""
    return GPT(cfg).init_weights(generator)


def cosine_lr(step: int, *, base_lr: float, warmup: int, total_steps: int, min_ratio: float = 0.1) -> float:
    """Linear warmup then cosine decay, restarted at every phase (PLAN.md:
    "Restart the schedule at each phase so every arm sees the same per-phase
    budget"). `step` is 0-based within the phase.

    Replay does not enter here: it enlarges the batch, not the step count, so
    the schedule is byte-identical across arms with the same data.
    """
    if total_steps <= 0:
        return 0.0
    if warmup > 0 and step < warmup:
        return base_lr * (step + 1) / warmup
    denom = max(1, total_steps - warmup)
    progress = min(1.0, max(0.0, (step - warmup) / denom))
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_ratio + (1.0 - min_ratio) * coeff)
