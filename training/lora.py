"""LoRA for arms C and D: rank 16 on the attention and MLP projections,
alpha 32 (PLAN.md, "Model and compute").

The one number that has to be right in two places is the scaling factor
``alpha / rank``. The forward pass computes ``W x + (alpha/rank) B A x`` and the
merge writes ``W <- W + (alpha/rank) B A``; if the two disagree, arm C's merge
is not the model arm C trained, H3 compares the wrong things, and nothing
crashes. `tests/test_lora.py::test_merge_equals_forward` is the check.

Adapters are attached by *name pattern*, never by importing the model module,
so this file does not depend on `training/model.py`.
"""

from __future__ import annotations

import math
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

#: Dotted module-name suffixes that get an adapter: the attention and MLP
#: projections (PLAN.md). GPT-2 naming first, then the aliases a differently
#: named decoder might use. A module qualifies only if its *dotted suffix*
#: matches, so an unrelated `c_proj` outside `attn`/`mlp` is not adapted, and
#: neither is the embedding or the tied LM head.
TARGET_SUFFIXES: tuple[str, ...] = (
    # GPT-2 style (what training/model.py is expected to use)
    "attn.c_attn",
    "attn.c_proj",
    "mlp.c_fc",
    "mlp.c_proj",
    # aliases, so a rename in model.py fails loudly rather than silently
    # adapting nothing
    "attn.q_proj",
    "attn.k_proj",
    "attn.v_proj",
    "attn.o_proj",
    "attn.qkv",
    "attn.out_proj",
    "mlp.fc1",
    "mlp.fc2",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def forward_logits(model: nn.Module, idx: torch.Tensor) -> torch.Tensor:
    """Logits for a batch of token ids, tolerant of the calling convention
    `training/model.py` settles on: a bare tensor, a `(logits, loss)` tuple
    (nanoGPT), or an object with a `.logits` attribute (HF)."""
    out = model(idx)
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)):
        return out[0]
    logits = getattr(out, "logits", None)
    if logits is None:
        raise TypeError(f"cannot find logits in model output of type {type(out)!r}")
    return logits


class LoRALinear(nn.Module):
    """`nn.Linear` plus a rank-`r` update, exactly identity at initialisation.

    y = W x + b + (alpha / r) * B (A x)

    `A` is [r, in] drawn from N(0, 1/r); `B` is [out, r] and starts at zero, so
    the adapted model's logits are bitwise the base model's until the first
    optimizer step. The base `nn.Linear` is kept as a child module and frozen.
    """

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: int,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        #: The scaling used by *both* the forward pass and `merged_delta()`.
        self.scaling = self.alpha / self.rank

        weight = base.weight
        out_features, in_features = weight.shape
        self.lora_A = nn.Parameter(
            torch.empty(self.rank, in_features, dtype=weight.dtype, device=weight.device)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(out_features, self.rank, dtype=weight.dtype, device=weight.device)
        )
        self.reset_parameters(generator)

        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

    @torch.no_grad()
    def reset_parameters(self, generator: torch.Generator | None = None) -> None:
        """A ~ N(0, 1/rank) (std = rank**-0.5), B = 0.

        Drawn from `generator`, which the caller seeds from the run's seed, so a
        rerun reproduces the adapter's initialisation.
        """
        std = 1.0 / math.sqrt(self.rank)
        sample = torch.empty(
            self.lora_A.shape, dtype=torch.float32, device="cpu"
        ).normal_(mean=0.0, std=std, generator=generator)
        self.lora_A.copy_(sample.to(self.lora_A.dtype).to(self.lora_A.device))
        self.lora_B.zero_()

    def merged_delta(self) -> torch.Tensor:
        """(alpha / rank) * B @ A, shaped like `base.weight` [out, in]."""
        return self.scaling * (self.lora_B @ self.lora_A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The same `self.scaling` the merge uses. Written as two matmuls so the
        # r-dimensional bottleneck is never materialised as a full weight.
        delta = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return self.base(x) + delta * self.scaling

    def extra_repr(self) -> str:  # pragma: no cover - debugging convenience
        return f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling}"


def find_target_modules(
    model: nn.Module, suffixes: tuple[str, ...] = TARGET_SUFFIXES
) -> list[str]:
    """Names of the `nn.Linear` modules that get an adapter."""
    names: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(name == s or name.endswith("." + s) for s in suffixes):
            names.append(name)
    return names


def _get_parent(model: nn.Module, dotted: str) -> tuple[nn.Module, str]:
    parts = dotted.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


#: Parameter-name suffixes of the token and position embeddings (the LM head is
#: tied to `wte`, so it moves with it). Trained in step 1 when
#: `train_embeddings=True` (docs/DECISIONS.md 2026-10-02: with them frozen, a
#: rank-16 or rank-64 adapter learns a new phase's vocabulary barely at all).
EMBEDDING_SUFFIXES: tuple[str, ...] = ("wte.weight", "wpe.weight")

#: Attribute holding step 1's starting embeddings, so a discard can restore
#: B_k exactly. Deliberately not a buffer: it never reaches a state_dict.
_SNAPSHOT_ATTR = "_lora_embedding_snapshot"


def embedding_parameter_names(model: nn.Module) -> list[str]:
    return [
        n for n, _ in model.named_parameters()
        if any(n == s or n.endswith("." + s) for s in EMBEDDING_SUFFIXES)
    ]


def apply_lora(
    model: nn.Module,
    rank: int,
    alpha: int,
    seed: int,
    suffixes: tuple[str, ...] = TARGET_SUFFIXES,
    train_embeddings: bool = False,
) -> nn.Module:
    """Attach adapters in place, freeze everything else, return the model.

    After this call *only* the adapter set has `requires_grad` (asserted by
    `assert_only_lora_trainable`): the LoRA parameters, plus the embeddings when
    `train_embeddings` is set. In that case the embeddings' starting values are
    snapshotted so `strip_lora(merge=False)` can put B_k back exactly. Raises if
    no module matched: a LoRA that adapts nothing trains nothing and would
    silently turn arm C and arm D into arm A with a wasted forward pass.
    """
    targets = find_target_modules(model, suffixes)
    if not targets:
        raise ValueError(
            "no LoRA target modules found; expected attention and MLP projections "
            f"with dotted suffixes {suffixes}. Adapting nothing would silently "
            "make arms C and D no-ops."
        )
    for p in model.parameters():
        p.requires_grad_(False)

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    for name in targets:
        parent, attr = _get_parent(model, name)
        base = getattr(parent, attr)
        setattr(parent, attr, LoRALinear(base, rank=rank, alpha=alpha, generator=generator))
    if train_embeddings:
        names = embedding_parameter_names(model)
        if not names:
            raise ValueError(f"train_embeddings=True but no parameter ends with {EMBEDDING_SUFFIXES}")
        params = dict(model.named_parameters())
        setattr(model, _SNAPSHOT_ATTR, {n: params[n].detach().clone() for n in names})
        for n in names:
            params[n].requires_grad_(True)
    return model


def lora_modules(model: nn.Module) -> Iterator[tuple[str, LoRALinear]]:
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            yield name, module


def lora_parameter_names(model: nn.Module) -> list[str]:
    return [n for n, p in model.named_parameters() if ".lora_A" in n or ".lora_B" in n]


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for n, p in model.named_parameters() if ".lora_A" in n or ".lora_B" in n]


def has_lora(model: nn.Module) -> bool:
    return any(True for _ in lora_modules(model))


def adapter_parameter_names(model: nn.Module) -> list[str]:
    """Step 1's trainable set: the LoRA parameters, plus the embeddings when
    `apply_lora(..., train_embeddings=True)` snapshotted them."""
    names = lora_parameter_names(model)
    if getattr(model, _SNAPSHOT_ATTR, None):
        names += embedding_parameter_names(model)
    return names


def adapter_parameters(model: nn.Module) -> list[nn.Parameter]:
    params = dict(model.named_parameters())
    return [params[n] for n in adapter_parameter_names(model)]


def assert_only_lora_trainable(model: nn.Module) -> list[str]:
    """Invariant for step 1: the trainable set is exactly the adapter set."""
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    if not lora_parameter_names(model):
        raise AssertionError("no LoRA parameters found; apply_lora was not called")
    expected = set(adapter_parameter_names(model))
    if not expected:
        raise AssertionError("no LoRA parameters found; apply_lora was not called")
    if trainable != expected:
        extra = sorted(trainable - expected)
        missing = sorted(expected - trainable)
        raise AssertionError(
            f"only LoRA parameters may require grad during step 1; "
            f"unexpectedly trainable: {extra}; unexpectedly frozen: {missing}"
        )
    return sorted(expected)


@torch.no_grad()
def strip_lora(model: nn.Module, merge: bool) -> nn.Module:
    """Remove every adapter, returning the plain module tree.

    `merge=True` folds the update into the base weight first
    (``W += (alpha/rank) B A``) -- arm C's night. `merge=False` discards it and
    leaves the base exactly as it was -- arm D's step 3 starts from B_k.

    Either way no LoRA parameter survives, so nothing reaches the optimizer or
    the checkpoint ("discard means gone"). Embeddings trained in step 1 follow
    the same rule: kept on a merge, restored to their snapshot on a discard.
    """
    for name, module in list(lora_modules(model)):
        base = module.base
        if merge:
            base.weight.add_(module.merged_delta().to(base.weight.dtype))
        parent, attr = _get_parent(model, name)
        setattr(parent, attr, base)
    snapshot = getattr(model, _SNAPSHOT_ATTR, None)
    if snapshot is not None:
        if not merge:
            params = dict(model.named_parameters())
            for n, value in snapshot.items():
                params[n].copy_(value.to(params[n].device, params[n].dtype))
        delattr(model, _SNAPSHOT_ATTR)
    return model


def merge_lora(model: nn.Module) -> nn.Module:
    """W += (alpha / rank) * B @ A for every adapter, then drop the adapters."""
    return strip_lora(model, merge=True)


def unfreeze_all(model: nn.Module) -> nn.Module:
    for p in model.parameters():
        p.requires_grad_(True)
    return model


def freeze_all(model: nn.Module) -> nn.Module:
    for p in model.parameters():
        p.requires_grad_(False)
    return model
