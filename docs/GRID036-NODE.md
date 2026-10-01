# grid036 rented-node run instructions

This is the full-size 0/3/6 experiment: 21 invocations, seven arms and three
seeds. Final training counts are 4,961 / 4,373 / 4,364; replay has 500 per phase.
Do not pass --pilot. One independent process uses each GPU; this is not DDP.

## Transfer and checkout

Keep `grid036-private.zip` private. It contains frozen exam text. Transfer it
from the local `D:\Lifespan\node_bundle` directory by SCP or the provider's
private upload facility. Never upload it to GitHub or a public dataset.
Unpack it under `/workspace/grid036` using Python's zipfile CLI:

```console
python -m zipfile -e /workspace/grid036-private.zip /workspace/grid036
git clone https://github.com/P1ayer-1/Lifespan.git /workspace/Lifespan
```

Check out the exact
40-character training commit recorded in the bundle's `release.json`:

```console
git -C /workspace/Lifespan checkout --detach <training_commit>
```

Start with a CUDA-enabled PyTorch environment supplied by the node image.
From `/workspace/Lifespan`, install requirements with `python -m pip install
-r requirements.txt`. These are minimum versions, not a lockfile. Record the
resolved runtime before starting, keep it unchanged across all arms, and retain
`python -m pip freeze` output alongside results.

## Preflight

Run `python /workspace/grid036/verify_bundle.py /workspace/grid036` to verify
all transferred file hashes. Then, from `/workspace/Lifespan`:

```console
python -m training.guard --train-dir /workspace/grid036/train --exam-dir /workspace/grid036/exam --manifest /workspace/grid036/exam/manifest.json --experiment-id lifespan-grid036
nvidia-smi
python -c "import torch; assert torch.cuda.device_count()==8; print(torch.__version__, torch.version.cuda); print([torch.cuda.get_device_name(i) for i in range(8)])"
python -m pip freeze > /workspace/grid036/runtime.txt
python -m runbook.grid --gpus 0,1,2,3,4,5,6,7 --phases 0,3,6 --micro-batch 8 --experiment-id lifespan-grid036 --manifest /workspace/grid036/exam/manifest.json --train-dir /workspace/grid036/train --exam-dir /workspace/grid036/exam --results-root /workspace/grid036/results --phase0-dir /workspace/grid036/shared --state-path /workspace/grid036/state.json --ledger-path /workspace/grid036/ledger.json --dry-run
```

Inspect that the dry run lists 21 invocations. All eight devices must be RTX
3090s with usable memory. The training code selects supported precision.
Micro-batch 8 is a memory setting, with gradient accumulation preserving the
full batch. If it fails for memory, reduce it to 4 and restart the same command;
do not change model sizes, epochs, learning rates or exams.

## Launch and resume

Use a persistent terminal (for example tmux). Set the allocator in the same
terminal, then repeat the grid command above without `--dry-run`:

```console
export PYTORCH_ALLOC_CONF=expandable_segments:True
```

The runner prepares shared initialization per seed before parallel dispatch,
waits for each seed's phase0 before dependent arms, records state and validates
each result. Logs live under `results/_logs`. Repeating the identical command
resumes interrupted runs and skips completed valid runs. Keep the same commit,
input files, runtime, state path and shared directory. Never restore pre-pilot
checkpoints here. Measure throughput from the first completed phase before
estimating remaining rental time; no fixed runtime is promised.

## Finish before terminating the node

Run `python -m results.validate --results-root /workspace/grid036/results`.
Require the grid summary to say 21 invocations with zero not complete; the
validator alone also succeeds when no folders exist, so it is not sufficient.
Download results, shared checkpoints, state.json, ledger.json, runtime.txt and
scheduler logs to local storage. Verify transferred archive hashes and re-run
result validation locally. Only then terminate the rental. Keep exam data private
when archiving or sharing results.

## Provenance

The frozen manifest SHA256 is
`25cee546e68cbc371b3add0ac02d93ab326a02a8a89e3e5c1ece8a6fbf8cb031`.
Its generator dirty flag records the actual state at freeze time and remains
unchanged. Subsequent commits archive the corpus/review evidence; they do not
retroactively make generation clean. The training checkout itself must be clean.
Phase 0 passed a 10% audit; phases 3 and 6 received full factual review. Age fit
was sampled for every phase. Review does not guarantee every remaining claim.

