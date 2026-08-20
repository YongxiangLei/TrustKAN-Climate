# Reproducible GPU execution

## Environment gate

Use a fresh environment with a CUDA-enabled PyTorch build compatible with the
target NVIDIA driver. A machine having an NVIDIA GPU is not sufficient:
`torch.cuda.is_available()` must be true in the exact Python environment used
for the experiments.

Before launching any neural shard, run:

```bash
python scripts/check_gpu_environment.py \
  --device cuda:0 \
  --out results/campaigns/ghcn_publication/gpu_environment.json
```

The preflight fails if CUDA is unavailable, the device index is invalid, or two
identically seeded convolutional training replays differ. It records the exact
Python, NumPy, pandas, PyTorch, CUDA and cuDNN versions; GPU name, compute
capability and memory; deterministic backend flags; optional `nvidia-smi`
metadata; and the preflight code hash. A final publication campaign is not
complete without an eligible preflight record.

## Frozen numerical policy

Publication and smoke configs explicitly set:

```yaml
training:
  deterministic_algorithms: true
  deterministic_warn_only: false
```

The runner configures seeded Python/NumPy/PyTorch generators,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, deterministic PyTorch algorithms, disabled
cuDNN benchmarking, and deterministic cuDNN behavior. Unsupported
nondeterministic operations fail rather than degrade to a warning. Automatic
mixed precision is intentionally disabled; introducing it would define a new
experiment and require rerunning all affected comparisons.

## CPU/GPU campaign separation

Generate the formal campaign on the execution environment:

```bash
python scripts/plan_ghcn_campaign.py \
  --outdir results/campaigns/ghcn_publication \
  --gpu-device cuda:0
python scripts/plan_cet_campaign.py \
  --outdir results/campaigns/cet_publication \
  --gpu-device cuda:0
python scripts/plan_jena_campaign.py \
  --outdir results/campaigns/jena_publication \
  --gpu-device cuda:0
```

The output contains:

- `run_cpu.ps1`: persistence, SVR and random-forest shards;
- `run_gpu.ps1`: GPU preflight followed by neural and reliability shards;
- `collect.ps1`: ledger reconstruction, five-seed/five-region aggregation and
  the final campaign audit;
- `run_all.ps1`: the same operations in one fail-fast sequential script.

CPU and GPU scripts may run on different machines if the standard `results/`
artifact and record paths are synchronized before collection. The manifest's
argument vectors are the source of truth; do not edit the frozen YAML configs
to allocate hardware.

For multiple GPUs, partition entries marked `accelerator: gpu` across workers.
Expose one physical GPU to each worker with `CUDA_VISIBLE_DEVICES`; inside that
worker the manifest's `cuda:0` remains correct. One process per GPU is the safe
default. Never run the same shard concurrently against one result directory.

## Fairness and out-of-memory handling

Use one GPU family and one software environment for the primary neural matrix
where practical. If hardware families must be mixed, preserve the recorded
metadata and treat hardware sensitivity as a limitation rather than assuming
bitwise cross-device equivalence.

Do not reduce batch size, enable mixed precision, or shorten training only for
a model that encounters out-of-memory errors. Such a change alters the frozen
protocol. Define it prospectively, update the configuration, and rerun every
affected comparison under the same policy.

Smoke values and CPU fallback checks remain engineering evidence only.
