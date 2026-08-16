# GHCN publication campaign execution

## Purpose

The frozen GHCN paper matrix contains 1,260 deterministic-forecasting runs and
150 TrustKAN reliability runs. It is too large to treat as one fragile serial
process. Execution filters therefore route immutable jobs without editing the
frozen YAML files or changing their configuration hashes.

Generate the checksum-bound campaign manifest and PowerShell runner with:

```bash
python scripts/plan_ghcn_campaign.py \
  --benchmark-config configs/ghcn.yaml \
  --reliability-config configs/ghcn_reliability.yaml \
  --outdir results/campaigns/ghcn_publication
```

The planner fails unless the publication configuration retains all five
regions, both protocols, horizons 1/7/30, five aligned seeds, the required
baseline suite, and the frozen conformal/reliability policy. It produces:

- `campaign.json`: hashes, publication checks, the exact matrix, argument
  vectors, and expected atomic-run counts;
- `run_all.ps1`: a fail-fast sequential execution path followed by collection,
  strict aggregation, and a completion audit.

## Sharded execution

Each entry under `benchmark_jobs` or `reliability_jobs` in `campaign.json` is an
independent shard. A typical deterministic shard is:

```bash
python scripts/run_ghcn_benchmark.py \
  --config configs/ghcn.yaml --resume --defer-collection \
  --horizon 1 --model trustkan --seed 11
```

A reliability shard omits `--model`. Protocol and target-region filters are
also available for finer routing. Seed `-1` is reserved for deterministic
models such as persistence and SVR. Do not create modified copies of the full
configuration to distribute jobs: doing so changes the experiment identity.

Shards write unique raw artifacts and atomic JSON run records. Config and panel
snapshots are also replaced atomically, so non-overlapping shards may execute
on machines that share a synchronized result directory. Do not launch the same
shard concurrently in that directory.

## Collection and progress audit

After copying all run records and artifacts into their standard result paths,
rebuild ledgers and enforce the paper gates:

```bash
python scripts/run_ghcn_benchmark.py \
  --config configs/ghcn.yaml --collect-only
python scripts/aggregate_results.py \
  --input results/aggregated/ghcn_full_runs.csv \
  --outdir results/tables/ghcn_full --min-seeds 5 --min-regions 5

python scripts/run_ghcn_reliability.py \
  --config configs/ghcn_reliability.yaml --collect-only
python scripts/aggregate_reliability.py \
  --input results/reliability/aggregated/ghcn_reliability_full_runs.csv \
  --outdir results/tables/ghcn_reliability_full \
  --min-seeds 5 --min-regions 5
```

At any time, verify current hashes, artifact checksums, failures, runtime
summaries, and completion fractions:

```bash
python scripts/audit_ghcn_campaign.py \
  --campaign results/campaigns/ghcn_publication/campaign.json \
  --out results/campaigns/ghcn_publication/progress.json
```

Add `--require-complete` only for the final publication gate. Incomplete
campaigns are valid progress states but cannot support manuscript claims. Any
change to the frozen configuration or hashed runner/source code makes prior
records stale; they are excluded rather than silently reused.

## Engineering validation

Smoke configurations require the planner's explicit `--engineering` flag. Its
collection thresholds are derived from the smaller smoke matrix, but the
resulting manifest has `publication_campaign: false`. Smoke metrics remain
prohibited from paper tables.
