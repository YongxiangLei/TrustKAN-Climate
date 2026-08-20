# Jena publication campaign execution

The frozen hourly Jena paper matrix contains 126 deterministic-forecasting runs
and 15 TrustKAN reliability runs. Generate the checksum-bound campaign with:

```bash
python scripts/plan_jena_campaign.py \
  --benchmark-config configs/jena.yaml \
  --reliability-config configs/jena_reliability.yaml \
  --outdir results/campaigns/jena_publication
```

The planner fails unless the publication configuration retains horizons
1/6/24, history 168, five aligned seeds, the required baseline suite, the
shared frozen panel, and the frozen conformal/reliability policy. It writes
`campaign.json`, `run_cpu.ps1`, `run_gpu.ps1`, `collect.ps1`, and `run_all.ps1`.

Neural and reliability shards must pass the CUDA preflight in
`docs/GPU_EXECUTION.md`. CPU fallback results cannot fill GPU campaign cells.
Smoke configurations require `--engineering` and remain prohibited from paper
tables. Official 10-minute archives must first pass `scripts/audit_jena.py`
and `scripts/prepare_jena.py`; the campaign planner does not invent those
eligibility numbers.

```bash
python scripts/audit_jena_campaign.py \
  --campaign results/campaigns/jena_publication/campaign.json \
  --out results/campaigns/jena_publication/progress.json
```

Add `--require-complete` only for the final publication gate.
