# CET publication campaign execution

The frozen CET paper matrix contains 168 deterministic-forecasting runs and 20
TrustKAN reliability runs. Generate the checksum-bound campaign with:

```bash
python scripts/plan_cet_campaign.py \
  --benchmark-config configs/cet.yaml \
  --reliability-config configs/cet_reliability.yaml \
  --outdir results/campaigns/cet_publication
```

The planner fails unless the publication configuration retains horizons
1/7/30/90, history 365, five aligned seeds, the required baseline suite, and
the frozen conformal/reliability policy. It writes `campaign.json`,
`run_cpu.ps1`, `run_gpu.ps1`, `collect.ps1`, and `run_all.ps1`.

Neural and reliability shards must pass the CUDA preflight in
`docs/GPU_EXECUTION.md`. CPU fallback results cannot fill GPU campaign cells.
Smoke configurations require `--engineering` and remain prohibited from paper
tables.

```bash
python scripts/audit_cet_campaign.py \
  --campaign results/campaigns/cet_publication/campaign.json \
  --out results/campaigns/cet_publication/progress.json
```

Add `--require-complete` only for the final publication gate.
