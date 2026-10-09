# CPU smoke evidence

Three full smoke runs produced exactly identical measurement rows and dataset manifests. The attached manifest records the published source commit and a clean working tree for the last run. Twenty regression checks passed locally.

- 17 trained/analytic model cases plus 13 untrained controls
- Six held-out test regimes, two probe tasks and 360 sequence-task rows
- Two training epochs, one data seed and one initialization seed
- All numeric measurements finite; every recorded quantum gradient norm positive
- Exact measurement reproduction on the same tested CPU environment

These files establish execution and reproducibility, not quantum advantage or architecture rankings. Timing can vary. The full datasets, fitted models and probes can be regenerated with the attached configuration; large checkpoint binaries are omitted here.

```bash
python -m pilot.run --config configs/cpu_smoke.json --output pilot_runs/reproduce
```

Compare `sequence_metrics.csv` and `data_17/dataset_manifest.json` to the checked-in evidence. Bitwise agreement across different dependency builds or hardware is not promised. Source hashes, package versions and seeds are in `manifest.json`. `verification.json` records the repeated-run checks.
