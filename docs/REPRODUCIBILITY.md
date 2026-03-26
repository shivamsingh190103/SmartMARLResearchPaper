# Reproducibility and Evidence Checklist

This repository currently supports **simulation-first** reproducibility.
It does not ship camera/radar raw recordings, YOLO pretrained checkpoints, or embedded-device telemetry logs.

## What Is Reproducible Today

1. Training/evaluation runs in SUMO/TraCI (or mock for smoke testing).
2. Seed-level metrics (`ATT`, `AWT`, `Throughput`) in JSON/CSV form.
3. Ablation table generation from seed artifacts.
4. Statistical audit outputs for paper-facing claims.
5. Dashboard figure regeneration from artifacts.

## One-Command Artifact Regeneration

```bash
python scripts/reproduce_all.py --raw_dir results/raw --out_dir results/repro
```

Expected files:

- `results/repro/seed_metrics.csv`
- `results/repro/method_summary.csv`
- `results/repro/ablation_summary.csv`
- `results/repro/paper_claims_audit.json`
- `results/repro/paper_claims_audit.md`
- `results/repro/dashboard.png`

## Bundle for Reviewers

```bash
python scripts/export_repro_bundle.py --raw_dir results/raw --out_dir results/repro --bundle results/repro_bundle.zip
```

This creates a zip that can be attached to submissions or shared with reviewers.

## Scope and Limitations

- If only mock backend artifacts exist, results are development-only and should not be framed as real-world validation.
- Paper claims are auditable only when sufficient overlapping seeds are available for each compared method.
- Hardware claims (for example Jetson latency/power) require raw deployment logs and are not automatically reproducible from this repo unless those logs are added.
