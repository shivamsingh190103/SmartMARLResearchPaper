# Data Directory Contract

This folder is intentionally lightweight in Git and serves as a mount point for external datasets.

## Expected Inputs (Optional but Recommended)

- Public trajectory datasets for calibration (for example NGSIM CSV exports).
- Sensor-like replay files if you want to validate perception noise settings against real traces.

## Suggested Layout

```text
data/
  ngsim/
    us101.csv
    i80.csv
  calibration/
    metadata.json
```

## Notes

- Large files are not committed to keep the repository portable.
- Generated calibration outputs should go to `results/calibration/`.
