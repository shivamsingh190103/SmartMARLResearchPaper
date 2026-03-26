# Model and Hardware Artifact Policy

Use this directory for non-source artifacts required to verify paper claims:

- Pretrained checkpoints (`.pt`)
- Inference logs (latency traces, profiler exports)
- Device telemetry logs (power, temperature, utilization)

Recommended structure:

```text
artifacts/
  checkpoints/
  latency_logs/
  power_logs/
```

If you report hardware numbers in the paper, include the raw logs here (or provide stable external links) so claims are independently auditable.
