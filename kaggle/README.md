# Kaggle Parallel Execution Pack

This folder lets you move SmartMARL from one-machine training to parallel Kaggle runs.

## Files

- `prepare_bundle.sh` : builds `kaggle/output/smartmarl_kaggle.zip`
- `dataset-metadata.json` : Kaggle dataset metadata
- `upload_dataset.sh` : creates/updates Kaggle dataset via API
- `launch_kernels.sh` : pushes all prepared training kernels
- `monitor_kernels.sh` : polls kernel status, downloads outputs, merges JSONs, runs aggregation
- `notebooks/notebook1_standard_full_1_10.py`
- `notebooks/notebook2_standard_full_11_20.py`
- `notebooks/notebook3_standard_l7_1_29.py`
- `notebooks/notebook4_standard_full_21_29_optional.py`
- `kernels/` : Kaggle kernel folders + metadata for API push

## Local Preparation

```bash
cd /Users/shivamsingh/Desktop/ResearchPaper
./kaggle/prepare_bundle.sh
```

Output zip:

- `/Users/shivamsingh/Desktop/ResearchPaper/kaggle/output/smartmarl_kaggle.zip`

## Upload to Kaggle (optional API path)

```bash
pip install kaggle
mkdir -p ~/.kaggle
# place kaggle.json in ~/.kaggle and chmod 600
./kaggle/upload_dataset.sh
```

## Notebook Launch Plan

Run these in parallel on Kaggle:

1. Notebook 1: standard full seeds `1..10`
2. Notebook 2: standard full seeds `11..20`
3. Notebook 3: standard L7 seeds `1..29`
4. Optional Notebook 4: standard full seeds `21..29`

Each notebook script installs SUMO, verifies `Mock mode: False`, then runs assigned seeds.

## Fully Automated API Path

With `KAGGLE_API_TOKEN` configured:

```bash
cd /Users/shivamsingh/Desktop/ResearchPaper
./kaggle/upload_dataset.sh
./kaggle/launch_kernels.sh
./kaggle/monitor_kernels.sh
```

`monitor_kernels.sh` continuously:

1. polls kernel status
2. downloads outputs from completed kernels
3. merges `*seed*.json` into local `results/raw/`
4. runs `collect_results.py --scenario standard`

## Merging Results Back

Download each notebook output folder and merge JSON files into local:

- `/Users/shivamsingh/Desktop/ResearchPaper/results/raw/`

Then aggregate:

```bash
cd /Users/shivamsingh/Desktop/ResearchPaper
python collect_results.py --scenario standard
python collect_results.py --scenario indian_hetero \
  --out_csv results/ablation_table_indian.csv \
  --out_txt results/ablation_table_indian.txt
```
