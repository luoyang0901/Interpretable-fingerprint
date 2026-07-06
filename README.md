# Final Run Commands for the Revised OSC PCE Fingerprint Workflow

This file replaces the earlier A--E command list. The revised manuscript uses a finalized conventional-baseline protocol and two TextCNN(FPtand) 101-seed campaigns.

## 0. Suggested directory layout

```text
project_root/
  data/
    D-A.csv
    gao_fd_fp.npy
    gao_fa_fp.npy
    gao_fp_Y.npy
  Baseline.py
  Combine.py
  00_prepare_datasets.py
  01_run_101_seeds.py
  02_summarize_results.py
  03_make_submission_figures.py
  04_configure_runtime.py
  config_fixed_split.json
  config_repeated_split.json
```

## 1. Conventional baseline and unknown-pair diagnostic hold-out

```bash
python Baseline.py \
  --da-csv data/D-A.csv \
  --output-dir results_baseline_final \
  --run-repeated-split
```

Final protocol:

```text
836 parent structure-resolved D--A pairs
seed 42 fixed 20-pair diagnostic unknown-pair hold-out
816 pairs for conventional model development
radius = 3 Morgan fingerprints
70:10:20 fixed split with split seed 12
deterministic mutual-information feature selection
no-augmentation Random Forest for unknown-pair, SHAP-oriented baseline, and Block A
101 repeated random splits for conventional-baseline statistics
```

## 2. Prepare in-house radius-2 and joint arrays for TextCNN

```bash
python 00_prepare_datasets.py \
  --inhouse-csv data/D-A.csv \
  --unknown-pairs-csv results_baseline_final/fixed_unknown_20_pairs_seed42.csv \
  --gao-fd data/gao_fd_fp.npy \
  --gao-fa data/gao_fa_fp.npy \
  --gao-y data/gao_fp_Y.npy \
  --output-dir prepared_data
```

## 3. Configure runtime

GPU:

```bash
python 04_configure_runtime.py --configs config_fixed_split.json --max-workers 1
python 04_configure_runtime.py --configs config_repeated_split.json --max-workers 1
```

CPU-only:

```bash
python 04_configure_runtime.py --configs config_fixed_split.json --force-cpu --max-workers 1
python 04_configure_runtime.py --configs config_repeated_split.json --force-cpu --max-workers 1
```

## 4. Fixed-split TextCNN 101-seed campaign

```bash
python 01_run_101_seeds.py --config config_fixed_split.json --max-workers 1
```

Expected number of completed jobs:

```text
3 branches × 101 model seeds = 303
```

Check progress:

```bash
find results_fixed_split_101 -name run_summary.json | wc -l
```

Windows PowerShell:

```powershell
(Get-ChildItem '.\results_fixed_split_101' -Recurse -Filter run_summary.json).Count
```

## 5. Repeated-split TextCNN 101-seed campaign

```bash
python 01_run_101_seeds.py --config config_repeated_split.json --max-workers 1
```

Expected number of completed jobs:

```text
3 branches × 101 paired split/model seeds = 303
```

Check progress:

```bash
find results_repeated_split_101 -name run_summary.json | wc -l
```

Windows PowerShell:

```powershell
(Get-ChildItem '.\results_repeated_split_101' -Recurse -Filter run_summary.json).Count
```

## 6. Summarize and generate publication figures

```bash
python 02_summarize_results.py \
  --fixed-config config_fixed_split.json \
  --repeated-config config_repeated_split.json \
  --output-dir publication_outputs
```

```bash
python 03_make_submission_figures.py \
  --summary-dir publication_outputs \
  --output-dir publication_figures
```

## 7. Package final outputs

Linux/macOS:

```bash
zip -r OSC_PCE_final_outputs.zip \
  results_baseline_final \
  publication_outputs \
  publication_figures \
  config_fixed_split.json \
  config_repeated_split.json
```

Windows PowerShell:

```powershell
Compress-Archive -Path '.\results_baseline_final','.\publication_outputs','.\publication_figures','.\config_fixed_split.json','.\config_repeated_split.json' -DestinationPath '.\OSC_PCE_final_outputs.zip' -Force
```
