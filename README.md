---
title: OSC PCE Fingerprint Workflow
layout: default
---

# Code and Data Access

This repository contains the updated reproducible workflow corresponding to the revised manuscript:

**Interpretable Fingerprint-Based Prediction of Power Conversion Efficiency for Donor--Acceptor Pairs in Organic Solar Cells with External Fingerprint-Only Data Integration**

The workflow was updated after revision to keep the manuscript, Supporting Information, figures, tables, and code under one documented protocol. In particular, the updated version separates:

1. the **structure-resolved conventional baseline**, which supports interpretation and the diagnostic unknown-pair hold-out;
2. the **fixed-split TextCNN(FPtand) 101-seed ensemble**, which provides common-test-set parity plots and best-single/ensemble comparisons; and
3. the **repeated-split robustness analyses**, which report mean ± SD and 95% confidence intervals.

## Repository status after revision

The public code has been updated to match the revised manuscript. The previous A--E workflow has been replaced by the following final reproducible workflow:

- `Baseline.py` now implements the finalized conventional machine-learning baseline:
  - 836 pair-level in-house D--A records as the parent structure-resolved dataset;
  - fixed 20-pair diagnostic unknown-pair hold-out selected with seed 42;
  - model development on the remaining 816 pairs;
  - radius = 3 Morgan fingerprints;
  - fixed 70:10:20 split with split seed 12;
  - deterministic mutual-information feature selection;
  - no-augmentation Random Forest for the unknown-pair diagnostic hold-out, SHAP-oriented baseline, and Block A high-PCE cases;
  - 101 repeated random splits for conventional-baseline stability statistics.

- `Combine.py` now implements the finalized TextCNN(FPtand) training runner:
  - Gao-only, in-house radius-2, and joint in-house + Gao branches;
  - fixed HSPXY split for best-single and 101-model ensemble parity plots;
  - model seeds 0--100;
  - best single model selected by validation RMSE;
  - 101-model ensemble formed only on the common fixed test set;
  - repeated random 70:10:20 split campaign for partition-robustness statistics.

The revised GitHub workflow should therefore be used instead of the earlier `Gao.py` / A--E command sequence.

## Source datasets

Place the input files in the `data/` directory:

| File | Role |
|---|---|
| `D-A.csv` | structure-resolved in-house donor--acceptor dataset |
| `Donor.csv` | optional donor metadata table, if used by interpretation scripts |
| `Acceptor.csv` | optional acceptor metadata table, if used by interpretation scripts |
| `gao_fd_fp.npy` | Gao donor-side fingerprint array |
| `gao_fa_fp.npy` | Gao acceptor-side fingerprint array |
| `gao_fp_Y.npy` | Gao target PCE array |

The Gao source reports 566 D--A pairs. In the revised workflow, 535 retained fingerprint records are used after preprocessing of the available arrays.

## Main code modules

| Script | Purpose |
|---|---|
| `Baseline.py` | conventional ML baseline, fixed unknown-pair hold-out, repeated-split statistics, high-PCE Block A outputs |
| `Combine.py` | single TextCNN(FPtand) training runner for fixed or repeated split settings |
| `00_prepare_datasets.py` | prepares in-house radius-2 arrays and joint in-house + Gao arrays |
| `01_run_101_seeds.py` | batch driver for three TextCNN branches and seeds 0--100 |
| `02_summarize_results.py` | summarizes fixed-split and repeated-split TextCNN results into publication tables |
| `03_make_submission_figures.py` | generates manuscript-style parity plots in PNG/TIFF/PDF/EPS |
| `04_configure_runtime.py` | configures CPU/GPU device and runtime settings |
| `Explanation.py` | SHAP and interpretation-oriented analyses for the structure-resolved branch, if used |

## Final workflow overview

| Stage | Branch | Script(s) | Main outputs |
|---|---|---|---|
| 1 | Conventional baseline | `Baseline.py` | Figures S2--S4; Tables S7--S11; Block A high-PCE table |
| 2 | Dataset preparation for TextCNN | `00_prepare_datasets.py` | in-house radius-2 arrays; joint in-house + Gao arrays |
| 3 | TextCNN fixed split | `01_run_101_seeds.py` + `Combine.py` | best-single and 101-model ensemble predictions; Figures 4--6 / S5--S7 |
| 4 | TextCNN repeated split | `01_run_101_seeds.py` + `Combine.py` | mean ± SD and 95% CI for partition robustness |
| 5 | Publication outputs | `02_summarize_results.py`, `03_make_submission_figures.py` | final tables, high-PCE cases, parity plots, source data |

## Environment

A typical environment contains:

```bash
python >= 3.10
numpy
pandas
scipy
scikit-learn
matplotlib
rdkit
pytorch
```

Install the core packages, for example:

```bash
conda create -n osc_pce python=3.11 -y
conda activate osc_pce
conda install -c conda-forge rdkit numpy pandas scipy scikit-learn matplotlib -y
pip install torch torchvision torchaudio
```

For GPU runs, install a CUDA-enabled PyTorch build appropriate for the local NVIDIA driver.

## Reproducible run order

### 1. Run the finalized conventional baseline

```bash
python Baseline.py \
  --da-csv data/D-A.csv \
  --output-dir results_baseline_final \
  --run-repeated-split
```

This command generates the fixed 20-pair unknown-pair diagnostic hold-out, the 816-pair development set, the fixed-split conventional baseline tables, and the 101 repeated-split statistics.

Key outputs include:

```text
results_baseline_final/fixed_unknown_20_pairs_seed42.csv
results_baseline_final/development_816_pairs_after_unknown_exclusion.csv
results_baseline_final/Table_S7_repeated_split_statistics_wide.csv
results_baseline_final/Table_S8_fixed_split_all_model_metrics.csv
results_baseline_final/Table_S9_selected_baseline_models.csv
results_baseline_final/Table_S11_unknown_20_predictions_no_aug_random_forest.csv
results_baseline_final/Table_S18_BlockA_high_PCE_no_aug_random_forest.csv
results_baseline_final/baseline_run_summary.json
```

The unknown-pair diagnostic result reported in the revised manuscript corresponds to the **no-augmentation Random Forest** trained under this unified seed-12 protocol.

### 2. Prepare TextCNN datasets

```bash
python 00_prepare_datasets.py \
  --inhouse-csv data/D-A.csv \
  --unknown-pairs-csv results_baseline_final/fixed_unknown_20_pairs_seed42.csv \
  --gao-fd data/gao_fd_fp.npy \
  --gao-fa data/gao_fa_fp.npy \
  --gao-y data/gao_fp_Y.npy \
  --output-dir prepared_data
```

Expected prepared arrays include:

```text
prepared_data/inhouse_r2_fd.npy
prepared_data/inhouse_r2_fa.npy
prepared_data/inhouse_r2_y.npy
prepared_data/joint_fd.npy
prepared_data/joint_fa.npy
prepared_data/joint_y.npy
```

The joint branch contains 816 in-house radius-2 samples plus 535 retained Gao samples, giving 1351 samples.

### 3. Configure runtime for TextCNN

For GPU:

```bash
python 04_configure_runtime.py --configs config_fixed_split.json --max-workers 1
python 04_configure_runtime.py --configs config_repeated_split.json --max-workers 1
```

For CPU-only fallback:

```bash
python 04_configure_runtime.py --configs config_fixed_split.json --force-cpu --max-workers 1
python 04_configure_runtime.py --configs config_repeated_split.json --force-cpu --max-workers 1
```

### 4. Run fixed-split TextCNN 101-seed campaign

```bash
python 01_run_101_seeds.py --config config_fixed_split.json --max-workers 1
```

This runs 303 jobs:

```text
3 branches × 101 model seeds = 303 jobs
```

The fixed-split campaign uses a common HSPXY test set and is used for:

- validation-selected best-single models;
- 101-model ensemble predictions;
- common-test-set parity plots;
- cross-branch fixed-split comparison;
- high-PCE Block B case analysis.

### 5. Run repeated-split TextCNN 101-seed campaign

```bash
python 01_run_101_seeds.py --config config_repeated_split.json --max-workers 1
```

This also runs 303 jobs, but each seed uses a different random 70:10:20 split. These results are used only for metric-level robustness statistics. Predictions from repeated-split runs are not averaged because the test samples differ across runs.

### 6. Summarize TextCNN results and generate figures

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

Main outputs include:

```text
publication_outputs/Table_TextCNN_101seed_statistics_wide.csv
publication_outputs/Table_TextCNN_cross_branch.csv
publication_outputs/gao_only/high_PCE_cases.csv
publication_outputs/inhouse_r2/high_PCE_cases.csv
publication_outputs/joint_inhouse_gao/high_PCE_cases.csv
publication_figures/gao_only_best_single_and_ensemble.pdf
publication_figures/inhouse_r2_best_single_and_ensemble.pdf
publication_figures/joint_inhouse_gao_best_single_and_ensemble.pdf
publication_figures/Figure_TextCNN_three_branches_3x2.pdf
```

## Direct `Combine.py` example

The batch driver calls `Combine.py` internally. A direct single-branch fixed-split example is:

```bash
python Combine.py \
  --fd-path prepared_data/joint_fd.npy \
  --fa-path prepared_data/joint_fa.npy \
  --y-path prepared_data/joint_y.npy \
  --output-dir results_example_joint_fixed \
  --profile strong \
  --split-method hspxy \
  --split-seed 12 \
  --model-seeds 0,1,2 \
  --test-size 0.2 \
  --valid-fraction-of-trainval 0.125 \
  --batch-size 32 \
  --epochs 300 \
  --patience 40 \
  --lr 0.001 \
  --weight-decay 0.0001 \
  --grad-clip 5.0 \
  --max-len 200 \
  --embedding-dim 128 \
  --channels 128 \
  --dropout 0.35 \
  --kernel-sizes 3,5,7 \
  --hidden-dim 256 \
  --loss huber \
  --device cuda:0
```

Use this direct command for testing only. The manuscript results come from the full 101-seed campaigns.

## Important interpretation notes

- The fixed HSPXY split is used as a common benchmark for best-single selection, 101-model ensemble construction, and parity plots.
- The repeated-split campaigns quantify partition sensitivity and are reported as mean ± SD and 95% confidence intervals.
- Repeated-split predictions are not averaged across runs.
- The unknown-pair hold-out contains only 20 fixed pairs and is interpreted as a diagnostic extrapolation probe, not as broad external validation.
- Gao fingerprint-only data are used for representation-aligned merged modeling and are not treated as independent external validation.
- Structure-level interpretation is restricted to the in-house records with recoverable donor and acceptor structures.

## GitHub update note

The repository has been updated to contain the final `Baseline.py`, final `Combine.py`, updated run commands, fixed/repeated split protocol descriptions, and documentation matching the revised manuscript and Supporting Information.
