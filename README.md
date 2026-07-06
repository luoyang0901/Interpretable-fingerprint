# OSC PCE fingerprint workflow

This repository contains the code used for the revised manuscript:

**Interpretable Fingerprint-Based Prediction of Power Conversion Efficiency for Donor--Acceptor Pairs in Organic Solar Cells with External Fingerprint-Only Data Integration**

The current version uses the final revision protocol. The older A--E workflow and the separate `Gao.py` route have been replaced by `Baseline.py` and `Combine.py`.

## Data

Place the input files in `data/`:

```text
data/
  D-A.csv
  gao_fd_fp.npy
  gao_fa_fp.npy
  gao_fp_Y.npy
```

`D-A.csv` is the structure-resolved in-house dataset. The Gao arrays are the external fingerprint-only data. The Gao source reports 566 pairs; 535 retained fingerprint records are used after preprocessing.

## Code

| File | Role |
|---|---|
| `Baseline.py` | conventional baseline, fixed 20-pair hold-out, repeated-split metrics |
| `Combine.py` | one TextCNN(FPtand) training run |
| `00_prepare_datasets.py` | prepares in-house radius-2 and joint arrays |
| `01_run_101_seeds.py` | runs the three TextCNN branches over seeds 0--100 |
| `02_summarize_results.py` | combines fixed-split and repeated-split TextCNN outputs |
| `03_make_submission_figures.py` | optional manuscript figure generation from saved source data |

`Baseline.py` and `Combine.py` are data-first scripts. By default they save CSV/JSON/NPZ source data. They do not need to generate final manuscript figures or formatted tables directly. Use `--save-plots` only for quick local checks.

## Run

### 1. Conventional baseline

```bash
python Baseline.py \
  --da-csv data/D-A.csv \
  --output-dir baseline \
  --run-repeated-split
```

Protocol:

```text
836 parent in-house pairs
20 fixed unknown-pair samples selected with seed 42
816 pairs used for conventional model development
radius-3 Morgan fingerprints
70:10:20 split with seed 12
no-augmentation Random Forest for the unknown-pair diagnostic test
101 random repeated splits for conventional-baseline stability
```

Main output files:

```text
baseline/parent_pairs.csv
baseline/unknown_pairs.csv
baseline/development_pairs.csv
baseline/split_indices.npz
baseline/baseline_metrics.csv
baseline/selected_models.csv
baseline/baseline_parity.csv
baseline/unknown_predictions.csv
baseline/high_pce_block_a.csv
baseline/repeated_metrics.csv
baseline/repeated_summary.csv
baseline/summary.json
```

### 2. Prepare TextCNN data

```bash
python 00_prepare_datasets.py \
  --inhouse-csv data/D-A.csv \
  --unknown-pairs-csv baseline/unknown_pairs.csv \
  --gao-fd data/gao_fd_fp.npy \
  --gao-fa data/gao_fa_fp.npy \
  --gao-y data/gao_fp_Y.npy \
  --output-dir prepared_data
```

The joint branch contains 816 in-house radius-2 samples plus 535 Gao samples, giving 1351 samples.

### 3. Run TextCNN fixed split

```bash
python 01_run_101_seeds.py --config config_fixed_split.json --max-workers 1
```

This fixed HSPXY campaign is used for best-single models, 101-model ensemble predictions, and common-test-set parity data.

### 4. Run TextCNN repeated split

```bash
python 01_run_101_seeds.py --config config_repeated_split.json --max-workers 1
```

This campaign is used for repeated-split mean, SD, and 95% confidence intervals. Predictions from different repeated splits are not averaged because their test samples differ.

### 5. Summarize source data

```bash
python 02_summarize_results.py \
  --fixed-config config_fixed_split.json \
  --repeated-config config_repeated_split.json \
  --output-dir results
```

Optional figure generation:

```bash
python 03_make_submission_figures.py \
  --summary-dir results \
  --output-dir figures
```

## TextCNN run outputs

Each TextCNN branch/seed folder stores simple source files such as:

```text
split_indices.npz
seed_metrics.csv
best_single_predictions.csv
ensemble_predictions.csv
high_pce_cases.csv
summary.json
```

Per-model subfolders may also contain:

```text
train_history.csv
predictions.csv
```

These files are intended as source data for plotting, checking, and manuscript table assembly.

## Notes

- The 20-pair unknown-pair set is a diagnostic hold-out, not a broad external validation set.
- The fixed HSPXY split is used for common-test-set ensemble comparison.
- Repeated random splits are used for partition-sensitivity statistics.
- Gao data are fingerprint-only in this workflow and are used for representation-aligned prediction, not structure-level interpretation.
# Run commands

## Baseline

```bash
python Baseline.py --da-csv data/D-A.csv --output-dir baseline --run-repeated-split
```

## Prepare TextCNN arrays

```bash
python 00_prepare_datasets.py \
  --inhouse-csv data/D-A.csv \
  --unknown-pairs-csv baseline/unknown_pairs.csv \
  --gao-fd data/gao_fd_fp.npy \
  --gao-fa data/gao_fa_fp.npy \
  --gao-y data/gao_fp_Y.npy \
  --output-dir prepared_data
```

## Fixed-split TextCNN

```bash
python 01_run_101_seeds.py --config config_fixed_split.json --max-workers 1
```

## Repeated-split TextCNN

```bash
python 01_run_101_seeds.py --config config_repeated_split.json --max-workers 1
```

## Summarize

```bash
python 02_summarize_results.py \
  --fixed-config config_fixed_split.json \
  --repeated-config config_repeated_split.json \
  --output-dir results
```
Optional:

```bash
python 03_make_submission_figures.py --summary-dir results --output-dir figures
```
