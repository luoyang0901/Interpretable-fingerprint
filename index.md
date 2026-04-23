---
title: OSC PCE Fingerprint Workflow
layout: default
---

# Code and Data Access

This page provides the public workflow for reproducing the manuscript results.

## Manuscript title

**Interpretable Fingerprint-Based Prediction of Power Conversion Efficiency for Donor-Acceptor Pairs in Organic Solar Cells with External Fingerprint-Only Data Integration**

## Source datasets

- `D-A.csv`
- `Donor.csv`
- `Acceptor.csv`
- `gao_fd_fp.npy`
- `gao_fa_fp.npy`
- `gao_fp_Y.npy`

## Code modules

- **A / B / E:** `Baseline.py`
- **C:** `Gao.py`
- **D:** `Combine.py`
- **Interpretation:** `Explanation.py`

## Workflow overview

| Step | Branch | Script | Output role |
|---|---|---|---|
| 0 | Preparation | `Baseline.py` | Build in-house fingerprint arrays |
| E | Stability search | `Baseline.py` | Split × seed × preset search |
| A | Main baseline | `Baseline.py` | Main interpretable branch |
| B | Radius control | `Baseline.py` | Radius = 2 control branch |
| C | Gao-only | `Gao.py` | Transfer-style validation |
| D | Joint branch | `Baseline.py` / `Combine.py` | Best predictive branch |

## Recommended run order

### Step 0 — prepare the in-house fingerprint arrays

```bash
python Baseline.py --dataset "D-A.csv" --donor-col donor_smiles --acceptor-col acceptor_smiles --pce-col PCE --auto-prepare-only
```

This step generates the downstream fingerprint arrays used later in the workflow, including `fd_fp1.npy`, `fa_fp1.npy`, `fd_fp_augmented.npy`, `fa_fp_augmented.npy`, and `fp_Y_augmented.npy`.

### Step E — search the split × seed configuration

```bash
python Baseline.py --main-search 0 100 --output-tag E_seed_search
```

### Step A — main interpretable baseline

```bash
python Baseline.py --run-best-main --output-tag A_main_r3
```

### Step B — radius-control branch

```bash
python Baseline.py --run-best-core-r2 --output-tag B_core_r2
```

### Step C — Gao-only transfer branch

```bash
mkdir C_gao_only
cp Gao.py C_gao_only/
cp gao_fd_fp.npy C_gao_only/fd_fp.npy
cp gao_fa_fp.npy C_gao_only/fa_fp.npy
cp gao_fp_Y.npy C_gao_only/fp_Y.npy
cd C_gao_only
python Gao.py
```

### Step D — joint in-house + Gao branch

```bash
python Baseline.py --run-best-core-r2-plus-gao --gao-fd-path gao_fd_fp.npy --gao-fa-path gao_fa_fp.npy --gao-y-path gao_fp_Y.npy --output-tag D_core_r2_plus_gao
```

## Practical note

The refactored `Baseline.py` is presented around the original source files `D-A.csv`, `Donor.csv`, and `Acceptor.csv`, so the public workflow no longer needs to describe the older internal naming system.
