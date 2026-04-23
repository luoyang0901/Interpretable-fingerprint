# A-E Run Commands

## Step 0 — preparation

```bash
python Baseline.py --dataset "D-A.csv" --donor-col donor_smiles --acceptor-col acceptor_smiles --pce-col PCE --auto-prepare-only
```

## Step E — stability search

```bash
python Baseline.py --main-search 0 100 --output-tag E_seed_search
```

## Step A — main baseline

```bash
python Baseline.py --run-best-main --output-tag A_main_r3
```

## Step B — radius-control branch

```bash
python Baseline.py --run-best-core-r2 --output-tag B_core_r2
```

## Step C — Gao-only branch

```bash
mkdir C_gao_only
cp Gao.py C_gao_only/
cp gao_fd_fp.npy C_gao_only/fd_fp.npy
cp gao_fa_fp.npy C_gao_only/fa_fp.npy
cp gao_fp_Y.npy C_gao_only/fp_Y.npy
cd C_gao_only
python Gao.py
```

## Step D — joint in-house + Gao branch

```bash
python Baseline.py --run-best-core-r2-plus-gao --gao-fd-path gao_fd_fp.npy --gao-fa-path gao_fa_fp.npy --gao-y-path gao_fp_Y.npy --output-tag D_core_r2_plus_gao
```
