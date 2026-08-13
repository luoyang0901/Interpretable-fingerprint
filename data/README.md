# Input data used by the revised workflow

This directory contains the exact machine-readable inputs needed by the revised scripts.

- `D-A.csv`: 1,076 structure-available in-house literature records (donor SMILES, acceptor SMILES, PCE, and pair key) used before pair-level aggregation.
- `gao_fd_fp.npy`: 535 retained Gao donor fingerprint rows, 1,024 bits each.
- `gao_fa_fp.npy`: 535 retained Gao acceptor fingerprint rows, 1,024 bits each.
- `gao_fp_Y.npy`: PCE labels for the same 535 retained Gao rows.

The Gao arrays are fingerprint-only in this repository and are used for representation-aligned internal prediction, not independent external validation or structure-level interpretation.
