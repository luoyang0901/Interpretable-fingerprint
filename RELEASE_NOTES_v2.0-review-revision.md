# v2.0-review-revision

Major-revision archival package corresponding to the leakage-controlled manuscript revision.

## Main changes

- Replaces the former random 20-pair test with a strict donor- and acceptor-scaffold-disjoint 20-pair hold-out plus 101-pair scaffold quarantine.
- Uses target-blind, group-aware fixed partitions rather than target-informed principal splitting.
- Adds systematic exact/near cross-source overlap and representation-collision audits.
- Adds role-aware FPtand encoding and a matched legacy control; verifies zero sequence truncation at length 200.
- Separates partition variability from model-initialization variability with 10 split seeds × 5 model seeds.
- Rebuilds SHAP interpretation from one pre-specified structure-resolved Random Forest with one consistent top-10 feature rule.
- Adds calibration, residual, PCE-stratified error, ranking, enrichment, and high-PCE recall diagnostics.
- Adds exact fixed and nested train/validation/test assignment archives in CSV and NPZ formats, plus a split manifest and checksums.
- Adds corrected graphical abstract/TOC artwork emphasizing internal prediction and limited scaffold extrapolation.

## Primary revised counts

- 1,076 structure-available records
- 836 unique in-house D-A pairs
- 20 strict scaffold-disjoint hold-out pairs
- 101 scaffold-overlap quarantine pairs
- 715 in-house development pairs
- 535 Gao fingerprint-only records
- 1,250 joint radius-2 samples

## Recommended release tag

`v2.0-review-revision`

Create the GitHub release from the synchronized repository state used for submission and, where possible, archive the same release in Zenodo/Figshare for a permanent DOI.
