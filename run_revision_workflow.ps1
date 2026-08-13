$ErrorActionPreference = "Stop"

$required = @(
  "data/D-A.csv",
  "data/gao_fd_fp.npy",
  "data/gao_fa_fp.npy",
  "data/gao_fp_Y.npy"
)
foreach ($f in $required) {
  if (-not (Test-Path $f)) { throw "Missing required input file: $f" }
}

Write-Host "[1/10] Baseline + scaffold-disjoint audit"
python Baseline.py --da-csv data/D-A.csv --output-dir baseline --aggregation-rule maximum --scaffold-test-size 20 --scaffold-seed 42 --fixed-split-seed 12 --bootstrap-repeats 2000

Write-Host "[2/10] SHAP interpretation"
python 04_shap_interpretation.py --pipeline baseline/interpretation_pipeline.joblib --output-dir interpretation --top-k 10 --background-size 200 --background-seed 2026 --explain-scope test

Write-Host "[3/10] Aggregation sensitivity"
python 06_aggregation_sensitivity.py --raw-da-csv data/D-A.csv --excluded-pairs-csv baseline/excluded_from_model_development.csv --output-dir aggregation_sensitivity --split-seed 12 --model-seed 12

Write-Host "[4/10] Radius-2 preparation and audits"
python 00_prepare_datasets.py --inhouse-csv data/D-A.csv --excluded-pairs-csv baseline/excluded_from_model_development.csv --gao-fd data/gao_fd_fp.npy --gao-fa data/gao_fa_fp.npy --gao-y data/gao_fp_Y.npy --output-dir prepared_data --max-len 200

Write-Host "[5/10] Fixed role-aware campaign"
python 01_run_101_seeds.py --config config_fixed_structure_only_roleaware.json --max-workers 1

Write-Host "[6/10] Fixed legacy campaign"
python 01_run_101_seeds.py --config config_fixed_structure_only_legacy.json --max-workers 1

Write-Host "[7/10] Nested role-aware campaign"
python 01_run_101_seeds.py --config config_nested_roleaware.json --max-workers 1

Write-Host "[8/10] Summarize role-aware results"
python 02_summarize_results.py --fixed-config config_fixed_structure_only_roleaware.json --nested-config config_nested_roleaware.json --output-dir results_roleaware

Write-Host "[9/10] Summarize legacy results"
python 02_summarize_results.py --fixed-config config_fixed_structure_only_legacy.json --output-dir results_legacy

Write-Host "[10/10] Generate figures"
python 03_make_submission_figures.py --summary-dir results_roleaware --output-dir figures_roleaware

Write-Host "Workflow completed successfully."
