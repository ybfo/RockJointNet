# RockJointNet

RockJointNet is the working repository for the rock-joint shear-strength study built around the GeoSPIN model family. This repo keeps the paper-facing assets in one place: benchmark summaries, figure scripts, table builders, cached predictions, manuscript drafts, and the small checkpoints needed for reproducibility checks.

The current layout is meant for day-to-day research use rather than for a polished software release. The emphasis is on making the benchmark pipeline easy to rerun, the manuscript artifacts easy to trace back to source files, and the final figures/tables easy to update when experiments change.

## Quick start

From the repository root:

```powershell
python .\build_all.py
```

This regenerates the framework diagrams, paper figures, paper tables, the TNNLS title page Word file, and the checkpoint manifest from the cached experiment outputs already stored in the repo. It does not retrain the full benchmark suite.

If you only want a subset:

```powershell
python .\scripts\build_figures.py
python .\scripts\build_tables.py
python .\scripts\build_title_page_docx.py
python .\scripts\build_checkpoint_manifest.py
```

## Repository layout

```text
RockJointNet/
├── build_all.py
├── manuscript_tnnls.tex
├── requirements.txt
├── checkpoints/
├── data/
├── outputs/
├── scripts/
└── source_inputs/
```

`data/` stores the cached benchmark summaries, counterfactual sweep exports, local direct-shear curve caches, and manuscript-side table sources. `scripts/` contains the generators for figures, tables, title-page documents, manifests, and framework diagrams. `outputs/` holds the rendered assets that are ready to drop into the paper or supplementary package. `source_inputs/` keeps the original imported tables used to build the cleaned benchmark caches.

## Main generated files

The build step writes the paper-facing assets to `outputs/`.

Figures:

- `outputs/figures/figure_03_prediction_and_error_field.png`
- `outputs/figures/figure_04_feature_response_maps.png`
- `outputs/figures/figure_05_slope_and_curvature_maps.png`
- `outputs/figures/figure_06_stress_path_profiles.png`
- `outputs/figures/figure_09_prediction_panels.png`
- `outputs/figures/figure_10_curve_reconstruction.png`
- `outputs/figures/framework_main.png`

Tables:

- `outputs/tables/table_01_benchmark_results.tex`
- `outputs/tables/table_02_transform_ablation.tex`
- `outputs/tables/table_03_constraint_checks.tex`
- `outputs/tables/table_04_curve_reconstruction.tex`
- `outputs/tables/table_05_noise_robustness.tex`
- `outputs/tables/table_06_inverse_recovery.tex`

Documents:

- `outputs/title_page_tnnls.docx`
- `outputs/title_page_tnnls.pdf`
- `outputs/paper_draft_full.pdf`

## Data notes

The default scripts read from cached CSV files rather than from live training logs. The most important ones are:

- `data/experiment_results/counterfactual_feature_sweeps.csv`
- `data/experiment_results/prediction_cases.csv`
- `data/experiment_results/fixed_context_constraint_checks.csv`
- `data/experiment_results/transform_ablation_results.csv`
- `data/experiment_results/noise_robustness_results.csv`
- `data/experiment_results/inverse_recovery_results.csv`
- `data/recent_methods/benchmark_summary.csv`
- `data/local_curve/curve_reconstruction_predictions.csv`
- `data/local_curve/curve_reconstruction_metrics.csv`

That makes the figure/table build deterministic and fast, which is helpful when the manuscript is still moving.

## Checkpoints

This repo is not checkpoint-heavy, but it does keep a few model files for traceability and smoke testing. The canonical index is:

```text
checkpoints/checkpoint_manifest.csv
outputs/tables/checkpoint_manifest.csv
```

The main file used for a quick implementation sanity check is:

- `checkpoints/rockjointnet_smoke_checkpoint.pt`

The archived baseline checkpoints are kept only so the manuscript comparisons remain auditable.

## GitHub publishing

If you want to push this repository from a fresh machine, use:

```powershell
$env:GITHUB_TOKEN="your_token_here"
.\publish_github.ps1
```

The script assumes a GitHub repository named `ybfo/RockJointNet`.
