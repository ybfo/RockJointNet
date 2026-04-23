# RockJointNet Paper Figure/Table Reproduction Package

This folder is the clean reproduction package for the final paper figures and tables. It is intentionally separate from the original exploratory folders so the manuscript artifacts can be regenerated without touching raw data or old experiment outputs.

## Quick run

From `E:\xiongyuhu\stress_prediction\RockJointNet_paper_figure_table_code`:

```powershell
python .\run_all.py
```

The default run regenerates the final paper figures and tables from saved experiment CSVs. It does not retrain neural networks, so it is deterministic and fast.

## Outputs

- `outputs/figures/lit_fig03_solution_and_absolute_error_field.png`
- `outputs/figures/lit_fig04_multifeature_prediction_contours.png`
- `outputs/figures/lit_fig05_slope_and_curvature_fields.png`
- `outputs/figures/lit_fig06_derived_stress_path_quantities.png`
- `outputs/figures/lit_fig09_prediction_performance_panels.png`
- `outputs/figures/ours_vs_bb_full_shear_dilation.png`
- `outputs/tables/*.tex`, `outputs/tables/*.csv`, and `outputs/tables/*.md`
- `checkpoints/checkpoint_manifest.csv`
- `outputs/tnnls_title_page.docx`, `outputs/tnnls_title_page.pdf`, and `outputs/tnnls_title_page.tex`

Each figure is also saved as PDF.

`outputs/tnnls_title_page.docx` is the editable Word version of the editorial-only IEEE TNNLS title page, and `outputs/tnnls_title_page.pdf` is the compiled PDF check. The title page contains the manuscript title, author names and affiliations, correspondence address and email, and acknowledgments. Do not merge this title page into the anonymous manuscript file sent to peer reviewers.

The Word file can be regenerated without external Python packages by running:

```powershell
python .\scripts\make_tnnls_title_page_docx.py
```

## Input data used by the default run

- `data/experiment_results/feature_sensitivity_sweeps.csv`
- `data/experiment_results/prediction_exports.csv`
- `data/experiment_results/physics_validity_fixed_context_sweeps.csv`
- `data/experiment_results/nondimensional_transform_ablation.csv`
- `data/experiment_results/noise_robustness.csv`
- `data/experiment_results/synthetic_inverse_recovery.csv`
- `data/recent_methods/recent_methods_summary.csv`
- `data/local_curve/ours_vs_bb_full_curve_predictions.csv`
- `data/local_curve/ours_vs_bb_full_curve_metrics.csv`

## Checkpoints

The default figure/table scripts use saved prediction caches instead of reloading a model checkpoint. The available checkpoint locations are recorded in:

```text
checkpoints/checkpoint_manifest.csv
outputs/tables/checkpoint_manifest.csv
```

Important entries:

- `checkpoints/geospin_gpu_smoke_model.pt`: copied stress-path PyTorch smoke checkpoint.
- `checkpoints/archived_original_baselines/*.pth`: archived original KAN/ODE/Transformer/GIIN baselines for traceability.
- `checkpoints/archived_monotone_baselines/*.pth`: archived monotone audit baselines for traceability.
- `outputs/checkpoints/curve_surrogate_latest.pt`: reserved output path if the optional historical curve retraining script is modified/run to save a state dict.

## Historical source generators

The folder `scripts/source_generators/` contains the original exploratory generators copied from the working directories. They are kept for auditability. The default clean run uses `scripts/make_paper_figures.py` and `scripts/make_paper_tables.py`, which are the recommended reproducible artifact scripts.
