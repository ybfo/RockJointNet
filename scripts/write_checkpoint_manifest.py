from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "tables"
CHECKPOINTS = ROOT / "checkpoints"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "artifact": "GeoSPIN GPU smoke checkpoint",
            "path": str((CHECKPOINTS / "geospin_gpu_smoke_model.pt").resolve()),
            "status": "copied into this package" if (CHECKPOINTS / "geospin_gpu_smoke_model.pt").exists() else "missing",
            "used_by_default_plot_run": "no",
            "purpose": "PyTorch/device smoke checkpoint for the stress-path implementation.",
        },
        {
            "artifact": "GeoSPIN curve surrogate checkpoint",
            "path": str((ROOT / "outputs" / "checkpoints" / "curve_surrogate_latest.pt").resolve()),
            "status": "created only when the optional curve retraining source generator is run",
            "used_by_default_plot_run": "no",
            "purpose": "Optional local full-curve surrogate state dict; default figures use saved predictions.",
        },
        {
            "artifact": "Archived original KAN-PINN checkpoint",
            "path": str((CHECKPOINTS / "archived_original_baselines" / "KAN-PINN_best.pth").resolve()),
            "status": "copied for traceability" if (CHECKPOINTS / "archived_original_baselines" / "KAN-PINN_best.pth").exists() else "missing",
            "used_by_default_plot_run": "no",
            "purpose": "Original baseline checkpoint retained only for audit/traceability.",
        },
        {
            "artifact": "Archived original NeuralODE-PINN checkpoint",
            "path": str((CHECKPOINTS / "archived_original_baselines" / "NeuralODE-PINN_best.pth").resolve()),
            "status": "copied for traceability" if (CHECKPOINTS / "archived_original_baselines" / "NeuralODE-PINN_best.pth").exists() else "missing",
            "used_by_default_plot_run": "no",
            "purpose": "Original baseline checkpoint retained only for audit/traceability.",
        },
        {
            "artifact": "Archived monotone KAN checkpoint",
            "path": str((CHECKPOINTS / "archived_monotone_baselines" / "Mono-KAN-PINN_best.pth").resolve()),
            "status": "copied for traceability" if (CHECKPOINTS / "archived_monotone_baselines" / "Mono-KAN-PINN_best.pth").exists() else "missing",
            "used_by_default_plot_run": "no",
            "purpose": "Intermediate audit checkpoint; not part of the final manuscript model claim.",
        },
        {
            "artifact": "Saved prediction cache",
            "path": str((ROOT / "data" / "experiment_results" / "prediction_exports.csv").resolve()),
            "status": "available",
            "used_by_default_plot_run": "yes",
            "purpose": "Deterministic prediction cache used to regenerate performance-panel figures.",
        },
        {
            "artifact": "Saved full-curve prediction cache",
            "path": str((ROOT / "data" / "local_curve" / "ours_vs_bb_full_curve_predictions.csv").resolve()),
            "status": "available",
            "used_by_default_plot_run": "yes",
            "purpose": "Deterministic local direct-shear full-curve cache used to regenerate the 2x3 curve figure.",
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(CHECKPOINTS / "checkpoint_manifest.csv", index=False)
    df.to_csv(OUT / "checkpoint_manifest.csv", index=False)
    print(f"Saved checkpoint manifest to {CHECKPOINTS / 'checkpoint_manifest.csv'}")


if __name__ == "__main__":
    main()
