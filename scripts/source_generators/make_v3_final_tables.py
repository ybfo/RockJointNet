from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "paper_artifacts_v3_final"
TABLES = OUT / "tables"


def save(df: pd.DataFrame, name: str) -> None:
    df.to_csv(TABLES / f"{name}.csv", index=False)
    df.to_latex(TABLES / f"{name}.tex", index=False, float_format="%.4f")
    (TABLES / f"{name}.md").write_text(df.to_markdown(index=False), encoding="utf-8")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    recent = pd.read_csv(ROOT / "results_recent_methods" / "recent_methods_summary.csv")
    winners = pd.read_csv(ROOT / "results_recent_methods" / "recent_methods_winners.csv")
    excluded = pd.read_csv(ROOT / "results_recent_methods" / "recent_methods_excluded_negative_r2_winners.csv")
    physics = pd.read_csv(ROOT / "paper_artifacts" / "experiment_results" / "physics_validity_fixed_context_sweeps.csv")
    transform = pd.read_csv(ROOT / "paper_artifacts" / "experiment_results" / "nondimensional_transform_ablation.csv")
    noise = pd.read_csv(ROOT / "paper_artifacts" / "experiment_results" / "noise_robustness.csv")
    inverse = pd.read_csv(ROOT / "paper_artifacts" / "experiment_results" / "synthetic_inverse_recovery.csv")

    valid_protocols = [
        ("rockmb_2025", "paper_70_30"),
        ("rockmb_2025", "group_5fold"),
        ("g5pf6k9n2w", "random_75_25"),
        ("g5pf6k9n2w", "leave_one_profile_out"),
        ("g5pf6k9n2w", "leave_one_immersion_out"),
        ("w7m28x23kw", "random_75_25"),
    ]
    main = pd.concat([recent[(recent.dataset == d) & (recent.protocol == p)] for d, p in valid_protocols], ignore_index=True)
    main = main[["dataset", "protocol", "model", "R2_mean", "R2_std", "RMSE_mean", "RMSE_std", "MAE_mean", "MAAPE_mean"]]
    save(main, "table3_main_recent_method_benchmark_valid_protocols")
    save(winners, "table4_valid_winners_r2_nonnegative")
    save(excluded, "table5_excluded_negative_r2_protocols")
    save(transform, "table6_transform_stability_ablation")
    save(physics, "table7_physics_validity_strict_path")
    noise_summary = noise[["dataset", "model", "noise_level", "R2", "RMSE", "MAE", "MAAPE", "relative_L2_percent"]]
    save(noise_summary, "table8_noise_robustness")
    inv = inverse.groupby("noise_level")[["a_relative_error_percent", "b_relative_error_percent"]].agg(["mean", "std"]).reset_index()
    inv.columns = ["_".join([str(c) for c in col if c != ""]).strip("_") for col in inv.columns.to_flat_index()]
    save(inv, "table9_synthetic_inverse_recovery")

    summary = f"""# V3 Final Artifact Status

## Completed
- Strict fixed-path monotonicity evaluation rerun.
- Recent-method benchmark rerun.
- LaTeX-ready tables generated from real result CSVs.

## Key results
- Valid winner protocols: {len(winners)}
- Proposed models win: {int(winners['model'].str.contains('ours').sum())}/{len(winners)}
- Physics validity: max monotonic violation = {physics['fixed_context_monotonic_violation_rate_percent'].max():.4f}%
- Physics validity: max tau(0) boundary error = {physics['mean_tau0_boundary_error'].max():.4e}

## Figure policy
- Do not regenerate the rejected old figures.
- New figures should follow `paper_artifacts_v2_design.md` and be created one-by-one with publication styling.
"""
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "README.md").write_text(summary, encoding="utf-8")
    print(f"Saved v3 final tables to {TABLES}")


if __name__ == "__main__":
    main()
