from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RECENT = DATA / "recent_methods"
EXP = DATA / "experiment_results"
LOCAL = DATA / "local_curve"
MANUSCRIPT_TABLES = DATA / "manuscript_tables"
OUT = ROOT / "outputs" / "tables"


def latex_escape(text: str) -> str:
    return (
        text.replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def save_table(df: pd.DataFrame, name: str, caption: str, label: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"{name}.csv", index=False)
    (OUT / f"{name}.md").write_text(df.to_markdown(index=False), encoding="utf-8")
    tex = df.to_latex(
        OUT / f"{name}.tex",
        index=False,
        escape=False,
        float_format=lambda x: f"{x:.4f}",
        caption=caption,
        label=label,
    )
    if tex is None:
        return


def make_main_results() -> pd.DataFrame:
    summary = pd.read_csv(RECENT / "benchmark_summary.csv")
    protocol_short = {
        ("rockmb_2025", "paper_70_30"): "R70",
        ("g5pf6k9n2w", "random_75_25"): "G5-R",
        ("g5pf6k9n2w", "leave_one_profile_out"): "G5-LOP",
    }
    wanted = [
        ("rockmb_2025", "paper_70_30", "FTTransformer_2021", "FTTransformer"),
        ("rockmb_2025", "paper_70_30", "CatBoost", "CatBoost"),
        ("rockmb_2025", "paper_70_30", "KANLite_2024", "KANLite"),
        ("rockmb_2025", "paper_70_30", "RealMLP_2021", "RealMLP"),
        ("rockmb_2025", "paper_70_30", "TabM_2025", "TabM"),
        ("rockmb_2025", "paper_70_30", "ResidualPeriodicMonotone_ours", r"\textbf{GeoSPIN}"),
        ("g5pf6k9n2w", "random_75_25", "FTTransformer_2021", "FTTransformer"),
        ("g5pf6k9n2w", "random_75_25", "CatBoost", "CatBoost"),
        ("g5pf6k9n2w", "random_75_25", "KANLite_2024", "KANLite"),
        ("g5pf6k9n2w", "random_75_25", "PeriodicMLP_2022", "Periodic MLP"),
        ("g5pf6k9n2w", "random_75_25", "RealMLP_2021", "RealMLP"),
        ("g5pf6k9n2w", "random_75_25", "ResidualPeriodicMonotone_ours", r"\textbf{GeoSPIN}"),
        ("g5pf6k9n2w", "leave_one_profile_out", "FTTransformer_2021", "FTTransformer"),
        ("g5pf6k9n2w", "leave_one_profile_out", "CatBoost", "CatBoost"),
        ("g5pf6k9n2w", "leave_one_profile_out", "KANLite_2024", "KANLite"),
        ("g5pf6k9n2w", "leave_one_profile_out", "PeriodicMLP_2022", "Periodic MLP"),
        ("g5pf6k9n2w", "leave_one_profile_out", "RealMLP_2021", "RealMLP"),
        ("g5pf6k9n2w", "leave_one_profile_out", "ResidualPeriodicMonotone_ours", r"\textbf{GeoSPIN}"),
    ]
    rows = []
    for dataset, protocol, model, display in wanted:
        part = summary[(summary["dataset"] == dataset) & (summary["protocol"] == protocol) & (summary["model"] == model)]
        if part.empty:
            continue
        row = part.iloc[0]
        rows.append(
            {
                "Prot.": protocol_short.get((dataset, protocol), f"{dataset}, {protocol}"),
                "Method": display,
                r"$R^2$": row["R2_mean"],
                "RMSE": row["RMSE_mean"],
                "MAE": row["MAE_mean"],
                "MAAPE": row["MAAPE_mean"],
            }
        )
    return pd.DataFrame(rows)


def make_transform_ablation() -> pd.DataFrame:
    df = pd.read_csv(EXP / "transform_ablation_results.csv")
    names = {
        "raw_sigma_raw_tau_plain_mlp": r"Raw $\sigma$, raw $\tau$ MLP",
        "log_sigma_raw_tau_plain_mlp": r"Log $\sigma$, raw $\tau$ MLP",
        "raw_sigma_asinh_tau_plain_mlp": r"Raw $\sigma$, asinh $\tau$ MLP",
        "log_sigma_asinh_tau_plain_mlp": r"Log $\sigma$, asinh $\tau$ MLP",
        "stress_path_periodic_residual_monotone": r"\textbf{GeoSPIN integral}",
    }
    out = df[["experiment", "R2", "RMSE", "MAE", "MAAPE", "relative_L2_percent"]].copy()
    out["experiment"] = out["experiment"].map(names).fillna(out["experiment"])
    out = out.rename(
        columns={
            "experiment": "Variant",
            "R2": r"$R^2$",
            "relative_L2_percent": "Rel. $L_2$",
        }
    )
    return out


def make_physics_checks() -> pd.DataFrame:
    physics = pd.read_csv(EXP / "fixed_context_constraint_checks.csv")
    max_mono = physics["fixed_context_monotonic_violation_rate_percent"].max()
    max_neg = physics["negative_prediction_rate_percent"].max()
    max_boundary = physics["mean_tau0_boundary_error"].max()
    rows = [
        {
            "Quantity checked": "Zero-stress boundary",
            "Mathematical statement": r"$\hat{\tau}_p(\sigma_n=0,c)=0$",
            "GeoSPIN enforcement": "hard integral lower limit",
            "Diagnostic": f"max mean error = {max_boundary:.2e}",
        },
        {
            "Quantity checked": "Fixed-context monotonicity",
            "Mathematical statement": r"$\partial \hat{\tau}_p/\partial s \ge 0$",
            "GeoSPIN enforcement": "softplus positive rate",
            "Diagnostic": f"max violation = {max_mono:.2f}%",
        },
        {
            "Quantity checked": "Nonnegative prediction",
            "Mathematical statement": r"$\hat{\tau}_p\ge0$",
            "GeoSPIN enforcement": "integral of nonnegative rate",
            "Diagnostic": f"max negative rate = {max_neg:.2f}%",
        },
        {
            "Quantity checked": "Trusted empirical prior",
            "Mathematical statement": r"$w_{BB}(\sigma_n,c)\mathcal{L}_{BB}$ only in low-stress region",
            "GeoSPIN enforcement": "trusted-region weighting",
            "Diagnostic": "not a global label constraint",
        },
    ]
    return pd.DataFrame(rows)


def make_curve_metrics() -> pd.DataFrame:
    df = pd.read_csv(LOCAL / "curve_reconstruction_metrics.csv")
    target_map = {
        "shear_stress_mpa": "Shear stress",
        "dilation_proxy_mm": "Dilation proxy",
    }
    out = df[["target", "method", "normal_stress_mpa", "rmse", "r2"]].copy()
    out["target"] = out["target"].map(target_map).fillna(out["target"])
    out["normal_stress_mpa"] = out["normal_stress_mpa"].map(lambda x: f"{x:g} MPa")
    out = out.rename(
        columns={
            "target": "Target",
            "method": "Method",
            "normal_stress_mpa": r"$\sigma_n$",
            "rmse": "RMSE",
            "r2": r"$R^2$",
        }
    )
    return out


def make_noise_table() -> pd.DataFrame:
    df = pd.read_csv(EXP / "noise_robustness_results.csv")
    out = df[["dataset", "model", "noise_level", "R2", "RMSE", "MAE", "MAAPE", "relative_L2_percent"]].copy()
    out["model"] = out["model"].replace({"ResidualPeriodicMonotone": "GeoSPIN"})
    out = out.rename(columns={"relative_L2_percent": "Relative $L_2$ (%)", "R2": r"$R^2$"})
    return out


def make_inverse_table() -> pd.DataFrame:
    df = pd.read_csv(EXP / "inverse_recovery_results.csv")
    grouped = df.groupby("noise_level")[["a_relative_error_percent", "b_relative_error_percent"]].agg(["mean", "std"])
    grouped.columns = ["_".join([str(x) for x in col if x]) for col in grouped.columns.to_flat_index()]
    grouped = grouped.reset_index()
    return grouped.rename(
        columns={
            "noise_level": "Noise level",
            "a_relative_error_percent_mean": "$a$ rel. error mean (%)",
            "a_relative_error_percent_std": "$a$ rel. error std (%)",
            "b_relative_error_percent_mean": "$b$ rel. error mean (%)",
            "b_relative_error_percent_std": "$b$ rel. error std (%)",
        }
    )


def copy_manuscript_tables() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for src in MANUSCRIPT_TABLES.glob("*"):
        if src.is_file():
            dst = OUT / f"manuscript_{src.name}" if not src.name.startswith("manuscript_") else OUT / src.name
            dst.write_bytes(src.read_bytes())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    save_table(
        make_main_results(),
        "table_01_benchmark_results",
        "Peak shear-strength prediction on public rock-joint benchmarks.",
        "tab:benchmark_results",
    )
    save_table(
        make_transform_ablation(),
        "table_02_transform_ablation",
        "Transformation and structural-design ablation on RockMB-2025.",
        "tab:transform_ablation",
    )
    save_table(
        make_physics_checks(),
        "table_03_constraint_checks",
        "Constraint-enforcement and diagnostic summary on fixed-context stress paths.",
        "tab:constraint_checks",
    )
    save_table(
        make_curve_metrics(),
        "table_04_curve_reconstruction",
        "Local full-curve reconstruction against C-BB, the calibrated Barton--Bandis curve law.",
        "tab:curve_reconstruction",
    )
    save_table(
        make_noise_table(),
        "table_05_noise_robustness",
        "Noise robustness under repeated forward prediction.",
        "tab:noise_robustness",
    )
    save_table(
        make_inverse_table(),
        "table_06_inverse_recovery",
        "Synthetic inverse parameter recovery under observation noise.",
        "tab:inverse_recovery",
    )
    copy_manuscript_tables()
    print(f"Saved paper tables to {OUT}")


if __name__ == "__main__":
    main()
