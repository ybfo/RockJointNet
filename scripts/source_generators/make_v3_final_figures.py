from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "paper_artifacts" / "experiment_results"
RECENT = ROOT / "results_recent_methods"
OUT = ROOT / "paper_artifacts_v3_final" / "figures"

COLORS = {
    "ours": "#9A3A22",
    "tabular": "#2E6F95",
    "neutral": "#57534E",
    "light": "#E9E1D6",
    "grid": "#D8CFC2",
    "danger": "#A23B45",
    "accent": "#C18A2B",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 320,
            "font.family": "DejaVu Serif",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "axes.edgecolor": "#3F3A35",
            "axes.linewidth": 0.8,
            "xtick.color": "#3F3A35",
            "ytick.color": "#3F3A35",
            "text.color": "#2D2926",
            "axes.labelcolor": "#2D2926",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def savefig(fig: plt.Figure, name: str, manifest: list[dict[str, str]], source: str, note: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    manifest.append({"figure": name, "source": source, "note": note})


def clean_axis(ax: plt.Axes) -> None:
    ax.grid(True, color=COLORS["grid"], linewidth=0.6, alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def metric_text(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    denom = np.maximum(np.abs(y_true), 1e-8)
    maape = float(np.mean(np.arctan(np.abs(err) / denom)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    return f"$R^2$={r2:.3f}\nRMSE={rmse:.3g}\nMAE={mae:.3g}\nMAAPE={maape:.3g}"


def plot_measured_predicted(manifest: list[dict[str, str]]) -> None:
    df = pd.read_csv(RESULTS / "prediction_exports.csv")
    panels = [
        ("rockmb_2025", "paper_70_30", "ResidualPeriodicMonotone", "RockMB paper split"),
        ("rockmb_2025", "group_5fold_fold1", "MonotonePeakNet", "RockMB group fold"),
        ("g5pf6k9n2w", "leave_one_profile_JC", "ResidualPeriodicMonotone", "G5 leave profile"),
        ("g5pf6k9n2w", "leave_one_immersion_360", "PeriodicMonotone", "G5 leave immersion"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.4))
    for ax, (dataset, protocol, model, title) in zip(axes.ravel(), panels):
        part = df[(df.dataset == dataset) & (df.protocol == protocol) & (df.model == model)].copy()
        y = part.y_true.to_numpy(float)
        yp = part.y_pred.to_numpy(float)
        lim_max = max(float(np.max(y)), float(np.max(yp))) * 1.08
        ax.scatter(y, yp, s=26, color=COLORS["ours"], edgecolor="white", linewidth=0.5, alpha=0.86)
        ax.plot([0, lim_max], [0, lim_max], color="#2D2926", linewidth=1.0, linestyle="--")
        ax.set_xlim(0, lim_max)
        ax.set_ylim(0, lim_max)
        ax.set_title(title)
        ax.set_xlabel("Measured peak shear strength (MPa)")
        ax.set_ylabel("Predicted peak shear strength (MPa)")
        ax.text(
            0.04,
            0.96,
            metric_text(y, yp),
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox=dict(facecolor="white", edgecolor=COLORS["grid"], boxstyle="round,pad=0.35", alpha=0.92),
        )
        clean_axis(ax)
    savefig(
        fig,
        "fig4_measured_vs_predicted_protocol_panels",
        manifest,
        "prediction_exports.csv",
        "Four protocol-aware measured-vs-predicted panels for the best valid proposed model.",
    )


def plot_benchmark_ranking(manifest: list[dict[str, str]]) -> None:
    df = pd.read_csv(RECENT / "recent_methods_summary.csv")
    keep = [
        ("rockmb_2025", "paper_70_30"),
        ("rockmb_2025", "group_5fold"),
        ("g5pf6k9n2w", "leave_one_profile_out"),
        ("g5pf6k9n2w", "leave_one_immersion_out"),
        ("g5pf6k9n2w", "random_75_25"),
        ("w7m28x23kw", "random_75_25"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(8.0, 9.0))
    for ax, (dataset, protocol) in zip(axes.ravel(), keep):
        part = df[(df.dataset == dataset) & (df.protocol == protocol)].copy()
        part = part.sort_values("RMSE_mean", ascending=True)
        labels = [m.replace("_ours", "").replace("_", "\n") for m in part.model]
        colors = [COLORS["ours"] if "ours" in m else COLORS["tabular"] for m in part.model]
        y = np.arange(len(part))
        ax.barh(y, part.RMSE_mean, xerr=part.RMSE_std.fillna(0.0), color=colors, edgecolor="white", linewidth=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7.7)
        ax.invert_yaxis()
        ax.set_title(f"{dataset} / {protocol}")
        ax.set_xlabel("RMSE lower is better")
        clean_axis(ax)
    legend = [
        Line2D([0], [0], color=COLORS["ours"], lw=6, label="proposed physics-structured"),
        Line2D([0], [0], color=COLORS["tabular"], lw=6, label="recent tabular baseline"),
    ]
    fig.legend(handles=legend, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.015))
    savefig(
        fig,
        "fig5_recent_method_ranked_benchmark",
        manifest,
        "recent_methods_summary.csv",
        "Protocol-specific RMSE rankings against 2021-2025 recent methods.",
    )


def plot_physics_validity_overlay(manifest: list[dict[str, str]]) -> None:
    summary = pd.read_csv(RECENT / "recent_methods_summary.csv")
    physics = pd.read_csv(RESULTS / "physics_validity_fixed_context_sweeps.csv")
    models = ["MonotonePeakNet_ours", "PeriodicMonotone_ours", "ResidualPeriodicMonotone_ours"]
    labels = ["Monotone\nPeakNet", "Periodic\nMonotone", "Residual\nPeriodic"]
    key = summary[(summary.dataset == "rockmb_2025") & (summary.protocol == "paper_70_30") & (summary.model.isin(models))]
    key = key.set_index("model").reindex(models)
    phys = physics.groupby("model")[["fixed_context_monotonic_violation_rate_percent", "mean_tau0_boundary_error"]].max()
    phys = phys.reindex(["MonotonePeakNet", "PeriodicMonotone", "ResidualPeriodicMonotone"])

    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.8))
    axes[0].bar(labels, key.RMSE_mean, color=COLORS["ours"], edgecolor="white")
    axes[0].set_ylabel("RMSE on RockMB paper split")
    axes[0].set_title("Accuracy")
    axes[1].bar(labels, phys.fixed_context_monotonic_violation_rate_percent, color=COLORS["accent"], edgecolor="white")
    axes[1].set_ylabel("Violation rate (%)")
    axes[1].set_title("Fixed-context monotonicity")
    axes[2].bar(labels, phys.mean_tau0_boundary_error, color=COLORS["neutral"], edgecolor="white")
    axes[2].set_ylabel(r"Mean $|\tau(0)|$")
    axes[2].set_title("Zero-stress boundary")
    for ax in axes:
        clean_axis(ax)
        ax.tick_params(axis="x", labelsize=8)
    savefig(
        fig,
        "fig6_architecture_ablation_physics_overlay",
        manifest,
        "recent_methods_summary.csv; physics_validity_fixed_context_sweeps.csv",
        "Accuracy and hard-constraint validity for the three proposed monotone variants.",
    )


def plot_transform_stability(manifest: list[dict[str, str]]) -> None:
    df = pd.read_csv(RESULTS / "nondimensional_transform_ablation.csv")
    labels = [
        "raw sigma\nraw tau",
        "log sigma\nraw tau",
        "raw sigma\nasinh tau",
        "log sigma\nasinh tau",
        "stress-path\nmonotone",
    ]
    x = np.arange(len(df))
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.9), sharex=True)
    axes[0].bar(x, df.RMSE, color=[COLORS["tabular"]] * 4 + [COLORS["ours"]], edgecolor="white")
    axes[0].set_ylabel("RMSE")
    axes[0].set_title("Prediction error")
    axes[1].bar(x, df.relative_L2_percent, color=[COLORS["tabular"]] * 4 + [COLORS["ours"]], edgecolor="white")
    axes[1].set_ylabel("Relative L2 error (%)")
    axes[1].set_title("TNNLS-style metric")
    axes[2].bar(x[:4], df.median_grad_norm.iloc[:4], color=COLORS["neutral"], edgecolor="white")
    axes[2].text(4, 0.03, "not\napplicable", ha="center", va="bottom", fontsize=8, color=COLORS["neutral"])
    axes[2].set_ylabel("Median grad norm")
    axes[2].set_title("Optimization stability")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7.5)
        clean_axis(ax)
    savefig(
        fig,
        "fig7_transform_stability_ablation",
        manifest,
        "nondimensional_transform_ablation.csv",
        "Nondimensionalization and target-transform ablation with relative L2 reporting.",
    )


def plot_stress_path_curves(manifest: list[dict[str, str]]) -> None:
    df = pd.read_csv(RESULTS / "feature_sensitivity_sweeps.csv")
    part = df[(df.dataset == "rockmb_2025") & (df.feature == "normal_stress_mpa")].copy()
    sample_ids = sorted(part.sample_id.unique())[:8]
    fig, ax = plt.subplots(figsize=(6.7, 4.3))
    for sid in sample_ids:
        g = part[part.sample_id == sid].sort_values("feature_value")
        ax.plot(g.feature_value, g.prediction, lw=1.3, color=COLORS["ours"], alpha=0.42)
    median_curve = part.groupby("feature_value", as_index=False).prediction.median()
    ax.plot(median_curve.feature_value, median_curve.prediction, lw=2.6, color="#2D2926", label="median predicted path")
    ax.axvspan(part.feature_value.min(), np.quantile(part.feature_value, 0.35), color=COLORS["light"], alpha=0.55, label="low-stress trusted-prior zone")
    ax.set_xlabel("Normal stress sweep (MPa)")
    ax.set_ylabel("Predicted peak shear strength (MPa)")
    ax.set_title("Fixed-context monotone stress-path sweeps")
    ax.legend(frameon=False, loc="upper left")
    clean_axis(ax)
    savefig(
        fig,
        "fig8_fixed_context_stress_path_curves",
        manifest,
        "feature_sensitivity_sweeps.csv",
        "Representative fixed-context sigma sweeps showing monotone stress-path behavior.",
    )


def plot_noise_inverse(manifest: list[dict[str, str]]) -> None:
    noise = pd.read_csv(RESULTS / "noise_robustness.csv")
    inv = pd.read_csv(RESULTS / "synthetic_inverse_recovery.csv")
    fig, axes = plt.subplots(1, 3, figsize=(8.8, 3.0))
    for model, g in noise.groupby("model"):
        color = COLORS["ours"] if "Residual" in model else COLORS["tabular"]
        axes[0].plot(g.noise_level * 100, g.RMSE, marker="o", lw=1.8, label=model, color=color)
        axes[1].plot(g.noise_level * 100, g.relative_L2_percent, marker="o", lw=1.8, label=model, color=color)
    inv_summary = inv.groupby("noise_level")[["a_relative_error_percent", "b_relative_error_percent"]].mean().reset_index()
    axes[2].plot(inv_summary.noise_level * 100, inv_summary.a_relative_error_percent, marker="o", lw=1.8, color=COLORS["accent"], label="a parameter")
    axes[2].plot(inv_summary.noise_level * 100, inv_summary.b_relative_error_percent, marker="s", lw=1.8, color=COLORS["danger"], label="b parameter")
    axes[0].set_ylabel("RMSE")
    axes[1].set_ylabel("Relative L2 error (%)")
    axes[2].set_ylabel("Inverse relative error (%)")
    for ax, title in zip(axes, ["Noisy-data RMSE", "Noisy-data relative L2", "Synthetic inverse recovery"]):
        ax.set_xlabel("Training noise level (%)")
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=7.5)
        clean_axis(ax)
    savefig(
        fig,
        "fig10_noise_robustness_inverse_recovery",
        manifest,
        "noise_robustness.csv; synthetic_inverse_recovery.csv",
        "TNNLS-style noisy forward prediction and synthetic inverse recovery experiment.",
    )


def plot_error_anatomy(manifest: list[dict[str, str]]) -> None:
    df = pd.read_csv(RESULTS / "prediction_exports.csv")
    df["abs_error"] = (df.y_pred - df.y_true).abs()
    df["signed_error"] = df.y_pred - df.y_true
    fig, axes = plt.subplots(1, 3, figsize=(8.7, 3.0))
    axes[0].scatter(df.y_true, df.abs_error, s=18, color=COLORS["ours"], alpha=0.7, edgecolor="white", linewidth=0.35)
    axes[0].set_xlabel("Measured peak shear strength (MPa)")
    axes[0].set_ylabel("Absolute error (MPa)")
    axes[0].set_title("Error grows with target scale")
    order = df.groupby("protocol").abs_error.median().sort_values().index
    data = [df[df.protocol == p].abs_error for p in order]
    axes[1].boxplot(data, tick_labels=[p.replace("_", "\n") for p in order], patch_artist=True, boxprops=dict(facecolor=COLORS["light"], color=COLORS["neutral"]), medianprops=dict(color=COLORS["danger"]))
    axes[1].set_ylabel("Absolute error (MPa)")
    axes[1].set_title("Protocol-level error")
    axes[1].tick_params(axis="x", labelsize=6.8)
    axes[2].hist(df.signed_error, bins=22, color=COLORS["tabular"], edgecolor="white", alpha=0.9)
    axes[2].axvline(0, color="#2D2926", linestyle="--", linewidth=1.0)
    axes[2].set_xlabel("Signed error (MPa)")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Residual distribution")
    for ax in axes:
        clean_axis(ax)
    savefig(
        fig,
        "fig11_error_anatomy",
        manifest,
        "prediction_exports.csv",
        "Failure-mode diagnostics across exported prediction protocols.",
    )


def plot_excluded_stress_test(manifest: list[dict[str, str]]) -> None:
    df = pd.read_csv(RECENT / "recent_methods_summary.csv")
    part = df[(df.dataset == "w7m28x23kw") & (df.protocol == "leave_one_joint_type_out")].copy()
    part = part.sort_values("R2_mean", ascending=False)
    colors = [COLORS["ours"] if "ours" in m else COLORS["tabular"] for m in part.model]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    y = np.arange(len(part))
    ax.barh(y, part.R2_mean, xerr=part.R2_std.fillna(0.0), color=colors, edgecolor="white")
    ax.axvline(0, color=COLORS["danger"], linestyle="--", linewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels([m.replace("_ours", "").replace("_", "\n") for m in part.model], fontsize=7.8)
    ax.invert_yaxis()
    ax.set_xlabel(r"$R^2$ on leave-one-joint-type stress test")
    ax.set_title("Excluded OOD stress test: no method reaches nonnegative R2")
    ax.text(
        0.98,
        0.08,
        "Transparent failure disclosure:\nexcluded from SOTA claims",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        bbox=dict(facecolor="white", edgecolor=COLORS["grid"], boxstyle="round,pad=0.35"),
    )
    clean_axis(ax)
    savefig(
        fig,
        "fig12_excluded_joint_type_ood_stress_test",
        manifest,
        "recent_methods_summary.csv",
        "Leave-one-joint-type result retained as a failure-disclosure stress test.",
    )


def plot_protocol_map(manifest: list[dict[str, str]]) -> None:
    winners = pd.read_csv(RECENT / "recent_methods_winners.csv")
    protocols = winners[["dataset", "protocol", "model"]].copy()
    protocols["claim_status"] = np.where(protocols.model.str.contains("ours"), "proposed wins", "baseline wins")
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ax.axis("off")
    cols = ["Dataset", "Protocol", "Status"]
    x = [0.02, 0.34, 0.68]
    for xi, col in zip(x, cols):
        ax.text(xi, 0.94, col, weight="bold", transform=ax.transAxes)
    for i, row in protocols.iterrows():
        y = 0.84 - i * 0.115
        status_color = COLORS["ours"] if row.claim_status == "proposed wins" else COLORS["tabular"]
        ax.add_patch(Rectangle((0.01, y - 0.035), 0.96, 0.075, transform=ax.transAxes, facecolor=COLORS["light"] if i % 2 == 0 else "white", edgecolor="none"))
        ax.text(x[0], y, row.dataset, transform=ax.transAxes, va="center")
        ax.text(x[1], y, row.protocol, transform=ax.transAxes, va="center")
        ax.text(x[2], y, row.claim_status, transform=ax.transAxes, va="center", color=status_color, weight="bold")
    ax.set_title("Dataset and protocol map for valid benchmark claims", loc="left", pad=8)
    savefig(
        fig,
        "fig3_dataset_protocol_map",
        manifest,
        "recent_methods_winners.csv",
        "Compact protocol map separating valid SOTA claims from stress-test disclosure.",
    )


def main() -> None:
    setup_style()
    manifest: list[dict[str, str]] = []
    plot_protocol_map(manifest)
    plot_measured_predicted(manifest)
    plot_benchmark_ranking(manifest)
    plot_physics_validity_overlay(manifest)
    plot_transform_stability(manifest)
    plot_stress_path_curves(manifest)
    plot_noise_inverse(manifest)
    plot_error_anatomy(manifest)
    plot_excluded_stress_test(manifest)
    pd.DataFrame(manifest).to_csv(OUT / "figure_manifest.csv", index=False)
    lines = [
        "# V3 Final Figures",
        "",
        "These figures were generated from rerun result CSVs only. The previously rejected figure set was not reused.",
        "",
        "Each figure is saved as both PNG and PDF. See `figure_manifest.csv` for data provenance.",
        "",
        "## Figure index",
    ]
    for item in manifest:
        lines.append(f"- `{item['figure']}`: {item['note']} Source: `{item['source']}`.")
    lines.extend(
        [
            "",
            "## Usage notes",
            "- Main text candidates: Fig. 3, Fig. 4, Fig. 5, Fig. 6, Fig. 7, Fig. 8, and Fig. 10.",
            "- Appendix candidates: Fig. 11 and Fig. 12.",
            "- Fig. 12 is intentionally framed as failure disclosure, not as a SOTA claim.",
            "- Figure 1 and Figure 2 remain schematic/model-diagram items and should be drawn manually or with vector tools rather than generated from result CSVs.",
        ]
    )
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {len(manifest)} figures to {OUT}")


if __name__ == "__main__":
    main()
