from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
EXP = DATA / "experiment_results"
RECENT = DATA / "recent_methods"
LOCAL = DATA / "local_curve"
OUT = ROOT / "outputs" / "figures"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "mathtext.fontset": "dejavuserif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 160,
            "savefig.dpi": 380,
        }
    )


def save(
    fig: plt.Figure,
    name: str,
    manifest: list[dict[str, str]],
    source: str,
    note: str,
    rect: tuple[float, float, float, float] | None = None,
) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if rect is None:
        fig.tight_layout()
    else:
        fig.tight_layout(rect=rect)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    manifest.append({"figure": name, "source": source, "note": note})


def _pivot_feature(feature: str) -> pd.DataFrame:
    sweeps = pd.read_csv(EXP / "counterfactual_feature_sweeps.csv")
    part = sweeps[sweeps["feature"] == feature].copy()
    table = part.pivot_table(index="sample_id", columns="feature_value", values="prediction")
    table = table.assign(_median=table.median(axis=1)).sort_values("_median").drop(columns="_median")
    return table


def metric_text(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    maape = float(np.mean(np.arctan(np.abs(err) / np.maximum(np.abs(y_true), 1e-8))))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / max(ss_tot, 1e-12))
    return f"$R^2$={r2:.3f}\nRMSE={rmse:.3g}\nMAE={mae:.3g}\nMAAPE={maape:.3g}"


def fig03_solution_and_error(manifest: list[dict[str, str]]) -> None:
    normal_table = _pivot_feature("normal_stress_mpa")
    xs = normal_table.columns.to_numpy(float)
    ranks = np.arange(len(normal_table))
    xx, yy = np.meshgrid(xs, ranks)
    zz = normal_table.to_numpy(float)

    pred = pd.read_csv(EXP / "prediction_cases.csv")
    pred["abs_error"] = (pred["y_pred"] - pred["y_true"]).abs()
    pred["case"] = pred["dataset"] + " / " + pred["protocol"]
    cases = [
        "rockmb_2025 / paper_70_30",
        "rockmb_2025 / group_5fold_fold1",
        "g5pf6k9n2w / leave_one_profile_JC",
        "g5pf6k9n2w / leave_one_immersion_360",
    ]
    rows = []
    max_len = 0
    for case in cases:
        part = pred[pred["case"] == case].sort_values("sample_index")
        values = part["abs_error"].to_numpy(float)
        rows.append(values)
        max_len = max(max_len, len(values))
    err_field = np.full((len(rows), max_len), np.nan)
    for i, values in enumerate(rows):
        err_field[i, : len(values)] = values

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.35))
    levels = np.linspace(0.0, max(float(np.nanmax(zz)), 1e-6), 26)
    cf0 = axes[0].contourf(xx, yy, zz, levels=levels, cmap="turbo")
    axes[0].contour(xx, yy, zz, levels=levels[::4], colors="white", linewidths=0.25, alpha=0.35)
    axes[0].set_title(r"(a) Predicted stress-path field $\hat{\tau}_p$")
    axes[0].set_xlabel(r"Normal stress $\sigma_n$ (MPa)")
    axes[0].set_ylabel("Fixed-context rank")
    fig.colorbar(cf0, ax=axes[0], fraction=0.046, pad=0.025)

    im = axes[1].imshow(err_field, aspect="auto", cmap="turbo", interpolation="nearest")
    axes[1].set_title(r"(b) Absolute-error field $|\tau_p-\hat{\tau}_p|$")
    axes[1].set_xlabel("Test-sample order inside protocol")
    axes[1].set_yticks(np.arange(len(cases)))
    axes[1].set_yticklabels([c.replace(" / ", "\n") for c in cases], fontsize=7.4)
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.025)
    save(
        fig,
        "figure_03_prediction_and_error_field",
        manifest,
        "counterfactual_feature_sweeps.csv; prediction_cases.csv",
        "Predicted stress-path solution field and protocol-wise absolute-error field.",
    )


def fig04_multifeature_contours(manifest: list[dict[str, str]]) -> None:
    features = [
        ("normal_stress_mpa", r"(a) $\hat{\tau}_p$ vs $\sigma_n$", r"Normal stress $\sigma_n$ (MPa)"),
        ("jrc", r"(b) $\hat{\tau}_p$ vs JRC", "JRC"),
        ("ucs_mpa", r"(c) $\hat{\tau}_p$ vs UCS", "UCS (MPa)"),
        ("specimen_length_mm", r"(d) $\hat{\tau}_p$ vs specimen length", "Specimen length (mm)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 5.9), sharey=True)
    for panel_idx, (ax, (feature, title, xlabel)) in enumerate(zip(axes.ravel(), features)):
        table = _pivot_feature(feature)
        xs = table.columns.to_numpy(float)
        ranks = np.arange(len(table))
        xx, yy = np.meshgrid(xs, ranks)
        zz = table.to_numpy(float)
        levels = np.linspace(0.0, max(float(np.nanmax(zz)), 1e-6), 24)
        cf = ax.contourf(xx, yy, zz, levels=levels, cmap="turbo")
        ax.contour(xx, yy, zz, levels=levels[::4], colors="white", linewidths=0.25, alpha=0.35)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Fixed-context rank" if panel_idx in (0, 2) else "")
        ax.grid(True, ls=":", lw=0.45, alpha=0.75)
        fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.025, label=r"$\hat{\tau}_p$ (MPa)")
    fig.text(
        0.5,
        0.01,
        "Each horizontal row is one fixed rock-joint context; rows are sorted by median predicted strength within each panel.",
        ha="center",
        fontsize=9.2,
    )
    save(
        fig,
        "figure_04_feature_response_maps",
        manifest,
        "counterfactual_feature_sweeps.csv",
        "Counterfactual response contours for major variables under fixed rock-joint contexts.",
    )


def fig05_slope_curvature(manifest: list[dict[str, str]]) -> None:
    features = ["normal_stress_mpa", "jrc", "ucs_mpa", "youngs_modulus_gpa"]
    fig, axes = plt.subplots(2, 4, figsize=(13.2, 5.3))
    for col, feature in enumerate(features):
        table = _pivot_feature(feature)
        xs = table.columns.to_numpy(float)
        yy_idx = np.arange(len(table))
        xx, yy = np.meshgrid(xs, yy_idx)
        zz = table.to_numpy(float)
        d1 = np.gradient(zz, xs, axis=1)
        curv = np.abs(np.gradient(d1, xs, axis=1))
        cf0 = axes[0, col].contourf(xx, yy, np.maximum(d1, 0.0), levels=22, cmap="turbo")
        axes[0, col].set_title(f"({chr(97 + col)}) positive slope: {feature.replace('_', ' ')}")
        axes[0, col].set_xlabel(feature.replace("_", " "))
        axes[0, col].set_ylabel("context rank")
        fig.colorbar(cf0, ax=axes[0, col], fraction=0.046, pad=0.025)
        cf1 = axes[1, col].contourf(xx, yy, curv, levels=22, cmap="Blues")
        axes[1, col].set_title(f"({chr(101 + col)}) curvature magnitude")
        axes[1, col].set_xlabel(feature.replace("_", " "))
        axes[1, col].set_ylabel("context rank")
        fig.colorbar(cf1, ax=axes[1, col], fraction=0.046, pad=0.025)
    save(
        fig,
        "figure_05_slope_and_curvature_maps",
        manifest,
        "counterfactual_feature_sweeps.csv",
        "Derivative and curvature maps from fixed-context counterfactual stress sweeps.",
    )


def fig06_derived_quantities(manifest: list[dict[str, str]]) -> None:
    table = _pivot_feature("normal_stress_mpa")
    sigma = table.columns.to_numpy(float)
    sample_ids = list(table.index[:6])
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 5.6))
    titles = ["Peak shear strength", r"Stress-path slope $d\tau/d\sigma$", r"Curvature $d^2\tau/d\sigma^2$"]
    for row in range(2):
        for col in range(3):
            ax = axes[row, col]
            for sid in sample_ids[row * 3 : row * 3 + 3]:
                path = table.loc[sid].to_numpy(float)
                slope = np.gradient(path, sigma)
                curv = np.gradient(slope, sigma)
                series = [path, slope, curv][col]
                ax.plot(sigma, series, marker="o", ms=2.7, lw=1.2, label=f"context {int(sid)}")
            ax.set_title(f"({chr(97 + row * 3 + col)}) {titles[col]}")
            ax.set_xlabel(r"$\sigma_n$ (MPa)")
            ax.set_ylabel([r"$\hat{\tau}_p$ (MPa)", "slope", "curvature"][col])
            ax.grid(True, ls=":", lw=0.45)
            if col == 0:
                ax.legend(fontsize=7, frameon=True)
    save(
        fig,
        "figure_06_stress_path_profiles",
        manifest,
        "counterfactual_feature_sweeps.csv",
        "Stress-path curves and derived slope/curvature quantities for fixed contexts.",
    )


def fig09_prediction_performance(manifest: list[dict[str, str]]) -> None:
    df = pd.read_csv(EXP / "prediction_cases.csv")
    panels = [
        ("g5pf6k9n2w", "leave_one_immersion_360", "PeriodicMonotone", "G5 leave immersion"),
        ("g5pf6k9n2w", "leave_one_profile_JC", "ResidualPeriodicMonotone", "G5 leave profile"),
        ("rockmb_2025", "group_5fold_fold1", "MonotonePeakNet", "RockMB group fold"),
        ("rockmb_2025", "paper_70_30", "ResidualPeriodicMonotone", "RockMB paper split"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.6))
    for ax, (dataset, protocol, model, title) in zip(axes.ravel(), panels):
        part = df[(df["dataset"] == dataset) & (df["protocol"] == protocol) & (df["model"] == model)].copy()
        part = part.sort_values("sample_index").reset_index(drop=True)
        x = np.arange(len(part))
        y = part["y_true"].to_numpy(float)
        yp = part["y_pred"].to_numpy(float)
        err = np.abs(yp - y)
        ax2 = ax.twinx()
        ax2.bar(x, err, color="#D0D0D0", width=0.82, alpha=0.72, label="absolute error", zorder=1)
        ax.plot(x, y, color="#111111", marker="o", ms=2.4, lw=1.15, label="actual strength", zorder=3)
        ax.plot(x, yp, color="#D62728", marker=".", ms=3.0, lw=1.05, label="predicted strength", zorder=4)
        ax.set_title(f"{dataset} / {protocol}")
        ax.set_xlabel("Data No.")
        ax.set_ylabel("Strength (MPa)")
        ax2.set_ylabel("Error (MPa)")
        ax.grid(True, ls=":", lw=0.45, alpha=0.65)
        ax.text(
            0.02,
            0.97,
            title + "\n" + metric_text(y, yp),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox=dict(facecolor="white", edgecolor="#D8D8D8", boxstyle="round,pad=0.30", alpha=0.94),
        )
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    bar_handles = [plt.Rectangle((0, 0), 1, 1, color="#D0D0D0", alpha=0.72)]
    fig.legend(
        handles + bar_handles,
        labels + ["absolute error"],
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
    )
    save(
        fig,
        "figure_09_prediction_panels",
        manifest,
        "prediction_cases.csv",
        "Actual/predicted/error performance panels with error bars placed behind line plots.",
        rect=(0, 0.065, 1, 1),
    )


def fig_full_curve(manifest: list[dict[str, str]]) -> None:
    pred = pd.read_csv(LOCAL / "curve_reconstruction_predictions.csv")
    normals = sorted(pred["normal_stress_mpa"].unique())
    fig, axes = plt.subplots(2, len(normals), figsize=(12.0, 6.2), sharex=True)
    if len(normals) == 1:
        axes = axes.reshape(2, 1)
    styles = {
        "Experiment": {"color": "#222222", "lw": 2.1, "ls": "-", "zorder": 3},
        "GeoSPIN": {"color": "#D62728", "lw": 2.0, "ls": "-", "zorder": 4},
        "C-BB": {"color": "#6E6E6E", "lw": 1.7, "ls": "--", "zorder": 2},
    }
    for col, normal in enumerate(normals):
        group = pred[pred["normal_stress_mpa"] == normal].sort_values("shear_displacement_mm")
        x = group["shear_displacement_mm"].to_numpy(float)
        top = axes[0, col]
        bottom = axes[1, col]
        top.plot(x, group["calibrated_shear_stress_mpa"], label="Experiment", **styles["Experiment"])
        top.plot(x, group["ours_shear_stress_mpa"], label="GeoSPIN", **styles["GeoSPIN"])
        top.plot(x, group["bb_shear_stress_mpa"], label="C-BB", **styles["C-BB"])
        bottom.plot(x, group["dilation_proxy_mm"], label="Experiment", **styles["Experiment"])
        bottom.plot(x, group["ours_dilation_proxy_mm"], label="GeoSPIN", **styles["GeoSPIN"])
        bottom.plot(x, group["bb_dilation_proxy_mm"], label="C-BB", **styles["C-BB"])
        top.set_title(rf"$\sigma_n={normal:g}$ MPa")
        for ax in (top, bottom):
            ax.grid(True, ls=":", lw=0.5, color="#BDBDBD")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    axes[0, 0].set_ylabel("Shear stress (MPa)")
    axes[1, 0].set_ylabel("Dilation proxy (mm)")
    for ax in axes[1, :]:
        ax.set_xlabel("Shear displacement (mm)")
    axes[0, 0].text(-0.18, 1.08, "(a)", transform=axes[0, 0].transAxes, fontsize=12, weight="bold")
    axes[1, 0].text(-0.18, 1.08, "(b)", transform=axes[1, 0].transAxes, fontsize=12, weight="bold")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(
        fig,
        "figure_10_curve_reconstruction",
        manifest,
        "curve_reconstruction_predictions.csv",
        "Two-row by three-column local direct-shear full-curve comparison: experiment, GeoSPIN, and C-BB.",
    )


def main() -> None:
    setup_style()
    manifest: list[dict[str, str]] = []
    fig03_solution_and_error(manifest)
    fig04_multifeature_contours(manifest)
    fig05_slope_curvature(manifest)
    fig06_derived_quantities(manifest)
    fig09_prediction_performance(manifest)
    fig_full_curve(manifest)
    print(f"Saved {len(manifest)} paper figures to {OUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerate final paper figures from saved experiment CSVs.")
    _ = parser.parse_args()
    main()
