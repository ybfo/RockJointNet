from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from sklearn.model_selection import train_test_split

from run_physics_grade_experiments import (
    ResidualPeriodicMonotonePeakNet,
    fit_monotone_model,
    load_rockmb,
    predict_monotone,
    predict_monotone_path,
)


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "paper_artifacts" / "experiment_results"
RECENT = ROOT / "results_recent_methods"
OUT = ROOT / "paper_artifacts_v3_final" / "figures_literature_style"


def style() -> None:
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
            "savefig.dpi": 360,
        }
    )


def save(fig: plt.Figure, name: str, manifest: list[dict[str, str]], source: str, note: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    manifest.append({"figure": name, "source": source, "note": note})


def train_final_model():
    x, y, groups = load_rockmb()
    train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=0.30, random_state=42)
    model, scaler, sigma_ref = fit_monotone_model(
        ResidualPeriodicMonotonePeakNet,
        x[train_idx],
        y[train_idx],
        seed=909,
        epochs=260,
    )
    return x, y, groups, train_idx, test_idx, model, scaler, sigma_ref


def model_grid(x: np.ndarray, model, scaler, sigma_ref: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sigma = np.linspace(0.0, float(np.percentile(x[:, 0], 98)), 130)
    jrc = np.linspace(float(np.percentile(x[:, 3], 2)), float(np.percentile(x[:, 3], 98)), 120)
    ss, jj = np.meshgrid(sigma, jrc)
    base = np.tile(np.median(x, axis=0), (ss.size, 1)).astype("float32")
    base[:, 0] = ss.ravel()
    base[:, 3] = jj.ravel()
    pred = predict_monotone(model, scaler, sigma_ref, base).reshape(ss.shape)
    return ss, jj, pred


def fig_solution_error_field(x, y, test_idx, model, scaler, sigma_ref, manifest):
    ss, jj, pred = model_grid(x, model, scaler, sigma_ref)
    test_x = x[test_idx]
    test_y = y[test_idx]
    test_pred = predict_monotone(model, scaler, sigma_ref, test_x)
    abs_err = np.abs(test_pred - test_y)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.25))
    levels_pred = np.linspace(float(np.min(pred)), float(np.max(pred)), 24)
    cf0 = axes[0].contourf(ss, jj, pred, levels=levels_pred, cmap="turbo")
    axes[0].set_title(r"(a) Predicted field $\hat{\tau}_p(\sigma_n,\mathrm{JRC})$")
    axes[0].set_xlabel(r"Normal stress $\sigma_n$ (MPa)")
    axes[0].set_ylabel("JRC")
    fig.colorbar(cf0, ax=axes[0], fraction=0.046, pad=0.025)

    triang = mtri.Triangulation(test_x[:, 0], test_x[:, 3])
    levels_err = np.linspace(0.0, max(float(np.max(abs_err)), 1e-6), 20)
    cf1 = axes[1].tricontourf(triang, abs_err, levels=levels_err, cmap="turbo")
    axes[1].scatter(test_x[:, 0], test_x[:, 3], s=9, c="k", marker=".", alpha=0.55)
    axes[1].set_title(r"(b) Absolute error $|\tau_p-\hat{\tau}_p|$")
    axes[1].set_xlabel(r"Normal stress $\sigma_n$ (MPa)")
    axes[1].set_ylabel("JRC")
    fig.colorbar(cf1, ax=axes[1], fraction=0.046, pad=0.025)
    save(
        fig,
        "lit_fig03_solution_and_absolute_error_field",
        manifest,
        "rockmb_2025_dataset.csv; trained ResidualPeriodicMonotone",
        "PINN-style predicted solution field and absolute-error field over sigma-JRC coordinates.",
    )


def fig_feature_contour_atlas(manifest):
    sweeps = pd.read_csv(RESULTS / "feature_sensitivity_sweeps.csv")
    features = ["normal_stress_mpa", "jrc", "ucs_mpa", "specimen_length_mm"]
    titles = [
        r"(a) $\hat{\tau}_p$ response field vs $\sigma_n$",
        r"(b) $\hat{\tau}_p$ response field vs JRC",
        r"(c) $\hat{\tau}_p$ response field vs UCS",
        r"(d) $\hat{\tau}_p$ response field vs specimen length",
    ]
    xlabels = [
        r"Normal stress $\sigma_n$ (MPa)",
        "JRC",
        "UCS (MPa)",
        "Specimen length (mm)",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 5.9), sharey=True)
    for panel_idx, (ax, feat, title) in enumerate(zip(axes.ravel(), features, titles)):
        part = sweeps[sweeps.feature == feat].copy()
        table = part.pivot_table(index="sample_id", columns="feature_value", values="prediction")
        table = table.assign(_median=table.median(axis=1)).sort_values("_median").drop(columns="_median")
        xs = table.columns.to_numpy(float)
        ranks = np.arange(len(table))
        xx, yy = np.meshgrid(xs, ranks)
        values = table.to_numpy(float)
        levels = np.linspace(0.0, max(float(np.nanmax(values)), 1e-6), 24)
        cf = ax.contourf(xx, yy, values, levels=levels, cmap="turbo")
        ax.contour(xx, yy, values, levels=levels[::3], colors="white", linewidths=0.25, alpha=0.35)
        ax.set_title(title)
        ax.set_xlabel(xlabels[features.index(feat)])
        if panel_idx in (0, 2):
            ax.set_ylabel("Fixed-context rank\n(low to high median strength)")
        else:
            ax.set_ylabel("")
        fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.025, label=r"$\hat{\tau}_p$ (MPa)")
        ax.grid(True, ls=":", lw=0.45, alpha=0.8)
    fig.text(
        0.5,
        0.01,
        "Each horizontal row is one fixed rock-joint context; rows are sorted by the median predicted peak strength within each panel.",
        ha="center",
        fontsize=9.2,
    )
    save(
        fig,
        "lit_fig04_multifeature_prediction_contours",
        manifest,
        "feature_sensitivity_sweeps.csv",
        "Counterfactual response-field contours for major rock-joint variables with fixed contexts sorted by median predicted strength.",
    )


def fig_curvature_error_atlas(manifest):
    sweeps = pd.read_csv(RESULTS / "feature_sensitivity_sweeps.csv")
    features = ["normal_stress_mpa", "jrc", "ucs_mpa", "youngs_modulus_gpa"]
    fig, axes = plt.subplots(2, 4, figsize=(13.2, 5.3))
    for col, feat in enumerate(features):
        part = sweeps[sweeps.feature == feat].copy()
        table = part.pivot_table(index="sample_id", columns="feature_value", values="prediction")
        xx, yy = np.meshgrid(table.columns.to_numpy(float), table.index.to_numpy(float))
        zz = table.to_numpy(float)
        d1 = np.gradient(zz, axis=1)
        curv = np.abs(np.gradient(d1, axis=1))
        cf0 = axes[0, col].contourf(xx, yy, np.maximum(d1, 0.0), levels=22, cmap="turbo")
        axes[0, col].set_title(f"({chr(97 + col)}) positive slope: {feat}")
        axes[0, col].set_xlabel(feat.replace("_", " "))
        axes[0, col].set_ylabel("context id")
        fig.colorbar(cf0, ax=axes[0, col], fraction=0.046, pad=0.025)
        cf1 = axes[1, col].contourf(xx, yy, curv, levels=22, cmap="Blues")
        axes[1, col].set_title(f"({chr(101 + col)}) curvature magnitude")
        axes[1, col].set_xlabel(feat.replace("_", " "))
        axes[1, col].set_ylabel("context id")
        fig.colorbar(cf1, ax=axes[1, col], fraction=0.046, pad=0.025)
    save(
        fig,
        "lit_fig05_slope_and_curvature_fields",
        manifest,
        "feature_sensitivity_sweeps.csv",
        "TNNLS-style derivative/curvature field maps from fixed-context counterfactual sweeps.",
    )


def fig_derived_quantities(x, test_idx, model, scaler, sigma_ref, manifest):
    sample_ids = test_idx[:6]
    sigma = np.linspace(0.0, float(np.percentile(x[:, 0], 98)), 45)
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 5.6))
    titles = ["Peak shear strength", r"Stress-path slope $d\tau/d\sigma$", r"Curvature $d^2\tau/d\sigma^2$"]
    for row in range(2):
        for col in range(3):
            ax = axes[row, col]
            for k, idx in enumerate(sample_ids[row * 3 : row * 3 + 3]):
                path = predict_monotone_path(model, scaler, sigma_ref, x[idx : idx + 1], sigma)
                slope = np.gradient(path, sigma + 1e-9)
                curv = np.gradient(slope, sigma + 1e-9)
                series = [path, slope, curv][col]
                ax.plot(sigma, series, marker="o", ms=2.8, lw=1.2, label=f"context {int(idx)}")
            ax.set_title(f"({chr(97 + row * 3 + col)}) {titles[col]}")
            ax.set_xlabel(r"$\sigma_n$ (MPa)")
            ax.set_ylabel([r"$\hat{\tau}_p$ (MPa)", "slope", "curvature"][col])
            ax.grid(True, ls=":", lw=0.45)
            if col == 0:
                ax.legend(fontsize=7, frameon=True)
    save(
        fig,
        "lit_fig06_derived_stress_path_quantities",
        manifest,
        "trained ResidualPeriodicMonotone; rockmb_2025_dataset.csv",
        "Curve-family plot of predicted strength and derived stress-path quantities.",
    )


def fig_collocation_and_sweep_map(x, y, train_idx, test_idx, manifest):
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.6))
    axes[0].scatter(x[train_idx, 0], x[train_idx, 3], s=13, c="royalblue", label="training data", alpha=0.82)
    axes[0].scatter(x[test_idx, 0], x[test_idx, 3], s=20, c="crimson", marker="x", label="test data", alpha=0.9)
    axes[0].axvline(0.0, color="k", lw=1.0, ls="--", label=r"hard boundary $\tau(0)=0$")
    for q in [25, 50, 75]:
        axes[0].axvline(np.percentile(x[:, 0], q), color="tomato", lw=0.8, alpha=0.55)
    axes[0].set_xlabel(r"Normal stress $\sigma_n$ (MPa)")
    axes[0].set_ylabel("JRC")
    axes[0].set_title("(a) Data, boundary, and sweep anchors")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, ls=":", lw=0.45)

    axes[1].scatter(x[:, 0], y, s=13, c="royalblue", alpha=0.66, label="observations")
    sigma = np.linspace(0.0, float(np.percentile(x[:, 0], 98)), 80)
    for jrc in [np.percentile(x[:, 3], 20), np.percentile(x[:, 3], 50), np.percentile(x[:, 3], 80)]:
        axes[1].plot(sigma, np.interp(sigma, [sigma.min(), sigma.max()], [0, np.percentile(y, 35 + jrc)]), lw=0.9, alpha=0.1)
    axes[1].axvline(0.0, color="k", lw=1.0, ls="--")
    axes[1].set_xlabel(r"Normal stress $\sigma_n$ (MPa)")
    axes[1].set_ylabel(r"Peak shear strength $\tau_p$ (MPa)")
    axes[1].set_title("(b) Observation map for supervised residual")
    axes[1].grid(True, ls=":", lw=0.45)
    save(
        fig,
        "lit_fig07_data_collocation_boundary_map",
        manifest,
        "rockmb_2025_dataset.csv",
        "PINN-style map showing training/test points, zero-stress boundary, and sweep anchor locations.",
    )


def fig_barton_bandis_proxy(x, model, scaler, sigma_ref, manifest):
    sigma = np.linspace(0.02, float(np.percentile(x[:, 0], 95)), 120)
    base = np.median(x, axis=0).astype("float32")
    jrc_values = [6.0, 10.0, 14.0]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    for jrc in jrc_values:
        context = base.copy()
        context[3] = jrc
        context[1] = float(np.percentile(x[:, 1], 70))
        pred = predict_monotone_path(model, scaler, sigma_ref, context[None, :], sigma)
        phi_b = np.deg2rad(30.0)
        jcs = max(float(context[1]), 1e-6)
        bb_angle = phi_b + np.deg2rad(jrc * np.log10(np.maximum(jcs / sigma, 1.0)))
        bb = sigma * np.tan(np.clip(bb_angle, 0, np.deg2rad(85)))
        axes[0].plot(sigma, pred, lw=1.8, label=f"model JRC={jrc:g}")
        axes[0].plot(sigma, bb, "k--", lw=1.0, alpha=0.58)
        axes[1].plot(sigma, np.abs(pred - bb), lw=1.6, label=f"JRC={jrc:g}")
    axes[0].set_title("(a) Proposed model vs BB proxy prior")
    axes[0].set_xlabel(r"Normal stress $\sigma_n$ (MPa)")
    axes[0].set_ylabel(r"Peak shear strength $\tau_p$ (MPa)")
    axes[0].legend(fontsize=8)
    axes[1].set_title("(b) Absolute deviation from proxy")
    axes[1].set_xlabel(r"Normal stress $\sigma_n$ (MPa)")
    axes[1].set_ylabel("Absolute deviation (MPa)")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(True, ls=":", lw=0.45)
    fig.text(
        0.5,
        -0.04,
        "Note: RockMB does not provide phi_b/JCS directly; UCS is used as a proxy JCS and phi_b=30 deg only for local-prior visualization.",
        ha="center",
        fontsize=9,
    )
    save(
        fig,
        "lit_fig08_barton_bandis_proxy_curves",
        manifest,
        "trained ResidualPeriodicMonotone; rockmb_2025_dataset.csv",
        "Barton-Bandis-style local-prior comparison using explicitly labeled proxy assumptions.",
    )


def fig_prediction_performance_grid(manifest):
    pred = pd.read_csv(RESULTS / "prediction_exports.csv")
    panels = list(pred.groupby(["dataset", "protocol", "model"]).groups.keys())
    fig = plt.figure(figsize=(11.4, 8.4))
    outer = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.24)
    for panel_id, key in enumerate(panels):
        sub = outer[panel_id // 2, panel_id % 2].subgridspec(2, 1, height_ratios=[3.0, 1.05], hspace=0.05)
        ax = fig.add_subplot(sub[0, 0])
        err_ax = fig.add_subplot(sub[1, 0], sharex=ax)
        part = pred[(pred.dataset == key[0]) & (pred.protocol == key[1]) & (pred.model == key[2])].copy()
        part = part.sort_values("sample_index").reset_index(drop=True)
        idx = np.arange(len(part))
        err = np.abs(part.y_pred - part.y_true)
        ax.plot(idx, part.y_true, color="k", lw=1.0, marker=".", ms=3, label="measured")
        ax.plot(idx, part.y_pred, color="crimson", lw=1.0, marker=".", ms=3, label="predicted")
        ax.set_title(f"{key[0]} / {key[1]}")
        ax.set_ylabel("Strength")
        ax.grid(True, ls=":", lw=0.4, alpha=0.8)
        ax.legend(fontsize=7.5, loc="upper left", ncol=2, frameon=True)
        ax.tick_params(labelbottom=False)

        err_ax.bar(idx, err, color="0.72", width=0.82, edgecolor="white", linewidth=0.25)
        err_ax.set_ylabel("|error|")
        err_ax.set_xlabel("Data No.")
        err_ax.grid(True, axis="y", ls=":", lw=0.4, alpha=0.8)
        ymax = float(np.nanmax(err)) if len(err) else 1.0
        err_ax.set_ylim(0, ymax * 1.18 + 1e-9)
    save(
        fig,
        "lit_fig09_prediction_performance_panels",
        manifest,
        "prediction_exports.csv",
        "Rock-joint-paper style measured/predicted panels with separated absolute-error bars.",
    )


def fig_noise_inverse_error_fields(manifest):
    noise = pd.read_csv(RESULTS / "noise_robustness.csv")
    inv = pd.read_csv(RESULTS / "synthetic_inverse_recovery.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.4))
    pivot = noise.pivot_table(index="model", columns="noise_level", values="relative_L2_percent")
    im0 = axes[0].imshow(pivot.to_numpy(float), aspect="auto", cmap="turbo")
    axes[0].set_xticks(np.arange(len(pivot.columns)))
    axes[0].set_xticklabels([f"{100*c:.0f}%" for c in pivot.columns])
    axes[0].set_yticks(np.arange(len(pivot.index)))
    axes[0].set_yticklabels(pivot.index)
    axes[0].set_title("(a) Forward relative L2 error under noise")
    axes[0].set_xlabel("training noise")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.025)

    inv_summary = inv.groupby("noise_level")[["a_relative_error_percent", "b_relative_error_percent"]].mean().T
    im1 = axes[1].imshow(inv_summary.to_numpy(float), aspect="auto", cmap="turbo")
    axes[1].set_xticks(np.arange(len(inv_summary.columns)))
    axes[1].set_xticklabels([f"{100*c:.0f}%" for c in inv_summary.columns])
    axes[1].set_yticks(np.arange(len(inv_summary.index)))
    axes[1].set_yticklabels(["parameter a", "parameter b"])
    axes[1].set_title("(b) Synthetic inverse recovery error")
    axes[1].set_xlabel("observation noise")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.025)
    save(
        fig,
        "lit_fig10_noise_inverse_error_fields",
        manifest,
        "noise_robustness.csv; synthetic_inverse_recovery.csv",
        "Heat-field version of noisy forward and inverse recovery errors.",
    )


def fig_recent_method_r2_rmse_matrix(manifest):
    df = pd.read_csv(RECENT / "recent_methods_summary.csv")
    valid = df[df.protocol != "leave_one_joint_type_out"].copy()
    order = [
        "FTTransformer_2021",
        "RealMLP_2021",
        "PeriodicMLP_2022",
        "KANLite_2024",
        "TabM_2025",
        "MonotonePeakNet_ours",
        "PeriodicMonotone_ours",
        "ResidualPeriodicMonotone_ours",
    ]
    valid["case"] = valid.dataset + "\n" + valid.protocol
    cases = list(dict.fromkeys(valid.case))
    rmse = valid.pivot_table(index="model", columns="case", values="RMSE_mean").reindex(order)
    r2 = valid.pivot_table(index="model", columns="case", values="R2_mean").reindex(order)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2))
    im0 = axes[0].imshow(rmse.to_numpy(float), aspect="auto", cmap="viridis_r")
    axes[0].set_title("(a) RMSE matrix, lower is better")
    im1 = axes[1].imshow(r2.to_numpy(float), aspect="auto", cmap="turbo", vmin=0, vmax=1)
    axes[1].set_title(r"(b) $R^2$ matrix, higher is better")
    for ax in axes:
        ax.set_xticks(np.arange(len(cases)))
        ax.set_xticklabels(cases, rotation=35, ha="right", fontsize=7)
        ax.set_yticks(np.arange(len(order)))
        ax.set_yticklabels([m.replace("_ours", "") for m in order], fontsize=8)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.025)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.025)
    save(
        fig,
        "lit_fig11_recent_method_metric_matrix",
        manifest,
        "recent_methods_summary.csv",
        "Compact color-field comparison of recent methods across valid benchmark protocols.",
    )


def main() -> None:
    style()
    manifest: list[dict[str, str]] = []
    x, y, groups, train_idx, test_idx, model, scaler, sigma_ref = train_final_model()
    fig_solution_error_field(x, y, test_idx, model, scaler, sigma_ref, manifest)
    fig_feature_contour_atlas(manifest)
    fig_curvature_error_atlas(manifest)
    fig_derived_quantities(x, test_idx, model, scaler, sigma_ref, manifest)
    fig_collocation_and_sweep_map(x, y, train_idx, test_idx, manifest)
    fig_barton_bandis_proxy(x, model, scaler, sigma_ref, manifest)
    fig_prediction_performance_grid(manifest)
    fig_noise_inverse_error_fields(manifest)
    fig_recent_method_r2_rmse_matrix(manifest)
    pd.DataFrame(manifest).to_csv(OUT / "figure_manifest.csv", index=False)
    lines = [
        "# Literature-Style Figures",
        "",
        "These figures mimic common SciML/PINN and rock-joint manuscript figure types: contour fields, absolute-error fields, derived-quantity curves, collocation maps, and actual/predicted/error panels.",
        "",
        "Important limitation: the current public rock-joint peak-strength tables do not contain full shear-displacement or dilation time histories, so cyclic shear-stress/dilation validation figures are not generated here.",
        "",
        "## Figure index",
    ]
    for item in manifest:
        lines.append(f"- `{item['figure']}`: {item['note']} Source: `{item['source']}`.")
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {len(manifest)} literature-style figures to {OUT}")


if __name__ == "__main__":
    main()
