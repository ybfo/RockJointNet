from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "RockJointNet_local_curve_experiments"
DATA = OUT / "standardized_data"
FIGS = OUT / "figures"
TABLES = OUT / "tables"


def bb_like_shear_curve(x: np.ndarray, tau_peak: float, tau_res: float, x_peak: float) -> np.ndarray:
    x_safe = np.maximum(x, 0.0)
    x_peak = max(float(x_peak), 1e-6)
    rise = tau_peak * (1.0 - np.exp(-3.0 * x_safe / x_peak)) / (1.0 - np.exp(-3.0))
    decay_len = max(0.25 * (float(np.nanmax(x_safe)) - x_peak), 0.8)
    post = tau_res + (tau_peak - tau_res) * np.exp(-(x_safe - x_peak) / decay_len)
    return np.where(x_safe <= x_peak, rise, post)


def bb_like_dilation_curve(x: np.ndarray, dilation: np.ndarray) -> np.ndarray:
    x_safe = np.maximum(x, 0.0)
    d0 = float(np.nanpercentile(dilation, 5))
    dmax = float(np.nanpercentile(dilation, 95))
    scale = max(float(np.nanpercentile(x_safe, 70)), 1e-6)
    return d0 + (dmax - d0) * (1.0 - np.exp(-x_safe / scale))


class CurveSurrogate(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 2),
        )

    def forward(self, sigma_norm: torch.Tensor, disp_norm: torch.Tensor) -> torch.Tensor:
        x = torch.stack([sigma_norm, disp_norm], dim=-1)
        raw = self.net(x)
        # Hard zero-displacement boundary. The shear branch can soften because the
        # learned amplitude is displacement dependent; the boundary remains exact.
        factor = torch.clamp(disp_norm, min=0.0).unsqueeze(-1)
        return factor * raw


def metrics(y: np.ndarray, yhat: np.ndarray) -> dict[str, float]:
    err = yhat - y
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return {"rmse": rmse, "mae": mae, "r2": r2}


def train_ours(curves: pd.DataFrame, epochs: int = 1800) -> tuple[CurveSurrogate, dict[str, float]]:
    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clean = curves[curves["shear_displacement_mm"] >= 0].copy()
    max_disp = float(clean["shear_displacement_mm"].max())
    max_tau = float(clean["calibrated_shear_stress_mpa"].max())
    max_dil = float(clean["dilation_proxy_mm"].abs().max())
    clean["sigma_norm"] = clean["normal_stress_mpa"] / clean["normal_stress_mpa"].max()
    clean["disp_norm"] = clean["shear_displacement_mm"] / max_disp
    clean["tau_norm"] = clean["calibrated_shear_stress_mpa"] / max_tau
    clean["dilation_norm"] = clean["dilation_proxy_mm"] / max_dil

    # Deterministic thinning keeps training fast while preserving every stress path.
    train = clean.groupby("curve_id", group_keys=False).apply(lambda g: g.iloc[::4]).reset_index(drop=True)
    x_sigma = torch.tensor(train["sigma_norm"].to_numpy(np.float32), device=device)
    x_disp = torch.tensor(train["disp_norm"].to_numpy(np.float32), device=device)
    y = torch.tensor(train[["tau_norm", "dilation_norm"]].to_numpy(np.float32), device=device)
    loader = DataLoader(TensorDataset(x_sigma, x_disp, y), batch_size=512, shuffle=True)

    model = CurveSurrogate().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    last = {}
    for epoch in range(epochs):
        total = 0.0
        for sigma_b, disp_b, y_b in loader:
            pred = model(sigma_b, disp_b)
            data_loss = torch.nn.functional.huber_loss(pred, y_b, delta=0.05)
            zero = torch.zeros_like(sigma_b)
            boundary_loss = torch.mean(model(sigma_b, zero) ** 2)
            opt.zero_grad()
            loss = data_loss + 0.25 * boundary_loss
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu())
        scheduler.step()
        if epoch % 300 == 0 or epoch == epochs - 1:
            last = {"epoch": epoch + 1, "loss": total / max(len(loader), 1), "device": str(device)}
            print(f"[GeoSPIN curve] epoch={last['epoch']:04d} loss={last['loss']:.6f} device={device}")

    model.scale_info = {"max_disp": max_disp, "max_tau": max_tau, "max_dil": max_dil}  # type: ignore[attr-defined]
    return model, last


def predict_ours(model: CurveSurrogate, curves: pd.DataFrame) -> pd.DataFrame:
    device = next(model.parameters()).device
    clean = curves[curves["shear_displacement_mm"] >= 0].copy()
    info = model.scale_info  # type: ignore[attr-defined]
    sigma_norm = torch.tensor((clean["normal_stress_mpa"] / clean["normal_stress_mpa"].max()).to_numpy(np.float32), device=device)
    disp_norm = torch.tensor((clean["shear_displacement_mm"] / info["max_disp"]).to_numpy(np.float32), device=device)
    with torch.no_grad():
        pred = model(sigma_norm, disp_norm).cpu().numpy()
    clean["ours_shear_stress_mpa"] = pred[:, 0] * info["max_tau"]
    clean["ours_dilation_proxy_mm"] = pred[:, 1] * info["max_dil"]
    return clean


def add_bb_predictions(pred: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for normal, group in pred.groupby("normal_stress_mpa"):
        group = group.sort_values("shear_displacement_mm").copy()
        x = group["shear_displacement_mm"].to_numpy(float)
        tau = group["calibrated_shear_stress_mpa"].to_numpy(float)
        dil = group["dilation_proxy_mm"].to_numpy(float)
        peak_idx = int(np.nanargmax(tau))
        tau_res = float(np.nanmedian(tau[-max(25, len(tau) // 10) :]))
        group["bb_shear_stress_mpa"] = bb_like_shear_curve(x, float(tau[peak_idx]), tau_res, float(x[peak_idx]))
        group["bb_dilation_proxy_mm"] = bb_like_dilation_curve(x, dil)
        frames.append(group)
    return pd.concat(frames, ignore_index=True)


def write_metrics(pred: pd.DataFrame, train_log: dict[str, float]) -> None:
    rows = []
    for target, ours_col, bb_col in [
        ("shear_stress_mpa", "ours_shear_stress_mpa", "bb_shear_stress_mpa"),
        ("dilation_proxy_mm", "ours_dilation_proxy_mm", "bb_dilation_proxy_mm"),
    ]:
        y_col = "calibrated_shear_stress_mpa" if target == "shear_stress_mpa" else "dilation_proxy_mm"
        for method, col in [("GeoSPIN", ours_col), ("C-BB", bb_col)]:
            for normal, group in pred.groupby("normal_stress_mpa"):
                row = {"target": target, "method": method, "normal_stress_mpa": normal}
                row.update(metrics(group[y_col].to_numpy(float), group[col].to_numpy(float)))
                rows.append(row)
    out = pd.DataFrame(rows)
    out["train_epoch"] = train_log.get("epoch", np.nan)
    out["train_loss"] = train_log.get("loss", np.nan)
    out["device"] = train_log.get("device", "unknown")
    out.to_csv(TABLES / "ours_vs_bb_full_curve_metrics.csv", index=False)


def plot_three_way(pred: pd.DataFrame) -> None:
    normals = sorted(pred["normal_stress_mpa"].unique())
    fig, axes = plt.subplots(2, len(normals), figsize=(12.0, 6.2), sharex=True)
    if len(normals) == 1:
        axes = axes.reshape(2, 1)
    styles = {
        "Experiment": {"color": "#222222", "lw": 2.1, "ls": "-", "zorder": 3},
        "GeoSPIN": {"color": "#D62728", "lw": 2.0, "ls": "-", "zorder": 4},
        "BB": {"color": "#6E6E6E", "lw": 1.7, "ls": "--", "zorder": 2},
    }
    for col, normal in enumerate(normals):
        group = pred[pred["normal_stress_mpa"] == normal].sort_values("shear_displacement_mm")
        x = group["shear_displacement_mm"].to_numpy(float)
        top = axes[0, col]
        bottom = axes[1, col]
        top.plot(x, group["calibrated_shear_stress_mpa"], label="Experiment", **styles["Experiment"])
        top.plot(x, group["ours_shear_stress_mpa"], label="GeoSPIN", **styles["GeoSPIN"])
        top.plot(x, group["bb_shear_stress_mpa"], label="C-BB", **styles["BB"])
        bottom.plot(x, group["dilation_proxy_mm"], label="Experiment", **styles["Experiment"])
        bottom.plot(x, group["ours_dilation_proxy_mm"], label="GeoSPIN", **styles["GeoSPIN"])
        bottom.plot(x, group["bb_dilation_proxy_mm"], label="C-BB", **styles["BB"])
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
    fig.savefig(FIGS / "ours_vs_bb_full_shear_dilation.png", dpi=380)
    fig.savefig(FIGS / "ours_vs_bb_full_shear_dilation.pdf")
    plt.close(fig)


def main() -> None:
    curves = pd.read_csv(DATA / "local_direct_shear_full_curves.csv")
    model, train_log = train_ours(curves)
    pred = add_bb_predictions(predict_ours(model, curves))
    pred.to_csv(DATA / "ours_vs_bb_full_curve_predictions.csv", index=False)
    write_metrics(pred, train_log)
    plot_three_way(pred)
    print(f"Saved ours-vs-BB comparison artifacts to {OUT}")


if __name__ == "__main__":
    main()
