from __future__ import annotations

from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from benchmark_recent_methods_external import (
    DEVICE,
    FTTransformerNet,
    PeriodicMLPNet,
    PeriodicMonotonePeakNet,
    ResidualPeriodicMonotonePeakNet,
    MonotonePeakNet,
    fit_torch_model,
)


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "paper_artifacts" / "experiment_results"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metrics(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    pred = np.clip(np.asarray(pred, dtype=np.float32), 0.0, None)
    return {
        "R2": float(r2_score(y_true, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, pred))),
        "MAE": float(mean_absolute_error(y_true, pred)),
        "MAAPE": float(np.mean(np.arctan(np.abs((y_true - pred) / np.maximum(np.abs(y_true), 1e-6))))),
        "relative_L2_percent": float(np.linalg.norm(pred - y_true) / max(np.linalg.norm(y_true), 1e-8) * 100.0),
    }


class PlainMLP(nn.Module):
    def __init__(self, n_features: int, hidden: int = 96) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def load_rockmb() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(ROOT / "rockmb_2025_dataset.csv")
    x = df[["normal_stress_mpa", "ucs_mpa", "youngs_modulus_gpa", "jrc", "specimen_length_mm"]].to_numpy(dtype=np.float32)
    y = df["peak_shear_stress_mpa"].to_numpy(dtype=np.float32)
    groups = df["reference"].to_numpy()
    return x, y, groups


def load_g5() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(ROOT / "datasets" / "g5pf6k9n2w_v1_parsed" / "shear_curve_peaks.csv")
    profile = pd.get_dummies(df["joint_profile"], prefix="profile")
    x = np.column_stack(
        [
            df["normal_stress_mpa"].to_numpy(dtype=np.float32),
            df["immersion_days"].to_numpy(dtype=np.float32),
            profile.to_numpy(dtype=np.float32),
        ]
    )
    y = df["peak_shear_stress_mpa"].to_numpy(dtype=np.float32)
    return x, y, df["joint_profile"].to_numpy(), df["immersion_days"].to_numpy()


def fit_plain(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    seed: int,
    log_sigma: bool,
    asinh_target: bool,
    epochs: int = 220,
) -> tuple[np.ndarray, dict[str, float]]:
    set_seed(seed)
    xtr = x_train.copy()
    xte = x_test.copy()
    if log_sigma:
        xtr[:, 0] = np.log1p(np.maximum(xtr[:, 0], 0.0))
        xte[:, 0] = np.log1p(np.maximum(xte[:, 0], 0.0))
    scaler = StandardScaler()
    xtr = scaler.fit_transform(xtr).astype("float32")
    xte = scaler.transform(xte).astype("float32")
    target = np.arcsinh(y_train).astype("float32") if asinh_target else y_train.astype("float32")
    model = PlainMLP(xtr.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    xt = torch.tensor(xtr, device=DEVICE)
    yt = torch.tensor(target, device=DEVICE)
    grad_norms = []
    losses = []
    for _ in range(epochs):
        pred = model(xt)
        loss = F.huber_loss(pred, yt, delta=0.2) if asinh_target else F.mse_loss(pred, yt)
        opt.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        losses.append(float(loss.detach().cpu()))
        grad_norms.append(float(grad_norm.detach().cpu() if torch.is_tensor(grad_norm) else grad_norm))
    with torch.no_grad():
        pred = model(torch.tensor(xte, device=DEVICE)).cpu().numpy()
    pred = np.sinh(pred) if asinh_target else pred
    diagnostics = {
        "final_train_loss": losses[-1],
        "median_grad_norm": float(np.median(grad_norms)),
        "max_grad_norm": float(np.max(grad_norms)),
        "loss_reduction_ratio": float(losses[0] / max(losses[-1], 1e-12)),
    }
    return np.clip(pred, 0.0, None), diagnostics


def fit_monotone_model(
    model_cls: type[nn.Module],
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int,
    epochs: int = 260,
) -> tuple[nn.Module, StandardScaler, float]:
    set_seed(seed)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_train).astype("float32")
    sigma = torch.tensor(x_train[:, :1], dtype=torch.float32, device=DEVICE)
    ctx = torch.tensor(x_scaled[:, 1:], dtype=torch.float32, device=DEVICE)
    y = torch.tensor(y_train, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    sigma_ref = float(np.median(np.maximum(x_train[:, 0], 1e-6)))
    model = model_cls(context_dim=ctx.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    for _ in range(epochs):
        pred = model(ctx, sigma, sigma_ref)
        loss = F.huber_loss(torch.asinh(pred), torch.asinh(y), delta=0.2)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model, scaler, sigma_ref


def predict_monotone(model: nn.Module, scaler: StandardScaler, sigma_ref: float, x: np.ndarray) -> np.ndarray:
    x_scaled = scaler.transform(x).astype("float32")
    sigma = torch.tensor(x[:, :1], dtype=torch.float32, device=DEVICE)
    ctx = torch.tensor(x_scaled[:, 1:], dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        return np.clip(model(ctx, sigma, sigma_ref).squeeze(1).cpu().numpy(), 0.0, None)


def fixed_context_violation(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_ref: np.ndarray,
    *,
    seed: int,
) -> dict[str, float | str]:
    if model_name == "MonotonePeakNet":
        model, scaler, sigma_ref = fit_monotone_model(MonotonePeakNet, x_train, y_train, seed=seed)
        predictor = lambda z: predict_monotone(model, scaler, sigma_ref, z)
        path_predictor = lambda context, sigmas: predict_monotone_path(model, scaler, sigma_ref, context, sigmas)
    elif model_name == "PeriodicMonotone":
        model, scaler, sigma_ref = fit_monotone_model(PeriodicMonotonePeakNet, x_train, y_train, seed=seed)
        predictor = lambda z: predict_monotone(model, scaler, sigma_ref, z)
        path_predictor = lambda context, sigmas: predict_monotone_path(model, scaler, sigma_ref, context, sigmas)
    elif model_name == "ResidualPeriodicMonotone":
        model, scaler, sigma_ref = fit_monotone_model(ResidualPeriodicMonotonePeakNet, x_train, y_train, seed=seed)
        predictor = lambda z: predict_monotone(model, scaler, sigma_ref, z)
        path_predictor = lambda context, sigmas: predict_monotone_path(model, scaler, sigma_ref, context, sigmas)
    else:
        raise ValueError(model_name)

    sigma_min = 0.0
    sigma_max = float(np.percentile(x_train[:, 0], 97.5))
    total_pairs = 0
    violations = 0
    negative = 0
    boundary_errors = []
    for i in range(min(40, len(x_ref))):
        base = np.repeat(x_ref[i : i + 1], 64, axis=0)
        base[:, 0] = np.linspace(sigma_min, sigma_max, 64)
        pred = path_predictor(x_ref[i : i + 1], base[:, 0])
        diffs = np.diff(pred)
        violations += int(np.sum(diffs < -1e-5))
        total_pairs += len(diffs)
        negative += int(np.sum(pred < -1e-6))
        zero = base[:1].copy()
        zero[:, 0] = 0.0
        boundary_errors.append(float(abs(predictor(zero)[0])))
    return {
        "model": model_name,
        "fixed_context_monotonic_violation_rate_percent": violations / max(total_pairs, 1) * 100.0,
        "negative_prediction_rate_percent": negative / max(64 * min(40, len(x_ref)), 1) * 100.0,
        "mean_tau0_boundary_error": float(np.mean(boundary_errors)),
    }


def predict_monotone_path(model: nn.Module, scaler: StandardScaler, sigma_ref: float, context_row: np.ndarray, sigma_values: np.ndarray) -> np.ndarray:
    base = np.repeat(context_row.copy(), len(sigma_values), axis=0)
    base[:, 0] = sigma_values
    x_scaled = scaler.transform(base).astype("float32")
    ctx = torch.tensor(x_scaled[:1, 1:], dtype=torch.float32, device=DEVICE)
    sigma_path = torch.tensor(sigma_values[None, :], dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        return np.clip(model.forward_path(ctx, sigma_path, sigma_ref).squeeze(0).cpu().numpy(), 0.0, None)


def run_transform_ablation() -> None:
    x, y, _ = load_rockmb()
    train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=0.30, random_state=42)
    rows = []
    configs = [
        ("raw_sigma_raw_tau_plain_mlp", False, False),
        ("log_sigma_raw_tau_plain_mlp", True, False),
        ("raw_sigma_asinh_tau_plain_mlp", False, True),
        ("log_sigma_asinh_tau_plain_mlp", True, True),
    ]
    for name, log_sigma, asinh_target in configs:
        pred, diag = fit_plain(x[train_idx], y[train_idx], x[test_idx], seed=11, log_sigma=log_sigma, asinh_target=asinh_target)
        rows.append({"experiment": name, **diag, **metrics(y[test_idx], pred)})
    model, scaler, sigma_ref = fit_monotone_model(ResidualPeriodicMonotonePeakNet, x[train_idx], y[train_idx], seed=11)
    pred = predict_monotone(model, scaler, sigma_ref, x[test_idx])
    rows.append({"experiment": "stress_path_periodic_residual_monotone", "final_train_loss": np.nan, "median_grad_norm": np.nan, "max_grad_norm": np.nan, "loss_reduction_ratio": np.nan, **metrics(y[test_idx], pred)})
    pd.DataFrame(rows).to_csv(OUT / "nondimensional_transform_ablation.csv", index=False)


def run_physics_validity() -> None:
    rows = []
    x, y, groups = load_rockmb()
    train_idx, test_idx = next(GroupKFold(n_splits=5).split(x, y, groups))
    for name in ["MonotonePeakNet", "PeriodicMonotone", "ResidualPeriodicMonotone"]:
        rows.append({"dataset": "rockmb_2025_group_fold1", **fixed_context_violation(name, x[train_idx], y[train_idx], x[test_idx], seed=101)})
    x, y, profile, _ = load_g5()
    train_idx = np.where(profile != "JC")[0]
    test_idx = np.where(profile == "JC")[0]
    for name in ["MonotonePeakNet", "PeriodicMonotone", "ResidualPeriodicMonotone"]:
        rows.append({"dataset": "g5_leave_JC_profile", **fixed_context_violation(name, x[train_idx], y[train_idx], x[test_idx], seed=202)})
    pd.DataFrame(rows).to_csv(OUT / "physics_validity_fixed_context_sweeps.csv", index=False)


def run_noise_robustness() -> None:
    x, y, _ = load_rockmb()
    train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=0.30, random_state=42)
    rows = []
    for noise in [0.0, 0.05, 0.10, 0.20]:
        set_seed(500 + int(noise * 100))
        x_train = x[train_idx].copy()
        y_train = y[train_idx].copy()
        x_scale = np.maximum(np.std(x_train, axis=0, keepdims=True), 1e-6)
        y_scale = max(float(np.std(y_train)), 1e-6)
        x_noisy = x_train + np.random.normal(0.0, noise, size=x_train.shape).astype("float32") * x_scale
        y_noisy = y_train + np.random.normal(0.0, noise, size=y_train.shape).astype("float32") * y_scale
        x_noisy[:, 0] = np.clip(x_noisy[:, 0], 0.0, None)
        y_noisy = np.clip(y_noisy, 0.0, None)
        for model_name, model_cls in [
            ("PeriodicMonotone", PeriodicMonotonePeakNet),
            ("ResidualPeriodicMonotone", ResidualPeriodicMonotonePeakNet),
        ]:
            model, scaler, sigma_ref = fit_monotone_model(model_cls, x_noisy, y_noisy, seed=600 + int(noise * 100))
            pred = predict_monotone(model, scaler, sigma_ref, x[test_idx])
            rows.append({"dataset": "rockmb_2025_paper_70_30", "model": model_name, "noise_level": noise, **metrics(y[test_idx], pred)})
    pd.DataFrame(rows).to_csv(OUT / "noise_robustness.csv", index=False)


def run_inverse_recovery_synthetic() -> None:
    rows = []
    rng = np.random.default_rng(42)
    sigmas = np.linspace(0.5, 12.0, 16)
    for noise in [0.0, 0.05, 0.10, 0.20]:
        for case in range(20):
            a_true = rng.uniform(0.8, 2.5)
            b_true = rng.uniform(0.15, 0.65)
            tau = a_true * np.log1p(sigmas / b_true)
            tau_noisy = tau + rng.normal(0.0, noise * np.std(tau), size=tau.shape)
            a = torch.tensor(1.0, device=DEVICE, requires_grad=True)
            log_b = torch.tensor(np.log(0.3), device=DEVICE, requires_grad=True)
            opt = torch.optim.Adam([a, log_b], lr=0.05)
            s_t = torch.tensor(sigmas, dtype=torch.float32, device=DEVICE)
            y_t = torch.tensor(tau_noisy, dtype=torch.float32, device=DEVICE)
            for _ in range(500):
                pred = F.softplus(a) * torch.log1p(s_t / F.softplus(log_b))
                loss = F.mse_loss(pred, y_t)
                opt.zero_grad()
                loss.backward()
                opt.step()
            a_hat = float(F.softplus(a).detach().cpu())
            b_hat = float(F.softplus(log_b).detach().cpu())
            rows.append(
                {
                    "synthetic_case": case,
                    "noise_level": noise,
                    "a_true": a_true,
                    "a_hat": a_hat,
                    "b_true": b_true,
                    "b_hat": b_hat,
                    "a_relative_error_percent": abs(a_hat - a_true) / a_true * 100.0,
                    "b_relative_error_percent": abs(b_hat - b_true) / b_true * 100.0,
                }
            )
    pd.DataFrame(rows).to_csv(OUT / "synthetic_inverse_recovery.csv", index=False)


def run_prediction_exports() -> None:
    rows = []
    x, y, groups = load_rockmb()
    train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=0.30, random_state=42)
    model, scaler, sigma_ref = fit_monotone_model(ResidualPeriodicMonotonePeakNet, x[train_idx], y[train_idx], seed=42)
    pred = predict_monotone(model, scaler, sigma_ref, x[test_idx])
    for idx, yt, yp in zip(test_idx, y[test_idx], pred):
        rows.append({"dataset": "rockmb_2025", "protocol": "paper_70_30", "sample_index": int(idx), "y_true": float(yt), "y_pred": float(yp), "model": "ResidualPeriodicMonotone"})
    train_idx, test_idx = next(GroupKFold(n_splits=5).split(x, y, groups))
    model, scaler, sigma_ref = fit_monotone_model(MonotonePeakNet, x[train_idx], y[train_idx], seed=101)
    pred = predict_monotone(model, scaler, sigma_ref, x[test_idx])
    for idx, yt, yp in zip(test_idx, y[test_idx], pred):
        rows.append({"dataset": "rockmb_2025", "protocol": "group_5fold_fold1", "sample_index": int(idx), "y_true": float(yt), "y_pred": float(yp), "model": "MonotonePeakNet"})

    x, y, profile, immersion = load_g5()
    train_idx = np.where(profile != "JC")[0]
    test_idx = np.where(profile == "JC")[0]
    model, scaler, sigma_ref = fit_monotone_model(ResidualPeriodicMonotonePeakNet, x[train_idx], y[train_idx], seed=203)
    pred = predict_monotone(model, scaler, sigma_ref, x[test_idx])
    for idx, yt, yp in zip(test_idx, y[test_idx], pred):
        rows.append({"dataset": "g5pf6k9n2w", "protocol": "leave_one_profile_JC", "sample_index": int(idx), "y_true": float(yt), "y_pred": float(yp), "model": "ResidualPeriodicMonotone"})
    train_idx = np.where(immersion != 360)[0]
    test_idx = np.where(immersion == 360)[0]
    model, scaler, sigma_ref = fit_monotone_model(PeriodicMonotonePeakNet, x[train_idx], y[train_idx], seed=304)
    pred = predict_monotone(model, scaler, sigma_ref, x[test_idx])
    for idx, yt, yp in zip(test_idx, y[test_idx], pred):
        rows.append({"dataset": "g5pf6k9n2w", "protocol": "leave_one_immersion_360", "sample_index": int(idx), "y_true": float(yt), "y_pred": float(yp), "model": "PeriodicMonotone"})
    pd.DataFrame(rows).to_csv(OUT / "prediction_exports.csv", index=False)


def run_sensitivity_exports() -> None:
    rows = []
    x, y, _ = load_rockmb()
    train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=0.30, random_state=42)
    model, scaler, sigma_ref = fit_monotone_model(ResidualPeriodicMonotonePeakNet, x[train_idx], y[train_idx], seed=42)
    feature_names = ["normal_stress_mpa", "ucs_mpa", "youngs_modulus_gpa", "jrc", "specimen_length_mm"]
    base_rows = x[test_idx][: min(40, len(test_idx))]
    for fi, name in enumerate(feature_names):
        vals = np.linspace(np.percentile(x[:, fi], 5), np.percentile(x[:, fi], 95), 50)
        for sample_id, base in enumerate(base_rows):
            sweep = np.repeat(base[None, :], len(vals), axis=0)
            sweep[:, fi] = vals
            pred = predict_monotone(model, scaler, sigma_ref, sweep)
            for v, p in zip(vals, pred):
                rows.append({"dataset": "rockmb_2025", "model": "ResidualPeriodicMonotone", "feature": name, "sample_id": sample_id, "feature_value": float(v), "prediction": float(p)})
    pd.DataFrame(rows).to_csv(OUT / "feature_sensitivity_sweeps.csv", index=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    run_transform_ablation()
    run_physics_validity()
    run_noise_robustness()
    run_inverse_recovery_synthetic()
    run_prediction_exports()
    run_sensitivity_exports()
    (OUT / "run_config.json").write_text(json.dumps({"device": str(DEVICE), "seed_policy": "fixed per experiment"}, indent=2), encoding="utf-8")
    print(f"Saved physics-grade experiment results to {OUT}")


if __name__ == "__main__":
    main()
