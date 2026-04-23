from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut, train_test_split


ROOT = Path(__file__).resolve().parent
PEAKS_PATH = ROOT / "datasets" / "g5pf6k9n2w_v1_parsed" / "shear_curve_peaks.csv"
OUTPUT_DIR = ROOT / "results_catboost"


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    pred = np.clip(np.asarray(y_pred, dtype=np.float32), 0.0, None)
    return {
        "R2": float(r2_score(y_true, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, pred))),
        "MAE": float(mean_absolute_error(y_true, pred)),
        "MAAPE": float(np.mean(np.arctan(np.abs((y_true - pred) / np.maximum(np.abs(y_true), 1e-6))))),
    }


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(PEAKS_PATH)
    profile_onehot = pd.get_dummies(df["joint_profile"], prefix="profile")
    x = np.column_stack(
        [
            df["normal_stress_mpa"].to_numpy(dtype=np.float32),
            df["immersion_days"].to_numpy(dtype=np.float32),
            profile_onehot.to_numpy(dtype=np.float32),
        ]
    )
    y = df["peak_shear_stress_mpa"].to_numpy(dtype=np.float32)
    profile_groups = df["joint_profile"].to_numpy()
    immersion_groups = df["immersion_days"].to_numpy()
    return x, y, profile_groups, immersion_groups


def fit_catboost(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, seed: int) -> np.ndarray:
    model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        depth=4,
        iterations=300,
        learning_rate=0.035,
        l2_leaf_reg=3.0,
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(x_train, y_train)
    return model.predict(x_test)


def run() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    x, y, profile_groups, immersion_groups = load_data()
    rows: list[dict[str, object]] = []

    print("[CatBoost G5] random_75_25")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
    pred = fit_catboost(x_train, y_train, x_test, seed=42)
    rows.append({"dataset": "g5pf6k9n2w", "protocol": "random_75_25", "model": "CatBoost"} | metrics(y_test, pred))

    logo = LeaveOneGroupOut()
    for fold, (train_idx, test_idx) in enumerate(logo.split(x, y, profile_groups), start=1):
        heldout = str(profile_groups[test_idx][0])
        print(f"[CatBoost G5] leave_one_profile_out fold={fold} heldout={heldout}")
        pred = fit_catboost(x[train_idx], y[train_idx], x[test_idx], seed=100 + fold)
        rows.append(
            {
                "dataset": "g5pf6k9n2w",
                "protocol": "leave_one_profile_out",
                "fold": fold,
                "heldout_group": heldout,
                "model": "CatBoost",
            }
            | metrics(y[test_idx], pred)
        )

    for fold, (train_idx, test_idx) in enumerate(logo.split(x, y, immersion_groups), start=1):
        heldout = int(immersion_groups[test_idx][0])
        print(f"[CatBoost G5] leave_one_immersion_out fold={fold} heldout={heldout}")
        pred = fit_catboost(x[train_idx], y[train_idx], x[test_idx], seed=200 + fold)
        rows.append(
            {
                "dataset": "g5pf6k9n2w",
                "protocol": "leave_one_immersion_out",
                "fold": fold,
                "heldout_group": heldout,
                "model": "CatBoost",
            }
            | metrics(y[test_idx], pred)
        )

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "g5_catboost_results.csv", index=False)
    numeric_cols = ["R2", "RMSE", "MAE", "MAAPE"]
    summary = out.groupby(["dataset", "protocol", "model"])[numeric_cols].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join([str(c) for c in col if c != ""]).strip("_") for col in summary.columns.to_flat_index()]
    summary.to_csv(OUTPUT_DIR / "g5_catboost_summary.csv", index=False)
    (OUTPUT_DIR / "run_config.json").write_text(
        json.dumps({"dataset": str(PEAKS_PATH), "model": "CatBoostRegressor", "protocols": 3}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved CatBoost G5 results to {OUTPUT_DIR}")


if __name__ == "__main__":
    run()
