"""Train 1D-CNN on raw channel profiles (140 punktow x 2 kanaly).

Siec konwolucyjna na surowych profilach spectral_shift. Inne wejscie
niz baseline/SVR/MLP - tutaj uzywamy bezposrednio surowego sygnalu.

Usage:
    uv run python scripts/07_train_cnn1d.py \
        --zarr data_processed/dataset.zarr \
        --output-dir reports/metrics

Produces:
    cnn1d_per_target.csv, cnn1d_per_condition.csv,
    cnn1d_loco_summary.csv, cnn1d_loco_folds.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xarray as xr
from torch.utils.data import DataLoader, TensorDataset

from fiber_ml.eval.metrics import (
    per_condition_metrics,
    per_target_metrics,
    summarise_loco,
)
from fiber_ml.models.baseline import TARGETS
from fiber_ml.models.splits import loco_cv, replicate_split
from fiber_ml.preprocessing.channels import extract_channels

logger = logging.getLogger(__name__)

# Hiperparametry treningu
EPOCHS = 200
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 30


# ==================== MODEL ====================
class CNN1D(nn.Module):
    """1D CNN na profilu (2 kanaly, 140 punktow)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2),  # 2 wyjscia: T, RH
        )

    def forward(self, x):
        return self.head(self.conv(x))


# ==================== DANE ====================
def build_profile_dataset(zarr_path: str | Path):
    """Wyciaga surowe profile 2x140 i etykiety z Zarr.

    Zwraca:
        X: ndarray (n_pomiarow, 2_kanaly, 140_punktow), float32
        y: ndarray (n_pomiarow, 2), float32 (T, RH)
        meta: DataFrame z kolumnami experiment_id, T, RH, replicate
    """
    logger.info("Loading profiles from %s ...", zarr_path)
    ds = xr.open_zarr(str(zarr_path))
    n = ds.sizes["experiment"]

    X = np.zeros((n, 2, 140), dtype=np.float32)
    y = np.zeros((n, 2), dtype=np.float32)
    meta_rows = []

    channel_names = ds["channel"].values.tolist()

    for i in range(n):
        arr = ds["data"].isel(experiment=i).values
        raw = pd.DataFrame(arr, columns=channel_names)
        paired = extract_channels(raw)

        X[i, 0, :] = paired.ch1["spectral_shift_ghz"].values
        X[i, 1, :] = paired.ch2["spectral_shift_ghz"].values
        y[i, 0] = float(ds["T"].values[i])
        y[i, 1] = float(ds["RH"].values[i])
        meta_rows.append({
            "experiment_id": str(ds["experiment_id"].values[i]),
            "T": float(ds["T"].values[i]),
            "RH": float(ds["RH"].values[i]),
            "replicate": int(ds["replicate"].values[i]),
        })

    meta = pd.DataFrame(meta_rows)
    logger.info("Loaded %d profiles, X shape=%s, y shape=%s",
                n, X.shape, y.shape)
    return X, y, meta


def standardize_input(X_train, X_other):
    """Standaryzacja per-kanal na podstawie X_train.

    Liczy mean i std po wymiarach (samples, points), osobno dla kazdego kanalu.
    Zwraca przeskalowane X_train i X_other.
    """
    mu = X_train.mean(axis=(0, 2), keepdims=True)
    sigma = X_train.std(axis=(0, 2), keepdims=True) + 1e-8
    return (X_train - mu) / sigma, (X_other - mu) / sigma, (mu, sigma)


def standardize_target(y_train, y_other):
    """Standaryzacja targetow - liczona na trainie."""
    mu = y_train.mean(axis=0)
    sigma = y_train.std(axis=0) + 1e-8
    return (y_train - mu) / sigma, (y_other - mu) / sigma, (mu, sigma)


# ==================== TRENING ====================
def train_model(X_train, y_train, X_val, y_val, device: str = "cpu",
                seed: int = 42) -> tuple[nn.Module, tuple]:
    """Trenuje CNN1D z early stopping.

    Zwraca: (model, (y_mu, y_sigma)) - parametry do denormalizacji predykcji.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Standaryzacja
    X_train_n, X_val_n, _ = standardize_input(X_train, X_val)
    y_train_n, y_val_n, (y_mu, y_sigma) = standardize_target(y_train, y_val)

    # Konwersja na tensory
    X_train_t = torch.tensor(X_train_n, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train_n, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val_n, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val_n, dtype=torch.float32).to(device)

    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = CNN1D().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    patience_count = 0

    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        # Walidacja
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                logger.info("  Early stopping at epoch %d (val_loss=%.5f)", epoch, best_val_loss)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, (y_mu, y_sigma), (X_train_n.mean(axis=(0,2), keepdims=True),
                                     X_train_n.std(axis=(0,2), keepdims=True) + 1e-8)


def predict(model: nn.Module, X: np.ndarray, X_stats, y_stats, device: str = "cpu") -> np.ndarray:
    """Predykcja z denormalizacja."""
    mu_x, sigma_x = X_stats
    y_mu, y_sigma = y_stats
    # X_stats sa juz z znormalizowanego trainu, dlatego liczymy je z surowego mean/std
    # Ale dla bezpieczenstwa standaryzujemy X z parametrami trainu
    X_n = (X - mu_x) / sigma_x
    X_t = torch.tensor(X_n, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        pred_n = model(X_t).cpu().numpy()
    return pred_n * y_sigma + y_mu


# ==================== MAIN ====================
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr", type=Path, required=True,
                        help="Path to Zarr dataset.")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("reports/metrics"),
                        help="Where to write the CSV outputs.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    # Wczytaj profile
    X, y, meta = build_profile_dataset(args.zarr)

    # ===================================================================
    # CZĘŚĆ 1: Replicate split
    # ===================================================================
    split = replicate_split(meta)
    logger.info("Replicate split: train=%d, val=%d, test=%d",
                len(split.train), len(split.val), len(split.test))

    X_train, y_train = X[split.train], y[split.train]
    X_val, y_val = X[split.val], y[split.val]
    X_test, y_test = X[split.test], y[split.test]

    logger.info("Training CNN1D on replicate split ...")
    model, y_stats, X_stats = train_model(X_train, y_train, X_val, y_val,
                                          device=device, seed=args.seed)

    y_pred = predict(model, X_test, X_stats, y_stats, device=device)

    pt = per_target_metrics(y_test, y_pred)
    pt["model"] = "cnn1d"
    logger.info("CNN1D test metrics:\n%s", pt.to_string(index=False))

    df_test = meta.iloc[split.test].reset_index(drop=True)
    pc = per_condition_metrics(df_test, y_test, y_pred)
    pc["model"] = "cnn1d"

    pt.to_csv(args.output_dir / "cnn1d_per_target.csv", index=False)
    pc.to_csv(args.output_dir / "cnn1d_per_condition.csv", index=False)
    logger.info("Wrote cnn1d_per_target.csv and cnn1d_per_condition.csv")

    # ===================================================================
    # CZĘŚĆ 2: LOCO CV
    # ===================================================================
    logger.info("Running LOCO CV with CNN1D (35 folds) ...")
    logger.info("Each fold trains a fresh network — this takes a few minutes.")

    fold_results = []
    for fold_idx, ((T, RH), fold_split) in enumerate(loco_cv(meta), start=1):
        X_fold_train, y_fold_train = X[fold_split.train], y[fold_split.train]
        X_fold_test, y_fold_test = X[fold_split.test], y[fold_split.test]

        # Holdoutowy fold nie ma val, wiec uzywamy male wycinki trenu jako val
        # (deterministyczne - bierzemy ostatnie 10%)
        n_val = max(20, len(X_fold_train) // 10)
        X_fv = X_fold_train[-n_val:]
        y_fv = y_fold_train[-n_val:]
        X_ft = X_fold_train[:-n_val]
        y_ft = y_fold_train[:-n_val]

        fold_model, fold_ystats, fold_xstats = train_model(
            X_ft, y_ft, X_fv, y_fv, device=device, seed=args.seed,
        )
        y_fold_pred = predict(fold_model, X_fold_test, fold_xstats, fold_ystats,
                              device=device)

        pt_fold = per_target_metrics(y_fold_test, y_fold_pred)
        fold_results.append({"T": T, "RH": RH, "per_target": pt_fold})

        logger.info("Fold %2d/35  T=%s RH=%s  MAE T=%.3f  MAE RH=%.3f",
                    fold_idx, T, RH,
                    pt_fold[pt_fold["target"] == "T"]["mae"].iloc[0],
                    pt_fold[pt_fold["target"] == "RH"]["mae"].iloc[0])

    folds_long = []
    for fr in fold_results:
        f = fr["per_target"].copy()
        f["T_holdout"] = fr["T"]
        f["RH_holdout"] = fr["RH"]
        folds_long.append(f)
    pd.concat(folds_long, ignore_index=True).to_csv(
        args.output_dir / "cnn1d_loco_folds.csv", index=False,
    )
    summarise_loco(fold_results).to_csv(
        args.output_dir / "cnn1d_loco_summary.csv", index=False,
    )
    logger.info("Wrote cnn1d_loco_folds.csv and cnn1d_loco_summary.csv")
    logger.info("Done.")


if __name__ == "__main__":
    main()