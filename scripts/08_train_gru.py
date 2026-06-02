"""Train GRU on raw channel profiles (140 punktow x 2 kanaly).

Siec rekurencyjna - traktuje 140 punktow wzdluz czujnika jak sekwencje.
Inne wejscie niz CNN: (batch, 140, 2) zamiast (batch, 2, 140).

Usage:
    uv run python scripts/08_train_gru.py \
        --zarr data_processed/dataset.zarr \
        --output-dir reports/metrics

Produces:
    gru_per_target.csv, gru_per_condition.csv,
    gru_loco_summary.csv, gru_loco_folds.csv
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

EPOCHS = 200
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 30


# ==================== MODEL ====================
class GRUNet(nn.Module):
    """GRU bidirectional na sekwencji (140 punktow, 2 kanaly)."""

    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(
            input_size=2,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )
        # Hidden 64 * 2 (bidirectional) = 128
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        # x: (batch, 140, 2)
        out, _ = self.gru(x)
        # Bierzemy ostatni krok sekwencji
        return self.head(out[:, -1, :])


# ==================== DANE ====================
def build_profile_dataset(zarr_path: str | Path):
    """Wyciaga profile w formacie (n, 140, 2) - sekwencja wzdluz czujnika."""
    logger.info("Loading profiles from %s ...", zarr_path)
    ds = xr.open_zarr(str(zarr_path))
    n = ds.sizes["experiment"]

    # Uwaga: dla GRU shape = (n, 140, 2), nie (n, 2, 140)
    X = np.zeros((n, 140, 2), dtype=np.float32)
    y = np.zeros((n, 2), dtype=np.float32)
    meta_rows = []

    channel_names = ds["channel"].values.tolist()

    for i in range(n):
        arr = ds["data"].isel(experiment=i).values
        raw = pd.DataFrame(arr, columns=channel_names)
        paired = extract_channels(raw)

        X[i, :, 0] = paired.ch1["spectral_shift_ghz"].values
        X[i, :, 1] = paired.ch2["spectral_shift_ghz"].values
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
    """Standaryzacja per-kanal. X shape: (n, 140, 2)."""
    # Mean/std po wymiarach (samples, points), osobno per kanal
    mu = X_train.mean(axis=(0, 1), keepdims=True)
    sigma = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    return (X_train - mu) / sigma, (X_other - mu) / sigma, (mu, sigma)


def standardize_target(y_train, y_other):
    mu = y_train.mean(axis=0)
    sigma = y_train.std(axis=0) + 1e-8
    return (y_train - mu) / sigma, (y_other - mu) / sigma, (mu, sigma)


# ==================== TRENING ====================
def train_model(X_train, y_train, X_val, y_val, device: str = "cpu",
                seed: int = 42) -> tuple[nn.Module, tuple, tuple]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train_n, X_val_n, X_stats = standardize_input(X_train, X_val)
    y_train_n, y_val_n, y_stats = standardize_target(y_train, y_val)

    X_train_t = torch.tensor(X_train_n, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train_n, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val_n, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val_n, dtype=torch.float32).to(device)

    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = GRUNet().to(device)
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

    return model, X_stats, y_stats


def predict(model: nn.Module, X: np.ndarray, X_stats, y_stats,
            device: str = "cpu") -> np.ndarray:
    mu_x, sigma_x = X_stats
    y_mu, y_sigma = y_stats
    X_n = (X - mu_x) / sigma_x
    X_t = torch.tensor(X_n, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        pred_n = model(X_t).cpu().numpy()
    return pred_n * y_sigma + y_mu


# ==================== MAIN ====================
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/metrics"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    X, y, meta = build_profile_dataset(args.zarr)

    # ============ Replicate split ============
    split = replicate_split(meta)
    logger.info("Replicate split: train=%d, val=%d, test=%d",
                len(split.train), len(split.val), len(split.test))

    X_train, y_train = X[split.train], y[split.train]
    X_val, y_val = X[split.val], y[split.val]
    X_test, y_test = X[split.test], y[split.test]

    logger.info("Training GRU on replicate split ...")
    model, X_stats, y_stats = train_model(
        X_train, y_train, X_val, y_val, device=device, seed=args.seed,
    )
    y_pred = predict(model, X_test, X_stats, y_stats, device=device)

    pt = per_target_metrics(y_test, y_pred)
    pt["model"] = "gru"
    logger.info("GRU test metrics:\n%s", pt.to_string(index=False))

    df_test = meta.iloc[split.test].reset_index(drop=True)
    pc = per_condition_metrics(df_test, y_test, y_pred)
    pc["model"] = "gru"

    pt.to_csv(args.output_dir / "gru_per_target.csv", index=False)
    pc.to_csv(args.output_dir / "gru_per_condition.csv", index=False)
    logger.info("Wrote gru_per_target.csv and gru_per_condition.csv")

    # ============ LOCO CV ============
    logger.info("Running LOCO CV with GRU (35 folds) ...")

    fold_results = []
    for fold_idx, ((T, RH), fold_split) in enumerate(loco_cv(meta), start=1):
        X_fold_train, y_fold_train = X[fold_split.train], y[fold_split.train]
        X_fold_test, y_fold_test = X[fold_split.test], y[fold_split.test]

        n_val = max(20, len(X_fold_train) // 10)
        X_fv = X_fold_train[-n_val:]
        y_fv = y_fold_train[-n_val:]
        X_ft = X_fold_train[:-n_val]
        y_ft = y_fold_train[:-n_val]

        fold_model, fold_xstats, fold_ystats = train_model(
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
        args.output_dir / "gru_loco_folds.csv", index=False,
    )
    summarise_loco(fold_results).to_csv(
        args.output_dir / "gru_loco_summary.csv", index=False,
    )
    logger.info("Wrote gru_loco_folds.csv and gru_loco_summary.csv")
    logger.info("Done.")


if __name__ == "__main__":
    main()