"""EDA plots for the static dataset.

Five canonical plots that should be re-generated whenever the Zarr dataset
changes:

1. State coverage — scatter of (T, RH) conditions, one point per condition,
   colored by replicate count. Quickly reveals missing or oversampled cells.
2. Channel profiles — spectral shift vs sensor position for a few corner
   conditions, both channels side by side.
3. Response surface — mean spectral shift vs (T, RH), one heatmap per channel.
4. Replicate variance — std of mean shift across the 20 replicates of each
   condition; high values flag noisy or drifting states.
5. Quality distribution — histogram of ``spectral_shift_quality`` per
   channel with the dropout threshold marked.

All plots accept either an ``xarray.Dataset`` (loaded from Zarr) or the
intermediate aggregated DataFrame and return a matplotlib Figure so the
caller can save / embed as needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fiber_ml.preprocessing.channels import (
    CH1_REF_T_RANGE_M,
    CH2_PI_TRH_RANGE_M,
    DROPOUT_QUALITY_THRESHOLD,
    extract_channels,
)

if TYPE_CHECKING:
    import xarray as xr


def _channels_from_zarr_row(ds: "xr.Dataset", i: int) -> pd.DataFrame:
    """Reconstruct a 5-column DataFrame from one Zarr experiment row."""
    arr = ds["data"].isel(experiment=i).values
    return pd.DataFrame(arr, columns=ds["channel"].values.tolist())


def aggregate_from_zarr(ds: "xr.Dataset") -> pd.DataFrame:
    """Build per-experiment aggregated features from the Zarr dataset.

    Output columns: ``experiment_id``, ``T``, ``RH``, ``replicate``,
    plus per-channel ``{ch}_mean``, ``{ch}_std``, ``{ch}_q_mean``.
    One row per experiment (700 rows for the full dataset).
    """
    rows: list[dict[str, object]] = []
    for i in range(ds.sizes["experiment"]):
        raw = _channels_from_zarr_row(ds, i)
        paired = extract_channels(raw)
        rows.append(
            {
                "experiment_id": str(ds["experiment_id"].values[i]),
                "T": int(ds["T"].values[i]),
                "RH": int(ds["RH"].values[i]),
                "replicate": int(ds["replicate"].values[i]),
                "ch1_mean": paired.ch1["spectral_shift_ghz"].mean(),
                "ch1_std": paired.ch1["spectral_shift_ghz"].std(),
                "ch1_q_mean": paired.ch1["spectral_shift_quality"].mean(),
                "ch2_mean": paired.ch2["spectral_shift_ghz"].mean(),
                "ch2_std": paired.ch2["spectral_shift_ghz"].std(),
                "ch2_q_mean": paired.ch2["spectral_shift_quality"].mean(),
            }
        )
    return pd.DataFrame(rows)


def plot_state_coverage(agg: pd.DataFrame) -> plt.Figure:
    """Scatter of (T, RH) conditions colored by replicate count."""
    counts = (
        agg.groupby(["T", "RH"]).size().reset_index(name="n_files")
    )
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sc = ax.scatter(
        counts["T"], counts["RH"], c=counts["n_files"],
        s=140, cmap="viridis", edgecolor="black", linewidth=0.5,
    )
    plt.colorbar(sc, ax=ax, label="files per condition (expected: 20)")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Relative humidity (%)")
    ax.set_title(
        f"State coverage — {len(counts)} conditions, "
        f"{int(counts['n_files'].sum())} files"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_channel_profiles(
    ds: "xr.Dataset", state_indices: list[int] | None = None
) -> plt.Figure:
    """Spectral shift along sensor position for a handful of corner states.

    If ``state_indices`` is None, picks the four (T, RH) corners of the grid.
    """
    if state_indices is None:
        df = pd.DataFrame(
            {
                "T": ds["T"].values,
                "RH": ds["RH"].values,
                "rep": ds["replicate"].values,
            }
        )
        # Pick replicate-1 rows at the four corners of the (T, RH) grid
        t_lo, t_hi = df["T"].min(), df["T"].max()
        rh_lo, rh_hi = df["RH"].min(), df["RH"].max()
        corners = [(t_lo, rh_lo), (t_lo, rh_hi), (t_hi, rh_lo), (t_hi, rh_hi)]
        state_indices = []
        for tc, rc in corners:
            sub = df[(df["T"] == tc) & (df["RH"] == rc) & (df["rep"] == 1)]
            if not sub.empty:
                state_indices.append(int(sub.index[0]))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharex=True)
    for i in state_indices:
        raw = _channels_from_zarr_row(ds, i)
        paired = extract_channels(raw)
        T = int(ds["T"].values[i])
        RH = int(ds["RH"].values[i])
        label = f"T={T}°C, RH={RH}%"
        axes[0].plot(paired.ch1["spectral_shift_ghz"].values, label=label, lw=1.2)
        axes[1].plot(paired.ch2["spectral_shift_ghz"].values, label=label, lw=1.2)

    axes[0].set_title(
        f"CH1_REF_T (length_2_m ∈ [{CH1_REF_T_RANGE_M[0]}, {CH1_REF_T_RANGE_M[1]}])"
    )
    axes[1].set_title(
        f"CH2_PI_TRH (length_2_m ∈ [{CH2_PI_TRH_RANGE_M[0]}, {CH2_PI_TRH_RANGE_M[1]}])"
    )
    for ax in axes:
        ax.set_xlabel("sensor position index")
        ax.set_ylabel("spectral shift (GHz)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Spectral shift profiles, replicate 1", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_response_surface(agg: pd.DataFrame) -> plt.Figure:
    """Heatmaps of mean spectral shift vs (T, RH) for both channels."""
    grouped = agg.groupby(["T", "RH"])[["ch1_mean", "ch2_mean"]].mean().reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, col, title in zip(
        axes,
        ["ch1_mean", "ch2_mean"],
        ["CH1_REF_T — mean spectral shift", "CH2_PI_TRH — mean spectral shift"],
        strict=True,
    ):
        piv = grouped.pivot(index="RH", columns="T", values=col)
        im = ax.imshow(
            piv.values,
            aspect="auto", origin="lower", cmap="RdBu_r",
            extent=(piv.columns.min(), piv.columns.max(),
                    piv.index.min(), piv.index.max()),
        )
        plt.colorbar(im, ax=ax, label="mean shift (GHz)")
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("Relative humidity (%)")
        ax.set_title(title)

    fig.suptitle("Response surface: channel mean vs (T, RH)", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_replicate_variance(agg: pd.DataFrame) -> plt.Figure:
    """Std of per-experiment mean shift across the 20 replicates of each state."""
    var = agg.groupby(["T", "RH"])[["ch1_mean", "ch2_mean"]].std().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, col, title in zip(
        axes,
        ["ch1_mean", "ch2_mean"],
        ["CH1_REF_T — std across replicates", "CH2_PI_TRH — std across replicates"],
        strict=True,
    ):
        sc = ax.scatter(
            var["T"], var["RH"], c=var[col],
            s=150, cmap="magma", edgecolor="black", linewidth=0.5,
        )
        plt.colorbar(sc, ax=ax, label="std of mean shift (GHz)")
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("Relative humidity (%)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    fig.suptitle(
        "Inter-replicate repeatability (lower is better)", fontweight="bold"
    )
    fig.tight_layout()
    return fig


def plot_quality_distribution(
    ds: "xr.Dataset",
    threshold: float = DROPOUT_QUALITY_THRESHOLD,
    sample_n: int = 50,
) -> plt.Figure:
    """Histogram of spectral_shift_quality for both channels.

    Args:
        ds: Zarr dataset.
        threshold: Dropout flag threshold to mark on the plot.
        sample_n: How many experiments to sample (full 700 also fine but slower).
    """
    n_total = ds.sizes["experiment"]
    rng = np.random.default_rng(42)
    indices = rng.choice(n_total, size=min(sample_n, n_total), replace=False)

    ch1_q: list[float] = []
    ch2_q: list[float] = []
    for i in indices:
        raw = _channels_from_zarr_row(ds, int(i))
        paired = extract_channels(raw)
        ch1_q.extend(paired.ch1["spectral_shift_quality"].dropna().tolist())
        ch2_q.extend(paired.ch2["spectral_shift_quality"].dropna().tolist())

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    bins = np.linspace(0, 1, 51)
    ax.hist(ch1_q, bins=bins, alpha=0.6, density=True, label="CH1_REF_T")
    ax.hist(ch2_q, bins=bins, alpha=0.6, density=True, label="CH2_PI_TRH")
    ax.axvline(
        threshold, color="red", linestyle="--", linewidth=1.5,
        label=f"dropout threshold = {threshold}",
    )
    ax.set_xlabel("spectral_shift_quality")
    ax.set_ylabel("density")
    ax.set_title(
        f"Quality distribution across {len(indices)} sampled experiments"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
