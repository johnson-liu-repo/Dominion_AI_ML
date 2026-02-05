#!/usr/bin/env python
"""
Plot episode reward and score_diff over time from a CSV file.

Examples:
  python scripts/plot_episode_metrics.py data/training/training_002/episode_data_over_time.csv
  python scripts/plot_episode_metrics.py data/training/training_002/episode_data_over_time.csv --out plots/episode_metrics.png
  python scripts/plot_episode_metrics.py data/training/training_002/episode_data_over_time.csv --smooth 25 --show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def _pick_x_column(df: pd.DataFrame, requested: str | None) -> tuple[str | None, pd.Series]:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"Requested x column '{requested}' not found in CSV.")
        return requested, df[requested]

    if "episode" in df.columns:
        return "episode", df["episode"]
    if "timestamp" in df.columns:
        return "timestamp", df["timestamp"]

    return None, pd.Series(range(1, len(df) + 1))


def _maybe_datetime_for_timestamp(series: pd.Series) -> pd.Series:
    # Heuristic: treat as ms if values are very large.
    try:
        max_value = float(series.max())
    except Exception:
        return series

    unit = "ms" if max_value > 1e12 else "s"
    converted = pd.to_datetime(series, unit=unit, errors="coerce")
    if converted.isna().all():
        return series
    return converted


def _smooth(series: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return series
    return series.rolling(window=window, min_periods=1).mean()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot episode reward and score_diff over time from a CSV file."
    )
    parser.add_argument("csv_path", help="Path to episode_data_over_time.csv")
    parser.add_argument(
        "--x",
        dest="x_column",
        default=None,
        help="Column to use for the x-axis (default: episode, then timestamp, else index).",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=1,
        help="Moving average window for smoothing (default: 1 = no smoothing).",
    )
    parser.add_argument("--out", default=None, help="Optional output image path.")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot window (useful when --out is also provided).",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional plot title. Default is based on the CSV file name.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    for required in ("reward", "score_diff"):
        if required not in df.columns:
            raise ValueError(f"Required column '{required}' not found in CSV.")

    x_name, x_series = _pick_x_column(df, args.x_column)
    if x_name == "timestamp":
        x_series = _maybe_datetime_for_timestamp(x_series)

    title = args.title or f"Episode Metrics: {csv_path.name}"

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 6), sharex=True)
    fig.suptitle(title)

    metrics = [
        ("reward", "Episode Reward"),
        ("score_diff", "Score Diff"),
    ]

    for ax, (col, label) in zip(axes, metrics):
        series = df[col]
        if args.smooth > 1:
            ax.plot(x_series, series, alpha=0.25, linewidth=1.0, label=f"{label} (raw)")
            ax.plot(
                x_series,
                _smooth(series, args.smooth),
                linewidth=2.0,
                label=f"{label} (ma{args.smooth})",
            )
        else:
            ax.plot(x_series, series, linewidth=1.5, label=label)

        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    if x_name:
        axes[-1].set_xlabel(x_name)
    else:
        axes[-1].set_xlabel("episode_index")

    fig.tight_layout()

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")

    if args.show or not args.out:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
