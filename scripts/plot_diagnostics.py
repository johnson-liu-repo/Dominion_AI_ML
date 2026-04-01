"""Generate diagnostic plots from training diagnostics CSVs.

Usage:
    python scripts/plot_diagnostics.py <diagnostics_dir>

where <diagnostics_dir> contains episode_summary.csv, rolling_metrics.csv,
and optionally buy_decisions.csv.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))
sys.path.append(str(repo_root / "src"))


def load_csvs(diag_dir: Path):
    episode = pd.read_csv(diag_dir / "episode_summary.csv")
    rolling = pd.read_csv(diag_dir / "rolling_metrics.csv")
    buy = None
    buy_path = diag_dir / "buy_decisions.csv"
    if buy_path.exists() and buy_path.stat().st_size > 0:
        buy = pd.read_csv(buy_path)
    return episode, rolling, buy


def plot_series(ax, x, y, *, label=None, color=None, linewidth=1.5,
                alpha=1.0, marker_when_sparse=True):
    """Plot a series so very small runs still produce visible output."""
    series = pd.to_numeric(y, errors="coerce")
    valid = series.notna()
    x_valid = x[valid]
    y_valid = series[valid]
    if y_valid.empty:
        return

    marker = "o" if marker_when_sparse and len(y_valid) <= 2 else None
    ax.plot(
        x_valid,
        y_valid,
        linewidth=linewidth,
        label=label,
        color=color,
        alpha=alpha,
        marker=marker,
        markersize=5 if marker else None,
    )


def adaptive_window(n_rows: int, cap: int = 200) -> tuple[int, int]:
    """Choose rolling params that still work for short runs."""
    window = max(1, min(cap, n_rows))
    min_periods = 1 if n_rows < 10 else min(10, window)
    return window, min_periods


def plot_a_win_rate(rolling: pd.DataFrame, out: Path):
    """Plot A: Win rate over time."""
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_series(ax, rolling["episode"], rolling["rolling_win_rate"], linewidth=1.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Rolling Win Rate")
    ax.set_title("Plot A: Win Rate Over Time")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "plot_a_win_rate.png", dpi=150)
    plt.close(fig)


def plot_b_score_diff(episode: pd.DataFrame, rolling: pd.DataFrame, out: Path):
    """Plot B: Score diff over time."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(episode["episode"], episode["final_score_diff"],
               alpha=0.08, s=4, color="gray", label="raw")
    plot_series(ax, rolling["episode"], rolling["rolling_mean_score_diff"],
                linewidth=1.5, color="blue", label="rolling mean")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Score Diff (RL - Opponent)")
    ax.set_title("Plot B: Score Difference Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "plot_b_score_diff.png", dpi=150)
    plt.close(fig)


def plot_c_coin_generation(rolling: pd.DataFrame, out: Path):
    """Plot C: Buy-phase coin generation over time."""
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_series(ax, rolling["episode"], rolling["rolling_mean_buy_phase_coins_rl"],
                linewidth=1.5, label="rolling mean coins")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Buy-Phase Coins")
    ax.set_title("Plot C: Buy-Phase Coin Generation Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "plot_c_coin_generation.png", dpi=150)
    plt.close(fig)


def plot_d_affordability(rolling: pd.DataFrame, out: Path):
    """Plot D: Affordability rates over time."""
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_series(ax, rolling["episode"], rolling["rolling_afford_gold_rate_rl"],
                linewidth=1.5, label="Gold affordable rate")
    plot_series(ax, rolling["episode"], rolling["rolling_afford_province_rate_rl"],
                linewidth=1.5, label="Province affordable rate")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Rate (fraction of buy opportunities)")
    ax.set_title("Plot D: Affordability Rates Over Time")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "plot_d_affordability.png", dpi=150)
    plt.close(fig)


def plot_e_conditional_choice(rolling: pd.DataFrame, out: Path):
    """Plot E: Conditional choice rates over time (most important)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    cols = {
        "rolling_buy_gold_given_affordable_rl": "Buy Gold | Gold affordable",
        "rolling_buy_province_given_affordable_rl": "Buy Province | Province affordable",
        "rolling_buy_silver_given_gold_affordable_rl": "Buy Silver | Gold affordable",
        "rolling_buy_silver_given_province_affordable_rl": "Buy Silver | Province affordable",
    }
    for col, label in cols.items():
        plot_series(ax, rolling["episode"], rolling[col], linewidth=1.5, label=label)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Conditional Buy Rate")
    ax.set_title("Plot E: Conditional Choice Rates Over Time")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "plot_e_conditional_choice.png", dpi=150)
    plt.close(fig)


def plot_f_buy_counts(rolling: pd.DataFrame, episode: pd.DataFrame, out: Path):
    """Plot F: Buy counts over time by card (rolling rates)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    cols = {
        "rolling_buy_gold_rate_rl": "Gold",
        "rolling_buy_province_rate_rl": "Province",
    }
    for col, label in cols.items():
        plot_series(ax, rolling["episode"], rolling[col], linewidth=1.5, label=label)

    # Compute rolling buy rates for other cards from episode data
    window, min_periods = adaptive_window(len(episode))
    for card, color in [("Copper", "brown"), ("Silver", "silver"),
                        ("Estate", "green"), ("Duchy", "olive"),
                        ("Curse", "purple")]:
        col = f"buy_{card.lower()}_count_rl"
        if col in episode.columns:
            rate = episode[col].rolling(window, min_periods=min_periods).sum() / \
                   episode["num_buy_phases_rl"].rolling(window, min_periods=min_periods).sum()
            plot_series(ax, episode["episode"], rate, linewidth=1.0, alpha=0.8,
                        color=color, label=card)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Buy Rate (fraction of buy decisions)")
    ax.set_title("Plot F: Buy Counts Over Time by Card")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "plot_f_buy_counts.png", dpi=150)
    plt.close(fig)


def plot_g_threshold_timing(episode: pd.DataFrame, out: Path):
    """Plot G: First-threshold timing over time."""
    fig, ax = plt.subplots(figsize=(10, 5))
    window, min_periods = adaptive_window(len(episode))
    for col, label, color in [
        ("first_turn_reach_5_coins_rl", "First turn >= 5 coins", "green"),
        ("first_turn_reach_6_coins_rl", "First turn >= 6 coins", "blue"),
        ("first_turn_reach_8_coins_rl", "First turn >= 8 coins", "red"),
    ]:
        vals = pd.to_numeric(episode[col], errors="coerce")
        rolling_mean = vals.rolling(window, min_periods=min_periods).mean()
        plot_series(ax, episode["episode"], rolling_mean, linewidth=1.5,
                    color=color, label=label)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Turn Number (rolling mean)")
    ax.set_title("Plot G: First-Threshold Timing Over Time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "plot_g_threshold_timing.png", dpi=150)
    plt.close(fig)


def plot_h_greening_timing(episode: pd.DataFrame, out: Path):
    """Plot H: First-greening timing over time."""
    fig, ax = plt.subplots(figsize=(10, 5))
    window, min_periods = adaptive_window(len(episode))
    for col, label, color in [
        ("first_turn_buy_duchy_rl", "First Duchy buy", "olive"),
        ("first_turn_buy_province_rl", "First Province buy", "darkgreen"),
    ]:
        vals = pd.to_numeric(episode[col], errors="coerce")
        rolling_mean = vals.rolling(window, min_periods=min_periods).mean()
        plot_series(ax, episode["episode"], rolling_mean, linewidth=1.5,
                    color=color, label=label)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Turn Number (rolling mean)")
    ax.set_title("Plot H: First-Greening Timing Over Time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "plot_h_greening_timing.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate diagnostic plots.")
    parser.add_argument("diagnostics_dir", type=Path,
                        help="Path to the diagnostics/ directory")
    args = parser.parse_args()

    diag_dir = args.diagnostics_dir
    if not diag_dir.is_dir():
        print(f"Error: {diag_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    episode, rolling, buy = load_csvs(diag_dir)
    out = diag_dir

    print(f"Loaded {len(episode)} episode summaries, {len(rolling)} rolling rows")

    plot_a_win_rate(rolling, out)
    plot_b_score_diff(episode, rolling, out)
    plot_c_coin_generation(rolling, out)
    plot_d_affordability(rolling, out)
    plot_e_conditional_choice(rolling, out)
    plot_f_buy_counts(rolling, episode, out)
    plot_g_threshold_timing(episode, out)
    plot_h_greening_timing(episode, out)

    print(f"Plots saved to {out}")


if __name__ == "__main__":
    main()
