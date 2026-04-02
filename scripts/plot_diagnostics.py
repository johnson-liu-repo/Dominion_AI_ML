"""Generate diagnostic plots from training diagnostics CSVs.

Usage:
    python scripts/plot_diagnostics.py <diagnostics_dir>
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
    buy = pd.read_csv(diag_dir / "buy_decisions.csv") if (diag_dir / "buy_decisions.csv").exists() else None
    bucket = pd.read_csv(diag_dir / "coin_bucket_summary.csv") if (diag_dir / "coin_bucket_summary.csv").exists() else None
    return episode, rolling, buy, bucket


def plot_series(ax, x, y, *, label=None, color=None, linewidth=1.5):
    s = pd.to_numeric(y, errors="coerce")
    valid = s.notna()
    if valid.sum() == 0:
        return
    ax.plot(x[valid], s[valid], linewidth=linewidth, label=label, color=color)


def plot_economy(rolling: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    for col, label in [
        ("rolling_mean_buy_phase_coins_rl", "Rolling buy-phase coins"),
        ("rolling_rate_coins_ge_6_rl", "Rate of 6+ coin turns"),
        ("rolling_rate_coins_ge_8_rl", "Rate of 8+ coin turns"),
        ("rolling_fraction_episodes_reach_8_coins", "Fraction episodes reaching 8"),
    ]:
        if col in rolling.columns:
            plot_series(ax, rolling["episode"], rolling[col], label=label)
    ax.set_title("Economy Development")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Rolling metric")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "plot_economy_development.png", dpi=150)
    plt.close(fig)


def plot_purchase_milestones(episode: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    window = max(1, min(200, len(episode)))
    for col, label in [
        ("first_turn_buy_silver_rl", "First Silver"),
        ("first_turn_buy_gold_rl", "First Gold"),
        ("first_turn_buy_green_rl", "First Green"),
        ("first_turn_buy_duchy_rl", "First Duchy"),
        ("first_turn_buy_province_rl", "First Province"),
    ]:
        if col in episode.columns:
            vals = pd.to_numeric(episode[col], errors="coerce").rolling(window, min_periods=1).mean()
            plot_series(ax, episode["episode"], vals, label=label)
    ax.set_title("Purchase Timing Milestones (rolling mean turn)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Turn")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "plot_purchase_milestones.png", dpi=150)
    plt.close(fig)


def plot_conditionals(rolling: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(11, 5))
    cols = [
        "rolling_buy_silver_given_affordable_rl",
        "rolling_buy_gold_given_affordable_rl",
        "rolling_buy_province_given_affordable_rl",
        "rolling_buy_duchy_given_duchy_affordable_no_province_rl",
        "rolling_buy_silver_given_gold_affordable_rl",
        "rolling_buy_something_else_given_province_affordable_rl",
        "rolling_pass_given_gold_affordable_rl",
        "rolling_pass_given_province_affordable_rl",
    ]
    for col in cols:
        if col in rolling.columns:
            plot_series(ax, rolling["episode"], rolling[col], label=col.replace("rolling_", ""))
    ax.set_title("Conditional Decision Metrics")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Rate")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "plot_conditional_decisions.png", dpi=150)
    plt.close(fig)


def plot_coin_buckets(bucket: pd.DataFrame, out: Path):
    if bucket is None or bucket.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot = bucket.pivot_table(index="episode", columns="coin_bucket", values="pass_rate", aggfunc="mean")
    for col in ["2", "3", "4", "5", "6", "7", "8+"]:
        if col in pivot.columns:
            plot_series(ax, pivot.index.to_series(), pivot[col], label=f"pass rate @ {col}")
    ax.set_title("Coin-Bucket Pass/NONE Rate")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Pass rate")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "plot_coin_bucket_pass_rates.png", dpi=150)
    plt.close(fig)


def plot_deck_quality(episode: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    for col in ["final_treasure_density_rl", "final_green_density_rl", "final_vp_per_card_rl"]:
        if col in episode.columns:
            window = max(1, min(200, len(episode)))
            s = pd.to_numeric(episode[col], errors="coerce").rolling(window, min_periods=1).mean()
            plot_series(ax, episode["episode"], s, label=col.replace("final_", "").replace("_rl", ""))
    ax.set_title("Deck Quality Trends")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Value")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "plot_deck_quality_trends.png", dpi=150)
    plt.close(fig)


def plot_training_process(rolling: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    for col in [
        "rolling_episode_loss_mean",
        "rolling_episode_td_error_abs_mean",
        "rolling_episode_q_value_mean",
        "rolling_episode_q_value_max",
        "rolling_episode_epsilon_mean",
        "rolling_random_action_rate",
        "rolling_greedy_action_rate",
    ]:
        if col in rolling.columns:
            plot_series(ax, rolling["episode"], rolling[col], label=col.replace("rolling_", ""))
    ax.set_title("Training Process Metrics")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Rolling value")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "plot_training_process.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate diagnostic plots.")
    parser.add_argument("diagnostics_dir", type=Path)
    args = parser.parse_args()

    episode, rolling, buy, bucket = load_csvs(args.diagnostics_dir)
    results = args.diagnostics_dir / "results" / "plots"
    results.mkdir(parents=True, exist_ok=True)

    plot_economy(rolling, results)
    plot_purchase_milestones(episode, results)
    plot_conditionals(rolling, results)
    plot_coin_buckets(bucket, results)
    plot_deck_quality(episode, results)
    plot_training_process(rolling, results)

    print(f"Plots saved to {results}")


if __name__ == "__main__":
    main()
