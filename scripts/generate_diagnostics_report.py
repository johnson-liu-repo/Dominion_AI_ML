"""Generate diagnostics summary report from training CSVs.

Usage:
    python scripts/generate_diagnostics_report.py <diagnostics_dir>
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def pct(num, den):
    return (num / den) if den else np.nan


def generate_report(diag_dir: Path) -> str:
    episode = pd.read_csv(diag_dir / "episode_summary.csv")
    rolling = pd.read_csv(diag_dir / "rolling_metrics.csv")
    bucket = pd.read_csv(diag_dir / "coin_bucket_summary.csv") if (diag_dir / "coin_bucket_summary.csv").exists() else None

    n = len(episode)
    lines = ["=" * 72, "DOMINION RL DIAGNOSTICS REPORT", "=" * 72, f"Episodes: {n}", ""]

    lines.append("[1] Economy development")
    lines.append(f"- Avg first turn reaching 5 coins: {pd.to_numeric(episode['first_turn_reach_5_coins_rl'], errors='coerce').mean():.2f}")
    lines.append(f"- Avg first turn reaching 6 coins: {pd.to_numeric(episode['first_turn_reach_6_coins_rl'], errors='coerce').mean():.2f}")
    lines.append(f"- Avg first turn reaching 8 coins: {pd.to_numeric(episode['first_turn_reach_8_coins_rl'], errors='coerce').mean():.2f}")
    lines.append(f"- Fraction episodes ever reach 8 coins: {episode['first_turn_reach_8_coins_rl'].notna().mean():.2%}")
    lines.append(f"- Avg buy-phase coins: {episode['mean_buy_phase_coins_rl'].mean():.3f}")
    lines.append("")

    lines.append("[2] Purchase timing milestones")
    for col, label in [
        ("first_turn_buy_silver_rl", "first Silver buy"),
        ("first_turn_buy_gold_rl", "first Gold buy"),
        ("first_turn_buy_green_rl", "first green-card buy"),
        ("first_turn_buy_duchy_rl", "first Duchy buy"),
        ("first_turn_buy_province_rl", "first Province buy"),
    ]:
        lines.append(f"- Avg {label}: {pd.to_numeric(episode[col], errors='coerce').mean():.2f}")
    lines.append(f"- Fraction episodes with Province buy: {(episode['buy_province_count_rl'] > 0).mean():.2%}")
    lines.append("")

    lines.append("[3] Greening / tempo")
    for col in [
        "treasure_buys_before_first_green_rl",
        "green_buys_before_first_gold_rl",
        "green_buys_before_first_8_coin_turn_rl",
    ]:
        lines.append(f"- Avg {col}: {episode[col].mean():.3f}")
    lines.append(f"- Avg opponent first Province turn: {pd.to_numeric(episode['opponent_first_turn_buy_province'], errors='coerce').mean():.2f}")
    lines.append(f"- Avg RL minus opponent first Province turn: {pd.to_numeric(episode['rl_minus_opp_first_province_turn'], errors='coerce').mean():.2f}")
    lines.append("")

    lines.append("[4] Conditional decision metrics")
    total_aff_silver = episode["afford_silver_count_rl"].sum()
    total_aff_gold = episode["afford_gold_count_rl"].sum()
    total_aff_prov = episode["afford_province_count_rl"].sum()
    total_aff_duchy_no_prov = (episode["afford_duchy_count_rl"] - episode["afford_province_count_rl"]).clip(lower=0).sum()
    lines.append(f"- buy Silver | Silver affordable: {pct(episode['buy_silver_when_affordable_count_rl'].sum(), total_aff_silver):.2%}")
    lines.append(f"- buy Gold | Gold affordable: {pct(episode['buy_gold_when_affordable_count_rl'].sum(), total_aff_gold):.2%}")
    lines.append(f"- buy Province | Province affordable: {pct(episode['buy_province_when_affordable_count_rl'].sum(), total_aff_prov):.2%}")
    lines.append(f"- buy Duchy | Duchy affordable and Province not: {pct(episode['buy_duchy_when_duchy_affordable_no_province_count_rl'].sum(), total_aff_duchy_no_prov):.2%}")
    lines.append(f"- buy Silver | Gold affordable: {pct(episode['buy_silver_when_gold_affordable_count_rl'].sum(), total_aff_gold):.2%}")
    lines.append(f"- buy something else | Province affordable: {pct(episode['buy_something_else_when_province_affordable_count_rl'].sum(), total_aff_prov):.2%}")
    lines.append(f"- pass | Gold affordable: {pct(episode['pass_when_gold_affordable_count_rl'].sum(), total_aff_gold):.2%}")
    lines.append(f"- pass | Province affordable: {pct(episode['pass_when_province_affordable_count_rl'].sum(), total_aff_prov):.2%}")
    lines.append("")

    lines.append("[5] Coin-bucket action analysis")
    if bucket is None or bucket.empty:
        lines.append("- coin_bucket_summary.csv not present")
    else:
        grp = bucket.groupby("coin_bucket")["pass_rate"].mean().sort_index()
        for b, v in grp.items():
            lines.append(f"- Avg pass/NONE rate at {b} coins: {v:.2%}")
    lines.append("")

    lines.append("[6] Deck-quality metrics")
    for col in [
        "final_count_copper_rl", "final_count_silver_rl", "final_count_gold_rl",
        "final_count_estate_rl", "final_count_duchy_rl", "final_count_province_rl",
        "final_count_curse_rl", "final_deck_size_rl", "final_treasure_count_rl",
        "final_green_count_rl", "final_treasure_density_rl", "final_green_density_rl",
        "final_treasure_to_green_ratio_rl", "final_nominal_treasure_value_rl", "final_vp_per_card_rl",
    ]:
        lines.append(f"- Avg {col}: {pd.to_numeric(episode[col], errors='coerce').mean():.4f}")
    lines.append("")

    lines.append("[7] Training-process metrics")
    for col in [
        "episode_loss_mean", "episode_td_error_abs_mean", "episode_q_value_mean", "episode_q_value_max",
        "episode_epsilon_mean", "episode_random_action_rate", "episode_greedy_action_rate",
    ]:
        lines.append(f"- Avg {col}: {pd.to_numeric(episode[col], errors='coerce').mean():.6f}")
    if not rolling.empty:
        last = rolling.iloc[-1]
        lines.append(f"- Last rolling random/greedy action rates: {last.get('rolling_random_action_rate', np.nan):.4f} / {last.get('rolling_greedy_action_rate', np.nan):.4f}")

    lines.append("\nArtifacts:")
    lines.append(f"- Episode summary: {diag_dir / 'episode_summary.csv'}")
    lines.append(f"- Buy decisions: {diag_dir / 'buy_decisions.csv'}")
    lines.append(f"- Rolling metrics: {diag_dir / 'rolling_metrics.csv'}")
    lines.append(f"- Coin buckets: {diag_dir / 'coin_bucket_summary.csv'}")
    lines.append(f"- Training steps: {diag_dir / 'training_step_metrics.csv'}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("diagnostics_dir", type=Path)
    args = parser.parse_args()

    report = generate_report(args.diagnostics_dir)
    results = args.diagnostics_dir / "results"
    results.mkdir(parents=True, exist_ok=True)
    path = results / "diagnostics_report.txt"
    path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
