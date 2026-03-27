"""Generate a plain-text diagnostics summary report from training CSVs.

Usage:
    python scripts/generate_diagnostics_report.py <diagnostics_dir>
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np


def generate_report(diag_dir: Path) -> str:
    episode = pd.read_csv(diag_dir / "episode_summary.csv")
    buy_path = diag_dir / "buy_decisions.csv"
    buy = pd.read_csv(buy_path) if buy_path.exists() and buy_path.stat().st_size > 100 else None

    n = len(episode)
    lines = []
    lines.append("=" * 70)
    lines.append("  DOMINION RL AGENT -- DIAGNOSTICS REPORT")
    lines.append("=" * 70)
    lines.append(f"Total episodes analysed: {n}")
    lines.append("")

    # --- Overall performance ---
    win_rate = episode["did_rl_agent_win"].mean()
    avg_score_diff = episode["final_score_diff"].mean()
    avg_reward = episode["episode_reward_total"].mean()
    lines.append("--- Overall Performance ---")
    lines.append(f"  Win rate:                {win_rate:.2%}")
    lines.append(f"  Average final score diff: {avg_score_diff:.2f}")
    lines.append(f"  Average episode reward:   {avg_reward:.2f}")
    lines.append("")

    # --- Economy ---
    avg_coins = episode["mean_buy_phase_coins_rl"].mean()
    lines.append("--- Economy ---")
    lines.append(f"  Average buy-phase coins:  {avg_coins:.2f}")
    lines.append("")

    # --- Affordability ---
    total_buys = episode["num_buy_phases_rl"].sum()
    total_afford_gold = episode["afford_gold_count_rl"].sum()
    total_afford_province = episode["afford_province_count_rl"].sum()
    pct_afford_gold = total_afford_gold / total_buys if total_buys > 0 else 0
    pct_afford_province = total_afford_province / total_buys if total_buys > 0 else 0
    lines.append("--- Affordability ---")
    lines.append(f"  % of buy opportunities with Gold affordable:     {pct_afford_gold:.2%}")
    lines.append(f"  % of buy opportunities with Province affordable: {pct_afford_province:.2%}")
    lines.append("")

    # --- Conditional choices ---
    total_buy_gold = episode["buy_gold_count_rl"].sum()
    total_buy_province = episode["buy_province_count_rl"].sum()
    total_buy_gold_when_aff = episode["buy_gold_when_affordable_count_rl"].sum()
    total_buy_prov_when_aff = episode["buy_province_when_affordable_count_rl"].sum()

    gold_conv = total_buy_gold_when_aff / total_afford_gold if total_afford_gold > 0 else float("nan")
    prov_conv = total_buy_prov_when_aff / total_afford_province if total_afford_province > 0 else float("nan")

    lines.append("--- Conditional Choice Rates ---")
    lines.append(f"  When Gold affordable, bought Gold:         {gold_conv:.2%}")
    lines.append(f"  When Province affordable, bought Province: {prov_conv:.2%}")

    # What does the agent buy instead of Province?
    if buy is not None and "was_province_affordable" in buy.columns:
        prov_aff = buy[buy["was_province_affordable"] == 1]
        if len(prov_aff) > 0:
            not_province = prov_aff[prov_aff["bought_province"] == 0]
            if len(not_province) > 0:
                alt = not_province["chosen_card"].value_counts().head(5)
                lines.append(f"  When Province affordable, most common alternatives:")
                for card, count in alt.items():
                    pct = count / len(prov_aff)
                    lines.append(f"    {card}: {count} ({pct:.1%})")
    lines.append("")

    # --- Timing ---
    reach_8 = pd.to_numeric(episode["first_turn_reach_8_coins_rl"], errors="coerce")
    buy_prov_turn = pd.to_numeric(episode["first_turn_buy_province_rl"], errors="coerce")
    avg_reach_8 = reach_8.mean()
    avg_first_prov = buy_prov_turn.mean()
    lines.append("--- Timing ---")
    lines.append(f"  Average first turn reaching 8 coins:    {avg_reach_8:.1f}" if not np.isnan(avg_reach_8) else "  Average first turn reaching 8 coins:    N/A (never reached)")
    lines.append(f"  Average first turn buying Province:      {avg_first_prov:.1f}" if not np.isnan(avg_first_prov) else "  Average first turn buying Province:      N/A (never bought)")
    lines.append("")

    # --- Final deck composition ---
    avg_gold = episode["final_count_gold_rl"].mean()
    avg_province = episode["final_count_province_rl"].mean()
    lines.append("--- Final Deck Composition (RL Agent Avg) ---")
    lines.append(f"  Copper:   {episode['final_count_copper_rl'].mean():.1f}")
    lines.append(f"  Silver:   {episode['final_count_silver_rl'].mean():.1f}")
    lines.append(f"  Gold:     {avg_gold:.1f}")
    lines.append(f"  Estate:   {episode['final_count_estate_rl'].mean():.1f}")
    lines.append(f"  Duchy:    {episode['final_count_duchy_rl'].mean():.1f}")
    lines.append(f"  Province: {avg_province:.1f}")
    lines.append(f"  Curse:    {episode['final_count_curse_rl'].mean():.1f}")
    lines.append(f"  Deck size:{episode['final_deck_size_rl'].mean():.1f}")
    lines.append("")

    # --- Interpretation ---
    lines.append("=" * 70)
    lines.append("  INTERPRETATION")
    lines.append("=" * 70)

    if pct_afford_province < 0.05:
        lines.append("  BOTTLENECK: Economy generation.")
        lines.append("  The agent rarely reaches 8 coins. It cannot afford Province.")
        lines.append("  Focus: improve treasure accumulation (buy more Gold/Silver).")
    elif not np.isnan(prov_conv) and prov_conv < 0.3:
        lines.append("  BOTTLENECK: Action choice after affordability.")
        lines.append(f"  Province is affordable {pct_afford_province:.1%} of buy opportunities,")
        lines.append(f"  but the agent only buys it {prov_conv:.1%} of the time.")
        lines.append("  The agent reaches the money threshold but makes bad buy decisions.")
    elif not np.isnan(prov_conv) and prov_conv >= 0.3 and win_rate < 0.4:
        lines.append("  MIXED: Agent buys Province sometimes but still loses.")
        lines.append("  Possible issues: timing (greening too early/late),")
        lines.append("  deck dilution, or suboptimal non-Province buy decisions.")
    elif win_rate >= 0.4:
        lines.append("  The agent shows reasonable performance.")
        lines.append("  Further analysis of conditional choice rates may reveal")
        lines.append("  opportunities for improvement.")
    else:
        lines.append("  Insufficient data to determine clear bottleneck.")
        lines.append("  Run more episodes for clearer signal.")

    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate diagnostics text report.")
    parser.add_argument("diagnostics_dir", type=Path,
                        help="Path to the diagnostics/ directory")
    args = parser.parse_args()

    diag_dir = args.diagnostics_dir
    if not diag_dir.is_dir():
        print(f"Error: {diag_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    report = generate_report(diag_dir)
    print(report)

    report_path = diag_dir / "diagnostics_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
