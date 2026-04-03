"""Generate diagnostics summary report from training CSVs.

Usage:
    python scripts/generate_diagnostics_report.py <diagnostics_dir>
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def pct(num, den):
    return (num / den) if den else np.nan


def mean_numeric(frame, column):
    if column not in frame.columns:
        return np.nan
    return pd.to_numeric(frame[column], errors="coerce").mean()


def sum_numeric(frame, column):
    if column not in frame.columns:
        return 0.0
    return pd.to_numeric(frame[column], errors="coerce").fillna(0).sum()


def series_numeric(frame, column):
    if column not in frame.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def format_float(value, decimals=2, na="N/A"):
    if pd.isna(value):
        return na
    return f"{value:.{decimals}f}"


def format_pct(value, decimals=2, na="N/A"):
    if pd.isna(value):
        return na
    return f"{value:.{decimals}%}"


def add_metric(lines, label, value):
    lines.append(f"  {label:<34} {value}")


def generate_report(diag_dir: Path) -> str:
    episode = pd.read_csv(diag_dir / "episode_summary.csv")

    rolling_path = diag_dir / "rolling_metrics.csv"
    rolling = pd.read_csv(rolling_path) if rolling_path.exists() else pd.DataFrame()

    bucket_path = diag_dir / "coin_bucket_summary.csv"
    bucket = pd.read_csv(bucket_path) if bucket_path.exists() else None

    buy_path = diag_dir / "buy_decisions.csv"
    buy = pd.read_csv(buy_path) if buy_path.exists() and buy_path.stat().st_size > 100 else None

    n = len(episode)
    lines = []
    lines.append("=" * 70)
    lines.append("  DOMINION RL AGENT -- DIAGNOSTICS REPORT")
    lines.append("=" * 70)
    lines.append(f"Total episodes analysed: {n}")
    lines.append("")

    win_rate = mean_numeric(episode, "did_rl_agent_win")
    avg_score_diff = mean_numeric(episode, "final_score_diff")
    avg_reward = mean_numeric(episode, "episode_reward_total")
    lines.append("--- Overall Performance ---")
    add_metric(lines, "Win rate:", format_pct(win_rate))
    add_metric(lines, "Average final score diff:", format_float(avg_score_diff))
    add_metric(lines, "Average episode reward:", format_float(avg_reward))
    lines.append("")

    avg_coins = mean_numeric(episode, "mean_buy_phase_coins_rl")
    reach_5 = mean_numeric(episode, "first_turn_reach_5_coins_rl")
    reach_6 = mean_numeric(episode, "first_turn_reach_6_coins_rl")
    reach_8 = mean_numeric(episode, "first_turn_reach_8_coins_rl")
    reach_8_fraction = series_numeric(episode, "first_turn_reach_8_coins_rl").notna().mean()
    lines.append("--- Economy ---")
    add_metric(lines, "Average buy-phase coins:", format_float(avg_coins))
    add_metric(lines, "Average first turn reaching 5 coins:", format_float(reach_5))
    add_metric(lines, "Average first turn reaching 6 coins:", format_float(reach_6))
    add_metric(lines, "Average first turn reaching 8 coins:", format_float(reach_8))
    add_metric(lines, "Episodes ever reaching 8 coins:", format_pct(reach_8_fraction))
    lines.append("")

    total_buys = sum_numeric(episode, "num_buy_phases_rl")
    total_afford_silver = sum_numeric(episode, "afford_silver_count_rl")
    total_afford_gold = sum_numeric(episode, "afford_gold_count_rl")
    total_afford_province = sum_numeric(episode, "afford_province_count_rl")
    total_afford_duchy = sum_numeric(episode, "afford_duchy_count_rl")
    total_afford_duchy_no_province = max(total_afford_duchy - total_afford_province, 0.0)

    pct_afford_gold = pct(total_afford_gold, total_buys)
    pct_afford_province = pct(total_afford_province, total_buys)
    gold_conv = pct(sum_numeric(episode, "buy_gold_when_affordable_count_rl"), total_afford_gold)
    province_conv = pct(sum_numeric(episode, "buy_province_when_affordable_count_rl"), total_afford_province)
    silver_conv = pct(sum_numeric(episode, "buy_silver_when_affordable_count_rl"), total_afford_silver)
    duchy_no_province_conv = pct(
        sum_numeric(episode, "buy_duchy_when_duchy_affordable_no_province_count_rl"),
        total_afford_duchy_no_province,
    )
    silver_when_gold = pct(sum_numeric(episode, "buy_silver_when_gold_affordable_count_rl"), total_afford_gold)
    something_else_when_province = pct(
        sum_numeric(episode, "buy_something_else_when_province_affordable_count_rl"),
        total_afford_province,
    )
    pass_when_gold = pct(sum_numeric(episode, "pass_when_gold_affordable_count_rl"), total_afford_gold)
    pass_when_province = pct(sum_numeric(episode, "pass_when_province_affordable_count_rl"), total_afford_province)

    lines.append("--- Affordability & Conditional Choice Rates ---")
    add_metric(lines, "% of buy opportunities with Gold affordable:", format_pct(pct_afford_gold))
    add_metric(lines, "% of buy opportunities with Province affordable:", format_pct(pct_afford_province))
    add_metric(lines, "When Silver affordable, bought Silver:", format_pct(silver_conv))
    add_metric(lines, "When Gold affordable, bought Gold:", format_pct(gold_conv))
    add_metric(lines, "When Province affordable, bought Province:", format_pct(province_conv))
    add_metric(
        lines,
        "When Duchy affordable without Province, bought Duchy:",
        format_pct(duchy_no_province_conv),
    )
    add_metric(lines, "When Gold affordable, bought Silver:", format_pct(silver_when_gold))
    add_metric(lines, "When Province affordable, bought something else:", format_pct(something_else_when_province))
    add_metric(lines, "When Gold affordable, passed:", format_pct(pass_when_gold))
    add_metric(lines, "When Province affordable, passed:", format_pct(pass_when_province))

    if buy is not None and "was_province_affordable" in buy.columns and "bought_province" in buy.columns:
        prov_aff = buy[buy["was_province_affordable"] == 1]
        if len(prov_aff) > 0 and "chosen_card" in prov_aff.columns:
            not_province = prov_aff[prov_aff["bought_province"] == 0]
            if len(not_province) > 0:
                alt = not_province["chosen_card"].value_counts().head(5)
                lines.append("  When Province affordable, most common alternatives:")
                for card, count in alt.items():
                    alt_pct = count / len(prov_aff)
                    lines.append(f"    {card}: {count} ({alt_pct:.1%})")

    if bucket is None or bucket.empty or "coin_bucket" not in bucket.columns or "pass_rate" not in bucket.columns:
        lines.append("  Coin-bucket pass/NONE rates: N/A")
    else:
        grouped = bucket.groupby("coin_bucket")["pass_rate"].mean().sort_index()
        lines.append("  Coin-bucket pass/NONE rates:")
        for coin_bucket, pass_rate in grouped.items():
            lines.append(f"    {coin_bucket}: {pass_rate:.2%}")
    lines.append("")

    first_silver = mean_numeric(episode, "first_turn_buy_silver_rl")
    first_gold = mean_numeric(episode, "first_turn_buy_gold_rl")
    first_green = mean_numeric(episode, "first_turn_buy_green_rl")
    first_duchy = mean_numeric(episode, "first_turn_buy_duchy_rl")
    first_province = mean_numeric(episode, "first_turn_buy_province_rl")
    province_buy_fraction = series_numeric(episode, "buy_province_count_rl").fillna(0).gt(0).mean()
    treasure_before_green = mean_numeric(episode, "treasure_buys_before_first_green_rl")
    green_before_gold = mean_numeric(episode, "green_buys_before_first_gold_rl")
    green_before_8 = mean_numeric(episode, "green_buys_before_first_8_coin_turn_rl")
    opp_first_province = mean_numeric(episode, "opponent_first_turn_buy_province")
    rl_minus_opp_province = mean_numeric(episode, "rl_minus_opp_first_province_turn")

    lines.append("--- Timing & Tempo ---")
    add_metric(lines, "Average first Silver buy:", format_float(first_silver))
    add_metric(lines, "Average first Gold buy:", format_float(first_gold))
    add_metric(lines, "Average first green-card buy:", format_float(first_green))
    add_metric(lines, "Average first Duchy buy:", format_float(first_duchy))
    add_metric(lines, "Average first turn reaching 8 coins:", format_float(reach_8, decimals=1))
    add_metric(lines, "Average first Province buy:", format_float(first_province, decimals=1))
    add_metric(lines, "Episodes with a Province buy:", format_pct(province_buy_fraction))
    add_metric(lines, "Treasure buys before first green:", format_float(treasure_before_green, decimals=3))
    add_metric(lines, "Green buys before first Gold:", format_float(green_before_gold, decimals=3))
    add_metric(lines, "Green buys before first 8-coin turn:", format_float(green_before_8, decimals=3))
    add_metric(lines, "Average opponent first Province turn:", format_float(opp_first_province))
    add_metric(lines, "Average RL minus opponent Province turn:", format_float(rl_minus_opp_province))
    lines.append("")

    lines.append("--- Final Deck Composition & Quality ---")
    for label, column in [
        ("Copper:", "final_count_copper_rl"),
        ("Silver:", "final_count_silver_rl"),
        ("Gold:", "final_count_gold_rl"),
        ("Estate:", "final_count_estate_rl"),
        ("Duchy:", "final_count_duchy_rl"),
        ("Province:", "final_count_province_rl"),
        ("Curse:", "final_count_curse_rl"),
        ("Deck size:", "final_deck_size_rl"),
    ]:
        add_metric(lines, label, format_float(mean_numeric(episode, column), decimals=1))

    for label, column in [
        ("Treasure count:", "final_treasure_count_rl"),
        ("Green count:", "final_green_count_rl"),
        ("Treasure density:", "final_treasure_density_rl"),
        ("Green density:", "final_green_density_rl"),
        ("Treasure to green ratio:", "final_treasure_to_green_ratio_rl"),
        ("Nominal treasure value:", "final_nominal_treasure_value_rl"),
        ("VP per card:", "final_vp_per_card_rl"),
    ]:
        add_metric(lines, label, format_float(mean_numeric(episode, column), decimals=4))
    lines.append("")

    lines.append("--- Training Process ---")
    for label, column in [
        ("Average episode loss:", "episode_loss_mean"),
        ("Average absolute TD error:", "episode_td_error_abs_mean"),
        ("Average Q value:", "episode_q_value_mean"),
        ("Average max Q value:", "episode_q_value_max"),
        ("Average epsilon:", "episode_epsilon_mean"),
        ("Average random action rate:", "episode_random_action_rate"),
        ("Average greedy action rate:", "episode_greedy_action_rate"),
    ]:
        decimals = 6 if "rate" not in label.lower() and "epsilon" not in label.lower() else 6
        value = mean_numeric(episode, column)
        add_metric(lines, label, format_float(value, decimals=decimals))

    if not rolling.empty:
        last = rolling.iloc[-1]
        random_rate = pd.to_numeric(pd.Series([last.get("rolling_random_action_rate", np.nan)]), errors="coerce").iloc[0]
        greedy_rate = pd.to_numeric(pd.Series([last.get("rolling_greedy_action_rate", np.nan)]), errors="coerce").iloc[0]
        add_metric(
            lines,
            "Last rolling random / greedy rates:",
            f"{format_float(random_rate, decimals=4)} / {format_float(greedy_rate, decimals=4)}",
        )
    else:
        add_metric(lines, "Last rolling random / greedy rates:", "N/A")
    lines.append("")

    lines.append("=" * 70)
    lines.append("  INTERPRETATION")
    lines.append("=" * 70)

    if not pd.isna(pct_afford_province) and pct_afford_province < 0.05:
        lines.append("  BOTTLENECK: Economy generation.")
        lines.append("  The agent rarely reaches 8 coins. It cannot afford Province.")
        lines.append("  Focus: improve treasure accumulation (buy more Gold/Silver).")
    elif not pd.isna(province_conv) and province_conv < 0.3:
        lines.append("  BOTTLENECK: Action choice after affordability.")
        lines.append(f"  Province is affordable {pct_afford_province:.1%} of buy opportunities,")
        lines.append(f"  but the agent only buys it {province_conv:.1%} of the time.")
        lines.append("  The agent reaches the money threshold but makes bad buy decisions.")
    elif not pd.isna(province_conv) and province_conv >= 0.3 and not pd.isna(win_rate) and win_rate < 0.4:
        lines.append("  MIXED: Agent buys Province sometimes but still loses.")
        lines.append("  Possible issues: timing (greening too early/late),")
        lines.append("  deck dilution, or suboptimal non-Province buy decisions.")
    elif not pd.isna(win_rate) and win_rate >= 0.4:
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
    parser.add_argument("diagnostics_dir", type=Path, help="Path to the diagnostics/ directory")
    args = parser.parse_args()

    diag_dir = args.diagnostics_dir
    if not diag_dir.is_dir():
        print(f"Error: {diag_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    report = generate_report(diag_dir)
    results = diag_dir / "results"
    results.mkdir(parents=True, exist_ok=True)
    report_path = results / "diagnostics_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved: {report_path}")


if __name__ == "__main__":
    main()
