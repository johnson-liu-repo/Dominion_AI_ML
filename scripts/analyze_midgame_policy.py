"""Reusable midgame policy diagnostics for Dominion RL buy decisions.

Usage:
    python scripts/analyze_midgame_policy.py <diagnostics_dir>

The script reads existing diagnostics artifacts and writes reusable summaries,
filtered CSVs, and a markdown report under the same diagnostics directory.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def to_int(value: str, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def to_float(value: str, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def pct(n: int, d: int) -> float:
    return (100.0 * n / d) if d else 0.0


def write_csv(path: Path, fieldnames: List[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def normalize_choice(row: Dict[str, str]) -> str:
    card = (row.get("chosen_card") or "").strip()
    action = (row.get("chosen_action") or "").strip().upper()
    bought_none = to_int(row.get("bought_none", "0")) == 1
    if bought_none or card in {"", "NONE", "PASS"} or action in {"PASS", "NONE", "NO_BUY"}:
        return "NONE"
    return card


def coin_bucket(coins: int) -> Optional[str]:
    if coins == 4:
        return "4"
    if coins == 5:
        return "5"
    if coins == 6:
        return "6"
    if coins == 7:
        return "7"
    if coins >= 8:
        return "8+"
    return None


@dataclass
class EpisodeThresholds:
    first_turn_reach_8: Optional[int]
    first_turn_buy_gold: Optional[int]


def parse_optional_turn(value: str) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        turn = int(float(value))
        return turn if turn > 0 else None
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze midgame buy policy diagnostics.")
    parser.add_argument("diagnostics_dir", type=Path, help="Path to diagnostics directory")
    args = parser.parse_args()

    diag_dir = args.diagnostics_dir
    buy_path = diag_dir / "buy_decisions.csv"
    ep_path = diag_dir / "episode_summary.csv"
    rolling_path = diag_dir / "rolling_metrics.csv"

    if not diag_dir.is_dir():
        raise SystemExit(f"Not a directory: {diag_dir}")
    if not buy_path.exists():
        raise SystemExit(f"Missing buy_decisions.csv at {buy_path}")

    with buy_path.open(newline="", encoding="utf-8") as f:
        buy_rows = list(csv.DictReader(f))
    episode_rows = []
    if ep_path.exists():
        with ep_path.open(newline="", encoding="utf-8") as f:
            episode_rows = list(csv.DictReader(f))
    rolling_rows = []
    if rolling_path.exists():
        with rolling_path.open(newline="", encoding="utf-8") as f:
            rolling_rows = list(csv.DictReader(f))

    rl_rows = [r for r in buy_rows if r.get("episode")]
    for row in rl_rows:
        row["normalized_choice"] = normalize_choice(row)
        row["coins_int"] = to_int(row.get("coins_available", "0"))
        row["turn_int"] = to_int(row.get("turn_index", "0"))
        row["episode_int"] = to_int(row.get("episode", "0"))

    episode_thresholds: Dict[int, EpisodeThresholds] = {}
    for row in episode_rows:
        ep = to_int(row.get("episode", "0"))
        if ep <= 0:
            continue
        episode_thresholds[ep] = EpisodeThresholds(
            first_turn_reach_8=parse_optional_turn(row.get("first_turn_reach_8_coins_rl", "")),
            first_turn_buy_gold=parse_optional_turn(row.get("first_turn_buy_gold_rl", "")),
        )

    # A) Buy distribution by coin bucket.
    bucket_counts: Dict[str, Counter] = defaultdict(Counter)
    bucket_totals: Counter = Counter()
    for row in rl_rows:
        bucket = coin_bucket(row["coins_int"])
        if bucket is None:
            continue
        choice = row["normalized_choice"]
        bucket_counts[bucket][choice] += 1
        bucket_totals[bucket] += 1

    dist_rows: List[Dict[str, object]] = []
    for bucket in ["4", "5", "6", "7", "8+"]:
        total = bucket_totals[bucket]
        for choice, count in bucket_counts[bucket].most_common():
            dist_rows.append(
                {
                    "coin_bucket": bucket,
                    "total_decisions": total,
                    "choice": choice,
                    "count": count,
                    "pct": round(pct(count, total), 2),
                }
            )
    write_csv(diag_dir / "buy_distribution_by_coins.csv", ["coin_bucket", "total_decisions", "choice", "count", "pct"], dist_rows)

    # B) Gold affordable not bought.
    gold_not_rows = [
        r for r in rl_rows if to_int(r.get("was_gold_affordable", "0")) == 1 and to_int(r.get("bought_gold", "0")) == 0
    ]
    gold_alt = Counter(r["normalized_choice"] for r in gold_not_rows)
    gold_by_coin: Dict[int, Counter] = defaultdict(Counter)
    for r in gold_not_rows:
        gold_by_coin[r["coins_int"]][r["normalized_choice"]] += 1

    suspicious_subs = {"Copper", "Estate", "NONE", "Silver", "Curse"}
    gold_suspicious = sum(1 for r in gold_not_rows if r["normalized_choice"] in suspicious_subs)

    write_csv(
        diag_dir / "gold_affordable_not_bought.csv",
        ["episode", "turn_index", "coins_available", "chosen_card", "normalized_choice", "was_province_affordable", "total_deck_size_rl_current"],
        [
            {
                "episode": r.get("episode", ""),
                "turn_index": r.get("turn_index", ""),
                "coins_available": r.get("coins_available", ""),
                "chosen_card": r.get("chosen_card", ""),
                "normalized_choice": r.get("normalized_choice", ""),
                "was_province_affordable": r.get("was_province_affordable", ""),
                "total_deck_size_rl_current": r.get("total_deck_size_rl_current", ""),
            }
            for r in gold_not_rows
        ],
    )

    gold_summary_rows: List[Dict[str, object]] = []
    for choice, count in gold_alt.most_common():
        gold_summary_rows.append({"view": "overall", "coins_available": "ALL", "choice": choice, "count": count, "pct": round(pct(count, len(gold_not_rows)), 2)})
    for coins in sorted(gold_by_coin):
        total = sum(gold_by_coin[coins].values())
        for choice, count in gold_by_coin[coins].most_common():
            gold_summary_rows.append({"view": "by_coins", "coins_available": coins, "choice": choice, "count": count, "pct": round(pct(count, total), 2)})
    write_csv(diag_dir / "gold_affordable_not_bought_summary.csv", ["view", "coins_available", "choice", "count", "pct"], gold_summary_rows)

    # C) Province affordable not bought.
    prov_not_rows = [
        r for r in rl_rows if to_int(r.get("was_province_affordable", "0")) == 1 and to_int(r.get("bought_province", "0")) == 0
    ]
    prov_alt = Counter(r["normalized_choice"] for r in prov_not_rows)
    prov_coin = Counter(r["coins_int"] for r in prov_not_rows)
    write_csv(
        diag_dir / "province_affordable_not_bought.csv",
        ["episode", "turn_index", "coins_available", "chosen_card", "normalized_choice", "was_gold_affordable", "total_deck_size_rl_current"],
        [
            {
                "episode": r.get("episode", ""),
                "turn_index": r.get("turn_index", ""),
                "coins_available": r.get("coins_available", ""),
                "chosen_card": r.get("chosen_card", ""),
                "normalized_choice": r.get("normalized_choice", ""),
                "was_gold_affordable": r.get("was_gold_affordable", ""),
                "total_deck_size_rl_current": r.get("total_deck_size_rl_current", ""),
            }
            for r in prov_not_rows
        ],
    )
    write_csv(
        diag_dir / "province_affordable_not_bought_summary.csv",
        ["view", "metric", "value", "count", "pct"],
        [
            *[
                {"view": "alternatives", "metric": "choice", "value": choice, "count": count, "pct": round(pct(count, len(prov_not_rows)), 2)}
                for choice, count in prov_alt.most_common()
            ],
            *[
                {"view": "coins", "metric": "coins_available", "value": coins, "count": count, "pct": round(pct(count, len(prov_not_rows)), 2)}
                for coins, count in sorted(prov_coin.items())
            ],
        ],
    )

    # D) Early greening heuristic analysis.
    # Heuristic: early green if Duchy/Estate bought before first Gold OR before first turn reaching 8 coins.
    first_green: Dict[Tuple[int, str], int] = {}
    for r in rl_rows:
        ch = r["normalized_choice"]
        if ch not in {"Duchy", "Estate"}:
            continue
        key = (r["episode_int"], ch)
        if key not in first_green or r["turn_int"] < first_green[key]:
            first_green[key] = r["turn_int"]

    early_rows: List[Dict[str, object]] = []
    duchy_episodes = set()
    estate_episodes = set()
    early_duchy = 0
    early_estate = 0
    for (ep, card), turn in sorted(first_green.items()):
        if ep <= 0:
            continue
        if card == "Duchy":
            duchy_episodes.add(ep)
        if card == "Estate":
            estate_episodes.add(ep)
        thresholds = episode_thresholds.get(ep, EpisodeThresholds(None, None))
        before_gold = thresholds.first_turn_buy_gold is None or turn < thresholds.first_turn_buy_gold
        before_8 = thresholds.first_turn_reach_8 is None or turn < thresholds.first_turn_reach_8
        early_flag = before_gold or before_8
        if card == "Duchy" and early_flag:
            early_duchy += 1
        if card == "Estate" and early_flag:
            early_estate += 1
        early_rows.append(
            {
                "episode": ep,
                "green_card": card,
                "first_green_turn": turn,
                "first_turn_buy_gold": thresholds.first_turn_buy_gold or "",
                "first_turn_reach_8": thresholds.first_turn_reach_8 or "",
                "before_first_gold": int(before_gold),
                "before_reach_8": int(before_8),
                "early_green_flag": int(early_flag),
            }
        )
    write_csv(
        diag_dir / "early_greening_analysis.csv",
        ["episode", "green_card", "first_green_turn", "first_turn_buy_gold", "first_turn_reach_8", "before_first_gold", "before_reach_8", "early_green_flag"],
        early_rows,
    )

    # E) Silver vs Gold.
    silver_rows = [r for r in rl_rows if r["normalized_choice"] == "Silver"]
    gold_rows = [r for r in rl_rows if r["normalized_choice"] == "Gold"]
    silver_when_gold_aff = sum(1 for r in silver_rows if to_int(r.get("was_gold_affordable", "0")) == 1)
    silver_when_prov_aff = sum(1 for r in silver_rows if to_int(r.get("was_province_affordable", "0")) == 1)
    silver_coin_bucket = Counter(coin_bucket(r["coins_int"]) or "<4" for r in silver_rows)

    sg_rows = [
        {"metric": "total_silver_buys", "value": len(silver_rows)},
        {"metric": "total_gold_buys", "value": len(gold_rows)},
        {"metric": "silver_to_gold_ratio", "value": round((len(silver_rows) / len(gold_rows)), 3) if gold_rows else "inf"},
        {"metric": "silver_when_gold_affordable", "value": silver_when_gold_aff},
        {"metric": "silver_when_province_affordable", "value": silver_when_prov_aff},
    ]
    for bucket, count in sorted(silver_coin_bucket.items()):
        sg_rows.append({"metric": f"silver_buys_coin_bucket_{bucket}", "value": count})

    if rolling_rows:
        for row in rolling_rows:
            ep = row.get("episode", "")
            silver_rate = to_float(row.get("rolling_buy_silver_given_gold_affordable_rl", "0"))
            gold_rate = to_float(row.get("rolling_buy_gold_given_affordable_rl", "0"))
            ratio = round((silver_rate / gold_rate), 4) if gold_rate > 0 else ""
            sg_rows.append({"metric": "rolling_ratio_silver_given_gold_to_gold_given_gold", "value": ratio, "episode": ep})

    write_csv(diag_dir / "silver_vs_gold_summary.csv", sorted({k for row in sg_rows for k in row.keys()}), sg_rows)

    # F) NONE/pass analysis.
    none_rows = [r for r in rl_rows if r["normalized_choice"] == "NONE"]
    none_coin = Counter(r["coins_int"] for r in none_rows)
    suspicious_none = [r for r in none_rows if r["coins_int"] >= 5 and (to_int(r.get("was_gold_affordable", "0")) == 1 or to_int(r.get("was_silver_affordable", "0")) == 1)]

    write_csv(
        diag_dir / "none_buy_analysis.csv",
        ["episode", "turn_index", "coins_available", "affordable_cards", "was_gold_affordable", "was_province_affordable", "suspicious_pass_flag"],
        [
            {
                "episode": r.get("episode", ""),
                "turn_index": r.get("turn_index", ""),
                "coins_available": r.get("coins_available", ""),
                "affordable_cards": r.get("affordable_cards", ""),
                "was_gold_affordable": r.get("was_gold_affordable", ""),
                "was_province_affordable": r.get("was_province_affordable", ""),
                "suspicious_pass_flag": int(r in suspicious_none),
            }
            for r in none_rows
        ],
    )

    # G) Weak-deck greening proxy.
    # Proxy flag: Duchy/Province bought when no prior Gold in episode OR before first 8-coin turn OR deck size <= 18.
    first_gold_by_ep: Dict[int, Optional[int]] = defaultdict(lambda: None)
    for r in rl_rows:
        if r["normalized_choice"] == "Gold":
            ep = r["episode_int"]
            t = r["turn_int"]
            if ep > 0 and (first_gold_by_ep[ep] is None or t < first_gold_by_ep[ep]):
                first_gold_by_ep[ep] = t

    weak_rows = []
    weak_count = 0
    total_green_major = 0
    for r in rl_rows:
        if r["normalized_choice"] not in {"Duchy", "Province"}:
            continue
        total_green_major += 1
        ep = r["episode_int"]
        turn = r["turn_int"]
        deck_size = to_int(r.get("total_deck_size_rl_current", "0"))
        thresholds = episode_thresholds.get(ep, EpisodeThresholds(None, None))
        first_gold = first_gold_by_ep.get(ep)
        before_first_gold = first_gold is None or turn < first_gold
        before_reach_8 = thresholds.first_turn_reach_8 is None or turn < thresholds.first_turn_reach_8
        small_payload = deck_size <= 18
        weak = before_first_gold or before_reach_8 or small_payload
        if weak:
            weak_count += 1
        weak_rows.append(
            {
                "episode": ep,
                "turn_index": turn,
                "green_card": r["normalized_choice"],
                "coins_available": r["coins_int"],
                "deck_size": deck_size,
                "first_gold_turn": first_gold or "",
                "first_turn_reach_8": thresholds.first_turn_reach_8 or "",
                "before_first_gold": int(before_first_gold),
                "before_reach_8": int(before_reach_8),
                "small_payload_proxy": int(small_payload),
                "weak_deck_green_flag": int(weak),
            }
        )
    write_csv(
        diag_dir / "weak_deck_greening_analysis.csv",
        ["episode", "turn_index", "green_card", "coins_available", "deck_size", "first_gold_turn", "first_turn_reach_8", "before_first_gold", "before_reach_8", "small_payload_proxy", "weak_deck_green_flag"],
        weak_rows,
    )

    # Build markdown report.
    lines: List[str] = []
    lines.append("# Midgame Policy Analysis")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append(f"- Decisions analyzed: **{len(rl_rows)}** RL buy decisions across **{len({r['episode_int'] for r in rl_rows if r['episode_int'] > 0})}** episodes.")
    lines.append(f"- Gold affordable but not bought: **{len(gold_not_rows)}** decisions ({pct(len(gold_not_rows), sum(1 for r in rl_rows if to_int(r.get('was_gold_affordable','0'))==1)):.1f}% of Gold-affordable states).")
    lines.append(f"- Province affordable but not bought: **{len(prov_not_rows)}** decisions ({pct(len(prov_not_rows), sum(1 for r in rl_rows if to_int(r.get('was_province_affordable','0'))==1)):.1f}% of Province-affordable states).")
    lines.append(f"- Silver vs Gold buys: **{len(silver_rows)} Silver** vs **{len(gold_rows)} Gold** (ratio {round(len(silver_rows)/len(gold_rows),2) if gold_rows else 'inf'}).")
    lines.append(f"- NONE/pass decisions: **{len(none_rows)}** total, including **{len(suspicious_none)}** suspicious passes at 5+ coins.")
    lines.append("")

    lines.append("## A) Buy Distribution by Coin Total")
    for bucket in ["4", "5", "6", "7", "8+"]:
        total = bucket_totals[bucket]
        lines.append(f"- Coins {bucket}: {total} decisions")
        for choice, count in bucket_counts[bucket].most_common(5):
            lines.append(f"  - {choice}: {count} ({pct(count, total):.1f}%)")
    lines.append("")

    lines.append("## B) Gold-Affordable but Not Bought")
    lines.append(f"- Total: {len(gold_not_rows)}")
    lines.append(f"- Suspicious substitutions (Copper/Estate/NONE/Silver/Curse): {gold_suspicious} ({pct(gold_suspicious, len(gold_not_rows)):.1f}%)")
    lines.append("- Most common alternatives:")
    for choice, count in gold_alt.most_common(8):
        lines.append(f"  - {choice}: {count} ({pct(count, len(gold_not_rows)):.1f}%)")
    lines.append("")

    lines.append("## C) Province-Affordable but Not Bought")
    lines.append(f"- Total: {len(prov_not_rows)}")
    lines.append("- Most common alternatives:")
    for choice, count in prov_alt.most_common(8):
        lines.append(f"  - {choice}: {count} ({pct(count, len(prov_not_rows)):.1f}%)")
    lines.append("- Coin distribution in these cases:")
    for coins, count in sorted(prov_coin.items()):
        lines.append(f"  - {coins} coins: {count} ({pct(count, len(prov_not_rows)):.1f}%)")
    lines.append("")

    lines.append("## D) Early Greening Analysis")
    lines.append("Heuristic: a first Duchy/Estate buy is flagged early if it occurs before first Gold buy OR before first 8-coin turn in that episode.")
    lines.append(f"- Episodes with Duchy buys: {len(duchy_episodes)}; early Duchy episodes: {early_duchy} ({pct(early_duchy, len(duchy_episodes)):.1f}%).")
    lines.append(f"- Episodes with Estate buys: {len(estate_episodes)}; early Estate episodes: {early_estate} ({pct(early_estate, len(estate_episodes)):.1f}%).")
    sample_early = [r for r in early_rows if r["early_green_flag"] == 1][:5]
    if sample_early:
        lines.append("- Example early-greening episodes (first 5):")
        for r in sample_early:
            lines.append(
                f"  - Ep {r['episode']} {r['green_card']} at turn {r['first_green_turn']} (first Gold={r['first_turn_buy_gold'] or 'N/A'}, first 8-coins={r['first_turn_reach_8'] or 'N/A'})"
            )
    lines.append("")

    lines.append("## E) Silver vs Gold Tendency")
    lines.append(f"- Total Silver buys: {len(silver_rows)}")
    lines.append(f"- Total Gold buys: {len(gold_rows)}")
    lines.append(f"- Silver buys when Gold affordable: {silver_when_gold_aff}")
    lines.append(f"- Silver buys when Province affordable: {silver_when_prov_aff}")
    lines.append("- Silver buys by coin bucket:")
    for bucket, count in sorted(silver_coin_bucket.items()):
        lines.append(f"  - {bucket}: {count}")
    lines.append("")

    lines.append("## F) NONE / Pass Analysis")
    lines.append(f"- Total NONE/pass decisions: {len(none_rows)}")
    lines.append("- Coin distribution for NONE/pass:")
    for coins, count in sorted(none_coin.items()):
        lines.append(f"  - {coins}: {count} ({pct(count, len(none_rows)):.1f}%)")
    lines.append(f"- Suspicious NONE/pass at 5+ coins with useful buys affordable: {len(suspicious_none)} ({pct(len(suspicious_none), len(none_rows)):.1f}% of all passes)")
    lines.append("")

    lines.append("## G) Weak-Deck Greening Analysis")
    lines.append("Proxy rule: Duchy/Province buy is flagged weak-deck greening if before first Gold, before first 8-coin turn, or at deck size <= 18.")
    lines.append(f"- Major green buys (Duchy/Province): {total_green_major}")
    lines.append(f"- Flagged weak-deck green buys: {weak_count} ({pct(weak_count, total_green_major):.1f}%)")
    lines.append("")

    lines.append("## Bottom-Line Diagnosis")
    diagnosis = []
    if len(gold_rows) <= max(3, len(silver_rows) // 3):
        diagnosis.append("insufficient Gold acquisition")
    if len(gold_not_rows) and pct(gold_suspicious, len(gold_not_rows)) > 50:
        diagnosis.append("weak 4–6 coin decision-making")
    if len(suspicious_none) > 0:
        diagnosis.append("too many pass/NONE decisions")
    if total_green_major > 0 and pct(weak_count, total_green_major) > 40:
        diagnosis.append("premature greening")
    if not diagnosis:
        diagnosis = ["combination of mild decision-quality and economy-timing issues"]
    lines.append(f"Main remaining issue appears to be: **{', '.join(diagnosis)}**.")
    lines.append("Recommended single follow-up experiment: add a targeted diagnostics slice for 4–6 coin states with legal-action ranking and compare top-1 RL choice vs top affordable money card baseline (analysis-only A/B reporting, no training changes).")

    report_path = diag_dir / "midgame_policy_analysis.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()
