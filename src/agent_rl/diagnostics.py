"""Diagnostics collector for Dominion RL training.

Collects per-episode summaries, per-buy-decision event logs, and rolling
aggregate metrics.  All output goes to CSV files under a run-specific
diagnostics/ directory.  This module is strictly observational -- it does
not modify the training algorithm, reward function, or environment.
"""

import csv
import numpy as np
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional


TRACKED_CARDS = ["Copper", "Silver", "Gold", "Estate", "Duchy", "Province", "Curse"]

EPISODE_SUMMARY_HEADER = [
    "episode", "winner", "did_rl_agent_win",
    "final_score_rl", "final_score_opponent", "final_score_diff",
    "episode_reward_total", "episode_length_turns",
    # Economy / buy power
    "mean_buy_phase_coins_rl", "max_buy_phase_coins_rl", "num_buy_phases_rl",
    "count_buy_phase_coins_ge_3_rl", "count_buy_phase_coins_ge_5_rl",
    "count_buy_phase_coins_ge_6_rl", "count_buy_phase_coins_ge_8_rl",
    # Affordability opportunity counts
    "afford_copper_count_rl", "afford_silver_count_rl", "afford_gold_count_rl",
    "afford_estate_count_rl", "afford_duchy_count_rl", "afford_province_count_rl",
    "afford_curse_count_rl",
    # Actual buy counts
    "buy_copper_count_rl", "buy_silver_count_rl", "buy_gold_count_rl",
    "buy_estate_count_rl", "buy_duchy_count_rl", "buy_province_count_rl",
    "buy_curse_count_rl", "buy_none_count_rl",
    # Conditional conversion counts
    "buy_gold_when_affordable_count_rl", "buy_province_when_affordable_count_rl",
    "buy_duchy_when_province_affordable_count_rl",
    "buy_silver_when_gold_affordable_count_rl",
    "buy_silver_when_province_affordable_count_rl",
    "buy_estate_when_duchy_affordable_count_rl",
    # Turn timing
    "first_turn_reach_5_coins_rl", "first_turn_reach_6_coins_rl",
    "first_turn_reach_8_coins_rl",
    "first_turn_buy_gold_rl", "first_turn_buy_province_rl",
    "first_turn_buy_duchy_rl",
    # Deck composition at episode end (RL)
    "final_count_copper_rl", "final_count_silver_rl", "final_count_gold_rl",
    "final_count_estate_rl", "final_count_duchy_rl", "final_count_province_rl",
    "final_count_curse_rl", "final_deck_size_rl",
    # Opponent summary (optional)
    "mean_buy_phase_coins_opp",
    "final_count_copper_opp", "final_count_silver_opp", "final_count_gold_opp",
    "final_count_estate_opp", "final_count_duchy_opp", "final_count_province_opp",
    "final_deck_size_opp",
]

BUY_DECISION_HEADER = [
    "episode", "turn_index", "buy_decision_index",
    "coins_available", "buys_available",
    "score_rl_current", "score_opp_current", "score_diff_current",
    "provinces_remaining", "duchies_remaining", "estates_remaining",
    "total_deck_size_rl_current",
    "affordable_cards", "chosen_action", "chosen_card",
    "was_gold_affordable", "was_province_affordable", "was_duchy_affordable",
    "was_silver_affordable", "was_curse_affordable",
    "bought_gold", "bought_province", "bought_duchy", "bought_silver",
    "bought_estate", "bought_copper", "bought_curse", "bought_none",
    "epsilon", "q_value_chosen", "top_k_actions_with_q_values",
]

ROLLING_HEADER = [
    "episode",
    "rolling_win_rate", "rolling_mean_score_diff", "rolling_mean_episode_reward",
    "rolling_mean_buy_phase_coins_rl",
    "rolling_afford_gold_rate_rl", "rolling_afford_province_rate_rl",
    "rolling_buy_gold_rate_rl", "rolling_buy_province_rate_rl",
    "rolling_buy_gold_given_affordable_rl",
    "rolling_buy_province_given_affordable_rl",
    "rolling_buy_silver_given_gold_affordable_rl",
    "rolling_buy_silver_given_province_affordable_rl",
    "rolling_first_turn_reach_8_mean_rl",
    "rolling_first_turn_buy_province_mean_rl",
]


def _safe_div(num, den, ndigits=4):
    """Divide with zero-denominator safety; returns empty string when undefined."""
    return round(num / den, ndigits) if den > 0 else ""


def get_player_deck_counts(player) -> Dict[str, int]:
    """Return {card_name: count} for all cards a player owns."""
    counts: Dict[str, int] = {}
    for card in player.get_all_cards():
        counts[card.name] = counts.get(card.name, 0) + 1
    return counts


class DiagnosticsCollector:
    """Accumulates per-episode and per-decision diagnostics, writing to CSV."""

    def __init__(self, run_dir, window_size=200):
        self.diag_dir = Path(run_dir) / "diagnostics"
        self.diag_dir.mkdir(parents=True, exist_ok=True)
        self.window_size = window_size

        self.episode_csv_path = self.diag_dir / "episode_summary.csv"
        self.buy_csv_path = self.diag_dir / "buy_decisions.csv"
        self.rolling_csv_path = self.diag_dir / "rolling_metrics.csv"

        self._init_csv(self.episode_csv_path, EPISODE_SUMMARY_HEADER)
        self._init_csv(self.buy_csv_path, BUY_DECISION_HEADER)
        self._init_csv(self.rolling_csv_path, ROLLING_HEADER)

        self._recent_summaries: deque = deque(maxlen=window_size)

        # Per-episode accumulators (reset each episode)
        self._reset_accumulators(0)

    # ------------------------------------------------------------------ #
    #  CSV helpers                                                        #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _init_csv(path, header):
        if not path.exists():
            with path.open("w", newline="") as f:
                csv.writer(f).writerow(header)

    # ------------------------------------------------------------------ #
    #  Per-episode lifecycle                                              #
    # ------------------------------------------------------------------ #
    def reset_episode(self, episode_id: int):
        """Call at the start of every episode."""
        self._reset_accumulators(episode_id)

    def _reset_accumulators(self, episode_id):
        self._ep_id = episode_id
        self._ep_coins: List[int] = []
        self._ep_buy_decisions: List[Dict[str, Any]] = []
        self._ep_buy_counts = {card: 0 for card in TRACKED_CARDS}
        self._ep_buy_counts["NONE"] = 0
        self._ep_afford_counts = {card: 0 for card in TRACKED_CARDS}
        self._ep_conditional = {
            "buy_gold_when_affordable": 0,
            "buy_province_when_affordable": 0,
            "buy_duchy_when_province_affordable": 0,
            "buy_silver_when_gold_affordable": 0,
            "buy_silver_when_province_affordable": 0,
            "buy_estate_when_duchy_affordable": 0,
        }
        self._ep_first_turn = {
            "reach_5": None, "reach_6": None, "reach_8": None,
            "buy_gold": None, "buy_province": None, "buy_duchy": None,
        }
        self._ep_decision_idx = 0

    # ------------------------------------------------------------------ #
    #  Per-decision recording                                             #
    # ------------------------------------------------------------------ #
    def record_buy_decision(
        self,
        snapshot: Dict[str, Any],
        action_idx: int,
        card_name: str,
        epsilon: Optional[float] = None,
        q_value_chosen: Optional[float] = None,
        top_k_q: Optional[str] = None,
    ):
        """Record one RL buy decision using a pre-step snapshot."""
        coins = snapshot["coins_available"]
        turn = snapshot["turn"]
        affordable = snapshot["affordable"]
        supply = snapshot["supply_remaining"]

        self._ep_coins.append(coins)
        self._ep_decision_idx += 1

        # -- Coin threshold tracking --
        if coins >= 5 and self._ep_first_turn["reach_5"] is None:
            self._ep_first_turn["reach_5"] = turn
        if coins >= 6 and self._ep_first_turn["reach_6"] is None:
            self._ep_first_turn["reach_6"] = turn
        if coins >= 8 and self._ep_first_turn["reach_8"] is None:
            self._ep_first_turn["reach_8"] = turn

        # -- Affordability counts --
        for card in TRACKED_CARDS:
            if affordable.get(card, False):
                self._ep_afford_counts[card] += 1

        # -- Identify what was bought --
        is_none = card_name in ("NONE", "PASS", "ILLEGAL") or card_name is None
        bought_card = None if is_none else card_name

        if is_none:
            self._ep_buy_counts["NONE"] += 1
        elif bought_card in self._ep_buy_counts:
            self._ep_buy_counts[bought_card] += 1

        # -- First-buy timing --
        if bought_card == "Gold" and self._ep_first_turn["buy_gold"] is None:
            self._ep_first_turn["buy_gold"] = turn
        if bought_card == "Province" and self._ep_first_turn["buy_province"] is None:
            self._ep_first_turn["buy_province"] = turn
        if bought_card == "Duchy" and self._ep_first_turn["buy_duchy"] is None:
            self._ep_first_turn["buy_duchy"] = turn

        # -- Conditional buy tracking --
        gold_aff = affordable.get("Gold", False)
        province_aff = affordable.get("Province", False)
        duchy_aff = affordable.get("Duchy", False)

        if gold_aff and bought_card == "Gold":
            self._ep_conditional["buy_gold_when_affordable"] += 1
        if province_aff and bought_card == "Province":
            self._ep_conditional["buy_province_when_affordable"] += 1
        if province_aff and bought_card == "Duchy":
            self._ep_conditional["buy_duchy_when_province_affordable"] += 1
        if gold_aff and bought_card == "Silver":
            self._ep_conditional["buy_silver_when_gold_affordable"] += 1
        if province_aff and bought_card == "Silver":
            self._ep_conditional["buy_silver_when_province_affordable"] += 1
        if duchy_aff and bought_card == "Estate":
            self._ep_conditional["buy_estate_when_duchy_affordable"] += 1

        # -- Build the per-decision CSV row --
        affordable_list = sorted(
            [name for name, aff in affordable.items() if aff]
        )

        row = {
            "episode": self._ep_id,
            "turn_index": turn,
            "buy_decision_index": self._ep_decision_idx,
            "coins_available": coins,
            "buys_available": snapshot["buys_available"],
            "score_rl_current": snapshot["score_rl"],
            "score_opp_current": snapshot["score_opp"],
            "score_diff_current": snapshot["score_diff"],
            "provinces_remaining": supply.get("Province", ""),
            "duchies_remaining": supply.get("Duchy", ""),
            "estates_remaining": supply.get("Estate", ""),
            "total_deck_size_rl_current": snapshot["deck_size_rl"],
            "affordable_cards": ";".join(affordable_list),
            "chosen_action": action_idx,
            "chosen_card": bought_card or "NONE",
            "was_gold_affordable": int(gold_aff),
            "was_province_affordable": int(province_aff),
            "was_duchy_affordable": int(duchy_aff),
            "was_silver_affordable": int(affordable.get("Silver", False)),
            "was_curse_affordable": int(affordable.get("Curse", False)),
            "bought_gold": int(bought_card == "Gold"),
            "bought_province": int(bought_card == "Province"),
            "bought_duchy": int(bought_card == "Duchy"),
            "bought_silver": int(bought_card == "Silver"),
            "bought_estate": int(bought_card == "Estate"),
            "bought_copper": int(bought_card == "Copper"),
            "bought_curse": int(bought_card == "Curse"),
            "bought_none": int(is_none),
            "epsilon": f"{epsilon:.4f}" if epsilon is not None else "",
            "q_value_chosen": f"{q_value_chosen:.4f}" if q_value_chosen is not None else "",
            "top_k_actions_with_q_values": top_k_q or "",
        }
        self._ep_buy_decisions.append(row)

    # ------------------------------------------------------------------ #
    #  Episode finalization                                               #
    # ------------------------------------------------------------------ #
    def finalize_episode(
        self,
        did_win: bool,
        rl_score: int,
        opp_score: int,
        ep_reward: float,
        ep_length: int,
        rl_deck_counts: Optional[Dict[str, int]] = None,
        opp_deck_counts: Optional[Dict[str, int]] = None,
    ):
        """Compute summary, write episode + decision CSVs, update rolling metrics."""
        coins = self._ep_coins
        num_buys = len(coins)

        if did_win:
            winner = "RL"
        elif rl_score == opp_score:
            winner = "Tie"
        else:
            winner = "Opponent"

        rl_deck = rl_deck_counts or {}
        opp_deck = opp_deck_counts or {}

        summary = {
            "episode": self._ep_id,
            "winner": winner,
            "did_rl_agent_win": int(did_win),
            "final_score_rl": rl_score,
            "final_score_opponent": opp_score,
            "final_score_diff": rl_score - opp_score,
            "episode_reward_total": round(ep_reward, 4),
            "episode_length_turns": ep_length,
            # Economy
            "mean_buy_phase_coins_rl": round(float(np.mean(coins)), 2) if coins else 0,
            "max_buy_phase_coins_rl": max(coins) if coins else 0,
            "num_buy_phases_rl": num_buys,
            "count_buy_phase_coins_ge_3_rl": sum(1 for c in coins if c >= 3),
            "count_buy_phase_coins_ge_5_rl": sum(1 for c in coins if c >= 5),
            "count_buy_phase_coins_ge_6_rl": sum(1 for c in coins if c >= 6),
            "count_buy_phase_coins_ge_8_rl": sum(1 for c in coins if c >= 8),
            # Affordability
            "afford_copper_count_rl": self._ep_afford_counts["Copper"],
            "afford_silver_count_rl": self._ep_afford_counts["Silver"],
            "afford_gold_count_rl": self._ep_afford_counts["Gold"],
            "afford_estate_count_rl": self._ep_afford_counts["Estate"],
            "afford_duchy_count_rl": self._ep_afford_counts["Duchy"],
            "afford_province_count_rl": self._ep_afford_counts["Province"],
            "afford_curse_count_rl": self._ep_afford_counts["Curse"],
            # Buy counts
            "buy_copper_count_rl": self._ep_buy_counts["Copper"],
            "buy_silver_count_rl": self._ep_buy_counts["Silver"],
            "buy_gold_count_rl": self._ep_buy_counts["Gold"],
            "buy_estate_count_rl": self._ep_buy_counts["Estate"],
            "buy_duchy_count_rl": self._ep_buy_counts["Duchy"],
            "buy_province_count_rl": self._ep_buy_counts["Province"],
            "buy_curse_count_rl": self._ep_buy_counts["Curse"],
            "buy_none_count_rl": self._ep_buy_counts["NONE"],
            # Conditional
            "buy_gold_when_affordable_count_rl": self._ep_conditional["buy_gold_when_affordable"],
            "buy_province_when_affordable_count_rl": self._ep_conditional["buy_province_when_affordable"],
            "buy_duchy_when_province_affordable_count_rl": self._ep_conditional["buy_duchy_when_province_affordable"],
            "buy_silver_when_gold_affordable_count_rl": self._ep_conditional["buy_silver_when_gold_affordable"],
            "buy_silver_when_province_affordable_count_rl": self._ep_conditional["buy_silver_when_province_affordable"],
            "buy_estate_when_duchy_affordable_count_rl": self._ep_conditional["buy_estate_when_duchy_affordable"],
            # Turn timing (empty string for never-reached)
            "first_turn_reach_5_coins_rl": self._ep_first_turn["reach_5"] if self._ep_first_turn["reach_5"] is not None else "",
            "first_turn_reach_6_coins_rl": self._ep_first_turn["reach_6"] if self._ep_first_turn["reach_6"] is not None else "",
            "first_turn_reach_8_coins_rl": self._ep_first_turn["reach_8"] if self._ep_first_turn["reach_8"] is not None else "",
            "first_turn_buy_gold_rl": self._ep_first_turn["buy_gold"] if self._ep_first_turn["buy_gold"] is not None else "",
            "first_turn_buy_province_rl": self._ep_first_turn["buy_province"] if self._ep_first_turn["buy_province"] is not None else "",
            "first_turn_buy_duchy_rl": self._ep_first_turn["buy_duchy"] if self._ep_first_turn["buy_duchy"] is not None else "",
            # Final deck (RL)
            "final_count_copper_rl": rl_deck.get("Copper", 0),
            "final_count_silver_rl": rl_deck.get("Silver", 0),
            "final_count_gold_rl": rl_deck.get("Gold", 0),
            "final_count_estate_rl": rl_deck.get("Estate", 0),
            "final_count_duchy_rl": rl_deck.get("Duchy", 0),
            "final_count_province_rl": rl_deck.get("Province", 0),
            "final_count_curse_rl": rl_deck.get("Curse", 0),
            "final_deck_size_rl": sum(rl_deck.values()),
            # Opponent (optional)
            "mean_buy_phase_coins_opp": "",
            "final_count_copper_opp": opp_deck.get("Copper", 0),
            "final_count_silver_opp": opp_deck.get("Silver", 0),
            "final_count_gold_opp": opp_deck.get("Gold", 0),
            "final_count_estate_opp": opp_deck.get("Estate", 0),
            "final_count_duchy_opp": opp_deck.get("Duchy", 0),
            "final_count_province_opp": opp_deck.get("Province", 0),
            "final_deck_size_opp": sum(opp_deck.values()),
        }

        # -- Write episode summary row --
        with self.episode_csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=EPISODE_SUMMARY_HEADER)
            writer.writerow(summary)

        # -- Write buy decision rows --
        if self._ep_buy_decisions:
            with self.buy_csv_path.open("a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=BUY_DECISION_HEADER)
                for row in self._ep_buy_decisions:
                    writer.writerow(row)

        # -- Update rolling window and write rolling metrics --
        self._recent_summaries.append(summary)
        if len(self._recent_summaries) >= 10:
            self._write_rolling_metrics()

    # ------------------------------------------------------------------ #
    #  Rolling aggregate metrics                                          #
    # ------------------------------------------------------------------ #
    def _write_rolling_metrics(self):
        summaries = list(self._recent_summaries)
        n = len(summaries)

        total_buys = sum(s["num_buy_phases_rl"] for s in summaries)

        total_afford_gold = sum(s["afford_gold_count_rl"] for s in summaries)
        total_afford_province = sum(s["afford_province_count_rl"] for s in summaries)
        total_buy_gold = sum(s["buy_gold_count_rl"] for s in summaries)
        total_buy_province = sum(s["buy_province_count_rl"] for s in summaries)
        total_buy_gold_when_aff = sum(s["buy_gold_when_affordable_count_rl"] for s in summaries)
        total_buy_prov_when_aff = sum(s["buy_province_when_affordable_count_rl"] for s in summaries)
        total_buy_silver_gold_aff = sum(s["buy_silver_when_gold_affordable_count_rl"] for s in summaries)
        total_buy_silver_prov_aff = sum(s["buy_silver_when_province_affordable_count_rl"] for s in summaries)

        reach_8_vals = [
            s["first_turn_reach_8_coins_rl"] for s in summaries
            if s["first_turn_reach_8_coins_rl"] != ""
        ]
        reach_8_mean = round(float(np.mean(reach_8_vals)), 2) if reach_8_vals else ""

        buy_prov_vals = [
            s["first_turn_buy_province_rl"] for s in summaries
            if s["first_turn_buy_province_rl"] != ""
        ]
        buy_prov_mean = round(float(np.mean(buy_prov_vals)), 2) if buy_prov_vals else ""

        row = {
            "episode": summaries[-1]["episode"],
            "rolling_win_rate": _safe_div(
                sum(s["did_rl_agent_win"] for s in summaries), n
            ),
            "rolling_mean_score_diff": round(
                float(np.mean([s["final_score_diff"] for s in summaries])), 2
            ),
            "rolling_mean_episode_reward": round(
                float(np.mean([s["episode_reward_total"] for s in summaries])), 2
            ),
            "rolling_mean_buy_phase_coins_rl": round(
                float(np.mean([s["mean_buy_phase_coins_rl"] for s in summaries])), 2
            ),
            "rolling_afford_gold_rate_rl": _safe_div(total_afford_gold, total_buys),
            "rolling_afford_province_rate_rl": _safe_div(total_afford_province, total_buys),
            "rolling_buy_gold_rate_rl": _safe_div(total_buy_gold, total_buys),
            "rolling_buy_province_rate_rl": _safe_div(total_buy_province, total_buys),
            "rolling_buy_gold_given_affordable_rl": _safe_div(total_buy_gold_when_aff, total_afford_gold),
            "rolling_buy_province_given_affordable_rl": _safe_div(total_buy_prov_when_aff, total_afford_province),
            "rolling_buy_silver_given_gold_affordable_rl": _safe_div(total_buy_silver_gold_aff, total_afford_gold),
            "rolling_buy_silver_given_province_affordable_rl": _safe_div(total_buy_silver_prov_aff, total_afford_province),
            "rolling_first_turn_reach_8_mean_rl": reach_8_mean,
            "rolling_first_turn_buy_province_mean_rl": buy_prov_mean,
        }

        with self.rolling_csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ROLLING_HEADER)
            writer.writerow(row)
