"""Diagnostics collector for Dominion RL training.

Collects per-episode summaries, per-buy-decision event logs, per-coin-bucket
analysis, and rolling aggregate metrics. All output goes to CSV files under a
run-specific diagnostics/ directory. This module is strictly observational --
it does not modify the training algorithm, reward function, or environment.
"""

import csv
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


TRACKED_CARDS = ["Copper", "Silver", "Gold", "Estate", "Duchy", "Province", "Curse"]
GREEN_CARDS = {"Estate", "Duchy", "Province"}
TREASURE_CARDS = {"Copper", "Silver", "Gold"}
COIN_BUCKETS = ["2", "3", "4", "5", "6", "7", "8+"]

EPISODE_SUMMARY_HEADER = [
    "episode", "winner", "did_rl_agent_win",
    "final_score_rl", "final_score_opponent", "final_score_diff",
    "episode_reward_total", "episode_length_turns",
    # Economy / buy power
    "mean_buy_phase_coins_rl", "max_buy_phase_coins_rl", "num_buy_phases_rl",
    "count_buy_phase_coins_ge_3_rl", "count_buy_phase_coins_ge_5_rl",
    "count_buy_phase_coins_ge_6_rl", "count_buy_phase_coins_ge_8_rl",
    "first_turn_reach_5_coins_rl", "first_turn_reach_6_coins_rl", "first_turn_reach_8_coins_rl",
    # Purchase milestones
    "first_turn_buy_silver_rl", "first_turn_buy_gold_rl", "first_turn_buy_green_rl",
    "first_turn_buy_duchy_rl", "first_turn_buy_province_rl",
    # Greening / tempo
    "treasure_buys_before_first_green_rl",
    "green_buys_before_first_gold_rl", "green_buys_before_first_8_coin_turn_rl",
    "opponent_first_turn_buy_province", "rl_minus_opp_first_province_turn",
    # Affordability opportunity counts
    "afford_copper_count_rl", "afford_silver_count_rl", "afford_gold_count_rl",
    "afford_estate_count_rl", "afford_duchy_count_rl", "afford_province_count_rl",
    "afford_curse_count_rl",
    # Actual buy counts
    "buy_copper_count_rl", "buy_silver_count_rl", "buy_gold_count_rl",
    "buy_estate_count_rl", "buy_duchy_count_rl", "buy_province_count_rl",
    "buy_curse_count_rl", "buy_none_count_rl",
    # Conditional conversion counts
    "buy_silver_when_affordable_count_rl",
    "buy_gold_when_affordable_count_rl", "buy_province_when_affordable_count_rl",
    "buy_duchy_when_duchy_affordable_no_province_count_rl",
    "buy_silver_when_gold_affordable_count_rl",
    "buy_something_else_when_province_affordable_count_rl",
    "pass_when_gold_affordable_count_rl", "pass_when_province_affordable_count_rl",
    # Deck composition at episode end (RL)
    "final_count_copper_rl", "final_count_silver_rl", "final_count_gold_rl",
    "final_count_estate_rl", "final_count_duchy_rl", "final_count_province_rl",
    "final_count_curse_rl", "final_deck_size_rl",
    # Deck quality at episode end (RL)
    "final_treasure_count_rl", "final_green_count_rl",
    "final_treasure_density_rl", "final_green_density_rl", "final_treasure_to_green_ratio_rl",
    "final_nominal_treasure_value_rl", "final_vp_per_card_rl",
    # Opponent summary (optional)
    "mean_buy_phase_coins_opp",
    "final_count_copper_opp", "final_count_silver_opp", "final_count_gold_opp",
    "final_count_estate_opp", "final_count_duchy_opp", "final_count_province_opp",
    "final_deck_size_opp",
    # Training-process episode aggregates
    "episode_loss_mean", "episode_td_error_abs_mean", "episode_q_value_mean", "episode_q_value_max",
    "episode_epsilon_mean", "episode_random_action_rate", "episode_greedy_action_rate",
]

BUY_DECISION_HEADER = [
    "episode", "turn_index", "buy_decision_index",
    "coins_available", "coin_bucket", "buys_available",
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
    "rolling_rate_coins_ge_6_rl", "rolling_rate_coins_ge_8_rl",
    "rolling_fraction_episodes_reach_8_coins",
    "rolling_fraction_episodes_with_province_buy",
    "rolling_afford_gold_rate_rl", "rolling_afford_province_rate_rl", "rolling_afford_silver_rate_rl",
    "rolling_buy_gold_rate_rl", "rolling_buy_province_rate_rl",
    "rolling_buy_gold_given_affordable_rl",
    "rolling_buy_province_given_affordable_rl",
    "rolling_buy_silver_given_affordable_rl",
    "rolling_buy_duchy_given_duchy_affordable_no_province_rl",
    "rolling_buy_silver_given_gold_affordable_rl",
    "rolling_buy_something_else_given_province_affordable_rl",
    "rolling_pass_given_gold_affordable_rl",
    "rolling_pass_given_province_affordable_rl",
    "rolling_first_turn_reach_8_mean_rl",
    "rolling_first_turn_buy_province_mean_rl",
    "rolling_rl_minus_opp_first_province_turn_mean",
    "rolling_episode_loss_mean", "rolling_episode_td_error_abs_mean",
    "rolling_episode_q_value_mean", "rolling_episode_q_value_max",
    "rolling_episode_epsilon_mean", "rolling_random_action_rate", "rolling_greedy_action_rate",
]

COIN_BUCKET_HEADER = [
    "episode", "coin_bucket", "total_decisions", "pass_rate",
    "buy_copper_rate", "buy_silver_rate", "buy_gold_rate",
    "buy_estate_rate", "buy_duchy_rate", "buy_province_rate", "buy_curse_rate",
    "buy_other_rate",
]

TRAINING_STEP_HEADER = [
    "episode", "global_step", "loss", "td_error_mean", "td_error_abs_mean", "td_error_max",
    "q_sa_mean", "q_sa_max", "target_mean", "epsilon",
]


def _safe_div(num, den, ndigits=4):
    return round(num / den, ndigits) if den > 0 else ""


def _safe_ratio(num, den):
    return (num / den) if den > 0 else None


def _coin_bucket(coins: int) -> str:
    if coins >= 8:
        return "8+"
    if coins <= 2:
        return "2"
    return str(int(coins))


def get_player_deck_counts(player) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for card in player.get_all_cards():
        counts[card.name] = counts.get(card.name, 0) + 1
    return counts


class DiagnosticsCollector:
    """Accumulates per-episode and per-decision diagnostics, writing to CSV."""

    def __init__(self, run_dir, window_size=200):
        self.diag_dir = Path(run_dir) / "diagnostics"
        self.results_dir = self.diag_dir / "results"
        self.diag_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.window_size = window_size

        self.episode_csv_path = self.diag_dir / "episode_summary.csv"
        self.buy_csv_path = self.diag_dir / "buy_decisions.csv"
        self.rolling_csv_path = self.diag_dir / "rolling_metrics.csv"
        self.coin_bucket_csv_path = self.diag_dir / "coin_bucket_summary.csv"
        self.training_step_csv_path = self.diag_dir / "training_step_metrics.csv"

        self._init_csv(self.episode_csv_path, EPISODE_SUMMARY_HEADER)
        self._init_csv(self.buy_csv_path, BUY_DECISION_HEADER)
        self._init_csv(self.rolling_csv_path, ROLLING_HEADER)
        self._init_csv(self.coin_bucket_csv_path, COIN_BUCKET_HEADER)
        self._init_csv(self.training_step_csv_path, TRAINING_STEP_HEADER)

        self._recent_summaries: deque = deque(maxlen=window_size)
        self._reset_accumulators(0)

    @staticmethod
    def _init_csv(path, header):
        if not path.exists():
            with path.open("w", newline="") as f:
                csv.writer(f).writerow(header)

    def reset_episode(self, episode_id: int):
        self._reset_accumulators(episode_id)

    def _reset_accumulators(self, episode_id):
        self._ep_id = episode_id
        self._ep_coins: List[int] = []
        self._ep_buy_decisions: List[Dict[str, Any]] = []
        self._ep_training_steps: List[Dict[str, Any]] = []
        self._ep_random_actions = 0
        self._ep_greedy_actions = 0
        self._ep_buy_counts = {card: 0 for card in TRACKED_CARDS}
        self._ep_buy_counts["NONE"] = 0
        self._ep_afford_counts = {card: 0 for card in TRACKED_CARDS}
        self._ep_conditional = {
            "buy_silver_when_affordable": 0,
            "buy_gold_when_affordable": 0,
            "buy_province_when_affordable": 0,
            "buy_duchy_when_duchy_affordable_no_province": 0,
            "buy_silver_when_gold_affordable": 0,
            "buy_something_else_when_province_affordable": 0,
            "pass_when_gold_affordable": 0,
            "pass_when_province_affordable": 0,
        }
        self._ep_first_turn = {
            "reach_5": None, "reach_6": None, "reach_8": None,
            "buy_silver": None, "buy_gold": None, "buy_green": None,
            "buy_province": None, "buy_duchy": None,
        }
        self._ep_treasure_buys_before_first_green = 0
        self._ep_green_buys_before_first_gold = 0
        self._ep_green_buys_before_first_8_coin = 0
        self._ep_decision_idx = 0

    def record_buy_decision(
        self,
        snapshot: Dict[str, Any],
        action_idx: int,
        card_name: str,
        was_random: bool,
        epsilon: Optional[float] = None,
        q_value_chosen: Optional[float] = None,
        top_k_q: Optional[str] = None,
    ):
        coins = int(snapshot["coins_available"])
        turn = snapshot["turn"]
        affordable = snapshot["affordable"]
        supply = snapshot["supply_remaining"]

        self._ep_coins.append(coins)
        self._ep_decision_idx += 1
        if was_random:
            self._ep_random_actions += 1
        else:
            self._ep_greedy_actions += 1

        if coins >= 5 and self._ep_first_turn["reach_5"] is None:
            self._ep_first_turn["reach_5"] = turn
        if coins >= 6 and self._ep_first_turn["reach_6"] is None:
            self._ep_first_turn["reach_6"] = turn
        if coins >= 8 and self._ep_first_turn["reach_8"] is None:
            self._ep_first_turn["reach_8"] = turn

        for card in TRACKED_CARDS:
            if affordable.get(card, False):
                self._ep_afford_counts[card] += 1

        is_none = card_name in ("NONE", "PASS", "ILLEGAL") or card_name is None
        bought_card = None if is_none else card_name

        if is_none:
            self._ep_buy_counts["NONE"] += 1
        elif bought_card in self._ep_buy_counts:
            self._ep_buy_counts[bought_card] += 1

        if bought_card == "Silver" and self._ep_first_turn["buy_silver"] is None:
            self._ep_first_turn["buy_silver"] = turn
        if bought_card == "Gold" and self._ep_first_turn["buy_gold"] is None:
            self._ep_first_turn["buy_gold"] = turn
        if bought_card in GREEN_CARDS and self._ep_first_turn["buy_green"] is None:
            self._ep_first_turn["buy_green"] = turn
        if bought_card == "Province" and self._ep_first_turn["buy_province"] is None:
            self._ep_first_turn["buy_province"] = turn
        if bought_card == "Duchy" and self._ep_first_turn["buy_duchy"] is None:
            self._ep_first_turn["buy_duchy"] = turn

        if self._ep_first_turn["buy_green"] is None and bought_card in TREASURE_CARDS:
            self._ep_treasure_buys_before_first_green += 1
        if self._ep_first_turn["buy_gold"] is None and bought_card in GREEN_CARDS:
            self._ep_green_buys_before_first_gold += 1
        if self._ep_first_turn["reach_8"] is None and bought_card in GREEN_CARDS:
            self._ep_green_buys_before_first_8_coin += 1

        gold_aff = affordable.get("Gold", False)
        province_aff = affordable.get("Province", False)
        duchy_aff = affordable.get("Duchy", False)
        silver_aff = affordable.get("Silver", False)

        if silver_aff and bought_card == "Silver":
            self._ep_conditional["buy_silver_when_affordable"] += 1
        if gold_aff and bought_card == "Gold":
            self._ep_conditional["buy_gold_when_affordable"] += 1
        if province_aff and bought_card == "Province":
            self._ep_conditional["buy_province_when_affordable"] += 1
        if duchy_aff and not province_aff and bought_card == "Duchy":
            self._ep_conditional["buy_duchy_when_duchy_affordable_no_province"] += 1
        if gold_aff and bought_card == "Silver":
            self._ep_conditional["buy_silver_when_gold_affordable"] += 1
        if province_aff and (bought_card not in (None, "Province")):
            self._ep_conditional["buy_something_else_when_province_affordable"] += 1
        if gold_aff and is_none:
            self._ep_conditional["pass_when_gold_affordable"] += 1
        if province_aff and is_none:
            self._ep_conditional["pass_when_province_affordable"] += 1

        affordable_list = sorted([name for name, aff in affordable.items() if aff])

        self._ep_buy_decisions.append({
            "episode": self._ep_id,
            "turn_index": turn,
            "buy_decision_index": self._ep_decision_idx,
            "coins_available": coins,
            "coin_bucket": _coin_bucket(coins),
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
            "was_silver_affordable": int(silver_aff),
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
        })

    def record_training_step(
        self,
        global_step: int,
        loss: float,
        td_error: np.ndarray,
        q_sa: np.ndarray,
        target: np.ndarray,
        epsilon: float,
    ):
        td_arr = np.asarray(td_error, dtype=np.float32)
        q_arr = np.asarray(q_sa, dtype=np.float32)
        tgt_arr = np.asarray(target, dtype=np.float32)
        row = {
            "episode": self._ep_id,
            "global_step": int(global_step),
            "loss": round(float(loss), 6),
            "td_error_mean": round(float(td_arr.mean()), 6),
            "td_error_abs_mean": round(float(np.abs(td_arr).mean()), 6),
            "td_error_max": round(float(np.max(np.abs(td_arr))), 6),
            "q_sa_mean": round(float(q_arr.mean()), 6),
            "q_sa_max": round(float(q_arr.max()), 6),
            "target_mean": round(float(tgt_arr.mean()), 6),
            "epsilon": round(float(epsilon), 6),
        }
        self._ep_training_steps.append(row)
        with self.training_step_csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=TRAINING_STEP_HEADER)
            writer.writerow(row)

    def finalize_episode(
        self,
        did_win: bool,
        rl_score: int,
        opp_score: int,
        ep_reward: float,
        ep_length: int,
        rl_deck_counts: Optional[Dict[str, int]] = None,
        opp_deck_counts: Optional[Dict[str, int]] = None,
        turn_events: Optional[List[Dict[str, Any]]] = None,
        rl_player_id: str = "RL",
    ):
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
        deck_size = int(sum(rl_deck.values()))
        treasure_count = sum(rl_deck.get(card, 0) for card in TREASURE_CARDS)
        green_count = sum(rl_deck.get(card, 0) for card in GREEN_CARDS)
        treasure_density = _safe_ratio(treasure_count, deck_size)
        green_density = _safe_ratio(green_count, deck_size)
        treasure_to_green = _safe_ratio(treasure_count, green_count)
        nominal_treasure_value = (
            rl_deck.get("Copper", 0)
            + 2 * rl_deck.get("Silver", 0)
            + 3 * rl_deck.get("Gold", 0)
        )
        nominal_vp = (
            rl_deck.get("Estate", 0)
            + 3 * rl_deck.get("Duchy", 0)
            + 6 * rl_deck.get("Province", 0)
            - rl_deck.get("Curse", 0)
        )

        first_opp_prov = ""
        if turn_events:
            opp_prov_turns = [
                int(event.get("turn", 0))
                for event in turn_events
                if "Province" in event.get("buys", []) and event.get("player_id") != rl_player_id
            ]
            if opp_prov_turns:
                first_opp_prov = min(opp_prov_turns)

        first_rl_prov = self._ep_first_turn["buy_province"]
        rl_minus_opp = ""
        if first_rl_prov is not None and first_opp_prov != "":
            rl_minus_opp = first_rl_prov - first_opp_prov

        train_rows = self._ep_training_steps
        loss_mean = float(np.mean([r["loss"] for r in train_rows])) if train_rows else None
        td_abs_mean = float(np.mean([r["td_error_abs_mean"] for r in train_rows])) if train_rows else None
        q_mean = float(np.mean([r["q_sa_mean"] for r in train_rows])) if train_rows else None
        q_max = float(np.max([r["q_sa_max"] for r in train_rows])) if train_rows else None
        eps_mean = float(np.mean([r["epsilon"] for r in train_rows])) if train_rows else None

        summary = {
            "episode": self._ep_id,
            "winner": winner,
            "did_rl_agent_win": int(did_win),
            "final_score_rl": rl_score,
            "final_score_opponent": opp_score,
            "final_score_diff": rl_score - opp_score,
            "episode_reward_total": round(ep_reward, 4),
            "episode_length_turns": ep_length,
            "mean_buy_phase_coins_rl": round(float(np.mean(coins)), 2) if coins else 0,
            "max_buy_phase_coins_rl": max(coins) if coins else 0,
            "num_buy_phases_rl": num_buys,
            "count_buy_phase_coins_ge_3_rl": sum(1 for c in coins if c >= 3),
            "count_buy_phase_coins_ge_5_rl": sum(1 for c in coins if c >= 5),
            "count_buy_phase_coins_ge_6_rl": sum(1 for c in coins if c >= 6),
            "count_buy_phase_coins_ge_8_rl": sum(1 for c in coins if c >= 8),
            "first_turn_reach_5_coins_rl": self._ep_first_turn["reach_5"] if self._ep_first_turn["reach_5"] is not None else "",
            "first_turn_reach_6_coins_rl": self._ep_first_turn["reach_6"] if self._ep_first_turn["reach_6"] is not None else "",
            "first_turn_reach_8_coins_rl": self._ep_first_turn["reach_8"] if self._ep_first_turn["reach_8"] is not None else "",
            "first_turn_buy_silver_rl": self._ep_first_turn["buy_silver"] if self._ep_first_turn["buy_silver"] is not None else "",
            "first_turn_buy_gold_rl": self._ep_first_turn["buy_gold"] if self._ep_first_turn["buy_gold"] is not None else "",
            "first_turn_buy_green_rl": self._ep_first_turn["buy_green"] if self._ep_first_turn["buy_green"] is not None else "",
            "first_turn_buy_duchy_rl": self._ep_first_turn["buy_duchy"] if self._ep_first_turn["buy_duchy"] is not None else "",
            "first_turn_buy_province_rl": self._ep_first_turn["buy_province"] if self._ep_first_turn["buy_province"] is not None else "",
            "treasure_buys_before_first_green_rl": self._ep_treasure_buys_before_first_green,
            "green_buys_before_first_gold_rl": self._ep_green_buys_before_first_gold,
            "green_buys_before_first_8_coin_turn_rl": self._ep_green_buys_before_first_8_coin,
            "opponent_first_turn_buy_province": first_opp_prov,
            "rl_minus_opp_first_province_turn": rl_minus_opp,
            "afford_copper_count_rl": self._ep_afford_counts["Copper"],
            "afford_silver_count_rl": self._ep_afford_counts["Silver"],
            "afford_gold_count_rl": self._ep_afford_counts["Gold"],
            "afford_estate_count_rl": self._ep_afford_counts["Estate"],
            "afford_duchy_count_rl": self._ep_afford_counts["Duchy"],
            "afford_province_count_rl": self._ep_afford_counts["Province"],
            "afford_curse_count_rl": self._ep_afford_counts["Curse"],
            "buy_copper_count_rl": self._ep_buy_counts["Copper"],
            "buy_silver_count_rl": self._ep_buy_counts["Silver"],
            "buy_gold_count_rl": self._ep_buy_counts["Gold"],
            "buy_estate_count_rl": self._ep_buy_counts["Estate"],
            "buy_duchy_count_rl": self._ep_buy_counts["Duchy"],
            "buy_province_count_rl": self._ep_buy_counts["Province"],
            "buy_curse_count_rl": self._ep_buy_counts["Curse"],
            "buy_none_count_rl": self._ep_buy_counts["NONE"],
            "buy_silver_when_affordable_count_rl": self._ep_conditional["buy_silver_when_affordable"],
            "buy_gold_when_affordable_count_rl": self._ep_conditional["buy_gold_when_affordable"],
            "buy_province_when_affordable_count_rl": self._ep_conditional["buy_province_when_affordable"],
            "buy_duchy_when_duchy_affordable_no_province_count_rl": self._ep_conditional["buy_duchy_when_duchy_affordable_no_province"],
            "buy_silver_when_gold_affordable_count_rl": self._ep_conditional["buy_silver_when_gold_affordable"],
            "buy_something_else_when_province_affordable_count_rl": self._ep_conditional["buy_something_else_when_province_affordable"],
            "pass_when_gold_affordable_count_rl": self._ep_conditional["pass_when_gold_affordable"],
            "pass_when_province_affordable_count_rl": self._ep_conditional["pass_when_province_affordable"],
            "final_count_copper_rl": rl_deck.get("Copper", 0),
            "final_count_silver_rl": rl_deck.get("Silver", 0),
            "final_count_gold_rl": rl_deck.get("Gold", 0),
            "final_count_estate_rl": rl_deck.get("Estate", 0),
            "final_count_duchy_rl": rl_deck.get("Duchy", 0),
            "final_count_province_rl": rl_deck.get("Province", 0),
            "final_count_curse_rl": rl_deck.get("Curse", 0),
            "final_deck_size_rl": deck_size,
            "final_treasure_count_rl": treasure_count,
            "final_green_count_rl": green_count,
            "final_treasure_density_rl": round(treasure_density, 4) if treasure_density is not None else "",
            "final_green_density_rl": round(green_density, 4) if green_density is not None else "",
            "final_treasure_to_green_ratio_rl": round(treasure_to_green, 4) if treasure_to_green is not None else "",
            "final_nominal_treasure_value_rl": nominal_treasure_value,
            "final_vp_per_card_rl": round(nominal_vp / deck_size, 4) if deck_size > 0 else "",
            "mean_buy_phase_coins_opp": "",
            "final_count_copper_opp": opp_deck.get("Copper", 0),
            "final_count_silver_opp": opp_deck.get("Silver", 0),
            "final_count_gold_opp": opp_deck.get("Gold", 0),
            "final_count_estate_opp": opp_deck.get("Estate", 0),
            "final_count_duchy_opp": opp_deck.get("Duchy", 0),
            "final_count_province_opp": opp_deck.get("Province", 0),
            "final_deck_size_opp": sum(opp_deck.values()),
            "episode_loss_mean": round(loss_mean, 6) if loss_mean is not None else "",
            "episode_td_error_abs_mean": round(td_abs_mean, 6) if td_abs_mean is not None else "",
            "episode_q_value_mean": round(q_mean, 6) if q_mean is not None else "",
            "episode_q_value_max": round(q_max, 6) if q_max is not None else "",
            "episode_epsilon_mean": round(eps_mean, 6) if eps_mean is not None else "",
            "episode_random_action_rate": _safe_div(self._ep_random_actions, num_buys),
            "episode_greedy_action_rate": _safe_div(self._ep_greedy_actions, num_buys),
        }

        with self.episode_csv_path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=EPISODE_SUMMARY_HEADER).writerow(summary)

        if self._ep_buy_decisions:
            with self.buy_csv_path.open("a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=BUY_DECISION_HEADER)
                for row in self._ep_buy_decisions:
                    writer.writerow(row)

        self._write_coin_bucket_rows()
        self._recent_summaries.append(summary)
        self._write_rolling_metrics()

    def _write_coin_bucket_rows(self):
        decisions = self._ep_buy_decisions
        if not decisions:
            return
        with self.coin_bucket_csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COIN_BUCKET_HEADER)
            for bucket in COIN_BUCKETS:
                rows = [r for r in decisions if r["coin_bucket"] == bucket]
                total = len(rows)
                if total == 0:
                    continue
                card_counts = {}
                for r in rows:
                    card = r["chosen_card"]
                    card_counts[card] = card_counts.get(card, 0) + 1
                known_total = sum(card_counts.get(c, 0) for c in TRACKED_CARDS)
                writer.writerow({
                    "episode": self._ep_id,
                    "coin_bucket": bucket,
                    "total_decisions": total,
                    "pass_rate": _safe_div(card_counts.get("NONE", 0), total),
                    "buy_copper_rate": _safe_div(card_counts.get("Copper", 0), total),
                    "buy_silver_rate": _safe_div(card_counts.get("Silver", 0), total),
                    "buy_gold_rate": _safe_div(card_counts.get("Gold", 0), total),
                    "buy_estate_rate": _safe_div(card_counts.get("Estate", 0), total),
                    "buy_duchy_rate": _safe_div(card_counts.get("Duchy", 0), total),
                    "buy_province_rate": _safe_div(card_counts.get("Province", 0), total),
                    "buy_curse_rate": _safe_div(card_counts.get("Curse", 0), total),
                    "buy_other_rate": _safe_div(total - card_counts.get("NONE", 0) - known_total, total),
                })

    def _write_rolling_metrics(self):
        summaries = list(self._recent_summaries)
        n = len(summaries)
        total_buys = sum(s["num_buy_phases_rl"] for s in summaries)

        total_afford_gold = sum(s["afford_gold_count_rl"] for s in summaries)
        total_afford_province = sum(s["afford_province_count_rl"] for s in summaries)
        total_afford_silver = sum(s["afford_silver_count_rl"] for s in summaries)
        total_afford_duchy_no_prov = sum(max(0, s["afford_duchy_count_rl"] - s["afford_province_count_rl"]) for s in summaries)
        total_buy_gold = sum(s["buy_gold_count_rl"] for s in summaries)
        total_buy_province = sum(s["buy_province_count_rl"] for s in summaries)

        reach_8_vals = [s["first_turn_reach_8_coins_rl"] for s in summaries if s["first_turn_reach_8_coins_rl"] != ""]
        buy_prov_vals = [s["first_turn_buy_province_rl"] for s in summaries if s["first_turn_buy_province_rl"] != ""]
        rl_minus_opp_vals = [s["rl_minus_opp_first_province_turn"] for s in summaries if s["rl_minus_opp_first_province_turn"] != ""]

        row = {
            "episode": summaries[-1]["episode"],
            "rolling_win_rate": _safe_div(sum(s["did_rl_agent_win"] for s in summaries), n),
            "rolling_mean_score_diff": round(float(np.mean([s["final_score_diff"] for s in summaries])), 2),
            "rolling_mean_episode_reward": round(float(np.mean([s["episode_reward_total"] for s in summaries])), 2),
            "rolling_mean_buy_phase_coins_rl": round(float(np.mean([s["mean_buy_phase_coins_rl"] for s in summaries])), 2),
            "rolling_rate_coins_ge_6_rl": _safe_div(sum(s["count_buy_phase_coins_ge_6_rl"] for s in summaries), total_buys),
            "rolling_rate_coins_ge_8_rl": _safe_div(sum(s["count_buy_phase_coins_ge_8_rl"] for s in summaries), total_buys),
            "rolling_fraction_episodes_reach_8_coins": _safe_div(sum(int(s["first_turn_reach_8_coins_rl"] != "") for s in summaries), n),
            "rolling_fraction_episodes_with_province_buy": _safe_div(sum(int(s["buy_province_count_rl"] > 0) for s in summaries), n),
            "rolling_afford_gold_rate_rl": _safe_div(total_afford_gold, total_buys),
            "rolling_afford_province_rate_rl": _safe_div(total_afford_province, total_buys),
            "rolling_afford_silver_rate_rl": _safe_div(total_afford_silver, total_buys),
            "rolling_buy_gold_rate_rl": _safe_div(total_buy_gold, total_buys),
            "rolling_buy_province_rate_rl": _safe_div(total_buy_province, total_buys),
            "rolling_buy_gold_given_affordable_rl": _safe_div(sum(s["buy_gold_when_affordable_count_rl"] for s in summaries), total_afford_gold),
            "rolling_buy_province_given_affordable_rl": _safe_div(sum(s["buy_province_when_affordable_count_rl"] for s in summaries), total_afford_province),
            "rolling_buy_silver_given_affordable_rl": _safe_div(sum(s["buy_silver_when_affordable_count_rl"] for s in summaries), total_afford_silver),
            "rolling_buy_duchy_given_duchy_affordable_no_province_rl": _safe_div(sum(s["buy_duchy_when_duchy_affordable_no_province_count_rl"] for s in summaries), total_afford_duchy_no_prov),
            "rolling_buy_silver_given_gold_affordable_rl": _safe_div(sum(s["buy_silver_when_gold_affordable_count_rl"] for s in summaries), total_afford_gold),
            "rolling_buy_something_else_given_province_affordable_rl": _safe_div(sum(s["buy_something_else_when_province_affordable_count_rl"] for s in summaries), total_afford_province),
            "rolling_pass_given_gold_affordable_rl": _safe_div(sum(s["pass_when_gold_affordable_count_rl"] for s in summaries), total_afford_gold),
            "rolling_pass_given_province_affordable_rl": _safe_div(sum(s["pass_when_province_affordable_count_rl"] for s in summaries), total_afford_province),
            "rolling_first_turn_reach_8_mean_rl": round(float(np.mean(reach_8_vals)), 2) if reach_8_vals else "",
            "rolling_first_turn_buy_province_mean_rl": round(float(np.mean(buy_prov_vals)), 2) if buy_prov_vals else "",
            "rolling_rl_minus_opp_first_province_turn_mean": round(float(np.mean(rl_minus_opp_vals)), 2) if rl_minus_opp_vals else "",
            "rolling_episode_loss_mean": round(float(np.mean([s["episode_loss_mean"] for s in summaries if s["episode_loss_mean"] != ""])), 6) if any(s["episode_loss_mean"] != "" for s in summaries) else "",
            "rolling_episode_td_error_abs_mean": round(float(np.mean([s["episode_td_error_abs_mean"] for s in summaries if s["episode_td_error_abs_mean"] != ""])), 6) if any(s["episode_td_error_abs_mean"] != "" for s in summaries) else "",
            "rolling_episode_q_value_mean": round(float(np.mean([s["episode_q_value_mean"] for s in summaries if s["episode_q_value_mean"] != ""])), 6) if any(s["episode_q_value_mean"] != "" for s in summaries) else "",
            "rolling_episode_q_value_max": round(float(np.mean([s["episode_q_value_max"] for s in summaries if s["episode_q_value_max"] != ""])), 6) if any(s["episode_q_value_max"] != "" for s in summaries) else "",
            "rolling_episode_epsilon_mean": round(float(np.mean([s["episode_epsilon_mean"] for s in summaries if s["episode_epsilon_mean"] != ""])), 6) if any(s["episode_epsilon_mean"] != "" for s in summaries) else "",
            "rolling_random_action_rate": round(float(np.mean([float(s["episode_random_action_rate"]) for s in summaries if s["episode_random_action_rate"] != ""])), 6) if any(s["episode_random_action_rate"] != "" for s in summaries) else "",
            "rolling_greedy_action_rate": round(float(np.mean([float(s["episode_greedy_action_rate"]) for s in summaries if s["episode_greedy_action_rate"] != ""])), 6) if any(s["episode_greedy_action_rate"] != "" for s in summaries) else "",
        }

        with self.rolling_csv_path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=ROLLING_HEADER).writerow(row)
