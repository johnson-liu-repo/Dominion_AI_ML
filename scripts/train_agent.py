import argparse
import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))
sys.path.append(str(repo_root / "src"))

from pyminion_master.pyminion.expansions import base
from pyminion_master.pyminion.game import Game
from pyminion_master.pyminion.bots.examples import BigMoney, BigMoneyUltimate

from agent_rl.card_catalog import BASE_CARDS
from agent_rl.train_dqn import train_buy_phase
from src.agent_rl.dominion_env_factory import make_env


BOT_FACTORIES = {
    "BigMoney": BigMoney,
    "BigMoneyUltimate": BigMoneyUltimate,
}

CARD_SETS = {
    "BASE_CARDS": BASE_CARDS,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Dominion RL buy-phase agent using a JSON config file."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=repo_root / "config" / "train_agent.json",
        help="Path to training config JSON file.",
    )
    return parser.parse_args()


def _resolve_resume_from(value: str | None) -> Path | None:
    if value in (None, ""):
        return None

    candidate = Path(value)
    return candidate if candidate.is_absolute() else repo_root / candidate


def _build_opponents(bot_names: list[str]) -> list:
    opponents = []
    for bot_name in bot_names:
        factory = BOT_FACTORIES.get(bot_name)
        if factory is None:
            available = ", ".join(sorted(BOT_FACTORIES))
            raise ValueError(
                f"Unknown bot '{bot_name}'. Supported bot names: {available}."
            )
        opponents.append(factory())
    return opponents


def load_training_configuration(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    cards_key = raw.get("cards", "BASE_CARDS")
    cards_used_in_game = CARD_SETS.get(cards_key)
    if cards_used_in_game is None:
        available = ", ".join(sorted(CARD_SETS))
        raise ValueError(
            f"Unknown cards value '{cards_key}'. Supported values: {available}."
        )

    seed = raw.get("seed", 4991)
    opponent_bot_names = raw.get("opponent_bots", ["BigMoney"])
    opponents = _build_opponents(opponent_bot_names)

    phase_env = make_env(
        cards_used_in_game=cards_used_in_game,
        seed=seed,
        opponent_bots=opponents,
    )()

    training = raw.get("training", {})
    return {
        "env": phase_env,
        "episodes": training.get("episodes", 10),
        "turn_limit": training.get("turn_limit", 250),
        "batch_size": training.get("batch_size", 64),
        "gamma": training.get("gamma", 0.99),
        "epsilon": training.get("epsilon", 1.0),
        "eps_decay": training.get("eps_decay", 0.9995),
        "eps_min": training.get("eps_min", 0.05),
        "target_update": training.get("target_update", 1000),
        "resume_from": _resolve_resume_from(raw.get("resume_from")),
        "checkpoint_every": training.get("checkpoint_every", 200),
        "latest_every": training.get("latest_every", 1),
        "save_turns": training.get("save_turns", True),
        "save_turns_every": training.get("save_turns_every", 1),
        "progress_bar": training.get("progress_bar", True),
    }


if __name__ == "__main__":
    args = parse_args()
    training_configuration = load_training_configuration(args.config)
    train_buy_phase(training_configuration)
