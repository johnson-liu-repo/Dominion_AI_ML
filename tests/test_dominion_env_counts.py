from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "src"))

from pyminion_master.pyminion.bots.examples import BigMoney
from pyminion_master.pyminion.expansions.base import copper, estate, silver

from src.agent_rl.card_catalog import BASE_CARDS
from src.agent_rl.dominion_env_factory import make_env


def _make_env():
    return make_env(BASE_CARDS, seed=4991, opponent_bots=[BigMoney()])()


def test_zone_card_count_matches_deck_hand_discard():
    env = _make_env()
    env.reset()

    expected = (
        len(env.bot.deck.cards)
        + len(env.bot.hand.cards)
        + len(env.bot.discard_pile.cards)
    )

    assert env._player_zone_card_count(env.bot) == expected


def test_named_zone_card_count_aggregates_across_deck_hand_discard():
    env = _make_env()
    env.reset()

    env.bot.deck.cards = [copper, estate]
    env.bot.hand.cards = [silver]
    env.bot.discard_pile.cards = [copper, copper]

    assert env._count_named_player_zone_card(env.bot, "Copper") == 3
    assert env._count_named_player_zone_card(env.bot, "Silver") == 1
    assert env._count_named_player_zone_card(env.bot, "Village Green") == 0


def test_owned_card_counts_include_playmat_but_zone_counts_do_not():
    env = _make_env()
    env.reset()

    original_owned = env._player_owned_card_count(env.bot)
    original_zone = env._player_zone_card_count(env.bot)

    moved_card = env.bot.discard_pile.cards.pop()
    env.bot.playmat.cards.append(moved_card)

    assert env._player_zone_card_count(env.bot) == original_zone - 1
    assert env._player_owned_card_count(env.bot) == original_owned
    assert env._count_named_player_zone_card(env.bot, moved_card.name) == (
        env._count_named_player_owned_card(env.bot, moved_card.name) - 1
    )


def test_observation_uses_zone_limited_counts_and_buy_updates_discard_counts():
    env = _make_env()
    obs, _ = env.reset()
    n = len(env.card_names)

    zone_counts = env._count_player_zone_cards(env.bot)
    np.testing.assert_array_equal(obs[2 * n:3 * n], zone_counts)

    silver_idx = env.card_names.index("Silver")
    before = env._count_named_player_zone_card(env.bot, "Silver")
    reward, is_valid = env._apply_buy(silver_idx)

    assert is_valid is True
    assert reward != -1
    assert env._count_named_player_zone_card(env.bot, "Silver") == before + 1
    assert env.bot.discard_pile.cards[-1].name == "Silver"
