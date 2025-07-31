

import sys

dominion_dir = 'C:/Users/johns/OneDrive/Desktop/projects/Dominion_AI_ML'
sys.path.append(dominion_dir)

import random
import numpy as np

from pyminion_master.pyminion.expansions import base
from pyminion_master.pyminion.game import Game

from agent_RL.dominion_env import DominionBuyPhaseEnv
from agent_RL.dummie_bot import DummieBot


def make_env(cards_used_in_game, seed=None):
    def thunk():
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        bot1 = DummieBot("RL_Agent")

        selected_expansions = [base.curriculum_simple]
        players = [bot1]

        game = Game(
            players=players,
            expansions=selected_expansions,
            random_order=True  # or False for deterministic
        )

        env = DominionBuyPhaseEnv(
            game = game,
            player_bot = bot1,
            all_cards = cards_used_in_game
        )

        # Set Gym-style seed (important for Gym compliance)
        env.seed(seed)

        return env

    return thunk

