

import sys

dominion_dir = 'C:/Users/johns/Desktop/projects/Dominion_AI_ML'
sys.path.append(dominion_dir)

import random
import numpy as np

from pyminion_master.pyminion.expansions import base
from pyminion_master.pyminion.game import Game

from src.agent_rl.wrappers import BuyPhaseEnv
from src.agent_rl.dummie_bot import DummieBot


def make_env(cards_used_in_game, seed=None, opponent_bots=None):
    def thunk():
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        bot1 = DummieBot("RL_Agent")
        opponents = list(opponent_bots) if opponent_bots else []

        selected_expansions = [base.curriculum_basic] # <--- adjust curriculum cards here
        players = [bot1] + opponents

        game = Game(
            players=players,
            expansions=selected_expansions,
            random_order=False  # True|False for nondeterministic|deterministic
        )

        # ------------------------------------------------------------------
        #  core_env  = underlying Pyminion `Game`
        #  phase_env = single-decision Gym wrapper we actually train on
        # ------------------------------------------------------------------
        core_env  = game
        phase_env = BuyPhaseEnv(
            game        = core_env,
            player_bot  = bot1,
            card_names  = cards_used_in_game,
            opponent_bots=opponents
        )

        # Reproducibility
        if seed is not None:
            phase_env.action_space.seed(seed)

        return phase_env

    return thunk
