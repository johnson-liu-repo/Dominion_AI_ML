
import sys

dominion_dir = 'C:/Users/johns/Desktop/projects/Dominion_AI_ML'
sys.path.append(dominion_dir)

from pyminion_master.pyminion.expansions import base
from pyminion_master.pyminion.game import Game

from src.agent_rl.train_dqn import train_buy_phase
from src.agent_rl.card_catalog import BASE_CARDS_NAMES
from src.agent_rl.dominion_env_factory import make_env


import logging
logger = logging.getLogger()


if __name__ == "__main__":

  
    # Train here...
    logger.info("-------------------------------------")
    logger.info("---- Training Buy Phase with DQN ----")
    logger.info("-------------------------------------")


    phase_env = make_env(BASE_CARDS_NAMES)()      # call the thunk
    train_buy_phase(phase_env)

### Notes:
# 1. Right now, scope is limited to training only a few selected cards. But if we want the agent
#    to generalize its knowledge to other cards that it has not seen before / trained on, we will
#    have to figure out how to encode card details into the agent's observation space.
# 1. Extension of project -- game theory to model multiplayer interactions?
