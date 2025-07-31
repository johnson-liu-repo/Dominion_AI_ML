
import sys

dominion_dir = 'C:/Users/johns/OneDrive/Desktop/projects/Dominion_AI_ML'
sys.path.append(dominion_dir)

from pyminion_master.pyminion.expansions import base
from pyminion_master.pyminion.game import Game

from agent_RL.train_dqn import train_buy_phase
from agent_RL.card_catalog import BASE_CARDS
from agent_RL.dominion_env_factory import make_env


import logging
logger = logging.getLogger()


if __name__ == "__main__":

  
    # Train here...
    logger.info("-------------------------------------")
    logger.info("---- Training Buy Phase with DQN ----")
    logger.info("-------------------------------------")


    env = make_env(BASE_CARDS)

    train_buy_phase(env)




### Notes:
# 1. Right now, scope is limited to training only a few selected cards. But if we want the agent
#    to generalize its knowledge to other cards that it has not seen before / trained on, we will
#    have to figure out how to encode card details into the agent's observation space.
# 1. Extension of project -- game theory to model multiplayer interactions?