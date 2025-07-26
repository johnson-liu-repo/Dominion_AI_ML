
import sys

dominion_dir = 'C:/Users/johns/OneDrive/Desktop/projects/Dominion_AI_ML'
sys.path.append(dominion_dir)

from pyminion_master.pyminion.expansions import base
from pyminion_master.pyminion.game import Game

from agent_RL.train_dqn import train_buy_phase
from agent_RL.card_catalog import BASE_CARDS


import logging
logger = logging.getLogger()


if __name__ == "__main__":

  
    # Train here...
    logger.info("-------------------------------------")
    logger.info("---- Training Buy Phase with DQN ----")
    logger.info("-------------------------------------")


    base_set_cards = BASE_CARDS

    train_buy_phase(
            cards_used_in_game = base_set_cards,
            episodes=2,
            episode_timeout=1,
            report_interval=1,
            n_envs=1
        )


    logger.info("----------------------------------------------------")
    logger.info("----------------------------------------------------")
    logger.info("----------------------------------------------------")
    logger.info("----------------------------------------------------")
    logger.info("FIGURE OUT HOW TO TRAIN BUY PHASE SEPARATELY FROM\n"
                "ACTION PHASE IN A WAY THAT LETS THE AGENT LEARN\n"
                "HOW TO USE THE CARDS THAT IT BUYS...\n"
                "...READ THROUGH NOTES IN CODE...")
    logger.info("----------------------------------------------------")
    logger.info("----------------------------------------------------")
    logger.info("----------------------------------------------------")
    logger.info("----------------------------------------------------")



### Notes:
# 1. Right now, scope is limited to training only a few selected cards. But if we want the agent
#    to generalize its knowledge to other cards that it has not seen before / trained on, we will
#    have to figure out how to encode card details into the agent's observation space.
# 1. Extension of project -- game theory to model multiplayer interactions?