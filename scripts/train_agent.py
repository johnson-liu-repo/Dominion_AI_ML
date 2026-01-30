
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))
sys.path.append(str(repo_root / "src"))

from pyminion_master.pyminion.expansions import base
from pyminion_master.pyminion.game import Game
from pyminion_master.pyminion.bots.examples import BigMoney, BigMoneyUltimate


from agent_rl.train_dqn import train_buy_phase
from agent_rl.card_catalog import BASE_CARDS_NAMES
from src.agent_rl.dominion_env_factory import make_env


import logging
logger = logging.getLogger()


if __name__ == "__main__":

  
    # Train here...
    logger.info("-------------------------------------")
    logger.info("---- Training Buy Phase with DQN ----")
    logger.info("-------------------------------------")

    bm = BigMoney()
    # bm_ultimate = BigMoneyUltimate()

    phase_env = make_env(
            cards_used_in_game = BASE_CARDS_NAMES,
            seed = 4991,
            opponent_bots = [ bm ]
        )()
    train_buy_phase(phase_env)

### Notes:
# 1. Right now, scope is limited to training only a few selected cards. But if we want the agent
#    to generalize its knowledge to other cards that it has not seen before / trained on, we will
#    have to figure out how to encode card details into the agent's observation space.
# 1. Extension of project -- game theory to model multiplayer interactions?
