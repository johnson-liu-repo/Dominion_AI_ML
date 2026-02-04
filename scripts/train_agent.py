
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))
sys.path.append(str(repo_root / "src"))

from pyminion_master.pyminion.expansions import base
from pyminion_master.pyminion.game import Game
from pyminion_master.pyminion.bots.examples import BigMoney, BigMoneyUltimate


from agent_rl.train_dqn import train_buy_phase
from agent_rl.card_catalog import BASE_CARDS
from agent_rl.logging_utils import configure_training_logging
from src.agent_rl.dominion_env_factory import make_env


if __name__ == "__main__":
    # logger.info("-------------------------------------")
    # logger.info("---- Training Buy Phase with DQN ----")
    # logger.info("-------------------------------------")

    configure_training_logging()

    bm = BigMoney()
    # bm_ultimate = BigMoneyUltimate()

    phase_env = make_env(
            cards_used_in_game = BASE_CARDS,
            seed = 4991,
            opponent_bots = [ bm ]
        )()

    training_configuration = {
        'env': phase_env,
        'episodes': 200000,      # total number of episodes to train
        'turn_limit': 250,       # max turns per episode
        'batch_size': 64,        # batch size for optimization
        'gamma': 0.99,           # discount factor
        'epsilon': 1.0,          # starting epsilon for epsilon-greedy
        'eps_decay': 0.9995,     # epsilon decay rate per step
        'eps_min': 0.05,         # minimum epsilon
        'target_update': 1000    # sync target network every N steps
    }

    train_buy_phase(training_configuration)

### Notes:
# 1. Right now, scope is limited to training only a few selected cards. But if we want the agent
#    to generalize its knowledge to other cards that it has not seen before / trained on, we will
#    have to figure out how to encode card details into the agent's observation space.
# 1. Extension of project -- game theory to model multiplayer interactions?
