from dominion_env import DominionEnv
from dummy_agent import run_dummy_agent
from train_dqn import train_buy_phase

pyminion_dir = 'C:/Users/johns/OneDrive/Desktop/projects/Dominion_AI_ML/pyminion_master'
import sys
sys.path.append(pyminion_dir)

from pyminion.expansions import base
from pyminion.game import Game
from dummie_bot import DummieBot

def get_all_card_types(expansions):
    # Safely extract all possible card names from the selected expansions
    return sorted({card.name for expansion in expansions for card in expansion})


if __name__ == "__main__":
    # Step 1: Choose your bots
    bot1 = DummieBot("RL_Agent")

    # Step 2: Select expansion set and derive card types (before game is started)
    selected_expansions = [base.test_set]
    # Step 3: Create the unstarted game
    game = Game(players=[bot1], expansions=selected_expansions)



    obs_card_types = ["Gardens", "Smithy", "Estate", "Duchy", "Province", "Copper", "Silver", "Gold"]

    # Step 4: Wrap game into Gym-like environment
    env = DominionEnv(game, bot1, None, all_card_types=obs_card_types)

    # Step 5: Run tests or train

    # Test here...
    # print("\n--- Running Dummy Agent ---")
    # run_dummy_agent(env)

    # exit()

    # Train here...
    print("\n--- Training Buy Phase with DQN ---")
    train_buy_phase(env)


### Notes:
# 1. Reward things like card draw and increasing turn money?
# 1. Curriculum learning?
# 1. Rewards based on victory points at the end of the game?
# 1. Right now, scope is limited to training only a few selected cards. But if we want the agent
#    to generalize its knowledge to other cards that it has not seen before / trained on, we will
#    have to figure out how to encode card details into the agent's observation space.
# 1. Extension of project -- game theory to model multiplayer interactions?