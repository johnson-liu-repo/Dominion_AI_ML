from dominion_env import DominionEnv
from dummy_agent import run_dummy_agent
from train_dqn import train_buy_phase

pyminion_dir = 'C:/Users/johns/OneDrive/Desktop/projects/Dominion_AI_ML/pyminion_master'
import sys
sys.path.append(pyminion_dir)

from pyminion.expansions import base
from pyminion.game import Game
from pyminion.bots.custom_bots import DummieBot


def get_all_card_types(expansions):
    # Safely extract all possible card names from the selected expansions
    return sorted({card.name for expansion in expansions for card in expansion})


if __name__ == "__main__":
    # Step 1: Choose your bots
    bot1 = DummieBot("RL_Agent")
    bot2 = DummieBot("Bot")

    # Step 2: Select expansion set and derive card types (before game is started)
    selected_expansions = [base.test_set]
    all_card_types = get_all_card_types(selected_expansions)

    # Step 3: Create the unstarted game
    game = Game(players=[bot1, bot2], expansions=selected_expansions)

    # Step 4: Wrap game into Gym-like environment
    env = DominionEnv(game, bot1, bot2, all_card_types)

    # Step 5: Run tests or train
    print("\n--- Running Dummy Agent ---")
    run_dummy_agent(env)

    exit()

    print("\n--- Training Buy Phase with DQN ---")
    train_buy_phase(env)
