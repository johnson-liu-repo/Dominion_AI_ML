

# import dominion_wrapper_01
# import dominion_wrapper_02



# if __name__ == '__main__':
#     game = dominion_wrapper_02.DominionEnv()



from dominion_env import DominionEnv
from dummy_agent import run_dummy_agent
from train_dqn import train_buy_phase

import sys
pyminion_dir = r'C:\Users\johns\OneDrive\Desktop\projects\Dominion_AI_ML\pyminion_master'
sys.path.append(pyminion_dir)

from pyminion.expansions import base
from pyminion.game import Game
from pyminion.bots.custom_bots import DummieBot

# Setup
bot1 = DummieBot("RL_Agent")
bot2 = DummieBot("Bot")
game = Game(players=[bot1, bot2], expansions=[base.test_set])
game.start()
all_card_types = sorted([pile.name for pile in game.supply.piles])

# Create env
env = DominionEnv(game, bot1, bot2, all_card_types)

# Test
print('testing...')
run_dummy_agent(env)

# Train
# train_buy_phase(env)
