import sys
import os

# Windows-style path to the pyminion library.
pyminion_dir = r'C:\Users\johns\OneDrive\Desktop\projects\Dominion_AI_ML\pyminion-master'
sys.path.append(pyminion_dir)

from pyminion.expansions import base
from pyminion.game import Game
from pyminion.bots.examples import BigMoney, BigMoneySmithy
from pyminion.bots.custom_bots import DummieBot
from pyminion.simulator import Simulator


# Initialize and bot.
bm = BigMoney( player_id = 'Big_Money' )
bm_smithy = BigMoneySmithy( player_id = 'Big_Money_Smithy' )

dummie_bot = DummieBot()

# Set up the game.
game = Game( players = [ dummie_bot, bm_smithy ],
             expansions = [ base.base_set ],
             kingdom_cards = [ base.smithy ],
             log_stdout = False,
             log_file = True,
             log_file_name = 'test_b01.log' )


sim = Simulator( game, iterations = 1 )

if __name__ == '__main__':
    result = sim.run()
    print(result)
