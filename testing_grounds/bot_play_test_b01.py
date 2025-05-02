import sys
import os

pyminion_dir = '/mnt/c/Users/johns/OneDrive/Desktop/projects/domAInion/pyminion-master'
sys.path.append(pyminion_dir)

from pyminion.expansions import base
from pyminion.game import Game
from pyminion.bots.examples import BigMoney, BigMoneySmithy
from pyminion.bots.custom_bots import DummieBot
from pyminion.simulator import Simulator


# Initialize human and bot.
bm = BigMoney( player_id = 'Big_Money' )
bm_smithy = BigMoneySmithy( player_id = 'Big_Money_Smithy' )

bot = DummieBot()
bot1 = DummieBot()
bot2 = DummieBot()

# Set up the game.
game = Game( players = [ bm, bm_smithy ],
             expansions = [ base.base_set ],
             kingdom_cards = [ base.smithy ],
             log_stdout = False,
             log_file = True,
             log_file_name = 'test_b01.log' )


sim = Simulator( game, iterations = 1 )

if __name__ == '__main__':
    result = sim.run()
    print(result)
