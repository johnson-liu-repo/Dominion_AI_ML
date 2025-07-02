import sys

### Should figure out how to fix this later.
# Windows-style path to the pyminion library.
pyminion_dir = r'C:\Users\johns\OneDrive\Desktop\projects\Dominion_AI_ML\pyminion_master'
sys.path.append(pyminion_dir)

from pyminion.expansions import base
from pyminion.game import Game

from pyminion.simulator import Simulator

from pyminion.bots.examples import BigMoney, BigMoneySmithy, ChapelBot
from pyminion.bots.custom_bots import DummieBot

# Initialize and bot.
# bm = BigMoney( player_id = 'Big_Money' )
# bm_smithy = BigMoneySmithy( player_id = 'Big_Money_Smithy' )
chapel_bot = ChapelBot( player_id = 'Chapel_Bot' )

dummie_bot = DummieBot()

# Set up the game.
game = Game( players = [ dummie_bot, dummie_bot ],
             expansions = [ base.test_set ],
             log_stdout = False,
             log_file = False
            )


sim = Simulator( game, iterations = 1 )

if __name__ == '__main__':
    result = sim.run()
    print(result)
