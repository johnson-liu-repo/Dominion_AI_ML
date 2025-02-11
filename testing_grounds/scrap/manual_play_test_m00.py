import sys
import os

pyminion_dir = '/mnt/c/Users/johns/OneDrive/Desktop/projects/dominion_ai/pyminion-master'
sys.path.append(pyminion_dir)

from pyminion.expansions import base, intrigue, seaside, alchemy
from pyminion.game import Game
from pyminion.bots.examples import BigMoney
from pyminion.human import Human

# Initialize human and bot.
human = Human()
bot = BigMoney()

# Set up the game.
game = Game( players = [ human, bot ], expansions = [ base.base_set, intrigue.intrigue_set, seaside.seaside_set, alchemy.alchemy_set ] )


# Play the game.
game.play()

