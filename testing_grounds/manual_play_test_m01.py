import sys
import os

# Windows-style path to the pyminion library.
pyminion_dir = r'C:\Users\johns\OneDrive\Desktop\projects\Dominion_AI_ML\pyminion-master'
sys.path.append(pyminion_dir)

from pyminion.expansions import base
from pyminion.game import Game
from pyminion.bots.custom_bots import DummieBot
from pyminion.human import Human

# Initialize human and bot.
human = Human()
bot = DummieBot()

# Set up the game.
game = Game(players=[human, bot], expansions=[base.test_set])

# Play the game.
game.play()

