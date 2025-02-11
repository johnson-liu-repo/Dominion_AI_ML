import sys
import os

pyminion_dir = '/mnt/c/Users/johns/OneDrive/Desktop/projects/dominion_ai/pyminion-master/'
sys.path.append(pyminion_dir)

from pyminion.bots.examples import BigMoney, BigMoneySmithy
from pyminion.expansions.base import base_set, smithy
from pyminion.game import Game
from pyminion.simulator import Simulator

bm = BigMoney()
bm_smithy = BigMoneySmithy()

game = Game(players=[bm, bm_smithy], expansions=[base_set], kingdom_cards=[smithy], log_stdout=False)
sim = Simulator(game, iterations=1000)
result = sim.run()
print(result)
