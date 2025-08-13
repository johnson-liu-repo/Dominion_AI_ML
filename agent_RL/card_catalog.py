


import sys

dominion_dir = 'C:/Users/john/Desktop/projects/Dominion_AI_ML'
sys.path.append(dominion_dir)

from pyminion_master.pyminion.expansions import base


BASE_TREASURE_NAMES = base.base_set_treasures_names
BASE_VICTORY_NAMES  = base.base_set_victory_names
BASE_CURSE_NAMES    = base.base_set_curses_names
BASE_ACTION_NAMES   = base.base_set_actions_names

BASE_CARDS_NAMES    = BASE_TREASURE_NAMES + BASE_VICTORY_NAMES + BASE_CURSE_NAMES + BASE_ACTION_NAMES


if __name__ == "__main__":

    pass