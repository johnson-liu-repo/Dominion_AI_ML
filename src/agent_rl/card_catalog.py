


import sys

dominion_dir = 'C:/Users/john/Desktop/projects/Dominion_AI_ML'
sys.path.append(dominion_dir)

from pyminion_master.pyminion.expansions import base


BASE_TREASURE = base.base_set_treasures_names
BASE_VICTORY  = base.base_set_victory_names
BASE_CURSE    = base.base_set_curses_names
BASE_ACTION   = base.base_set_actions_names

BASE_CARDS    = BASE_TREASURE + BASE_VICTORY + BASE_CURSE + BASE_ACTION
BASE_CARDS_C1 = BASE_TREASURE + BASE_VICTORY + BASE_CURSE

if __name__ == "__main__":

    pass