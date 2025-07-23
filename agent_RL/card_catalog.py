


import sys

dominion_dir = 'C:/Users/johns/OneDrive/Desktop/projects/Dominion_AI_ML'
sys.path.append(dominion_dir)

from pyminion_master.pyminion.expansions import base


BASE_TREASURE = base.base_set_treasures
BASE_VICTORY  = base.base_set_victory
BASE_CURSE    = base.base_set_curses
BASE_ACTION   = base.base_set_actions

BASE_CARDS    = BASE_TREASURE + BASE_VICTORY + BASE_CURSE + BASE_ACTION

# BASE_CARD_NAMES     = [c.name for c in BASE_CARDS]
# CARD2IDX            = {name: i for i, name in enumerate(BASE_CARD_NAMES)}
# NUM_CARDS           = len(BASE_CARD_NAMES)
# PASS_IDX            = NUM_CARDS


if __name__ == "__main__":

    pass