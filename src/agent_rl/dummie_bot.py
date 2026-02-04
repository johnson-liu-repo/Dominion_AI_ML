"""Baseline Dominion bot used for debugging and simple opponents."""

import logging
from typing import Iterator
import sys

# NOTE: Local development path used to import the bundled Pyminion package.
dominion_dir = 'C:/Users/johns/Desktop/projects/Dominion_AI_ML'
sys.path.append(dominion_dir)

from pyminion_master.pyminion.bots.bot import Bot, BotDecider
from pyminion_master.pyminion.core import Card
from pyminion_master.pyminion.expansions.base import province, duchy, estate
from pyminion_master.pyminion.player import Player
from pyminion_master.pyminion.game import Game

logger = logging.getLogger()

class DummieBotDecider(BotDecider):
    """
    Only buys Provinces, Duchies, and Estates in that order.
    - Only possible to buy Duchies and Estates without any other directive
    because the maximum amount of usable treasure is (5).

    """

    # The bot has no action priority.
    def action_priority(self, player: Player, game: Game) -> Iterator[Card]:
        """Return an empty action priority list to skip action cards."""
        return iter([])

    def buy_priority(self, player: Player, game: Game) -> Iterator[Card]:
        """Return an empty buy priority list (buys handled elsewhere)."""
        return iter([])


class DummieBot(Bot):
    """Minimal bot wrapper that uses DummieBotDecider."""
    def __init__(
        self,
        player_id: str = "dummie_bot",
    ):
        super().__init__(decider=DummieBotDecider(), player_id=player_id)

        # ----- Placeholder -----
        self.actions = 1
        self.buys = 1
