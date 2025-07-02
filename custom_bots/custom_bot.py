import logging
from typing import Iterator

from pyminion.bots.bot import Bot, BotDecider
from pyminion.core import Card
from pyminion.expansions.base import province, duchy, estate
from pyminion.player import Player
from pyminion.game import Game

logger = logging.getLogger()

class LearnerBotDecider(BotDecider):
    """

    """

    def action_priority(self, player: Player, game: Game) -> Iterator[Card]:
        return iter([])

    def buy_priority(self, player: Player, game: Game) -> Iterator[Card]:
        money = player.state.money
        if money >= 8:
            yield province
        if money >= 5:
            yield duchy
        if money >= 2:
            yield estate


class LearnerBot(Bot):
    def __init__(
        self,
        player_id: str = "leaner_bot",
    ):
        super().__init__(decider=LearnerBotDecider(), player_id=player_id)
