import logging
from typing import Iterator

from pyminion.bots.bot import Bot, BotDecider
from pyminion.core import Card
from pyminion.expansions.base import province, duchy, estate
from pyminion.player import Player
from pyminion.game import Game

logger = logging.getLogger()

class DummieBotDecider(BotDecider):
    """
    Only buys Provinces, Duchies, and Estates in that order.
    - Only possible to buy Duchies and Estates without any other directive
    because the maximum amount of usable treasure is (5).

    """

    # The bot has no action priority.
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


class DummieBot(Bot):
    def __init__(
        self,
        player_id: str = "dummie_bot",
    ):
        super().__init__(decider=DummieBotDecider(), player_id=player_id)
