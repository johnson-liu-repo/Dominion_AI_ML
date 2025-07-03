import numpy as np
import gym
from gym import spaces

class DominionEnv(gym.Env):
    def __init__(self, game, player_bot, opponent_bot, all_card_types):
        super().__init__()

        self.game = game
        self.player_bot = player_bot
        self.opponent_bot = opponent_bot
        self.all_card_types = all_card_types  # List of all card names as strings
        self.current_player = self.player_bot
        self.phase = "action"

        self.action_space_action = spaces.Discrete(len(all_card_types) + 1)  # Last index = skip
        self.action_space_buy = spaces.Discrete(len(all_card_types) + 1)    # Last index = skip

        self.action_spaces = {
            "action": self.action_space_action,
            "buy": self.action_space_buy
        }

        # Dummy observation space size
        obs_size = len(all_card_types) + 3 + len(all_card_types)  # hand + scalars + supply
        self.observation_space = spaces.Box(low=0, high=100, shape=(obs_size,), dtype=np.float32)

    def reset(self):
        self.game.start()
        self.all_card_types = extract_card_names(self.game)
        self.phase = "action"
        self.current_player = self.player_bot
        return self._get_observation()

    def step(self, action):
        if self.phase == "action":
            self._apply_action_phase(action)
            self.phase = "money"
            return self._get_observation(), 0.0, False, {}

        elif self.phase == "money":
            self._play_all_treasures()
            self.phase = "buy"
            return self._get_observation(), 0.0, False, {}

        elif self.phase == "buy":
            reward, done = self._apply_buy_phase(action)
            self.phase = "action"
            return self._get_observation(), reward, done, {}

    def _get_observation(self):
        player = self.player_bot
        hand_counts = self._count_card_types(player.hand.cards)
        supply_counts = self._count_card_types_from_supply()

        scalar_features = np.array([
            player.actions,
            player.buys,
            player.coins
        ])

        return np.concatenate([hand_counts, scalar_features, supply_counts]).astype(np.float32)

    def _count_card_types(self, cards):
        count = np.zeros(len(self.all_card_types))
        for card in cards:
            if card.name in self.all_card_types:
                idx = self.all_card_types.index(card.name)
                count[idx] += 1
        return count

    def _count_card_types_from_supply(self):
        count = np.zeros(len(self.all_card_types))
        for pile in self.game.supply.piles:
            if pile.name in self.all_card_types:
                idx = self.all_card_types.index(pile.name)
                count[idx] = len(pile)
        return count

    def _play_all_treasures(self):
        player = self.player_bot
        treasures = [card for card in player.hand.cards if 'Treasure' in card.types]
        for card in treasures:
            card.play(player, self.game)

    def _apply_action_phase(self, action_idx):
        if action_idx == len(self.all_card_types):
            return  # skip

        card_name = self.all_card_types[action_idx]
        player = self.player_bot

        for card in player.hand.cards:
            if card.name == card_name and 'Action' in card.types:
                card.play(player, self.game)
                break

    def _apply_buy_phase(self, action_idx):
        reward = 0.0
        done = False

        if action_idx == len(self.all_card_types):
            pass  # skip buy
        else:
            card_name = self.all_card_types[action_idx]
            card = next((pile.cards[0] for pile in self.game.supply.piles if pile.name == card_name and len(pile) > 0), None)
            if card:
                self.player_bot.buy(card, self.game)

        if self.game.is_over():
            done = True
            winners = self.game.get_winners()
            if self.player_bot in winners:
                reward = 1.0 if len(winners) == 1 else 0.5
            else:
                reward = 0.0

        return reward, done

    def extract_card_names(game):
        card_names = [pile.name for pile in game.supply.piles]
        return sorted(card_names)

import ace_tools as tools; tools.display_dataframe_to_user(name="DominionEnv Template", dataframe=None)
