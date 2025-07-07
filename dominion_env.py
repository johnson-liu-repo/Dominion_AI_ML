import numpy as np
import gym
from gym import spaces

class DominionEnv(gym.Env):
    def __init__(self, game, player_bot, opponent_bot=None, all_card_types=[]):
        super().__init__()
        self.game = game
        self.player_bot = player_bot
        self.all_card_types = all_card_types
        self.phase = "action"
        self.current_player = self.player_bot

        n = len(all_card_types)
        self.action_spaces = {
            "action": spaces.Discrete(n + 1),
            "buy": spaces.Discrete(n + 1)
        }

        obs_size = n + 3 + n  # hand vector + 3 scalars + supply vector
        self.observation_space = spaces.Box(low=0, high=100, shape=(obs_size,), dtype=np.float32)

    def reset(self):
        self.game.start() # <--- Fix this. Right now, this is restarting the entire game every reset.
        self.phase = "action"
        return self._get_observation()

    def step(self, choice):
        if self.phase == "action":
            reward = self._apply_action_phase(choice)
            self.phase = "money" # <--- self.phase alternates between action, money, buy
            return self._get_observation(), reward, False, {}

        elif self.phase == "money":
            self._play_all_treasures()
            self.phase = "buy" # <--- self.phase alternates between action, money, buy
            return self._get_observation(), 0.0, False, {}

        elif self.phase == "buy":
            reward, done = self._apply_buy_phase(choice)
            self.phase = "action" # <--- self.phase alternates between action, money, buy
            return self._get_observation(), reward, done, {}

    def _get_observation(self):
        player = self.player_bot
        hand_counts = self._count_card_types(player.hand.cards)
        supply_counts = self._count_card_types_from_supply()
        scalars = np.array([player.actions, player.buys, player.state.money])
        return np.concatenate([hand_counts, scalars, supply_counts]).astype(np.float32)

    def _count_card_types(self, cards):
        counts = np.zeros(len(self.all_card_types))
        for card in cards:
            if card.name in self.all_card_types:
                idx = self.all_card_types.index(card.name)
                counts[idx] += 1
        return counts

    def _count_card_types_from_supply(self):
        counts = np.zeros(len(self.all_card_types))
        for pile in self.game.supply.piles:
            if pile.name in self.all_card_types:
                idx = self.all_card_types.index(pile.name)
                counts[idx] = len(pile)
        return counts

    def _play_all_treasures(self):
        print(f"Bot is told to play all treasure cards...")
        for card in list(self.player_bot.hand.cards):
            if 'Treasure' in card.type:
                print(f"Bot is playing (treasure) {card.name}...which is worth {card.money}")
                card.play(self.player_bot, self.game)

    def _apply_action_phase(self, action):
        reward = 0.0

        if action == len(self.all_card_types):
            print("Bot is passing the action phase...")
            return reward

        valid_play = False
        name = self.all_card_types[action]

        print(f"Bot is attempting to play {name}...")

        for card in self.player_bot.hand.cards:
            if card.name == name and 'Action' in card.types:
                card.play(self.player_bot, self.game)
                valid_play = True
                print(f"Bot successfully played {name}...")
                break
        
        if valid_play == False:
            print(f"Bot was unable to play {name} because it was not in hand...")
            reward = -1

        return reward

    def _apply_buy_phase(self, action):
        reward, done = 0.0, True

        if action == len(self.all_card_types):
            print("Bot is passing the buy phase...")
            return reward, done

        valid_buy = False
        name = self.all_card_types[action]

        for pile in self.game.supply.piles:
            if pile.name == name and len(pile) > 0:
                print(f"Bot is attempting to buy {pile.name}...")

                cost = pile.cards[0].base_cost
                print(f"{name} costs {cost}...")

                money = self.player_bot.state.money

                if money >= cost:
                    self.player_bot.buy(pile.cards[0], self.game)
                    valid_buy = True
                    break

                else:
                    print(f"Bot does not have enough money to buy {name}")
        
        if valid_buy == False:
            reward = -1


        if self.game.is_over():
            done = True
            winners = self.game.get_winners()
            reward = 1.0 if self.player_bot in winners and len(winners) == 1 else 0.5 if self.player_bot in winners else 0.0

        return reward, done
