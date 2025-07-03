


import numpy as np
import gym
from gym import spaces



class DominionEnv(gym.Env):
    def __init__(self):
        super().__init__()

        import sys

        ### Should figure out how to fix this later.
        # Windows-style path to the pyminion library.
        pyminion_dir = r'C:\Users\johns\OneDrive\Desktop\projects\Dominion_AI_ML\pyminion_master'
        sys.path.append(pyminion_dir)

        from pyminion.expansions import base
        from pyminion.game import Game
        from pyminion.bots.custom_bots import DummieBot



        self.player_bot = DummieBot(player_id="RL_Agent")
        self.opponent_bot = DummieBot(player_id="Opponent")

        self.game = Game(players=[self.player_bot, self.opponent_bot],
                         expansions=[base.test_set],
                         log_stdout=False,
                         log_file=False)
        
        self.current_player = self.player_bot
        self.phase = "action"
        
        ### !!!
        # ---->
        # Example: assume 10 possible actions (to be replaced)
        N = 2
        self.action_space_action = gym.spaces.Discrete(N + 1)  # 0 to N-1 = play card, N = skip
        # ---->
        ### !!!

        # For now, the bot should always play all money available in their hand.
        # self.action_space_money = gym.spaces.Discrete(0)

        ### !!!
        # ---->
        # Example: assume observation is a 100-dim vector (to be replaced)
        M = 8 # 2 kingdom cards, 3 money cards, 3 victory cards.
        self.action_space_buy = gym.spaces.Discrete(M + 1)  # 0 to M-1 = buy card, M = skip
        # ---->
        ### !!!

        self.action_spaces = {
            "action": self.action_space_action,
            "buy": self.action_space_buy
        }

    def reset(self):
        self.game.start()
        self.phase = "action"
        self.current_player = self.player_bot
        obs = self._get_observation()
        return obs

    '''
    def step(self, action):
        ### !!!
        # ---->
        # TODO: map action index to game move
        # ---->
        ### !!!
        reward = 0.0
        done = False

        # For now, simulate a full turn
        self.game.play_turn(self.player_bot)

        if self.game.is_over():
            done = True
            winners = self.game.get_winners()
            if self.player_bot in winners:
                reward = 1.0 if len(winners) == 1 else 0.5
            else:
                reward = 0.0

        obs = self._get_observation()
        return obs, reward, done, {}
        '''
    def step(self, action):
        if self.phase == "action":
            self._perform_action_phase(action)
            self.phase = "money"
            obs = self._get_observation()
            return obs, 0.0, False, {}
        
        elif self.phase == "money":
            self._play_all_treasures()
            self.phase = "buy"
            obs = self._get_observation()
            return obs, 0.0, False, {}

        elif self.phase == "buy":
            reward, done = self._perform_buy_phase(action)
            self.phase = "action"  # Next turn
            obs = self._get_observation()
            return obs, reward, done, {}

    # def _get_observation(self):
    #     ### !!!
    #     # ---->
    #     # TODO: convert player + game state into fixed-length vector
    #     vec = np.zeros(100)
    #     # ---->
    #     ### !!!
        
    #     return vec

    def _get_observation(self):
        player = self.player_bot

        # Count of each card in hand (one-hot or fixed order)
        hand_counts = self._count_card_types(player.hand.cards, self.all_card_types)

        # Simple scalars
        num_actions = 1 # For now, limit number of actions to 1.
        # num_actions = player.actions
        num_buys = 1 # For now, limit number of buys to 1.
        # num_buys = player.buys
        current_coins = player.coins

        # Supply info: number of each card left in supply
        supply_counts = self._count_card_types_from_supply()

        # Combine all into a single fixed-size vector
        obs_vector = np.concatenate(
                [
                    hand_counts,
                    [ num_actions, num_buys, current_coins ],
                    supply_counts
                ]
            ).astype(np.float32)

        return obs_vector

    def _count_card_types(self, cards, card_type_list):
        count = np.zeros(len(card_type_list))
        for card in cards:
            idx = card_type_list.index(card.name)
            count[idx] += 1
        return count

    def _count_card_types_from_supply(self):
        count = np.zeros(len(self.all_card_types))
        for pile in self.game.supply.piles:
            idx = self.all_card_types.index(pile.name)
            count[idx] = len(pile)
        return count

    def render(self, mode="human"):
        print(f"Turn: {self.player_bot.turns}, Score: {self.player_bot.get_victory_points()}")
