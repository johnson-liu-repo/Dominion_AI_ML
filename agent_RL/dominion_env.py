

import random
import numpy as np
import copy


import gym
from gym import spaces
import torch


import logging
logger = logging.getLogger()



class DominionEnv(gym.Env):
    def __init__(
        self,
        game,
        player_bot,
        opponent_bot = None,
        all_cards = []
    ):

        super().__init__()
        self.game_object = game
        self.game = copy.copy(self.game_object)

        self.player_bot = player_bot
        self.all_cards = all_cards
        self.phase = "action"

        n = len(all_cards)
        self.action_spaces = {
            "action": spaces.Discrete(n + 1),
            "buy": spaces.Discrete(n + 1)
        }

        self.action_space = self.action_spaces["buy"]  # for Gym compatibility

        num_card_types = len(all_cards)
        obs_len = num_card_types + num_card_types + 3  # supply + hand + [actions, buys, total_money]

        self.observation_space = spaces.Box(low=0, high=100, shape=(obs_len,), dtype=np.float32)


    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        random.seed(seed)
        np.random.seed(seed)
        return [seed]

    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.game = copy.copy(self.game_object)
        self.game.start()
        self.phase = "action"
        obs = self._get_observation()
        info = {}  # no extra info yet, but Gym expects this structure
        return obs, info


    def step(self, action):
        """
        Gym-compatible step for buy-phase: takes one int (action),
        applies it, updates the game, and returns the observation.
        """

        logger.info("Bot is starting turn...")
        self.player_bot.start_turn(self.game, False)

        logger.info("The board state is...")
        logger.info(self.game.supply.get_pretty_string(self.player_bot, self.game))

        logger.info("Bot is trying to perform an action...")
        logger.info("should be nothing since bot has no action instructions...")
        logger.info("DummieBotDecider.action_priority returns iter([])")
        self.player_bot.start_action_phase(self.game)

        self.game.current_phase = self.game.Phase.Buy

        logger.info(f"Bot is told to play all treasure cards...")
        self._play_all_treasures()

        logger.info("Bot is starting buy phase..."
                    "( training happens here )...")

        reward, terminated = self._apply_buy_phase(action)
        ###### vvv CHECK THIS OUT LATER vvv ######
        truncated = False                # unless you introduce a max‑turn cutoff
        ###### ^^^ CHECK THIS OUT LATER ^^^ ######
        
        logger.info("Bot is starting cleanup phase...")
        self.player_bot.start_cleanup_phase(self.game)

        logger.info("Bot is ending turn...\n\n")
        self.player_bot.end_turn(self.game)

        # cleanup …
        obs  = self._get_observation()
        info = {}

        return obs, reward, terminated, truncated, info



    ### ---> NEED TO FIGURE OUT WHAT THE OBSERVATIONS SHOULD BE. <--- ###
    def _get_observation(self):
        supply_counts = self._count_card_types_from_supply()
        hand_counts = self._count_card_types()

        # Compute total money available this turn from treasures in hand
        total_money = sum(card.money for card in self.player_bot.hand.cards if 'Treasure' in card.type)

        scalars = np.array([self.player_bot.actions, self.player_bot.buys, total_money])

        # Final observation
        obs = np.concatenate([supply_counts, hand_counts, scalars]).astype(np.float32)
        
        return obs
    
    def return_observation(self):
        return self._get_observation()
    

    # ----------  BUY-ACTION MASK  ----------
    def _get_valid_buy_mask(self) -> np.ndarray:
        """
        Boolean mask of length |action_space| indicating which buy-indices
        are legal **right now**. 1|0 -- legal|illegal
        Rule: a buy is legal if
          • the pile still has cards, AND
          • player has ≥ money + potions cost, AND
          • player still has buys remaining.
        The last index (len(all_cards)) is the <pass> action → always legal.
        """
        mask = np.zeros(self.action_spaces["buy"].n, dtype=np.float32)

        # Quick exit if the player has no buys left
        if self.player_bot.state.buys == 0:
            mask[-1] = 1.0          # only “pass” is legal
            return mask

        money   = self.player_bot.state.money
        potions = self.player_bot.state.potions

        for i, name in enumerate(self.all_cards):
            for pile in self.game.supply.piles:
                if pile.name == name:
                    affordable = (pile.cards[0].base_cost.money  <= money and
                                   pile.cards[0].base_cost.potions <= potions)
                    if len(pile) > 0 and affordable:
                        mask[i] = 1.0
                    break  # found the pile – jump to next card type

        mask[-1] = 1.0  # “pass / buy nothing” is always an option
        return mask

    def _count_card_types(self):
        counts = np.zeros(len(self.all_cards))
        for card in self.player_bot.hand.cards:
            if card.name in self.all_cards:
                idx = self.all_cards.index(card.name)
                counts[idx] += 1
        return counts

    def _count_card_types_from_supply(self):
        counts = np.zeros(len(self.all_cards))
        for pile in self.game.supply.piles:
            if pile.name in self.all_cards:
                idx = self.all_cards.index(pile.name)
                counts[idx] = len(pile)
        return counts

    def _play_all_treasures(self):
        for card in list(self.player_bot.hand.cards):
            if 'Treasure' in card.type:
                logger.info(f"Bot is playing (treasure) {card.name}...which is worth {card.money}")
                card.play(self.player_bot, self.game)

    def _apply_buy_phase(self, action):
        reward, done = 0.0, False

        if action == len(self.all_cards):
            logger.info(f"action choice: {action}, Bot is passing the buy phase...")
            reward = -0.01
            return reward, done

        valid_buy = False
        card_name = self.all_cards[action]

        logger.info(f"action choice: {action}, card_name: {card_name}")

        for pile in self.game.supply.piles:
            if pile.name == card_name and len(pile) > 0:
                logger.info(f"Bot is attempting to buy {pile.name}...")

                cost = pile.cards[0].base_cost
                logger.info(f"{card_name} costs {cost}...")

                money = self.player_bot.state.money
                logger.info(f"money: ${money}, cost: {cost}")

                if money >= cost:
                    logger.info(f"Bot buys {card_name}...")
                    self.player_bot.buy(pile.cards[0], self.game)
                    valid_buy = True
                    reward = 0.1
                    break

                else:
                    logger.info(f"Bot does not have enough money to buy {name}")

        if valid_buy == False:
            reward = -1

        if self.game.is_over():
            done = True
            winners = self.game.get_winners()
            reward = 1.0 if self.player_bot in winners and len(winners) == 1 else 0.5 if self.player_bot in winners else 0.0

        return reward, done
