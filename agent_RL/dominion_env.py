import numpy as np
import gym
from gym import spaces


import logging
logger = logging.getLogger()



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

        num_card_types = len(all_card_types)
        obs_len = num_card_types + num_card_types + 3  # supply + hand + [actions, buys, total_money]

        self.observation_space = spaces.Box(low=0, high=100, shape=(obs_len,), dtype=np.float32)

    def reset(self):
        self.game.start() # <--- Fix this. Right now, this is restarting the entire game every reset.
                          # Actually, is the problem here or somewhere else? Since the game is supposed
                          # to restart at the end of an episode.
        self.phase = "action"
        # return self._get_observation()

    def step_train_buy(self, choice):
        logger.info("Bot is starting turn...")
        self.player_bot.start_turn(self.game, False)

        logger.info("Bot is trying to perform an action..."
                    "( should be nothing since bot has no action instructions )...")
        self.player_bot.start_action_phase(self.game)

        self.game.current_phase = self.game.Phase.Buy
        self._play_all_treasures()

        logger.info("Bot is starting buy phase..."
                    "( training happens here )...")
        reward, done = self._apply_buy_phase(choice)

        logger.info("Bot is starting cleanup phase...")
        self.player_bot.start_cleanup_phase(self.game)

        logger.info("Bot is ending turn...")
        self.player_bot.end_turn(self.game)

        return self._get_observation(), reward, done, {}

    ### ---> NEED TO FIGURE OUT WHAT THE OBSERVATIONS SHOULD BE. <--- ###
    def _get_observation(self):
        player = self.player_bot
        hand_counts = self._count_card_types(player.hand.cards)
        supply_counts = self._count_card_types_from_supply()

        # Compute total money available this turn from treasures in hand
        # treasure_names = {"Copper", "Silver", "Gold"}
        total_money = sum(card.money for card in player.hand.cards if 'Treasure' in card.type)

        # total_money = sum(card.get_coin_value(game=self.game, player=player) for card in player.hand.cards if card.is_treasure())

        scalars = np.array([player.actions, player.buys, total_money])

        # Final observation
        obs = np.concatenate([supply_counts, hand_counts, scalars]).astype(np.float32)
        return obs
    
    def return_observation(self):
        return self._get_observation()

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
        logger.info(f"Bot is told to play all treasure cards...")
        for card in list(self.player_bot.hand.cards):
            if 'Treasure' in card.type:
                logger.info(f"Bot is playing (treasure) {card.name}...which is worth {card.money}")
                card.play(self.player_bot, self.game)

    def _apply_action_phase(self, action):
        reward = 0.0

        if action == len(self.all_card_types):
            logger.info("Bot is passing the action phase...")
            return reward

        valid_play = False
        name = self.all_card_types[action]

        logger.info(f"Bot is attempting to play {name}...")

        for card in self.player_bot.hand.cards:
            if card.name == name and 'Action' in card.types:
                card.play(self.player_bot, self.game)
                valid_play = True
                logger.info(f"Bot successfully played {name}...")
                break
        
        if valid_play == False:
            logger.info(f"Bot was unable to play {name} because it was not in hand...")
            reward = -1

        return reward

    def _apply_buy_phase(self, action):
        reward, done = 0.0, False

        if action == len(self.all_card_types):
            logger.info("Bot is passing the buy phase...")
            return reward, done

        valid_buy = False
        name = self.all_card_types[action]

        logger.info(f"action choice: {action}, name: {name}")

        for pile in self.game.supply.piles:
            if pile.name == name and len(pile) > 0:
                logger.info(f"Bot is attempting to buy {pile.name}...")

                cost = pile.cards[0].base_cost
                logger.info(f"{name} costs {cost}...")

                money = self.player_bot.state.money
                logger.info(f"money: ${money}, cost: {cost}")

                if money >= cost:
                    logger.info(f"Bot buys {name}...")
                    self.player_bot.buy(pile.cards[0], self.game)
                    valid_buy = True

                    # logger.info(f"[DEBUG] Supply left: {[card.name for card in self.game.supply.cards if card.count > 0]}")
                    # logger.info(f"[DEBUG] Game over? {self.game.is_over()}")

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
