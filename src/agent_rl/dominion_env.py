"""Gymnasium environment wrapper for Dominion buy-phase training."""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


from pyminion_master.pyminion.core import CardType

# from agent_rl.logging_utils import get_train_logger

# logger = get_train_logger()


class DominionBuyPhaseEnv(gym.Env):
    """
    A one-phase wrapper: the *agent* decides only the BUY action.
    All other phases are executed internally so every step == exactly
    one *complete* player turn.
    """
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, game, player_bot, card_names, opponent_bots=None):
        super().__init__()
        self.game          = game
        self.bot           = player_bot          # convenience alias
        self.opponent_bots = list(opponent_bots) if opponent_bots else []
        if len(self.opponent_bots) != 1:
            raise ValueError(
                "DominionBuyPhaseEnv currently supports exactly one opponent bot "
                "(2-player games only)."
            )
        self.opponent = self.opponent_bots[0]
        if len(getattr(self.game, "players", [])) != 2:
            raise ValueError(
                "DominionBuyPhaseEnv requires game.players to contain exactly "
                "two players: the RL agent and one opponent."
            )
        if self.bot not in self.game.players or self.opponent not in self.game.players:
            raise ValueError(
                "DominionBuyPhaseEnv requires both the RL agent and the "
                "configured opponent to be present in game.players."
            )
        self.card_names    = card_names          # Ordered list of card types encoded in
                                                 # the action/observation vectors.
                                                 # (In current training this includes
                                                 # base treasure, victory, curse,
                                                 # and action card names, not just
                                                 # kingdom cards.)
        self.pass_idx      = len(card_names)     # final index is "pass / buy nothing"

        n = len(card_names)
        self.action_space = spaces.Discrete(n + 1)                  # 0..n  (n==pass)
        self.observation_space = spaces.Box(
            low=-100,
            high=100,
            shape=(4 * n + 6,),  #######################################################
                                 # n is len(card_names), meaning the number of
                                 # card types represented in the action and
                                 # observation encoding. In the current setup,
                                 # card_names includes base treasure, victory,
                                 # curse, and action card names, even though
                                 # only some of those piles may actually be
                                 # present in the match supply.
                                 ########################################################
                                 # 4 n-length card-count vectors:
                                 # ======================================================
                                 # 1.  The number of cards remaining in each
                                 #     supply pile.
                                 #     --------------------------------------------------
                                 #     An n-vector where each element corresponds to
                                 #     the count of cards left in that supply pile.
                                 # ======================================================
                                 # 2.  The number of each card type in the
                                 #     agent's current hand.
                                 #     --------------------------------------------------
                                 #     An n-vector where each element corresponds to
                                 #     the count of that card type in the agent's hand.
                                 # ======================================================
                                 # 3.  The number of each card type in the
                                 #     agent's deck/hand/discard zones.
                                 #     --------------------------------------------------
                                 #     An n-vector where each element corresponds to
                                 #     the count of that card that the agent currently 
                                 #     owns across their deck, hand, and discard pile.
                                 # ======================================================
                                 # 4.  The number of each card type in the
                                 #     single opponent's deck/hand/discard
                                 #     zones.
                                 #     --------------------------------------------------
                                 #     An n-vector where each element corresponds to 
                                 #     the count of that card that the opponent
                                 #     currently owns across their deck, hand,
                                 #     and discard pile.
                                 # ======================================================
                                 # 6 scalar values:
                                 # ======================================================
                                 # 5.  The agent's remaining actions.
                                 # 6.  The agent's remaining buys.
                                 # 7.  The agent's available money.
                                 # 8.  The current turn number.
                                 # 9.  The current score difference versus the
                                 #     opponent.
                                 # 10. The total number of cards in the agent's 
                                 #     deck/hand/discard zones.
            dtype=np.float32,
        )

        # internal flags
        self._done  = False
        self._turn  = 0
        self.turn_events = []
        self._agent_hand_snapshot = None

    # --------------------------------------------------------------------- #
    #  Public API                                                           #
    # --------------------------------------------------------------------- #
    def reset(self, *, seed=None, options=None):
        """Reset the underlying game and return the initial observation."""
        super().reset(seed=seed)
        self.game.start()                   # full game restart
        self._turn, self._done = 0, False
        self.turn_events = []
        self._agent_hand_snapshot = None
        self._start_new_turn()              # plays start-action & treasure phase
        return self._obs(), {}

    def step(self, action: int):
        """Apply a buy action and advance the game by one full turn."""
        assert self.action_space.contains(action), "invalid action index"

        # self.bot.start_buy_phase(self.game)
        reward, _ = self._apply_buy(action)

        # --- wrap up the turn -------------------------------------------- #
        self.bot.start_cleanup_phase(self.game)
        self.bot.end_turn(self.game)

        # terminal check (province pile / empty-3) ------------------------- #
        if self.game.is_over():
            self._done = True
            info = self._terminal_info()
            reward += info["score_diff"] + 0.5 if info["won"] else -0.5
            return self._obs(), reward, True, info

        # play opponents, if any ------------------------------------------ #
        self._play_opponents()
        if self.game.is_over():
            self._done = True
            info = self._terminal_info()
            reward += info["score_diff"]
            return self._obs(), reward, True, info

        # otherwise start next turn & return obs -------------------------- #
        self._start_new_turn()
        return self._obs(), reward, False, {}

    # --------------------------------------------------------------------- #
    #  Internal helpers                                                     #
    # --------------------------------------------------------------------- #
    def _start_new_turn(self):
        """Advance to the buy phase for the controlled agent."""
        self._turn += 1
        self.game.current_player = self.bot
        self.bot.start_turn(self.game, is_extra_turn=False)
        self._agent_hand_snapshot = [card.name for card in self.bot.hand.cards]
        # logger.info(f"Hand ({self.bot.player_id}): {self.bot.hand}")
        self.bot.start_action_phase(self.game)
        self.bot.start_treasure_phase(self.game)

        self.game.current_phase = self.game.Phase.Buy

    def _play_opponents(self):
        """Play the single opponent turn before the next agent turn."""
        opponent = self.opponent
        hand_snapshot = [card.name for card in opponent.hand.cards]
        # logger.info(f"Hand ({opponent.player_id}): {opponent.hand}")
        self.game.current_player = opponent
        self.game.play_turn(opponent)
        buys = [
            card for phase, card in opponent.last_turn_gains
            if phase == self.game.Phase.Buy
        ]
        # if buys:
            # for card in buys:
                # logger.info(
                    # f"Buy phase: {opponent.player_id} bought {card} (turn {self._turn})"
                # )
        # else:
            # logger.info(f"Buy phase: {opponent.player_id} passed (turn {self._turn})")
        buy_names = [card.name for card in buys] if buys else ["PASS"]
        self._record_turn_event(opponent, hand_snapshot, buy_names, self._turn)

    def _terminal_info(self):
        """Compute terminal win/loss metadata for logging and reward shaping."""
        agent_score = self.bot.get_victory_points()
        opponent_score = self.opponent.get_victory_points()
        score_diff = agent_score - opponent_score
        winners = self.game.get_winners()
        return {
            "won": self.bot in winners,
            "agent_score": agent_score,
            "opponent_score": opponent_score,
            "score_diff": score_diff,
        }


    # ---------- diagnostics snapshot --------------------------------------- #
    def buy_phase_snapshot(self):
        """Return a dict describing the current buy-phase state for diagnostics.

        Must be called *before* env.step() so the state reflects pre-buy conditions.
        """
        money = self.bot.state.money
        affordable = {}
        supply_remaining = {}
        for name in self.card_names:
            pile = self._pile_for_card(name)
            if pile and len(pile) > 0:
                affordable[name] = pile.cards[0].base_cost.money <= money
            else:
                affordable[name] = False
            supply_remaining[name] = len(pile) if pile else 0

        agent_score = self.bot.get_victory_points()
        opponent_score = self.opponent.get_victory_points()

        return {
            "coins_available": money,
            "buys_available": self.bot.state.buys,
            "score_rl": agent_score,
            "score_opp": opponent_score,
            "score_diff": agent_score - opponent_score,
            "affordable": affordable,
            "supply_remaining": supply_remaining,
            "deck_size_rl": self._player_owned_card_count(self.bot),
            "turn": self._turn,
        }

    # ---------- buy logic ------------------------------------------------- #
    def _apply_buy(self, action_idx: int):
        """
        Execute the chosen buy.  Returns (reward, valid_flag).
        Reward scheme:
          • 
          • 
        """
        # logger.info(f"{self.bot.player_id} has {self.bot.state.money} money available...")
        # logger.info(f"{self.bot.player_id} has chosen action index {action_idx}...")
        # logger.info(f"The cards are {self.card_names}...\n")

        #  mask invalid indices
        mask = self.valid_action_mask()
        if mask[action_idx] == 0:
            # logger.debug(f"Buy phase: {self.bot.player_id} tried illegal action {action_idx} (turn {self._turn})")
            self._record_turn_event(self.bot, self._agent_hand_snapshot, ["ILLEGAL"], self._turn)
            return -0.1, False

        if action_idx == self.pass_idx:         # no purchase
            # logger.info(f"Buy phase: {self.bot.player_id} passed (turn {self._turn})")
            self._record_turn_event(self.bot, self._agent_hand_snapshot, ["PASS"], self._turn)
            return -0.01, True

        name = self.card_names[action_idx]
        pile = self._pile_for_card(name)

        try:
            card = pile.cards[0]

            self.bot.buy(pile.cards[0], self.game)
            # logger.info(f"Buy phase: {self.bot.player_id} bought {name} (turn {self._turn})")
            reward = 0.0

            # 
            if CardType.Curse in card.type:
                reward += -1

            if CardType.Victory in card.type:
                reward += self._victory_card_density()
            
            elif CardType.Treasure in card.type:
                reward += self._treasure_density()


            # penalty for diluting the deck
            # penalty becomes larger as the deck grows, but is bounded by -1 for very large decks
            reward += -(self._player_owned_card_count(self.bot)) / (self._player_owned_card_count(self.bot) + 1)

            # log the buy event with the card name and hand snapshot
            self._record_turn_event(self.bot, self._agent_hand_snapshot, [name], self._turn)

            return reward, True

        except Exception:                       # money / buys / empty pile
            self._record_turn_event(self.bot, self._agent_hand_snapshot, ["ILLEGAL"], self._turn)
            return -1, False

    # ---------- compute observables -------------------------------------- #
    def _treasure_density(self):
        total_copper_cards = self._count_named_player_owned_card(self.bot, "Copper")
        total_silver_cards = self._count_named_player_owned_card(self.bot, "Silver")
        total_gold_cards = self._count_named_player_owned_card(self.bot, "Gold")

        total_treasure_cards = total_copper_cards + total_silver_cards + total_gold_cards

        treasure_density = (total_silver_cards + 3*total_gold_cards)/total_treasure_cards if total_treasure_cards > 0 else 0.0
        
        return treasure_density
    
    def _victory_card_density(self):
        total_estate_cards = self._count_named_player_owned_card(self.bot, "Estate")
        total_duchy_cards = self._count_named_player_owned_card(self.bot, "Duchy")
        total_province_cards = self._count_named_player_owned_card(self.bot, "Province")

        total_victory_cards = total_estate_cards + total_duchy_cards + total_province_cards

        victory_card_density = (total_duchy_cards + 3*total_province_cards)/total_victory_cards if total_victory_cards > 0 else 0.0
        
        return victory_card_density


    # ---------- observation + mask --------------------------------------- #
    def _obs(self):
        """Build observation vector of supply, hand, and scalar state."""
        # supply: count of cards remaining in each supply pile
        s = self._count_supply()

        # hand: count of cards in bot's hand aligned to card_names
        h = self._count_deck(self.bot.hand.cards)

        # zone cards: count of cards in deck + hand + discard aligned to card_names
        d = self._count_player_zone_cards(self.bot)

        # opponent zone cards: count of cards in deck + hand + discard aligned to card_names
        od = self._count_player_zone_cards(self.opponent)


        # scalars: actions, buys, money, turn number, current score diff (agent vs opponent),
        # count of cards in deck + hand + discard

        scalars = np.array(
                            [self.bot.state.actions,
                                self.bot.state.buys,
                                self.bot.state.money,
                                self._turn,
                                self._terminal_info()["score_diff"],
                                self._player_zone_card_count(self.bot)
                            ], dtype=np.float32
                        )

        return np.concatenate([s, h, d, od, scalars])

    def valid_action_mask(self):
        """
        A binary (n+1,) mask where 1 == legal.
        • Legal if pile non-empty and cost ≤ money and bot has buys.
        The last slot (pass) is always legal.
        """
        money, buys = self.bot.state.money, self.bot.state.buys
        mask = np.zeros(len(self.card_names) + 1, dtype=np.float32)

        if buys == 0:
            mask[self.pass_idx] = 1.0
            return mask

        for i, name in enumerate(self.card_names):
            pile = self._pile_for_card(name)
            if pile is None:
                continue  # unmatched => stays illegal
            affordable = (len(pile) > 0 and pile.cards[0].base_cost.money <= money)
            if affordable:
                mask[i] = 1.0

        mask[self.pass_idx] = 1.0
        return mask

    # ---------- util counts ---------------------------------------------- #
    def _player_zone_cards(self, player):
        """Return cards in the player's deck, hand, and discard pile only."""
        return list(player.deck.cards) + list(player.hand.cards) + list(player.discard_pile.cards)

    def _player_zone_card_count(self, player):
        """Count the total number of cards in the player's deck, hand, and discard pile."""
        return len(player.deck.cards) + len(player.hand.cards) + len(player.discard_pile.cards)

    def _player_owned_card_count(self, player):
        """Count the total number of cards the player owns across all zones."""
        return player.get_all_cards_count()

    def _count_deck(self, cards):
        """Count card occurrences in a list of cards aligned to card_names."""
        cnt = np.zeros(len(self.card_names), dtype=np.float32)
        # build a fast map using the same tolerant match
        name_to_idx = {}
        for i, name in enumerate(self.card_names):
            p = self._pile_for_card(name)
            if p is not None:
                name_to_idx[p.name] = i  # use the canonical pile name
        for c in cards:
            idx = name_to_idx.get(c.name)
            if idx is not None:
                cnt[idx] += 1
        return cnt

    def _count_player_zone_cards(self, player):
        """Count card occurrences across deck, hand, and discard only."""
        return self._count_deck(self._player_zone_cards(player))

    def _count_player_owned_cards(self, player):
        """Count card occurrences across all cards the player currently owns."""
        return self._count_deck(player.get_all_cards())

    def _count_named_card(self, counts, name):
        """Safely read a named card count from a count vector."""
        try:
            return counts[self.card_names.index(name)]
        except ValueError:
            return 0.0

    def _count_named_player_zone_card(self, player, name):
        """Count a specific named card across deck, hand, and discard only."""
        return self._count_named_card(self._count_player_zone_cards(player), name)

    def _count_named_player_owned_card(self, player, name):
        """Count a specific named card across all owned zones."""
        return self._count_named_card(self._count_player_owned_cards(player), name)

    def _count_supply(self):
        """Count remaining cards in each supply pile."""
        cnt = np.zeros(len(self.card_names), dtype=np.float32)
        for i, name in enumerate(self.card_names):
            pile = self._pile_for_card(name)
            if pile is not None:
                cnt[i] = len(pile)
        return cnt

    def _pile_for_card(self, name: str):
        """Find the supply pile corresponding to `name`."""
        ln = name.lower()
        for pile in self.game.supply.piles:
            pn = pile.name.lower()
            # exact match or match against slash-joined multi-name piles
            if pn == ln or ln in pn.split('/'):
                return pile
        return None

    def _record_turn_event(self, player, hand_snapshot, buys, turn):
        self.turn_events.append({
            "turn": int(turn),
            "player_id": player.player_id,
            "hand": list(hand_snapshot) if hand_snapshot else [],
            "buys": list(buys) if buys else [],
        })

    def consume_turn_events(self):
        events = self.turn_events
        self.turn_events = []
        return events
