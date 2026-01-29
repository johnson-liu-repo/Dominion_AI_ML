# dominion_env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import logging
logger = logging.getLogger(__name__)


class DominionBuyPhaseEnv(gym.Env):
    """
    A one-phase wrapper: the *agent* decides only the BUY action.
    All other phases are executed internally so every step == exactly
    one *complete* player turn.
    """
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, game, player_bot, card_names):
        super().__init__()
        self.game        = game
        self.bot         = player_bot          # convenience alias
        self.card_names  = card_names          # ordered list of |K| kingdom cards
        self.pass_idx    = len(card_names)     # final index is "pass / buy nothing"

        n = len(card_names)
        self.action_space = spaces.Discrete(n + 1)                  # 0..n  (n==pass)
        self.observation_space = spaces.Box(low=0, high=100,
                                            shape=(2 * n + 3,),     # supply + hand + scalars
                                            dtype=np.float32)

        # internal flags
        self._done  = False
        self._turn  = 0

    # --------------------------------------------------------------------- #
    #  Public API                                                           #
    # --------------------------------------------------------------------- #
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.game.start()                   # full game restart
        self._turn, self._done = 0, False
        self._start_new_turn()              # plays start-action & treasure phase
        return self._obs(), {}

    def step(self, action: int):
        assert self.action_space.contains(action), "invalid action index"

        # self.bot.start_buy_phase(self.game)
        reward, _ = self._apply_buy(action)

        # --- wrap up the turn -------------------------------------------- #
        self.bot.start_cleanup_phase(self.game)
        self.bot.end_turn(self.game)

        # terminal check (province pile / empty-3) ------------------------- #
        if self.game.is_over():
            self._done = True
            winners = self.game.get_winners()
            info = {"won": self.bot in winners}
            # overwrite reward with final outcome bonus
            reward += 1.0 if info["won"] else 0.0
            return self._obs(), reward, True, info

        # otherwise start next turn & return obs -------------------------- #
        self._start_new_turn()
        return self._obs(), reward, False, {}

    # --------------------------------------------------------------------- #
    #  Internal helpers                                                     #
    # --------------------------------------------------------------------- #
    def _start_new_turn(self):
        self._turn += 1
        self.bot.start_turn(self.game, is_extra_turn=False)
        self.bot.start_action_phase(self.game)

        self.game.current_phase = self.game.Phase.Buy
        
        # auto-play all treasures
        for card in list(self.bot.hand.cards):
            if 'Treasure' in card.type:
                card.play(self.bot, self.game)


    # ---------- buy logic ------------------------------------------------- #
    def _apply_buy(self, action_idx: int):
        """
        Execute the chosen buy.  Returns (reward, valid_flag).
        Reward scheme:
          • +0.01  valid buy (or pass)
          • -0.01  illegal selection (not enough $, empty pile, mask==0)
        """
        logger.info(f"{self.bot.player_id} has {self.bot.state.money} money available...")
        logger.info(f"{self.bot.player_id} has chosen action index {action_idx}...")
        logger.info(f"The cards are {self.card_names}...\n")

        #  mask invalid indices
        mask = self.valid_action_mask()
        if mask[action_idx] == 0:
            logger.info(f"Buy phase: {self.bot.player_id} tried illegal action {action_idx} (turn {self._turn})")
            return -0.1, False

        if action_idx == self.pass_idx:         # no purchase
            logger.info(f"Buy phase: {self.bot.player_id} passed (turn {self._turn})")
            return -0.01, True

        name = self.card_names[action_idx]
        pile = self.game.supply.get_pile(name)
        try:
            self.bot.buy(pile.cards[0], self.game)
            logger.info(f"Buy phase: {self.bot.player_id} bought {name} (turn {self._turn})")
            return +0.01, True
        except Exception:                       # money / buys / empty pile
            return -0.01, False

    # ---------- observation + mask --------------------------------------- #
    def _obs(self):
        h = self._count_deck(self.bot.hand.cards)
        s = self._count_supply()
        scalars = np.array([self.bot.state.actions,
                            self.bot.state.buys,
                            self.bot.state.money], dtype=np.float32)
        return np.concatenate([s, h, scalars])

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
    def _count_deck(self, cards):
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

    def _count_supply(self):
        cnt = np.zeros(len(self.card_names), dtype=np.float32)
        for i, name in enumerate(self.card_names):
            pile = self._pile_for_card(name)
            if pile is not None:
                cnt[i] = len(pile)
        return cnt

    def _pile_for_card(self, name: str):
        """Robustly find the supply pile corresponding to `name`."""
        ln = name.lower()
        for pile in self.game.supply.piles:
            pn = pile.name.lower()
            # exact match or match against slash-joined multi-name piles
            if pn == ln or ln in pn.split('/'):
                return pile
        return None