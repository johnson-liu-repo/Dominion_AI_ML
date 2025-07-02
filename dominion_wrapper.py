


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
        self.action_space = spaces.Discrete(10)
        # ---->
        ### !!!

        ### !!!
        # ---->
        # Example: assume observation is a 100-dim vector (to be replaced)
        self.observation_space = spaces.Box(low=0, high=100, shape=(100,), dtype=np.float32)
        # ---->
        ### !!!

    def reset(self):
        self.game.start()
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

    def _get_observation(self):
        ### !!!
        # ---->
        # TODO: convert player + game state into fixed-length vector
        vec = np.zeros(100)
        # ---->
        ### !!!
        
        return vec

    def render(self, mode="human"):
        print(f"Turn: {self.player_bot.turns}, Score: {self.player_bot.get_victory_points()}")
