import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np


import logging
logger = logging.getLogger()



class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

##############################################################################
### ---> Figure out different rewards for buy phase and action phase. <--- ###
##############################################################################

def train_buy_phase(
        env, 
        episodes=1
    ):
    report_interval = 1
    episode_timeout = 10

    input_dim = env.observation_space.shape[0]
    output_dim = env.action_spaces["buy"].n

    policy_net = DQN(input_dim, output_dim)
    target_net = DQN(input_dim, output_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    replay_buffer = deque(maxlen=10000)

    best_win_rate = 0.0
    batch_size = 64
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.1

    win_counter = 0

    for episode in range(episodes):
        env.reset()
        done = False
        reward_sum = 0.0
        steps = 0

        while not done:
            obs = env.return_observation()

            choice, next_obs, reward, done, _ = env.step_train_buy(epsilon, policy_net)

            reward_sum += reward
            steps += 1

            replay_buffer.append((obs, choice, reward, next_obs, done))

            # Why do we have this?
            # obs = next_obs

            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                obs_b, act_b, rew_b, next_obs_b, done_b = zip(*batch)
                obs_b = torch.tensor(obs_b, dtype=torch.float32)
                act_b = torch.tensor(act_b).unsqueeze(1)
                rew_b = torch.tensor(rew_b)
                next_obs_b = torch.tensor(next_obs_b, dtype=torch.float32)
                with torch.no_grad():
                    # build a batch of masks for the NEXT obs
                    masks = []
                    for single_next_obs in next_obs_b:
                        env.game._dummy_state_for_vectorised_use(single_next_obs)  # <-- see note*
                        masks.append(env._get_valid_buy_mask())
                    mask_tensor = torch.tensor(masks, dtype=torch.float32)

                    next_q = target_net(next_obs_b)
                    next_q[mask_tensor == 0] = -1e9
                    next_q = next_q.max(1)[0]

                # done_b = torch.tensor(done_b)
                done_b = torch.tensor(done_b, dtype=torch.float32)

                q_vals = policy_net(obs_b).gather(1, act_b).squeeze()
                next_q = target_net(next_obs_b).max(1)[0]
                target = rew_b + gamma * next_q * (1 - done_b)

                loss = criterion(q_vals, target.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if steps == episode_timeout:
                done = True
                logger.info(f"Episode timeout reached: {steps} steps...")

        if env.player_bot in env.game.get_winners():
            win_counter += 1

        if episode % report_interval == 0:
            target_net.load_state_dict(policy_net.state_dict())
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            win_rate = win_counter / report_interval
            logger.info(f"[Buy Phase] Ep {episode:04d} | epsilon={epsilon:.3f} | win_rate={win_rate:.2f} | avg_reward={reward_sum / max(steps,1):.3f}")
            win_counter = 0