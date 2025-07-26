

import random
from collections import deque
import numpy as np



import torch
import torch.nn as nn
import torch.optim as optim

from gym.vector import SyncVectorEnv

from dominion_env_factory import make_env



import logging
logger = logging.getLogger()



#####################################################
##### ---> Try different NN architectures. <--- #####
#####################################################

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



def get_valid_buy_mask_from_obs(obs, cards):
    """
    Given a single observation vector (from env._get_observation()),
    return a binary mask of valid buy actions.

    Assumes:
    - obs[0:len(cards_names)]       = supply counts
    - obs[len(cards_names):2*len()] = hand counts
    - obs[-3:]                      = [actions, buys, total_money]
    """
    n = len(cards)
    supply = obs[:n]
    buys = obs[-2]
    money = obs[-1]

    mask = np.zeros(n + 1, dtype=np.float32)  # +1 for "pass"

    mask[-1] = 1.0

    if buys <= 0:
        return mask

    card_costs = {card.name: card.base_cost.money for card in cards}

    for i in range(n):
        if supply[i] > 0 and card_costs[cards[i].name] <= money:
            mask[i] = 1.0

    mask[-1] = 1.0  # "pass" is always legal

    return mask



##################################################################################
##### ---> Figure out different rewards for buy phase and action phase. <--- #####
##################################################################################


def train_buy_phase(
        cards_used_in_game,
        episodes=1,
        episode_timeout=1,
        report_interval=1,
        n_envs=1
    ):

    card_names = [card.name for card in cards_used_in_game]

    env_fns = [make_env(card_names, seed=i) for i in range(n_envs)]
    envs = SyncVectorEnv(env_fns)

    input_dim = envs.single_observation_space.shape[0]
    output_dim = envs.single_action_space.n

    policy_net = DQN(input_dim, output_dim)
    target_net = DQN(input_dim, output_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # -------------------------------------------------------------------
    
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

    # -------------------------------------------------------------------

    replay_buffer = deque(maxlen=10000)

    for episode in range(episodes):
        obs, info = envs.reset()
        dones = [False] * n_envs
        reward_sums = np.zeros(n_envs)
        steps = 0

        while not all(dones):
            obs_tensor = torch.tensor(obs, dtype=torch.float32)

            with torch.no_grad():
                q_vals = policy_net(obs_tensor)

            actions = np.zeros(n_envs, dtype=int)

            for i in range(n_envs):
                mask = get_valid_buy_mask_from_obs(obs[i], cards_used_in_game )
                valid_idxs = np.flatnonzero(mask)

                # Explore or exploit.
                if random.random() < epsilon:
                    actions[i] = int(np.random.choice(valid_idxs))
                else:
                    q_vals_mod = q_vals[i].clone()
                    q_vals_mod[mask == 0] = -1e9
                    actions[i] = int(torch.argmax(q_vals_mod).item())


            next_obs, rewards, terminated, truncated, infos = envs.step(actions)
            dones = np.logical_or(terminated, truncated)

            for i in range(n_envs):
                replay_buffer.append((obs[i], actions[i], rewards[i], next_obs[i], dones[i]))

            obs = next_obs
            reward_sums += rewards
            steps += 1

            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                obs_b, act_b, rew_b, next_obs_b, done_b = zip(*batch)
                obs_b_np = np.asarray(obs_b, dtype=np.float32)
                obs_b = torch.tensor(obs_b_np, dtype=torch.float32)
                act_b = torch.as_tensor(act_b, dtype=torch.long).unsqueeze(1)
                rew_b = torch.tensor(rew_b, dtype=torch.float32)
                next_obs_b = torch.tensor(next_obs_b, dtype=torch.float32)
                done_b = torch.tensor(done_b, dtype=torch.float32)

                q_vals = policy_net(obs_b).gather(1, act_b).squeeze()
                next_q_vals = target_net(next_obs_b).max(1)[0]
                target = rew_b + gamma * next_q_vals * (1 - done_b)

                loss = criterion(q_vals, target.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if steps >= episode_timeout:
                break
                # dones = [True] * n_envs

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        avg_reward = reward_sums.mean()
        logger.info(f"[Vec-BuyPhase] Ep {episode:04d} | epsilon={epsilon:.3f} | avg_reward={avg_reward:.3f}")
        target_net.load_state_dict(policy_net.state_dict())