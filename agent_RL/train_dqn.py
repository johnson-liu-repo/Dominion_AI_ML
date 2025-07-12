import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque


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

def train_buy_phase(env, episodes=1):
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
            
            # Choose to explore or exploit.
            if random.random() < epsilon:
                # Explore.
                action = random.randint(0, output_dim - 1)
            else:
                # Exploit.
                with torch.no_grad():
                    q_vals = policy_net(torch.tensor(obs, dtype=torch.float32))
                    action = q_vals.argmax().item()

            next_obs, reward, done, _ = env.step_train_buy(action)

            # Reward shaping
            if action == output_dim - 1:  # Skip - the agent is passing the phase
                reward -= 0.01            #        without doing anything.

            elif reward == 0.0:
                reward += 0.01          # Give the agent a tiny reward to teach
                                        # it that doing something (anything) is
                                        # better than doing nothing (passing).

            reward_sum += reward
            steps += 1

            replay_buffer.append((obs, action, reward, next_obs, done))
            obs = next_obs

            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                obs_b, act_b, rew_b, next_obs_b, done_b = zip(*batch)
                obs_b = torch.tensor(obs_b, dtype=torch.float32)
                act_b = torch.tensor(act_b).unsqueeze(1)
                rew_b = torch.tensor(rew_b)
                next_obs_b = torch.tensor(next_obs_b, dtype=torch.float32)
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

        ########################################################################################
        ### Take a look at this ... Does an episode constitute a single turn? Or does an episode
        ###                         constitute an entire game? A single episode is an entire
        ###                         game.
        ########################################################################################
        if reward >= 1.0:                  # This tells us that the agent won
            win_counter += 1               # the game.

        if episode % report_interval == 0:
            target_net.load_state_dict(policy_net.state_dict())
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            win_rate = win_counter / report_interval
            logger.info(f"[Buy Phase] Ep {episode:04d} | epsilon={epsilon:.3f} | win_rate={win_rate:.2f} | avg_reward={reward_sum / max(steps,1):.3f}")
            win_counter = 0
        ########################################################################################