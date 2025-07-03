import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

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

def train_buy_phase(env, episodes=1000):
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_spaces["buy"].n

    policy_net = DQN(input_dim, output_dim)
    target_net = DQN(input_dim, output_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    replay_buffer = deque(maxlen=10000)

    batch_size = 64
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.1

    for episode in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            if env.phase != "buy":
                obs, _, done, _ = env.step(0)
                continue

            if random.random() < epsilon:
                action = random.randint(0, output_dim - 1)
            else:
                with torch.no_grad():
                    action = policy_net(torch.tensor(obs, dtype=torch.float32)).argmax().item()

            next_obs, reward, done, _ = env.step(action)
            replay_buffer.append((obs, action, reward, next_obs, done))
            obs = next_obs

            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                obs_b, act_b, rew_b, next_obs_b, done_b = zip(*batch)
                obs_b = torch.tensor(obs_b, dtype=torch.float32)
                act_b = torch.tensor(act_b).unsqueeze(1)
                rew_b = torch.tensor(rew_b)
                next_obs_b = torch.tensor(next_obs_b, dtype=torch.float32)
                done_b = torch.tensor(done_b)

                q_vals = policy_net(obs_b).gather(1, act_b).squeeze()
                next_q = target_net(next_obs_b).max(1)[0]
                target = rew_b + gamma * next_q * (1 - done_b)

                loss = criterion(q_vals, target.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if episode % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            print(f"Ep {episode}  ε={epsilon:.3f}")
