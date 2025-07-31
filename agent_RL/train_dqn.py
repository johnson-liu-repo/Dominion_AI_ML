
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def select_action(obs, policy_net, mask, epsilon, n_actions):
    """Return an int in [0, n_actions).  Mask is a 1/0 numpy array."""
    if random.random() < epsilon:
        # ---------- random but legal ----------
        valid_idxs = np.flatnonzero(mask)
        return int(np.random.choice(valid_idxs))

    # ---------- greedy but legal -------------
    with torch.no_grad():
        q = policy_net(torch.as_tensor(obs, dtype=torch.float32, device=DEVICE))
        # put -inf on illegal indices so argmax ignores them
        illegal = (mask == 0.0)
        q[illegal] = -1e9
        return int(torch.argmax(q).item())



##################################################################################
##### ---> Figure out different rewards for buy phase and action phase. <--- #####
##################################################################################


def train_buy_phase(env, episodes=2, buffer_size=10_000, batch_size=64):
    input_dim  = env.observation_space.shape[0]
    n_actions  = env.action_space.n                   #  |card_names| + 1  (pass)

    policy_net = DQN(input_dim, n_actions).to(DEVICE)
    target_net = DQN(input_dim, n_actions).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    opt      = optim.Adam(policy_net.parameters(), lr=1e-3)
    loss_fn  = nn.MSELoss()
    replay   = deque(maxlen=buffer_size)

    gamma          = 0.99
    epsilon        = 1.0
    eps_decay      = 0.995
    eps_min        = 0.1
    target_update  = 10            # sync target every N episodes

    for ep in range(episodes):
        obs, _   = env.reset()
        done      = False
        step_ctr  = 0
        ep_reward = 0.0

        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_vals = policy_net(obs_tensor)
        print("Obs shape:", obs_tensor.shape)
        print("Q-values shape:", q_vals.shape)

        while not done:
            mask   = env.valid_action_mask()         #  ← NEW
            act    = select_action(obs, policy_net, mask, epsilon, n_actions)

            next_obs, r, done, _ = env.step(act)

            #  store transition -------------------------------------------------
            replay.append((obs, act, r, next_obs, done))
            obs        = next_obs
            ep_reward += r
            step_ctr  += 1

            #  ----------------------  SGD update  -----------------------------
            if len(replay) >= batch_size:
                batch = random.sample(replay, batch_size)
                # unpack and tensorise
                obs_b, act_b, rew_b, nxt_b, done_b = zip(*batch)
                obs_b  = torch.as_tensor(obs_b,  dtype=torch.float32, device=DEVICE)
                act_b  = torch.as_tensor(act_b,  dtype=torch.long,   device=DEVICE).unsqueeze(1)
                rew_b  = torch.as_tensor(rew_b,  dtype=torch.float32,device=DEVICE)
                nxt_b  = torch.as_tensor(nxt_b,  dtype=torch.float32,device=DEVICE)
                done_b = torch.as_tensor(done_b, dtype=torch.float32,device=DEVICE)

                # Q(s,a)
                q_sa   = policy_net(obs_b).gather(1, act_b).squeeze()

                # V(s')  (no mask in bootstrapping; agent can learn from it)
                with torch.no_grad():
                    v_nxt = target_net(nxt_b).max(1)[0]

                target = rew_b + gamma * v_nxt * (1.0 - done_b)
                loss   = loss_fn(q_sa, target)

                opt.zero_grad()
                loss.backward()
                opt.step()

        # --------------- episode end  -----------------
        epsilon = max(eps_min, epsilon * eps_decay)
        print(f"Ep {ep:04d} | steps={step_ctr:3d} | ε={epsilon:.3f} | reward={ep_reward:.3f}")

        if (ep + 1) % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
