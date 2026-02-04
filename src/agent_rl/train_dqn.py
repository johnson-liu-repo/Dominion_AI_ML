
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

from agent_rl.logging_utils import configure_training_logging


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
        return self.net(x.to(next(self.parameters()).device))


def select_action(
        obs,
        policy_net,
        mask,
        epsilon,
        n_actions
    ):
    """Return an int in [0, n_actions).  Mask is a 1/0 numpy array."""
    if random.random() < epsilon:
        # ---------- random but legal ----------
        # logger.info(f"Random action selection...")
        valid_idxs = np.flatnonzero(mask)
        # logger.info(f"Valid indices for random selection: {valid_idxs}...")
        choice = int(np.random.choice(valid_idxs))
        # logger.info(f"Randomly selected action index {choice}...")
        return choice

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


def train_buy_phase(
        config,
        buffer_size=10000
    ):

    env          = config['env']
    episodes     = config.get('episodes', 500)
    turn_limit   = config.get('turn_limit', 50)
    batch_size   = config.get('batch_size', 64)
    gamma        = config.get('gamma', 0.99)
    epsilon      = config.get('epsilon', 1.0)
    eps_decay    = config.get('eps_decay', 0.995)
    eps_min      = config.get('eps_min', 0.1)
    target_update= config.get('target_update', 10)


    logger = configure_training_logging()
    input_dim  = env.observation_space.shape[0]
    n_actions  = env.action_space.n                   #  |card_names| + 1 (pass)

    policy_net = DQN(input_dim, n_actions).to(DEVICE)
    target_net = DQN(input_dim, n_actions).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    opt      = optim.Adam(policy_net.parameters(), lr=1e-3)
    loss_fn  = nn.MSELoss()
    replay   = deque(maxlen=buffer_size)

    for ep in range(episodes):
        logger.info(f"\n=== Starting episode {ep:04d} ===")
        obs, _    = env.reset()

        # logger.info("card_names -> supply piles mapping:")
        for name in env.card_names:
            pile = env._pile_for_card(name)
            # logger.info(f"  {name}  -->  {getattr(pile, 'name', 'UNMATCHED')}")
            
        done      = False
        step_ctr  = 0
        ep_reward = 0.0

        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            q_vals = policy_net(obs_tensor)
        # logger.info(f"Obs shape: {obs_tensor.shape}")
        # logger.info(f"Q-values shape: {q_vals.shape}")

        turn = 0

        # last_info = {}
        while not done:
            mask   = env.valid_action_mask()
            # logger.info(f"Money: {env.bot.state.money}\nBuys: {env.bot.state.buys}\nLegal: {np.flatnonzero(mask)}")
            act    = select_action(obs, policy_net, mask, epsilon, n_actions)

            next_obs, r, done, _ = env.step(act)
            # last_info = info or {}
            next_mask = env.valid_action_mask() if not done else np.zeros_like(mask)

            #  store transition -------------------------------------------------
            replay.append((obs, act, r, next_obs, done, next_mask))
            obs        = next_obs
            ep_reward += r
            step_ctr  += 1

            #  ----------------------  SGD update  -----------------------------
            if len(replay) >= batch_size:
                batch = random.sample(replay, batch_size)
                # unpack and tensorise
                obs_b, act_b, rew_b, nxt_b, done_b, nxt_mask_b = zip(*batch)
                obs_b  = torch.as_tensor(obs_b,  dtype=torch.float32, device=DEVICE)
                act_b  = torch.as_tensor(act_b,  dtype=torch.long,   device=DEVICE).unsqueeze(1)
                rew_b  = torch.as_tensor(rew_b,  dtype=torch.float32,device=DEVICE)
                nxt_b  = torch.as_tensor(nxt_b,  dtype=torch.float32,device=DEVICE)
                done_b = torch.as_tensor(done_b, dtype=torch.float32,device=DEVICE)
                nxt_mask_b = torch.as_tensor(nxt_mask_b, dtype=torch.float32, device=DEVICE)

                # Q(s,a)
                q_sa   = policy_net(obs_b).gather(1, act_b).squeeze()

                # V(s')  (mask illegal actions in bootstrapping)
                with torch.no_grad():
                    q_nxt = target_net(nxt_b)
                    q_nxt[nxt_mask_b == 0.0] = -1e9
                    v_nxt = q_nxt.max(1)[0]

                target = rew_b + gamma * v_nxt * (1.0 - done_b)
                loss   = loss_fn(q_sa, target)

                opt.zero_grad()
                loss.backward()
                opt.step()

            # can extract turn from bot object.
            turn += 1
            if turn >= turn_limit:
                done = True
                # logger.info("Turn limit reached, ending episode.")

        # --------------- episode end  -----------------
        epsilon = max(eps_min, epsilon * eps_decay)
        # score_diff = last_info.get("score_diff")

        RL_agent_score = env.bot.get_victory_points()

        opponents = [bot for bot in env.opponent_bots if bot != env.bot]
        opponent_scores = [opp.get_victory_points() for opp in opponents]

        logger.info("\n\n--- Episode Summary ---")

        logger.info(f"RL agent score: {RL_agent_score}")
        logger.info("Opponent scores:")
        for i, opp_score in enumerate(opponent_scores):
            logger.info(f"  {opponents[i].player_id}: {opp_score}")

        score_diff = np.average([RL_agent_score - opp_score for opp_score in opponent_scores]) if opponent_scores else None
        ep_reward += score_diff if score_diff is not None else 0.0

        if score_diff is not None:
            logger.info(f"Ep {ep:04d} | steps={step_ctr:3d} | epsilon={epsilon:.3f} | reward={ep_reward:.3f} | score_diff={score_diff:.1f}")
        else:
            logger.info(f"Ep {ep:04d} | steps={step_ctr:3d} | epsilon={epsilon:.3f} | reward={ep_reward:.3f}")

        if (ep + 1) % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
