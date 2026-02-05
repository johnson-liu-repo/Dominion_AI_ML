"""Minimal DQN training loop for the Dominion buy-phase environment."""

import sys
import time
import shutil
from pathlib import Path
from collections import deque
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# from agent_rl.logging_utils import configure_training_logging
from agent_rl.training_io import TrainingRunWriter, load_checkpoint, resolve_run_dir


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _format_seconds(seconds):
    if seconds is None or seconds == float("inf"):
        return "?"
    seconds = max(0, int(seconds))
    mins, secs = divmod(seconds, 60)
    hours, mins = divmod(mins, 60)
    if hours:
        return f"{hours:d}h{mins:02d}m{secs:02d}s"
    if mins:
        return f"{mins:d}m{secs:02d}s"
    return f"{secs:d}s"


class EpisodeProgress:
    def __init__(self, total_episodes, start_time, bar_width=30, stream=None):
        self.total_episodes = max(0, int(total_episodes))
        self.start_time = start_time
        self.bar_width = max(10, int(bar_width))
        self.stream = stream or sys.stdout
        self._last_len = 0
        self._term_cols_fallback = 80

    def update(self, episode_id):
        if self.total_episodes <= 0:
            return
        episode_id = max(0, int(episode_id))
        completed = min(episode_id, self.total_episodes)
        frac = completed / self.total_episodes if self.total_episodes else 1.0

        elapsed = time.time() - self.start_time
        rate = completed / elapsed if elapsed > 0 else 0.0
        remaining = self.total_episodes - completed
        eta = remaining / rate if rate > 0 else None

        # Keep the line short enough to avoid terminal wrapping, which can look
        # like the progress bar "prints on new lines" even when using `\r`.
        cols = self._term_cols_fallback
        try:
            cols = shutil.get_terminal_size(fallback=(self._term_cols_fallback, 20)).columns
        except OSError:
            cols = self._term_cols_fallback
        max_cols = max(20, int(cols) - 1)  # stay under the last column to avoid wrap

        elapsed_s = _format_seconds(elapsed)
        eta_s = _format_seconds(eta)
        pct_s = f"{frac * 100:5.1f}%"

        # Prefer showing both elapsed+ETA, then only ETA, then only percent.
        suffix_candidates = [
            f"] {pct_s} elapsed {elapsed_s} ETA {eta_s}",
            f"] {pct_s} ETA {eta_s}",
            f"] {pct_s}",
        ]

        prefix = f"Ep {completed}/{self.total_episodes} ["
        chosen_suffix = suffix_candidates[-1]
        bar_width = self.bar_width
        for suffix in suffix_candidates:
            allowed = max_cols - len(prefix) - len(suffix)
            if allowed >= 5:
                chosen_suffix = suffix
                bar_width = min(self.bar_width, allowed)
                break
            if len(prefix) + len(suffix) <= max_cols:
                # Fits without a bar.
                chosen_suffix = suffix
                bar_width = 0
                break

        if bar_width > 0:
            filled = int(bar_width * frac)
            bar = "#" * filled + "-" * (bar_width - filled)
            line = prefix + bar + chosen_suffix
        else:
            # No room for a meaningful bar; just print stats.
            show_elapsed = "elapsed " in chosen_suffix
            show_eta = "ETA " in chosen_suffix
            line = f"Ep {completed}/{self.total_episodes} {pct_s}"
            if show_elapsed:
                line += f" elapsed {elapsed_s}"
            if show_eta:
                line += f" ETA {eta_s}"

        if len(line) > max_cols:
            line = line[:max_cols]

        # Update the same line in-place.
        pad = " " * max(0, self._last_len - len(line))
        self.stream.write("\r" + line + pad)
        self.stream.flush()
        self._last_len = len(line)

    def close(self):
        self.stream.write("\n")
        self.stream.flush()


#####################################################
##### ---> Try different NN architectures. <--- #####
#####################################################


class DuelingDQN(nn.Module):
    """Dueling network with a 3-layer MLP trunk for discrete action selection."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.adv_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        """Forward pass with device-aware tensor conversion."""
        squeeze = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze = True
        x = x.to(next(self.parameters()).device)
        h = self.trunk(x)
        v = self.value_head(h)
        a = self.adv_head(h)
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q.squeeze(0) if squeeze else q


def select_action(
        obs,
        policy_net,
        mask,
        epsilon,
        n_actions
    ):
    """
    Choose an action index with epsilon-greedy exploration.

    Args:
        obs: Current observation vector.
        policy_net: Torch model used to score actions.
        mask: 1/0 numpy array indicating legal actions.
        epsilon: Probability of random action.
        n_actions: Total number of action slots (including pass).
    """
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
        if q.dim() == 1:
            q[illegal] = -1e9
        else:
            q[:, illegal] = -1e9
        return int(torch.argmax(q).item())



##################################################################################
##### ---> Figure out different rewards for buy phase and action phase. <--- #####
##################################################################################


def train_buy_phase(
        config,
        buffer_size=10000
    ):

    env          = config['env']
    episodes     = config.get('episodes', 200_000)
    turn_limit   = config.get('turn_limit', 250)
    batch_size   = config.get('batch_size', 64)
    gamma        = config.get('gamma', 0.99)
    epsilon      = config.get('epsilon', 1.0)
    eps_decay    = config.get('eps_decay', 0.9995)
    eps_min      = config.get('eps_min', 0.05)
    target_update= config.get('target_update', 1_000)
    output_dir   = config.get('output_dir')
    run_dir      = config.get('run_dir')
    resume_from  = config.get('resume_from')
    checkpoint_every = config.get('checkpoint_every', 100)
    latest_every = config.get('latest_every', 10)
    save_turns   = config.get('save_turns', True)
    save_turns_every = config.get('save_turns_every', 1)
    progress_bar = config.get('progress_bar', True)


    # logger = configure_training_logging()
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = Path(output_dir) if output_dir else (repo_root / "data" / "training")
    run_dir = resolve_run_dir(output_dir, run_dir=run_dir, resume_from=resume_from)
    writer = TrainingRunWriter(run_dir)
    input_dim  = env.observation_space.shape[0]
    n_actions  = env.action_space.n                   #  |card_names| + 1 (pass)

    policy_net = DuelingDQN(input_dim, n_actions).to(DEVICE)
    target_net = DuelingDQN(input_dim, n_actions).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    opt      = optim.Adam(policy_net.parameters(), lr=1e-3)
    loss_fn  = nn.MSELoss()
    replay   = deque(maxlen=buffer_size)

    global_step = 0
    start_episode = 0
    start_time = time.time()
    progress = EpisodeProgress(episodes, start_time) if progress_bar else None

    if resume_from:
        payload = load_checkpoint(
            resume_from,
            policy_net=policy_net,
            target_net=target_net,
            optimizer=opt,
            device=DEVICE,
        )
        epsilon = payload.get("epsilon", epsilon)
        global_step = payload.get("global_step", global_step)
        start_episode = payload.get("episode", start_episode) + 1

    total_episodes = max(0, episodes - start_episode)
    for ep in range(start_episode, episodes):
        # logger.info(f"\n=== Starting episode {ep:04d} ===")
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
            global_step += 1

            epsilon = max(eps_min, epsilon * eps_decay)

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

                # Double DQN target (mask illegal actions before argmax)
                with torch.no_grad():
                    q_online = policy_net(nxt_b)
                    q_online[nxt_mask_b == 0.0] = -1e9
                    next_act = q_online.argmax(1, keepdim=True)

                    q_target = target_net(nxt_b)
                    v_nxt = q_target.gather(1, next_act).squeeze(1)

                target = rew_b + gamma * v_nxt * (1.0 - done_b)
                loss   = loss_fn(q_sa, target)

                opt.zero_grad()
                loss.backward()
                opt.step()

            if global_step % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())

            # can extract turn from bot object.
            turn += 1
            if turn >= turn_limit:
                done = True
                # logger.info("Turn limit reached, ending episode.")

        # --------------- episode end  -----------------
        RL_agent_score = env.bot.get_victory_points()

        opponents = [bot for bot in env.opponent_bots if bot != env.bot]
        opponent_scores = [opp.get_victory_points() for opp in opponents]

        # logger.info("\n\n--- Episode Summary ---")

        # logger.info(f"RL agent score: {RL_agent_score}")
        # logger.info("Opponent scores:")
        # for i, opp_score in enumerate(opponent_scores):
            # logger.info(f"  {opponents[i].player_id}: {opp_score}")

        score_diff = np.average([RL_agent_score - opp_score for opp_score in opponent_scores]) if opponent_scores else None
        ep_reward += score_diff if score_diff is not None else 0.0

        # if score_diff is not None:
            # logger.info(f"Ep {ep:04d} | steps={step_ctr:3d} | epsilon={epsilon:.3f} | reward={ep_reward:.3f} | score_diff={score_diff:.1f}")
        # else:
            # logger.info(f"Ep {ep:04d} | steps={step_ctr:3d} | epsilon={epsilon:.3f} | reward={ep_reward:.3f}")

        episode_id = ep + 1
        writer.log_episode({
            "episode": episode_id,
            "steps": step_ctr,
            "reward": float(ep_reward),
            "score_diff": float(score_diff) if score_diff is not None else "",
            "epsilon": float(epsilon),
            "timestamp": time.time(),
        })

        if hasattr(env, "game") and hasattr(env.game, "players"):
            writer.log_final_decks(episode_id, env.game.players)

        turn_events = env.consume_turn_events()
        should_save_turns = save_turns and save_turns_every and (episode_id % save_turns_every == 0)
        if should_save_turns:
            for event in turn_events:
                event["episode"] = episode_id
            writer.write_turns(episode_id, turn_events)

        should_save_latest = latest_every and (episode_id % latest_every == 0)
        should_save_checkpoint = checkpoint_every and (episode_id % checkpoint_every == 0)
        if ep == episodes - 1:
            should_save_latest = True
            should_save_checkpoint = True

        if should_save_latest or should_save_checkpoint:
            payload = {
                "episode": ep,
                "global_step": global_step,
                "epsilon": epsilon,
                "policy_state": policy_net.state_dict(),
                "target_state": target_net.state_dict(),
                "opt_state": opt.state_dict(),
            }

        if should_save_latest:
            writer.save_checkpoint(payload, "checkpoint_latest")

        if should_save_checkpoint:
            ckpt_path = writer.save_checkpoint(payload, f"checkpoint_ep_{episode_id:06d}")
            writer.log_weights_checkpoint(episode_id, ckpt_path)

        if progress:
            progress.update(episode_id)

    if progress:
        progress.close()
