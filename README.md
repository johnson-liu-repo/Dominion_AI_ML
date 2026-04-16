# Dominion AI — Reinforcement Learning for Dominion

——— Work in progress ———

This project explores reinforcement learning approaches to training bots that can play the deck-building board game *Dominion*. It builds on the [Pyminion](https://github.com/evanofslack/pyminion) library, providing an environment wrapper and experimental reinforcement learning agents.

Johnson Liu\
<sub><small>
GitHub: [@johnson-liu-code](https://github.com/johnson-liu-code)\
</small></sub>
<sup><small>
Email: [liujohnson.jl@gmail.com](mailto:liujohnson.jl@gmail.com)
</small></sup>

---

## Table of Contents

- [Project Overview](#project-overview)
  - [Goal](#goal)
  - [Current Progress](#current-progress)
  - [Future Plans](#future-plans)
- [General Structure](#general-structure)
- [Some notes](#some-notes)
- [Module Relationships & Data Flow](#module-relationships--data-flow)
  - [Dependency Graph (ASCII Diagram)](#dependency-graph-ascii-diagram)
  - [Module Relationship Table](#module-relationship-table)
  - [Data Flow Pipeline](#data-flow-pipeline)
  - [Output Artifact Structure](#output-artifact-structure)
- [Diagnostics](#diagnostics)
  - [Enable diagnostics](#enable-diagnostics)
  - [Diagnostics outputs](#diagnostics-outputs)
  - [Generate the plots and report](#generate-the-plots-and-report)
  - [Interpreting small runs](#interpreting-small-runs)
- [Design Decisions (potentially outdated section)](#design-decisions-potentially-outdated-section)
- [Detailed DQN Structure Overview (Comprehensive)](#detailed-dqn-structure-overview-comprehensive)
  - [DQN Structure (at a glance)](#dqn-structure-at-a-glance)
  - [1) Learning Setup: What the Agent Actually Learns](#1-learning-setup-what-the-agent-actually-learns)
  - [2) State (Observation) Structure](#2-state-observation-structure)
  - [3) Action-Space Constraints via Legal Masking](#3-action-space-constraints-via-legal-masking)
  - [4) Network Architecture: Dueling DQN MLP](#4-network-architecture-dueling-dqn-mlp)
  - [5) Exploration Strategy](#5-exploration-strategy)
  - [6) Replay Buffer and Off-Policy Learning](#6-replay-buffer-and-off-policy-learning)
  - [7) Target Computation: Double DQN with Masked Argmax](#7-target-computation-double-dqn-with-masked-argmax)
  - [8) Optimization and Stability Mechanisms](#8-optimization-and-stability-mechanisms)
  - [9) Reward Composition and Learning Signal](#9-reward-composition-and-learning-signal)
  - [10) Temporal Granularity and Credit Assignment](#10-temporal-granularity-and-credit-assignment)
  - [11) Logging, Checkpointing, and Diagnostics Integration](#11-logging-checkpointing-and-diagnostics-integration)
  - [12) Why This DQN Design Fits the Current Project Stage](#12-why-this-dqn-design-fits-the-current-project-stage)

---

## Project Overview

### Goal

Create a framework to train reinforcement learning agents to play Dominion.

### Current Progress

- **Environment**: Dominion game environment built on top of Pyminion (`dominion_env.py`, `dominion_env_factory.py`, `wrappers.py`)
- **Agents**: Includes `DummieBot` baseline and early DQN agent implementations (`train_dqn.py`)
- **Training I/O**: Full logging pipeline (`training_io.py`) with checkpoints, metrics, and turn events

### Future Plans

- Train robust DQN agents that outperform scripted baselines
- Experiment with curriculum learning (start with basic cards, progressively add complexity)
- Extend to multi-phase agents (action + buy decision-making)
- Compare shared vs. separate models for multi-phase learning
- Evaluate agent performance using custom metrics and comparisons to human-style strategies

---

## General Structure

The repository is organized around a small RL stack layered on top of a bundled copy of Pyminion:

```
Dominion_AI_ML/
├── README.md                        # Project overview + onboarding report
├── docs/                            # Research notes and reports
│   ├── reports/                     # Internal reports
│   └── research/                    # External reading material
├── scripts/                         # Standalone entry points and analysis
│   ├── train_agent.py               # Entry point for training experiments
│   ├── plot_episode_metrics.py      # Plot reward/score trends from CSV logs
│   ├── plot_final_deck_card_trends.py # Analyze deck composition evolution
│   ├── reorganize_run_artifacts.py  # Utility for managing output structure
│   └── logs/                        # Historical training logs
├── src/                             # Core RL implementation
│   ├── __init__.py
│   └── agent_rl/                    # RL environments, bots, and training loops
│       ├── __init__.py
│       ├── dominion_env.py          # DominionBuyPhaseEnv (Gymnasium wrapper)
│       ├── dominion_env_factory.py  # make_env() factory for seeded environments
│       ├── train_dqn.py             # train_buy_phase() DQN training loop
│       ├── dummie_bot.py            # DummieBot baseline agent
│       ├── training_io.py           # TrainingRunWriter, checkpoint I/O
│       ├── logging_utils.py         # Logging configuration
│       ├── run_dummy_agent.py       # Manual agent runner (debugging tool)
│       ├── wrappers.py              # Gym-style environment wrappers/aliases
│       ├── card_catalog.py          # BASE_CARDS enumeration
│       └── __pycache__/
├── data/                            # Training data and outputs
│   └── training/                    # Training run directories
│       ├── training_005/            # Example completed training run
│       │   ├── episode_data_over_time.csv    # Metrics: episode, steps, reward
│       │   ├── model_weights_over_time.csv   # Network parameters evolution
│       │   ├── final_decks.json              # Card composition per player
│       │   ├── episodes/                     # Turn-by-turn event logs (JSONL)
│       │   │   ├── episode_000000_turns.jsonl
│       │   │   ├── episode_000001_turns.jsonl
│       │   │   └── ...
│       │   └── checkpoints/                  # Model snapshots (.pt)
│       │       ├── checkpoint_latest.pt
│       │       ├── checkpoint_best.pt
│       │       └── ...
│       ├── training_006/
│       └── training_007/
├── claude/                          # Claude model evaluation data
│   ├── episode_010000_turns.jsonl   # Turn-by-turn records from Claude runs
│   ├── episode_010001_turns.jsonl
│   └── ...
├── pyminion_master/                 # Bundled Pyminion repo (vendor—do not edit)
│   ├── LICENSE
│   ├── README.md
│   ├── setup.py
│   ├── pyminion/                    # Game engine (Game, Player, Bot, cards)
│   │   ├── core.py                  # Card, DeckCounter, Supply abstractions
│   │   ├── game.py                  # Game loop and phase management
│   │   ├── player.py                # Player state (deck, hand, discard)
│   │   ├── bot.py                   # Bot interface and examples
│   │   ├── effects.py               # Card effect system
│   │   ├── exceptions.py
│   │   ├── duration.py
│   │   ├── decider.py
│   │   ├── human.py
│   │   ├── result.py
│   │   ├── simulator.py
│   │   ├── expansions/              # Card definitions (base, intrigue, seaside, alchemy)
│   │   │   ├── base.py              # Base set cards (Copper, Silver, Province, etc.)
│   │   │   ├── intrigue.py
│   │   │   ├── seaside.py
│   │   │   └── alchemy.py
│   │   ├── bots/                    # Bot implementations
│   │   │   ├── bot.py               # Base Bot class
│   │   │   ├── optimized_bot.py
│   │   │   └── custom_bots/
│   │   └── __pycache__/
│   ├── tests/
│   └── examples/
├── replay_viewer/                   # HTML replay viewer
│   └── dominion_replay_viewer.html
├── __pycache__/
└── (Note: __pycache__/ and .pyc files are auto-generated and ignored)
```

---

## Some notes

- **Pyminion is the underlying game engine.** All game rules, card definitions, and turn mechanics live in `pyminion_master`. Treat it as vendor code; keep custom logic in `src/agent_rl`.
- **The RL environment is buy-phase only.** `DominionBuyPhaseEnv` (in `dominion_env.py`) internally advances the game through action and treasure phases, exposing only buy decisions to the agent.
- **Action space is fixed.** The environment uses a fixed action space (one per card + pass).
- **The training loop is intentionally minimal.** `train_dqn.py` is a small DQN baseline with a replay buffer, masking logic, and target network syncs. This is just a starting point.

---

## Module Relationships & Data Flow

### Dependency Graph (ASCII Diagram)

```
scripts/train_agent.py  [CONFIG & ENTRY POINT]
    │
    ├─→ agent_rl.dominion_env_factory.make_env()
    │       │
    │       ├─→ pyminion.Game (from pyminion_master/)
    │       ├─→ pyminion.expansions.base (BASE_CARDS)
    │       ├─→ agent_rl.dummie_bot.DummieBot [RL Agent]
    │       │       └─→ pyminion.bots.Bot (base class)
    │       │
    │       └─→ agent_rl.wrappers.DominionBuyPhaseEnv
    │               └─→ agent_rl.dominion_env.DominionBuyPhaseEnv (Gymnasium wrapper)
    │                   └─→ pyminion.Game (manages game loop)
    │
    ├─→ agent_rl.train_dqn.train_buy_phase() [TRAINING LOOP]
    │       │
    │       ├─→ torch.nn.Module (DQN network)
    │       ├─→ agent_rl.training_io.TrainingRunWriter [LOGGING]
    │       │       └─→ data/training/training_XXX/ (checkpoints, CSV, JSONL)
    │       │
    │       └─→ agent_rl.training_utils.EpisodeProgress [METRICS]
    │
    └─→ agent_rl.card_catalog.BASE_CARDS [CARD ENUMERATION]

ANALYSIS PIPELINE (post-training):
    agent_rl.training_io [Read artifacts]
        │
        ├─→ scripts/plot_episode_metrics.py [Episode metrics plotting]
        ├─→ scripts/plot_final_deck_card_trends.py [Deck composition analysis]
        └─→ scripts/reorganize_run_artifacts.py [Output structure management]
```

### Module Relationship Table

| **Module** | **Key Classes/Functions** | **Purpose** | **Depends On** | **Used By** |
|---|---|---|---|---|
| **dominion_env.py** | `DominionBuyPhaseEnv` (class) | Wraps Pyminion Game as Gymnasium environment; exposes buy-phase decisions; internally runs action/treasure/cleanup phases | `pyminion.Game`, `numpy`, `gymnasium` | `dominion_env_factory.make_env()`, `train_dqn.train_buy_phase()` |
| **dominion_env_factory.py** | `make_env(cards, seed, opponent)` | Factory function; creates seeded 2-player environments with configurable card sets and one scripted opponent | `DominionBuyPhaseEnv`, `DummieBot`, `pyminion` | `train_agent.py` (config entry point) |
| **train_dqn.py** | `train_buy_phase(config)` | Main DQN training loop; epsilon-greedy exploration, replay buffer, target network syncs, action masking | `torch`, `training_io.TrainingRunWriter`, environment from `make_env()` | `train_agent.py` |
| **dummie_bot.py** | `DummieBot` (class) | Simple baseline bot with hardcoded buy priority (Province > Duchy > Estate); implements `pyminion.bots.Bot` interface | `pyminion.bots.Bot`, `pyminion.expansions.base` | `dominion_env_factory.make_env()` (as RL agent) |
| **training_io.py** | `TrainingRunWriter` (class) | Serializes training artifacts; saves checkpoints (.pt), episode metrics (CSV), turn events (JSONL), manages data/training/run_XXX/ structure | `pathlib`, `torch`, `json`, `csv` | `train_dqn.train_buy_phase()` |
| **logging_utils.py** | `setup_logging()` | Configures Python logging for training sessions | `logging` | `train_agent.py`, training modules |
| **wrappers.py** | `DominionBuyPhaseEnv` (alias) | Lightweight wrapper/alias; placeholder for future multi-phase extensions (e.g., `ActionPhaseEnv`) | `dominion_env.DominionBuyPhaseEnv` | `dominion_env_factory.make_env()` |
| **card_catalog.py** | `BASE_CARDS` (dict/list) | Enumerated list of available cards in base Dominion set; card name → ID mappings | — | `dominion_env.py`, `dominion_env_factory.py` |
| **run_dummy_agent.py** | `main()` | Manual agent runner for debugging; runs environment without training | `dominion_env_factory.make_env()` | Direct CLI invocation |

### Data Flow Pipeline

**Typical training execution:**

1. **Configuration** → `scripts/train_agent.py` loads hyperparameters (gamma, epsilon, LR, card set, seed)

2. **Environment Creation** → `dominion_env_factory.make_env()` instantiates:
   - Pyminion `Game` object with the specified card set
   - `DummieBot` as the scripted RL agent
   - Wraps it as a `DominionBuyPhaseEnv` (Gymnasium interface)

3. **Episode Loop** (inside `train_dqn.train_buy_phase()`):
   - `env.reset()` → Initial observation (supply state, hand state, resources)
   - DQN Q-network selects action via ε-greedy: $Q(s, a) = \text{DQN}(s)$ masked to valid actions
   - `env.step(action)` → 
     - Internally executes: buy phase → cleanup phase → opponent turns → next buy phase
     - Returns: `(observation', reward, done, info)`
   - Store `(s, a, r, s', done)` in replay buffer
   - Sample minibatch → compute TD loss → gradient step
   - Update target network every $N$ steps
   - Decay ε

4. **Logging** (via `training_io.TrainingRunWriter`):
   - Every episode: append to `episode_data_over_time.csv` (episode#, total_reward, epsilon, etc.)
   - Every turn: append to `episodes/episode_XXXXX_turns.jsonl` (hand, buys, supply, money snapshot)
   - Every checkpoint: save model weights to `checkpoints/checkpoint_latest.pt` and roll up best model
   - Final: save `final_decks.json` (deck composition per player)

5. **Evaluation** (post-training):
   - `scripts/plot_episode_metrics.py`: reads `episode_data_over_time.csv` → plots reward trends
   - `scripts/plot_final_deck_card_trends.py`: reads `final_decks.json` + episodes/ → analyzes card acquisition patterns
   - `scripts/plot_diagnostics.py`: reads `diagnostics/*.csv` → generates training-diagnostics plots
   - `scripts/generate_diagnostics_report.py`: reads `diagnostics/*.csv` → writes a plain-text diagnostics summary
   - `scripts/reorganize_run_artifacts.py`: tidies output directory structure

### Output Artifact Structure

Each training run in `data/training/training_XXX/` produces:

| **File/Dir** | **Content** | **Format** |
|---|---|---|
| `episode_data_over_time.csv` | Aggregate metrics per episode: episode #, total steps, total reward, epsilon, loss | CSV (header: episode, steps, reward, epsilon, loss, ...) |
| `model_weights_over_time.csv` | Network parameter statistics (optional); can track weight norms, gradient magnitudes | CSV |
| `final_decks.json` | Terminal deck composition for each player at end of training | JSON dict: `{player_name: [card, card, ...]}` |
| `episodes/episode_XXXXX_turns.jsonl` | Turn-by-turn event log per episode; each line is a turn snapshot | JSONL (one JSON object per turn with keys: hand, buys, money, supply_state, etc.) |
| `checkpoints/checkpoint_latest.pt` | Most recent model weights + optimizer state; for resuming training | PyTorch .pt (serialized state_dict) |
| `checkpoints/checkpoint_best.pt` | Best model by eval metric (highest reward or win rate); for inference | PyTorch .pt |
| `checkpoints/checkpoint_XXXXX.pt` | Periodic snapshots (every N episodes) | PyTorch .pt |

---

## Diagnostics

The training pipeline can collect extra diagnostics about the RL agent's buy decisions, affordability windows, and end-of-episode deck composition.

### Enable diagnostics

Set `"enable_diagnostics": true` under `"training"` in `config/train_agent.json`.

Example:

```json
{
  "training": {
    "enable_diagnostics": true
  }
}
```

Then run training normally:

```powershell
python scripts/train_agent.py --config config/train_agent.json
```

Each run writes a `diagnostics/` directory under `data/training/training_XXX/`.

### Diagnostics outputs

The `diagnostics/` directory contains:

- `episode_summary.csv`: one row per episode with win/loss, reward, coin thresholds, buy counts, and final deck composition
- `buy_decisions.csv`: one row per RL buy decision with current coins, affordable cards, selected action, epsilon, and Q-value context
- `rolling_metrics.csv`: rolling aggregates for economy development, purchase timing, greening tempo, conditional choices, and training-process metrics
- `coin_bucket_summary.csv`: per-episode coin-bucket action and pass-rate summaries (buckets 2,3,4,5,6,7,8+)
- `training_step_metrics.csv`: per-update loss/TD/Q diagnostics for train-process analysis
- `results/plots/*.png`: generated diagnostics charts
- `results/diagnostics_report.txt`: plain-text summary and interpretation

### Generate the plots and report

Replace `training_XXX` with the run you want to inspect.

```powershell
python scripts/plot_diagnostics.py data/training/training_XXX/diagnostics
python scripts/generate_diagnostics_report.py data/training/training_XXX/diagnostics
```

Example:

```powershell
python scripts/plot_diagnostics.py data/training/training_010/diagnostics
python scripts/generate_diagnostics_report.py data/training/training_010/diagnostics
```

### Interpreting small runs

Short runs produce noisy metrics. With very small numbers of episodes, rolling diagnostics are still generated, but the charts should be treated as sanity checks rather than evidence of learning. For meaningful trend analysis, use substantially more than 10 episodes.

---

## Design Decisions (potentially outdated section)

### 1. Single-Player Simplification

We treat Dominion as a single-player optimization problem (the RL agent plays against a fixed scripted opponent). This simplifies game state representation and agent training.

This eliminates need to model opponent behavior; focuses learning on agent's buy/action strategy against a known opponent.

### 2. Buy-Phase Only vs. Multi-Phase Agent

**Current**: Buy-phase only
- Simplifies observation/action space
- RL agent sees buying decisions; internal turn automation handles action/treasure/cleanup phases

**Future**: Multi-phase agent
- Extend `DominionBuyPhaseEnv` to `DominionFullEnv` that exposes action phase
- Share or separate Q-networks between action and buy phases
- Options: multi-headed network, phase-encoded observation, or separate models with curriculum learning

### 3. Separate Models vs. Shared Model (Action + Buy Phases)

If extending to multi-phase agents, options include:

| Approach | Advantage | Disadvantage |
|----------|-----------|--------------|
| **Separate models** | Clear separation of concerns; independent skill learning | 2× parameters; potential redundancy in learning |
| **Multi-headed network** | Shared body learns common features (coins, hand state); separate output heads | Slightly more complex; coordination overhead |
| **Phase-encoded single network** | Single model handles both phases; phase signal in observation | All outputs must have consistent size; less modular |
| **Curriculum learning** | Train separate models first, then fine-tune joint model | Multi-stage training; more bookkeeping |


---

## Detailed DQN Structure Overview

This section provides a deeper technical breakdown of the DQN implementation used in `src/agent_rl/train_dqn.py` and how it interacts with the buy-phase environment.

### DQN Structure (at a glance)

The implemented network is a **dueling DQN** with one shared MLP trunk and two output heads:

| Stage | Layer | Output shape (single sample) | Purpose |
|---|---|---|---|
| Input | Observation vector | `(4K + 6,)` | Encodes supply, hand, deck zones, opponent zones, and scalar turn/economy features |
| Normalize | `LayerNorm(input_dim)` | `(4K + 6,)` | Stabilizes feature scales across heterogeneous numeric inputs |
| Trunk-1 | `Linear(input_dim, 256)` + ReLU | `(256,)` | First shared representation layer |
| Trunk-2 | `Linear(256, 256)` + ReLU | `(256,)` | Higher-capacity shared feature extraction |
| Trunk-3 | `Linear(256, 128)` + ReLU | `(128,)` | Compressed latent state representation |
| Value stream | `Linear(128,128)` + ReLU + `Linear(128,1)` | `(1,)` | Estimates scalar state value `V(s)` |
| Advantage stream | `Linear(128,128)` + ReLU + `Linear(128,K+1)` | `(K+1,)` | Estimates per-action advantages `A(s,a)` |
| Aggregation | `Q = V + (A - mean(A))` | `(K+1,)` | Produces final Q-values over all buy actions + pass |

In other words, for each observation, the network outputs one Q-value per action slot in the fixed action space (`K` cards + pass). Illegal actions are then masked out before greedy action selection and before next-state argmax in Double DQN target computation.

### 1) Learning Setup: What the Agent Actually Learns

The current agent learns a **buy-phase policy** only. At each environment step, it chooses one discrete action from:

- `0..K-1`: buy one of the `K` tracked cards in `card_names`
- `K`: pass (buy nothing)

The action dimension is therefore `K + 1`.

Formally, the network approximates:

\[
Q_\theta(s, a) \approx \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t r_{t+1} \mid s_0=s, a_0=a\right]
\]

where the transition dynamics are induced by full Dominion turn progression (agent buy + cleanup + opponent turns + next agent turn).

---

### 2) State (Observation) Structure

The environment observation is a flat vector with shape:

\[
\text{obs\_dim} = 4K + 6
\]

concatenating:

1. **Supply counts** (`K`): remaining cards per supply pile
2. **Agent hand counts** (`K`): card histogram for current hand
3. **Agent zone counts** (`K`): deck + hand + discard histogram
4. **Opponent zone counts** (`K`): single-opponent deck + hand + discard histogram
5. **Scalar features** (`6`):
   - actions remaining
   - buys remaining
   - money available
   - turn index
   - current score difference (agent minus opponent)
   - total owned cards in agent zones

This representation mixes **economy state**, **deck composition**, **supply depletion**, and **game progression** in one vector suitable for MLP processing.

---

### 3) Action-Space Constraints via Legal Masking

Although the action space is fixed-size, Dominion legality changes by state. The implementation constructs a binary mask each step:

- action is legal if pile exists, pile non-empty, cost ≤ current money, and buys > 0
- pass action is always legal

Masking is used in two places:

1. **Behavior policy** (selection): illegal actions are forced to very negative Q before argmax
2. **Bootstrap target** (Double DQN): next-state argmax is computed only over legal actions

This is critical: it prevents the learner from exploiting impossible actions and reduces overestimation artifacts from invalid max operations.

---

### 4) Network Architecture: Dueling DQN MLP

The model class is `DuelingDQN`, with a shared trunk plus value/advantage heads.

#### Shared trunk

- `LayerNorm(input_dim)`
- `Linear(input_dim, 256)` + ReLU
- `Linear(256, 256)` + ReLU
- `Linear(256, 128)` + ReLU

#### Heads

- **Value head**: `128 -> 128 -> 1`
- **Advantage head**: `128 -> 128 -> (K+1)`

Combined as:

\[
Q(s,a)=V(s) + \left(A(s,a) - \frac{1}{|\mathcal{A}|}\sum_{a'}A(s,a')\right)
\]

This dueling factorization helps when many actions have similar effects by allowing the network to separately model:

- overall state quality (`V(s)`)
- relative preference among actions (`A(s,a)`)

LayerNorm at input stabilizes training under heterogeneous feature scales (supply counts, score deltas, money, turn index, etc.).

---

### 5) Exploration Strategy

The policy is **epsilon-greedy** with multiplicative decay:

- Start: `epsilon` (default 1.0)
- Per step update: `epsilon <- max(eps_min, epsilon * eps_decay)`
- Floor: `eps_min` (default 0.05)

When exploring, a random action is sampled **only from legal actions**. This keeps exploration realistic and avoids wasting transitions on impossible moves.

---

### 6) Replay Buffer and Off-Policy Learning

Transitions are stored in a uniform replay buffer (`deque`), each tuple containing:

\[
(s_t, a_t, r_t, s_{t+1}, done_t, mask_{t+1})
\]

where `mask_{t+1}` is explicitly persisted so target computation can respect future legality at train time.

Training starts once replay size reaches `batch_size`, then each update samples a random minibatch to reduce temporal correlation and improve sample efficiency.

---

### 7) Target Computation: Double DQN with Masked Argmax

For each sampled transition, the training code computes:

1. **Current estimate**

$Q_\theta(s_t,a_t)$

2. **Action selection network** (online net, masked legality)

$a^* = \arg\max_a Q_\theta(s_{t+1},a)\quad\text{with illegal actions excluded}$

3. **Action evaluation network** (target net)

$V_{t+1}=Q_{\bar\theta}(s_{t+1},a^*)$

4. **Bootstrapped target**

$y_t = r_t + \gamma(1-d_t)V_{t+1}$

5. **Loss**

$\mathcal{L}=\text{MSE}\left(Q_\theta(s_t,a_t), y_t\right)$

This is Double DQN because argmax uses the online network while value extraction uses the target network.

---

### 8) Optimization and Stability Mechanisms

- Optimizer: **Adam** (`lr=1e-3`)
- Loss: **MSELoss**
- Gradient step: one update per environment step after warmup threshold
- Target sync: hard copy `policy_net -> target_net` every `target_update` global steps
- Device: CUDA if available, else CPU

The hard-sync target network reduces moving-target instability during bootstrapped learning.

---

### 9) Reward Composition and Learning Signal

The step reward in the buy-phase environment is shaped from multiple terms:

- Illegal action penalty
- Pass penalty (small negative)
- Card-type-dependent shaping (curse/victory/treasure effects)
- Deck dilution penalty based on owned-card count
- Terminal/game-end score-based adjustments
- End-of-episode score differential augmentation in trainer

This produces a dense training signal intended to accelerate learning of economically coherent purchase behavior, while still tying outcomes to final Dominion scoring dynamics.

---

### 10) Temporal Granularity and Credit Assignment

One RL `step` corresponds to **one agent buy decision**, but the environment transition spans:

1. agent buy
2. cleanup/end-turn
3. full opponent turns
4. start of next agent buy phase

Therefore the MDP is effectively semi-aggregated at decision points. The Q-function must learn long-horizon consequences of buy choices through intervening opponent and draw dynamics.

---

### 11) Logging, Checkpointing, and Diagnostics Integration

The DQN loop is tightly integrated with run artifact tooling:

- episode metrics CSV logging
- turn-event JSONL persistence
- periodic/latest checkpoint snapshots (`policy`, `target`, optimizer, epsilon, step counters)
- optional diagnostics:
  - buy decisions + chosen Q values
  - top-k legal Q candidates
  - training-step loss / TD-error traces
  - episode summary and rolling metrics

This makes the implementation useful not only for training, but for **policy-behavior analysis** and debugging failure modes (e.g., over-passing, greening too early, low-economy traps).

---

### 12) Why This DQN Design Fits the Current Project Stage

The current design is a pragmatic baseline for Dominion buy-phase RL because it combines:

- simple vector observations (fast experimentation)
- legal-action masking (domain correctness)
- dueling architecture (better action discrimination)
- Double DQN targets (reduced overestimation)
- replay + target net (standard deep RL stability toolkit)
- rich artifact logging (analysis-first workflow)

In short, it is not yet the final architecture for full Dominion mastery, but it is a strong and extensible baseline for iterative research on buy-phase policy quality, reward shaping, and deck-building dynamics.

---

<!-- ![Test image 2](test_02.png) -->
<!-- ![Test image 3](test_03.png) -->
