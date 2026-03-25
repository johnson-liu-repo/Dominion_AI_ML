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

## Important Things to Know

- **Pyminion is the underlying game engine.** All game rules, card definitions, and turn mechanics live in `pyminion_master`. Treat it as vendor code; keep custom logic in `src/agent_rl`.
- **The RL environment is buy-phase only.** `DominionBuyPhaseEnv` (in `dominion_env.py`) internally advances the game through action and treasure phases, exposing only buy decisions to the agent.
- **Action space is fixed.** The environment uses a fixed action space (one per card + pass). This maintains DQN compatibility but requires masking invalid actions at runtime.
- **The training loop is intentionally minimal.** `train_dqn.py` is a small DQN baseline with a replay buffer, masking logic, and target network syncs. Use it as a starting point.
- **Local-path hacks exist.** Several files append a hardcoded path to `sys.path` to load the bundled Pyminion. This should be replaced with repo-relative imports for portability.

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
| **dominion_env_factory.py** | `make_env(cards, seed, opponent)` | Factory function; creates seeded environments with configurable card sets and scripted opponents | `DominionBuyPhaseEnv`, `DummieBot`, `pyminion` | `train_agent.py` (config entry point) |
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

## Design Decisions

### 1. Single-Player Simplification

We treat Dominion as a single-player optimization problem (the RL agent plays against a fixed scripted opponent). This simplifies game state representation and agent training.

**Rationale**: Eliminates need to model opponent behavior; focuses learning on agent's buy/action strategy against a known opponent.

### 2. Fixed vs. Dynamic Action Space

**Fixed Action Space (Chosen):**
- One action per card in supply + one "pass" action
- Size remains constant throughout an episode

| Aspect | Benefit | Cost |
|--------|---------|------|
| **Framework Compatibility** | DQN and most RL algorithms require fixed output size | Wastes action space early (many masks) |
| **Masked Learning** | Agent learns not to select invalid actions via masking + penalty | Slower convergence without careful reward shaping |
| **Implementation** | Simple to implement; no custom infrastructure | Requires robust invalid-action masking at every step |

**Dynamic Action Space (Alternative, not chosen):**
- Only expose valid actions at each state
- **Pros**: No wasted actions, faster exploration. **Cons**: Breaks DQN (requires variable output size); needs custom infrastructure.

### 3. Buy-Phase Only vs. Multi-Phase Agent

**Current**: Buy-phase only
- Simplifies observation/action space
- RL agent sees buying decisions; internal turn automation handles action/treasure/cleanup phases

**Future**: Multi-phase agent (optional)
- Extend `DominionBuyPhaseEnv` to `DominionFullEnv` that exposes action phase
- Share or separate Q-networks between action and buy phases
- Options: multi-headed network, phase-encoded observation, or separate models with curriculum learning

### 4. Reward Shaping

Observation: Raw terminal reward (final score difference) is sparse and leads to slow learning.

**Solution**: Shaped reward combining:
- Immediate reward: coins acquired during buy phase
- Terminal reward: final score - opponent score (scaled)
- Win bonus: +positive reward on victory

This accelerates learning without undermining long-term goals.

### 5. Checkpointing & Resumption

Training saves periodic checkpoints to `checkpoints/` directory:
- `checkpoint_latest.pt`: resumable state (model weights + optimizer)
- `checkpoint_best.pt`: highest performance (for inference)
- `checkpoint_XXXXX.pt`: snapshots at intervals

Enables long training runs that can be paused and resumed.

### 6. Separate Models vs. Shared Model (Action + Buy Phases)

If extending to multi-phase agents, options include:

| Approach | Advantage | Disadvantage |
|----------|-----------|--------------|
| **Separate models** | Clear separation of concerns; independent skill learning | 2× parameters; potential redundancy in learning |
| **Multi-headed network** | Shared body learns common features (coins, hand state); separate output heads | Slightly more complex; coordination overhead |
| **Phase-encoded single network** | Single model handles both phases; phase signal in observation | All outputs must have consistent size; less modular |
| **Curriculum learning** | Train separate models first, then fine-tune joint model | Multi-stage training; more bookkeeping |

**Recommendation**: Start with separate models for clarity. If memory/time is tight, move to multi-headed architecture.

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
