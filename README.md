# Dominion AI — Reinforcement Learning for Dominion

This project explores reinforcement learning approaches to training bots that can play the deck-building board game *Dominion*. It builds on the [Pyminion](https://github.com/evanofslack/pyminion) library, providing an environment wrapper and experimental reinforcement learning agents.


Johnson Liu\
<sub><small>
GitHub: [@johnson-liu-code](https://github.com/johnson-liu-code)\
</small></sub>
<sup><small>
Email: [liujohnson.jl@gmail.com](mailto:liujohnson.jl@gmail.com)
</small></sup>

<!-- ## __Contents__
1. [Resources Used](#resources-used)
    1. [Pyminion Reposity](#pyminion-repository) -->

### Pyminion Repository
By Evan\
[https://github.com/evanofslack/pyminion](https://github.com/evanofslack/pyminion)

---

## Project Overview

### Goal
- Create an AI framework to train reinforcement learning agents to play Dominion.

### Current Progress  
- **Environment**: Dominion game environment built on top of Pyminion (`dominion_env.py`, `dominion_env_factory.py`, `wrappers.py`).  
- **Agents**: Includes a `dummie_bot.py` as a baseline and early RL agent implementations (`train_dqn.py`).  

### Future Plans  
- Extend beyond dummy agents to train working DQN agents against scripted opponents.  
- Experiment with curriculum learning for progressively harder card sets.  
- Evaluate agent performance using custom metrics and comparisons to human-style strategies.  

---

## Repository Structure  
```
Dominion_AI_ML/
│── README.md                  # Project documentation
│── docs/                      # Reports and research material
│   ├── reports/
│   │   └── refactor_report.pdf # Organization refactor summary
│   └── research/
│       └── Reinforcement_Learning_for_Board_Game_AI_Comprehensive_Overview.pdf
│── scripts/                   # Runnable entry points
│   └── train_agent.py          # Entry point for experiments
│── src/
│   └── agent_rl/               # RL environment and agent code
│       ├── dominion_env.py     # Dominion environment
│       ├── dominion_env_factory.py # Environment factory
│       ├── dummie_bot.py       # Simple baseline bot
│       ├── run_dummy_agent.py  # Run dummy agent
│       ├── train_dqn.py        # Initial DQN training loop
│       ├── wrappers.py         # Gym-style wrappers
│── pyminion_master/            # Bundled Pyminion repo
│── requirements.txt (TBD)
```


---

### Notes - format this later
1. design decisions

    1. treat this as a single player game to simplify the game state

    1. fixed action space over dynamic action space...
        1. fixed action space

            Pros:
           
            1. Keeps the action space size constant → compatible with DQN and most RL frameworks.
            1. The bot can learn from state (e.g., hand cards, coins) not to select invalid actions.

            Cons:

            1. Wastes actions early in training.
            1. Slower convergence unless you use invalid-action penalties or masking.

        1. dynamic action space

            Pros:

            1. Efficient exploration (agent can’t select invalid actions).
            1. Faster early training.

            Cons:

            1. Requires more complex custom infrastructure.
            1. DQN assumes fixed output size → this breaks it.
    
    1. applying Deep Q-learning to action and buy phases separately? can also use a shared model between action and buy phases.
    
        1. shared model

            1. Option 1: Multi-Headed Network
            
                One shared body (input → hidden layers)

                Two separate output “heads”: one for action, one for buy.

            2. Option 2: Single Policy, Phase Encoded
            
                Encode the phase ("action" or "buy") into the observation vector.

                Train a single DQN with one output head, always with the same output size.

            3. Why Do This?
                
                Share learning: common features (e.g., coins, hand state) don’t need to be relearned in two separate networks.

                Save memory/training time: fewer networks = less duplication.

                Enable joint optimization: the model gets better at choosing actions and buys simultaneously.

            4. Tradeoff

                Slightly more complexity: the training loop must switch heads or encode phases.

                You’ll need a way to organize experience replay to track which phase each sample came from.

        2. separate models - maybe try both and compare?

        3. curriculum learning

            1. train separate models for actions and buying first, then train them together in a shared model after the agent has learned the basics of each part of the game.

            Separate Models - Early curriculum stages - Independent skill learning

            Shared Model - Later curriculum or fine-tuning - Integrated decision-making
        
        1. train buying phase model first
        
            Remove all Action cards.

            Agent plays one buy phase per turn with only basic treasures and VP cards (e.g., Copper, Silver, Gold, Estate, Duchy, Province).

            Goal: learn economy and scoring trade-offs.



