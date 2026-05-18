<div align="justify">

*— Work in progress —*

# Reinforcement Learning: Deep Q-Learning for Dominion


### Contents



## Reinforcement Learning
Reinforcement learning is a subtype of machine learning where an agent learns to optimize its behavior in a given environment through learning by trial and error.
The agent learns from its experience what it should and shouldn't do within the state that it currently occupies.

### Historical Background

blah blah blah

### Sequential Decision Making

Given states $S_t$ at time $t$, future states $S_{t+1}$, the rewards $R_t$ for state transitions, and the actions $A_t$ that map $S_t$ into $S_{t+1}$, the reinforcement agent seeks to maximize its return

$$
\begin{aligned}
G_t &= R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \ldots \\
G_t &= \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
\end{aligned}
$$

where $\gamma \in [0,1]$ is a discount factor for future returns.

===================================

*write something about ...
In machine learning, we look for a function \(f\) in a hypothesis space \(\mathcal{H}\) that maps inputs to outputs.
We can define the subsets by their mapping structures:Supervised Learning (\(SL\)): \(f: \mathcal{X} \to \mathcal{Y}\).
The dataset is a static set of pairs: \(\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N\).
Reinforcement Learning (\(RL\)): \(f: \mathcal{S} \to \mathcal{A}\) (where \(f\) is a policy \(\pi \), mapping states to actions).The fundamental difference lies in the loss function.
In supervised learning, the loss is evaluated per independent sample.
In RL, the objective function is a conditional expectation over time, where actions change the future data distribution:*

===================================

### Markov Decision Processes

A Markov decision process (MDP) is a 4-tuple defined as

$$
(\mathscr{S}, \mathscr{A}, P, R)
$$

where $\mathscr{S}$ is the state space, $\mathscr{A}$ is the action space, $P = P(s' \mid s, a)$ is the probability of transitioning from state $s$ to $s'$ through action $a$, and $R(s,a,s')$ is the reward for the transition from state $s$ to $s'$ through action $a$.
MDPs have the property that the probability of an event happening is independent of the process's history:

$$
P \big(S_{t_1} \mid S_t, A_t \big) = P \big( S_{t+1} \mid S_t, A_t, S_{t-1}, A_{t-1}, \ldots \big) \ .
$$

### Policies, Returns, and Value Functions

Following a deterministic policy, the agent's behavior is defined by

$$
a = \pi(s)
$$

where $\pi(s)$ is a policy mapping each state $s$ to the action $a$.
While in a given state, the agent picks the action that gives it the greatest reward.
A stochastic policy gives the agent 

### Bellman Equations

### Temporal-Difference Learning

### Q-Learning

### Exploration and ϵ-Greedy Action Selection

### Tabular Q-Learning

### Strengths and Limitations of the Tabular Q-learning


Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.

---

### Q-Learning

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.

#### Tabular Q-Learning Example

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.

---

### Deep Q-Learning

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.

#### Neural Networks

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.

#### State Value and Action Value Approximation Using Neural Networks

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.


## Deep Q-Learning for Dominion

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.



</div>
