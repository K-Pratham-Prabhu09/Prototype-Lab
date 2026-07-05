# GridWorld

A custom-built Python simulator designed to bridge the gap between mathematical Reinforcement Learning (RL) theory and practical implementation. This project implements a grid-based environment and an autonomous agent that learns to navigate using fundamental RL algorithms.

## Features
* **Custom Environment (`GridWorld`)**: Configurable states, goals, rewards, and step costs.
* **Autonomous Agent (`Agent`)**: Supports pure exploration (random walk) and exploitation ($\epsilon$-greedy).
* **Multiple Learning Algorithms**:
  * First-Visit Monte Carlo Prediction
  * Temporal Difference, TD(0)
* **Real-time Diagnostics**: Visual mapping of the value function ($V$), Q-table ($Q$), and greedy policies.

---

## Architecture

### 1. `GridWorld` (Environment)
Handles the transition dynamics, state space, and reward system.
* **State Space ($S$)**: Represented as `(row, column)` tuples.
* **Action Space ($A$)**: Discrete movements: `1:Up, 2:Down, 3:Left, 4:Right`.
* **Reward Function ($R$)**: Returns $+2$ for the goal, and a living penalty of $-0.1$ for normal steps to penalize laziness and prevent infinite loops.
* **Transition Dynamics ($P$)**: Deterministic. Moving UP guarantees landing in the cell above, assuming it is not a wall.

### 2. `Agent` (Learner)
Interacts with the `GridWorld` and updates its internal memory mapping based on experience.
* **Policy ($\pi$)**: Uses $\epsilon$-greedy action selection (`choose_action`) to balance exploration and exploitation.
* **Value Table (`self.V`)**: Maps a state to an expected return. Used in Model-Based/TD(0) learning.
* **Q-Table (`self.Q`)**: Maps a (state, action) pair to an expected return. Used in Model-Free Q-learning.

---

## Math to Code

| Theoretical Term | Math Symbol | Python Implementation |
| :--- | :--- | :--- |
| **Model Dynamics** | $P(s' \mid s, a)$ | `GridWorld.move()` logic |
| **Reward Function** | $R(s, a)$ | Values inside `GridWorld.grid` |
| **Policy** | $\pi(a \mid s)$ | `Agent.choose_action()` |
| **Discount Factor** | $\gamma$ | `gamma=0.9` parameter |
| **Return** | $G_t$ | `G` variable (sum of discounted rewards) |
| **State Value** | $V(s)$ | `Agent.V` dictionary |
| **Action Value** | $Q(s, a)$ | `Agent.Q` nested dictionary |

---

## 🧠 Learning Algorithms Implemented

### 1. First-Visit Monte Carlo Prediction
Calculates the expected return by waiting for the episode to finish, then working backward to distribute the rewards.
* **Pros**: Simple, unbiased.
* **Cons**: High variance, cannot handle non-terminating loops without a step cost.

### 2. Temporal Difference TD(0)
Updates the state value immediately after a single step by bootstrapping.
$$V(S) \leftarrow V(S) + \alpha [R + \gamma V(S') - V(S)]$$
* **Pros**: Learns step-by-step, breaks infinite loops in real-time, lower variance.
