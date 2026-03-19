# Deep Reinforcement Learning

**DOTE 6635: Artificial Intelligence for Business Research (Spring 2026)**

**Instructor: Renyu (Philip) Zhang**

## Abstract

This article provides a comprehensive introduction to deep reinforcement learning (Deep RL), the powerful combination of deep neural networks with reinforcement learning algorithms that has enabled agents to tackle problems with enormous state and action spaces. The content is based on the lecture slides from the course "DOTE 6635: Artificial Intelligence for Business Research" and is supplemented with additional explanations and references to foundational literature. We begin with a review of where we stand in the RL landscape, connecting model-based methods, model-free methods, and model-free control. We then introduce value function approximation as the bridge from tabular RL to scalable methods, covering both linear and nonlinear approximators. Next, we explore Deep Q-Networks (DQN), the landmark algorithm that achieved human-level performance on Atari games, examining its key innovations—experience replay and fixed targets—along with extensions and applications to business research. We then turn to policy-based methods, which optimize the policy directly rather than deriving it from a value function, covering policy gradient theory, the REINFORCE algorithm, variance reduction techniques, and actor-critic methods including A3C. Building on this foundation, we examine modern deep RL algorithms—Trust Region Policy Optimization (TRPO) and Proximal Policy Optimization (PPO)—that address the instability of vanilla policy gradients through constrained optimization in policy space. Finally, we discuss the increasingly important application of RL to large language models (LLMs), covering supervised fine-tuning (SFT), reward modeling, reinforcement learning from human feedback (RLHF), reward hacking, Direct Preference Optimization (DPO), chain-of-thought reasoning, reinforcement learning with verifiable rewards (RLVR), Group Relative Policy Optimization (GRPO), and the emergence of reasoning capabilities in models like DeepSeek-R1, which together form the foundation of modern AI alignment and reasoning capabilities.

## 1. Where Are We?

Before proceeding to deep reinforcement learning, it is helpful to situate ourselves within the broader RL curriculum (see Sutton and Barto, 2018 [1] for a comprehensive textbook treatment, and Silver, 2015 [2] for an excellent lecture series):

**Model-Based Methods:**
- We know the action space, the state space, the transition probability matrix, and the reward function.
- The Bellman equation is used to evaluate a policy, and the Bellman optimality operator is used to find the optimal policy.
- Solution algorithms include value iteration, policy iteration, and linear programming.

**Model-Free Methods:**
- Monte Carlo (MC) learns directly from episodes of experience: sample the entire reward process and use empirical mean return to approximate expected return.
- Temporal Difference (TD) combines MC and the Bellman operator: bootstrap to update the value function based on the existing estimate.
- MC has zero bias and high variance; TD has some bias and low variance.

**Model-Free Control:**
- MC policy iteration combines MC policy evaluation and $\varepsilon$-greedy for policy improvement.
- SARSA is an on-policy method to update $Q(S,A)$ using TD and $\varepsilon$-greedy for policy improvement.
- Q-learning is an off-policy method to learn $Q(S,A)$ for the greedy target policy using $\varepsilon$-greedy as the behavior policy through TD update.

All of the above methods, however, rely on **tabular representations**—maintaining a separate value for every state or state-action pair. This works well for small, discrete problems but becomes infeasible when the state or action space is large or continuous. The central question of this lecture is: **how do we scale reinforcement learning to real-world problems?**

## 2. Value Function Approximation

### 2.1. The Limits of Tabular RL

In tabular RL, we represent the value functions by a lookup table:
- Every state $s$ has an entry $V(s)$.
- Every state-action pair $(s, a)$ has an entry $Q(s, a)$.

This is sometimes visualized as a **Q-table**, with states along one axis and actions along the other. Each cell stores the estimated value of taking that action in that state.

**Challenges with large MDPs:**
- **Too many states or actions to store in memory.** For example, the game of Go has approximately $3^{19 \times 19} \approx 10^{137}$ states—far exceeding the estimated number of atoms in the observable universe ($\sim 10^{80}$). Self-driving cars operate in continuous state spaces. Dynamic pricing involves continuous action spaces.
- **Too slow to learn the value of each state individually.** Even if we could store the table, visiting every state enough times to obtain reliable estimates would take prohibitively long.

### 2.2. The Idea of Function Approximation

The solution is to **avoid explicitly learning or storing values for every single state**. Instead, we estimate approximated value and policy functions parameterized by a weight vector $\mathbf{w}$:

$$ \hat{v}(s, \mathbf{w}) \approx v_\pi(s) $$

$$ \hat{q}(s, a, \mathbf{w}) \approx q_\pi(s, a) $$

$$ \hat{\pi}(a, s, \mathbf{w}) \approx \pi(a|s) $$

The key benefits of function approximation are:

1. **Generalization from seen to unseen states.** By learning a parameterized function, we can estimate values for states that have never been visited, as long as they share features with states that have been visited.
2. **Compact representation.** Instead of storing $|\mathcal{S}|$ (or $|\mathcal{S}| \times |\mathcal{A}|$) values, we store only the parameter vector $\mathbf{w}$, whose dimension can be vastly smaller.
3. **Scalability.** The parameter $\mathbf{w}$ is updated using MC or TD learning, making the approach compatible with the model-free methods we have already studied.

There are several function design choices. The approximator can take:
- State $s$ as input and output $\hat{v}(s, \mathbf{w})$.
- State $s$ and action $a$ as inputs and output $\hat{q}(s, a, \mathbf{w})$.
- State $s$ as input and output $\hat{q}(s, a_1, \mathbf{w}), \hat{q}(s, a_2, \mathbf{w}), \ldots, \hat{q}(s, a_m, \mathbf{w})$ for all actions simultaneously.

We consider **differentiable functions** as the approximators, since they can deal with non-stationary and non-i.i.d. data—both hallmarks of RL. Two important families are:

- **Linear feature representations:** $\hat{v}(s, \mathbf{w}) = \mathbf{x}(s)^\top \mathbf{w}$
- **Neural networks:** $\hat{v}(s, \mathbf{w}) = f_{\mathbf{w}}(s)$, where $f$ is a deep neural network.

### 2.3. Value Function Approximation with an Oracle

To build intuition, we first consider the idealized setting where an **oracle** provides the true value $v^\pi(s)$ for any given state $s$. The objective is to find the best approximate representation of $v^\pi(s)$.

**Loss Function:** We use the mean squared error (MSE) between the true value and the approximation:

$$ J(\mathbf{w}) = \mathbb{E}_\pi \left[ \left( v^\pi(s) - \hat{v}(s, \mathbf{w}) \right)^2 \right] $$

**Gradient Descent Update:** We minimize this loss using gradient descent:

$$ \Delta \mathbf{w} = -\frac{1}{2} \alpha \nabla_\mathbf{w} J(\mathbf{w}) $$

$$ \mathbf{w}_{t+1} = \mathbf{w}_t + \Delta \mathbf{w} $$

where $\alpha$ is the learning rate (step size).

### 2.4. Linear Approximation with an Oracle

A natural starting point is to represent states using a finite feature vector with $n$ variables:

$$ \mathbf{x}(s) = \begin{pmatrix} x_1(s) \\ x_2(s) \\ \vdots \\ x_n(s) \end{pmatrix} $$

The value function is then approximated by a linear combination of features:

$$ \hat{V}(s; \mathbf{w}) = \sum_{j=1}^{n} x_j(s) w_j = \mathbf{x}(s)^\top \mathbf{w} $$

The loss function is:

$$ J(\mathbf{w}) = \mathbb{E}_\pi \left[ \left( V^\pi(s) - \hat{V}(s; \mathbf{w}) \right)^2 \right] $$

Using stochastic gradient descent (SGD) to update the weights:

$$ \Delta \mathbf{w} = -\frac{1}{2} \alpha \nabla_\mathbf{w} J(\mathbf{w}) $$

Since the function is linear and the loss is MSE, the gradient simplifies to:

$$ \Delta \mathbf{w} = \alpha \left( v^\pi(s) - \hat{v}(s, \mathbf{w}) \right) \mathbf{x}(s) $$

> **Property:** Because the loss is convex in $\mathbf{w}$, SGD will converge to the **global minimum**.

For a general (nonlinear) approximation function, SGD will only converge to a local minimum. The general update rule is:

$$ \Delta \mathbf{w} = \alpha \left( v^\pi(s) - \hat{v}(s, \mathbf{w}) \right) \nabla_\mathbf{w} \hat{v}(s_t, \mathbf{w}) $$

### 2.5. Model-Free Evaluation with Approximation

In practice, we do not have oracles—we only have **rewards** collected by **interacting with the environment**. But we can obtain the **target** using samples.

**Monte Carlo with Function Approximation:**

For MC, the target is simply the actual return $G_t$:

$$ \Delta \mathbf{w} = \alpha \left( G_t - \hat{v}(s_t, \mathbf{w}) \right) \nabla_\mathbf{w} \hat{v}(s_t, \mathbf{w}) $$

Key properties:
- $G_t$ is an **unbiased estimate** of the oracle $v^\pi(s)$.
- MC prediction converges, in both linear and nonlinear value function approximation.

**TD(0) with Function Approximation:**

For TD(0), the target is the TD target $R_{t+1} + \gamma \hat{v}(s_{t+1}, \mathbf{w})$:

$$ \Delta \mathbf{w} = \alpha \left( R_{t+1} + \gamma \hat{v}(s_{t+1}, \mathbf{w}) - \hat{v}(s_t, \mathbf{w}) \right) \nabla_\mathbf{w} \hat{v}(s_t, \mathbf{w}) $$

Key properties:
- The TD target $R_{t+1} + \gamma \hat{v}(s_{t+1}, \mathbf{w})$ is a **biased estimate** of the oracle $v^\pi(s)$.
- This is called a **semi-gradient** method, because we ignore the effect of changing $\mathbf{w}$ on the target (i.e., we do not differentiate through $\hat{v}(s_{t+1}, \mathbf{w})$ in the target).
- Linear TD(0) converges close to the global minimum.

**Algorithms: MC + SGD vs. TD + Semi-Gradient SGD**

The MC algorithm samples a complete trajectory $\tau$ before computing the gradient:

```
while (not converged):
    s₀ ~ p₀, t = 0
    while (sₜ ≠ terminal):
        aₜ ~ π(· | sₜ)
        (rₜ, sₜ₊₁) ~ p(·, · | sₜ, aₜ)
        t = t + 1
    end
    T = t
    g = (V_φ(s₀) - Σₜ₌₀ᵀ⁻¹ γᵗrₜ) ∇_φ V_φ(s₀)
    update φ using g with an optimizer
end
```

The TD algorithm updates after each single transition:

```
while (not converged):
    s₀ ~ p₀, a₀ ~ π(· | s₀), (r₀, s₁) ~ p(·, · | s₀, a₀)
    y = r₀ + γV_φ(s₁)    if s₁ ≠ terminal
        r₀                if s₁ = terminal
    g = (V_φ(s₀) - y) ∇_φ V_φ(s₀)
    update φ using g with an optimizer
end
```

The fundamental difference is clear: MC requires completing entire episodes before updating, while TD updates after a single step, making it suitable for continuing (non-episodic) tasks and faster learning in practice.

## 3. Generalized Policy Iteration with Value Function Approximation

### 3.1. The Framework

Just as in tabular RL, we can combine approximate policy evaluation with policy improvement to perform **generalized policy iteration (GPI)** with function approximation. The procedure alternates between two steps:

1. **Policy Evaluation:** Approximate policy evaluation, $\hat{q}(\cdot, \cdot, \mathbf{w}) \approx q^\pi$.
2. **Policy Improvement:** $\varepsilon$-greedy policy improvement based on the approximate Q-function.

Starting from an initial weight vector $\mathbf{w}$, the process iteratively refines the Q-function approximation and improves the policy, converging toward the optimal Q-function $q_*$.

### 3.2. Model-Free Control with Linear Approximation

With a similar idea to value function approximation, we represent state-action pairs using a finite feature vector:

$$ \mathbf{x}(s, a) = \begin{pmatrix} x_1(s, a) \\ x_2(s, a) \\ \vdots \\ x_n(s, a) \end{pmatrix} $$

The Q-function is approximated by a linear combination of features:

$$ \hat{Q}(s, a; \mathbf{w}) = \mathbf{x}(s, a)^\top \mathbf{w} = \sum_{j=1}^{n} x_j(s, a) w_j $$

The objective is to minimize the MSE between the approximate Q-function and the oracle Q-function:

$$ J(\mathbf{w}) = \mathbb{E}_\pi \left[ \left( q^\pi(s, a) - \hat{q}(s, a, \mathbf{w}) \right)^2 \right] $$

The stochastic gradient descent update is:

$$ \Delta \mathbf{w} = \alpha \left( q^\pi(s, a) - \hat{q}(s, a, \mathbf{w}) \right) \mathbf{x}(s, a) $$

### 3.3. Incremental Control Algorithms

In practice, we do not have the oracle state-action function $q^\pi(s, a)$—we only have rewards collected by interacting with the environment. Different choices of target lead to different algorithms:

**MC Control with Approximation:**

For MC, the target is the actual return $G_t$:

$$ \Delta \mathbf{w} = \alpha \left( G_t - \hat{q}(s_t, a_t, \mathbf{w}) \right) \nabla_\mathbf{w} \hat{q}(s_t, a_t, \mathbf{w}) $$

**SARSA with Approximation:**

For SARSA, the target is the TD target $R_{t+1} + \gamma \hat{q}(s_{t+1}, a_{t+1}, \mathbf{w})$:

$$ \Delta \mathbf{w} = \alpha \left( R_{t+1} + \gamma \hat{q}(s_{t+1}, a_{t+1}, \mathbf{w}) - \hat{q}(s_t, a_t, \mathbf{w}) \right) \nabla_\mathbf{w} \hat{q}(s_t, a_t, \mathbf{w}) $$

**Q-Learning with Approximation:**

For Q-learning, the target is $R_{t+1} + \gamma \max_a \hat{q}(s_{t+1}, a, \mathbf{w})$:

$$ \Delta \mathbf{w} = \alpha \left( R_{t+1} + \gamma \max_a \hat{q}(s_{t+1}, a, \mathbf{w}) - \hat{q}(s_t, a_t, \mathbf{w}) \right) \nabla_\mathbf{w} \hat{q}(s_t, a_t, \mathbf{w}) $$

### 3.4. Convergence of Control Algorithms

An important practical concern is whether these approximate control algorithms converge. The key insight is that **TD with value function approximation does NOT follow the gradient of any loss function.** The updates essentially involve an *approximate Bellman backup* combined with *fitting the underlying value function*—two operations that do not jointly minimize a single objective.

TD may diverge in **off-policy** (e.g., Q-learning) or **nonlinear approximation** settings (Tsitsiklis and Van Roy, 1997 [15]; Bertsekas and Tsitsiklis, 1996 [16]). This is the challenge for off-policy control: the behavior policy and target policy are not identical, so the value function approximation may diverge.

The following table summarizes convergence guarantees:

| Algorithm | Table Lookup | Linear | Non-Linear |
|-----------|:------------:|:------:|:----------:|
| Monte-Carlo Control | $\checkmark$ | ($\checkmark$) | $\times$ |
| SARSA | $\checkmark$ | ($\checkmark$) | $\times$ |
| Q-Learning | $\checkmark$ | $\times$ | $\times$ |

Here ($\checkmark$) indicates that the algorithm moves around the near-optimal value function (it oscillates near the optimum but does not converge exactly).

> **The Deadly Triad:** The combination of (1) **bootstrapping**, (2) **function approximation**, and (3) **off-policy learning** is known as the "deadly triad" in RL. When all three are present simultaneously, training instability and divergence become significant risks. Q-learning with nonlinear function approximation is a prime example, as it combines all three elements.

## 4. Going Beyond Linear: Deep Reinforcement Learning

### 4.1. Motivation

So far, we have focused on linear approximators. Linear approximations work well given the right set of features, but they require **manual design** of the feature set—a process that demands significant domain expertise and may not capture complex nonlinear relationships in the data.

We know that deep neural networks (DNNs) are much better representation tools when we have a large data set. They can automatically learn hierarchical feature representations from raw data, eliminating the need for hand-crafted features.

The central question is: **How can we leverage deep learning for function approximation and model-free control of MDPs?**

### 4.2. Deep Reinforcement Learning

Deep reinforcement learning leverages DNNs to represent:
- **Value functions** ($Q$ and $V$)
- **Policy functions** ($\pi$)
- **World models** (for model-based approaches)

The loss function is optimized by stochastic gradient descent (SGD), just as in supervised deep learning.

**Core Challenges:**
- **Data inefficiency:** Too many model parameters to optimize, requiring vast amounts of interaction data.
- **The Deadly Triad creates instability and divergence in training**, arising from:
  - Nonlinear function approximation
  - Bootstrapping
  - Off-policy training

Deep Q-learning (DQN) addresses two critical issues—**correlations between samples** and **non-stationary targets**—through two key innovations:

1. **Experience Replay:** Sample experience randomly from stored data to update weights, reducing correlations between data points.
2. **Fixed Q-Targets:** Fix the target network's weights while updating the main network, improving stability.

## 5. Deep Q-Network (DQN) for Atari

### 5.1. The Breakthrough

The landmark paper by Mnih et al. (2015) [3], "Human-level control through deep reinforcement learning," published in *Nature*, demonstrated that a single DQN agent could achieve **professional human-level performance** across many Atari 2600 games using the **same network architecture and hyperparameters** for all games. This was a groundbreaking result: the same algorithm, without any game-specific tuning, could learn to play dozens of different video games from raw pixel inputs.

DQN represents the action-value function with a deep neural network (DNN) approximator. The performance comparison between linear approximators and deep networks across several Atari games illustrates the dramatic advantage of deep representations:

| Game | Linear | Deep Network |
|------|:------:|:------------:|
| Breakout | 3 | 3 |
| Enduro | 62 | 29 |
| River Raid | 2345 | 1453 |
| Seaquest | 656 | 275 |
| Space Invaders | 301 | 302 |

*(Note: Scores represent percentage of human performance; deep networks eventually far surpass these early results with full DQN training.)*

### 5.2. Architecture and Setup

The DQN architecture for Atari is an **end-to-end learning** system for $Q(s, a)$ directly from input pixel frames:

- **Input state** $s$: A stack of raw pixels from the latest 4 frames (to capture motion information). The input is preprocessed to $84 \times 84 \times 4$.
- **Output** of $Q(s, a)$: 18 values, one for each possible joystick action.
- **Reward**: The change in game score for that step.
- **Network architecture**: A convolutional neural network (CNN) with:
  - $8 \times 8 \times 4$ filters with stride 4 → $20 \times 20 \times 16$ feature maps
  - $4 \times 4 \times 16$ filters with stride 2 → $9 \times 9 \times 32$ feature maps
  - Fully connected layer → 256 hidden units
  - Fully connected output layer → $4$–$18$ action values
- **Network architecture and hyperparameters are fixed across all games.**

The crucial design choice is to output Q-values for **all actions simultaneously** given a state, rather than taking both state and action as input. This allows efficient computation of $\max_a Q(s, a)$ in a single forward pass.

### 5.3. Experience Replay

A fundamental problem in training neural networks with RL data is that consecutive samples are **highly correlated**—the agent's trajectory through the environment produces a sequence of states where each state is very similar to the previous one. Standard SGD assumes i.i.d. samples, so training on correlated data leads to poor learning and instability.

**The Solution:** Store the transition $(s_t, a_t, r_t, s_{t+1})$ in a **replay memory** $\mathcal{D}$. Then, instead of training on the most recent transition, sample random mini-batches from $\mathcal{D}$.

To perform experience replay, repeat the following:

1. **Sample** an experience tuple from the dataset: $(s, a, r, s') \sim \mathcal{D}$.
2. **Compute** the target value for the sampled tuple: $r + \gamma \max_{a'} \hat{Q}(s', a', \mathbf{w})$.
3. **Use SGD** to update the network weights $\mathbf{w}$:

$$ \Delta \mathbf{w} = \alpha \left( r + \gamma \max_{a'} \hat{Q}(s', a', \mathbf{w}) - Q(s, a, \mathbf{w}) \right) \nabla_\mathbf{w} \hat{Q}(s, a, \mathbf{w}) $$

**Benefits of Experience Replay:**
- Breaks temporal correlations between consecutive samples.
- Each experience can be used for multiple weight updates, improving data efficiency.
- The replay buffer effectively smooths the training distribution over many past behaviors.

**Limitation:** Using the target as a scalar works for the current update, but network weights will get updated on the next round, changing the target value. This introduces non-stationarity in the targets.

### 5.4. Fixed Q-Targets

A second source of instability is that the target used to compute the TD error depends on the **same weights** $\mathbf{w}$ that we are trying to update. In the original Q-learning with value function approximation, both the Q estimation and the Q target shift at each time step. This is analogous to a cat (Q estimation) chasing a mouse (Q target): if both the cat and the mouse are moving, the cat's trajectory will oscillate wildly, producing an **oscillated training history**.

**The Solution:** Fix the target weights used in the target calculation for multiple updates.

Let $\mathbf{w}^-$ be a separate set of parameters used in the target (the **target network**), and $\mathbf{w}$ be the weights that are being updated (the **online network**). The target network's weights are held fixed and only periodically synchronized with the online network.

To perform experience replay with fixed targets, repeat the following:

1. **Sample** an experience tuple from the dataset: $(s, a, r, s') \sim \mathcal{D}$.
2. **Compute** the target value for the sampled tuple: $r + \gamma \max_{a'} \hat{Q}(s', a', \mathbf{w}^-)$.
3. **Use stochastic gradient descent** to update the network weights:

$$ \Delta \mathbf{w} = \alpha \left( r + \gamma \max_{a'} \hat{Q}(s', a', \mathbf{w}^-) - Q(s, a, \mathbf{w}) \right) \nabla_\mathbf{w} \hat{Q}(s, a, \mathbf{w}) $$

The key difference is that $\mathbf{w}^-$ in the target is **not** updated at every step. Instead, $\mathbf{w}^-$ is periodically copied from $\mathbf{w}$ every $C$ steps.

### 5.5. The Complete DQN Algorithm

Putting it all together, the full DQN algorithm is as follows:

```
Input: C (target network update frequency), α (learning rate), D = {} (replay buffer)
Initialize: w, w⁻ = w, t = 0

1.  Get initial state s₀
2.  loop:
3.      Sample action aₜ given ε-greedy policy for current Q̂(sₜ, a; w)
4.      Observe reward rₜ and next state sₜ₊₁
5.      Store transition (sₜ, aₜ, rₜ, sₜ₊₁) in replay buffer D
6.      Sample random minibatch of tuples (sⱼ, aⱼ, rⱼ, sⱼ₊₁) from D
7.      For j in minibatch do:
8.          if episode terminated at step j + 1 then
9.              yⱼ = rⱼ
10.         else
11.             yⱼ = rⱼ + γ maxₐ' Q̂(sⱼ₊₁, a', w⁻)
12.         end if
13.         Do gradient descent step on (yⱼ - Q̂(sⱼ, aⱼ; w))²:
                Δw = α(yⱼ - Q̂(sⱼ, aⱼ; w)) ∇_w Q̂(sⱼ, aⱼ; w)
14.     end for
15.     t = t + 1
16.     if mod(t, C) == 0 then
17.         w⁻ ← w
18.     end if
19. end loop
```

The algorithm elegantly combines three ideas: (1) Q-learning for off-policy control, (2) experience replay for breaking correlations, and (3) fixed targets for training stability.

### 5.6. Performance of DQN on Atari

The empirical results of DQN on Atari 2600 games were striking. DQN achieved scores comparable to or exceeding professional human testers across a diverse set of 49 games, all using the same architecture and hyperparameters.

An ablation study reveals the importance of each component:

| Game | Replay + Fixed-Q | Replay + Q-learning | No Replay + Fixed-Q | No Replay + Q-learning |
|------|:----------------:|:-------------------:|:-------------------:|:----------------------:|
| Breakout | 316.81 | 240.73 | 10.16 | 3.17 |
| Enduro | 1006.3 | 831.25 | 141.89 | 29.1 |
| River Raid | 7446.62 | 4102.81 | 2867.66 | 1453.02 |
| Seaquest | 2894.4 | 822.55 | 1003 | 275.81 |
| Space Invaders | 1088.94 | 826.33 | 373.22 | 301.99 |

The results demonstrate that:
- **Experience replay** provides a massive performance boost across all games. Without it, performance drops dramatically.
- **Fixed Q-targets** further improve performance on top of experience replay.
- The combination of both innovations (**Replay + Fixed-Q**) consistently achieves the best performance.

## 6. Extensions to DQN

The success of DQN on Atari inspired a rich line of follow-up work improving deep reinforcement learning:

**Double DQN** (Van Hasselt, Guez, and Silver, AAAI 2016) [4]: Standard Q-learning is known to overestimate action values because it uses the same network to both select and evaluate actions ($\max_a Q(s, a)$ selects the best action and simultaneously uses that Q-value as the estimate). Double DQN decouples these two operations: the online network selects the best action, but the target network evaluates its value. This significantly reduces overestimation bias and improves performance.

**Prioritized Replay** (Schaul, Quan, Antonoglou, and Silver, ICLR 2016) [5]: Instead of sampling uniformly from the replay buffer, prioritized experience replay samples transitions with larger TD errors more frequently. The intuition is that transitions where the agent's prediction is most wrong are the most informative for learning. This approach stores the last encountered TD error along with each transition in the replay buffer and uses it to define sampling probabilities.

**Dueling DQN** (Wang, Schaul, Hessel et al., ICML 2016, Best Paper) [6]: The dueling architecture decomposes the Q-function into two streams: a **value stream** $V(s)$ that estimates how good it is to be in state $s$, and an **advantage stream** $A(s, a)$ that estimates the relative advantage of each action. The Q-value is then reconstructed as $Q(s, a) = V(s) + A(s, a)$. This decomposition allows the network to learn which states are valuable (or not) without having to learn the effect of each action in each state separately, leading to more efficient learning.

## 7. Applications: Deep RL for Business Decision Making

Deep reinforcement learning has found compelling applications across a range of business domains. We highlight several recent papers published in top journals that illustrate the breadth and depth of these applications.

### 7.1. Dynamic Coupon Targeting

Liu (2023) [8] in *Marketing Science* applies batch deep reinforcement learning (BDRL) to dynamic coupon targeting in a livestream shopping context.

**Problem Setting:**
- The firm must decide how to dynamically target coupons to consumers over time.
- The state space is high-dimensional (capturing consumer characteristics, browsing history, and purchase behavior).
- The action space involves selecting coupon types and discount levels.
- Transition dynamics and rewards are unknown and must be learned from observational data.

**Methodology:**
- The paper uses a batch (offline) variant of deep Q-learning, where the agent learns from a fixed dataset of historical interactions rather than through online exploration.
- Deep neural networks are used to represent the high-dimensional state space and alleviate the curse of dimensionality.
- The approach incorporates a doubly robust estimator for policy evaluation.

**Key Findings:**
- The BDRL solution increases the platform's revenue significantly compared to static targeting policies.
- Gains come from more effective and automatic targeting of consumers based on heterogeneity and dynamics, using counterfactually rich temporal differences among consumers over time.
- The approach demonstrates the practical value of deep RL for high-frequency, high-dimensional business decision-making.

### 7.2. Ensembling Experimentation Along the Customer Journey

Song and Sun (2024) [9] in *Management Science* propose using deep reinforcement learning to optimize interventions along the customer journey based on an ensemble of historical experiments.

**Problem Setting:**
- Firms run multiple randomized experiments at different stages of the customer journey (e.g., ad exposure, landing page design, promotional offers), but these experiments are typically analyzed in isolation.
- The challenge is to learn an optimal *sequence* of interventions across stages, accounting for intertemporal effects and heterogeneity.

**Methodology:**
- The paper proposes a Bayesian Recurrent Q-Network Estimation (BRQN) framework that combines Bayesian deep learning with recurrent neural networks to handle partial observability and sequential decision-making.
- The framework learns from an ensemble of historical experiments to guide future intervention trials, bridging the gap between RL and classical experimentation.

**Key Findings:**
- Learning from multiple historical experiments jointly yields significantly higher average rewards than learning from any single experiment in isolation.
- The approach shows the promise of fusing RL with multiple experiments for optimizing multi-stage customer journeys.

### 7.3. Sequential Targeting

Wang, Li, Luo, and Wang (2023) [10] in *Management Science* design a DRL-based personalized targeting strategy in a sequential setting, addressing three important challenges: (1) forward looking—balancing exploration and exploitation, (2) scalability—coping with a high-dimensional state and policy space, and (3) adaptivity—learning while continuously interacting with consumers. The proposed DRL agent generates substantially more long-term revenue than can the conventional bandit-based and greedy approaches.

### 7.4. Career Path Recommendations

Kekouos and Ipeirotis (2021) [11] in *Management Science* apply reinforcement learning to provide demand-aware career path recommendations for contractors on an online labor market. The framework combines reinforcement learning, Bayesian inference, and guided learning to promote future career path recommendations while optimizing current market trends. The framework uses market information to identify current trends and project future career paths, and recommends skills that contractors should learn to maximize long-term earnings.

## 8. Policy-Based Methods

> For a thorough and accessible exposition of the material in this section, we recommend Weng (2018) [17], which provides an excellent tutorial on policy gradient algorithms with detailed derivations and intuitive explanations.

### 8.1. From Value to Policy Approximations

So far, we have focused on approximating the value functions $v^\pi(s)$ and $q^\pi(s, a)$ given the policy $\pi$. An entirely different perspective is to **parameterize the policy function directly**:

$$ \pi_\theta(a|s) $$

where $\theta$ is the parameter vector (e.g., the weights of a deep neural network) that determines the policy.

The central optimization problem becomes:

$$ \max_\theta \quad \mathcal{J}(\theta) = \mathbb{E}_{s_0 \sim p_0} \left[ V^{\pi_\theta}(s_0) \right] = \mathbb{E}_{s_0 \sim p_0}^{\pi} \left[ \sum_{t=0}^{T-1} \gamma^t r_t \right] $$

where $\pi_\theta$ is represented by a DNN with parameter $\theta$. The action and reward from the policy are all we need to optimize the policy directly—no value function estimation is required.

### 8.2. Value-Based RL vs. Policy-Based RL

There are three major paradigms for deep reinforcement learning:

- **Value-based RL** learns the value function and derives the policy from it (e.g., by acting greedily with respect to the Q-function). DQN is a prime example.
- **Policy-based RL** learns the optimal policy directly without explicitly computing a value function. The policy gradient methods we discuss below fall into this category.
- **Actor-critic** methods learn *both* a policy and a value function simultaneously. The policy (the "actor") decides which actions to take, while the value function (the "critic") evaluates how good those actions are.

These three approaches can be visualized as a Venn diagram: value-based and policy-based methods occupy distinct regions, while actor-critic methods sit at their intersection, combining elements of both.

### 8.3. Pros and Cons of Policy-Based RL

**Advantages:**
- **Stochastic policies.** Policy-based methods can directly learn stochastic policies, which is important for games (e.g., rock-paper-scissors, where a deterministic policy is trivially exploitable) and for partially observable environments.
- **High-dimensional or continuous action spaces.** Policy-based RL is effective when the action space is high-dimensional or continuous (e.g., robotic control, LLM token generation), where computing $\max_a Q(s, a)$ is intractable.
- **Better convergence properties.** Policy gradient methods enjoy convergence guarantees similar to policy iteration, since the objective function is smooth in the policy parameters.

**Disadvantages:**
- **High variance.** Policy gradient estimators tend to have high variance, which can slow learning and make training unstable.
- **Local optima.** Policy gradient methods typically converge to a local optimum rather than the global optimum. The entire policy space is harder to explore and optimize than the value function space.

### 8.4. Policy Gradient for One-Step MDPs

To build intuition, consider a simple MDP with **one step**: start with $s \sim d(s)$, take one action, and terminate with reward $r = R(s, a)$.

The expected reward of policy $\pi_\theta(s, a)$ is:

$$ J(\theta) = \mathbb{E}_{\pi_\theta}[r] = \sum_{s \in \mathcal{S}} d(s) \sum_{a \in \mathcal{A}} \pi_\theta(s, a) \, r $$

The gradient is:

$$ \nabla_\theta J(\theta) = \sum_{s \in \mathcal{S}} d(s) \sum_{a \in \mathcal{A}} \pi_\theta(s, a) \, \nabla_\theta \log \pi_\theta(s, a) \, r $$

This uses the **log-likelihood trick** (also called the **score function** trick or the **REINFORCE** trick):

$$ \nabla_\theta \pi_\theta(s, a) = \pi_\theta(s, a) \frac{\nabla_\theta \pi_\theta(s, a)}{\pi_\theta(s, a)} = \pi_\theta(s, a) \nabla_\theta \log \pi_\theta(s, a) $$

The term $\nabla_\theta \log \pi_\theta(s, a)$ is called the **score function** or the **gradient of the log-likelihood ratio**. The gradient can thus be written as an expectation:

$$ \nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ r \, \nabla_\theta \log \pi_\theta(s, a) \right] $$

This is remarkable: we can estimate the gradient by sampling trajectories from the policy, without needing to know the environment dynamics.

### 8.5. Policy Gradient for Multi-Step MDPs

Now consider a full multi-step MDP. A **state-action trajectory** from one episode is:

$$ \tau = (s_0, a_0, r_1, \ldots, s_{T-1}, a_{T-1}, r_T, s_T) \sim (\pi_\theta, P(s_{t+1}|s_t, a_t)) $$

Define $R(\tau) = \sum_{t=0}^{T} R(s_t, a_t)$ as the total reward for trajectory $\tau$ (assuming $\gamma = 1$ for simplicity). The policy value is:

$$ J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \sum_{t=0}^{T-1} R(s_t, a_t) \right] = \sum_\tau P(\tau; \theta) R(\tau) $$

where $P(\tau; \theta) = \mu(s_0) \prod_{t=0}^{T-1} \pi_\theta(a_t | s_t) p(s_{t+1} | s_t, a_t)$ is the probability of trajectory $\tau$ under policy $\pi_\theta$.

Our goal is to find the optimal policy parameter:

$$ \theta^* = \arg\max_\theta J(\theta) = \arg\max_\theta \sum_\tau P(\tau; \theta) R(\tau) $$

### 8.6. Taking the Gradient

To compute the gradient $\nabla_\theta J(\theta)$, we apply the log-likelihood trick at the trajectory level:

$$ \nabla_\theta J(\theta) = \nabla_\theta \sum_\tau P(\tau; \theta) R(\tau) = \sum_\tau \nabla_\theta P(\tau; \theta) R(\tau) = \sum_\tau P(\tau; \theta) \nabla_\theta \log P(\tau; \theta) \, R(\tau) $$

Now we decompose $\nabla_\theta \log P(\tau; \theta)$. Since:

$$ P(\tau; \theta) = \mu(s_0) \prod_{t=0}^{T-1} \pi_\theta(a_t | s_t) \, p(s_{t+1} | s_t, a_t) $$

Taking the log:

$$ \log P(\tau; \theta) = \log \mu(s_0) + \sum_{t=0}^{T-1} \log \pi_\theta(a_t | s_t) + \log p(s_{t+1} | s_t, a_t) $$

The gradient with respect to $\theta$ eliminates terms that do not depend on $\theta$:

$$ \nabla_\theta \log P(\tau; \theta) = \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t | s_t) $$

This is a crucial result: **the gradient does not require knowledge of the transition dynamics** $p(s_{t+1}|s_t, a_t)$. All we need is the ability to compute $\nabla_\theta \log \pi_\theta(a_t | s_t)$, which depends only on the parameterized policy.

### 8.7. The REINFORCE Algorithm

Combining the results above, the policy gradient can be estimated as:

$$ \nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p_\theta(\tau)} \left[ \left( \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) \right) \left( \sum_{t=1}^{T} r(s_t, a_t) \right) \right] $$

This expectation can be approximated by sampling $N$ trajectories:

$$ \nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \left( \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(a_{i,t} | s_{i,t}) \right) \left( \sum_{t=1}^{T} r(s_{i,t}, a_{i,t}) \right) $$

The full **REINFORCE** algorithm (Williams, 1992 [14]; also called vanilla policy gradient + Monte Carlo) is:

```
Repeat:
    1. Sample trajectories {τⁱ} from π_θ(aₜ | sₜ)
    2. Estimate gradient: ∇_θ J(θ) ≈ Σᵢ (Σₜ ∇_θ log π_θ(aⁱₜ | sⁱₜ)) (Σₜ r(sⁱₜ, aⁱₜ))
    3. Update: θ ← θ + α ∇_θ J(θ)
```

The algorithm is conceptually simple: run the current policy to collect data, compute the gradient estimate, and take a gradient ascent step.

### 8.8. Policy Gradient vs. Maximum Likelihood

It is instructive to compare the policy gradient estimator with the maximum likelihood estimator used in supervised learning (also called **imitation learning** or **behavioral cloning**):

**Policy gradient estimator:**

$$ \nabla_\theta J(\theta) \approx \frac{1}{M} \sum_{m=1}^{M} \left( \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(a_{t,m} | s_{t,m}) \right) \left( \sum_{t=1}^{T} r(s_{t,m}, a_{t,m}) \right) $$

**Maximum likelihood estimator:**

$$ \nabla_\theta J_{ML}(\theta) \approx \frac{1}{M} \sum_{m=1}^{M} \left( \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(a_{t,m} | s_{t,m}) \right) $$

The only difference is that the policy gradient **weights each trajectory's log-likelihood by its total reward**. The interpretation is intuitive: **good actions (those that lead to high rewards) are made more likely, and bad actions are made less likely.** In maximum likelihood (imitation learning), all demonstrated actions are treated equally—the agent simply mimics the observed behavior regardless of quality.

### 8.9. The Policy Gradient Theorem

The policy gradient theorem (Sutton et al., 1999 [13]) generalizes the likelihood ratio approach to a broader set of objective functions:

> **Theorem (Policy Gradient Theorem).** For any differentiable policy $\pi_\theta(s, a)$, for any of the policy objective functions $J = J_1$ (episodic reward), $J_{avR}$ (average reward per time step), or $\frac{1}{1-\gamma} J_{avV}$ (average value), the policy gradient is:
>
> $$ \nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(s, a) \, Q^{\pi_\theta}(s, a) \right] $$

The policy gradient theorem says that the **policy gradient equals the Q-function following the policy, weighted by the score function of the policy**. This result is fundamental because it provides a unified expression for the gradient regardless of which objective function we choose.

### 8.10. Score Function

The score function $\nabla_\theta \log \pi_\theta(s, a)$ takes different forms depending on the policy parameterization:

**Discrete action policy (Softmax):**

$$ \pi_\theta(s, a) = \frac{\exp(\phi(s, a)^\top \theta)}{\sum_{a'} \exp(\phi(s, a')^\top \theta)} $$

The score function is:

$$ \nabla_\theta \log \pi_\theta(s, a) = \phi(s, a) - \mathbb{E}_{\pi_\theta}[\phi(s, \cdot)] $$

This is the feature vector for the chosen action minus the expected feature vector under the policy—it points in the direction that makes the chosen action more likely relative to the average.

**Continuous action policy (Gaussian):**

The policy is Gaussian with mean linear in state features:

$$ \mu(s) = \phi(s)^\top \theta, \quad a \sim \mathcal{N}(\mu(s), \sigma^2) $$

The score function is:

$$ \nabla_\theta \log \pi_\theta(s, a) = \frac{(a - \mu(s)) \, \phi(s)}{\sigma^2} $$

This pushes the mean toward the selected action, proportional to how far the action is from the current mean.

### 8.11. Reducing Variance of Policy Gradient

The REINFORCE gradient estimator is **unbiased but very noisy** (high variance). The variance arises because the total reward $R(\tau)$ for an entire trajectory is used to weight every action's log-probability in that trajectory, even though some of that reward may have nothing to do with a particular action. Two key techniques reduce this variance:

**Temporal Causality:**

The principle of temporal causality states that a policy at time $t'$ cannot affect a reward at time $t$ for $t < t'$. Exploiting this:

$$ \nabla_\theta J(\theta) = \mathbb{E}_\tau \left[ \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t | s_t) \sum_{t'=t}^{T-1} r_{t'} \right] = \mathbb{E}_\tau \left[ \sum_{t=0}^{T-1} G_t \cdot \nabla_\theta \log \pi_\theta(a_t | s_t) \right] $$

where $G_t = \sum_{t'=t}^{T-1} r_{t'}$ is the **reward-to-go** from time step $t$ onward. This replaces the total trajectory reward with only the future rewards from each time step, significantly reducing variance.

The estimated gradient becomes:

$$ \nabla_\theta \mathbb{E}[R] \approx \frac{1}{m} \sum_{i=1}^{m} \sum_{t=0}^{T-1} G_t^{(i)} \cdot \nabla_\theta \log \pi_\theta(a_t^i | s_t^i) $$

**Subtracting a Baseline:**

We can subtract a **baseline** $b(s_t) = \mathbb{E}[r_t + r_{t+1} + \cdots + r_{T-1}]$ from the return without introducing bias:

$$ \nabla_\theta \mathbb{E}_{\tau \sim \pi_\theta}[R] = \mathbb{E}_\tau \left[ \sum_{t=0}^{T-1} (G_t - b(s_t)) \cdot \nabla_\theta \log \pi_\theta(a_t | s_t) \right] $$

Why is this unbiased? Because $\mathbb{E}_\tau \left[ \nabla_\theta \log \pi_\theta(a_t | s_t) \, b(s_t) \right] = 0$ for any baseline that depends only on the state.

The key insight is that subtracting the baseline changes the gradient from asking "was this action good?" to asking "**was this action better than expected?**" This increases the log-probability of actions proportional to how much their returns exceed the expected return, and is both **unbiased** and **reduces variance**.

### 8.12. Vanilla Policy Gradient with Baseline

The complete vanilla policy gradient algorithm with a learned baseline is:

```
procedure POLICY GRADIENT(α)
    Initialize policy parameters θ and baseline values b(s) for all s, e.g. to 0
    for iteration = 1, 2, ... do
        Collect a set of m trajectories by executing the current policy π_θ
        for each time step t of each trajectory τ⁽ⁱ⁾ do
            Compute the return G_t⁽ⁱ⁾ = Σₜ'₌ₜᵀ⁻¹ rₜ'
            Compute the advantage estimate Â_t⁽ⁱ⁾ = G_t⁽ⁱ⁾ - b(sₜ)
        Re-fit the baseline to the empirical returns by updating w to minimize
            Σᵢ₌₁ᵐ Σₜ₌₀ᵀ⁻¹ ‖b(sₜ) - G_t⁽ⁱ⁾‖²
        Update policy parameters θ using the policy gradient estimate ĝ:
            ĝ = Σᵢ₌₁ᵐ Σₜ₌₀ᵀ⁻¹ Â_t⁽ⁱ⁾ ∇_θ log π_θ(aₜ⁽ⁱ⁾ | sₜ⁽ⁱ⁾)
        with an optimizer like SGD (θ ← θ + α · ĝ) or Adam
    return θ and baseline values b(s)
```

Note that vanilla policy gradient is **on-policy**: it uses only data collected from the current policy $\pi_\theta$ to update the parameters. This raises the question: how can we generalize to **off-policy** settings, where we reuse data from older policies?

### 8.13. Off-Policy Policy Gradient

What if we want to use samples collected from a different policy $\bar{\pi}$ to update our current policy $\pi_{\theta'}$? This is the off-policy setting, which is important for data efficiency since we can reuse past experience.

The idea is to use **importance sampling**. Given a proposal distribution $q(x)$ and a target distribution $p(x)$:

$$ \mathbb{E}_{x \sim p(x)}[f(x)] = \int p(x) f(x) \, dx = \int q(x) \frac{p(x)}{q(x)} f(x) \, dx = \mathbb{E}_{x \sim q(x)} \left[ \frac{p(x)}{q(x)} f(x) \right] $$

Applying this at the trajectory level, the importance weight is $\frac{p_\theta(\tau)}{p_{\bar{\theta}}(\tau)} = \prod_{t=1}^{T} \frac{\pi_\theta(a_t | s_t)}{\bar{\pi}(a_t | s_t)}$, which can become very small or very large for longer trajectories.

To mitigate this, we can instead apply importance sampling at the **per-timestep** level rather than the trajectory level, which is much less likely to explode or vanish:

$$ \nabla_{\theta'} J(\theta') \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \frac{\pi_{\theta'}(s_{i,t}, a_{i,t})}{\pi_\theta(s_{i,t}, a_{i,t})} \nabla_{\theta'} \log \pi_{\theta'}(a_{i,t} | s_{i,t}) \left( \left( \sum_{t'=t}^{T} r(s_{i,t'}, a_{i,t'}) \right) - b \right) $$

In practice, the per-timestep importance ratio $\frac{\pi_{\theta'}(s_t, a_t)}{\pi_\theta(s_t, a_t)}$ is often approximated as 1, simplifying the computation.

**Key challenge:** If the policy changes too much before sampling new data, the data no longer reflects the states the updated policy would visit. To address this, we can **constrain the policy not to change too much** between updates:

$$ \mathbb{E}_{s \sim \pi_\theta} \left[ D_{KL}(\pi_{\theta'}(\cdot | s) \| \pi_\theta(\cdot | s)) \right] \leq \delta $$

This constraint is the foundation of trust region methods like TRPO (Trust Region Policy Optimization) and PPO (Proximal Policy Optimization), which have become widely used in practice, including in the training of large language models.

### 8.14. Reducing Variance with a Critic

Recall the policy gradient update with baseline:

$$ \nabla_\theta \mathbb{E}[R] \approx \frac{1}{m} \sum_{i=1}^{m} \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t, s_t) (G_t^{(i)} - b(s_t)) $$

The MC rollout return $G_t^{(i)}$ is an unbiased estimate of the Q-function, but it comes with high variance. We can instead use a **critic**—a learned approximation of the Q-function—to reduce this variance.

**The Actor-Critic Architecture:**
- **Critic:** Updates the Q-function approximation $Q_\mathbf{w}(s, a)$ with parameter $\mathbf{w}$, which is policy evaluation through function approximation.
- **Actor:** Updates the policy function $\pi_\theta$ using the Q-function produced by the critic.
- **Baseline:** The average reward $V(s_t) = \mathbb{E}_{a_t \sim \pi_\theta(\cdot | s_t)}[Q(s_t, a_t)]$ serves as a natural baseline.

The key quantity is the **advantage function**:

$$ A^{\pi_\theta}(s, a) = Q^{\pi_\theta}(s, a) - V^{\pi_\theta}(s) $$

The advantage function measures how much better action $a$ is compared to the average action under the current policy. It can be approximated by the **TD error**:

$$ \delta^{\pi_\theta} = r + \gamma V^{\pi_\theta}(s') - V^{\pi_\theta}(s) $$

In expectation, the TD error equals the advantage: $\mathbb{E}_{\pi_\theta}[\delta^{\pi_\theta} | s, a] = A^{\pi_\theta}(s, a)$.

The policy gradient then becomes:

$$ \nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(s, a) \, A^{\pi_\theta}(s, a) \right] $$

### 8.15. Compatible Function Approximation Theorem

A natural concern arises: since the critic's Q-function approximation has errors, does this bias the policy gradient updates?

> **Theorem (Compatible Function Approximation).** If the following two conditions are satisfied:
>
> 1. The value function approximator is **compatible** with the policy: $\nabla_\mathbf{w} Q_\mathbf{w}(s, a) = \nabla_\theta \log \pi_\theta(s, a)$
> 2. The value function parameters $\mathbf{w}$ minimize the mean-squared error: $\varepsilon = \mathbb{E}_{\pi_\theta} \left[ (Q^{\pi_\theta}(s, a) - Q_\mathbf{w}(s, a))^2 \right]$
>
> Then the policy gradient is **exact**: $\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} [\nabla_\theta \log \pi_\theta(s, a) \, Q_\mathbf{w}(s, a)]$

This theorem provides reassurance that if the critic satisfies certain compatibility conditions, the actor's gradient updates are not biased by the approximation error in the critic. In practice, the conditions need not hold exactly for the approach to work well.

### 8.16. Simple Action-Value Actor-Critic (QAC) Algorithm

A concrete actor-critic algorithm using linear value function approximation is the **Simple QAC** (Q Actor-Critic):

- **Critic:** Use a linear value function approximation: $Q_\mathbf{w}(s, a) = \psi(s, a)^\top \mathbf{w}$. Update $\mathbf{w}$ by linear TD(0).
- **Actor:** Update $\theta$ by policy gradient.

```
Algorithm: Simple QAC
1: for each step do
2:     Generate sample s, a, r, s', a' following π_θ
3:     δ = r + γ Q_w(s', a') - Q_w(s, a)          # TD error
4:     w ← w + β δ ψ(s, a)                         # Critic update
5:     θ ← θ + α ∇_θ log π_θ(s, a) Q_w(s, a)      # Actor update
6: end for
```

### 8.17. Asynchronous Advantage Actor-Critic (A3C)

The **A3C** algorithm (Mnih et al., ICML 2016 [7]; see also Weng, 2018 [17] for a tutorial exposition) scales actor-critic methods through **parallel training of multiple actors**. It is a policy gradient method where critics learn the value function while multiple actors are trained in parallel, each interacting with its own copy of the environment.

**Key Design Principles:**
1. **Global parameters** $\theta$ and $\mathbf{w}$ are shared, with thread-specific copies $\theta'$ and $\mathbf{w}'$.
2. Each thread runs independently: reset gradients, synchronize with global parameters, sample a trajectory, and compute local gradients.
3. Gradients are accumulated locally over a short trajectory segment and then used to **asynchronously update** the global parameters.

The algorithm outline is:

```
While T ≤ T_MAX:
    1. Reset gradient: dθ = 0 and dw = 0
    2. Synchronize thread-specific parameters: θ' = θ and w' = w
    3. Sample a starting state sₜ
    4. While (sₜ ≠ TERMINAL) and (t - t_start ≤ t_max):
        - Pick action Aₜ ~ π_θ'(Aₜ | Sₜ) and observe Rₜ, sₜ₊₁
        - t = t + 1
    5. Initialize return estimate:
        R = 0 if sₜ is TERMINAL, else V_w'(sₜ)
    6. For i = t-1, ..., t_start:
        R ← γR + Rᵢ   (MC estimate of Gᵢ)
        Accumulate gradients w.r.t. θ': dθ ← dθ + ∇_θ' log π_θ'(aᵢ|sᵢ)(R - V_w'(sᵢ))
        Accumulate gradients w.r.t. w': dw ← dw + 2(R - V_w'(sᵢ))∇_w'(R - V_w'(sᵢ))
    7. Update asynchronously: θ using dθ, and w using dw
```

**Advantages of A3C:**
- **Parallelism for stability:** Running multiple actors in parallel with different exploration patterns provides diverse experience, reducing the correlation between updates. This serves a similar purpose to experience replay in DQN, but without the memory overhead.
- **On-policy learning at scale:** Unlike DQN which is off-policy, A3C remains on-policy while achieving data efficiency through parallelism.
- **Lower memory requirements:** No need for a large replay buffer; each thread only needs to store a short trajectory segment.

## 9. Application: Deep RL for Inventory Control

Gijsbrechts, Boute, Van Mieghem, and Zhang (2022) [12] in *Manufacturing & Service Operations Management* rigorously evaluate deep reinforcement learning for inventory management across three classic and intractable problem settings: lost sales, dual-sourcing, and multi-echelon inventory systems.

**Problem Setting:**
- Some inventory control problems are **notoriously challenging**: lost-sales systems (where unmet demand is lost rather than backordered), dual-sourcing (choosing between fast and slow suppliers), and multi-echelon networks (coordinating inventory across supply chain tiers).
- These problems lack tractable closed-form solutions and have historically relied on hand-crafted heuristics.

**Methodology:**
- The authors formulate each inventory problem as a Markov decision process and apply the Asynchronous Advantage Actor-Critic (A3C) algorithm.
- The DRL agent learns inventory replenishment policies directly from simulated demand and cost data.

**Key Findings:**
- The A3C algorithm **matches the performance of well-designed heuristics**, with limited changes to the tuning parameters across different problem structures.
- Although initial tuning was computationally demanding and time-consuming, only small adjustments to the tuning parameters were needed for the other studied problems.
- However, **generating structural policy insight and specialized near-optimal policies remains desirable**—DRL provides a powerful numerical tool, but the learned policies can be difficult to interpret and do not replace the conceptual understanding offered by structural results.

## 10. Modern Deep RL: From TRPO to PPO

The policy gradient methods discussed in the previous sections—REINFORCE, vanilla policy gradient with baselines, and actor-critic algorithms—suffer from a critical practical limitation: **they are highly sensitive to the step size**. In supervised learning, an overly large learning rate leads to poor convergence but the model can typically recover. In RL, the consequences are far more severe: a bad policy update leads to bad data collection, which leads to an even worse policy, creating a vicious cycle that can **collapse overall performance irreversibly** [17, 18].

This section introduces two landmark algorithms—**Trust Region Policy Optimization (TRPO)** and **Proximal Policy Optimization (PPO)**—that address this instability by constraining how much the policy can change in each update step.

### 10.1. Recap: The Policy Gradient Landscape

Before diving into modern methods, it is useful to recall the different forms the policy gradient can take:

$$ \nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}[\nabla_\theta \log \pi_\theta(s, a) \, G_t] \quad \text{— REINFORCE} $$

$$ = \mathbb{E}_{\pi_\theta}[\nabla_\theta \log \pi_\theta(s, a) \, Q^{\mathbf{w}}(s, a)] \quad \text{— Q Actor-Critic} $$

$$ = \mathbb{E}_{\pi_\theta}[\nabla_\theta \log \pi_\theta(s, a) \, A^{\mathbf{w}}(s, a)] \quad \text{— Advantage Actor-Critic} $$

$$ = \mathbb{E}_{\pi_\theta}[\nabla_\theta \log \pi_\theta(s, a) \, \delta] \quad \text{— TD Actor-Critic} $$

The critic can use MC or TD learning to estimate $Q^\pi(s,a)$, $A^\pi(s,a)$, or $V^\pi(s)$.

### 10.2. Issues with Vanilla Policy Gradient

Vanilla policy gradient is **on-policy**, meaning it uses only data collected from the current policy $\pi_\theta$ to compute gradient updates. This has two major consequences:

1. **Poor sample efficiency:** Each batch of data is used for a single gradient update and then discarded. Collecting new data after every update is expensive.
2. **Catastrophic sensitivity to step size:** Unlike supervised learning where data and model are independent, in RL the data distribution depends on the policy. If the step is too large:
   - The policy becomes bad → it collects bad data → the next update makes the policy even worse.
   - The training may **never recover** from such a collapse.

The solution is to combine **off-policy learning** (via importance sampling) with a **trust region** constraint that limits how much the policy can change. This leads to Trust Region Policy Optimization (TRPO) and its simpler successor, Proximal Policy Optimization (PPO).

### 10.3. Relative Policy Performance

The key theoretical insight underlying TRPO begins with analyzing how policy performance changes when we move from one policy to another.

**Steepest ascent in parameter space (standard gradient):**

$$ d^* = \nabla_\theta J(\theta) = \lim_{\epsilon \to 0} \frac{1}{\epsilon} \arg\max_d J(\theta + d), \quad \text{s.t. } \|d\| \leq \epsilon $$

This characterizes the direction of steepest improvement using the **Euclidean metric** on parameter space.

**Steepest ascent in distribution space (natural gradient):**

$$ d^* = \arg\max_d J(\theta + d), \quad \text{s.t. } KL(\pi_\theta \| \pi_{\theta+d}) = c $$

This characterizes the direction of steepest improvement using the **KL divergence** to measure distances between policy distributions. The KL divergence is defined as:

$$ KL(\pi_\theta \| \pi_{\theta'}) = E_{\pi_\theta}[\log \pi_\theta] - E_{\pi_\theta}[\log \pi_{\theta'}] $$

The second-order Taylor expansion of the KL divergence gives:

$$ KL(\pi_\theta \| \pi_{\theta+d}) \approx \frac{1}{2} d^T F d $$

where $F$ is the **Fisher Information Matrix**:

$$ F = E_{\pi_\theta}[\nabla \log \pi_\theta \nabla \log \pi_\theta^T] $$

**The performance difference between two policies** can be expressed exactly as:

$$ J(\pi') - J(\pi) = \mathbb{E}_{\tau \sim \pi'}\left[\sum_{t=0}^{\infty} \gamma^t A^\pi(s_t, a_t)\right] = \frac{1}{1 - \gamma} \mathbb{E}_{\substack{s \sim d^{\pi'} \\ a \sim \pi'}} [A^\pi(s, a)] $$

where $d^\pi(s) = (1 - \gamma) \sum_{t=0}^{\infty} \gamma^t P(s_t = s | \pi)$ is the discounted state visitation distribution. This identity says: the improvement of $\pi'$ over $\pi$ equals the expected advantage of $\pi'$'s actions under $\pi$'s value function, weighted by $\pi'$'s state distribution.

### 10.4. Importance Sampling for Policy Optimization

The performance difference formula requires sampling states from the **new** policy $\pi'$, but we only have data from the **old** policy $\pi_{\theta_0}$. Applying importance sampling:

$$ \mathcal{J}(\theta) - \mathcal{J}(\theta_0) = \mathbb{E}_{\tau \sim (p_0, \pi_\theta, p)} \left[ \sum_{t=0}^{T-1} \gamma^t A^{\pi_{\theta_0}}(s_t, a_t) \right] $$

Using importance sampling to correct for the distribution mismatch:

$$ = \mathbb{E}_{\tau \sim (p_0, \pi_{\theta_0}, p)} \left[ \sum_{t=0}^{T-1} \gamma^t \frac{\pi_\theta(a_t' | s_t)}{\pi_{\theta_0}(a_t' | s_t)} A^{\pi_{\theta_0}}(s_t, a_t') \right] $$

where states are sampled from the **new** policy but actions are sampled from the **original** policy, with importance weights correcting the mismatch.

### 10.5. The Surrogate Objective

If the new policy is **sufficiently close** to the original one, we can define a **surrogate objective** that can be estimated entirely from data collected under $\pi_{\theta_0}$:

$$ \mathcal{K}(\theta; \theta_0) = \mathbb{E}_{\tau \sim (p_0, \pi_{\theta_0}, p)} \left[ \sum_{t=0}^{T-1} \gamma^t \frac{\pi_\theta(s_t, a_t)}{\pi_{\theta_0}(s_t, a_t)} A^{\pi_{\theta_0}}(s_t, a_t) \right] + C $$

Here both **states and actions** are sampled from the **original** policy, making this estimable from collected data. The surrogate can be approximated as:

$$ \mathcal{L}_{\theta_0}(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=0}^{T^{(i)}-1} \gamma^t \frac{\pi_\theta(s_t^{(i)}, a_t^{(i)})}{\pi_{\theta_0}(s_t^{(i)}, a_t^{(i)})} \hat{A}_t^{(i)} $$

We then maximize this surrogate objective subject to a **trust-region constraint** that keeps $\theta$ close to $\theta_0$:

$$ \max_{\theta \in \mathbb{R}^p} \frac{1}{N} \sum_{i=1}^{N} \sum_{t=0}^{T^{(i)}-1} \gamma^t \frac{\pi_\theta(s_t^{(i)}, a_t^{(i)})}{\pi_{\theta_0}(s_t^{(i)}, a_t^{(i)})} \hat{A}_t^{(i)}, \quad \text{subject to } \theta \text{ and } \theta_0 \text{ close} $$

### 10.6. Trust Region Policy Optimization (TRPO)

**TRPO** (Schulman et al., 2015) [18] operationalizes the surrogate objective by constraining policy updates to stay within a **trust region** defined by KL divergence:

$$ KL(\pi_{\theta_{\text{old}}}(\cdot | s_t) \| \pi_\theta(\cdot | s_t)) \leq \delta $$

```
Algorithm: Trust Region Policy Optimization (TRPO)
Input: initial policy parameters θ₀
For k = 0, 1, 2, ... do:
    1. Collect set of trajectories D_k on policy π_k = π(θ_k)
    2. Estimate advantages Â using any advantage estimation algorithm
    3. Form sample estimates for:
       • Policy gradient ĝ_k (using advantage estimates)
       • KL-divergence Hessian-vector product function f(v) = Ĥ_k v
    4. Use conjugate gradient (CG) with n_cg iterations to obtain x_k ≈ Ĥ_k⁻¹ ĝ_k
    5. Estimate proposed step Δ_k ≈ √(2δ / (x_kᵀ Ĥ_k x_k)) · x_k
    6. Perform backtracking line search with exponential decay to obtain final update:
       θ_{k+1} = θ_k + αʲ Δ_k
End for
```

### 10.7. Limitations of TRPO

While theoretically elegant, TRPO has significant **scalability issues**:

1. **Expensive Fisher Information Matrix computation:** Computing the Fisher Information Matrix $H$ for the current policy requires a large batch of rollouts:

$$ H = \nabla_\theta^2 KL(\pi_{\theta_t} \| \pi_\theta) = E_{a, s \sim \pi_{\theta_t}} \left[ \nabla_\theta \log \pi_\theta(a, s) \nabla_\theta \log \pi_\theta(a, s)^T \right] $$

2. **Conjugate gradient solver:** Determining the step size involves an approximate Newton method, solving $H^{-1}g$ via a conjugate gradient (CG) solver—a second-order method that is complex and computationally expensive.

Using a first-order Taylor approximation of the objective and second-order approximation of the KL constraint, the update becomes:

$$ \theta_{t+1} = \arg\max_\theta g^T(\theta - \theta_t) \quad \text{s.t. } \frac{1}{2}(\theta - \theta_t)^T H (\theta - \theta_t) \leq \delta $$

This can be solved analytically:

$$ \theta_{t+1} = \theta_t + \sqrt{\frac{2\delta}{g^T H^{-1} g}} H^{-1} g $$

### 10.8. Proximal Policy Optimization (PPO)

**PPO** (Schulman et al., 2017) [19] achieves performance comparable to TRPO while being **much simpler to implement**. Instead of solving a constrained optimization problem with second-order methods, PPO uses first-order methods (SGD/Adam) with a modified objective.

**PPO-Penalty:** The trust region constraint is reformulated as an **unconstrained optimization** with KL divergence as a regularizer:

$$ \max_\theta \mathbb{E}_t \left[ \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)} A_t \right] - \beta \, \mathbb{E}_t \left[ KL[\pi_{\theta_{\text{old}}}(\cdot | s_t), \pi_\theta(\cdot | s_t)] \right] $$

The penalty coefficient $\beta$ is **adaptive** between iterations to approximately enforce the KL-divergence constraint:
- If $\bar{D}_{KL}(\theta_{k+1} \| \theta_k) \geq 1.5\delta$: $\beta_{k+1} = 2\beta_k$ (policy moved too far, increase penalty)
- If $\bar{D}_{KL}(\theta_{k+1} \| \theta_k) \leq \delta / 1.5$: $\beta_{k+1} = \beta_k / 2$ (policy moved too little, decrease penalty)

```
Algorithm: PPO with Adaptive KL Penalty
Input: initial policy parameters θ₀, initial KL penalty β₀, target KL-divergence δ
For k = 0, 1, 2, ... do:
    1. Collect set of partial trajectories D_k on policy π_k = π(θ_k)
    2. Estimate advantages Â using any advantage estimation algorithm
    3. Compute policy update:
       θ_{k+1} = arg max_θ L_{θ_k}(θ) - β_k D̄_KL(θ || θ_k)
       by taking K steps of minibatch SGD (via Adam)
    4. If D̄_KL(θ_{k+1} || θ_k) ≥ 1.5δ then β_{k+1} = 2β_k
       else if D̄_KL(θ_{k+1} || θ_k) ≤ δ/1.5 then β_{k+1} = β_k / 2
End for
```

### 10.9. PPO-Clip

The most widely used variant of PPO is **PPO-Clip**, which avoids computing KL divergence entirely by using a **clipped surrogate objective**:

Define the probability ratio:

$$ r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_k}(a_t | s_t)} $$

Note that $r_t(\theta)$ is close to 1 when the new policy is close to the original one. The clipped objective is:

$$ \mathcal{L}_{\theta_k}^{CLIP}(\theta) = \mathbb{E}_{\tau \sim \pi_k} \left[ \sum_{t=0}^{T} \left[ \min\left( r_t(\theta) \hat{A}_t^{\pi_k}, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t^{\pi_k} \right) \right] \right] $$

where $\epsilon$ is a hyperparameter (typically $\epsilon = 0.2$). The policy update is simply:

$$ \theta_{k+1} = \arg\max_\theta \mathcal{L}_{\theta_k}^{CLIP}(\theta) $$

**How clipping works:**
- When the advantage $A > 0$ (good action): the objective increases with $r_t(\theta)$, but clipping at $1 + \epsilon$ prevents the ratio from growing too large.
- When the advantage $A < 0$ (bad action): the objective increases as $r_t(\theta)$ decreases, but clipping at $1 - \epsilon$ prevents the ratio from shrinking too much.

The final clipped objective is a **lower (pessimistic) bound** of the unclipped objective, which means it errs on the side of caution. The clipping removes the incentive for the policy to move far from $\theta_k$, providing a simple mechanism for trust-region enforcement without any KL divergence computation.

```
Algorithm: PPO with Clipped Objective
Input: initial policy parameters θ₀, clipping threshold ε
For k = 0, 1, 2, ... do:
    1. Collect set of partial trajectories D_k on policy π_k = π(θ_k)
    2. Estimate advantages Â using any advantage estimation algorithm
    3. Compute policy update:
       θ_{k+1} = arg max_θ L^CLIP_{θ_k}(θ)
       by taking K steps of minibatch SGD (via Adam), where:
       L^CLIP_{θ_k}(θ) = E_{τ~π_k} [Σ_t min(r_t(θ) Â_t, clip(r_t(θ), 1-ε, 1+ε) Â_t)]
End for
```

**Key advantages of PPO-Clip:**
- PPOs have the **stability and reliability of TRPO** but are **much simpler to implement**.
- Only requires first-order optimization (SGD or Adam).
- No need to compute KL divergence or the Fisher Information Matrix.
- Achieves comparable or better performance than TRPO on standard benchmarks.

### 10.10. PPO: Implementation Challenges

Despite its apparent simplicity, PPO implementation is **tricky and challenging** in practice. Hsu, Mendler-Dünner, and Hardt (2020) [20] and Engstrom et al. (2020) [21] highlight several important caveats:

1. **Failure modes of standard PPO:**
   - On continuous action spaces, standard PPO is unstable when rewards vanish outside bounded support.
   - On discrete action spaces with sparse high rewards, standard PPO often gets stuck at suboptimal actions.
   - The policy is sensitive to initialization when there are locally optimal actions close to initialization.

2. **Implementation details matter enormously:** "Code-level optimizations" that are described as auxiliary details in implementations turn out to have a **major impact on agent behavior**. These optimizations are responsible for most of PPO's gain in cumulative reward over TRPO, fundamentally changing how the RL methods function.

These findings serve as a reminder that many algorithmic design choices in deep RL are tied to specific simulation environments, and standard design choices should not be implicitly accepted as universal defaults.

## 11. Reinforcement Learning for Large Language Models

One of the most impactful applications of modern deep RL—and of PPO in particular—is in the **post-training of large language models (LLMs)**. Reinforcement learning from human feedback (RLHF) has become a central technique for aligning LLMs with human preferences, addressing issues of safety, helpfulness, and instruction-following that supervised training alone cannot fully resolve [22, 23] (see [26] for an accessible illustrated tutorial and [29] for a hands-on short course).

### 11.1. The LLM Training Pipeline

Modern LLMs are trained in two major phases:

**Pretraining:**
- Very large models are trained via **unsupervised learning** on a gigantic web-scale dataset of texts and documents—essentially the entire archive of human written knowledge.
- Key enablers: **GPUs** for fast computation, **data** freely available from the Internet, the **Transformer** architecture, and substantial **financial investment**.
- The output is a **base LLM** that can generate fluent text but may not follow instructions well, may produce harmful content, or may hallucinate.

**Post-training:**
- The base LLM is refined for specific downstream tasks through **supervised fine-tuning** and **reinforcement learning**.
- The goal is to slightly adjust the pre-trained model for subsequent tasks, particularly to address **alignment** and **safety** issues.
- RLHF has become the dominant technique for this alignment phase [22].

### 11.2. Post-training Components

The post-training pipeline consists of several components, which can be organized into two tracks:

**Track 1: Reasoning Models (RL-CoT)**
1. **Supervised Fine-Tuning (SFT)** → RL with Chain-of-Thought (CoT) reasoning → Reasoning Model

**Track 2: Non-reasoning Models (RLHF)**
1. **Supervised Fine-Tuning (SFT)** + **Reward Model (RM) Training** → RLHF → Non-reasoning Model

The four key components are:

1. **Supervised Fine-Tuning (SFT):** Behavioral cloning of human or expert behaviors. The model learns to mimic high-quality demonstrations.
2. **Reward Model (RM) Training:** Learning a model of human preferences from comparative judgments.
3. **RLHF:** Optimizing the fine-tuned LLM against the reward model using RL (typically PPO).
4. **RL without RM:** Reasoning with (long) Chain-of-Thoughts (CoTs), enabling test-time scaling without an explicit reward model.

### 11.3. Reward Modeling

**Motivation:** Supervised fine-tuning has limitations:
- Open-ended questions lack a single correct answer.
- Some token prediction errors are more serious than others.
- It is expensive to create high-quality demonstration data.

**Approach:** Instead of demonstrating the correct answer, human labelers are asked to **rank $K$ LLM-generated responses** to a prompt. This comparative judgment is much easier and cheaper than writing ideal responses.

The reward model $r_\theta$ is trained using the **Bradley-Terry model** (a classic method of paired comparisons) with the following loss function:

$$ \text{loss}(\theta) = -\frac{1}{\binom{K}{2}} E_{(x, y_w, y_l) \sim D} \left[ \log \sigma\left( r_\theta(x, y_w) - r_\theta(x, y_l) \right) \right] $$

where $x$ is the prompt, $y_w$ is the winning (preferred) response, $y_l$ is the losing response, and $\sigma$ is the sigmoid function.

**Key properties of reward models:**
- The RM is typically a "small" language model (e.g., GPT-3 with 6B parameters).
- RMs help **generalize LLM evaluations to difficult-to-verify tasks**, where correctness is hard to define.
- RMs **save huge costs** compared to recruiting human labelers for every evaluation.
- However, RMs are subject to **reward hacking**—the LLM may learn to exploit weaknesses in the RM to achieve high reward scores without actually improving output quality (Weng, 2024) [24].

### 11.4. The MDP Formulation of RLHF

RLHF finetunes the LLM $\pi$ further so that the completion achieves high reward as measured by the reward model $r(\cdot)$. To apply RL, we must first formulate the problem as an MDP [25]:

**What is the MDP in RLHF?**
- LLMs are **autoregressive** models, so they are not Markovian but history-dependent: the policy is not of the form $\pi(a_t | s_t)$ but rather $\pi(a_t | s_1, s_2, \ldots, s_t)$.
- Therefore, the **state** should be the user prompt and the tokens generated so far: $(u_1, u_2, \ldots, u_l)$.
- Each **action** is the generation of one token. The policy is randomized (sampling from the distribution over tokens), but the transition dynamics is **deterministic** (appending the chosen token to the sequence).

**The RL Setup:**
- Each timestep is a **BPE token**.
- The LLM $\pi_\theta(u_{l+1} | u_1, \ldots, u_l)$ is our policy mapping the current state (token sequence) to a distribution on the action (next token) $u_{l+1}$.
- Response generation is an **episode**, and an episode terminates when the LM generates `<EOS>`.
- **No discount** is used, i.e., discount factor $\gamma = 1$.
- Reward (by reward model $r_\psi$) is only provided at the **end of the episode**. There are no intermediate rewards. This is called the **"contextual bandit" setting**.
- Sampling temperature $\beta = 1$.

### 11.5. PPO for RLHF

Let $\pi_\theta(u_{l+1} | u_1, \ldots, u_l)$ be our LLM and the RL policy. Let $x$ be a text prompt and $y = y_{1:T+1}$ be its completion by $\pi_\theta$ (so $y_{T+1}$ = `<EOS>`). Let $y_{1:t}$ denote the partial completion up to token $t$. The PPO-Clip ratio is set to $\varepsilon = 0.2$.

PPO maintains a **value function model** $V_\phi(x, y_{1:t})$: given the prompt and partial completion $(x, y_{1:t})$, what is the expected reward if we continue generation with $\pi_\theta$?

The **advantage** is estimated as:

$$ \hat{A} = r_\psi(x, y_{1:T+1}) - V_\phi(x, y_{1:t}) $$

This measures: how good is the total completion $y_{t+1:T+1}$ compared to what $V_\phi$ was expecting based on $y_{1:t}$?

- If $\hat{A} = r_\psi(x, y_{1:T+1}) - V_\phi(x, y_{1:t}) > 0$, then $y_{t+1:T+1}$ was a **good completion**. We should adjust $\pi_\theta$ to make those actions more likely.
- If $\hat{A} = r_\psi(x, y_{1:T+1}) - V_\phi(x, y_{1:t}) < 0$, then $y_{t+1:T+1}$ was a **bad completion**. We should adjust $\pi_\theta$ to make those actions less likely.

(In practice, Generalized Advantage Estimation (GAE) is used for $\hat{A}$, but the simpler advantage estimate above conveys the key intuition.)

### 11.6. PPO v0: Susceptibility to Reward Hacking

The basic PPO algorithm for RLHF (PPO v0) proceeds as follows:

```
While (not converged):
    1. Sample N trajectories (x⁽ⁱ⁾, y⁽ⁱ⁾_{1:T⁽ⁱ⁾+1}) ~ (p₀, π^RL_{θ_curr})  for i = 1,...,N
    2. Compute advantages:
       Â⁽ⁱ⁾_t = r_ψ(x⁽ⁱ⁾, y⁽ⁱ⁾_{1:T⁽ⁱ⁾+1}) - V_φ(x⁽ⁱ⁾, y⁽ⁱ⁾_{1:t})  for all i, t
    3. Solve (policy update):
       maximize_{θ_next} Σᵢ Σₜ C_ε(π^RL_{θ_next}(y⁽ⁱ⁾_{t+1}|x⁽ⁱ⁾,y⁽ⁱ⁾_{1:t}) / π^RL_{θ_curr}(y⁽ⁱ⁾_{t+1}|x⁽ⁱ⁾,y⁽ⁱ⁾_{1:t}), Â⁽ⁱ⁾_t)
    4. Set θ_curr = θ_next
    5. Solve (value function update):
       minimize_φ (1/N) Σᵢ (1/T⁽ⁱ⁾) Σₜ ½(r_ψ(x⁽ⁱ⁾, y⁽ⁱ⁾_{1:T⁽ⁱ⁾+1}) - V_φ(x⁽ⁱ⁾, y⁽ⁱ⁾_{1:t}))²
End
```

However, this basic version is **susceptible to reward hacking**.

### 11.7. Reward Hacking and Goodhart's Law

**Goodhart's Law** states: *"When a measure becomes a target, it ceases to be a good measure."* This principle is directly relevant to RLHF [24]:

- The reward model is **imperfect**. Maximizing an imperfect reward model too aggressively will exploit the model's imperfections and result in **adversarial generations**.
- The reward model is trained on the SFT LLM $\pi_{\text{SFT}}$. Therefore, the reward model is only informative about responses generated by the RL policy if $\pi_{\text{RL}}$ is **close** to $\pi_{\text{SFT}}$.
- Moving away from $\pi_{\text{SFT}}$ too much will cause the language model to **lose its main capabilities**. Fine-tuning too much can break the model, causing it to output nonsense tokens.

**Solution:** Impose a **KL-divergence penalty term**, ensuring that $\pi_{\text{RL}}$ stays close to $\pi_{\text{SFT}}$.

### 11.8. KL-Penalty and Pre-training Loss

RLHF with KL-penalty maximizes the following objective:

$$ \mathcal{J}(\theta) = \mathbb{E}_{(x,y) \sim D_{\pi_\theta^{RL}}} \left[ r_\psi(x, y) \right] - \beta \, \mathbb{E}_{x \sim \mathcal{D}} \left[ D_{KL}\left( \pi_\theta^{RL}(\cdot | x) \| \pi^{SFT}(\cdot | x) \right) \right] $$

The KL divergence can be decomposed at the token level:

$$ = \mathbb{E}_{(x,y) \sim D_{\pi_\theta^{RL}}} \left[ r_\psi(x, y) - \beta \log \frac{\pi_\theta^{RL}(y | x)}{\pi^{SFT}(y | x)} \right] $$

$$ = \mathbb{E}_{(x,y) \sim D_{\pi_\theta^{RL}}} \left[ r_\psi(x, y) - \beta \sum_{t=0}^{T} \log \frac{\pi_\theta^{RL}(y_{t+1} | x, y_{1:t})}{\pi^{SFT}(y_{t+1} | x, y_{1:t})} \right] $$

The KL-penalty encourages $\pi_{\text{RL}}$ to stay **close** to $\pi_{\text{SFT}}$. (See Schulman, 2020 [28] for practical considerations on approximating KL divergence in this setting.)

**Absorbing the KL penalty into the MDP:** Maximizing $\mathcal{J}(\theta)$ is equivalent to solving RL on an MDP with the same transition dynamics but **modified rewards** $r_0, r_1, \ldots, r_T$:

$$ r_t = -\beta \log \frac{\pi_\theta^{RL}(y_{t+1} | x, y_{1:t})}{\pi^{SFT}(y_{t+1} | x, y_{1:t})}, \quad t = 0, \ldots, T-1 $$

$$ r_T = r_\psi(x, y_{1:T+1}) - \beta \log \frac{\pi_\theta^{RL}(y_{T+1} | x, y_{1:T})}{\pi^{SFT}(y_{T+1} | x, y_{1:T})} $$

The modified MDP has **immediate rewards** (at every token) which absorb the KL penalty. The reward depends on the parameter $\theta$, but all the math goes through.

**Adding the pre-training loss:** Even with the KL-penalty, the base language model capability can be compromised. Adding a **pre-training loss** term preserves general language abilities:

$$ \mathcal{J}(\theta) = \mathbb{E}_{(x,y) \sim D_{\pi_\theta^{RL}}} \left[ r_\psi(x, y) - \beta \log(\pi_\theta^{RL}(y|x) / \pi^{SFT}(y|x)) \right] + \gamma \, \mathbb{E}_{x \sim D_{\text{pretrain}}} \left[ \log(\pi_\theta^{RL}(x)) \right] $$

where $\gamma > 0$ and the last term is the next-token prediction loss used in pre-training. PPO and pre-training updates are performed **simultaneously or in alternating fashion**.

> **Connection to PPO:** The KL penalty in the RLHF objective plays exactly the same role as the trust region constraint in PPO—preventing the policy from moving too far from its starting point. This is why PPO (especially PPO-Penalty) is a natural fit for RLHF: the KL divergence between the RL policy and the SFT policy is precisely the trust-region constraint that PPO enforces.

### 11.9. Full PPO Algorithm for RLHF

The complete PPO algorithm with KL-penalty for RLHF is:

```
While (not converged):
    1. Sample N trajectories (x⁽ⁱ⁾, y⁽ⁱ⁾_{1:T⁽ⁱ⁾+1}) ~ (p₀, π^RL_{θ_curr})  for i = 1,...,N
    2. Compute advantages with modified rewards:
       Â⁽ⁱ⁾_t = r_ψ(x⁽ⁱ⁾, y⁽ⁱ⁾_{1:T⁽ⁱ⁾+1})
             - β Σₛ log(π^RL_{θ_next}(y⁽ⁱ⁾_{s+1}|x⁽ⁱ⁾,y⁽ⁱ⁾_{1:s}) / π^SFT(y⁽ⁱ⁾_{s+1}|x⁽ⁱ⁾,y⁽ⁱ⁾_{1:s}))
             - V_φ(x⁽ⁱ⁾, y⁽ⁱ⁾_{1:t})
    3. Solve (policy update with PPO-Clip):
       maximize_{θ_next} Σᵢ Σₜ C_ε(π^RL_{θ_next}/π^RL_{θ_curr}, Â⁽ⁱ⁾_t)
       where C_ε(ℓ, A) = min(ℓA, clip(ℓ, 1-ε, 1+ε) A)
    4. Set θ_curr = θ_next
    5. Solve (value function update):
       minimize_φ (1/N) Σᵢ (1/T⁽ⁱ⁾) Σₜ ½(Ĝ⁽ⁱ⁾_t - V_φ(x⁽ⁱ⁾, y⁽ⁱ⁾_{1:t}))²
End
```

### 11.10. Empirical Performance of RLHF

The effectiveness of RLHF has been demonstrated in landmark studies:

**Learning to Summarize (Stiennon et al., 2020) [27]:** Models trained with human feedback significantly outperform both pretrain-only and supervised learning baselines on summarization tasks. Importantly, RLHF performance improves with model size, and human feedback provides gains that scale better than supervised learning alone.

**InstructGPT (Ouyang et al., 2022) [22]:** RLHF-trained models (PPO and PPO-ptx variants) consistently outperform SFT models on both the GPT and Instruct distributions. The improvements are measured by win rates against reference summaries, evaluated by both held-out workers and training workers. GPT models fine-tuned with PPO achieve the highest preference rates across model sizes from 1.3B to 175B parameters.

### 11.11. RLHF Can Be (Too) Complex

Despite its success, the full RLHF pipeline is **computationally expensive and tricky** to implement [30]:

- **RL optimization is expensive:** The pipeline requires maintaining and coordinating multiple models (policy LM, reference SFT model, reward model, value model/critic).
- **Value function fitting is required:** A separate value network must be trained alongside the policy.
- **Online sampling is slow:** Generating completions from the current policy at each iteration is a major bottleneck.
- **Performance is highly sensitive to hyperparameters**, including the KL penalty coefficient $\beta$, learning rates, and clipping thresholds.

These complexities motivate the search for simpler alternatives.

### 11.12. Direct Preference Optimization (DPO)

**Direct Preference Optimization (DPO)** (Rafailov et al., 2023) [31] elegantly sidesteps the complexity of RLHF by observing that the RL problem has a **closed-form solution**.

**Standard RLHF** requires two steps:
1. Fit the reward model based on human preference data.
2. Optimize the instruction-finetuned LLM given the reward function (via PPO).

**DPO's key insight:** There is another perspective:
1. Derive the reward model based on an RL policy, parameterized by $\theta$.
2. Optimize the parameter $\theta$ by **fitting the reward model to the true human preference data**, instead of fitting a given reward model.

**Closed-form solution for KL-regularized RL:** Ignoring the pre-training loss, the KL-regularized RL objective for any reward model $r$ is:

$$ \max_\theta \mathcal{J}(\pi_\theta; r) = \mathbb{E}_{\substack{x \sim \mathcal{D} \\ y \sim \pi_\theta}} \left[ r(x, y) - \beta \log \frac{\pi_\theta(y | x)}{\pi^{SFT}(y | x)} \right] $$

Transforming this as a loss and completing the algebra:

$$ -\frac{1}{\beta} \mathcal{J}(\pi_\theta; r) = \mathbb{E}_{\substack{x \sim \mathcal{D} \\ y \sim \pi_\theta}} \left[ \log \frac{\pi_\theta(y|x)}{\pi^{SFT}(y|x)} - \frac{1}{\beta} r(x, y) \right] $$

$$ = \mathbb{E}_{x \sim \mathcal{D}} \left[ D_{KL}(\pi_\theta(\cdot | x) \| \pi_r(\cdot | x)) \right] - \log Z(x) $$

where the **optimal policy** $\pi_r$ is:

$$ \pi_r(y | x) = \frac{\pi^{SFT}(y | x) \exp\left(\frac{1}{\beta} r(x, y)\right)}{Z(x)}, \quad Z(x) = \sum_y \pi^{SFT}(y | x) \exp\left(\frac{1}{\beta} r(x, y)\right) $$

Therefore, maximizing $\mathcal{J}(\pi_\theta; r)$ over $\theta$ is equivalent to minimizing $D_{KL}(\pi_\theta(\cdot | x) \| \pi_r(\cdot | x))$, i.e., **the optimal policy is $\pi_\theta = \pi_r$**.

**Inversely**, the reward function that makes any policy $\pi$ optimal is:

$$ r_\pi(x, y) = \beta \log \frac{\pi(y | x)}{\pi^{SFT}(y | x)} + \beta \log Z(x) $$

### 11.13. DPO: The Final Objective

Recall that the reward model was trained via the Bradley-Terry (BT) method:

$$ \min_\psi \sum_{\substack{(x, y_i, y_j) \in \mathcal{D} \\ y_i \succ y_j}} -\log \sigma(r_\psi(x, y_i) - r_\psi(x, y_j)) $$

DPO substitutes the closed-form reward $r_\pi$ into this loss. Since $\beta \log Z(x)$ cancels in the difference $r_\pi(x, y_i) - r_\pi(x, y_j)$, the DPO loss becomes:

$$ \mathcal{L}^{DPO}(\theta) = \sum_{\substack{(x, y_i, y_j) \in \mathcal{D} \\ y_i \succ y_j}} -\log \sigma\left( \beta \log \frac{\pi_\theta(y_i | x)}{\pi^{SFT}(y_i | x)} - \beta \log \frac{\pi_\theta(y_j | x)}{\pi^{SFT}(y_j | x)} \right) $$

**Key advantages of DPO:**
- There is **no need to train a reward model** and no need to train a value network.
- DPO effectively converts the RL problem into a **supervised learning problem**, so it is **much easier to execute**.

**The DPO gradient** is:

$$ \nabla_\theta \mathcal{L}^{DPO}(\theta) = -\beta \sum_{\substack{(x, y_i, y_j) \in \mathcal{D} \\ y_i \succ y_j}} \sigma\left(-(r_{\pi_\theta}(x, y_i) - r_{\pi_\theta}(x, y_j))\right) \left( \nabla_\theta \log \pi_\theta(y_i | x) - \nabla_\theta \log \pi_\theta(y_j | x) \right) $$

**Interpretation:** DPO is doing **ascent on the good outcome $y_i$** (increasing its likelihood) while doing **descent on the bad completion $y_j$** (decreasing its likelihood). The gradient is accentuated when the implicitly defined reward model $r_{\pi_\theta}$ **disagrees** with the human preference $y_i \succ y_j$—this is where the model needs the most correction.

There is ongoing debate on DPO vs. PPO for LLM alignment (see Xu et al., 2024 [32]).

### 11.14. Empirical Performance of DPO

DPO achieves competitive or superior performance compared to RLHF on standard benchmarks:

- On **summarization helpfulness**, DPO achieves the highest win rates against ground truth, outperforming PPO, Best-of-128 sampling, Preference-Filtered Training (PFT), and SFT.
- On **dialogue helpfulness**, DPO similarly leads in win rates, demonstrating that the closed-form approach does not sacrifice quality.
- The pipeline is dramatically simpler: no reward model training, no value function fitting, no online sampling, and standard supervised learning optimizers suffice.

### 11.15. Application: DPO for Balancing Engagement and Polarization

Chang, Obi, and Yoganarasimhan (2025) [33] demonstrate a compelling business application of DPO: using LLMs to generate news content that **balances engagement and polarization**.

**Problem:** Media firms face a multi-objective challenge—making content more engaging while maintaining a preferred level of polarization consistent with the firm's editorial policy. Using news articles from The New York Times, the authors show that more engaging human-written content tends to be more polarizing. Further, naively applying standard DPO approaches to generate more engaging content using LLMs without explicitly controlling for polarization can also increase polarization.

**Solution:** The authors propose **Multi-Objective Direct Preference Optimization (MODPO)** [34], a novel approach that integrates DPO with multi-objective optimization techniques. They build an open-source LLM that simultaneously makes content more engaging while maintaining a preferred editorial stance. Their model achieves this by modifying content characteristics strongly associated with polarization but that have a relatively smaller impact on engagement.

**Key takeaway:** This work illustrates how preference optimization techniques from RL can be applied to real-world content generation problems where multiple, potentially conflicting objectives must be balanced.

### 11.16. Chain-of-Thought (CoT) Reasoning

Beyond RLHF and DPO, another powerful technique for improving LLM performance is **Chain-of-Thought (CoT) prompting** (Wei et al., 2022 [35]; Kojima et al., 2022 [36]):

- CoT is a technique for LLMs to **"think aloud to itself"** before producing an answer.
- CoT **greatly improves the performance** compared with immediately producing an answer, especially on arithmetic, commonsense, and symbolic reasoning tasks.
- An easy way to induce CoT: append **"Let's think step by step"** to the prompt (zero-shot CoT).
- CoT can also be induced by **in-context prompting**—providing a few examples of step-by-step reasoning (few-shot CoT).
- Modern LLMs are increasingly **instruction-finetuned** to exhibit CoT behavior natively, enabling them to produce reasoning chains without explicit prompting.

**Connection to RL:** Recall from Section 11.2 that the post-training pipeline includes an **RL-CoT track** for reasoning models. In this track, RL is used to train models to produce long chains of thought, enabling **test-time scaling**—the model can allocate more computation at inference time by reasoning through more steps, improving accuracy on harder problems without any explicit reward model.

### 11.17. Reinforcement Learning with Verifiable Rewards (RLVR)

A key challenge in RLHF is that the reward model is **imperfect**—it is a learned proxy for human preferences, subject to reward hacking. However, some domains offer a fundamentally different opportunity: **coding and mathematics** are challenging reasoning tasks with **verifiable rewards** (Chen et al., 2021 [37]; Shao et al., 2024 [38]).

- Automatic code generation was a longstanding challenge, until LLMs emerged as a solution. Code data with comments written in natural language is plentiful on the Internet, so modern LLMs are explicitly trained on code together with natural language data.
- More interestingly, coding improves non-coding reasoning capabilities as well, such as math.
- Training AI on math also improves general reasoning capabilities.

The key insight is that coding and math problems have **objectively verifiable solutions**: a code submission either passes the test suite or it does not; a mathematical answer is either correct or incorrect. This eliminates the need for a learned reward model entirely.

**Reinforcement Learning with Verifiable Rewards (RLVR)** exploits this structure:

$$ \text{High-quality reasoning data} \rightarrow \text{Base LLM} \rightarrow \text{Reinforcement Learning} \rightarrow \text{Reasoning LLM} $$

Since the reward is **exact** (verifiable), there is no risk of reward hacking or overfitting to an imperfect reward model. Only **outcome rewards** are provided (whether the final answer is correct). Intermediate rewards on partial credits, i.e., **process rewards**, are NOT used.

### 11.18. Group Relative Policy Optimization (GRPO)

**DeepSeekMath** (Shao et al., 2024) [38] proposes **Group Relative Policy Optimization (GRPO)**, a key algorithmic innovation that simplifies PPO for the RLVR setting.

**Motivation:** Standard PPO for RLHF requires maintaining four models simultaneously: the policy model, a reference model (for KL penalty), a reward model, and a **value model** (critic). The value model is expensive to train and maintain. GRPO eliminates the value model entirely by using **group-level relative advantages** instead.

**The GRPO Algorithm:**

```
While (not converged):
    1. Sample a problem x
    2. Sample G responses y⁽¹⁾, ..., y⁽ᴳ⁾ ~ π_{θ_curr}(· | x)
    3. Evaluate the rewards r = (r⁽¹⁾, ..., r⁽ᴳ⁾) for y⁽¹⁾, ..., y⁽ᴳ⁾
    4. Solve:
       maximize_{θ_next} Σᵢ₌₁ᴳ (1/|T⁽ⁱ⁾|) Σₜ₌₀ᵀ⁽ⁱ⁾ (term₁ + term₂)

       term₁ = C_ε(π_{θ_next}(y⁽ⁱ⁾_{t+1} | x, y⁽ⁱ⁾_{1:t}) / π_{θ_curr}(y⁽ⁱ⁾_{t+1} | x, y⁽ⁱ⁾_{1:t}),
                        (r⁽ⁱ⁾ - mean(r)) / (std(r) + ε_A))

       term₂ = -β (π^SFT(y⁽ⁱ⁾_{s+1} | x⁽ⁱ⁾, y⁽ⁱ⁾_{1:s}) / π_{θ_next}(y⁽ⁱ⁾_{s+1} | x⁽ⁱ⁾, y⁽ⁱ⁾_{1:s})
                     - log(π^SFT(y⁽ⁱ⁾_{s+1} | x⁽ⁱ⁾, y⁽ⁱ⁾_{1:s}) / π^RL_{θ_next}(y⁽ⁱ⁾_{s+1} | x⁽ⁱ⁾, y⁽ⁱ⁾_{1:s})))

    5. Set θ_curr = θ_next
End
```

**Key innovations of GRPO:**

1. **Group Relative Advantage:** Instead of training a value function to estimate advantages, GRPO samples a **group** of $G$ responses to the same problem and computes the advantage by **normalizing rewards within the group**:

$$ \hat{A}^{(i)} = \frac{r^{(i)} - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r}) + \varepsilon_A} $$

This eliminates the need for a separate value model—the group statistics serve as a natural baseline. Responses that are better than the group average receive positive advantage; those worse receive negative advantage.

2. **Length Normalization:** The objective divides by $|T^{(i)}|$ (the response length), preventing the optimization from favoring shorter or longer responses.

3. **Unbiased KL Estimate:** The KL penalty term uses an unbiased estimator of the KL divergence between the RL policy and the SFT reference policy, rather than the standard log-ratio approximation.

**Comparison with PPO:** While PPO maintains four models (policy, reference, reward, value), GRPO only needs three (policy, reference, reward)—or just two (policy, reference) when rewards are verifiable and no learned reward model is needed. This makes GRPO substantially more memory-efficient and simpler to implement.

### 11.19. DeepSeek-R1: Scaling Up RLVR

**DeepSeek-R1** (Guo et al., 2025) [39] scales up the RLVR approach to coding, math, science, and other reasoning tasks, demonstrating that RL with verifiable rewards can produce reasoning capabilities competitive with frontier models.

**Training Pipeline:** DeepSeek-R1 is built through a multi-stage process:

**Step 1: Knowledge Distillation (Open-R1-Distill-8B)**
- Directly imitate the outputs of a strong reasoning model (DeepSeek-R1) through supervised fine-tuning on distilled reasoning data.
- This produces a capable but not yet optimal model.

**Step 2: DeepSeek-R1-Zero**
- Directly apply RLVR to a base model (DeepSeek-V3) using GRPO with verifiable rewards—**no SFT, no reward model, just pure RL**.
- The model learns to produce long CoT reasoning chains.
- However, the resulting CoT often suffers from **poor readability** (mixing languages, disorganized reasoning steps).

**Step 3: DeepSeek-R1**
- **Cold-start with SFT**: First fine-tune on a small amount of curated reasoning data to establish good formatting and readability.
- Then combine **RLVR and RLHF with GRPO**: Apply RL with both verifiable rewards (for reasoning correctness) and human feedback (for readability and helpfulness).
- This yields the final DeepSeek-R1 model with both strong reasoning and good output quality.

**Key Results:**
- The model learns to utilize **CoT very extensively**, with average response lengths growing significantly during training (from ~2,500 tokens to ~12,500+ tokens over 10,000 training steps).
- On the AIME mathematical reasoning benchmark, DeepSeek-R1-Zero achieves accuracy that surpasses human participants, improving steadily throughout training.

### 11.20. Emergence of Reasoning Through RLVR

Perhaps the most striking finding from the DeepSeek-R1 work is that **RLVR induces the emergence of sophisticated reasoning behaviors** that are **not explicitly programmed** [39]:

- As test-time computation increases, the model develops key **cognitive behaviors for reasoning**, such as:
  - **Verification:** Checking intermediate results for correctness.
  - **Back-tracking:** Recognizing errors and revising the approach.
  - **Subgoal setting:** Breaking complex problems into manageable sub-problems.
  - **Backward chaining:** Working from the desired conclusion back to the premises.
  - **Self-evolution:** The model refines its own reasoning strategies over the course of training.

**The "Aha Moment" of DeepSeek-R1:** During training, the model exhibits a remarkable behavior: it pauses mid-solution, recognizes that something is wrong ("Wait, wait. Wait. That's an aha moment I can flag here."), re-evaluates its approach step by step, and arrives at the correct answer. This emergent self-correction behavior was never explicitly trained—it arose purely from RL optimization against verifiable rewards.

**Why does this work?** The argument is not that "RL didn't work before"—PPO on base models with verifiable rewards had shown strong results previously. Rather, the breakthrough is that with **sufficient base model knowledge** and **sufficient RL compute and context window length**, the model develops **long CoT with internal reasoning emergence**: verification, error-correction, and branching-like behavior that earlier, smaller models could not sustain.

> **The broader significance:** RLVR represents a paradigm where RL can train reasoning capabilities without any human-generated reasoning traces or learned reward models. Combined with the group relative advantage of GRPO that eliminates the value network, this approach dramatically simplifies the pipeline from base model to reasoning model: just RL with verifiable rewards.

## 12. Conclusion

Deep reinforcement learning represents a powerful synthesis of deep learning's representational capacity with reinforcement learning's sequential decision-making framework. The progression from tabular RL to function approximation to deep RL follows a natural path of increasing scalability and expressiveness:

1. **Value function approximation** replaces the lookup table with a parameterized function, enabling generalization across states and scaling to large problems.
2. **Linear approximation** provides a tractable starting point with convergence guarantees, but requires manual feature engineering.
3. **Deep Q-Networks** leverage neural networks for automatic feature learning, with experience replay and fixed targets to stabilize training.
4. **Policy gradient methods** optimize the policy directly, enabling handling of continuous and high-dimensional action spaces, stochastic policies, and settings where value-based methods struggle.
5. **Actor-critic methods** combine the best of both worlds: a critic for low-variance value estimation and an actor for direct policy optimization, culminating in scalable algorithms like A3C.
6. **Modern DRL algorithms**—TRPO and PPO—address the instability of vanilla policy gradients through constrained optimization in policy space, with PPO emerging as the practical workhorse due to its simplicity and effectiveness.
7. **RL for LLMs** applies these techniques—particularly PPO—to align large language models with human preferences through RLHF, representing one of the most impactful real-world applications of deep RL to date.
8. **Direct Preference Optimization (DPO)** simplifies RLHF by exploiting a closed-form solution for KL-regularized RL, converting the alignment problem into supervised learning and eliminating the need for reward models, value networks, and online sampling.
9. **Chain-of-Thought (CoT) reasoning** enables LLMs to "think step by step," dramatically improving performance on reasoning tasks and forming the basis of test-time scaling in modern reasoning models.
10. **RLVR and GRPO** demonstrate that for domains with verifiable rewards (coding, math, science), RL can train reasoning capabilities without learned reward models or value networks, and remarkably, sophisticated reasoning behaviors such as self-correction and back-tracking **emerge** from pure RL optimization.

The key takeaways for business researchers are:
- **The deadly triad** (bootstrapping + function approximation + off-policy learning) is a fundamental source of instability. DQN addresses it through engineering innovations (experience replay and fixed targets) rather than theoretical fixes.
- **Policy gradient methods** provide a complementary approach to value-based methods, with distinct advantages for continuous action spaces, stochastic policies, and LLM fine-tuning (e.g., RLHF).
- **Variance reduction** is central to making policy gradient methods practical. Temporal causality, baselines, and actor-critic architectures progressively reduce the variance of gradient estimates.
- **Trust regions and clipping** (TRPO and PPO) are essential for stable policy optimization. The insight that RL training can collapse from a single bad update—unlike supervised learning—motivates constraining policy changes, whether through KL divergence constraints or clipped objectives.
- **RLHF and reward modeling** have emerged as the dominant paradigm for post-training LLMs. The connection between PPO's trust-region approach and the KL penalty in RLHF highlights how foundational RL concepts directly enable modern AI alignment. However, reward hacking (Goodhart's Law) remains a fundamental challenge.
- **DPO offers a simpler alternative** to the full RLHF pipeline, with competitive empirical performance. Its application to multi-objective content generation (e.g., balancing engagement and polarization) demonstrates the versatility of preference optimization for business applications.
- Deep RL opens the door to applications with high-dimensional state spaces—such as dynamic pricing, personalized recommendations, adaptive marketing, inventory management, customer journey optimization, and LLM-powered content generation—where tabular methods are infeasible.
- **RLVR and emergent reasoning** represent a frontier where RL with verifiable rewards produces sophisticated cognitive behaviors—verification, back-tracking, self-correction—without explicit programming. GRPO's elimination of the value network further simplifies the training pipeline, pointing toward a future where reasoning capabilities arise naturally from scale and RL optimization.
- **Extensions** like Double DQN, Prioritized Replay, Dueling DQN, A3C, TRPO, PPO, DPO, GRPO, and CoT reasoning offer further improvements and continue to be active areas of research.

## References

[1] Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. [http://incompleteideas.net/book/the-book-2nd.html](http://incompleteideas.net/book/the-book-2nd.html)

[2] Silver, D. (2015). *Lectures on Reinforcement Learning*. University College London. [https://www.davidsilver.uk/teaching/](https://www.davidsilver.uk/teaching/)

[3] Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., et al. (2015). *Human-level control through deep reinforcement learning*. Nature, 518(7540), 529–533.

[4] Van Hasselt, H., Guez, A., & Silver, D. (2016). *Deep reinforcement learning with double Q-learning*. Proceedings of the AAAI Conference on Artificial Intelligence, 30(1).

[5] Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2016). *Prioritized experience replay*. Proceedings of the International Conference on Learning Representations (ICLR).

[6] Wang, Z., Schaul, T., Hessel, M., et al. (2016). *Dueling network architectures for deep reinforcement learning*. Proceedings of the International Conference on Machine Learning (ICML).

[7] Mnih, V., Badia, A. P., Mirza, M., Graves, A., et al. (2016). *Asynchronous methods for deep reinforcement learning*. Proceedings of the International Conference on Machine Learning (ICML).

[8] Liu, X. (2023). *Dynamic coupon targeting using batch deep reinforcement learning: An application to livestream shopping*. Marketing Science, 42(4), 610–636.

[9] Song, Y., & Sun, T. (2024). *Ensemble experiments to optimize interventions along the customer journey: A reinforcement learning approach*. Management Science, 70(8), 5117–5139.

[10] Wang, W., Li, B., Luo, X., & Wang, X. (2023). *Deep reinforcement learning for sequential targeting*. Management Science, 69(9), 5382–5404.

[11] Kekouos, M., & Ipeirotis, P. G. (2021). *Demand-aware career path recommendations: A reinforcement learning approach*. Management Science, 67(7), 4030–4050.

[12] Gijsbrechts, J., Boute, R. N., Van Mieghem, J. A., & Zhang, D. J. (2022). *Can deep reinforcement learning improve inventory management? Performance on lost sales, dual-sourcing, and multi-echelon problems*. Manufacturing & Service Operations Management, 24(3), 1664–1677.

[13] Sutton, R. S., McAllester, D., Singh, S., & Mansour, Y. (1999). *Policy gradient methods for reinforcement learning with function approximation*. Advances in Neural Information Processing Systems (NeurIPS), 12.

[14] Williams, R. J. (1992). *Simple statistical gradient-following algorithms for connectionist reinforcement learning*. Machine Learning, 8(3), 229–256.

[15] Tsitsiklis, J. N., & Van Roy, B. (1997). *An analysis of temporal-difference learning with function approximation*. IEEE Transactions on Automatic Control, 42(5), 674–690.

[16] Bertsekas, D. P., & Tsitsiklis, J. N. (1996). *Neuro-Dynamic Programming*. Athena Scientific.

[17] Weng, L. (2018). *Policy Gradient Algorithms*. Lil'Log. [https://lilianweng.github.io/posts/2018-04-08-policy-gradient/](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/)

[18] Schulman, J., Levine, S., Abbeel, P., Jordan, M., & Moritz, P. (2015). *Trust region policy optimization*. Proceedings of the International Conference on Machine Learning (ICML).

[19] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal policy optimization algorithms*. arXiv preprint arXiv:1707.06347.

[20] Hsu, C. C.-Y., Mendler-Dünner, C., & Hardt, M. (2020). *Revisiting design choices in proximal policy optimization*. arXiv preprint arXiv:2009.10897.

[21] Engstrom, L., Ilyas, A., Santurkar, S., Tsipras, D., Janoos, F., Rudolph, L., & Madry, A. (2020). *Implementation matters in deep policy gradients: A case study on PPO and TRPO*. Proceedings of the International Conference on Learning Representations (ICLR).

[22] Ouyang, L., Wu, J., Jiang, X., Almeida, D., et al. (2022). *Training language models to follow instructions with human feedback*. Advances in Neural Information Processing Systems (NeurIPS), 35.

[23] Ziegler, D. M., Stiennon, N., Wu, J., Brown, T. B., et al. (2019). *Fine-tuning language models from human preferences*. arXiv preprint arXiv:1909.08593.

[24] Weng, L. (2024). *Reward Hacking in Reinforcement Learning*. Lil'Log. [https://lilianweng.github.io/posts/2024-11-28-reward-hacking/](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/)

[25] Ryu, E. K. (2025). *RL for LLM*. Course Notes, Stanford University. [https://ernestryu.com/courses/RL-LLM/chapter1.pdf](https://ernestryu.com/courses/RL-LLM/chapter1.pdf)

[26] Hugging Face. (2022). *Illustrating Reinforcement Learning from Human Feedback (RLHF)*. Hugging Face Blog. [https://huggingface.co/blog/rlhf](https://huggingface.co/blog/rlhf)

[27] Stiennon, N., Ouyang, L., Wu, J., Ziegler, D. M., et al. (2020). *Learning to summarize with human feedback*. Advances in Neural Information Processing Systems (NeurIPS), 33. arXiv:2009.01325.

[28] Schulman, J. (2020). *Approximating KL Divergence*. [http://joschu.net/blog/kl-approx.html](http://joschu.net/blog/kl-approx.html)

[29] DeepLearning.AI. (2023). *Reinforcement Learning from Human Feedback*. Short Course. [https://learn.deeplearning.ai/courses/reinforcement-learning-from-human-feedback](https://learn.deeplearning.ai/courses/reinforcement-learning-from-human-feedback)

[30] Zheng, R., Dou, S., Gao, S., Hua, Y., et al. (2023). *Secrets of RLHF in large language models (Part I): PPO*. arXiv preprint arXiv:2307.04964.

[31] Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., & Finn, C. (2023). *Direct preference optimization: Your language model is secretly a reward model*. Advances in Neural Information Processing Systems (NeurIPS), 36. [https://arxiv.org/abs/2305.18290](https://arxiv.org/abs/2305.18290)

[32] Xu, S., Fu, W., Gao, J., Ye, W., Liu, W., Mei, Z., Wang, G., Yu, C., & Wu, Y. (2024). *Is DPO superior to PPO for LLM alignment? A comprehensive study*. arXiv preprint arXiv:2404.10719.

[33] Chang, M., Obi, E., & Yoganarasimhan, H. (2025). *Balancing engagement and polarization: Multi-objective alignment of news content using LLMs*. arXiv preprint arXiv:2504.13444.

[34] Li, Z., Liu, M., Chen, D., Lyu, M., Wang, S., & Zheng, Z. (2024). *Beyond one-preference-fits-all alignment: Multi-objective direct preference optimization*. Findings of the Association for Computational Linguistics (ACL). [https://aclanthology.org/2024.findings-acl.630/](https://aclanthology.org/2024.findings-acl.630/)

[35] Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., Chi, E., Le, Q., & Zhou, D. (2022). *Chain-of-thought prompting elicits reasoning in large language models*. Advances in Neural Information Processing Systems (NeurIPS), 35. [https://arxiv.org/abs/2201.11903](https://arxiv.org/abs/2201.11903)

[36] Kojima, T., Gu, S. S., Reid, M., Matsuo, Y., & Iwasawa, Y. (2022). *Large language models are zero-shot reasoners*. Advances in Neural Information Processing Systems (NeurIPS), 35. [https://arxiv.org/abs/2205.11916](https://arxiv.org/abs/2205.11916)

[37] Chen, M., Tworek, J., Jun, H., Yuan, Q., et al. (2021). *Evaluating large language models trained on code*. arXiv preprint arXiv:2107.03374.

[38] Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., & Bi, X. (2024). *DeepSeekMath: Pushing the limits of mathematical reasoning in open language models*. arXiv preprint arXiv:2402.03300.

[39] Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., et al. (2025). *DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning*. Nature. [https://www.nature.com/articles/s41586-025-09422-z](https://www.nature.com/articles/s41586-025-09422-z)
