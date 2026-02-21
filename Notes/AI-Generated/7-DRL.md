# Deep Reinforcement Learning

**DOTE 6635: Artificial Intelligence for Business Research (Spring 2026)**

**Instructor: Renyu (Philip) Zhang**

## Abstract

This article provides a comprehensive introduction to deep reinforcement learning (Deep RL), the powerful combination of deep neural networks with reinforcement learning algorithms that has enabled agents to tackle problems with enormous state and action spaces. The content is based on the lecture slides from the course "DOTE 6635: Artificial Intelligence for Business Research" and is supplemented with additional explanations and references to foundational literature. We begin with a review of where we stand in the RL landscape, connecting model-based methods, model-free methods, and model-free control. We then introduce value function approximation as the bridge from tabular RL to scalable methods, covering both linear and nonlinear approximators. Finally, we explore Deep Q-Networks (DQN), the landmark algorithm that achieved human-level performance on Atari games, examining its key innovations—experience replay and fixed targets—along with extensions and applications to business research.

## 1. Where Are We?

Before proceeding to deep reinforcement learning, it is helpful to situate ourselves within the broader RL curriculum:

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

TD may diverge in **off-policy** (e.g., Q-learning) or **nonlinear approximation** settings. This is the challenge for off-policy control: the behavior policy and target policy are not identical, so the value function approximation may diverge.

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

The landmark paper by Mnih et al. (2015), "Human-level control through deep reinforcement learning," published in *Nature*, demonstrated that a single DQN agent could achieve **professional human-level performance** across many Atari 2600 games using the **same network architecture and hyperparameters** for all games. This was a groundbreaking result: the same algorithm, without any game-specific tuning, could learn to play dozens of different video games from raw pixel inputs.

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
  - $4 \times 16$ filters with stride 2 → $9 \times 9 \times 32$ feature maps
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

**Double DQN** (Van Hasselt, Guez, and Silver, AAAI 2016): Standard Q-learning is known to overestimate action values because it uses the same network to both select and evaluate actions ($\max_a Q(s, a)$ selects the best action and simultaneously uses that Q-value as the estimate). Double DQN decouples these two operations: the online network selects the best action, but the target network evaluates its value. This significantly reduces overestimation bias and improves performance.

**Prioritized Replay** (Schaul, Quan, Antonoglou, and Silver, ICLR 2016): Instead of sampling uniformly from the replay buffer, prioritized experience replay samples transitions with larger TD errors more frequently. The intuition is that transitions where the agent's prediction is most wrong are the most informative for learning. This approach stores the last encountered TD error along with each transition in the replay buffer and uses it to define sampling probabilities.

**Dueling DQN** (Wang, Schaul, Hessel et al., ICML 2016, Best Paper): The dueling architecture decomposes the Q-function into two streams: a **value stream** $V(s)$ that estimates how good it is to be in state $s$, and an **advantage stream** $A(s, a)$ that estimates the relative advantage of each action. The Q-value is then reconstructed as $Q(s, a) = V(s) + A(s, a)$. This decomposition allows the network to learn which states are valuable (or not) without having to learn the effect of each action in each state separately, leading to more efficient learning.

## 7. Application: Deep Q-Learning for Dynamic Coupon Targeting

A compelling business application of deep reinforcement learning is presented by Liu (2023) in *Marketing Science*, who applies batch deep reinforcement learning (BDRL) to dynamic coupon targeting in a livestream shopping context.

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

## 8. Conclusion

Deep reinforcement learning represents a powerful synthesis of deep learning's representational capacity with reinforcement learning's sequential decision-making framework. The progression from tabular RL to function approximation to deep RL follows a natural path of increasing scalability:

1. **Value function approximation** replaces the lookup table with a parameterized function, enabling generalization across states and scaling to large problems.
2. **Linear approximation** provides a tractable starting point with convergence guarantees, but requires manual feature engineering.
3. **Deep Q-Networks** leverage neural networks for automatic feature learning, with experience replay and fixed targets to stabilize training.

The key takeaways for business researchers are:
- **The deadly triad** (bootstrapping + function approximation + off-policy learning) is a fundamental source of instability. DQN addresses it through engineering innovations rather than theoretical fixes.
- **Experience replay** and **fixed targets** are general-purpose stabilization techniques applicable beyond DQN.
- Deep RL opens the door to applications with high-dimensional state spaces—such as dynamic pricing, personalized recommendations, and adaptive marketing—where tabular methods are infeasible.
- **Extensions** like Double DQN, Prioritized Replay, and Dueling DQN offer further improvements and continue to be active areas of research.

## References

[1] Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. [http://incompleteideas.net/book/the-book-2nd.html](http://incompleteideas.net/book/the-book-2nd.html)

[2] Silver, D. (2015). *Lectures on Reinforcement Learning*. University College London. [https://www.davidsilver.uk/teaching/](https://www.davidsilver.uk/teaching/)

[3] Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., et al. (2015). *Human-level control through deep reinforcement learning*. Nature, 518(7540), 529–533.

[4] Van Hasselt, H., Guez, A., & Silver, D. (2016). *Deep reinforcement learning with double Q-learning*. Proceedings of the AAAI Conference on Artificial Intelligence, 30(1).

[5] Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2016). *Prioritized experience replay*. Proceedings of the International Conference on Learning Representations (ICLR).

[6] Wang, Z., Schaul, T., Hessel, M., et al. (2016). *Dueling network architectures for deep reinforcement learning*. Proceedings of the International Conference on Machine Learning (ICML).

[7] Liu, X. (2023). *Dynamic coupon targeting using batch deep reinforcement learning: An application to livestream shopping*. Marketing Science, 42(4), 610–636.

[8] Tsitsiklis, J. N., & Van Roy, B. (1997). *An analysis of temporal-difference learning with function approximation*. IEEE Transactions on Automatic Control, 42(5), 674–690.

[9] Bertsekas, D. P., & Tsitsiklis, J. N. (1996). *Neuro-Dynamic Programming*. Athena Scientific.
