# Introduction to Reinforcement Learning

**Course:** DOTE 6635: Artificial Intelligence for Business Research (Spring 2026)
**Instructor:** Renyu (Philip) Zhang

## Introduction: The New Era of Artificial Intelligence

The field of Artificial Intelligence (AI) is undergoing a significant transformation. As noted by pioneers like Richard Sutton and David Silver, we are entering an "Era of Experience" [1]. This new era is marked by a shift away from reliance on human-provided data and towards systems that learn and improve through direct interaction with their environment. This paradigm shift is encapsulated in the ancient proverb, "Knowledge from books is shallow; to truly understand, one must practice." Reinforcement Learning (RL) is at the heart of this evolution, providing the theoretical and algorithmic framework for creating agents that learn from experience.

Shunyu Yao has characterized the current moment as "AI's halftime" [2]. The first half of AI saw remarkable successes in solving well-defined problems with clear rules and objectives, such as chess and Go. These achievements were driven by innovations in search algorithms, deep learning, and large-scale computation. However, the second half of AI, which is beginning now, will be defined by a move from *solving* problems to *defining* them. In this new landscape, the ability of an agent to learn and adapt in complex, dynamic environments becomes paramount. This is where RL truly shines, as it provides a mechanism for agents to learn to make good sequences of decisions, even in the absence of complete information or a perfect model of the world.

### Why Reinforcement Learning?

The fundamental advantage of RL over other machine learning paradigms, such as Supervised Learning (SL), lies in its ability to generalize and discover novel strategies. As demonstrated by the performance of systems like AlphaGo Zero [3] and recent research on language models [4], an agent that learns through RL can surpass the performance of its teachers. While Supervised Fine-Tuning (SFT) is effective at memorizing patterns from a given dataset, RL enables an agent to generalize its knowledge to new, unseen situations. This is because RL is an active learning process: the agent is not a passive recipient of data but an active participant that explores its environment, makes decisions, and learns from the consequences of its actions.

This ability to learn through trial and error is what allows RL agents to achieve superhuman performance. For example, DeepSeek-R1, a large language model trained with RL, has demonstrated the ability to reason and solve complex mathematical problems, even discovering novel solutions and correcting its own mistakes along the way [5]. This self-evolving intelligence is a hallmark of RL and a key differentiator from other machine learning approaches.

## The Simplest Form of Reinforcement Learning: Multi-Armed Bandits

To understand the core principles of RL, we can start with the simplest possible setting: the Multi-Armed Bandit (MAB) problem. Imagine a gambler in a casino facing a row of slot machines, each with a different, unknown probability of paying out a reward. The gambler has a limited number of pulls and wants to maximize their total winnings. This is the essence of the MAB problem: how to allocate a limited set of resources among several competing choices to maximize the expected gain, when the properties of these choices are not known in advance.

In the MAB framework, each slot machine is an "arm," and the decision of which arm to pull at each time step is the central challenge. This problem encapsulates the fundamental trade-off in RL: the **exploitation-exploration dilemma**. 

*   **Exploitation** involves choosing the arm that currently appears to be the best, based on past experience. This is the strategy of maximizing the immediate, known reward.
*   **Exploration** involves trying out other arms to gather more information about their potential rewards. This may lead to a lower immediate reward but could result in discovering a better arm in the long run.

### Naive Approaches and the Need for a Strategy

A simple, yet flawed, approach is the **greedy algorithm**. This strategy involves always choosing the arm with the highest estimated reward based on the data collected so far. The problem with a purely greedy approach is that it can get stuck in a suboptimal strategy. If, due to random chance, a suboptimal arm initially appears to be the best, the greedy algorithm will continue to choose it, never exploring other arms that might be better. This leads to a situation where the agent is locked into a wrong decision forever.

To overcome this, we need a more sophisticated strategy that balances exploitation and exploration. The **epsilon-greedy** algorithm is a simple yet effective way to achieve this. With a small probability, epsilon (ε), the agent chooses an arm at random (exploration), and with probability 1-ε, it chooses the best-known arm (exploitation). The value of epsilon can be decreased over time, as the agent becomes more confident in its estimates of the rewards.

### Advanced Strategies: Optimism in the Face of Uncertainty

More advanced MAB algorithms are based on the principle of **Optimism in the Face of Uncertainty (OFU)**. The core idea is that the more uncertain we are about the reward of an arm, the higher the priority we should give to exploring it. Two popular algorithms that implement this principle are the **Upper Confidence Bound (UCB)** and **Thompson Sampling (TS)**.

*   **UCB** works by calculating an upper confidence bound for the reward of each arm. This bound is a combination of the current estimated reward and an uncertainty term that is larger for arms that have been tried less frequently. The algorithm then chooses the arm with the highest upper confidence bound.

*   **Thompson Sampling** is a Bayesian approach where we maintain a probability distribution for the reward of each arm. At each step, we sample a reward value from each arm's distribution and then choose the arm with the highest sampled value. This naturally balances exploration and exploitation, as arms with higher uncertainty will have a wider distribution, giving them a chance to be selected even if their current mean estimate is not the highest.

Both UCB and TS have been shown to be highly effective in practice and have strong theoretical guarantees on their performance, as measured by **regret** (the difference between the reward obtained by the algorithm and the reward that would have been obtained by an optimal strategy).

### MAB in Action: Real-World Applications

The MAB framework has found numerous applications in business and technology. It is widely used in online advertising to dynamically allocate ad impressions to different ad creatives to maximize click-through or conversion rates [6, 7]. Other applications include dynamic pricing [8], personalized recommendations [9], and A/B testing. These applications demonstrate the power of MABs to solve real-world decision-making problems under uncertainty.

## Sequential Decision Making: Markov Decision Processes

While MABs are a useful model for single-step decision problems, many real-world problems involve a sequence of decisions where the action taken at one step can influence the state of the world and the available actions at future steps. To model these more complex scenarios, we use the framework of **Markov Decision Processes (MDPs)**.

An MDP is a mathematical framework for modeling decision-making in situations where outcomes are partly random and partly under the control of a decision-maker. The core assumption of an MDP is the **Markov property**, which states that the future is independent of the past, given the present. In other words, the current state of the system contains all the information needed to make a decision, and the history of how the system arrived at that state is irrelevant.

### The Components of an MDP

An MDP is formally defined by a tuple (S, A, P, R, γ), where:

*   **S** is a set of states.
*   **A** is a set of actions.
*   **P** is the state transition probability function, P(s'|s, a), which gives the probability of transitioning to state s' from state s after taking action a.
*   **R** is the reward function, R(s, a, s'), which gives the reward received after transitioning from state s to state s' as a result of taking action a.
*   **γ** (gamma) is the discount factor, a number between 0 and 1 that represents the preference for immediate rewards over future rewards.

### From Markov Processes to MDPs

To understand MDPs, it's helpful to build up from simpler concepts:

1.  **Markov Process (or Markov Chain):** This is the simplest model, consisting of a set of states and a transition probability matrix that defines the probability of moving from one state to another. There are no actions or rewards in a Markov Process.

2.  **Markov Reward Process (MRP):** An MRP adds a reward function and a discount factor to a Markov Process. This allows us to calculate the **value** of being in a particular state, which is defined as the expected total discounted reward from that state onwards.

3.  **Markov Decision Process (MDP):** An MDP adds a set of actions to an MRP, giving the agent control over the transitions between states. The goal of the agent in an MDP is to learn a **policy** that maximizes the expected total discounted reward.

### Policies and Value Functions

A **policy**, denoted by π, is a mapping from states to actions. It specifies what action the agent should take in each state. The goal of RL is to find an optimal policy, π*, that maximizes the expected return.

To evaluate a policy, we use **value functions**. There are two main types of value functions:

*   **State-value function (v_π(s)):** The expected return starting from state s and following policy π.
*   **Action-value function (q_π(s, a)):** The expected return starting from state s, taking action a, and then following policy π.

These value functions can be calculated using the **Bellman equations**, which express the value of a state in terms of the values of its successor states. The Bellman equations form the basis for many RL algorithms.

### Finding the Optimal Policy

The ultimate goal in an MDP is to find the optimal policy, π*. The optimal policy is the one that achieves the highest possible expected return from every state. The corresponding value functions are called the optimal value functions, v*(s) and q*(s, a).

It can be shown that for any MDP, there exists at least one optimal policy. Once we have the optimal action-value function, q*(s, a), the optimal policy is simply to choose the action that maximizes q*(s, a) in each state.

The optimal value functions can be found by solving the **Bellman optimality equations**. These equations are similar to the Bellman equations for a given policy, but they include a maximization step over all possible actions.

## Solving MDPs with Dynamic Programming

When the full model of the MDP is known (i.e., we know the transition probabilities and reward function), we can use dynamic programming methods to find the optimal policy. The two main dynamic programming algorithms for solving MDPs are **Value Iteration** and **Policy Iteration**.

### Value Iteration

Value Iteration is an iterative algorithm that starts with an arbitrary value function and repeatedly updates it using the Bellman optimality equation. The algorithm converges to the optimal value function, from which the optimal policy can be easily extracted. The update rule for Value Iteration is as follows:

`v_{k+1}(s) = max_a Σ_{s'} P(s'|s, a) [R(s, a, s') + γv_k(s')]`

This process is repeated until the value function converges.

### Policy Iteration

Policy Iteration is another iterative algorithm that alternates between two steps:

1.  **Policy Evaluation:** Given a policy, calculate the corresponding value function. This is done by solving a system of linear equations or by iteratively applying the Bellman equation for the given policy.

2.  **Policy Improvement:** Improve the policy by acting greedily with respect to the current value function. For each state, we choose the action that maximizes the expected return based on the current value function.

This process of policy evaluation and improvement is repeated until the policy no longer changes, at which point we have found the optimal policy. The convergence of policy iteration is often visualized in a simple environment like a GridWorld, where the initial random policy quickly evolves into an optimal path-finding strategy within a few iterations.

### A Comparison of Dynamic Programming Methods

Both Value Iteration and Policy Iteration are guaranteed to converge to the optimal policy, but they do so in different ways. The choice between them involves a trade-off between the number of iterations and the computational cost per iteration.

| Algorithm | Pros | Cons |
| :--- | :--- | :--- |
| **Value Iteration** | Each iteration is computationally fast. | Convergence can be slow and is asymptotic, meaning it gets closer and closer to the optimum without necessarily reaching it in a finite number of steps. |
| **Policy Iteration** | Guaranteed to converge in a finite number of iterations, and often converges very quickly in practice. | The policy evaluation step in each iteration can be computationally expensive, as it involves solving a system of linear equations or running an iterative process until convergence. |

In practice, **Policy Iteration is often faster than Value Iteration** because it can converge in a much smaller number of iterations. However, it's important to note that if the discount factor, γ, is close to 1, both methods can become slow.

## An Alternative Perspective: Linear Programming

Beyond dynamic programming, MDPs can also be solved using **Linear Programming (LP)**. This approach provides a different mathematical perspective on the problem of finding the optimal value function. The core idea is to formulate the Bellman optimality equation as a set of constraints in an LP problem.

The primal form of the LP is formulated as follows:

**Minimize:** `Σ_s d_0(s)V(s)`
**Subject to:** `V(s) ≥ R(s, a) + γ Σ_{s'} P(s'|s, a)V(s')` for all `s ∈ S, a ∈ A`

Here, `V(s)` are the decision variables, representing the value of each state, and `d_0(s)` is an initial state distribution. The constraints ensure that the value function satisfies the Bellman optimality condition. The optimal solution to this LP is the optimal value function of the MDP.

This LP has `|S|` decision variables and `|S| * |A|` linear constraints. By taking the dual of this primal LP, we can obtain a formulation where the variables correspond to state-action visitation frequencies, and the optimal solution to the dual LP can be used to directly derive the optimal policy.

## Connections to Econometrics: Structural Estimation

The concepts and methods developed for solving MDPs have found significant applications in the field of econometrics, particularly in the **structural estimation of dynamic discrete choice models**. These models are used to analyze the behavior of individuals or firms making sequential decisions over time, such as a consumer's decision to purchase a product or a firm's decision to invest in a new technology.

Estimating these structural models is often computationally challenging. A traditional approach is the **nested fixed-point (NFXP)** algorithm, which involves repeatedly solving for the value function (the fixed point) within an optimization routine that searches for the model's parameters. This can be very slow.

In a seminal paper, Su and Judd (2012) proposed a more efficient approach based on **constrained optimization** [10]. Instead of nesting the fixed-point calculation, they formulate the estimation problem as a single large optimization problem where the Bellman equation is included as a set of constraints. This approach, often referred to as **Mathematical Programming with Equilibrium Constraints (MPEC)**, can be significantly faster than NFXP because it avoids the repeated solving of the structural model for every guess of the parameters.

This connection highlights the deep interplay between reinforcement learning, optimization, and econometrics. The tools and techniques developed in one field can often be leveraged to solve important problems in another.

## Conclusion

This lecture has provided an introduction to the fundamental concepts of Reinforcement Learning. We have seen how RL is driving a new era of AI, enabling agents to learn from experience and achieve superhuman performance. We have explored the core concepts of the exploitation-exploration trade-off, Markov Decision Processes, value functions, and policies. We have also discussed how to solve MDPs using dynamic programming methods when a model of the environment is available, and we have touched upon alternative solution methods like linear programming and the connections between RL and structural estimation in econometrics.

This is just the beginning of our journey into the fascinating world of RL. In future lectures, we will explore more advanced topics, including model-free RL methods that can be used when the model of the environment is unknown, as well as deep reinforcement learning, which combines the power of deep neural networks with RL to solve complex problems in high-dimensional state and action spaces.

## References

[1] Sutton, R. S., & Silver, D. (2024). *Welcome to the Era of Experience*. [Online]. Available: [https://www.youtube.com/watch?v=b_h3QO33-8A](https://www.youtube.com/watch?v=b_h3QO33-8A)
[2] Yao, S. (2024). *The Second Half of AI*. [Online]. Available: [https://xbench.org/](https://xbench.org/)
[3] Silver, D., Schrittwieser, J., Simonyan, K., Antonoglou, I., Huang, A., Guez, A., ... & Hassabis, D. (2017). Mastering the game of Go without human knowledge. *Nature*, 550(7676), 354-359.
[4] Gao, L., et al. (2025). *SFT Memorizes, RL Generalizes*. arXiv preprint arXiv:2501.17161.
[5] DeepSeek AI. (2025). DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning. *Nature*, 635(7981), 1-8.
[6] Schwartz, E. M., Bradlow, E. T., & Fader, P. S. (2017). Customer acquisition via display advertising using multi-armed bandit experiments. *Marketing Science*, 36(4), 500-522.
[7] Ye, Z., Zhang, D. J., Zhang, H., Zhang, R., Chen, X., & Xu, Z. (2022). Cold start to improve market thickness on online advertising platforms: Data-driven algorithms and field experiments. *Management Science*, 68(10), 7161-7183.
[8] Misra, K., Schwartz, E. M., & Abernethy, J. (2019). Dynamic online pricing with incomplete information using multiarmed bandit experiments. *Marketing Science*, 38(2), 226-252.
[9] Aramayo, N., Schiappacasse, M., & Goic, M. (2022). A multiarmed bandit approach for house ads recommendations. *Marketing Science*, 41(4), 679-700.
[10] Su, C. L., & Judd, K. L. (2012). Constrained optimization approaches to estimation of structural models. *Econometrica*, 80(5), 2213-2230.
