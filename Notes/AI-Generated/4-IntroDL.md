# An Introduction to Deep Learning for Business Research

**DOTE 6635: Artificial Intelligence for Business Research (Spring 2026)**

**Instructor: Renyu (Philip) Zhang**

## Abstract

This article provides a comprehensive overview of the foundational concepts of deep learning, tailored for doctoral students in business. The content is based on the lecture slides from the course "DOTE 6635: Artificial Intelligence for Business Research" and is supplemented with additional explanations, code examples, and references to seminal literature. The objective is to offer a clear and intuitive understanding of the mathematical underpinnings of deep learning models and their practical applications in research. We begin by revisiting the principles of supervised learning and model training, with a focus on gradient-based optimization. Subsequently, we delve into the architecture of deep neural networks, exploring their expressive power and the mechanisms behind their learning capabilities. Finally, we discuss the computational aspects of deep learning, including the hardware and resources required to train large-scale models.

## 1. The Framework of Supervised Learning and Optimization

Supervised learning constitutes a dominant paradigm in machine learning, wherein the primary objective is to learn a functional mapping from an input space to an output space. This is achieved by leveraging a dataset of labeled input-output pairs. Within the domain of deep learning, this mapping function is represented by a deep neural network. The process of discerning the optimal parameters for this function is termed "model training," an optimization problem aimed at minimizing a predefined loss function that quantifies the disparity between the model's predictions and the observed ground truth.

### 1.1. Gradient Descent: The Engine of Optimization

At the heart of training for most contemporary machine learning models lies the **gradient descent** algorithm. As a first-order iterative optimization method, its purpose is to identify a local minimum of a differentiable function. The core principle is intuitive: to minimize a function, one should take steps in the direction of the function's steepest descent. This direction is precisely the negative of the function's gradient. The parameter update rule for gradient descent is thus formulated as:

$$ \theta_{t+1} = \theta_t - \alpha \nabla L(\theta_t) $$

Here, $ \theta_t $ represents the vector of model parameters at iteration $ t $, $ \alpha $ is the learning rate—a critical hyperparameter that governs the magnitude of each step—and $ \nabla L(\theta_t) $ is the gradient of the loss function $ L $ with respect to the parameters $ \theta $ at the current iteration.

The convergence properties of gradient descent are contingent upon the characteristics of the loss function. For convex functions possessing a Lipschitz continuous gradient, the algorithm is guaranteed to converge to a global minimum. However, the rate of this convergence varies.

> **Theorem 1: Convergence of Gradient Descent**
> 
> Let $f$ be a convex and differentiable function with a Lipschitz continuous gradient, meaning there exists a constant $L > 0$ such that $ \|\nabla f(x) - \nabla f(y)\| \leq L \|x - y\| $ for all $x, y$. If the learning rate $ \alpha $ is chosen such that $ 0 < \alpha \leq 1/L $, then gradient descent converges.
> 
> 1.  For a general convex function, the convergence rate is sublinear, requiring $ O(1/\epsilon) $ iterations to achieve an accuracy of $ \epsilon $.
> 2.  If $f$ is also strongly convex, the convergence rate becomes linear (or geometric), requiring only $ O(\log(1/\epsilon)) $ iterations, which is substantially faster [1].

To provide a tangible example, consider the simple quadratic function $ f(x) = x^2 - 4x + 4 $. The gradient is $ f'(x) = 2x - 4 $. A basic implementation of gradient descent in Python would be:

```python
# A simple demonstration of gradient descent

# Define the function and its gradient
def f(x):
    return x**2 - 4*x + 4

def grad_f(x):
    return 2*x - 4

# Hyperparameters
learning_rate = 0.1
iterations = 50
current_x = 0.0 # Initial guess

print(f"Starting gradient descent at x = {current_x}")

for i in range(iterations):
    gradient = grad_f(current_x)
    current_x = current_x - learning_rate * gradient
    if (i+1) % 10 == 0:
        print(f"Iteration {i+1}: x = {current_x:.4f}, f(x) = {f(current_x):.4f}")

print(f"Minimum found at x = {current_x:.4f}")
```

### 1.2. Foundational Models as Illustrations

The principles of gradient-based optimization can be clearly observed in classical statistical models. In **Ordinary Least Squares (OLS)** regression, the objective is to minimize the sum of squared residuals, $ L(\beta) = \sum_{i=1}^{n} (y_i - x_i^T \beta)^2 $. While OLS benefits from a closed-form analytical solution ($ \hat{\beta} = (X^T X)^{-1} X^T y $), it serves as a useful pedagogical tool for demonstrating gradient descent. In contrast, **logistic regression**, used for binary classification, lacks a closed-form solution. Its parameters are estimated by minimizing the cross-entropy loss, a task for which iterative methods like gradient descent are indispensable.

### 1.3. Enhancements to Gradient-Based Optimization

The vanilla gradient descent algorithm, while foundational, is often too slow or unreliable for the high-dimensional, non-convex landscapes typical of deep learning. Consequently, several more sophisticated variants have been developed.

**Stochastic Gradient Descent (SGD)** addresses the computational bottleneck of large datasets by approximating the true gradient using only a small, random subset of the data (a "mini-batch") at each iteration. This introduces noise into the optimization process, which can help the algorithm escape shallow local minima, but it also results in a more erratic convergence path.

To smooth out this path and accelerate convergence, the **Momentum** method was introduced. It accumulates an exponentially decaying moving average of past gradients, analogous to a ball rolling down a hill that gathers momentum. This allows the optimizer to navigate ravines more effectively and dampen oscillations in directions of high curvature.

Perhaps the most ubiquitous optimizer in modern practice is **Adam (Adaptive Moment Estimation)** [2]. Adam synergistically combines the concept of momentum with adaptive learning rates. It maintains separate decaying averages of both the past gradients (the first moment, like momentum) and the past squared gradients (the second moment, which captures the variance). By using these estimates, Adam computes individualized, adaptive learning rates for each parameter, making it robust and often effective with minimal hyperparameter tuning.

## 2. The Architecture and Theory of Deep Neural Networks

Having established the optimization framework, we now turn to the models themselves: **Deep Neural Networks (DNNs)**. This section explores their architecture, the theoretical guarantees of their expressive power, and the practical considerations for their construction and training.

### 2.1. The Connectionist Paradigm

The genesis of neural networks lies in the **connectionist** movement, which sought to model intelligence by drawing inspiration from the parallel, distributed processing architecture of the human brain. This stands in contrast to the classical **symbolic AI** approach, which is predicated on rule-based manipulation of symbols. The 2024 Nobel Prize in Physics, awarded to John J. Hopfield and Geoffrey E. Hinton, celebrated their pioneering work on **Hopfield networks** and **Boltzmann machines**, which connected concepts from statistical mechanics to the learning dynamics of neural networks, laying a crucial theoretical foundation for the field [3, 4].

### 2.2. The Power of Representation: Universal Approximation

A cornerstone of neural network theory is the **Universal Approximation Theorem**. This result provides a powerful guarantee about the expressive capacity of even simple network architectures.

> **Theorem 2: The Universal Approximation Theorem (Cybenko, 1989; Hornik et al., 1989)**
> 
> Let $ \sigma(\cdot) $ be a non-constant, bounded, and monotonically-increasing continuous function (a "squashing" function, like the sigmoid). Let $ I_m $ denote the $m$-dimensional unit hypercube $ [0, 1]^m $. The space of continuous functions on $ I_m $ is denoted by $ C(I_m) $. Given any function $ f \in C(I_m) $ and any $ \epsilon > 0 $, there exists an integer $ N $ and real constants $ v_i, b_i \in \mathbb{R} $ and vectors $ w_i \in \mathbb{R}^m $ for $ i = 1, \dots, N $, such that the function $ F(x) $ defined as:
> 
> $$ F(x) = \sum_{i=1}^{N} v_i \sigma(w_i^T x + b_i) $$
> 
> provides an approximation of $f$ with arbitrary accuracy, i.e., $ |F(x) - f(x)| < \epsilon $ for all $ x \in I_m $ [5, 6].

In essence, the theorem states that a single hidden layer in a neural network is sufficient to approximate any continuous function to any desired degree of precision. However, it is an existence theorem; it does not prescribe how to find the network's parameters (weights and biases), nor does it suggest that a single-layer network is the most efficient or learnable representation for a given problem. The "deep" aspect of deep learning—the use of multiple hidden layers—is motivated by the empirical finding that hierarchical representations can learn complex features more efficiently and with fewer parameters than their shallow counterparts.

### 2.3. Are Simple Multilayer Perceptrons Outdated? Applications in Business Research

Given the rapid advancements in deep learning architectures—from convolutional neural networks (CNNs) to transformers and large language models (LLMs)—one might question whether the simple **Multilayer Perceptron (MLP)**, the most basic form of a feedforward neural network, has become obsolete. The answer, perhaps surprisingly, is a resounding "no." The key to leveraging MLPs effectively lies not in their architectural complexity, but in identifying **interesting and impactful applications** where their simplicity is a virtue.

For business researchers, the MLP remains a powerful and highly relevant tool. Its relative simplicity offers advantages in terms of interpretability, computational efficiency, and theoretical tractability—qualities that are often paramount in academic research where understanding *why* a model works is as important as its predictive accuracy. Several recent publications in top-tier management and operations research journals demonstrate the continued vitality of MLP-based approaches.

**Example 1: AI for Scientific Discovery**

A landmark paper published in *Nature* by Davies et al. (2021), titled "Advancing mathematics by guiding human intuition with AI," demonstrated how machine learning, including neural network models, can assist mathematicians in discovering new patterns and formulating conjectures [12]. This work, a collaboration between DeepMind and leading mathematicians, showcases how AI can augment human intuition in pure mathematics—a domain far removed from the typical "big data" applications of deep learning. The success of this project underscores that the value of neural networks often lies in their application to novel, high-impact problems rather than in the sheer scale of the model.

**Example 2: Causal Inference in Large-Scale Experiments**

In the domain of operations and marketing, a paper by Ye, Zhang, Zhang, Zhang, and Zhang (2024) published in *Management Science*, titled "Deep Learning-Based Causal Inference for Large-Scale Combinatorial Experiments," addresses a critical challenge faced by online platforms [13]. These platforms run thousands of A/B tests daily, but the sheer number of treatment combinations makes it infeasible to test every possibility. The authors develop a novel framework called "Debiased Deep Learning" (DeDL) that combines deep learning with doubly robust estimation to infer the causal effect of any treatment combination, even those that were never directly observed. This work highlights how neural networks can be integrated with established econometric techniques to solve practical business problems at scale.

**Example 3: Deep Learning in Asset Pricing**

In finance, Chen, Pelger, and Zhu (2023) published "Deep Learning in Asset Pricing" in *Management Science* [14]. They use deep neural networks to estimate an asset pricing model for individual stock returns. The model takes advantage of a vast amount of conditioning information, maintains a flexible functional form, and accounts for time variation. Their approach outperforms traditional benchmark models in terms of Sharpe ratio, explained variation, and pricing errors, demonstrating the power of neural networks to capture complex, non-linear relationships in financial data.

**Example 4: Neural Networks for Choice Modeling**

Another compelling example comes from the field of operations research. Wang, Gao, and Li have developed a "Neural-Network Mixed Logit Choice Model" that provides statistical and optimality guarantees [15]. The mixed logit model is a workhorse in marketing and econometrics for modeling consumer choice. The authors show that a single-hidden-layer neural network can effectively approximate the mixture distribution in the mixed logit model. Crucially, they provide theoretical guarantees that the approximation error does not suffer from the curse of dimensionality, and that stochastic gradient descent can find the global optimum of the regularized problem. This work is a prime example of how rigorous theoretical analysis can validate the use of simple neural network architectures in established econometric frameworks.

These examples collectively illustrate a vital point: the frontier of impactful research is not solely defined by the complexity of the model, but by the novelty and significance of the problem being addressed. For business school researchers, this means that mastering the fundamentals of MLPs and understanding their theoretical properties can open doors to a wide range of high-impact research opportunities.

### 2.4. The Mechanics of Learning: Backpropagation and Regularization

The training of a DNN is accomplished via the **backpropagation** algorithm, which is a highly efficient method for computing the gradients of the loss function with respect to all network parameters. It works by first performing a **forward pass**, where the input signal propagates through the network to produce a prediction, followed by a **backward pass**, where the error signal is propagated backward from the output layer. At each layer, the chain rule of calculus is applied to compute the local gradients, which are then used by an optimizer like Adam to update the weights.

Given their immense number of parameters, DNNs are highly susceptible to **overfitting**, a phenomenon where the model memorizes the training data, including its idiosyncrasies and noise, at the expense of its ability to generalize to unseen data. This is a manifestation of the classic **bias-variance tradeoff** [7]. To mitigate this, **regularization** techniques are essential.

Standard methods include **L1 (Lasso)** and **L2 (Ridge)** regularization, which add a penalty to the loss function based on the magnitude of the network's weights [8]. A more modern and highly effective technique specific to neural networks is **Dropout** [9]. During training, Dropout randomly sets the activations of a fraction of neurons to zero at each forward pass. This prevents complex co-adaptations between neurons and forces the network to learn more robust and redundant representations, akin to training an ensemble of many smaller networks.

Implementing dropout in a modern framework like Keras is straightforward:

```python
# Example of a simple Keras model with Dropout and L2 regularization

# This is a conceptual example and requires a full environment to run.
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.keras.regularizers import l2

# model = Sequential([
#     Dense(128, activation='relu', input_shape=(784,), kernel_regularizer=l2(0.001)),
#     Dropout(0.5), # Apply dropout with a 50% rate
#     Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
#     Dropout(0.5),
#     Dense(10, activation='softmax')
# ])

# model.summary()
```

## 3. The Computational Landscape of Deep Learning

The practical realization of deep learning is as much a story of computational engineering as it is of theoretical advances. The training of state-of-the-art models is a computationally demanding process that relies on a specialized ecosystem of software and hardware. This section provides a comprehensive overview of the tools, infrastructure, and economic considerations that underpin modern deep learning research.

### 3.1. Deep Learning Software Libraries

The good news for researchers entering the field is that you will likely never need to implement backpropagation from scratch. A rich ecosystem of software libraries has emerged to handle the mathematical and computational complexities of deep learning, allowing researchers to focus on model design and application.

At the foundation are **general parallel computing libraries**, primarily **TensorFlow** (developed by Google) and **PyTorch** (developed by Meta/Facebook). These libraries are conceptually similar to NumPy, the ubiquitous Python library for numerical computing, but with two critical advantages: they provide much better support for parallel computing on GPUs, and they automatically compute and store derivatives—a feature known as **automatic differentiation**. This latter capability is the core function of any deep learning library: to take and store the derivatives of complex computational graphs in an automated and efficient fashion, which is essential for the backpropagation algorithm.

Built on top of these foundational libraries are higher-level **deep learning frameworks** such as **Keras** (which provides a user-friendly API for TensorFlow) and **Hugging Face** (which offers a vast repository of pre-trained models and tools for natural language processing). These frameworks allow researchers to define, train, and deploy complex models with just a few lines of code.

The following table illustrates the correspondence between common NumPy operations and their PyTorch equivalents, highlighting the ease of transition for researchers already familiar with numerical computing in Python:

| Operation | NumPy | PyTorch |
|---|---|---|
| Array/Tensor Creation | `numpy.array()` | `torch.tensor()` |
| Dimensions | `array.ndim` | `tensor.dim()` |
| Shape | `array.shape` | `tensor.size()` |
| Sum over all elements | `numpy.sum(array)` | `torch.sum(tensor)` |
| Mean | `numpy.mean(array)` | `torch.mean(tensor)` |
| Standard Deviation | `numpy.std(array)` | `torch.std(tensor)` |
| Element-wise Sum | `array1 + array2` | `tensor1 + tensor2` |
| Element-wise Product | `array1 * array2` | `tensor1 * tensor2` |
| Matrix Multiplication | `numpy.dot(a, b)` | `torch.matmul(a, b)` |
| Reshape | `array.reshape()` | `tensor.view()` |
| Transpose | `array.T` | `tensor.t()` |

A simple example of defining and training a neural network in PyTorch demonstrates the elegance of these modern tools:

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple 2-layer neural network
class SimpleNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Instantiate model, loss function, and optimizer
model = SimpleNet(input_dim=10, hidden_dim=64, output_dim=1)
criterion = nn.MSELoss()  # Mean Squared Error for regression
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop (conceptual)
# for epoch in range(num_epochs):
#     for inputs, targets in dataloader:
#         optimizer.zero_grad()       # Clear previous gradients
#         outputs = model(inputs)     # Forward pass
#         loss = criterion(outputs, targets)
#         loss.backward()             # Backward pass (computes gradients)
#         optimizer.step()            # Update parameters
```

### 3.2. Computing Hardware: GPUs and Beyond

The computational demands of deep learning necessitate specialized hardware. While **Central Processing Units (CPUs)** are general-purpose processors optimized for sequential tasks, **Graphics Processing Units (GPUs)** feature a massively parallel architecture with thousands of smaller cores designed for simultaneous operations. This makes GPUs exceptionally well-suited for the matrix multiplications that dominate neural network computations.

Researchers have two primary options for accessing GPU compute:

**Self-Made Workstations and Servers:** Building a custom workstation with a high-end consumer GPU, such as the NVIDIA RTX 4090 (approximately US$1,500, though often difficult to procure), is a cost-effective option for individual researchers. For larger-scale needs, institutions may invest in dedicated High-Performance Computing (HPC) clusters. For example, the CUHK Business School AI Lab has deployed two HPC clusters equipped with 16 NVIDIA 4090s, 4 NVIDIA 3090s, and 24 NVIDIA H100s.

**Cloud Computing:** Major cloud providers offer on-demand access to powerful GPU instances, providing scalability and flexibility without the need for upfront capital investment. Pricing for high-end instances is substantial:

| Provider | Instance Type | Approximate Cost |
|---|---|---|
| Google Cloud Platform (GCP) | 8x NVIDIA H100 | ~US$38.84 per hour |
| Amazon Web Services (AWS) | 8x NVIDIA H100 | ~US$39.33 per hour |

The performance difference between GPU generations is significant. Benchmarks on the ResNet-50 model (a standard computer vision architecture) show that the NVIDIA H100 outperforms the A100, which in turn significantly outperforms the consumer-grade RTX 4090. Crucially, performance scales near-linearly when using multiple GPUs in parallel (e.g., 4 or 8 GPUs), enabling the training of very large models.

| Configuration | NVIDIA A100 80GB | NVIDIA H100 | NVIDIA RTX 4090 |
|---|---|---|---|
| 1 GPU | 2,535 points | 3,042 points | 1,720 points |
| 4 GPUs | 9,366 points | 11,989 points | 5,934 points |
| 8 GPUs | 21,479 points | 30,070 points | N/A |

*Table: GPU benchmark scores on ResNet-50 (FP16). Higher is better.*

### 3.3. Model Size, Training Time, and Computational Costs

The cost of training a deep learning model is directly proportional to two factors: the **size of the network** (i.e., the number of parameters) and the **size of the training data**. Understanding the relationship between model scale and training time is crucial for planning research projects.

The following examples provide concrete benchmarks for training well-known models on a DGX server (a high-end NVIDIA system with 8 A100 GPUs):

**ResNet-50 (12 million parameters):** This classic computer vision model, trained on the ImageNet dataset using TF32 precision, takes approximately **30 minutes** on a DGX server. On a single consumer-grade RTX 4090, the same task would take roughly 6-10 hours.

**BERT (110 million parameters):** This foundational Natural Language Processing (NLP) model, pre-trained on 170GB of text from BooksCorpus and Wikipedia, takes approximately **5 hours** on a DGX server. Fine-tuning BERT on a smaller, task-specific dataset like the Stanford Question Answering Dataset (SQuAD) takes only 3-5 minutes.

**GPT-3 (175 billion parameters):** Training this landmark large language model required an estimated $3.15 \times 10^{23}$ FLOPs (floating-point operations), trained on 300 billion tokens. To understand the scale, consider the calculation:

$$ \text{Total FLOPs} \approx \text{Parameters} \times \text{Tokens} \times 6 $$

where the factor of 6 represents the approximate compute per parameter per token in transformer architectures. Using 128 DGX servers (each with 8 A100 GPUs running at 80 TFLOP/s), the estimated training time is:

$$ \text{Training Time} = \frac{3.15 \times 10^{23}}{128 \times 8 \times 80 \times 10^{12} \times 24 \times 3600} \approx 51 \text{ to } 100 \text{ days} $$

This translates to roughly **3 months** of continuous training.

### 3.4. Scaling Laws and the Economics of Large Models

Recent research has focused on understanding the **scaling laws** of neural networks, which describe the empirical relationship between model performance, model size, dataset size, and the amount of compute used for training [10]. These laws, pioneered by researchers at OpenAI, suggest that performance (measured by loss) improves predictably as a power law of these factors:

$$ L(N, D, C) \propto N^{-\alpha_N} + D^{-\alpha_D} + C^{-\alpha_C} $$

where $L$ is the loss, $N$ is the number of parameters, $D$ is the dataset size, $C$ is the compute budget, and $\alpha$ values are empirically determined exponents. This finding has profound implications: it suggests that, within certain regimes, simply scaling up models, data, and compute will yield predictable performance gains.

The economics of training frontier models are staggering. The table below compares the training costs of two recent large language models:

| Model | Parameters | Training Cost (USD) |
|---|---|---|
| DeepSeek-V3 | 671 Billion (MoE) | ~$5.576 Million |
| Llama 3.1 | 405 Billion | ~$160-200 Million |

The dramatic cost difference highlights the importance of **architectural innovation**. DeepSeek-V3 employs a **Mixture-of-Experts (MoE)** architecture, which activates only a sparse subset of its 671 billion parameters (approximately 37 billion) for each input token [11]. This allows it to achieve the performance of a much larger dense model while significantly reducing training and inference costs. DeepSeek-V3 was trained on 14.8 trillion tokens, requiring a total of approximately $3.3 \times 10^{24}$ FLOPs and 2,664,000 H800 GPU hours (approximately $5.328 million at $2/GPU-hour).

### 3.5. Geopolitical Considerations: GPU Export Controls

The strategic importance of advanced AI has led to significant geopolitical considerations surrounding the hardware that powers it. Beginning in October 2022, the United States government implemented export controls restricting the sale of high-performance GPUs (such as the NVIDIA A100 and H100) to certain countries, most notably China. These controls were further tightened in January 2025, expanding the range of restricted chips.

These restrictions have spurred the development of alternative chips (like the NVIDIA H800, a variant of the H100 designed to comply with export rules) and have accelerated domestic chip development efforts in affected countries. For business researchers, understanding this geopolitical landscape is important, as it affects the global distribution of AI research capabilities and the competitive dynamics of the technology industry.

## 4. Conclusion

This article has traversed the foundational landscape of deep learning, from the mathematical principles of gradient-based optimization to the architectural and theoretical underpinnings of deep neural networks, and finally to the computational realities of their implementation. For the business researcher, a solid grasp of these concepts is indispensable. Understanding the mechanics of gradient descent and its variants provides the intuition for how models learn. Knowledge of network architecture, the Universal Approximation Theorem, and regularization techniques informs how models are designed and controlled. Finally, an appreciation for the computational costs and scaling laws provides a pragmatic perspective on the feasibility and scope of applying these powerful tools to substantive research questions.

## 5. Compute-Efficient Training with GPUs

While Section 3 highlights the macro-level cost structure of deep learning, effective research practice also depends on **micro-level training efficiency**. In modern workflows, the difference between a feasible experiment and an infeasible one often comes down to GPU-aware implementation details. The following principles, distilled from recent lecture materials and practitioner guidance [16–19], summarize how to train models efficiently without compromising rigor.

### 5.1. Hardware-Aware Implementation and Parallelism

Deep learning models should be implemented in a framework that is optimized for GPU execution (e.g., PyTorch). This is not merely a convenience; it is foundational to performance. The most important first step is to **keep computation on GPU** and avoid unnecessary data transfers between CPU and GPU. When multiple GPUs are available, the default strategy for scaling training is **data parallelism**, typically implemented via **Distributed Data Parallel (DDP)**. In DDP, each GPU processes a different mini-batch, gradients are synchronized across devices, and model parameters are updated collectively.

An additional practical rule is to **choose "nice" tensor dimensions**, especially batch sizes and sequence lengths. GPUs operate on **CUDA kernels** that are optimized for **power-of-two block sizes**. As a result, choosing batch sizes and sequence lengths that are multiples of $2^k$ (e.g., 64, 128, 256) often yields noticeably better throughput.

### 5.2. Batch Size, Gradient Accumulation, and Learning-Rate Scaling

For GPU utilization, **larger batch sizes are typically more efficient**, but GPU memory imposes a hard ceiling. When memory constraints prevent the desired batch size, **gradient accumulation** is a principled workaround. The idea is to split a large batch into several smaller mini-batches, accumulate gradients across them, and update parameters only after the full effective batch has been processed.

Example: If the GPU can only fit a mini-batch of 8, but the target batch size is 32, then set:

$$ \text{mini-batch size} = 8, \quad \text{accumulation steps} = 4. $$

The optimizer updates parameters once every 4 mini-batches, achieving the same gradient effect as a batch of 32. A key corollary is **learning-rate scaling**: when the batch size is multiplied by $k$, it is common (and often empirically effective) to multiply the learning rate by $k$ as well.

### 5.3. Precision Reduction and Efficient Attention

Modern GPU architectures support reduced-precision arithmetic (e.g., **BF16** and **FP8**). For many deep learning models, training in reduced precision yields minimal accuracy loss while substantially improving throughput and lowering memory usage. These gains can enable larger batch sizes or larger models within the same hardware budget.

Another major performance bottleneck in large language models is attention computation. Recent advances, such as **FlashAttention**, reformulate attention to reduce memory overhead and improve GPU utilization, making it a de facto standard in high-performance transformer implementations.

### 5.4. Optimization Hygiene: Schedules and Initialization

Compute efficiency also depends on *convergence efficiency*. A poorly initialized model or an unstable learning rate schedule can waste GPU cycles. Empirically, the following practices are widely used:

- **He initialization** for ReLU-based networks to stabilize the variance of activations.
- **Learning-rate schedules** (e.g., cosine decay, warmup) to accelerate early learning and avoid late-stage oscillations.
- **AdamW** as the default optimizer for transformer-style architectures, due to its stable convergence and decoupled weight decay.

Taken together, these practices highlight a central lesson: **efficient training is a system-level problem**, not a single algorithmic trick. Researchers who design experiments with hardware constraints in mind can iterate faster, explore larger design spaces, and produce more reproducible computational results.

## 5. Compute-Efficient Training with GPUs

While Section 3 highlights the macro-level cost structure of deep learning, effective research practice also depends on **micro-level training efficiency**. In modern workflows, the difference between a feasible experiment and an infeasible one often comes down to GPU-aware implementation details. The following principles, distilled from recent lecture materials and practitioner guidance [16–19], summarize how to train models efficiently without compromising rigor.

### 5.1. Hardware-Aware Implementation and Parallelism

Deep learning models should be implemented in a framework that is optimized for GPU execution (e.g., PyTorch). This is not merely a convenience; it is foundational to performance. The most important first step is to **keep computation on GPU** and avoid unnecessary data transfers between CPU and GPU. When multiple GPUs are available, the default strategy for scaling training is **data parallelism**, typically implemented via **Distributed Data Parallel (DDP)**. In DDP, each GPU processes a different mini-batch, gradients are synchronized across devices, and model parameters are updated collectively.

An additional practical rule is to **choose "nice" tensor dimensions**, especially batch sizes and sequence lengths. GPUs operate on **CUDA kernels** that are optimized for **power-of-two block sizes**. As a result, choosing batch sizes and sequence lengths that are multiples of $2^k$ (e.g., 64, 128, 256) often yields noticeably better throughput.

### 5.2. Batch Size, Gradient Accumulation, and Learning-Rate Scaling

For GPU utilization, **larger batch sizes are typically more efficient**, but GPU memory imposes a hard ceiling. When memory constraints prevent the desired batch size, **gradient accumulation** is a principled workaround. The idea is to split a large batch into several smaller mini-batches, accumulate gradients across them, and update parameters only after the full effective batch has been processed.

Example: If the GPU can only fit a mini-batch of 8, but the target batch size is 32, then set:

$$ \text{mini-batch size} = 8, \quad \text{accumulation steps} = 4. $$

The optimizer updates parameters once every 4 mini-batches, achieving the same gradient effect as a batch of 32. A key corollary is **learning-rate scaling**: when the batch size is multiplied by $k$, it is common (and often empirically effective) to multiply the learning rate by $k$ as well.

### 5.3. Precision Reduction and Efficient Attention

Modern GPU architectures support reduced-precision arithmetic (e.g., **BF16** and **FP8**). For many deep learning models, training in reduced precision yields minimal accuracy loss while substantially improving throughput and lowering memory usage. These gains can enable larger batch sizes or larger models within the same hardware budget.

Another major performance bottleneck in large language models is attention computation. Recent advances, such as **FlashAttention**, reformulate attention to reduce memory overhead and improve GPU utilization, making it a de facto standard in high-performance transformer implementations.

### 5.4. Optimization Hygiene: Schedules and Initialization

Compute efficiency also depends on *convergence efficiency*. A poorly initialized model or an unstable learning rate schedule can waste GPU cycles. Empirically, the following practices are widely used:

- **He initialization** for ReLU-based networks to stabilize the variance of activations.
- **Learning-rate schedules** (e.g., cosine decay, warmup) to accelerate early learning and avoid late-stage oscillations.
- **AdamW** as the default optimizer for transformer-style architectures, due to its stable convergence and decoupled weight decay.

Taken together, these practices highlight a central lesson: **efficient training is a system-level problem**, not a single algorithmic trick. Researchers who design experiments with hardware constraints in mind can iterate faster, explore larger design spaces, and produce more reproducible computational results.

## References

[1] Tibshirani, R. (2016). *Gradient Descent*. Convex Optimization, Fall 2016, Carnegie Mellon University. [https://www.stat.cmu.edu/~ryantibs/convexopt-F16/lectures/grad-descent.pdf](https://www.stat.cmu.edu/~ryantibs/convexopt-F16/lectures/grad-descent.pdf)

[2] Kingma, D. P., & Ba, J. (2014). *Adam: A Method for Stochastic Optimization*. arXiv preprint arXiv:1412.6980. [https://arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)

[3] The Royal Swedish Academy of Sciences. (2024). *Press release: The Nobel Prize in Physics 2024*. [https://www.nobelprize.org/prizes/physics/2024/press-release/](https://www.nobelprize.org/prizes/physics/2024/press-release/)

[4] Wang, J. Z., et al. (2024). *Hopfield and Hinton’s neural network revolution and the future of AI*. The Innovation, 5(6). [https://www.sciencedirect.com/science/article/pii/S2666389924002666](https://www.sciencedirect.com/science/article/pii/S2666389924002666)

[5] Cybenko, G. (1989). *Approximation by superpositions of a sigmoidal function*. Mathematics of Control, Signals and Systems, 2(4), 303-314.

[6] Hornik, K., Stinchcombe, M., & White, H. (1989). *Multilayer feedforward networks are universal approximators*. Neural Networks, 2(5), 359-366. [https://www.sciencedirect.com/science/article/pii/0893608089900208](https://www.sciencedirect.com/science/article/pii/0893608089900208)

[7] Belkin, M., Hsu, D., Ma, S., & Mandal, S. (2019). *Reconciling modern machine-learning practice and the classical bias–variance trade-off*. Proceedings of the National Academy of Sciences, 116(32), 15849-15854. [https://www.pnas.org/doi/abs/10.1073/pnas.1903070116](https://www.pnas.org/doi/abs/10.1073/pnas.1903070116)

[8] Tibshirani, R. (1996). *Regression Shrinkage and Selection via the Lasso*. Journal of the Royal Statistical Society: Series B (Methodological), 58(1), 267-288. [https://webdoc.agsci.colostate.edu/koontz/arec-econ535/papers/Tibshirani%20(JRSS-B%201996).pdf](https://webdoc.agsci.colostate.edu/koontz/arec-econ535/papers/Tibshirani%20(JRSS-B%201996).pdf)

[9] Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). *Dropout: a simple way to prevent neural networks from overfitting*. The journal of machine learning research, 15(1), 1929-1958. [http://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf](http://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)

[10] Kaplan, J., et al. (2020). *Scaling Laws for Neural Language Models*. arXiv preprint arXiv:2001.08361. [https://arxiv.org/abs/2001.08361](https://arxiv.org/abs/2001.08361)

[11] DeepSeek AI. (2024). *DeepSeek-V3 Technical Report*. arXiv preprint arXiv:2412.19437. [https://arxiv.org/html/2412.19437v1](https://arxiv.org/html/2412.19437v1)

[12] Davies, A., Veličković, P., Buesing, L., Blackwell, S., Zheng, D., Tomašev, N., ... & Kohli, P. (2021). *Advancing mathematics by guiding human intuition with AI*. Nature, 600(7887), 70-74. [https://www.nature.com/articles/s41586-021-04086-x](https://www.nature.com/articles/s41586-021-04086-x)

[13] Ye, Z., Zhang, Z., Zhang, D. J., Zhang, H., & Zhang, R. (2024). *Deep Learning-Based Causal Inference for Large-Scale Combinatorial Experiments: Theory and Empirical Evidence*. Management Science. [https://doi.org/10.1287/mnsc.2024.04625](https://doi.org/10.1287/mnsc.2024.04625)

[14] Chen, L., Pelger, M., & Zhu, J. (2023). *Deep Learning in Asset Pricing*. Management Science, 70(2). [https://doi.org/10.1287/mnsc.2023.4695](https://doi.org/10.1287/mnsc.2023.4695)

[15] Wang, Z., Gao, R., & Li, S. (Working Paper). *Neural-Network Mixed Logit Choice Model: Statistical and Optimality Guarantees*.
[16] Karpathy, A. (2024). *Compute Efficient Training with GPUs* (lecture). [https://www.youtube.com/watch?v=l8pRSuU81PU](https://www.youtube.com/watch?v=l8pRSuU81PU)
[17] Stanford CS336. (2025). *Language Modeling from Scratch*. [https://stanford-cs336.github.io/spring2025/](https://stanford-cs336.github.io/spring2025/)
[18] Dao, T., et al. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. arXiv preprint arXiv:2205.14135. [https://arxiv.org/pdf/2205.14135](https://arxiv.org/pdf/2205.14135)
[19] MIT 6.5940. (2024). *Efficient Deep Learning Computing*. [https://efficientml.ai](https://efficientml.ai)
[16] Karpathy, A. (2024). *Compute Efficient Training with GPUs* (lecture). [https://www.youtube.com/watch?v=l8pRSuU81PU](https://www.youtube.com/watch?v=l8pRSuU81PU)
[17] Stanford CS336. (2025). *Language Modeling from Scratch*. [https://stanford-cs336.github.io/spring2025/](https://stanford-cs336.github.io/spring2025/)
[18] Dao, T., et al. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. arXiv preprint arXiv:2205.14135. [https://arxiv.org/pdf/2205.14135](https://arxiv.org/pdf/2205.14135)
[19] MIT 6.5940. (2024). *Efficient Deep Learning Computing*. [https://efficientml.ai](https://efficientml.ai)