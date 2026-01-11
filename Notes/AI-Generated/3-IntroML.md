# Machine Learning Basics

**Course:** DOTE 6635: Artificial Intelligence for Business Research (Spring 2026)  

**Instructor:** Renyu (Philip) Zhang

---

## Abstract

This document serves as a comprehensive guide to the fundamental concepts of machine learning, tailored for business research applications. Moving beyond the "black box" perspective, we explore the theoretical underpinnings of supervised learning, the critical importance of model evaluation, and the mechanics of core algorithms such as K-Nearest Neighbors, Decision Trees, and Ensemble Methods. The goal is to provide a rigorous yet accessible framework for understanding how these tools can be applied to predict outcomes and infer relationships in complex datasets.

---

## 1. Introduction

In the era of big data, machine learning has emerged as a pivotal tool for extracting actionable insights from vast information repositories. For business researchers, the ability to not only apply these algorithms but also understand their internal mechanics is crucial. This lecture focuses on the supervised learning paradigm, where the objective is to learn a mapping from input features to target outputs.

We begin by establishing the mathematical framework of supervised learning and the dual goals of prediction and inference. We then delve into the Bias-Variance Trade-off, a central concept that governs the performance of all machine learning models. Finally, we examine specific algorithms—ranging from simple non-parametric methods to sophisticated ensemble techniques—and discuss their practical implementation and evaluation.

### Key Learning Objectives
*   **Understand the Bias-Variance Trade-off**: Grasp the tension between model simplicity and flexibility.
*   **Master Cross-Validation**: Learn robust methods for estimating model performance on unseen data.
*   **Apply Core Algorithms**: Gain proficiency in K-Nearest Neighbors, Decision Trees, and Ensemble Methods.

---

## 2. The Supervised Learning Framework

Supervised learning involves building a statistical model to predict or estimate an output based on one or more inputs. Let $X$ represent the vector of input features (predictors) and $Y$ represent the target variable (response). We assume there is a systematic relationship between $X$ and $Y$, which can be mathematically expressed as:

$$Y = f(X) + \epsilon$$

Here, the terms are defined as follows:
*   **$Y$ (Target Variable)**: The outcome we wish to predict (e.g., sales revenue, customer churn).
*   **$X$ (Feature Vector)**: The set of input variables or predictors (e.g., advertising spend, customer age).
*   **$f(X)$ (Systematic Function)**: The true underlying relationship between the inputs and the output. This represents the signal we aim to learn.
*   **$\epsilon$ (Irreducible Error)**: The random error term that captures noise, measurement errors, and the influence of unmeasured variables. It is assumed to have a mean of zero ($E[\epsilon] = 0$) and be independent of $X$.

### Prediction vs. Inference

The estimation of $f$, denoted as $\hat{f}$, serves two primary purposes in business research:

1.  **Prediction**: When the focus is on predicting the value of $Y$ for a new, unobserved input $X$. In this context, $\hat{f}$ is treated as a "black box," and the primary metric of success is the accuracy of the prediction $\hat{Y} = \hat{f}(X)$.
2.  **Inference**: When the goal is to understand the relationship between $X$ and $Y$. Researchers ask questions such as: Which predictors are most significant? Is the relationship linear or non-linear? In this case, the interpretability of $\hat{f}$ is paramount.

### Parametric vs. Non-Parametric Approaches

Methods for estimating $f$ generally fall into two categories:

*   **Parametric Methods**: These methods make an explicit assumption about the functional form of $f$ (e.g., assuming it is linear). This simplifies the problem to estimating a set of parameters. While easier to interpret and requiring less data, parametric methods suffer from high bias if the assumed form does not match the true underlying relationship.
    *   *Examples*: Linear Regression, Logistic Regression.
*   **Non-Parametric Methods**: These methods do not assume a specific functional form for $f$. Instead, they seek to fit the data as closely as possible. This flexibility allows them to model complex, non-linear relationships but requires a larger number of observations to avoid overfitting.
    *   *Examples*: K-Nearest Neighbors, Decision Trees.

---

## 3. Model Evaluation and the Bias-Variance Trade-off

Evaluating the performance of a machine learning model is critical. For regression problems, the standard metric is the **Mean Squared Error (MSE)**, which measures the average squared difference between the estimated values and the actual values:

$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{f}(x_i))^2$$

In this equation:
*   **$n$**: The total number of observations in the dataset.
*   **$y_i$**: The actual observed value of the target variable for the $i$-th observation.
*   **$\hat{f}(x_i)$**: The predicted value generated by the model for the $i$-th observation.
*   **$(y_i - \hat{f}(x_i))^2$**: The squared residual (error) for a single observation. Squaring ensures that positive and negative errors do not cancel each other out and penalizes larger errors more heavily.

While the training MSE (calculated on the data used to fit the model) is useful, the true test of a model is its performance on unseen data, known as the **Test MSE**.

### The Bias-Variance Decomposition

The expected test MSE for a given input point $x_0$ can be mathematically decomposed into three distinct components:

$$E[(y_0 - \hat{f}(x_0))^2] = \text{Var}(\hat{f}(x_0)) + [\text{Bias}(\hat{f}(x_0))]^2 + \text{Var}(\epsilon)$$

This equation breaks down the expected test error into three components:
1.  **$\text{Var}(\hat{f}(x_0))$ (Variance)**: Measures how much the model's prediction $\hat{f}(x_0)$ would change if we trained it on a different dataset. High variance implies the model is unstable and overfits the training data.
2.  **$[\text{Bias}(\hat{f}(x_0))]^2$ (Squared Bias)**: Measures the error introduced by approximating a complex real-world problem with a simplified model. High bias implies the model is too simple (underfitting) and misses key relationships.
3.  **$\text{Var}(\epsilon)$ (Irreducible Error)**: The inherent noise in the system that no model can eliminate.

This decomposition highlights the fundamental trade-off in machine learning:

*   **Variance**: Refers to the amount by which $\hat{f}$ would change if we estimated it using a different training data set. High variance indicates that the model is overfitting the training data, capturing noise rather than the underlying signal.
*   **Bias**: Refers to the error introduced by approximating a real-life problem (which may be extremely complicated) by a much simpler model. High bias indicates underfitting, where the model fails to capture important relations.
*   **Irreducible Error**: The variance of the error term $\epsilon$, which cannot be reduced by any model.

**Key Insight**: As model complexity increases, variance tends to increase while bias tends to decrease. The optimal model is found at the "sweet spot" where the sum of bias and variance is minimized.

### Cross-Validation

To estimate the test error without setting aside a large portion of data solely for testing (which would deprive the model of valuable training data), we employ **Cross-Validation**. The most common variant is **K-Fold Cross-Validation**.

In this approach, the dataset is randomly divided into $K$ equal-sized folds. The model is trained on $K-1$ folds and validated on the remaining fold. This process is repeated $K$ times, and the results are averaged:

$$CV_{(K)} = \frac{1}{K} \sum_{k=1}^{K} MSE_k$$

Where:
*   **$K$**: The number of folds (partitions) the data is split into.
*   **$MSE_k$**: The Mean Squared Error calculated on the $k$-th validation fold (the hold-out set) after training on the other $K-1$ folds.
*   **$CV_{(K)}$**: The final cross-validation estimate, which is simply the average of the $K$ individual MSE scores.

*   **Choice of K**: Typically, $K=5$ or $K=10$ is chosen. This balances the bias-variance trade-off in the error estimate itself. A very large $K$ (like Leave-One-Out Cross-Validation) has low bias but high variance and high computational cost.

#### Python Implementation: K-Fold CV

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

# Generate synthetic data
X, y = make_regression(n_samples=100, n_features=20, noise=0.1)

# Initialize model
model = LinearRegression()

# Perform 5-fold cross-validation
# Note: sklearn uses negative MSE for scoring, so we negate it to get positive MSE
scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
mse_scores = -scores

print(f"Mean MSE: {mse_scores.mean():.4f}")
```

---

## 4. K-Nearest Neighbors (k-NN)

The K-Nearest Neighbors (k-NN) algorithm is a simple yet powerful non-parametric method. It operates on the principle that similar data points likely have similar outcomes. Given a query point $x_0$ and a positive integer $K$, the algorithm identifies the $K$ points in the training data that are closest to $x_0$.

*   **For Classification**: The algorithm assigns $x_0$ to the class that is most common among its neighbors (a majority vote).
    $$P(Y=j | X=x_0) = \frac{1}{K} \sum_{i \in \mathcal{N}_0} I(y_i = j)$$

In this formula:
*   **$P(Y=j | X=x_0)$**: The estimated probability that the new point $x_0$ belongs to class $j$.
*   **$\mathcal{N}_0$**: The set of the $K$ nearest neighbors to $x_0$ in the training data.
*   **$I(y_i = j)$**: An indicator function that equals 1 if the neighbor $y_i$ belongs to class $j$, and 0 otherwise. Essentially, this counts the "votes" for class $j$.

*   **For Regression**: The algorithm assigns $x_0$ the average value of its neighbors.
    $$\hat{f}(x_0) = \frac{1}{K} \sum_{i \in \mathcal{N}_0} y_i$$

Here, $\hat{f}(x_0)$ is simply the arithmetic mean of the target values $y_i$ of the $K$ nearest neighbors.

### Implementation Considerations

*   **The Role of K**: The choice of $K$ controls the bias-variance trade-off. A small $K$ (e.g., $K=1$) leads to a highly flexible model with low bias but high variance (a jagged decision boundary). A large $K$ smooths the decision boundary, increasing bias but reducing variance.
*   **Feature Scaling**: Because k-NN relies on distance metrics (typically Euclidean), it is critical to standardize features so that variables with large scales do not dominate the distance calculation.
*   **Curse of Dimensionality**: k-NN suffers in high-dimensional spaces. As the number of features increases, data points become sparse, and the concept of "nearest" neighbor becomes less meaningful.

---

## 5. Decision Trees

Decision trees mimic human decision-making by learning a hierarchy of if/else questions. Structurally, a tree consists of a **root node** (containing all data), **internal nodes** (where splits occur based on features), and **leaf nodes** (where the final prediction is made).

### The CART Algorithm

The Classification and Regression Tree (CART) algorithm builds the tree by recursively splitting the data into regions $R_1, ..., R_J$ to minimize impurity.

*   **Gini Index (Classification)**: Measures the total variance across the $K$ classes. A small Gini index indicates a node that is predominantly one class (pure).
    $$G = \sum_{k=1}^{K} \hat{p}_{mk}(1 - \hat{p}_{mk})$$

Where:
*   **$\hat{p}_{mk}$**: The proportion of training observations in the $m$-th region (node) that belong to the $k$-th class.
*   **$G$**: The Gini Index. A value of 0 indicates perfect purity (all observations in the node belong to a single class).

*   **Entropy (Classification)**: An alternative measure of disorder or uncertainty.
    $$D = - \sum_{k=1}^{K} \hat{p}_{mk} \log \hat{p}_{mk}$$

Similar to Gini, **Entropy ($D$)** is minimized (approaches 0) when the node is pure. It is maximized when the classes are perfectly mixed (e.g., a 50/50 split in a binary case).

### Pruning to Prevent Overfitting

A major drawback of decision trees is their tendency to overfit by growing too deep and memorizing the training data. **Pruning** addresses this by reducing the size of the tree. **Cost Complexity Pruning** introduces a penalty term $\alpha$ to the error function, penalizing the number of terminal nodes $|T|$:

$$\sum_{m=1}^{|T|} \sum_{x_i \in R_m} (y_i - \hat{y}_{R_m})^2 + \alpha |T|$$

This cost function balances two competing objectives:
1.  **Model Fit**: The first term $\sum (y_i - \hat{y}_{R_m})^2$ measures the total squared error of the tree (how well it fits the training data).
2.  **Model Complexity**: The second term $\alpha |T|$ penalizes the size of the tree.
    *   **$|T|$**: The number of terminal nodes (leaves) in the tree.
    *   **$\alpha$ (Alpha)**: A tuning parameter that controls the strength of the penalty. If $\alpha = 0$, we get the full, unpruned tree. As $\alpha$ increases, we are forced to prune the tree to reduce $|T|$, leading to a simpler model.

---

## 6. Ensemble Methods

Ensemble methods represent a paradigm shift from finding a single "best" model to combining multiple "weak learners" to create a "strong learner." These methods often produce state-of-the-art results in predictive tasks.

### Bagging (Bootstrap Aggregating)

Bagging involves generating $B$ different bootstrapped training sets (sampling with replacement). A deep decision tree is trained on each set, and the predictions are aggregated (averaged for regression, majority vote for classification). By averaging multiple uncorrelated trees, Bagging significantly reduces **variance**.

### Random Forests

Random Forests improve upon Bagging by further decorrelating the trees. When building each tree, at each split, the algorithm is forced to consider only a **random subset of $m$ features** (typically $m \approx \sqrt{p}$). This prevents strong predictors from dominating every tree, ensuring diversity among the ensemble members and leading to better generalization.

### Boosting

Unlike Bagging and Random Forests, which build trees in parallel, Boosting grows trees **sequentially**. Each new tree is trained to correct the errors (residuals) of the previous tree.

1.  Fit a tree to the data.
2.  Calculate the residuals.
3.  Fit a new tree to the residuals.
4.  Update the model by adding a scaled version of the new tree.

$$\hat{f}(x) \leftarrow \hat{f}(x) + \lambda \hat{f}^b(x)$$

In this update step:
*   **$\hat{f}(x)$**: The current ensemble model before adding the new tree.
*   **$\hat{f}^b(x)$**: The new decision tree fitted to the residuals (errors) of the current model.
*   **$\lambda$ (Lambda)**: The **learning rate** (typically a small number like 0.01 or 0.1). It shrinks the contribution of the new tree, preventing the model from learning too quickly and overfitting. This "slow learning" approach is key to Boosting's success.

#### Python Example: Random Forest vs. Boosting

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate synthetic classification data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print(f"Random Forest Accuracy: {rf.score(X_test, y_test):.4f}")

# Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gb.fit(X_train, y_train)
print(f"Gradient Boosting Accuracy: {gb.score(X_test, y_test):.4f}")
```

---

## 7. Advanced Topic: Causal Forests

While standard Random Forests predict an outcome $Y$ given covariates $X$, **Causal Forests** are designed to estimate the **Heterogeneous Treatment Effect (HTE)**. The goal is to estimate $\tau(x) = E[Y^{(1)} - Y^{(0)} | X=x]$, representing the causal effect of a treatment (e.g., a marketing campaign) on an individual with characteristics $x$.

To understand this, we rely on the **Potential Outcomes Framework** (also known as the Rubin Causal Model). For any individual, there are two potential outcomes:
*   **$Y^{(1)}$ (Treatment Outcome)**: The outcome that *would* be observed if the individual received the treatment (e.g., the customer buys the product after seeing an ad).
*   **$Y^{(0)}$ (Control Outcome)**: The outcome that *would* be observed if the individual did *not* receive the treatment (e.g., the customer's behavior without seeing the ad).

The fundamental challenge of causal inference is that we can never observe both $Y^{(1)}$ and $Y^{(0)}$ for the same individual simultaneously—a problem known as the **Fundamental Problem of Causal Inference**. Causal Forests address this by estimating the *expected* difference between these potential outcomes for individuals with similar features $X$.

To ensure valid inference, Causal Forests employ "honesty": one half of the data is used to build the tree structure (deciding splits), and the other half is used to estimate the treatment effect within the leaves. This separation prevents overfitting and allows for accurate estimation of personalized causal effects.

---

## References

1.  James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An Introduction to Statistical Learning*. Springer. [Link](https://www.statlearning.com/)
2.  Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer. [Link](https://hastie.su.domains/ElemStatLearn/)
3.  Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32. [Link](https://link.springer.com/article/10.1023/A:1010933404324)
4.  Athey, S., & Imbens, G. (2016). Recursive partitioning for heterogeneous causal effects. *Proceedings of the National Academy of Sciences*, 113(27), 7353-7360. [Link](https://www.pnas.org/doi/10.1073/pnas.1510489113)
