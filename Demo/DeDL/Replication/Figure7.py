"""
Synthetic replication of Figure 7 from DeDL: track how estimation
MAPE for DeDL, SDL, and LR changes with DNN training epochs.

The script:
1) Simulates combinatorial treatments and nonlinear outcomes.
2) Trains a DNN outcome model over many epochs, logging training MSE.
3) Computes ATEs for all treatment combinations using:
   - LR (linear regression baseline, fixed across epochs)
   - SDL (plug-in from the DNN)
   - DeDL (doubly robust using the DNN + propensity)
4) Records MAPE versus true ATEs at each epoch and plots the curves.
"""

from __future__ import annotations

import itertools
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# -------------------------------
# Configuration
# -------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

m = 3  # number of experiments (binary treatments)
d = 10  # covariate dimension
n_train = 4000
n_eval = 1500
epochs = 400
batch_size = 128
learning_rate = 1e-3
eval_every = 1  # compute metrics every epoch for a smooth curve

# Save figure next to this script
OUTPUT_PLOT = os.path.join(os.path.dirname(__file__), "figure7_synthetic.png")


# -------------------------------
# Data generation
# -------------------------------
@dataclass
class Coefficients:
    main: np.ndarray  # shape (m, d)
    interact: List[np.ndarray]  # list of length m*(m-1)/2, each shape (d,)
    base: np.ndarray  # shape (d,)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def build_coefficients(rng: np.random.Generator) -> Coefficients:
    main = rng.normal(scale=1.0, size=(m, d))
    interact = [rng.normal(scale=0.7, size=d) for _ in range(int(m * (m - 1) / 2))]
    base = rng.normal(scale=0.5, size=d)
    return Coefficients(main=main, interact=interact, base=base)


def enumerate_combos(m: int) -> np.ndarray:
    combos = list(itertools.product([0, 1], repeat=m))
    return np.array(combos, dtype=np.int64)


def outcome_function(
    x: np.ndarray, t: np.ndarray, coefs: Coefficients, noise_std: float = 0.1, add_noise: bool = True
) -> float:
    """
    Nonlinear outcome with treatment interactions; designed so LR is misspecified
    while a well-trained DNN can capture the structure.
    """
    base_term = 1.0 + 0.4 * np.sin(x[:3].sum()) + 0.2 * x[3:6].sum()
    main_effect = 0.0
    for j in range(m):
        main_effect += (1.5 + 0.3 * j) * sigmoid(x @ coefs.main[j]) * t[j]

    # pairwise interactions
    pair_terms = [
        (t[0] * t[1], coefs.interact[0]),
        (t[0] * t[2], coefs.interact[1]),
        (t[1] * t[2], coefs.interact[2]),
    ]
    interact_effect = 0.0
    for idx, (flag, w) in enumerate(pair_terms):
        if flag:
            interact_effect += (2.0 + 0.2 * idx) * np.tanh(x @ w)

    nonlinear_trend = 0.8 * np.sin(x @ coefs.base)
    y = base_term + main_effect + interact_effect + nonlinear_trend
    if add_noise:
        y += np.random.normal(0.0, noise_std)
    return float(y)


def generate_dataset(
    rng: np.random.Generator, coefs: Coefficients, n: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = rng.normal(size=(n, d))
    combos = enumerate_combos(m)
    t_idx = rng.integers(0, len(combos), size=n)
    t = combos[t_idx]
    y = np.array([outcome_function(x[i], t[i], coefs, add_noise=True) for i in range(n)], dtype=np.float32)
    return x.astype(np.float32), t.astype(np.float32), y


def compute_true_ates(
    x: np.ndarray, combos: np.ndarray, coefs: Coefficients
) -> np.ndarray:
    """Noise-free ATE of each combo against the all-zero baseline."""
    base = combos[0]
    base_out = np.array([outcome_function(xx, base, coefs, add_noise=False) for xx in x])
    ates = []
    for t in combos:
        y_t = np.array([outcome_function(xx, t, coefs, add_noise=False) for xx in x])
        ates.append((y_t - base_out).mean())
    return np.array(ates, dtype=np.float32)


# -------------------------------
# Models
# -------------------------------
class OutcomeNet(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze(-1)


def fit_linear_regression(x: np.ndarray, t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    OLS: y ~ 1 + X + T (no higher-order interactions).
    Returns coefficients for design matrix columns.
    """
    design = np.concatenate([np.ones((len(x), 1)), x, t], axis=1)
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return beta


def predict_lr(beta: np.ndarray, x: np.ndarray, t: np.ndarray) -> np.ndarray:
    design = np.concatenate([np.ones((len(x), 1)), x, t], axis=1)
    return design @ beta


# -------------------------------
# Evaluation helpers
# -------------------------------
def mape(est: np.ndarray, true: np.ndarray, eps: float = 1e-6) -> float:
    denom = np.where(np.abs(true) < eps, eps, np.abs(true))
    return float(np.mean(np.abs((est - true) / denom)) * 100.0)


def compute_plugin_ates(
    net: OutcomeNet, x_eval: torch.Tensor, combos: np.ndarray, device: torch.device
) -> np.ndarray:
    net.eval()
    preds = []
    with torch.no_grad():
        for combo in combos:
            t_block = torch.tensor(np.repeat(combo[np.newaxis, :], len(x_eval), axis=0), dtype=torch.float32, device=device)
            inputs = torch.cat([x_eval, t_block], dim=1)
            preds.append(net(inputs).cpu().numpy())
    preds = np.stack(preds)  # shape: (num_combos, n_eval)
    base_pred = preds[0]
    plugin_ates = preds - base_pred
    return plugin_ates.mean(axis=1)


def compute_dedl_ates(
    net: OutcomeNet,
    x_eval: torch.Tensor,
    t_eval: np.ndarray,
    y_eval: np.ndarray,
    combos: np.ndarray,
    propensities: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """
    Doubly robust ATE for each combo against baseline.
    """
    net.eval()
    with torch.no_grad():
        preds_all = []
        for combo in combos:
            t_block = torch.tensor(np.repeat(combo[np.newaxis, :], len(x_eval), axis=0), dtype=torch.float32, device=device)
            inputs = torch.cat([x_eval, t_block], dim=1)
            preds_all.append(net(inputs).cpu().numpy())
    preds_all = np.stack(preds_all)  # (num_combos, n_eval)
    base_pred = preds_all[0]

    dr_ates = []
    base_idx = 0
    for idx, combo in enumerate(combos):
        plugin = (preds_all[idx] - base_pred).mean()

        # residual-based correction term
        indicator_t = (t_eval == idx).astype(np.float32)
        indicator_base = (t_eval == base_idx).astype(np.float32)
        p_t = max(propensities[idx], 1e-3)
        p_base = max(propensities[base_idx], 1e-3)

        resid_t = indicator_t * (y_eval - preds_all[idx].squeeze())
        resid_base = indicator_base * (y_eval - base_pred.squeeze())
        correction = (resid_t / p_t - resid_base / p_base).mean()
        dr_ates.append(plugin + correction)
    return np.array(dr_ates, dtype=np.float32)


# -------------------------------
# Main experiment
# -------------------------------
def main():
    rng = np.random.default_rng(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    combos = enumerate_combos(m)
    coefs = build_coefficients(rng)

    # Data
    x_train, t_train, y_train = generate_dataset(rng, coefs, n_train)
    x_eval, t_eval, y_eval = generate_dataset(rng, coefs, n_eval)
    true_ates = compute_true_ates(x_eval, combos, coefs)

    # LR baseline (fixed across epochs)
    beta_lr = fit_linear_regression(x_train, t_train, y_train)
    lr_preds = []
    for combo in combos:
        combo_block = np.repeat(combo[np.newaxis, :], len(x_eval), axis=0)
        lr_preds.append(predict_lr(beta_lr, x_eval, combo_block))
    lr_preds = np.stack(lr_preds)
    lr_ates = (lr_preds - lr_preds[0]).mean(axis=1)
    lr_mape = mape(lr_ates, true_ates)

    # DNN setup
    train_inputs = np.concatenate([x_train, t_train], axis=1)
    eval_inputs = np.concatenate([x_eval, t_eval], axis=1)
    train_ds = TensorDataset(
        torch.tensor(train_inputs, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    x_eval_tensor = torch.tensor(x_eval, dtype=torch.float32, device=device)

    net = OutcomeNet(input_dim=d + m).to(device)
    optim = torch.optim.Adam(net.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    # Propensity: empirical probability of each combo in evaluation set
    t_eval_idx = np.array([np.where((combos == t).all(axis=1))[0][0] for t in t_eval], dtype=np.int64)
    propensities = np.bincount(t_eval_idx, minlength=len(combos)) / len(t_eval_idx)

    history = {
        "epoch": [],
        "train_mse": [],
        "mape_dedl": [],
        "mape_sdl": [],
        "mape_lr": [],
    }

    for epoch in range(1, epochs + 1):
        net.train()
        running_loss = 0.0
        n_seen = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optim.zero_grad()
            preds = net(batch_x)
            loss = loss_fn(preds, batch_y)
            loss.backward()
            optim.step()

            batch_size_actual = batch_y.size(0)
            running_loss += float(loss.item()) * batch_size_actual
            n_seen += batch_size_actual

        if epoch % eval_every == 0:
            train_mse = running_loss / max(n_seen, 1)
            plugin_ates = compute_plugin_ates(net, x_eval_tensor, combos, device)
            dedl_ates = compute_dedl_ates(
                net, x_eval_tensor, t_eval_idx, y_eval, combos, propensities, device
            )
            history["epoch"].append(epoch)
            history["train_mse"].append(train_mse)
            history["mape_dedl"].append(mape(dedl_ates, true_ates))
            history["mape_sdl"].append(mape(plugin_ates, true_ates))
            history["mape_lr"].append(lr_mape)

            if epoch % 50 == 0 or epoch == 1:
                print(
                    f"Epoch {epoch:3d} | Train MSE {train_mse:.3f} | "
                    f"MAPE DeDL {history['mape_dedl'][-1]:.2f} | "
                    f"SDL {history['mape_sdl'][-1]:.2f} | LR {lr_mape:.2f}"
                )

    # Plot
    epochs_arr = np.array(history["epoch"])
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(epochs_arr, history["train_mse"], color="#8b1d40", label="Training MSE")
    ax1.set_xlabel("Training epoch")
    ax1.set_ylabel("Training MSE", color="#8b1d40")
    ax1.tick_params(axis="y", labelcolor="#8b1d40")

    ax2 = ax1.twinx()
    ax2.plot(epochs_arr, history["mape_dedl"], color="#305f8c", label="DeDL MAPE")
    ax2.plot(epochs_arr, history["mape_sdl"], color="#2c8c63", linestyle="--", label="SDL MAPE")
    ax2.plot(epochs_arr, history["mape_lr"], color="#9c6b30", linestyle=":", label="LR MAPE")
    ax2.set_ylabel("Estimation MAPE (%)")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUTPUT_PLOT, dpi=200)
    print(f"Saved plot to {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()
