"""
- Treats reward as Gaussian: predicts mean and variance, uses NLL loss
- Reports train/validation losses and reward log-likelihood on held-out data
"""

import math
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# synthetic data
from heartstep_simulator import HeartStepsSimulator


# ==========================
# 1. PyTorch Dataset wrapper
# ==========================

class TrajectoryDataset(Dataset):
    """
    Wraps a list of trajectories into a PyTorch Dataset.

    Each item is a dict with keys:
        'S': Tensor (T, s_dim)
        'A': Tensor (T,)  long
        'R': Tensor (T,)  float
    """

    def __init__(self, trajectories):
        self.traj = []
        for traj in trajectories:
            S = torch.tensor(traj["S"], dtype=torch.float32)        # (T, s_dim)
            A = torch.tensor(traj["A"], dtype=torch.long)           # (T,)
            R = torch.tensor(traj["R"], dtype=torch.float32)        # (T,)
            self.traj.append({"S": S, "A": A, "R": R})

    def __len__(self):
        return len(self.traj)

    def __getitem__(self, idx):
        return self.traj[idx]


# ==========================
# 2. LSTM 
# ==========================

class LSTMWorldModel(nn.Module):
    """
    steps:
        - embed S_t, A_t, R_t into vectors
        - concatenate embeddings
        - feed through LSTM to get hidden state h_t
        - from h_t, predict:
            S_{t+1} (continuous, via MSE loss)
            R_{t+1} as Gaussian:
                R_{t+1} ~ N(mu_{t+1}, sigma_{t+1}^2)
                output mu and log-variance (logvar)
                use Gaussian NLL loss
    """

    def __init__(self, s_dim: int, hidden_dim: int = 64, embed_dim: int = 32):
        super().__init__()
        self.s_dim = s_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim

        # State 
        self.s_net = nn.Sequential(
            nn.Linear(s_dim, embed_dim),
            nn.ReLU(),
        )

        # Action :  (0 or 1)
        self.a_emb = nn.Embedding(num_embeddings=2, embedding_dim=embed_dim)

        # Reward 
        self.r_net = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU(),
        )

        # LSTM over time
        self.lstm = nn.LSTM(
            input_size=embed_dim * 3,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        # Heads: predict next S and next R distribution
        self.s_head = nn.Linear(hidden_dim, s_dim)      # S_{t+1} mean
        self.r_head_mu = nn.Linear(hidden_dim, 1)       # R_{t+1} mean
        self.r_head_logvar = nn.Linear(hidden_dim, 1)   # log-variance for R_{t+1}

    def forward(self, S, A, R):
        """
        Args:
            S: (batch, T, s_dim)
            A: (batch, T)     (long)
            R: (batch, T, 1)

        Returns:
            pred_S: (batch, T, s_dim)       predictions for S_{t+1}
            pred_mu: (batch, T, 1)          mean for R_{t+1}
            pred_logvar: (batch, T, 1)      log-variance for R_{t+1}
        """
        # Encode state
        s_emb = self.s_net(S)             # (batch, T, embed_dim)

        # Encode action
        a_emb = self.a_emb(A)             # (batch, T, embed_dim)

        # Encode reward
        r_emb = self.r_net(R)             # (batch, T, embed_dim)

        # Concatenate embeddings
        x = torch.cat([s_emb, a_emb, r_emb], dim=-1)  # (batch, T, 3*embed_dim)

        lstm_out, _ = self.lstm(x)        # (batch, T, hidden_dim)

        pred_S = self.s_head(lstm_out)
        pred_mu = self.r_head_mu(lstm_out)
        pred_logvar = self.r_head_logvar(lstm_out)

        return pred_S, pred_mu, pred_logvar


# ==========================
# 3. Gaussian NLL loss for reward
# ==========================

def gaussian_nll(mu, logvar, target):
    """
    Gaussian negative log-likelihood:

        target ~ N(mu, sigma^2)
        sigma^2 = exp(logvar)

        NLL = 0.5 * [ (target - mu)^2 / sigma^2 + logvar + log(2*pi) ]
    """
    var = torch.exp(logvar)
    nll = 0.5 * ((target - mu) ** 2 / (var + 1e-8) + logvar + math.log(2 * math.pi))
    return nll.mean()


# ==========================
# 4. Training / evaluation loops
# ==========================

@dataclass
class TrainConfig:
    num_users: int = 1000
    T: int = 90
    sigma_noise: float = 30.0
    train_val_split: float = 0.8  # 80% train, 20% val
    batch_size: int = 32
    num_epochs: int = 20
    lr: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def split_train_val(trajectories, train_ratio=0.8):
    """
    Split trajectories by user into train/validation sets.
    """
    num = len(trajectories)
    indices = list(range(num))
    random.shuffle(indices)

    train_size = int(num * train_ratio)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_trajs = [trajectories[i] for i in train_indices]
    val_trajs = [trajectories[i] for i in val_indices]
    return train_trajs, val_trajs


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_nll = 0.0
    count = 0

    for batch in loader:
        S = batch["S"].to(device)              # (B, T, s_dim)
        A = batch["A"].to(device)              # (B, T)
        R = batch["R"].to(device)              # (B, T)

        # Use all but last time step as input,
        # predict the next step (autoregressive)
        S_in = S[:, :-1, :]
        A_in = A[:, :-1]
        R_in = R[:, :-1].unsqueeze(-1)

        S_target = S[:, 1:, :]
        R_target = R[:, 1:].unsqueeze(-1)

        pred_S, pred_mu, pred_logvar = model(S_in, A_in, R_in)

        # State reconstruction loss (MSE)
        loss_S = nn.functional.mse_loss(pred_S, S_target)

        # Reward NLL loss
        loss_R = gaussian_nll(pred_mu, pred_logvar, R_target)

        loss = loss_S + loss_R

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = S.size(0)
        total_loss += loss.item() * batch_size
        total_nll += loss_R.item() * batch_size
        count += batch_size

    avg_loss = total_loss / count
    avg_nll = total_nll / count
    return avg_loss, avg_nll


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_nll = 0.0
    count = 0

    with torch.no_grad():
        for batch in loader:
            S = batch["S"].to(device)
            A = batch["A"].to(device)
            R = batch["R"].to(device)

            S_in = S[:, :-1, :]
            A_in = A[:, :-1]
            R_in = R[:, :-1].unsqueeze(-1)

            S_target = S[:, 1:, :]
            R_target = R[:, 1:].unsqueeze(-1)

            pred_S, pred_mu, pred_logvar = model(S_in, A_in, R_in)

            loss_S = nn.functional.mse_loss(pred_S, S_target)
            loss_R = gaussian_nll(pred_mu, pred_logvar, R_target)
            loss = loss_S + loss_R

            batch_size = S.size(0)
            total_loss += loss.item() * batch_size
            total_nll += loss_R.item() * batch_size
            count += batch_size

    avg_loss = total_loss / count
    avg_nll = total_nll / count
    return avg_loss, avg_nll


# ==========================
# 5. Main
# ==========================

def main():
    cfg = TrainConfig()
    print("Using device:", cfg.device)

    # 1. Simulate dataset from YOUR synthetic HeartSteps simulator
    sim = HeartStepsSimulator(T=cfg.T, sigma_noise=cfg.sigma_noise, random_state=123)
    trajectories = sim.simulate_dataset(num_users=cfg.num_users, p_action=0.5)

    print(f"Simulated {len(trajectories)} user trajectories")

    # 2. Split into train / validation (by users)
    train_trajs, val_trajs = split_train_val(trajectories, train_ratio=cfg.train_val_split)
    print(f"Train users: {len(train_trajs)}, Val users: {len(val_trajs)}")

    # 3. Wrap in PyTorch Datasets and DataLoaders
    train_dataset = TrajectoryDataset(train_trajs)
    val_dataset = TrajectoryDataset(val_trajs)

    s_dim = train_dataset[0]["S"].shape[-1]

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

    # 4. Initialize LSTM model
    model = LSTMWorldModel(s_dim=s_dim, hidden_dim=64, embed_dim=32).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # 5. Training loop
    for epoch in range(1, cfg.num_epochs + 1):
        train_loss, train_nll = train_one_epoch(model, train_loader, optimizer, cfg.device)
        val_loss, val_nll = evaluate(model, val_loader, cfg.device)

        print(
            f"Epoch {epoch:02d} | "
            f"Train loss: {train_loss:.4f} (reward NLL {train_nll:.4f}) | "
            f"Val loss: {val_loss:.4f} (reward NLL {val_nll:.4f})"
        )

    torch.save(model.state_dict(), "lstm_heartsteps_worldmodel.pt")
    print("Saved model to lstm_heartsteps_worldmodel.pt")


if __name__ == "__main__":
    main()