import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

STATE_DIM = 4
ACTION_DIM = 2
STEP_INPUT_DIM = STATE_DIM + ACTION_DIM   # 6


class _StepRewardNet(nn.Module):
    """
    Predicts a per-step reward scalar from (state, action).
    Trajectory reward = Σ_t r(s_t, a_t).
    """

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STEP_INPUT_DIM, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, sa: torch.Tensor) -> torch.Tensor:
        """sa: (batch, 6) → (batch,)"""
        return self.net(sa).squeeze(-1)

    def trajectory_return(self, trajectory: list) -> torch.Tensor:
        """
        trajectory: list of (state: np.ndarray, action: int)
        Returns scalar tensor (summed per-step reward).
        """
        sa_pairs = []
        for s, a in trajectory:
            action_oh = np.eye(ACTION_DIM, dtype=np.float32)[int(a)]
            sa_pairs.append(np.concatenate([s.astype(np.float32), action_oh]))
        sa_t = torch.from_numpy(np.stack(sa_pairs))   # (T, 6)
        return self.forward(sa_t).sum()                # scalar


class EnsembleRewardModel:
    """
    Ensemble of step-reward networks trained with the Bradley-Terry loss on
    human (or oracle) trajectory preferences.

    Bradley-Terry: P(τ1 ≻ τ2) = σ(R(τ1) − R(τ2))
    Loss = −log σ(R(preferred) − R(rejected))

    Uncertainty = std of ensemble return predictions — high → reward model
    is uncertain → this pair is informative to query.
    """

    def __init__(self, n_models: int = 3, hidden_dim: int = 256, lr: float = 3e-4):
        self.n_models = n_models
        self.models = [_StepRewardNet(hidden_dim) for _ in range(n_models)]
        self.optimizers = [torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-4)
                           for m in self.models]
        self.preference_buffer: list[tuple] = []  # (traj1, traj2, label)

    # ── Preference buffer ──────────────────────────────────────────────────

    def add_preference(self, traj1: list, traj2: list, label: int) -> None:
        """
        label = 0 → traj1 preferred
        label = 1 → traj2 preferred
        """
        self.preference_buffer.append((traj1, traj2, int(label)))

    # ── Training ──────────────────────────────────────────────────────────

    def update(self, n_epochs: int = 20) -> float:
        """Re-train all ensemble members on current preference buffer."""
        if len(self.preference_buffer) < 2:
            return float("nan")

        total_loss = 0.0
        for model, opt in zip(self.models, self.optimizers):
            for _ in range(n_epochs):
                perm = np.random.permutation(len(self.preference_buffer))
                epoch_loss = 0.0
                for idx in perm:
                    traj1, traj2, label = self.preference_buffer[idx]
                    r1 = model.trajectory_return(traj1)
                    r2 = model.trajectory_return(traj2)

                    # Bradley-Terry loss
                    if label == 0:
                        loss = -torch.log(torch.sigmoid(r1 - r2) + 1e-8)
                    else:
                        loss = -torch.log(torch.sigmoid(r2 - r1) + 1e-8)

                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    epoch_loss += loss.item()
                total_loss += epoch_loss

        return total_loss / (self.n_models * n_epochs * len(self.preference_buffer))

    # ── Inference ─────────────────────────────────────────────────────────

    def predict_return(self, trajectory: list[tuple]) -> tuple[float, float]:
        """
        Returns (mean_return, reward_uncertainty).
        reward_uncertainty = std of ensemble predictions.
        """
        returns = []
        with torch.no_grad():
            for model in self.models:
                r = model.trajectory_return(trajectory)
                returns.append(r.item())
        return float(np.mean(returns)), float(np.std(returns))

    def step_reward(self, state: np.ndarray, action: int) -> float:
        """Mean per-step reward across ensemble (used during policy training)."""
        action_oh = np.eye(ACTION_DIM, dtype=np.float32)[int(action)]
        sa = torch.from_numpy(np.concatenate([state.astype(np.float32), action_oh])).unsqueeze(0)
        with torch.no_grad():
            rewards = [m(sa).item() for m in self.models]
        return float(np.mean(rewards))

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dicts": [m.state_dict() for m in self.models],
            "preference_buffer": self.preference_buffer,
        }
        torch.save(payload, path)

    def load(self, path: str) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        for model, sd in zip(self.models, payload["state_dicts"]):
            model.load_state_dict(sd)
        self.preference_buffer = payload.get("preference_buffer", [])
