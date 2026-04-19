import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm

STATE_DIM = 4
ACTION_DIM = 2
INPUT_DIM = STATE_DIM + ACTION_DIM


class _DynamicsNet(nn.Module):
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, STATE_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EnsembleDynamicsModel:
    """
    Ensemble of MLPs that learn P(s' | s, a) from an offline dataset.

    Uncertainty is measured as the std of next-state predictions across
    ensemble members — high std → out-of-distribution transition.
    """

    def __init__(self, n_models: int = 5, hidden_dim: int = 256, lr: float = 1e-3):
        self.n_models = n_models
        self.models = [_DynamicsNet(hidden_dim) for _ in range(n_models)]
        self.optimizers = [torch.optim.Adam(m.parameters(), lr=lr) for m in self.models]
        # Normalisation statistics (set during train)
        self.state_mean: np.ndarray = np.zeros(STATE_DIM, dtype=np.float32)
        self.state_std: np.ndarray = np.ones(STATE_DIM, dtype=np.float32)

    # ── Normalisation helpers ──────────────────────────────────────────────

    def _norm_state(self, s: np.ndarray) -> np.ndarray:
        return (s - self.state_mean) / (self.state_std + 1e-8)

    def _denorm_state(self, s: np.ndarray) -> np.ndarray:
        return s * (self.state_std + 1e-8) + self.state_mean

    # ── Training ──────────────────────────────────────────────────────────

    def train(
        self,
        dataset: list,
        n_epochs: int = 100,
        batch_size: int = 512,
    ) -> None:
        from .collect_data import dataset_to_arrays

        states, actions, next_states = dataset_to_arrays(dataset)

        # Fit normalisation on full dataset
        self.state_mean = states.mean(axis=0)
        self.state_std = states.std(axis=0) + 1e-8

        states_n = (states - self.state_mean) / self.state_std
        next_states_n = (next_states - self.state_mean) / self.state_std
        X = np.concatenate([states_n, actions], axis=1).astype(np.float32)  # (N, 6)
        Y = next_states_n.astype(np.float32)                                # (N, 4)
        N = len(X)

        for model_idx, (model, opt) in enumerate(zip(self.models, self.optimizers)):
            # Bootstrap sample — each model sees a different data subset
            boot_idx = np.random.choice(N, N, replace=True)
            Xb, Yb = X[boot_idx], Y[boot_idx]

            for epoch in tqdm(range(n_epochs), desc=f"Dynamics model {model_idx+1}/{self.n_models}", leave=False):
                perm = np.random.permutation(N)
                epoch_loss = 0.0
                for i in range(0, N, batch_size):
                    idx = perm[i : i + batch_size]
                    x_t = torch.from_numpy(Xb[idx])
                    y_t = torch.from_numpy(Yb[idx])
                    pred = model(x_t)
                    loss = nn.MSELoss()(pred, y_t)
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    epoch_loss += loss.item()

        print("Dynamics model training complete.")

    # ── Inference ─────────────────────────────────────────────────────────

    def predict(
        self,
        state: np.ndarray,
        action,
    ) -> tuple:
        """
        Returns (mean_next_state, transition_uncertainty).

        transition_uncertainty = mean std across state dims, averaged over ensemble.
        Higher value → ensemble disagrees → out-of-distribution.
        """
        state_n = self._norm_state(state).astype(np.float32)
        action_oh = np.eye(ACTION_DIM, dtype=np.float32)[int(action)]
        x = torch.from_numpy(np.concatenate([state_n, action_oh])).unsqueeze(0)  # (1, 6)

        preds = []
        with torch.no_grad():
            for model in self.models:
                preds.append(model(x).squeeze(0).numpy())  # (4,)

        preds = np.stack(preds)             # (n_models, 4)
        mean_n = preds.mean(axis=0)         # normalised space
        std_n = preds.std(axis=0)           # normalised space

        mean = self._denorm_state(mean_n)
        uncertainty = float(std_n.mean())   # scalar summary of disagreement
        return mean, uncertainty

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dicts": [m.state_dict() for m in self.models],
            "state_mean": self.state_mean,
            "state_std": self.state_std,
        }
        torch.save(payload, path)
        print(f"Dynamics model saved → {path}")

    def load(self, path: str) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        for model, sd in zip(self.models, payload["state_dicts"]):
            model.load_state_dict(sd)
        self.state_mean = payload["state_mean"]
        self.state_std = payload["state_std"]
        print(f"Dynamics model loaded ← {path}")
