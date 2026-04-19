"""
Policy network and REINFORCE trainer.

The policy is trained using the *learned* reward model as a surrogate signal.
Rollouts are generated inside the *real* CartPole environment so that we don't
compound dynamics-model errors during policy optimisation.
Evaluation always uses the true environment reward.
"""
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from pathlib import Path

STATE_DIM = 4
ACTION_DIM = 2


class PolicyNetwork(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, ACTION_DIM),
        )

    def forward(self, state_t: torch.Tensor) -> torch.distributions.Categorical:
        logits = self.net(state_t)
        return torch.distributions.Categorical(logits=logits)

    def select_action(self, state: np.ndarray) -> tuple[int, torch.Tensor]:
        s = torch.from_numpy(state.astype(np.float32)).unsqueeze(0)
        dist = self(s)
        action = dist.sample()
        return int(action.item()), dist.log_prob(action)


class REINFORCETrainer:
    """
    REINFORCE (Williams 1992) using the learned reward model as reward signal.

    Rollouts are collected in the real CartPole environment (not the dynamics
    model) so we avoid compounding prediction errors. The learned reward model
    replaces the true reward at training time — the algorithm never sees it.
    """

    def __init__(
        self,
        policy: PolicyNetwork,
        reward_model,
        lr: float = 1e-3,
        gamma: float = 0.99,
    ):
        self.policy = policy
        self.reward_model = reward_model
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        self.gamma = gamma

    def _collect_episode(self, env) -> tuple[list, list]:
        """Collect one episode using learned reward, return (log_probs, returns)."""
        state, _ = env.reset()
        log_probs, rewards = [], []

        done = False
        while not done:
            action, log_prob = self.policy.select_action(state)
            next_state, _, terminated, truncated, _ = env.step(action)
            r = self.reward_model.step_reward(state, action)   # learned reward
            log_probs.append(log_prob)
            rewards.append(r)
            state = next_state
            done = terminated or truncated

        # Discounted returns
        G, returns = 0.0, []
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        return log_probs, returns

    def train(self, n_episodes: int = 50) -> None:
        env = gym.make("CartPole-v1")
        self.policy.train()

        for ep in range(n_episodes):
            log_probs, returns = self._collect_episode(env)
            returns_t = torch.tensor(returns, dtype=torch.float32)
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

            loss = -torch.stack([lp * R for lp, R in zip(log_probs, returns_t)]).sum()
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimizer.step()

        env.close()
        self.policy.eval()

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str) -> None:
        self.policy.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))


def evaluate_policy(policy: PolicyNetwork, n_episodes: int = 20) -> tuple[float, float]:
    """
    Evaluate using the TRUE CartPole environment reward.
    This is only for measurement — the algorithm never calls this during training.
    """
    env = gym.make("CartPole-v1")
    policy.eval()
    returns = []

    for _ in range(n_episodes):
        state, _ = env.reset()
        total, done = 0.0, False
        while not done:
            with torch.no_grad():
                action, _ = policy.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
        returns.append(total)

    env.close()
    return float(np.mean(returns)), float(np.std(returns))
