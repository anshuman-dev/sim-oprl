import gymnasium as gym
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm

STATE_DIM = 4
ACTION_DIM = 2


def _heuristic_action(state, noise: float) -> int:
    """Simple pole-angle heuristic with configurable noise."""
    if np.random.random() < noise:
        return np.random.randint(ACTION_DIM)
    # Push in the direction the pole is leaning
    return 1 if state[2] > 0 else 0


def collect_trajectory(env, noise: float) -> list:
    """
    Collect one (state, action, next_state) trajectory.
    No reward labels stored — consistent with offline RLHF setup.
    """
    state, _ = env.reset()
    trajectory = []
    for _ in range(500):
        action = _heuristic_action(state, noise)
        next_state, _, terminated, truncated, _ = env.step(action)
        trajectory.append((state.copy(), int(action), next_state.copy()))
        state = next_state
        if terminated or truncated:
            break
    return trajectory


def collect_offline_dataset(n_trajectories=800, save_path="data/offline_dataset.pkl") -> list:
    """
    Collect a mixed-quality offline dataset from CartPole-v1.

    Quality levels:
      - noise=1.0  → fully random  (low quality)
      - noise=0.7  → mostly random (medium-low)
      - noise=0.3  → mostly heuristic (medium-high)
      - noise=0.05 → near-expert   (high quality)

    Returns list of trajectories, each a list of (s, a, s') tuples.
    """
    env = gym.make("CartPole-v1")
    noise_levels = [1.0, 0.7, 0.3, 0.05]
    per_level = n_trajectories // len(noise_levels)
    trajectories = []

    for noise in noise_levels:
        for _ in tqdm(range(per_level), desc=f"Collecting (noise={noise:.2f})", leave=False):
            trajectories.append(collect_trajectory(env, noise))

    env.close()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(trajectories, f)

    lengths = [len(t) for t in trajectories]
    print(f"Collected {len(trajectories)} trajectories | "
          f"len: min={min(lengths)} mean={np.mean(lengths):.1f} max={max(lengths)}")
    return trajectories


def load_dataset(path: str = "data/offline_dataset.pkl") -> list:
    with open(path, "rb") as f:
        return pickle.load(f)


def dataset_to_arrays(dataset: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten dataset into (states, actions_onehot, next_states) arrays."""
    states, actions, next_states = [], [], []
    for traj in dataset:
        for s, a, ns in traj:
            states.append(s)
            actions.append(np.eye(ACTION_DIM)[a])
            next_states.append(ns)
    return np.array(states, dtype=np.float32), np.array(actions, dtype=np.float32), np.array(next_states, dtype=np.float32)


def extract_sa_trajectories(dataset: list) -> list[list[tuple]]:
    """Convert (s, a, s') trajectories to (s, a) trajectories for reward model."""
    return [[(s, a) for s, a, ns in traj] for traj in dataset]
