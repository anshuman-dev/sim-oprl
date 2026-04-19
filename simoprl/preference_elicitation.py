"""
Three preference elicitation strategies from the Sim-OPRL paper.

UniformOPRL     — random pairs from offline dataset (naïve baseline)
UncertaintyOPRL — pairs where the reward model is most uncertain (active baseline)
SimOPRL         — simulate new trajectories with the dynamics model;
                   optimistic on reward uncertainty, pessimistic on
                   transition uncertainty (the paper's contribution)
"""
import random
import numpy as np
from .reward_model import EnsembleRewardModel
from .dynamics_model import EnsembleDynamicsModel


def oracle_preference(traj1: list, traj2: list, stochastic: bool = False) -> int:
    """
    Simulated oracle that uses the true CartPole reward (episode length)
    to label which trajectory is better.

    label = 0 → traj1 preferred
    label = 1 → traj2 preferred

    In CartPole every step returns reward=1, so the true return equals
    the number of steps the agent survived.
    """
    r1 = len(traj1)   # CartPole: reward = 1 per surviving step
    r2 = len(traj2)

    if stochastic:
        # Bradley-Terry noise — closer to a real human
        p = 1.0 / (1.0 + np.exp(-(r1 - r2)))
        return 0 if np.random.random() < p else 1
    else:
        return 0 if r1 >= r2 else 1


# ─────────────────────────────────────────────────────────────────────────────

class UniformOPRL:
    """Randomly sample trajectory pairs from the offline dataset."""

    def __init__(self, dataset: list):
        self.trajectories = [[(s, a) for s, a, ns in traj] for traj in dataset]

    def get_query_pair(self, reward_model=None, policy_fn=None) -> tuple[list, list]:
        idx1, idx2 = random.sample(range(len(self.trajectories)), 2)
        return self.trajectories[idx1], self.trajectories[idx2]


# ─────────────────────────────────────────────────────────────────────────────

class UncertaintyOPRL:
    """
    Sample from offline dataset; prioritise pairs with high reward-model
    uncertainty (disagreement across ensemble members).
    """

    def __init__(self, dataset: list, n_candidates: int = 64):
        self.trajectories = [[(s, a) for s, a, ns in traj] for traj in dataset]
        self.n_candidates = n_candidates

    def get_query_pair(self, reward_model: EnsembleRewardModel, policy_fn=None) -> tuple[list, list]:
        candidates = random.sample(self.trajectories,
                                   min(self.n_candidates, len(self.trajectories)))

        uncs = np.array([reward_model.predict_return(t)[1] for t in candidates])
        top2 = np.argsort(uncs)[-2:]
        return candidates[top2[0]], candidates[top2[1]]


# ─────────────────────────────────────────────────────────────────────────────

class SimOPRL:
    """
    Core contribution of the paper.

    Instead of querying the offline dataset directly, the agent:
    1. Samples starting states from the offline dataset.
    2. Simulates new trajectories using the learned dynamics model.
    3. Scores each trajectory pair by:
         score = reward_uncertainty − λ · transition_uncertainty

       reward_uncertainty  (optimistic)  → query pairs where we learn the most
       transition_uncertainty (pessimistic) → avoid OOD regions the model can't predict

    The pair with the highest score is the most informative query.
    """

    def __init__(
        self,
        dataset: list,
        dynamics_model: EnsembleDynamicsModel,
        horizon: int = 50,
        n_simulated: int = 30,
        lambda_: float = 1.0,
    ):
        # Pool of real starting states
        self.start_states = [s.copy() for traj in dataset for s, a, ns in traj]
        self.dynamics_model = dynamics_model
        self.horizon = horizon
        self.n_simulated = n_simulated
        self.lambda_ = lambda_

    def _simulate_trajectory(self, start_state, policy_fn) -> tuple:
        """
        Roll out a trajectory from start_state using the dynamics model.

        Returns trajectory and avg_trans_uncertainty."""
        state = start_state.copy()
        trajectory = []
        total_trans_unc = 0.0

        for _ in range(self.horizon):
            action = policy_fn(state)
            next_state, trans_unc = self.dynamics_model.predict(state, action)
            trajectory.append((state.copy(), int(action)))
            total_trans_unc += trans_unc
            state = next_state

        avg_trans_unc = total_trans_unc / self.horizon
        return trajectory, avg_trans_unc

    def get_query_pair(self, reward_model, policy_fn) -> tuple:
        """
        Generate n_simulated trajectory pairs and return the best one to query.
        """
        candidates = []
        for _ in range(self.n_simulated):
            start = random.choice(self.start_states)
            traj, trans_unc = self._simulate_trajectory(start, policy_fn)
            _, reward_unc = reward_model.predict_return(traj)

            # Sim-OPRL acquisition score (Eq. 2 in paper)
            score = reward_unc - self.lambda_ * trans_unc
            candidates.append((traj, score))

        # Sort descending by score; take top-2 distinct trajectories
        candidates.sort(key=lambda x: x[1], reverse=True)
        traj1 = candidates[0][0]
        traj2 = candidates[1][0]
        return traj1, traj2
