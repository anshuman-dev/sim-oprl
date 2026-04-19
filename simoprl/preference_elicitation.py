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

# CartPole termination thresholds (from gymnasium source)
_X_THRESH = 2.4
_THETA_THRESH = 12 * np.pi / 180   # 0.2094 rad


def _cartpole_quality(trajectory: list) -> int:
    """
    Physics-based quality for a CartPole trajectory (real or simulated).

    Returns the number of steps the pole stays within CartPole's valid bounds.
    This is the TRUE reward for CartPole (1 per surviving step), correctly
    evaluated even for simulated trajectories whose length is always `horizon`.

    Using len(traj) would give identical scores for all simulated trajectories
    because they are all rolled out to the same fixed horizon — making oracle
    labels meaningless. This function fixes that.
    """
    count = 0
    for s, a in trajectory:
        x, _, theta, _ = s
        if abs(x) > _X_THRESH or abs(theta) > _THETA_THRESH:
            break
        count += 1
    return count


def oracle_preference(traj1: list, traj2: list, stochastic: bool = False) -> int:
    """
    Simulated oracle using true CartPole physics to label preferences.

    label = 0 → traj1 preferred
    label = 1 → traj2 preferred
    """
    r1 = _cartpole_quality(traj1)
    r2 = _cartpole_quality(traj2)

    if r1 == r2:
        return random.randint(0, 1)

    if stochastic:
        p = 1.0 / (1.0 + np.exp(-(r1 - r2)))
        return 0 if np.random.random() < p else 1
    return 0 if r1 > r2 else 1


# ─────────────────────────────────────────────────────────────────────────────

class UniformOPRL:
    """Randomly sample trajectory pairs from the offline dataset."""

    def __init__(self, dataset: list):
        self.trajectories = [[(s, a) for s, a, ns in traj] for traj in dataset]

    def get_query_pair(self, reward_model=None, policy_fn=None):
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

    def get_query_pair(self, reward_model, policy_fn=None):
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
    1. Samples starting states from the offline dataset (preferring upright pole
       angles so simulated trajectories are long enough to differentiate).
    2. Simulates new trajectories using the learned dynamics model, using a
       mix of the current policy and random actions for diversity.
    3. Scores each trajectory by:
         score = reward_uncertainty − λ · transition_uncertainty

       reward_uncertainty  (optimistic)  → query where we learn the most
       transition_uncertainty (pessimistic) → avoid OOD regions

    The pair with the highest score is the most informative query to label.
    """

    def __init__(
        self,
        dataset: list,
        dynamics_model: EnsembleDynamicsModel,
        horizon: int = 40,
        n_simulated: int = 50,
        lambda_: float = 0.5,
        epsilon: float = 0.3,       # exploration in simulated rollouts
    ):
        self.dynamics_model = dynamics_model
        self.horizon = horizon
        self.n_simulated = n_simulated
        self.lambda_ = lambda_
        self.epsilon = epsilon

        # Prefer near-upright starting states: they produce longer, more
        # informative trajectories. Using near-failure states means simulated
        # trajectories all score 0 and the oracle can't distinguish them.
        all_states = [s.copy() for traj in dataset for s, a, ns in traj]
        upright = [s for s in all_states if abs(s[2]) < _THETA_THRESH * 0.7]
        self.start_states = upright if len(upright) > 20 else all_states

    def _simulate_trajectory(self, start_state, policy_fn):
        """
        Roll out one trajectory from start_state using the dynamics model.
        Actions are chosen by policy_fn with epsilon-greedy exploration.

        Returns (trajectory, avg_transition_uncertainty).
        """
        state = start_state.copy()
        trajectory = []
        total_trans_unc = 0.0

        for _ in range(self.horizon):
            # Epsilon-greedy: explore with random actions for diversity
            if np.random.random() < self.epsilon:
                action = np.random.randint(2)
            else:
                action = int(policy_fn(state))

            next_state, trans_unc = self.dynamics_model.predict(state, action)
            trajectory.append((state.copy(), action))
            total_trans_unc += trans_unc
            state = next_state

            # Early stop if predicted state is clearly out of CartPole bounds
            # (avoids accumulating dynamics errors past the point of no return)
            if abs(state[0]) > _X_THRESH * 1.5 or abs(state[2]) > _THETA_THRESH * 2:
                break

        avg_trans_unc = total_trans_unc / max(len(trajectory), 1)
        return trajectory, avg_trans_unc

    def get_query_pair(self, reward_model, policy_fn):
        """
        Generate n_simulated candidate trajectories and return the best pair.
        """
        candidates = []
        for _ in range(self.n_simulated):
            start = random.choice(self.start_states)
            traj, trans_unc = self._simulate_trajectory(start, policy_fn)
            _, reward_unc = reward_model.predict_return(traj)

            # Sim-OPRL acquisition score
            score = reward_unc - self.lambda_ * trans_unc
            candidates.append((traj, score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0], candidates[1][0]
