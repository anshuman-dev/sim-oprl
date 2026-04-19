"""
train.py — Run the full Sim-OPRL comparison experiment.

Compares three preference elicitation strategies across multiple seeds:
  1. Uniform OPRL  (random pairs from dataset)
  2. Uncertainty OPRL  (most uncertain pairs from dataset)
  3. Sim-OPRL  (simulated pairs from learned dynamics model)

Usage:
    python train.py

Results are saved to results/experiment_results.pkl and plotted with plot_results.py.
"""
import os
import pickle
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

from simoprl.collect_data import collect_offline_dataset, load_dataset
from simoprl.dynamics_model import EnsembleDynamicsModel
from simoprl.reward_model import EnsembleRewardModel
from simoprl.preference_elicitation import (
    UniformOPRL,
    UncertaintyOPRL,
    SimOPRL,
    oracle_preference,
)
from simoprl.policy import PolicyNetwork, REINFORCETrainer, evaluate_policy

DATA_PATH = "data/offline_dataset.pkl"
DYN_MODEL_PATH = "models/dynamics_model.pt"
RESULTS_PATH = "results/experiment_results.pkl"

QUERY_CHECKPOINTS = [5, 10, 20, 30, 50, 75, 100]
N_SEEDS = 5
POLICY_UPDATE_EVERY = 5    # retrain policy every K queries
POLICY_EPISODES = 50       # REINFORCE episodes per update
EVAL_EPISODES = 20


def run_one_seed(method_name: str, dataset: list, dynamics_model: EnsembleDynamicsModel, seed: int) -> dict:
    """
    Run one full experiment for a given method and random seed.
    Returns {query_count: mean_return} mapping.
    """
    np.random.seed(seed)
    import torch; torch.manual_seed(seed)

    # Fresh reward model and policy per seed
    reward_model = EnsembleRewardModel(n_models=3)
    policy = PolicyNetwork()
    trainer = REINFORCETrainer(policy, reward_model)

    # Instantiate elicitor
    if method_name == "uniform":
        elicitor = UniformOPRL(dataset)
    elif method_name == "uncertainty":
        elicitor = UncertaintyOPRL(dataset)
    elif method_name == "simoprl":
        elicitor = SimOPRL(dataset, dynamics_model, horizon=50, n_simulated=30)
    else:
        raise ValueError(f"Unknown method: {method_name}")

    def policy_fn(state):
        action, _ = policy.select_action(state)
        return action

    results = {}
    max_queries = max(QUERY_CHECKPOINTS)

    for q in tqdm(range(1, max_queries + 1), desc=f"  {method_name} seed={seed}", leave=False):
        # Get pair and oracle label
        fn = policy_fn if method_name == "simoprl" else None
        traj1, traj2 = elicitor.get_query_pair(reward_model, fn)
        label = oracle_preference(traj1, traj2, stochastic=True)
        reward_model.add_preference(traj1, traj2, label)
        reward_model.update(n_epochs=10)

        # Periodic policy update
        if q % POLICY_UPDATE_EVERY == 0:
            trainer.train(n_episodes=POLICY_EPISODES)

        # Record at checkpoints
        if q in QUERY_CHECKPOINTS:
            mean_ret, _ = evaluate_policy(policy, n_episodes=EVAL_EPISODES)
            results[q] = mean_ret
            tqdm.write(f"    [{method_name} seed={seed}] q={q:3d} → return={mean_ret:.1f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    parser.add_argument("--skip-collect", action="store_true")
    parser.add_argument("--skip-dynamics", action="store_true")
    args = parser.parse_args()

    # ── Dataset ──────────────────────────────────────────────────────────────
    if Path(DATA_PATH).exists() or args.skip_collect:
        print(f"Loading dataset from {DATA_PATH}")
        dataset = load_dataset(DATA_PATH)
    else:
        print("Collecting offline dataset …")
        dataset = collect_offline_dataset(n_trajectories=800, save_path=DATA_PATH)

    # ── Dynamics model ────────────────────────────────────────────────────────
    dynamics_model = EnsembleDynamicsModel(n_models=5)
    if Path(DYN_MODEL_PATH).exists() or args.skip_dynamics:
        print(f"Loading dynamics model from {DYN_MODEL_PATH}")
        dynamics_model.load(DYN_MODEL_PATH)
    else:
        print("Training dynamics model …")
        dynamics_model.train(dataset, n_epochs=100)
        dynamics_model.save(DYN_MODEL_PATH)

    # ── Comparison experiment ──────────────────────────────────────────────────
    methods = ["uniform", "uncertainty", "simoprl"]
    all_results = {m: [] for m in methods}

    for method in methods:
        print(f"\n{'='*60}")
        print(f"  Method: {method.upper()}")
        print(f"{'='*60}")
        for seed in range(args.seeds):
            seed_results = run_one_seed(method, dataset, dynamics_model, seed)
            all_results[method].append(seed_results)

    # ── Save results ──────────────────────────────────────────────────────────
    Path(RESULTS_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "wb") as f:
        pickle.dump({"results": all_results, "checkpoints": QUERY_CHECKPOINTS}, f)
    print(f"\nResults saved → {RESULTS_PATH}")
    print("Run `python plot_results.py` to generate the main figure.")


if __name__ == "__main__":
    main()
