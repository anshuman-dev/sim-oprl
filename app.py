"""
app.py — Interactive Sim-OPRL demo (Gradio).

The professor clicks which of two CartPole trajectories she prefers.
Each click updates the Bradley-Terry reward model.
Every 5 clicks, the policy is retrained using REINFORCE on the learned reward.
The agent's performance (true CartPole reward) is plotted live.

Deploy: gradio app.py   or   python app.py
HuggingFace Spaces: push this repo; set app.py as the entrypoint.
"""
import os
import pickle
import random
import tempfile
import numpy as np
import torch
import gymnasium as gym
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr
from pathlib import Path

from simoprl.collect_data import collect_offline_dataset, load_dataset
from simoprl.dynamics_model import EnsembleDynamicsModel
from simoprl.reward_model import EnsembleRewardModel
from simoprl.preference_elicitation import SimOPRL
from simoprl.policy import PolicyNetwork, REINFORCETrainer, evaluate_policy

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH = "data/offline_dataset.pkl"
DYN_MODEL_PATH = "models/dynamics_model.pt"
RESULTS_PATH = "results/experiment_results.pkl"

# ── Global mutable state (single-user demo) ───────────────────────────────────
class _State:
    dynamics_model: EnsembleDynamicsModel = None
    reward_model: EnsembleRewardModel = None
    policy: PolicyNetwork = None
    trainer: REINFORCETrainer = None
    elicitor: SimOPRL = None
    dataset: list = None
    query_count: int = 0
    return_history: list = []           # [(n_queries, mean_return)]
    current_traj1: list = None
    current_traj2: list = None
    initialized: bool = False

S = _State()


# ── Setup ─────────────────────────────────────────────────────────────────────

def _setup():
    """Train / load all components. Called once at startup."""
    if S.initialized:
        return

    # 1. Dataset
    if Path(DATA_PATH).exists():
        S.dataset = load_dataset(DATA_PATH)
        print(f"Dataset loaded: {len(S.dataset)} trajectories")
    else:
        print("Collecting offline dataset …")
        S.dataset = collect_offline_dataset(n_trajectories=800, save_path=DATA_PATH)

    # 2. Dynamics model (pre-trained; central to Sim-OPRL)
    S.dynamics_model = EnsembleDynamicsModel(n_models=5)
    if Path(DYN_MODEL_PATH).exists():
        S.dynamics_model.load(DYN_MODEL_PATH)
    else:
        print("Training dynamics model (first run — this takes a few minutes) …")
        S.dynamics_model.train(S.dataset, n_epochs=100)
        S.dynamics_model.save(DYN_MODEL_PATH)

    # 3. Reward model — starts blank; shaped entirely by the professor's clicks
    S.reward_model = EnsembleRewardModel(n_models=3)

    # 4. Policy — starts random; improves as reward model learns
    S.policy = PolicyNetwork()
    S.trainer = REINFORCETrainer(S.policy, S.reward_model, lr=1e-3)

    # 5. Sim-OPRL elicitor
    S.elicitor = SimOPRL(S.dataset, S.dynamics_model, horizon=50, n_simulated=40, lambda_=1.0)

    S.initialized = True
    print("Setup complete.")


# ── Trajectory simulation & rendering ────────────────────────────────────────

def _current_policy_fn(state: np.ndarray) -> int:
    if S.query_count < 5:
        return np.random.randint(2)
    action, _ = S.policy.select_action(state)
    return action


def _render_trajectory_to_gif(trajectory, path, fps=20) -> str:
    """
    Render a (state, action) trajectory to a GIF using CartPole's rgb_array renderer.
    For simulated trajectories the env state is set at each step.
    """
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env.reset()

    frames = []
    for state_arr, action in trajectory:
        # Clip to renderable range (dynamics model may predict slightly OOB states)
        clipped = np.array([
            np.clip(state_arr[0], -4.8, 4.8),
            np.clip(state_arr[1], -10.0, 10.0),
            np.clip(state_arr[2], -0.5, 0.5),
            np.clip(state_arr[3], -10.0, 10.0),
        ], dtype=np.float64)
        env.unwrapped.state = clipped
        frames.append(env.render())

    env.close()

    duration = 1.0 / fps
    imageio.mimwrite(path, frames, format="GIF", duration=duration, loop=0)
    return path


def _generate_and_render_pair() -> tuple[str, str]:
    """Ask Sim-OPRL for the next query pair and render both as GIFs."""
    traj1, traj2 = S.elicitor.get_query_pair(S.reward_model, _current_policy_fn)
    S.current_traj1 = traj1
    S.current_traj2 = traj2

    path1 = _render_trajectory_to_gif(traj1, "/tmp/simoprl_traj_A.gif")
    path2 = _render_trajectory_to_gif(traj2, "/tmp/simoprl_traj_B.gif")
    return path1, path2


# ── Plot ──────────────────────────────────────────────────────────────────────

def _make_return_plot() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.set_facecolor("#f5f5f5")
    fig.patch.set_facecolor("white")

    ax.axhline(y=21, color="#aaa", linestyle=":", linewidth=1.2, label="Random policy (~21 steps)")
    ax.axhline(y=500, color="#2ca02c", linestyle="--", linewidth=1, alpha=0.5, label="Max return (500)")

    if S.return_history:
        qs = [x[0] for x in S.return_history]
        means = np.array([x[1] for x in S.return_history])
        ax.plot(qs, means, "o-", color="#1f77b4", linewidth=2.5, markersize=7,
                label="Sim-OPRL (your preferences)")
        ax.fill_between(qs, means * 0.85, means * 1.15, alpha=0.15, color="#1f77b4")

    ax.set_xlabel("Number of Preference Queries", fontsize=12)
    ax.set_ylabel("Policy Return (True Reward)", fontsize=12)
    ax.set_title("How your preferences shape the agent", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 530)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    return fig


def _make_comparison_plot() -> plt.Figure:
    """Show pre-computed baseline comparison if results exist."""
    if not Path(RESULTS_PATH).exists():
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.text(0.5, 0.5, "Run python train.py to generate comparison figure",
                ha="center", va="center", transform=ax.transAxes, fontsize=12, color="gray")
        ax.axis("off")
        return fig

    with open(RESULTS_PATH, "rb") as f:
        data = pickle.load(f)

    results = data["results"]
    checkpoints = sorted(data["checkpoints"])
    colors = {"uniform": "#d62728", "uncertainty": "#ff7f0e", "simoprl": "#1f77b4"}
    labels = {"uniform": "Uniform OPRL", "uncertainty": "Uncertainty OPRL", "simoprl": "Sim-OPRL (paper)"}

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.set_facecolor("#f5f5f5")
    for method in ["uniform", "uncertainty", "simoprl"]:
        if method not in results:
            continue
        seed_results = results[method]
        qs = checkpoints
        means = np.array([np.mean([r.get(q, np.nan) for r in seed_results]) for q in qs])
        stds = np.array([np.std([r.get(q, np.nan) for r in seed_results]) for q in qs])
        ax.plot(qs, means, "-o", color=colors[method], linewidth=2 if method == "simoprl" else 1.5,
                markersize=5, label=labels[method])
        ax.fill_between(qs, means - stds, means + stds, alpha=0.12, color=colors[method])

    ax.axhline(y=500, color="green", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Preference Queries", fontsize=11)
    ax.set_ylabel("Policy Return", fontsize=11)
    ax.set_title("Sim-OPRL vs baselines (oracle preferences, 5 seeds)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# ── Gradio handlers ───────────────────────────────────────────────────────────

def on_load():
    _setup()
    gif1, gif2 = _generate_and_render_pair()
    plot = _make_return_plot()
    status = "Ready — click which trajectory keeps the pole balanced longer."
    return gif1, gif2, plot, status, _make_comparison_plot()


def on_preference(preferred: str):
    """Called when professor clicks 'Prefer A' or 'Prefer B'."""
    if S.current_traj1 is None:
        return on_load()

    label = 0 if preferred == "A" else 1
    S.reward_model.add_preference(S.current_traj1, S.current_traj2, label)
    S.reward_model.update(n_epochs=15)
    S.query_count += 1

    status = f"Query {S.query_count}: you preferred {'A' if label == 0 else 'B'}."

    # Retrain policy every 5 queries
    if S.query_count % 5 == 0:
        status += " Updating policy …"
        S.trainer.train(n_episodes=40)
        mean_ret, _ = evaluate_policy(S.policy, n_episodes=15)
        S.return_history.append((S.query_count, mean_ret))
        status += f" Policy return: {mean_ret:.1f}"

    gif1, gif2 = _generate_and_render_pair()
    return gif1, gif2, _make_return_plot(), status, _make_comparison_plot()


# ── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="Sim-OPRL Demo", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # Sim-OPRL: Preference Elicitation for Offline RL
    ### Pace · Schölkopf · Rätsch · Ramponi — ICLR 2025

    Two CartPole trajectories are simulated by a learned **dynamics model**, chosen by the
    **Sim-OPRL** acquisition strategy: high reward uncertainty (we learn the most here)
    and low transition uncertainty (the dynamics model is reliable here).

    **Click which run keeps the pole balanced longer.**
    Your preferences directly train the reward model via the Bradley-Terry loss.
    Every 5 clicks, the policy is re-optimised with REINFORCE on the learned reward.
    """)

    with gr.Row(equal_height=True):
        with gr.Column():
            vid_A = gr.Image(label="Trajectory A", type="filepath")
            btn_A = gr.Button("⬅  Prefer A", variant="primary", size="lg")
        with gr.Column():
            vid_B = gr.Image(label="Trajectory B", type="filepath")
            btn_B = gr.Button("Prefer B  ➡", variant="primary", size="lg")

    status_box = gr.Textbox(label="Status", interactive=False, lines=1)

    with gr.Tabs():
        with gr.Tab("Live: Your Preferences → Agent Return"):
            live_plot = gr.Plot(label="Return vs Queries (updates every 5 clicks)")
        with gr.Tab("Baseline Comparison (from train.py)"):
            comparison_plot = gr.Plot(label="Sim-OPRL vs Uniform OPRL vs Uncertainty OPRL")

    gr.Markdown("""
    ---
    ### How Sim-OPRL works

    | Step | What happens |
    |------|--------------|
    | 1 | Collect an unlabelled offline dataset (no rewards) |
    | 2 | Train an **ensemble dynamics model** on the dataset |
    | 3 | For each query: simulate trajectories, score by `reward_uncertainty − λ · transition_uncertainty` |
    | 4 | Ask for a preference on the highest-scoring pair |
    | 5 | Update the **Bradley-Terry reward model** with the preference |
    | 6 | Re-optimise the policy with REINFORCE on the learned reward |

    Sim-OPRL reaches higher returns with **fewer queries** than naïve baselines
    by asking *informative* questions, not random ones.
    """)

    # Wire up
    btn_A.click(
        fn=lambda: on_preference("A"),
        inputs=[],
        outputs=[vid_A, vid_B, live_plot, status_box, comparison_plot],
    )
    btn_B.click(
        fn=lambda: on_preference("B"),
        inputs=[],
        outputs=[vid_A, vid_B, live_plot, status_box, comparison_plot],
    )
    demo.load(
        fn=on_load,
        inputs=[],
        outputs=[vid_A, vid_B, live_plot, status_box, comparison_plot],
    )


if __name__ == "__main__":
    demo.launch(share=False)
