"""異なるシードで学習した複数NSSモデルによるアンサンブルと不確実性推定。"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from .dataset import generate_rollout_set
from .evaluate import load_model, rollout_error, rollout_model
from .simulator import PendulumSimulator
from .train import train_nss_model
from .utils import set_seed


def train_ensemble(
    output_dir: Path,
    n_members: int = 5,
    base_seed: int = 0,
    model_type: str = "residual",
    training_mode: str = "multistep",
    **train_kwargs,
):
    """異なるシードでn_members個のNSSモデルを学習し、output_dir/member_{i}/に保存する。"""
    output_dir = Path(output_dir)
    for i in range(n_members):
        member_dir = output_dir / f"member_{i}"
        print(f"--- training ensemble member {i} (seed={base_seed + i}) ---")
        train_nss_model(
            output_dir=member_dir,
            model_type=model_type,
            training_mode=training_mode,
            seed=base_seed + i,
            **train_kwargs,
        )


def load_ensemble(output_dir: Path, n_members: int):
    output_dir = Path(output_dir)
    return [load_model(output_dir / f"member_{i}") for i in range(n_members)]


def ensemble_rollout(members, x0: np.ndarray, u_sequence: np.ndarray):
    """各メンバーでロールアウトし、予測の集合・平均・不確実性(標準偏差ベース指標)を返す。

    Returns
    -------
    preds: shape (n_members, horizon+1, 2) 各メンバーの [theta, theta_dot] 予測
    mean_pred: shape (horizon+1, 2) メンバー間の単純平均予測
    uncertainty: shape (horizon+1,) sqrt(var(theta) + var(theta_dot))（メンバー間分散ベース）
    """
    preds = np.stack(
        [rollout_model(model, normalizer, x0, u_sequence) for model, normalizer in members],
        axis=0,
    )
    mean_pred = preds.mean(axis=0)
    var_theta = preds[..., 0].var(axis=0)
    var_theta_dot = preds[..., 1].var(axis=0)
    uncertainty = np.sqrt(var_theta + var_theta_dot)
    return preds, mean_pred, uncertainty


def evaluate_ensemble_uncertainty(
    output_dir: Path,
    n_members: int = 5,
    n_rollouts: int = 100,
    horizon: int = 200,
    seed: int = 1000,
    u_amplitude: float = 2.0,
):
    """アンサンブル分散(不確実性)が実際のロールアウト誤差・セパラトリクス近傍を予測できるかを検証する。"""
    set_seed(seed)
    output_dir = Path(output_dir)
    members = load_ensemble(output_dir, n_members)
    sim = PendulumSimulator()

    trajectories = generate_rollout_set(sim, n_rollouts, horizon, seed=seed, u_amplitude=u_amplitude)

    final_error = np.zeros(n_rollouts)
    final_uncertainty = np.zeros(n_rollouts)
    max_abs_theta_dot = np.zeros(n_rollouts)
    for i, traj in enumerate(trajectories):
        _, mean_pred, uncertainty = ensemble_rollout(members, traj.x0, traj.u_sequence)
        final_error[i] = rollout_error(traj.states_gt, mean_pred)[-1]
        final_uncertainty[i] = uncertainty[-1]
        max_abs_theta_dot[i] = np.abs(traj.states_gt[:, 1]).max()

    corr_err = float(np.corrcoef(final_uncertainty, final_error)[0, 1])
    corr_amp = float(np.corrcoef(max_abs_theta_dot, final_uncertainty)[0, 1])

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(final_uncertainty, final_error, alpha=0.6)
    ax.set_xlabel(f"ensemble uncertainty (step {horizon})")
    ax.set_ylabel(f"actual rollout error (step {horizon})")
    ax.set_title(f"uncertainty vs actual error (corr={corr_err:.3f})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "uncertainty_vs_error.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(max_abs_theta_dot, final_uncertainty, alpha=0.6, color="C2")
    ax.set_xlabel("max |theta_dot| over trajectory [rad/s]")
    ax.set_ylabel(f"ensemble uncertainty (step {horizon})")
    ax.set_title(f"uncertainty vs amplitude (corr={corr_amp:.3f})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "uncertainty_vs_amplitude.png", dpi=150)
    plt.close(fig)

    result = {"corr_uncertainty_vs_error": corr_err, "corr_uncertainty_vs_amplitude": corr_amp}
    with open(output_dir / "uncertainty_summary.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"corr(uncertainty, actual_error) = {corr_err:.3f}")
    print(f"corr(max|theta_dot|, uncertainty) = {corr_amp:.3f}")
    return result
