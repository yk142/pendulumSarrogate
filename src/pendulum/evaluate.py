"""NSSサロゲートのオープンループロールアウト誤差評価。"""
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from .dataset import generate_rollout_set
from .models import build_model, rollout_encoded
from .simulator import PendulumSimulator
from .utils import Normalizer, set_seed, wrap_angle


def load_model(output_dir: Path):
    output_dir = Path(output_dir)
    with open(output_dir / "model_config.json") as f:
        cfg = json.load(f)
    with open(output_dir / "normalizer.json") as f:
        norm_dict = json.load(f)
    model = build_model(
        cfg["model_type"],
        hidden_dim=cfg["hidden_dim"],
        n_hidden_layers=cfg["n_hidden_layers"],
        dt=cfg["dt"],
        theta_dot_std=cfg.get("theta_dot_std", 1.0),
    )
    model.load_state_dict(torch.load(output_dir / "nss_model.pt"))
    model.eval()
    normalizer = Normalizer.from_dict(norm_dict)
    return model, normalizer


@torch.no_grad()
def rollout_model(model, normalizer: Normalizer, x0: np.ndarray, u_sequence: np.ndarray) -> np.ndarray:
    """学習済みNSSモデルで自己回帰的にロールアウトする。

    Returns states array shape (len(u_sequence)+1, 2) = [theta, theta_dot]（thetaは連続量、非ラップ）。
    """
    theta0, theta_dot0 = x0
    x_enc0 = torch.tensor(
        [np.cos(theta0), np.sin(theta0), normalizer.normalize_theta_dot(theta_dot0)], dtype=torch.float32
    )
    u_norm_seq = torch.tensor(normalizer.normalize_u(u_sequence), dtype=torch.float32)

    encoded = rollout_encoded(model, x_enc0, u_norm_seq).numpy()  # (H+1, 3)
    theta_principal = np.arctan2(encoded[:, 1], encoded[:, 0])
    theta_dot = normalizer.denormalize_theta_dot(encoded[:, 2])

    # atan2の主値から連続量(unwrap)のthetaを復元する
    theta_unwrapped = np.zeros_like(theta_principal)
    theta_unwrapped[0] = theta0
    for i in range(1, len(theta_principal)):
        delta = wrap_angle(theta_principal[i] - theta_principal[i - 1])
        theta_unwrapped[i] = theta_unwrapped[i - 1] + delta

    return np.stack([theta_unwrapped, theta_dot], axis=-1)


def rollout_error(states_gt: np.ndarray, states_pred: np.ndarray) -> np.ndarray:
    """各ステップの誤差ノルム sqrt(wrap(dtheta)^2 + dtheta_dot^2) を返す。"""
    dtheta = wrap_angle(states_gt[:, 0] - states_pred[:, 0])
    dtheta_dot = states_gt[:, 1] - states_pred[:, 1]
    return np.sqrt(dtheta**2 + dtheta_dot**2)


def evaluate_rollout(
    output_dir: Path,
    n_rollouts: int = 20,
    horizon: int = 250,
    seed: int = 1000,
    u_amplitude: float = 2.0,
):
    set_seed(seed)
    output_dir = Path(output_dir)
    model, normalizer = load_model(output_dir)
    sim = PendulumSimulator()

    trajectories = generate_rollout_set(sim, n_rollouts, horizon, seed=seed, u_amplitude=u_amplitude)

    all_errors = np.zeros((n_rollouts, horizon + 1))
    for i, traj in enumerate(trajectories):
        states_pred = rollout_model(model, normalizer, traj.x0, traj.u_sequence)
        all_errors[i] = rollout_error(traj.states_gt, states_pred)

    mean_error = all_errors.mean(axis=0)
    max_error = all_errors.max(axis=0)
    min_error = all_errors.min(axis=0)

    fig, ax = plt.subplots(figsize=(8, 5))
    steps = np.arange(horizon + 1)
    ax.plot(steps, mean_error, label="mean error", color="C0")
    ax.fill_between(steps, min_error, max_error, alpha=0.2, color="C0", label="min-max range")
    ax.set_xlabel("rollout step")
    ax.set_ylabel("state error norm  sqrt(wrap(dtheta)^2 + dtheta_dot^2)")
    ax.set_title(f"NSS surrogate open-loop rollout error (n={n_rollouts} trajectories)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "rollout_error.png", dpi=150)
    plt.close(fig)

    np.savez(
        output_dir / "rollout_errors.npz",
        steps=steps,
        mean_error=mean_error,
        max_error=max_error,
        min_error=min_error,
    )

    result = {
        "mean_error_final": float(mean_error[-1]),
        "max_error_final": float(max_error[-1]),
        "mean_error_step10": float(mean_error[min(10, horizon)]),
    }
    with open(output_dir / "rollout_error_summary.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"rollout error @step10 (mean): {result['mean_error_step10']:.4f}")
    print(f"rollout error @final step {horizon} (mean): {result['mean_error_final']:.4f}")
    print(f"rollout error @final step {horizon} (max):  {result['max_error_final']:.4f}")
    print(f"plot saved to {output_dir / 'rollout_error.png'}")

    return result


def plot_timeseries_comparison(
    output_dir: Path,
    horizon: int = 250,
    seed: int = 2000,
    u_amplitude: float = 2.0,
    n_examples: int = 3,
    dt: float = 0.02,
):
    """物理モデル(ground truth)とNSSサロゲートの角度・角速度の時系列を並べて比較表示する。"""
    set_seed(seed)
    output_dir = Path(output_dir)
    model, normalizer = load_model(output_dir)
    sim = PendulumSimulator()

    trajectories = generate_rollout_set(sim, n_examples, horizon, seed=seed, u_amplitude=u_amplitude)
    t = np.arange(horizon + 1) * dt

    fig, axes = plt.subplots(n_examples, 2, figsize=(11, 3 * n_examples), squeeze=False)
    for i, traj in enumerate(trajectories):
        states_pred = rollout_model(model, normalizer, traj.x0, traj.u_sequence)
        theta_gt = wrap_angle(traj.states_gt[:, 0])
        theta_pred = wrap_angle(states_pred[:, 0])

        ax_theta, ax_omega = axes[i]
        ax_theta.plot(t, theta_gt, label="physical model (GT)", color="C0")
        ax_theta.plot(t, theta_pred, label="NSS surrogate", color="C1", linestyle="--")
        ax_theta.set_ylabel("theta [rad]")
        ax_theta.set_title(f"trajectory {i} : theta(t)")
        ax_theta.grid(True, alpha=0.3)
        if i == 0:
            ax_theta.legend()

        ax_omega.plot(t, traj.states_gt[:, 1], label="physical model (GT)", color="C0")
        ax_omega.plot(t, states_pred[:, 1], label="NSS surrogate", color="C1", linestyle="--")
        ax_omega.set_ylabel("theta_dot [rad/s]")
        ax_omega.set_title(f"trajectory {i} : theta_dot(t)")
        ax_omega.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("time [s]")
    axes[-1, 1].set_xlabel("time [s]")
    fig.tight_layout()
    save_path = output_dir / "timeseries_comparison.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"time-series comparison plot saved to {save_path}")
    return save_path


def analyze_error_vs_amplitude(
    output_dir: Path,
    n_rollouts: int = 100,
    horizon: int = 200,
    seed: int = 1000,
    u_amplitude: float = 2.0,
):
    """軌道の振幅・角速度の大きさと最終ロールアウト誤差の相関を診断する。"""
    set_seed(seed)
    output_dir = Path(output_dir)
    model, normalizer = load_model(output_dir)
    sim = PendulumSimulator()

    trajectories = generate_rollout_set(sim, n_rollouts, horizon, seed=seed, u_amplitude=u_amplitude)

    max_abs_theta_dot = np.zeros(n_rollouts)
    max_abs_theta = np.zeros(n_rollouts)
    final_error = np.zeros(n_rollouts)
    for i, traj in enumerate(trajectories):
        states_pred = rollout_model(model, normalizer, traj.x0, traj.u_sequence)
        errors = rollout_error(traj.states_gt, states_pred)
        final_error[i] = errors[-1]
        max_abs_theta_dot[i] = np.abs(traj.states_gt[:, 1]).max()
        max_abs_theta[i] = np.abs(wrap_angle(traj.states_gt[:, 0])).max()

    corr_theta_dot = float(np.corrcoef(max_abs_theta_dot, final_error)[0, 1])
    corr_theta = float(np.corrcoef(max_abs_theta, final_error)[0, 1])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(max_abs_theta_dot, final_error, alpha=0.6)
    axes[0].set_xlabel("max |theta_dot| over trajectory [rad/s]")
    axes[0].set_ylabel(f"final rollout error (step {horizon})")
    axes[0].set_title(f"corr={corr_theta_dot:.3f}")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(max_abs_theta, final_error, alpha=0.6, color="C1")
    axes[1].set_xlabel("max |theta| over trajectory [rad]")
    axes[1].set_ylabel(f"final rollout error (step {horizon})")
    axes[1].set_title(f"corr={corr_theta:.3f}")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"error vs amplitude/velocity (n={n_rollouts})")
    fig.tight_layout()
    save_path = output_dir / "error_vs_amplitude.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    result = {"corr_theta_dot": corr_theta_dot, "corr_theta": corr_theta}
    with open(output_dir / "error_vs_amplitude_summary.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"corr(max|theta_dot|, final_error) = {corr_theta_dot:.3f}")
    print(f"corr(max|theta|, final_error)     = {corr_theta:.3f}")
    print(f"plot saved to {save_path}")
    return result


def plot_comparison(run_specs, save_path: Path):
    """複数実験(label, output_dir)のロールアウト誤差成長カーブを1枚に重ね書きする。

    run_specs: list of (label: str, output_dir: Path)  各output_dirに rollout_errors.npz が必要
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, output_dir in run_specs:
        data = np.load(Path(output_dir) / "rollout_errors.npz")
        ax.plot(data["steps"], data["mean_error"], label=label)

    ax.set_xlabel("rollout step")
    ax.set_ylabel("mean state error norm  sqrt(wrap(dtheta)^2 + dtheta_dot^2)")
    ax.set_title("Rollout error growth comparison across mitigation strategies")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"comparison plot saved to {save_path}")
    return save_path
