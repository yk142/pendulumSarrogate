"""学習用1-stepデータセットとロールアウト評価用軌道の生成。"""
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from .simulator import PendulumSimulator
from .utils import Normalizer, generate_aprbs


def sample_initial_state(rng: np.random.Generator) -> np.ndarray:
    theta0 = rng.uniform(-np.pi, np.pi)
    theta_dot0 = rng.uniform(-8.0, 8.0)
    return np.array([theta0, theta_dot0])


def generate_transitions(
    sim: PendulumSimulator,
    n_trajectories: int,
    traj_len: int,
    seed: int,
    u_amplitude: float = 2.0,
):
    """短い軌道を多数生成し、(x_t, u_t, x_{t+1}) の集合を返す。"""
    rng = np.random.default_rng(seed)
    xs, us, x_nexts = [], [], []
    for _ in range(n_trajectories):
        x0 = sample_initial_state(rng)
        u_seq = generate_aprbs(traj_len, amplitude=u_amplitude, rng=rng)
        states = sim.rollout(x0, u_seq)
        xs.append(states[:-1])
        us.append(u_seq)
        x_nexts.append(states[1:])
    xs = np.concatenate(xs, axis=0)
    us = np.concatenate(us, axis=0)
    x_nexts = np.concatenate(x_nexts, axis=0)
    return xs, us, x_nexts


def compute_normalizer(us: np.ndarray, xs: np.ndarray) -> Normalizer:
    return Normalizer(
        theta_dot_mean=float(xs[:, 1].mean()),
        theta_dot_std=float(xs[:, 1].std() + 1e-8),
        u_mean=float(us.mean()),
        u_std=float(us.std() + 1e-8),
    )


def encode_state(theta: np.ndarray, theta_dot: np.ndarray, normalizer: Normalizer) -> np.ndarray:
    """状態を NN 入力表現 [cos(theta), sin(theta), theta_dot_norm] に変換する。"""
    return np.stack(
        [np.cos(theta), np.sin(theta), normalizer.normalize_theta_dot(theta_dot)], axis=-1
    )


class TransitionDataset(Dataset):
    """1-step学習用データセット。NN入出力表現へのエンコード込み。"""

    def __init__(self, xs: np.ndarray, us: np.ndarray, x_nexts: np.ndarray, normalizer: Normalizer):
        self.normalizer = normalizer
        state_enc = encode_state(xs[:, 0], xs[:, 1], normalizer)
        u_enc = normalizer.normalize_u(us)[:, None]
        next_enc = encode_state(x_nexts[:, 0], x_nexts[:, 1], normalizer)

        self.inputs = torch.tensor(
            np.concatenate([state_enc, u_enc], axis=-1), dtype=torch.float32
        )
        self.targets = torch.tensor(next_enc, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


@dataclass
class RolloutTrajectory:
    x0: np.ndarray
    u_sequence: np.ndarray
    states_gt: np.ndarray  # shape (horizon+1, 2), ground truth [theta, theta_dot]


def generate_rollout_set(
    sim: PendulumSimulator,
    n_rollouts: int,
    horizon: int,
    seed: int,
    u_amplitude: float = 2.0,
):
    """ロールアウト評価用の軌道集合を生成する（学習データとは別シード）。"""
    rng = np.random.default_rng(seed)
    trajectories = []
    for _ in range(n_rollouts):
        x0 = sample_initial_state(rng)
        u_seq = generate_aprbs(horizon, amplitude=u_amplitude, rng=rng)
        states_gt = sim.rollout(x0, u_seq)
        trajectories.append(RolloutTrajectory(x0=x0, u_sequence=u_seq, states_gt=states_gt))
    return trajectories
