"""学習用1-stepデータセットとロールアウト評価用軌道の生成。"""
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from .simulator import PendulumSimulator
from .utils import Normalizer, generate_aprbs


def separatrix_speed(sim: PendulumSimulator) -> float:
    """減衰・トルクなしの場合に振り子が頂点(theta=pi)を超えるために必要な臨界角速度 sqrt(4g/L)。

    診断の結果、このセパラトリクス近傍(振り子が頂点を超えるか反転するかの境界)で
    ロールアウト誤差が全モデル共通で最大化することが分かったため、学習データ生成時に
    この付近を重点的にサンプリングするための基準値として使う。
    """
    p = sim.params
    return np.sqrt(4 * p.g / p.L)


def sample_initial_state(
    rng: np.random.Generator,
    sim: PendulumSimulator = None,
    oversample_separatrix: bool = False,
    separatrix_frac: float = 0.3,
    separatrix_band: float = 1.5,
) -> np.ndarray:
    theta0 = rng.uniform(-np.pi, np.pi)
    if oversample_separatrix and sim is not None and rng.uniform() < separatrix_frac:
        v_c = separatrix_speed(sim)
        sign = rng.choice([-1.0, 1.0])
        theta_dot0 = sign * rng.uniform(v_c - separatrix_band, v_c + separatrix_band)
    else:
        theta_dot0 = rng.uniform(-8.0, 8.0)
    return np.array([theta0, theta_dot0])


def generate_transitions(
    sim: PendulumSimulator,
    n_trajectories: int,
    traj_len: int,
    seed: int,
    u_amplitude: float = 2.0,
    oversample_separatrix: bool = False,
):
    """短い軌道を多数生成し、(x_t, u_t, x_{t+1}) の集合を返す。"""
    rng = np.random.default_rng(seed)
    xs, us, x_nexts = [], [], []
    for _ in range(n_trajectories):
        x0 = sample_initial_state(rng, sim, oversample_separatrix=oversample_separatrix)
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


def generate_sequences(
    sim: PendulumSimulator,
    n_trajectories: int,
    traj_len: int,
    horizon: int,
    seed: int,
    u_amplitude: float = 2.0,
    stride: int = 1,
    oversample_separatrix: bool = False,
):
    """マルチステップ学習用に、長さ(horizon+1)の状態列と長さhorizonの入力列をスライディングウィンドウで抽出する。

    Returns
    -------
    x0s: shape (N, 2) 各シーケンスの初期状態
    u_seqs: shape (N, horizon)
    states_seqs: shape (N, horizon+1, 2)
    """
    assert traj_len >= horizon, "traj_len must be >= horizon"
    rng = np.random.default_rng(seed)
    x0s, u_seqs, states_seqs = [], [], []
    for _ in range(n_trajectories):
        x0 = sample_initial_state(rng, sim, oversample_separatrix=oversample_separatrix)
        u_seq = generate_aprbs(traj_len, amplitude=u_amplitude, rng=rng)
        states = sim.rollout(x0, u_seq)
        for start in range(0, traj_len - horizon + 1, stride):
            x0s.append(states[start])
            u_seqs.append(u_seq[start : start + horizon])
            states_seqs.append(states[start : start + horizon + 1])
    return np.array(x0s), np.array(u_seqs), np.array(states_seqs)


class SequenceDataset(Dataset):
    """マルチステップ学習用データセット。NN入出力表現へのエンコード込み。"""

    def __init__(self, x0s: np.ndarray, u_seqs: np.ndarray, states_seqs: np.ndarray, normalizer: Normalizer):
        self.normalizer = normalizer
        self.x0_enc = torch.tensor(encode_state(x0s[:, 0], x0s[:, 1], normalizer), dtype=torch.float32)
        self.u_norm_seq = torch.tensor(normalizer.normalize_u(u_seqs), dtype=torch.float32)
        target_enc = encode_state(states_seqs[..., 0], states_seqs[..., 1], normalizer)
        self.target_enc_seq = torch.tensor(target_enc, dtype=torch.float32)

    def __len__(self):
        return len(self.x0_enc)

    def __getitem__(self, idx):
        return self.x0_enc[idx], self.u_norm_seq[idx], self.target_enc_seq[idx]


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
