"""共通ユーティリティ: シード固定・角度処理・正規化統計。"""
from dataclasses import dataclass

import numpy as np
import torch


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def wrap_angle(theta: np.ndarray) -> np.ndarray:
    """角度を [-pi, pi] にラップする（可視化用）。"""
    return (theta + np.pi) % (2 * np.pi) - np.pi


def generate_aprbs(n_steps: int, hold_range=(5, 20), amplitude: float = 2.0, rng: np.random.Generator = None) -> np.ndarray:
    """区分的一定なランダム信号（APRBS的な励起入力）を生成する。"""
    rng = rng or np.random.default_rng()
    u = np.zeros(n_steps)
    i = 0
    while i < n_steps:
        hold = rng.integers(hold_range[0], hold_range[1] + 1)
        value = rng.uniform(-amplitude, amplitude)
        u[i : i + hold] = value
        i += hold
    return u


@dataclass
class Normalizer:
    """状態量[theta_dot]や入力[u]の標準化統計。cos/sinは正規化不要（既に[-1,1]）。"""

    theta_dot_mean: float
    theta_dot_std: float
    u_mean: float
    u_std: float

    def normalize_theta_dot(self, theta_dot):
        return (theta_dot - self.theta_dot_mean) / self.theta_dot_std

    def denormalize_theta_dot(self, theta_dot_n):
        return theta_dot_n * self.theta_dot_std + self.theta_dot_mean

    def normalize_u(self, u):
        return (u - self.u_mean) / self.u_std

    def to_dict(self):
        return {
            "theta_dot_mean": self.theta_dot_mean,
            "theta_dot_std": self.theta_dot_std,
            "u_mean": self.u_mean,
            "u_std": self.u_std,
        }

    @staticmethod
    def from_dict(d):
        return Normalizer(**d)
