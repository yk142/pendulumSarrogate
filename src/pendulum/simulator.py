"""単振子（トルク入力あり）の物理シミュレータ。"""
from dataclasses import dataclass

import numpy as np


@dataclass
class PendulumParams:
    m: float = 1.0
    L: float = 1.0
    g: float = 9.81
    b: float = 0.05  # 粘性減衰係数


class PendulumSimulator:
    """theta_ddot = -(g/L) sin(theta) - (b/(m L^2)) theta_dot + u/(m L^2)"""

    def __init__(self, params: PendulumParams = None, dt: float = 0.02):
        self.params = params or PendulumParams()
        self.dt = dt

    def dynamics(self, x: np.ndarray, u: float) -> np.ndarray:
        theta, theta_dot = x
        p = self.params
        theta_ddot = (
            -(p.g / p.L) * np.sin(theta)
            - (p.b / (p.m * p.L**2)) * theta_dot
            + u / (p.m * p.L**2)
        )
        return np.array([theta_dot, theta_ddot])

    def step(self, x: np.ndarray, u: float, dt: float = None) -> np.ndarray:
        """RK4で1ステップ積分する。"""
        dt = self.dt if dt is None else dt
        k1 = self.dynamics(x, u)
        k2 = self.dynamics(x + 0.5 * dt * k1, u)
        k3 = self.dynamics(x + 0.5 * dt * k2, u)
        k4 = self.dynamics(x + dt * k3, u)
        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def rollout(self, x0: np.ndarray, u_sequence: np.ndarray, dt: float = None) -> np.ndarray:
        """初期状態と入力列から軌道を生成する。

        Returns
        -------
        states: shape (len(u_sequence) + 1, 2) 各ステップの [theta, theta_dot]
        """
        dt = self.dt if dt is None else dt
        states = np.zeros((len(u_sequence) + 1, 2))
        states[0] = x0
        x = x0.copy()
        for i, u in enumerate(u_sequence):
            x = self.step(x, u, dt)
            states[i + 1] = x
        return states
