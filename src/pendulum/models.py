"""Neural State Space (NSS) サロゲートモデル。"""
import torch
import torch.nn as nn

from .utils import Normalizer


class NSSModel(nn.Module):
    """状態遷移 x_{t+1} = x_t + f_NN(x_t, u_t) を学習する残差形式MLP。

    入出力表現: [cos(theta), sin(theta), theta_dot_norm]（+入力 u_norm）
    """

    def __init__(self, hidden_dim: int = 64, n_hidden_layers: int = 2):
        super().__init__()
        state_dim = 3  # cos, sin, theta_dot_norm
        input_dim = state_dim + 1  # + u_norm

        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, state_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x_enc: torch.Tensor) -> torch.Tensor:
        """x_enc: [..., 4] = [cos, sin, theta_dot_norm, u_norm] -> 次状態 [cos, sin, theta_dot_norm]"""
        state = x_enc[..., :3]
        delta = self.net(x_enc)
        return state + delta

    @torch.no_grad()
    def predict_theta_thetadot(self, cos_t, sin_t, theta_dot_n, u_n, normalizer: Normalizer):
        """1ステップ先の (theta, theta_dot) を物理量として返すヘルパー（ロールアウト用）。"""
        x_enc = torch.stack([cos_t, sin_t, theta_dot_n, u_n], dim=-1)
        out = self.forward(x_enc)
        cos_next, sin_next, theta_dot_n_next = out[..., 0], out[..., 1], out[..., 2]
        # cos/sinは正規化崩れを補正するため再正規化
        norm = torch.sqrt(cos_next**2 + sin_next**2 + 1e-8)
        cos_next, sin_next = cos_next / norm, sin_next / norm
        theta_next = torch.atan2(sin_next, cos_next)
        theta_dot_next = normalizer.denormalize_theta_dot(theta_dot_n_next)
        return theta_next, theta_dot_next, cos_next, sin_next, theta_dot_n_next
