"""Neural State Space (NSS) サロゲートモデル。

共通インターフェース: forward(x_enc_and_u) -> next_x_enc
  x_enc_and_u: [..., 4] = [cos(theta), sin(theta), theta_dot_norm, u_norm]
  next_x_enc:  [..., 3] = [cos(theta), sin(theta), theta_dot_norm] (次ステップ, cos/sinは単位円上に再正規化済み)
"""
import torch
import torch.nn as nn


def _renormalize_cos_sin(cos_v: torch.Tensor, sin_v: torch.Tensor):
    norm = torch.sqrt(cos_v**2 + sin_v**2 + 1e-8)
    return cos_v / norm, sin_v / norm


class ResidualNSSModel(nn.Module):
    """x_{t+1} = x_t + f_NN(x_t, u_t) を学習する残差形式MLP（フェーズ1のベースラインモデル）。"""

    def __init__(self, hidden_dim: int = 64, n_hidden_layers: int = 2, dt: float = 0.02):
        super().__init__()
        self.dt = dt
        state_dim = 3  # cos, sin, theta_dot_norm
        input_dim = state_dim + 1  # + u_norm

        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, state_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x_enc_and_u: torch.Tensor) -> torch.Tensor:
        state = x_enc_and_u[..., :3]
        delta = self.net(x_enc_and_u)
        next_state = state + delta
        cos_next, sin_next = _renormalize_cos_sin(next_state[..., 0], next_state[..., 1])
        theta_dot_next = next_state[..., 2]
        return torch.stack([cos_next, sin_next, theta_dot_next], dim=-1)


class KinematicNSSModel(nn.Module):
    """物理制約(運動学)を構造に組み込んだNSSモデル。

    dtheta/dt = theta_dot は減衰・トルクの有無に依らず常に厳密に成り立つ関係のため、
    NNはtheta_dotのダイナミクスのみを学習し、thetaはtheta_dotから台形則で運動学的に積分する。
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        n_hidden_layers: int = 2,
        dt: float = 0.02,
        theta_dot_std: float = 1.0,
    ):
        super().__init__()
        self.dt = dt
        # theta_dot_norm の delta を theta_dot の物理単位に変換するためのスケール
        self.theta_dot_std = theta_dot_std
        input_dim = 4  # cos, sin, theta_dot_norm, u_norm

        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, 1))  # delta_theta_dot_norm のみ出力
        self.net = nn.Sequential(*layers)

    def forward(self, x_enc_and_u: torch.Tensor) -> torch.Tensor:
        cos_t = x_enc_and_u[..., 0]
        sin_t = x_enc_and_u[..., 1]
        theta_dot_n = x_enc_and_u[..., 2]

        delta_theta_dot_n = self.net(x_enc_and_u).squeeze(-1)
        theta_dot_n_next = theta_dot_n + delta_theta_dot_n

        # 台形則: theta_next = theta + 0.5*(theta_dot + theta_dot_next)*dt （運動学的に厳密）
        theta_dot = theta_dot_n * self.theta_dot_std
        theta_dot_next = theta_dot_n_next * self.theta_dot_std
        theta_t = torch.atan2(sin_t, cos_t)
        theta_next = theta_t + 0.5 * (theta_dot + theta_dot_next) * self.dt

        cos_next, sin_next = torch.cos(theta_next), torch.sin(theta_next)
        return torch.stack([cos_next, sin_next, theta_dot_n_next], dim=-1)


def build_model(model_type: str, hidden_dim: int = 64, n_hidden_layers: int = 2, dt: float = 0.02, theta_dot_std: float = 1.0):
    if model_type == "residual":
        return ResidualNSSModel(hidden_dim=hidden_dim, n_hidden_layers=n_hidden_layers, dt=dt)
    if model_type == "kinematic":
        return KinematicNSSModel(
            hidden_dim=hidden_dim, n_hidden_layers=n_hidden_layers, dt=dt, theta_dot_std=theta_dot_std
        )
    raise ValueError(f"unknown model_type: {model_type}")


def rollout_encoded(model: nn.Module, x_enc0: torch.Tensor, u_norm_seq: torch.Tensor) -> torch.Tensor:
    """符号化状態表現でのオープンループロールアウト（微分可能, 学習・評価共用）。

    x_enc0: [..., 3] = [cos, sin, theta_dot_norm] (初期状態)
    u_norm_seq: [..., H] 正規化済み入力列
    Returns: [..., H+1, 3] 各ステップの符号化状態（0番目は初期状態そのまま）
    """
    horizon = u_norm_seq.shape[-1]
    states = [x_enc0]
    x = x_enc0
    for t in range(horizon):
        u_t = u_norm_seq[..., t : t + 1]
        x = model(torch.cat([x, u_t], dim=-1))
        states.append(x)
    return torch.stack(states, dim=-2)
