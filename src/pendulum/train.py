"""NSSモデルの学習（1-step / マルチステップ・ロールアウト損失）。"""
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from .dataset import (
    SequenceDataset,
    TransitionDataset,
    compute_normalizer,
    generate_sequences,
    generate_transitions,
)
from .models import build_model, rollout_encoded
from .simulator import PendulumSimulator
from .utils import set_seed

DEFAULT_CURRICULUM_HORIZONS = [1, 2, 5, 10, 20]


def _save_artifacts(output_dir: Path, model, normalizer, model_type, dt, hidden_dim, n_hidden_layers, history):
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "nss_model.pt")
    with open(output_dir / "normalizer.json", "w") as f:
        json.dump(normalizer.to_dict(), f, indent=2)
    with open(output_dir / "model_config.json", "w") as f:
        json.dump(
            {
                "model_type": model_type,
                "dt": dt,
                "hidden_dim": hidden_dim,
                "n_hidden_layers": n_hidden_layers,
                "theta_dot_std": normalizer.theta_dot_std,
            },
            f,
            indent=2,
        )
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)


def _train_onestep(
    output_dir: Path,
    model_type: str,
    n_trajectories: int,
    traj_len: int,
    seed: int,
    n_epochs: int,
    batch_size: int,
    lr: float,
    hidden_dim: int,
    n_hidden_layers: int,
    val_fraction: float,
    dt: float,
):
    sim = PendulumSimulator(dt=dt)
    xs, us, x_nexts = generate_transitions(sim, n_trajectories, traj_len, seed=seed)
    normalizer = compute_normalizer(us, xs)

    dataset = TransitionDataset(xs, us, x_nexts, normalizer)
    n_val = int(len(dataset) * val_fraction)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = build_model(model_type, hidden_dim, n_hidden_layers, dt, normalizer.theta_dot_std)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            preds = model(inputs)
            loss = loss_fn(preds, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(inputs)
        train_loss /= len(train_set)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                preds = model(inputs)
                val_loss += loss_fn(preds, targets).item() * len(inputs)
        val_loss /= max(len(val_set), 1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if epoch % max(1, n_epochs // 10) == 0 or epoch == n_epochs - 1:
            print(f"epoch {epoch:4d}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

    return model, normalizer, history


def _train_multistep(
    output_dir: Path,
    model_type: str,
    n_trajectories: int,
    traj_len: int,
    seed: int,
    epochs_per_stage: int,
    batch_size: int,
    lr: float,
    hidden_dim: int,
    n_hidden_layers: int,
    val_fraction: float,
    dt: float,
    curriculum_horizons=None,
):
    curriculum_horizons = curriculum_horizons or DEFAULT_CURRICULUM_HORIZONS
    max_horizon = max(curriculum_horizons)

    sim = PendulumSimulator(dt=dt)
    x0s, u_seqs, states_seqs = generate_sequences(
        sim, n_trajectories, traj_len, max_horizon, seed=seed, stride=max(1, max_horizon // 4)
    )
    normalizer = compute_normalizer(u_seqs.reshape(-1), states_seqs.reshape(-1, 2))

    dataset = SequenceDataset(x0s, u_seqs, states_seqs, normalizer)
    n_val = int(len(dataset) * val_fraction)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = build_model(model_type, hidden_dim, n_hidden_layers, dt, normalizer.theta_dot_std)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    history = {"train_loss": [], "val_loss": [], "horizon": []}
    for horizon in curriculum_horizons:
        print(f"--- curriculum stage: horizon={horizon} ---")
        for epoch in range(epochs_per_stage):
            model.train()
            train_loss = 0.0
            for x0_enc, u_norm_seq, target_enc_seq in train_loader:
                u_h = u_norm_seq[:, :horizon]
                target_h = target_enc_seq[:, : horizon + 1]
                optimizer.zero_grad()
                pred_h = rollout_encoded(model, x0_enc, u_h)
                loss = loss_fn(pred_h, target_h)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(x0_enc)
            train_loss /= len(train_set)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x0_enc, u_norm_seq, target_enc_seq in val_loader:
                    u_h = u_norm_seq[:, :horizon]
                    target_h = target_enc_seq[:, : horizon + 1]
                    pred_h = rollout_encoded(model, x0_enc, u_h)
                    val_loss += loss_fn(pred_h, target_h).item() * len(x0_enc)
            val_loss /= max(len(val_set), 1)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["horizon"].append(horizon)
            if epoch % max(1, epochs_per_stage // 5) == 0 or epoch == epochs_per_stage - 1:
                print(f"  epoch {epoch:4d}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

    return model, normalizer, history


def train_nss_model(
    output_dir: Path,
    model_type: str = "residual",
    training_mode: str = "onestep",
    n_trajectories: int = 200,
    traj_len: int = 100,
    seed: int = 0,
    n_epochs: int = 100,
    epochs_per_stage: int = 40,
    curriculum_horizons=None,
    batch_size: int = 256,
    lr: float = 1e-3,
    hidden_dim: int = 64,
    n_hidden_layers: int = 2,
    val_fraction: float = 0.15,
    dt: float = 0.02,
):
    set_seed(seed)
    output_dir = Path(output_dir)

    if training_mode == "onestep":
        model, normalizer, history = _train_onestep(
            output_dir,
            model_type,
            n_trajectories,
            traj_len,
            seed,
            n_epochs,
            batch_size,
            lr,
            hidden_dim,
            n_hidden_layers,
            val_fraction,
            dt,
        )
    elif training_mode == "multistep":
        model, normalizer, history = _train_multistep(
            output_dir,
            model_type,
            n_trajectories,
            traj_len,
            seed,
            epochs_per_stage,
            batch_size,
            lr,
            hidden_dim,
            n_hidden_layers,
            val_fraction,
            dt,
            curriculum_horizons,
        )
    else:
        raise ValueError(f"unknown training_mode: {training_mode}")

    _save_artifacts(output_dir, model, normalizer, model_type, dt, hidden_dim, n_hidden_layers, history)
    return model, normalizer, history
