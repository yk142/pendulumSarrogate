"""NSSモデルの1-step教師あり学習。"""
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from .dataset import TransitionDataset, compute_normalizer, generate_transitions
from .models import NSSModel
from .simulator import PendulumSimulator
from .utils import set_seed


def train_nss_model(
    output_dir: Path,
    n_trajectories: int = 200,
    traj_len: int = 100,
    seed: int = 0,
    n_epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    hidden_dim: int = 64,
    n_hidden_layers: int = 2,
    val_fraction: float = 0.15,
):
    set_seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sim = PendulumSimulator()
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

    model = NSSModel(hidden_dim=hidden_dim, n_hidden_layers=n_hidden_layers)
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

    torch.save(model.state_dict(), output_dir / "nss_model.pt")
    with open(output_dir / "normalizer.json", "w") as f:
        json.dump(normalizer.to_dict(), f, indent=2)
    with open(output_dir / "model_config.json", "w") as f:
        json.dump({"hidden_dim": hidden_dim, "n_hidden_layers": n_hidden_layers}, f, indent=2)
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    return model, normalizer, history
