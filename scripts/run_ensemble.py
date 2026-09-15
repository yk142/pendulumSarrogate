#!/usr/bin/env python
"""アンサンブル学習と不確実性評価を実行するCLI。

Usage:
    python scripts/run_ensemble.py all --output-dir outputs/ensemble_residual_multistep
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pendulum.ensemble import evaluate_ensemble_uncertainty, train_ensemble

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "ensemble"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "evaluate", "all"], help="実行するステップ")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--n-members", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--model-type", choices=["residual", "kinematic"], default="residual")
    parser.add_argument("--training-mode", choices=["onestep", "multistep"], default="multistep")
    parser.add_argument("--n-trajectories", type=int, default=300)
    parser.add_argument("--traj-len", type=int, default=60)
    parser.add_argument("--epochs-per-stage", type=int, default=30)
    parser.add_argument("--n-epochs", type=int, default=60)
    parser.add_argument("--n-rollouts", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.command in ("train", "all"):
        print(f"=== training ensemble (n_members={args.n_members}, model_type={args.model_type}, training_mode={args.training_mode}) ===")
        train_ensemble(
            output_dir=output_dir,
            n_members=args.n_members,
            base_seed=args.base_seed,
            model_type=args.model_type,
            training_mode=args.training_mode,
            n_trajectories=args.n_trajectories,
            traj_len=args.traj_len,
            epochs_per_stage=args.epochs_per_stage,
            n_epochs=args.n_epochs,
        )

    if args.command in ("evaluate", "all"):
        print("=== evaluating ensemble uncertainty vs actual error ===")
        evaluate_ensemble_uncertainty(
            output_dir=output_dir,
            n_members=args.n_members,
            n_rollouts=args.n_rollouts,
            horizon=args.horizon,
            seed=args.seed + 1000,
        )


if __name__ == "__main__":
    main()
