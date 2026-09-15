#!/usr/bin/env python
"""単振子NSSサロゲート実験の一括実行CLI。

Usage:
    python scripts/run_experiment.py all
    python scripts/run_experiment.py train
    python scripts/run_experiment.py evaluate
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pendulum.evaluate import analyze_error_vs_amplitude, evaluate_rollout, plot_timeseries_comparison
from pendulum.train import train_nss_model

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["train", "evaluate", "timeseries", "diagnose", "all"], help="実行するステップ"
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-trajectories", type=int, default=200)
    parser.add_argument("--traj-len", type=int, default=100)
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--epochs-per-stage", type=int, default=40)
    parser.add_argument("--n-rollouts", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=250)
    parser.add_argument(
        "--model-type", choices=["residual", "kinematic"], default="residual",
        help="residual: フェーズ1のベースライン残差モデル / kinematic: 運動学的制約付きモデル",
    )
    parser.add_argument(
        "--training-mode", choices=["onestep", "multistep"], default="onestep",
        help="onestep: 1-step教師あり学習 / multistep: カリキュラム型ロールアウト損失学習",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.command in ("train", "all"):
        print(f"=== training NSS surrogate (model_type={args.model_type}, training_mode={args.training_mode}) ===")
        train_nss_model(
            output_dir=output_dir,
            model_type=args.model_type,
            training_mode=args.training_mode,
            n_trajectories=args.n_trajectories,
            traj_len=args.traj_len,
            seed=args.seed,
            n_epochs=args.n_epochs,
            epochs_per_stage=args.epochs_per_stage,
        )

    if args.command in ("evaluate", "all"):
        print("=== evaluating open-loop rollout error ===")
        evaluate_rollout(
            output_dir=output_dir,
            n_rollouts=args.n_rollouts,
            horizon=args.horizon,
            seed=args.seed + 1000,
        )

    if args.command in ("timeseries", "all"):
        print("=== plotting theta(t) / theta_dot(t) comparison ===")
        plot_timeseries_comparison(
            output_dir=output_dir,
            horizon=args.horizon,
            seed=args.seed + 2000,
        )

    if args.command == "diagnose":
        print("=== analyzing error vs amplitude/velocity ===")
        analyze_error_vs_amplitude(
            output_dir=output_dir,
            n_rollouts=max(args.n_rollouts, 100),
            horizon=args.horizon,
            seed=args.seed + 1000,
        )


if __name__ == "__main__":
    main()
