#!/usr/bin/env python
"""複数実験のロールアウト誤差成長カーブを比較プロットするCLI。

Usage:
    python scripts/compare_runs.py \
        --run "residual+onestep=outputs/residual_onestep" \
        --run "residual+multistep=outputs/residual_multistep" \
        --run "kinematic+onestep=outputs/kinematic_onestep" \
        --run "kinematic+multistep=outputs/kinematic_multistep" \
        --output outputs/comparison.png
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pendulum.evaluate import plot_comparison


def parse_run_spec(spec: str):
    label, path = spec.split("=", 1)
    return label, Path(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", action="append", required=True,
        help="label=output_dir の形式で指定（複数指定可）",
    )
    parser.add_argument("--output", default="outputs/comparison.png")
    args = parser.parse_args()

    run_specs = [parse_run_spec(spec) for spec in args.run]
    plot_comparison(run_specs, args.output)


if __name__ == "__main__":
    main()
