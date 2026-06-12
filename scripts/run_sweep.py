"""
Script: Sweep scripts/run_experiment.py over multiple envs/seeds and collect
results into a single CSV.

Usage:
    python scripts/run_sweep.py --envs hopper-medium-v2 walker2d-medium-v2 \
        --seeds 0 1 --algo iql --device cuda
    python scripts/run_sweep.py --envs hopper-medium-v2 --seeds 0 \
        --extra_args="--quick"  # for quick test (note the '=', required for flags)
"""

import argparse
import csv
import glob
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_args():
    p = argparse.ArgumentParser(description="Sweep RT experiments over envs/seeds")
    p.add_argument("--envs", type=str, nargs="+", default=["hopper-medium-v2"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--algo", type=str, default="iql", choices=["iql", "bcq"])
    p.add_argument("--exp_dir", type=str, default="experiments")
    p.add_argument("--output_csv", type=str, default="sweep_results.csv")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--extra_args", type=str, default="",
                   help="Extra raw args passed through to run_experiment.py, e.g. '--quick'")
    return p.parse_args()


def main():
    args = get_args()
    extra_args = args.extra_args.split() if args.extra_args else []

    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_experiment.py")

    for env in args.envs:
        for seed in args.seeds:
            cmd = [
                sys.executable, script_path,
                "--env", env, "--algo", args.algo, "--seed", str(seed),
                "--device", args.device, "--exp_dir", args.exp_dir,
            ] + extra_args
            print(f"\n{'=' * 60}")
            print(f"Running: {' '.join(cmd)}")
            print(f"{'=' * 60}")
            subprocess.run(cmd, check=True)

    rows = []
    for env in args.envs:
        for seed in args.seeds:
            pattern = os.path.join(args.exp_dir, f"{env}_{args.algo}_seed{seed}_*")
            matches = glob.glob(pattern)
            if not matches:
                print(f"  WARNING: no experiment dir found for {pattern}")
                continue
            exp_dir = max(matches, key=os.path.getmtime)
            results_path = os.path.join(exp_dir, "results.json")
            if not os.path.exists(results_path):
                print(f"  WARNING: no results.json in {exp_dir}")
                continue
            with open(results_path) as f:
                results = json.load(f)
            rows.append({
                "env": env,
                "algo": args.algo,
                "seed": seed,
                "best_normalized_score": results.get("best_normalized_score"),
                "final_normalized_score": results.get("final_normalized_score"),
                "exp_dir": exp_dir,
            })

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "env", "algo", "seed", "best_normalized_score", "final_normalized_score", "exp_dir",
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'=' * 60}")
    print(f"Sweep results (saved to {args.output_csv}):")
    for row in rows:
        print(f"  {row['env']} seed={row['seed']}: "
              f"best={row['best_normalized_score']:.2f} "
              f"final={row['final_normalized_score']:.2f}  ({row['exp_dir']})")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
