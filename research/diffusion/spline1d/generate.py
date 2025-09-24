#!/usr/bin/env python3
import argparse
from helper import generate_dataset, save_xy_csv, NaturalCubicSpline  # NaturalCubicSpline kept exported

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True)
    ap.add_argument("--noisy", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--sigma", type=float, default=0.08)
    ap.add_argument("--bmax", type=float, default=0.5)
    ap.add_argument("--num_regions", type=int, default=10)
    ap.add_argument("--num_noisy", type=int, default=5)
    ap.add_argument("--ctrl_points", type=int, default=4)
    ap.add_argument("--random_ctrl_x", action="store_true")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    data = generate_dataset(
        n=args.n, sigma=args.sigma, bmax=args.bmax,
        num_regions=args.num_regions, num_noisy=args.num_noisy,
        ctrl_points=args.ctrl_points, random_ctrl_x=args.random_ctrl_x,
        seed=args.seed
    )

    save_xy_csv(args.clean, data["x"], data["y_clean"], "x", "y_clean")
    save_xy_csv(args.noisy, data["x"], data["y_noisy"], "x", "y_noisy")

    if args.plot:
        import matplotlib.pyplot as plt
        x, y_clean, y_noisy, edges = data["x"], data["y_clean"], data["y_noisy"], data["edges"]
        plt.figure(figsize=(9,5))
        plt.plot(x, y_clean, label="clean")
        plt.plot(x, y_noisy, label="noisy", linewidth=1.25, alpha=0.9)
        for e in edges[1:-1]:
            plt.axvline(e, linestyle=":", alpha=0.2)
        plt.title("Generated data (random spline / random regions / per-region bias & noise)")
        plt.xlabel("x"); plt.ylabel("y")
        plt.legend(); plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()
