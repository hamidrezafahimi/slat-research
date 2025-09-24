#!/usr/bin/env python3
"""
visualize_splines.py
- Plot predicted spline (from control points) and GT spline (raw dense x,y).
- Optionally overlay a proximity heatmap (requires --file).
- plot_heatmap(...) returns an Axes; no plt.show()/save inside.

Usage:
  python visualize_splines.py \
      --spline-ctrl pred_ctrl.csv \
      --gt-spline gt_dense.csv \
      [--with-heatmap --file noisy_xy.csv --max-dist 1.0 --half-life 0.5 --res-y 400 --sample-x 800 --blur-cols 0] \
      [--out out.png]
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt

from helper import *

def main():
    ap = argparse.ArgumentParser(description="Visualize predicted (ctrl) spline vs GT (dense) spline; optional heatmap.")
    # Splines
    ap.add_argument("--spline-ctrl", required=False, help="CSV with predicted control points: x,y (x strictly increasing).")
    ap.add_argument("--gt-spline", required=False, help="CSV with GT raw/dense spline points: x,y (no fitting).")
    # Heatmap (optional)
    ap.add_argument("--with-heatmap", action="store_true", help="Overlay proximity heatmap (requires --file).")
    ap.add_argument("--file", type=str, required=True, default=None, help="CSV with x,y noisy points for the heatmap.")
    ap.add_argument("--max-dist", type=float, default=1.0)
    ap.add_argument("--half-life", type=float, default=None)
    ap.add_argument("--res-y", type=int, default=400)
    ap.add_argument("--sample-x", type=int, default=None)
    ap.add_argument("--blur-cols", type=int, default=0)
    # Output
    ap.add_argument("--out", type=str, default=None, help="If set, saves the figure to this path.")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(9, 6))

    # Determine x-range (from heatmap if used; else from splines)
    x_min = x_max = None

    if args.file:
        noisy_xy = read_xy_csv(args.file)
        ax.plot(noisy_xy[:,0], noisy_xy[:,1], label="Noisy curve", color="blue")


    # Optional heatmap first (so lines draw on top)
    if args.with_heatmap:
        xs, ys, heat_masked = compute_heatmap(
            noisy_xy,
            args.max_dist,
            args.half_life,
            args.res_y,
            args.sample_x,
            args.blur_cols,
        )
        x_min, x_max = float(xs.min()), float(xs.max())
        ax = plot_heatmap(xs, ys, heat_masked, noisy_xy=noisy_xy, ax=ax, cbar_label="Proximity Score")

    if args.spline_ctrl:
        # Predicted spline (from control points)
        xs_pred, ys_pred = build_pred_spline_xy(args.spline_ctrl, x_min=x_min, x_max=x_max, n=1200)
        ax.plot(xs_pred, ys_pred, lw=2.0, color="orange", label="Predicted (ctrl→spline)")
        # Load control points and noisy data
        ctrl_xy = read_xy_csv(args.spline_ctrl)   # shape (N,2)
        # Compute score
        psc = ProximityScoreCalculator(
            noisy_xy,
            max_dist=args.max_dist,
            half_life=args.half_life
        )
        spline_score = psc.score_spline_points(ctrl_xy)
        # Print to stdout
        print(f"[spline score] {spline_score:.6f}")
        # Annotate on the figure
        ax.text(
            0.01, 0.98,
            f"Spline proximity score: {spline_score:.6f}",
            transform=ax.transAxes, ha="left", va="top",
            bbox=dict(facecolor="white", alpha=0.85, boxstyle="round,pad=0.3"),
            fontsize=10,
        )

    if args.gt_spline:
        # GT spline (raw dense points)
        xs_gt, ys_gt = load_gt_dense_xy(args.gt_spline, x_min=x_min, x_max=x_max)
        ax.plot(xs_gt, ys_gt, lw=2.0, color="deepskyblue", label="GT (dense)")

    # Cosmetics
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    if args.out:
        plt.savefig(args.out, dpi=150)
        print(f"[saved plot → {args.out}]")
    else:
        plt.show()

if __name__ == "__main__":
    main()
