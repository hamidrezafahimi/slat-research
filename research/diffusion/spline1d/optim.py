#!/usr/bin/env python3
import argparse
import matplotlib.pyplot as plt

from helper import *
from pathlib import Path
import numpy as np

def _default_out_path(in_path: str) -> str:
    p = Path(in_path)
    return str(p.with_name(p.stem + "_opt.csv"))

def update_plot(ax, xs_a, ys_b, ctrl_xy_pert, score_orig, score_pert,
                pert_line, pert_scatter, score_box, first_label_done):
    """Update the plot for a new iteration, replacing previous perturbed elements."""
    if pert_line is not None:
        pert_line.remove()
    if pert_scatter is not None:
        pert_scatter.remove()
    if score_box is not None:
        score_box.remove()

    label_line = None if first_label_done else f"Spline"
    label_scatter = None if first_label_done else "Ctrl (pert)"
    pert_line, = ax.plot(xs_a, ys_b, lw=2.0, linestyle="--", color="green", label=label_line)
    pert_scatter = ax.scatter(
        ctrl_xy_pert[:, 0], ctrl_xy_pert[:, 1],
        s=28, marker="^", edgecolor="k", facecolor="lime", label=label_scatter
    )

    score_box = ax.text(
        0.01, 0.98,
        f"Score orig: {score_orig:.6f}\nScore pert: {score_pert:.6f}",
        transform=ax.transAxes, ha="left", va="top",
        bbox=dict(facecolor="white", alpha=0.85, boxstyle="round,pad=0.3"),
        fontsize=10,
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.2)
    ax.set_title("Original vs Perturbed Spline (with control points)")

    handles, labels = ax.get_legend_handles_labels()
    seen, uniq = set(), []
    for h, l in zip(handles, labels):
        if l not in seen:
            uniq.append((h, l))
            seen.add(l)
    if uniq:
        ax.legend(*zip(*uniq), loc="upper right")

    plt.tight_layout()
    plt.pause(1)
    return pert_line, pert_scatter, score_box

def numerical_gradient_all(psc, ctrl_xy_pert, score_prev, DY, dJdY, alpha):
    """Your GD logic generalized to all control points (coordinate descent)."""
    N = ctrl_xy_pert.shape[0]
    for i in range(N):
        oldy = ctrl_xy_pert[i, 1]
        ctrl_xy_pert[i, 1] -= DY[i]
        score_new = psc.score_spline_points(ctrl_xy_pert)

        dJdY[i] = (score_new - score_prev) / DY[i]
        DY[i]   = alpha * dJdY[i]
        score_prev = score_new
    return score_prev, DY, dJdY

def main():
    ap = argparse.ArgumentParser(description="Visualize spline optimization.")
    ap.add_argument("--spline-ctrl", required=True)
    ap.add_argument("--with-heatmap", action="store_true")
    ap.add_argument("--file", type=str, required=True)
    ap.add_argument("--iters", type=int, required=True)
    ap.add_argument("--max-dist", type=float, default=1.0)
    ap.add_argument("--half-life", type=float, default=None)
    ap.add_argument("--res-y", type=int, default=400)
    ap.add_argument("--sample-x", type=int, default=None)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--blur-cols", type=int, default=0)
    ap.add_argument("--init-dy", type=float, default=0.02)
    ap.add_argument("--fast", action="store_true", help="Run without plotting during iterations, only final plot")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(9, 6))

    # Heatmap optional
    x_min = x_max = None
    noisy_xy = None
    if args.with_heatmap:
        noisy_xy = read_xy_csv(args.file)
        xs, ys, heat_masked = compute_heatmap(
            noisy_xy, args.max_dist, args.half_life, args.res_y, args.sample_x, args.blur_cols
        )
        x_min, x_max = float(xs.min()), float(xs.max())
        ax = plot_heatmap(xs, ys, heat_masked, noisy_xy=None, ax=ax, cbar_label="Proximity Score")

    # Original control points
    ctrl_xy = read_xy_csv(args.spline_ctrl)
    N = ctrl_xy.shape[0]

    if x_min is None or x_max is None:
        x_all_min = ctrl_xy[:, 0].min()
        x_all_max = ctrl_xy[:, 0].max()
        pad = 0.01 * (x_all_max - x_all_min + 1e-9)
        x_min, x_max = x_all_min - pad, x_all_max + pad

    xs_a, ys_a = build_pred_spline_xy(ctrl_xy, x_min=x_min, x_max=x_max, n=1200)
    ax.scatter(ctrl_xy[:,0], ctrl_xy[:,1], s=28, marker="o", edgecolor="k", facecolor="orange", label="Ctrl (orig)")
    ax.plot(xs_a, ys_a, lw=2.2, color="orange", label="Spline (original ctrl)")

    noisy_xy = noisy_xy if noisy_xy is not None else read_xy_csv(args.file)
    psc = ProximityScoreCalculator(noisy_xy, max_dist=args.max_dist, half_life=args.half_life)
    score_orig = psc.score_spline_points(ctrl_xy)
    print(f"original spline score ={score_orig:.6f}")

    # state
    ctrl_xy_pert = ctrl_xy.copy()
    alpha = args.lr
    DY   = np.full(N, args.init_dy, dtype=float)
    dJdY = np.zeros(N, dtype=float)
    score_prev = psc.score_spline_points(ctrl_xy_pert)

    # holders for plotting (slow mode)
    pert_line = pert_scatter = score_box = None
    first_label_done = False

    # loop
    for it in range(args.iters):
        score_prev, DY, dJdY = numerical_gradient_all(psc, ctrl_xy_pert, score_prev, DY, dJdY, alpha)
        print(f"iteration {it}: score={score_prev:.6f}")

        if not args.fast:
            _, ys_b = build_pred_spline_xy(ctrl_xy_pert, x_min=x_min, x_max=x_max, n=1200)
            pert_line, pert_scatter, score_box = update_plot(
                ax, xs_a, ys_b, ctrl_xy_pert,
                score_orig, score_prev,
                pert_line, pert_scatter, score_box,
                first_label_done=first_label_done
            )
            first_label_done = True

    # Final visualization (fast mode only)
    if args.fast:
        _, ys_b = build_pred_spline_xy(ctrl_xy_pert, x_min=x_min, x_max=x_max, n=1200)
        ax.plot(xs_a, ys_b, lw=2.0, linestyle="--", color="green", label="Spline (final)")
        ax.scatter(ctrl_xy_pert[:,0], ctrl_xy_pert[:,1], s=28, marker="^", edgecolor="k", facecolor="lime", label="Ctrl (final)")
        ax.set_title("Final Optimized Spline")
        ax.legend(loc="upper right")
    
    # --- Always save optimized control points ---
    out_csv = _default_out_path(args.spline_ctrl)
    np.savetxt(out_csv, ctrl_xy_pert, delimiter=",", fmt="%.9g", header="x,y", comments="")
    print(f"[saved] optimized control points -> {out_csv}  shape={ctrl_xy_pert.shape}")

    plt.show()

if __name__ == "__main__":
    main()
