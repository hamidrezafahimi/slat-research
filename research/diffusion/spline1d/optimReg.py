#!/usr/bin/env python3
"""
visualize_splines_opt.py

- Optimizes ALL control-point y's via your coordinate-descent rule.
- Optional heatmap overlay.
- --fast: skips per-iteration plotting and shows only the final result.
- Always saves final optimized control points as <input_stem>_opt.csv
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from helper import (
    read_xy_csv,
    compute_heatmap,
    plot_heatmap,
    build_pred_spline_xy,
    ProximityScoreCalculator,
)

# ---------- Helpers ----------

def _default_out_path(in_path: str) -> str:
    p = Path(in_path)
    return str(p.with_name(p.stem + "_opt.csv"))

def augmented_score(psc, ctrl_xy_current, y0, reg_l2: float) -> float:
    """
    Data score (via psc) + L2 regularization on y vs initial y0.
    Mean-square scaling keeps it size-independent.
    """
    s = psc.score_spline_points(ctrl_xy_current)
    if reg_l2 != 0.0:
        diff = ctrl_xy_current[:, 1] - y0
        s += reg_l2 * float(np.dot(diff, diff)) / diff.size
    return s

def update_plot(ax, xs_a, ys_b, ctrl_xy_pert, score_orig, score_pert,
                pert_line, pert_scatter, score_box, first_label_done):
    """Replace previous perturbed artists with current ones."""
    # remove old
    if pert_line is not None:
        pert_line.remove()
    if pert_scatter is not None:
        pert_scatter.remove()
    if score_box is not None:
        score_box.remove()

    # draw new
    label_line = None if first_label_done else "Spline"
    label_scatter = None if first_label_done else "Ctrl (pert)"
    pert_line, = ax.plot(xs_a, ys_b, lw=2.0, linestyle="--", color="green", label=label_line)
    pert_scatter = ax.scatter(
        ctrl_xy_pert[:, 0], ctrl_xy_pert[:, 1],
        s=28, marker="^", edgecolor="k", facecolor="lime", label=label_scatter
    )

    # score box
    score_box = ax.text(
        0.01, 0.98,
        f"Score orig: {score_orig:.6f}\nScore pert: {score_pert:.6f}",
        transform=ax.transAxes, ha="left", va="top",
        bbox=dict(facecolor="white", alpha=0.85, boxstyle="round,pad=0.3"),
        fontsize=10,
    )

    # cosmetics
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.2)
    ax.set_title("Original vs Perturbed Spline (with control points)")

    # legend: unique labels only
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

def numerical_gradient_all(psc, ctrl_xy_pert, score_prev, DY, dJdY, alpha, y0, reg_l2):
    """
    YOUR GD logic applied to ALL control points (coordinate-descent).
    - y_i := y_i - DY[i]
    - slope (score_new - score_prev)/DY[i]
    - DY[i] := alpha * slope
    Uses augmented_score (data + reg). No extra prints here (fast mode friendly).
    Returns: new_score, DY, dJdY
    """
    N = ctrl_xy_pert.shape[0]
    for i in range(N):
        # one coordinate step
        ctrl_xy_pert[i, 1] -= DY[i]
        score_new = augmented_score(psc, ctrl_xy_pert, y0, reg_l2)

        # your finite-diff-on-the-move slope + DY update
        # (kept exactly: slope over the step that was just taken)
        dJdY[i] = (score_new - score_prev) / DY[i]
        DY[i]   = alpha * dJdY[i]

        # carry score forward like your loop
        score_prev = score_new

    return score_prev, DY, dJdY


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(description="Visualize and optimize a spline from control points; optional heatmap; fast mode.")
    # Required IO
    ap.add_argument("--spline-ctrl", required=True, help="CSV with predicted control points: x,y (x strictly increasing).")
    ap.add_argument("--file", type=str, required=True, help="CSV with x,y noisy points for scoring/heatmap.")
    # Optimization control
    ap.add_argument("--iters", type=int, required=True, help="Number of outer iterations.")
    ap.add_argument("--init-dy", type=float, default=0.02, help="Initial coordinate step for each y (magnitude & sign).")
    ap.add_argument("--fast", action="store_true", help="No per-iteration plotting; only final plot.")
    # Regularization
    ap.add_argument("--reg-l2", type=float, default=1e-3, help="L2 regularization weight on y vs initial y. Set 0 to disable.")
    # Heatmap (optional)
    ap.add_argument("--with-heatmap", action="store_true", help="Overlay proximity heatmap (ignored until final if --fast).")
    ap.add_argument("--max-dist", type=float, default=1.0)
    ap.add_argument("--half-life", type=float, default=None)
    ap.add_argument("--res-y", type=int, default=400)
    ap.add_argument("--sample-x", type=int, default=None)
    ap.add_argument("--blur-cols", type=int, default=0)

    args = ap.parse_args()

    # Load data
    ctrl_xy = read_xy_csv(args.spline_ctrl)       # (N,2)
    noisy_xy = read_xy_csv(args.file)             # used by scorer
    N = ctrl_xy.shape[0]

    # Build scorer
    psc = ProximityScoreCalculator(noisy_xy, max_dist=args.max_dist, half_life=args.half_life)

    # Working copy & anchors
    ctrl_xy_pert = ctrl_xy.copy()
    y0 = ctrl_xy[:, 1].copy()                     # anchor for L2 reg

    # x-range for plotting
    x_all_min = ctrl_xy[:, 0].min()
    x_all_max = ctrl_xy[:, 0].max()
    pad = 0.01 * (x_all_max - x_all_min + 1e-9)
    x_min, x_max = x_all_min - pad, x_all_max + pad

    # Precompute base line for original spline
    xs_a, ys_a = build_pred_spline_xy(ctrl_xy, x_min=x_min, x_max=x_max, n=1200)

    # Scores
    score_orig = psc.score_spline_points(ctrl_xy)
    # Running score uses augmented objective
    score_prev = augmented_score(psc, ctrl_xy_pert, y0, args.reg_l2)

    # GD state (your parameters)
    alpha = 1e-6                      # your LR factor used in DY update
    DY   = np.full(N, args.init_dy, dtype=float)
    dJdY = np.zeros(N, dtype=float)

    # Plotting holders (used only in slow mode)
    if not args.fast:
        fig, ax = plt.subplots(figsize=(9, 6))
        # optional heatmap (slow mode: plot now)
        if args.with_heatmap:
            xs_h, ys_h, heat_masked = compute_heatmap(
                noisy_xy, args.max_dist, args.half_life, args.res_y, args.sample_x, args.blur_cols
            )
            ax = plot_heatmap(xs_h, ys_h, heat_masked, noisy_xy=None, ax=ax, cbar_label="Proximity Score")

        # draw original
        ax.scatter(ctrl_xy[:,0], ctrl_xy[:,1], s=28, marker="o", edgecolor="k", facecolor="orange", label="Ctrl (orig)")
        ax.plot(xs_a, ys_a, lw=2.2, color="orange", label="Spline (original ctrl)")
        pert_line = pert_scatter = score_box = None
        first_label_done = False

    # ---- Optimization loop ----
    for it in range(args.iters):
        # one full pass with your rule (data + reg in score)
        score_prev, DY, dJdY = numerical_gradient_all(
            psc, ctrl_xy_pert, score_prev, DY, dJdY, alpha, y0, args.reg_l2
        )

        # print only iteration & score in both modes (keeps --fast clean)
        print(f"iteration {it}: score={score_prev:.6f}")

        # slow mode: visualize each iteration
        if not args.fast:
            _, ys_b = build_pred_spline_xy(ctrl_xy_pert, x_min=x_min, x_max=x_max, n=1200)
            pert_line, pert_scatter, score_box = update_plot(
                ax, xs_a, ys_b, ctrl_xy_pert,
                score_orig, score_prev,
                pert_line, pert_scatter, score_box,
                first_label_done=first_label_done
            )
            first_label_done = True

    # ---- Final visualization ----
    if args.fast:
        fig, ax = plt.subplots(figsize=(9, 6))
        # optional heatmap (fast mode: plot only now)
        if args.with_heatmap:
            xs_h, ys_h, heat_masked = compute_heatmap(
                noisy_xy, args.max_dist, args.half_life, args.res_y, args.sample_x, args.blur_cols
            )
            ax = plot_heatmap(xs_h, ys_h, heat_masked, noisy_xy=None, ax=ax, cbar_label="Proximity Score")

        # originals
        ax.scatter(ctrl_xy[:,0], ctrl_xy[:,1], s=28, marker="o", edgecolor="k", facecolor="orange", label="Ctrl (orig)")
        ax.plot(xs_a, ys_a, lw=2.2, color="orange", label="Spline (original ctrl)")

        # final
        _, ys_b = build_pred_spline_xy(ctrl_xy_pert, x_min=x_min, x_max=x_max, n=1200)
        ax.plot(xs_a, ys_b, lw=2.0, linestyle="--", color="green", label="Spline (final)")
        ax.scatter(ctrl_xy_pert[:,0], ctrl_xy_pert[:,1], s=28, marker="^", edgecolor="k", facecolor="lime", label="Ctrl (final)")
        ax.set_title("Final Optimized Spline")
        ax.legend(loc="upper right")

    # ---- Always save optimized control points ----
    out_csv = _default_out_path(args.spline_ctrl)
    np.savetxt(out_csv, ctrl_xy_pert, delimiter=",", fmt="%.9g", header="x,y", comments="")
    print(f"[saved] optimized control points -> {out_csv}  shape={ctrl_xy_pert.shape}")

    plt.show()


if __name__ == "__main__":
    main()
