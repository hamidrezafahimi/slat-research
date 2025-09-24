#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt

from helper import (
    read_xy_csv,
    compute_heatmap,            # background heatmap (uses ProximityScoreCalculator)
    ProximityScoreCalculator,   # scoring engine
    RandNeighborSplineGen
)


# ---- utilities ---------------------------------------------------------------

def _extract_spline_callable(candidate: dict):
    """
    Try several common keys to obtain the callable spline f(x)->y.
    """
    for k in ("spline", "rand_spline", "fit", "fn"):
        if k in candidate:
            return candidate[k]
    raise KeyError("Candidate dict has no spline callable under keys: 'spline'|'rand_spline'|'fit'|'fn'.")

def _extract_ctrl_points(candidate: dict) -> np.ndarray:
    """
    Returns control points as an (M,2) float array [[x,y],...].
    Tries different common layouts.
    """
    if "ctrl" in candidate and isinstance(candidate["ctrl"], (list, np.ndarray)):
        arr = np.asarray(candidate["ctrl"], dtype=float)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            return arr[:, :2]

    # separate x/y arrays
    if "ctrl_x" in candidate and "ctrl_y" in candidate:
        x = np.asarray(candidate["ctrl_x"], dtype=float)
        y = np.asarray(candidate["ctrl_y"], dtype=float)
        return np.column_stack([x, y])

    # generic names
    if "pts_ctrl" in candidate:
        arr = np.asarray(candidate["pts_ctrl"], dtype=float)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            return arr[:, :2]

    raise KeyError("Candidate dict has no recognizable control points: tried 'ctrl', ('ctrl_x','ctrl_y'), 'pts_ctrl'.")

def _evaluate_total_score(xs_eval: np.ndarray, spline_fn, psc: ProximityScoreCalculator) -> float:
    ys_eval = spline_fn(xs_eval)
    total = 0.0
    # Sum proximity score at each (x, y(x))
    for x, y in zip(xs_eval, ys_eval):
        s = float(psc.score_column(float(x), np.array([float(y)], dtype=float))[0])
        total += s
    return float(total)


# ---- CLI ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Generate N random splines via RandNeighborSplineGen, score each, plot all (gray) + best (red), and save the best control points."
    )
    # Data / scoring
    ap.add_argument("--file", required=True, help="CSV with columns x,y (reference data)")
    ap.add_argument("--max-dist", type=float, default=1.0, help="cutoff distance (>= => score 0)")
    ap.add_argument("--half-life", type=float, default=None,
                    help="distance half-life for score decay (default: max_dist/3)")

    # Generator contract: it will return N splines internally
    ap.add_argument("--n-splines", type=int, default=10, help="number of candidate splines to generate internally")

    # You can pass through a few common knobs; the generator may ignore any it doesn't use
    ap.add_argument("--k", type=int, default=3, help="spline degree (if applicable to the generator)")
    ap.add_argument("--K-ctrl", type=int, default=6, help="number of control points (if applicable)")
    ap.add_argument("--smooth", default="auto",
                    help='smoothing for baseline ("auto" or float if generator uses it)')
    ap.add_argument("--alpha-low", type=float, default=0.2, help="lower alpha bound (if generator uses it)")
    ap.add_argument("--alpha-high", type=float, default=0.8, help="upper alpha bound (if generator uses it)")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for the generator")

    # Heatmap / output
    ap.add_argument("--plot", action="store_true", help="visualize heatmap + candidates")
    ap.add_argument("--res-y", type=int, default=400, help="vertical resolution of heatmap grid")
    ap.add_argument("--sample-x", type=int, default=None, help="downsample unique x columns in the heatmap")
    ap.add_argument("--blur-cols", type=int, default=0, help="optional small odd width (e.g., 3) for heatmap x-blur")
    ap.add_argument("--out", type=str, default=None, help="optional path to save the figure (e.g., result.png)")
    ap.add_argument("--save-best-ctrl", type=str, default="best_ctrl.csv",
                    help="output CSV path to save the best spline's control points (x,y)")

    args = ap.parse_args()

    # Load data and evaluation x-grid (exactly the original unique x's)
    arr_ref = read_xy_csv(args.file)
    xs_unique = np.unique(arr_ref[:, 0])
    x_min, x_max = xs_unique.min(), xs_unique.max()

    # Scorer over the raw data
    psc = ProximityScoreCalculator(arr_ref, max_dist=args.max_dist, half_life=args.half_life)

    # Ask the generator to produce ALL candidates in one shot
    gen = RandNeighborSplineGen(
        k=args.k,
        K_ctrl=args.K_ctrl,
        smooth=(args.smooth if args.smooth != "auto" else "auto"),
        alpha_low=args.alpha_low,
        alpha_high=args.alpha_high,
        seed=args.seed,
    )

    # Contract: one call returns a LIST of candidate dicts
    # (each dict must contain a callable spline and its control points)
    candidates = gen.generate(arr_ref[:, 0], arr_ref[:, 1], n=args.n_splines, visualize=False)

    if not isinstance(candidates, (list, tuple)) or len(candidates) == 0:
        raise RuntimeError("RandNeighborSplineGen.generate(...) returned no candidates.")

    # Score each candidate at xs_unique; keep dense curves for plotting
    x_draw = np.linspace(x_min, x_max, 1200)
    curves = []
    scores = []
    best_idx = -1
    best_score = -np.inf
    best_curve = None
    best_ctrl = None

    for i, cand in enumerate(candidates):
        spline_fn = _extract_spline_callable(cand)
        ctrl_pts = _extract_ctrl_points(cand)

        total = psc.score_spline_points(ctrl_pts)
        scores.append(total)

        y_draw = spline_fn(x_draw)
        curves.append((x_draw, y_draw))

        if total > best_score:
            best_score = total
            best_idx = i
            best_curve = (x_draw, y_draw)
            best_ctrl = ctrl_pts

    scores = np.asarray(scores, dtype=float)

    # Report & save best control points
    print(f"num_candidates : {len(candidates)}")
    print(f"best_index     : {best_idx}")
    print(f"best_score     : {best_score:.6f}")
    print(f"mean_score     : {np.mean(scores):.6f}")
    print(f"std_score      : {np.std(scores):.6f}")

    if best_ctrl is None:
        raise RuntimeError("Best candidate control points not found in generator output.")
    np.savetxt(args.save_best_ctrl, np.asarray(best_ctrl, dtype=float), delimiter=",", header="x,y", comments="")
    print(f"[saved best spline control points → {args.save_best_ctrl}]")

    # Plot (no points; curves only)
    if args.plot:
        xs_hm, ys_hm, heat_masked = compute_heatmap(
            arr_ref,
            max_dist=args.max_dist,
            half_life=args.half_life,
            res_y=args.res_y,
            sample_x=args.sample_x,
            blur_cols=args.blur_cols,
        )

        fig, ax = plt.subplots(figsize=(11, 6))
        ax.set_facecolor("white")

        # Heatmap (white where score==0)
        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color="white")
        im = ax.imshow(
            heat_masked,
            extent=[xs_hm.min(), xs_hm.max(), ys_hm.min(), ys_hm.max()],
            origin="lower", aspect="auto",
            cmap=cmap, vmin=0.0, vmax=1.0, interpolation="nearest",
        )
        plt.colorbar(im, ax=ax, label="Proximity Score")

        # All candidates in thin gray
        for idx, (xx, yy) in enumerate(curves):
            if idx != best_idx:
                ax.plot(xx, yy, color=(0.45, 0.45, 0.45), lw=1.0, alpha=0.95)

        # Best in thick red
        if best_curve is not None:
            xx_b, yy_b = best_curve
            ax.plot(xx_b, yy_b, color="red", lw=2.6, label=f"best (idx={best_idx}, score={best_score:.3f})")

        ax.set_xlim(x_min, x_max)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("RandNeighborSplineGen: candidates over proximity heatmap (best in red)")
        ax.legend(loc="upper right")
        plt.tight_layout()

        if args.out:
            plt.savefig(args.out, dpi=150)
            print(f"[saved plot → {args.out}]")
        else:
            plt.show()


if __name__ == "__main__":
    main()
