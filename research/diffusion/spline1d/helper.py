#!/usr/bin/env python3
"""
Helper library for:
- randomized natural cubic spline generation
- noisy dataset synthesis (per-region bias/noise)
- CSV IO
- proximity scoring (strict same-x rule)
- heatmap computation and plotting with white background
"""

from __future__ import annotations
import csv, math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import UnivariateSpline

ArrayLike = Union[np.ndarray, pd.Series, list, tuple]


# ---------------------------
# Natural cubic spline
# ---------------------------
class NaturalCubicSpline:
    """Minimal natural cubic spline (second-deriv = 0 at ends)."""
    def __init__(self, x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        if np.any(np.diff(x) <= 0): raise ValueError("x must be strictly increasing")
        if x.shape != y.shape: raise ValueError("x and y must match")
        self.x, self.y = x, y
        self.h = np.diff(x)
        self.m = self._m()

    def _m(self):
        n = len(self.x); h = self.h; y = self.y
        A = np.zeros((n, n)); r = np.zeros(n)
        A[0, 0] = A[-1, -1] = 1.0
        for i in range(1, n - 1):
            A[i, i - 1] = h[i - 1] / 6
            A[i, i]     = (h[i - 1] + h[i]) / 3
            A[i, i + 1] = h[i] / 6
            r[i] = (y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1]
        return np.linalg.solve(A, r)

    def __call__(self, xq):
        xq = np.asarray(xq, float)
        xk, yk, m, h = self.x, self.y, self.m, self.h
        xq = np.clip(xq, xk[0], xk[-1])
        idx = np.searchsorted(xk, xq, side='right') - 1
        idx = np.clip(idx, 0, len(h) - 1)
        xi, xip1 = xk[idx], xk[idx + 1]
        hi       = h[idx]
        yi, yip1 = yk[idx], yk[idx + 1]
        mi, mip1 = m[idx], m[idx + 1]
        return (mi*(xip1-xq)**3 + mip1*(xq-xi)**3)/(6*hi) + \
               ((yi-mi*hi**2/6)*(xip1-xq) + (yip1-mip1*hi**2/6)*(xq-xi))/hi


# ---------------------------
# Data generation
# ---------------------------
def build_random_controls(ctrl_points: int, rng: np.random.Generator,
                          random_ctrl_x: bool) -> Tuple[np.ndarray, np.ndarray]:
    if ctrl_points < 3:
        raise ValueError("--ctrl_points must be >= 3")
    if random_ctrl_x and ctrl_points > 2:
        xin = np.sort(rng.uniform(0.0 + 1e-6, 100.0 - 1e-6, size=ctrl_points - 2))
        x_ctrl = np.concatenate(([0.0], xin, [100.0]))
    else:
        x_ctrl = np.linspace(0.0, 100.0, ctrl_points)

    y_ctrl = rng.normal(loc=0.0, scale=2.0, size=ctrl_points)
    y_ctrl[0] = 0.0
    y_ctrl[-1] = 1.0
    return x_ctrl, y_ctrl


def generate_dataset(n: int, sigma: float, bmax: float,
                     num_regions: int, num_noisy: int,
                     ctrl_points: int, random_ctrl_x: bool,
                     seed: Optional[int] = None):
    """Return dict with x, y_clean, y_noisy, region info, and controls."""
    rng = np.random.default_rng(seed)
    x_ctrl, y_ctrl = build_random_controls(ctrl_points, rng, random_ctrl_x)
    spl = NaturalCubicSpline(x_ctrl, y_ctrl)

    x = np.linspace(0.0, 100.0, n)
    y_clean = spl(x)

    # Regions via Dirichlet over lengths
    num_regions = max(1, int(num_regions))
    raw_lengths = rng.dirichlet(alpha=np.ones(num_regions))
    edges = np.concatenate(([0.0], np.cumsum(100.0 * raw_lengths)))
    edges[-1] = 100.0

    # Region id for each x
    region_ids = np.digitize(x, edges, right=False) - 1
    region_ids = np.clip(region_ids, 0, num_regions - 1)

    # Choose noisy regions
    num_noisy = max(0, min(int(num_noisy), num_regions))
    noisy_regions = np.sort(rng.choice(num_regions, size=num_noisy, replace=False)) if num_noisy > 0 \
                    else np.array([], dtype=int)

    # Per-region bias + noise
    bias = np.zeros_like(x)
    noise = np.zeros_like(x)
    for r in noisy_regions:
        b_r = rng.uniform(-bmax, bmax)
        idx = (region_ids == r)
        bias[idx] = b_r
        sigma_r = rng.uniform(0.5, 1.5) * sigma
        noise[idx] = rng.normal(0.0, sigma_r, size=idx.sum())

    y_noisy = y_clean + bias + noise

    return {
        "x": x, "y_clean": y_clean, "y_noisy": y_noisy,
        "edges": edges, "region_ids": region_ids,
        "x_ctrl": x_ctrl, "y_ctrl": y_ctrl,
    }


def save_xy_csv(path: str, x: np.ndarray, y: np.ndarray, x_name="x", y_name="y"):
    pd.DataFrame({x_name: x, y_name: y}).to_csv(path, index=False)


# ---------------------------
# CSV reader (x,y)
# ---------------------------
def read_xy_csv(path: str, x_col: int = 0, y_col: int = 1,
                has_header: Optional[bool] = None, delimiter: str = ",") -> np.ndarray:
    rows: List[Tuple[float, float]] = []
    with open(path, "r", newline="") as f:
        sniff = csv.Sniffer()
        sample = f.read(2048)
        f.seek(0)
        if has_header is None:
            has_header = sniff.has_header(sample)
        reader = csv.reader(f, delimiter=delimiter)
        if has_header:
            next(reader, None)
        for r in reader:
            if not r:
                continue
            try:
                rows.append((float(r[x_col]), float(r[y_col])))
            except (ValueError, IndexError):
                pass
    arr = np.asarray(rows, dtype=np.float64)
    if arr.size == 0:
        raise ValueError(f"No valid (x,y) rows in {path}")
    return arr

@dataclass
class ProximityScoreCalculator:
    data: np.ndarray                     # shape (N, 2): columns (x, y)
    max_dist: float = 1.0
    half_life: Optional[float] = None    # distance half-life for proximity decay

    # --- smoothness control (new) ---
    k_neighbors: int = 1                 # locality: use rolling median over ±k of |d2y/dx2|
    rough_half_life: Optional[float] = None  # half-life for curvature magnitude
    smooth_power: float = 0.7            # soften effect: w_smooth ** smooth_power  (0.5..0.9)
    smooth_blend: float = 0.7            # blend with neutral: alpha*w + (1-alpha)*1
    eps: float = 1e-12

    def __post_init__(self):
        self.max_dist = float(self.max_dist)
        self.half_life = float(self.half_life) if self.half_life else (self.max_dist / 3.0)

        self._x = self.data[:, 0].astype(float, copy=False)
        self._y = self.data[:, 1].astype(float, copy=False)

        # Map each x -> indices into self.data having that exact x
        self._map: Dict[float, List[int]] = {}
        for i, xv in enumerate(self._x):
            self._map.setdefault(xv, []).append(i)

        # Precompute per-x smoothness weights (LOCAL window)
        self._w_smooth: Dict[float, float] = self._build_smoothness_weights_local()

    # --- helpers ---

    @staticmethod
    def _exp_half_life(value: np.ndarray, half_life: float) -> np.ndarray:
        return np.exp(np.log(0.5) * (value / max(half_life, 1e-12)))

    def _score1d_vec(self, d: np.ndarray) -> np.ndarray:
        out = self._exp_half_life(d, self.half_life)
        out[d >= self.max_dist] = 0.0
        return out

    @staticmethod
    def _local_second_derivative(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """
        Local 2nd derivative for non-uniform xs using nearest neighbors:
        For interior i: d2y ≈ 2 * [ (y_{i+1}-y_i)/(x_{i+1}-x_i) - (y_i - y_{i-1})/(x_i - x_{i-1}) ] / (x_{i+1}-x_{i-1})
        Ends use forward/backward 3-pt stencils.
        """
        n = xs.size
        d2 = np.zeros(n, dtype=float)
        if n < 3:
            return d2

        # interior
        for i in range(1, n-1):
            dxp = xs[i+1] - xs[i]
            dxm = xs[i]   - xs[i-1]
            denom = (xs[i+1] - xs[i-1])
            if dxp <= 0 or dxm <= 0 or denom == 0:
                d2[i] = 0.0
                continue
            v1 = (ys[i+1] - ys[i]) / dxp
            v0 = (ys[i]   - ys[i-1]) / dxm
            d2[i] = 2.0 * (v1 - v0) / denom

        # endpoints: simple 3-pt one-sided (non-uniform aware via finite differences)
        # left end i=0
        dx1 = xs[1] - xs[0]
        dx2 = xs[2] - xs[1] if n > 2 else dx1
        if dx1 > 0 and dx2 > 0:
            # use quadratic fit via three points (0,1,2)
            x0,x1,x2 = xs[0], xs[1], xs[2] if n>2 else (xs[0], xs[1], xs[1]+dx1)
            y0,y1,y2 = ys[0], ys[1], ys[2] if n>2 else (ys[0], ys[1], ys[1])
            # Lagrange second derivative at x0
            denom01 = (x0-x1)*(x0-x2)
            denom11 = (x1-x0)*(x1-x2)
            denom21 = (x2-x0)*(x2-x1)
            if denom01 != 0 and denom11 != 0 and denom21 != 0:
                # d2/dx2 of Lagrange basis at x0 equals 2 * sum(yj * coeff_j) evaluated at x0
                # but simpler: compute quadratic coeffs via np.polyfit on (x0,x1,x2)
                coeffs = np.polyfit(np.array([x0,x1,x2]), np.array([y0,y1,y2]), 2)
                d2[0] = 2.0 * coeffs[0]
        # right end i=n-1
        dx1 = xs[-1] - xs[-2]
        dx2 = xs[-2] - xs[-3] if n > 2 else dx1
        if dx1 > 0 and dx2 > 0:
            x0,x1,x2 = xs[-3], xs[-2], xs[-1] if n>2 else (xs[-2]-dx1, xs[-2], xs[-1])
            y0,y1,y2 = ys[-3], ys[-2], ys[-1] if n>2 else (ys[-2], ys[-2], ys[-1])
            coeffs = np.polyfit(np.array([x0,x1,x2]), np.array([y0,y1,y2]), 2)
            d2[-1] = 2.0 * coeffs[0]

        return d2

    def _build_smoothness_weights_local(self) -> Dict[float, float]:
        xs_unique = np.array(sorted(self._map.keys()), dtype=float)
        if xs_unique.size < 3:
            return {float(x): 1.0 for x in xs_unique}

        # use median to be robust to multiple y per x
        y_center = np.array([np.median(self._y[self._map[x]]) for x in xs_unique], dtype=float)

        # Highly local curvature
        d2 = np.abs(self._local_second_derivative(xs_unique, y_center))

        # optionally widen slightly via rolling median over ±k (small, default 1)
        if self.k_neighbors > 0:
            k = int(self.k_neighbors)
            pad = np.pad(d2, (k, k), mode="edge")
            win = 2*k + 1
            # rolling median
            d2_sm = np.array([np.median(pad[i:i+win]) for i in range(len(pad)-win+1)], dtype=float)
            d2 = d2_sm

        # choose half-life (auto if None), scaled up to weaken effect
        if self.rough_half_life is None:
            med = np.median(d2)
            q75 = np.percentile(d2, 75.0)
            base = q75 if med < 10 * self.eps else med
            rough_hl = float(max(base, 1e-6)) * 2.0   # <- enlarge tolerance
        else:
            rough_hl = float(self.rough_half_life)

        # base weight
        w = self._exp_half_life(d2, rough_hl)
        w = np.clip(w, 0.0, 1.0)

        # soften: power + blend with 1.0
        w = (w ** float(self.smooth_power))
        alpha = float(self.smooth_blend)
        w = alpha * w + (1.0 - alpha) * 1.0

        return {float(x): float(wi) for x, wi in zip(xs_unique, w)}

    # --- public API ---
    def score_column(self, x: float, y_grid: np.ndarray) -> np.ndarray:
        idxs = self._map.get(x)
        if idxs is None:
            return np.zeros_like(y_grid, dtype=float)

        # Proximity as before
        yvals = self._y[idxs]
        d = np.abs(y_grid[:, None] - yvals[None, :]).min(axis=1)
        prox = self._score1d_vec(d)

        # Local smoothness multiplier (tamed)
        w_s = self._w_smooth.get(float(x), 1.0)
        return prox * w_s

    def score_spline_points(self, ctrl_xy: np.ndarray) -> float:
        """
        Build a NaturalCubicSpline from ctrl_xy and score it by evaluating
        the spline at the data's available x-keys (strict same-x rule),
        then summing proximity scores across those columns.
        """
        # Build spline from control points (x must be strictly increasing)
        x_ctrl = np.asarray(ctrl_xy[:, 0], dtype=float)
        y_ctrl = np.asarray(ctrl_xy[:, 1], dtype=float)
        spline = NaturalCubicSpline(x_ctrl, y_ctrl)

        # Only evaluate where the scorer has data columns (exact x matches)
        xs_keys = np.array(sorted(self._map.keys()), dtype=float)
        xmin, xmax = float(x_ctrl.min()), float(x_ctrl.max())
        mask = (xs_keys >= xmin) & (xs_keys <= xmax)
        xs_eval = xs_keys[mask]
        if xs_eval.size == 0:
            return 0.0

        # Evaluate spline at those x's, then accumulate per-column scores
        ys_eval = spline(xs_eval)  # vectorized eval
        total = 0.0
        for xi, yi in zip(xs_eval, ys_eval):
            s = self.score_column(float(xi), np.array([float(yi)], dtype=float))
            total += float(s[0])
        return float(total)


# ---------------------------
# Heatmap (white background)
# ---------------------------
def compute_heatmap(arr_xy: np.ndarray, max_dist: float, half_life: Optional[float],
                    res_y: int, sample_x: Optional[int] = None,
                    blur_cols: int = 0):
    """Return xs, ys, heat_masked (NaN where score==0)."""
    psc = ProximityScoreCalculator(arr_xy, max_dist=max_dist, half_life=half_life)

    uniq_x = np.unique(arr_xy[:, 0])
    if sample_x and sample_x < uniq_x.size:
        idx = np.linspace(0, uniq_x.size - 1, sample_x, dtype=int)
        xs = uniq_x[idx]
    else:
        xs = uniq_x

    y_min = arr_xy[:, 1].min() - max_dist
    y_max = arr_xy[:, 1].max() + max_dist
    ys = np.linspace(y_min, y_max, int(res_y))

    heat = np.zeros((ys.size, xs.size), dtype=np.float32)
    for j, x in enumerate(xs):
        heat[:, j] = psc.score_column(x, ys)

    if blur_cols and blur_cols > 1:
        heat = _box_blur_cols(heat, blur_cols)

    heat_masked = np.where(heat > 0, heat, np.nan)
    return xs, ys, heat_masked


def _box_blur_cols(img: np.ndarray, k: int) -> np.ndarray:
    """simple horizontal box blur (odd kernel)"""
    k = max(1, int(k))
    if k <= 1 or img.shape[1] <= 1:
        return img
    if k % 2 == 0:
        k += 1
    pad = k // 2
    padded = np.pad(img, ((0, 0), (pad, pad)), mode="edge")
    csum = np.cumsum(padded, axis=1, dtype=np.float64)
    win_sum = csum[:, k:] - csum[:, :-k]
    return (win_sum / float(k)).astype(img.dtype)

@dataclass
class RandNeighborSplineGen:
    # Fit a smoothing spline to (x,y), compute constant parallel envelopes
    # based on residual min/max, then generate a random in-between spline
    # with exactly K_ctrl control points.
    #
    # Parameters
    # ----------
    # k         : spline degree (for fitted spline and control-point spline)
    # K_ctrl    : number of control points for the random spline
    # smooth    : "auto" (two-pass residual estimate) or float for scipy's s
    # alpha_*   : range of random blending factors in [0,1]
    # seed      : RNG seed
    k: int = 3
    K_ctrl: int = 6
    smooth: Union[str, float] = "auto"
    alpha_low: float = 0.2
    alpha_high: float = 0.8
    seed: Optional[int] = None

    fitted_spline_: Optional[UnivariateSpline] = None
    rand_spline_: Optional[UnivariateSpline] = None
    bounds_: Optional[Tuple[float, float]] = None  # (lb, ub)

    # ------------- internals -------------
    def _ensure_xy(self, x: ArrayLike, y: ArrayLike):
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        if x.shape != y.shape:
            raise ValueError("x and y must have the same shape")
        order = np.argsort(x)
        return x[order], y[order]

    def _fit_smoothing(self, x: np.ndarray, y: np.ndarray) -> UnivariateSpline:
        N = len(x)
        if self.smooth == "auto":
            spl0 = UnivariateSpline(x, y, k=self.k, s=max(N, 1))
            resid = y - spl0(x)
            sigma = float(np.std(resid))
            s_final = max(N * (sigma**2), 1e-12)
        else:
            s_final = float(self.smooth)
        return UnivariateSpline(x, y, k=self.k, s=s_final)

    def _compute_bounds(self, x: np.ndarray, y: np.ndarray, spl: UnivariateSpline):
        r = y - spl(x)
        lb = float(np.min(r))  # ≤ 0
        ub = float(np.max(r))  # ≥ 0
        return lb, ub

    def _make_random_ctrl(self, x: np.ndarray, lb: float, ub: float, spl: UnivariateSpline):
        rng = np.random.default_rng(self.seed)
        xmin, xmax = float(np.min(x)), float(np.max(x))
        xk = np.linspace(xmin, xmax, self.K_ctrl)
        fxk = spl(xk)
        low = fxk + lb
        high = fxk + ub
        alphas = rng.uniform(self.alpha_low, self.alpha_high, size=self.K_ctrl)
        yk = low + alphas * (high - low)
        return xk, yk

    # ------------- public API -------------
    def __call__(
        self,
        x: ArrayLike,
        y: ArrayLike,
        save_ctrl: Optional[str] = None,
        save_bounds: Optional[str] = None,
        save_curves: Optional[str] = None,
        visualize: bool = False
    ) -> Dict[str, object]:
        # run the generator pipeline
        xs, ys = self._ensure_xy(x, y)

        fitted = self._fit_smoothing(xs, ys)
        self.fitted_spline_ = fitted

        lb, ub = self._compute_bounds(xs, ys, fitted)
        self.bounds_ = (lb, ub)

        ctrl_x, ctrl_y = self._make_random_ctrl(xs, lb, ub, fitted)
        rand_spline = UnivariateSpline(ctrl_x, ctrl_y, k=self.k, s=0.0)
        self.rand_spline_ = rand_spline

        # dense evaluation for convenience
        xx = np.linspace(xs.min(), xs.max(), 2000)
        y_fit = fitted(xx)
        y_lower = y_fit + lb
        y_upper = y_fit + ub
        y_rand = rand_spline(xx)

        # optional saves
        if save_ctrl:
            pd.DataFrame({"x": ctrl_x, "y": ctrl_y}).to_csv(save_ctrl, index=False)
        if save_bounds:
            import json
            with open(save_bounds, "w") as f:
                json.dump({"low_bound": lb, "up_bound": ub}, f, indent=2)
        if save_curves:
            pd.DataFrame({"x": xx, "y_fit": y_fit, "lower": y_lower,
                          "upper": y_upper, "y_rand": y_rand}).to_csv(save_curves, index=False)

        # optional visualization
        if visualize:
            import matplotlib.pyplot as plt  # matplotlib only
            plt.figure(figsize=(9, 5.5))
            plt.scatter(xs, ys, s=5, alpha=0.5, label="data")
            plt.plot(xx, y_fit, linewidth=2, label="fitted spline")
            plt.plot(xx, y_lower, linewidth=1.5, label="lower bound")
            plt.plot(xx, y_upper, linewidth=1.5, label="upper bound")
            plt.plot(xx, y_rand, linewidth=2, linestyle="--",
                     label=f"random spline ({self.K_ctrl} ctrl pts)")
            plt.scatter(ctrl_x, ctrl_y, s=40, marker="x", label="control points")
            plt.title("RandNeighborSplineGen result")
            plt.xlabel("x"); plt.ylabel("y")
            plt.legend(loc="best")
            plt.tight_layout()
            plt.show()

        return {
            "x_sorted": xs, "y_sorted": ys,
            "lb": lb, "ub": ub,
            "fitted_spline": fitted,
            "rand_spline": rand_spline,
            "ctrl_x": ctrl_x, "ctrl_y": ctrl_y,
            "xx": xx, "y_fit": y_fit, "y_lower": y_lower,
            "y_upper": y_upper, "y_rand": y_rand,
        }
    
    def generate(
        self,
        x: ArrayLike,
        y: ArrayLike,
        n: int = 10,
        visualize: bool = False
    ) -> list[Dict[str, object]]:
        """
        Generate `n` random spline candidates on the same (x,y) data.
        Each run jitters the RNG seed to ensure variation.

        Parameters
        ----------
        x, y : arrays of data
        n : number of random splines to return
        visualize : whether to plot each candidate (default False)

        Returns
        -------
        candidates : list of dicts, each exactly what __call__ returns
        """
        results = []
        base_seed = self.seed if self.seed is not None else 0
        for i in range(n):
            # re-seed for reproducibility but ensure each run differs
            self.seed = base_seed + i
            cand = self.__call__(x, y, visualize=visualize)
            results.append(cand)
        # restore original seed
        self.seed = base_seed if self.seed is not None else None
        return results

import os

def build_pred_spline_xy(ctrl, x_min=None, x_max=None, n=1200):
    """
    Build a dense (x,y) spline from control points.

    ctrl: either
      - str / os.PathLike -> CSV path with control points (x,y), x strictly increasing
      - array-like         -> shape (N,2) control points (x,y)
    """
    # Accept path or array
    if isinstance(ctrl, (str, os.PathLike)):
        ctrl_xy = read_xy_csv(ctrl)
    else:
        ctrl_xy = np.asarray(ctrl, dtype=float)
        if ctrl_xy.ndim != 2 or ctrl_xy.shape[1] != 2:
            raise ValueError(f"ctrl must be Nx2; got {ctrl_xy.shape}")
        
    x_ctrl, y_ctrl = ctrl_xy[:, 0], ctrl_xy[:, 1]
    spline = NaturalCubicSpline(x_ctrl, y_ctrl)

    if x_min is None:
        x_min = float(x_ctrl.min())
    if x_max is None:
        x_max = float(x_ctrl.max())

    xs = np.linspace(x_min, x_max, int(n))
    ys = spline(xs)
    return xs, ys

def load_gt_dense_xy(gt_csv: str, x_min: float | None = None, x_max: float | None = None):
    """
    Load GT raw (dense) spline points directly (no fitting).
    Ensures sorted by x; optionally crops to [x_min, x_max].
    """
    arr = read_xy_csv(gt_csv)  # (M,2): x,y
    # sort by x
    order = np.argsort(arr[:, 0])
    arr = arr[order]
    if x_min is not None:
        arr = arr[arr[:, 0] >= x_min]
    if x_max is not None:
        arr = arr[arr[:, 0] <= x_max]
    return arr[:, 0], arr[:, 1]

def plot_heatmap(xs, ys, heat_masked, noisy_xy=None, ax=None, cbar_label="Proximity Score"):
    """
    Draw a 2D heatmap (masked regions shown white).
    Returns the Axes used; does not finalize the figure.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6))

    cmap = plt.cm.viridis.copy()
    try:
        cmap.set_bad(color="white")
    except Exception:
        pass

    im = ax.imshow(
        heat_masked,
        extent=[xs.min(), xs.max(), ys.min(), ys.max()],
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=0.0, vmax=1.0,
        interpolation="nearest",
    )
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    if noisy_xy is not None:
        ax.scatter(noisy_xy[:, 0], noisy_xy[:, 1], c="red", s=6, label="Noisy curve")
    return ax
