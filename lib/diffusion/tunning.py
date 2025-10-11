import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/..")
from geom.surfaces import bspline_surface_mesh_from_ctrl
import threading
import numpy as np
from scipy.interpolate import griddata
from .config import BGPatternDiffuserConfig
from geom.surfaces import infer_grid
import collections

def log_point_update(args, i, oldz, newz, lr, dJdZ, label="dJdZ"):
    if args.verbosity == "full":
        if oldz is None:
            print(f"[idx={i}] Δz = lr * {label}  -->  + {lr:.3e} * {dJdZ:.3e}   (new z[{i}]={newz:.6f})")
        else:
            print(f"[idx={i}] new val = old val + lr * {label}  -->  {newz:.6f} = {oldz:.6f} + {lr:.3e} * {dJdZ:.3e}")

# Main function to extract adjacent rows with subgrid size and new index
def extract_adjacent_nodes(flattened_index, grid_width, grid_height, original_array):
    # Get the row and column of the node from the flattened index
    row, col = divmod(flattened_index, grid_width)
    # Determine the subgrid size based on the position of the node
    if row == 0 and col == 0:
        subgrid_width, subgrid_height = 2, 2  # Top-left corner
    elif row == 0 and col == grid_width - 1:
        subgrid_width, subgrid_height = 2, 2  # Top-right corner
    elif row == grid_height - 1 and col == 0:
        subgrid_width, subgrid_height = 2, 2  # Bottom-left corner
    elif row == grid_height - 1 and col == grid_width - 1:
        subgrid_width, subgrid_height = 2, 2  # Bottom-right corner
    elif row == 0 or row == grid_height - 1:
        subgrid_width, subgrid_height = 3, 2  # Top or bottom edge
    elif col == 0 or col == grid_width - 1:
        subgrid_width, subgrid_height = 2, 3  # Left or right edge
    else:
        subgrid_width, subgrid_height = 3, 3  # Internal point

    # Get the list of flattened indices for the node and its adjacencies
    indices = extract_adjacent_indices(flattened_index, grid_width, grid_height)

    # Extract the rows from the original array corresponding to these indices
    extracted_rows = original_array[indices]

    # Find the new index of the original node in the extracted subgrid
    new_index_in_subgrid = indices.index(flattened_index)

    return extracted_rows, (subgrid_width, subgrid_height), new_index_in_subgrid

def extract_adjacent_indices(flattened_index, grid_width, grid_height):
    row, col = divmod(flattened_index, grid_width)
    adjacent_indices = [flattened_index]  # Start with the original node itself
    # Loop through the possible adjacent positions (no diagonals, just the 4 immediate neighbors)
    for dr in [-1, 0, 1]:  # row offsets
        for dc in [-1, 0, 1]:  # column offsets
            # Skip the central node itself
            if dr == 0 and dc == 0:
                continue
            # Calculate the adjacent node's row and column
            adj_row = row + dr
            adj_col = col + dc
            # Check if the adjacent node is within bounds
            if 0 <= adj_row < grid_height and 0 <= adj_col < grid_width:
                # Convert to flattened index and add to the list
                adj_flat_index = adj_row * grid_width + adj_col
                adjacent_indices.append(adj_flat_index)

    return adjacent_indices


def get_submesh_on_index(ind, gw, gh, ctrl_pts, su, sv):
    nei_p, dims, k = extract_adjacent_nodes(25, gw, gh, ctrl_pts)
    target_grid_size = 10
    expanded_ctrl_pts = interpolate_ctrl_points_to_larger_grid(nei_p, target_grid_size)
    return bspline_surface_mesh_from_ctrl(expanded_ctrl_pts, target_grid_size, target_grid_size, su=su, sv=sv)


def interpolate_ctrl_points_to_larger_grid(ctrl_pts: np.ndarray, target_grid_size: int) -> np.ndarray:
    """Interpolate a smaller grid of control points (e.g., 3x3) to a larger grid (e.g., 10x10)."""
    # Define the size of the grid
    x = ctrl_pts[:, 0]
    y = ctrl_pts[:, 1]
    z = ctrl_pts[:, 2]
    
    # Create a mesh grid for the target size
    x_new = np.linspace(min(x), max(x), target_grid_size)
    y_new = np.linspace(min(y), max(y), target_grid_size)
    x_grid, y_grid = np.meshgrid(x_new, y_new)
    
    # Perform the interpolation
    z_grid = griddata((x, y), z, (x_grid, y_grid), method='cubic')

    # Stack the result into a new control points array
    return np.column_stack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()])

def strike(text: str) -> str:
    return ''.join(c + '\u0336' for c in text)

class Optimizer:
    """
    Numerical optimizer for z-values of control net.
    - Does NOT touch visualization.
    - Owns the ctrl ndarray (view into user's array or copy).
    - Methods:
        step_point(i) -> (i, oldz, newz, dJdZ, score)   # always per-point (GUI space)
        iterate_once() -> (score, z_vals)               # per-point (GUI A/O)
        iterate_once_move_all() -> (score, z_vals)      # uniform shift (GUI M)
        run_loop(max_iters, tol, ...)                   # per-point or uniform depending on --move_all (FAST only)
    """
    def __init__(self, config: BGPatternDiffuserConfig, _ctrl=None, _scorer=None, _alpha=None):
        """
        initial_guess: ndarray (N,3) - will be modified in place
        scorer: Projection3DScorer (initialized with base mesh)
        args: namespace with grid_w, grid_h, samples_u, samples_v, alpha, eps, tol, verbosity, fast...
        """
        self.config = config
        self.prevScore = None
        self.prevCtrl = None
        self.noisy = False
        if not _ctrl is None:
            assert not _scorer is None and not _alpha is None
            self.ctrl = _ctrl
            self.W, self.H = infer_grid(self.ctrl)
            self.N = _ctrl.shape[0]
            self.scorer = _scorer
            self.lr = _alpha
            print("LR is : ----> ", self.lr)
            self._score_window = collections.deque(maxlen=self.config.tunning_window)
            self._big_window_len = 10 * self.config.tunning_window
            self._score_window_big = collections.deque(maxlen=self._big_window_len)
            self.avgChangeTolBig = self.config.tunning_avgChangeTol * 10
            self.badLR = 1e6

    def resetConfirmUpdate(self):
        self.prevScore = None
        self.prev2ctrl = None
        self.prev3ctrl = None
        self.prevCtrl = None

    def confirmUpdate(self, score, ctrl):
        if self.prevScore is None or self.it < 3:
            self.prevScore = score
            self.ctrl = ctrl.copy()
            self.noisy = False
            return True
        self.noisy = score < self.prevScore
        self.prevScore = score
        if self.noisy:
            self.badLR = self.lr
            self.lr /= 32
            self.it -= 3
            self.ctrl = self.prev2ctrl
            if self.ctrl is None:
                self.ctrl = self.prev3ctrl
            self.resetConfirmUpdate()
            return False
        else:
            new_lr = 2 * self.lr
            if new_lr < self.badLR:
                self.lr = new_lr
            if self.prev2ctrl is not None:
                self.prev3ctrl = self.prev2ctrl
            if self.prevCtrl is not None:
                self.prev2ctrl = self.prevCtrl.copy()
            self.prevCtrl = self.ctrl.copy()
            self.ctrl = ctrl.copy()
            return True

    # ---- Headless run loop (FAST). --move_all toggles behavior here ONLY. ----
    def tune(self, initial_guess, scorer, iters, alpha):
        self.lr = alpha
        print("LR is : ----> ", self.lr)
        self.ctrl = initial_guess.copy()
        self.scorer = scorer
        self.N = initial_guess.shape[0]
        self.W, self.H = infer_grid(self.ctrl)
        self._score_window = collections.deque(maxlen=self.config.tunning_window)
        self._big_window_len = 10 * self.config.tunning_window
        self._score_window_big = collections.deque(maxlen=self._big_window_len)
        self.avgChangeTolBig = self.config.tunning_avgChangeTol * 10
        self.resetConfirmUpdate()
        self.badLR = 1e6
        self.it = -1
        while self.it < iters:
            self.it += 1
            if self.ctrl is not None:
                ctrl = self.ctrl.copy()
            for i in range(self.N):
                dJdZ = self.central_diff_grad(i)
                step = self.lr * dJdZ
                ctrl[i, 2] = float(self.ctrl[i, 2]) + step
            score, _ = self.scorer.score(bspline_surface_mesh_from_ctrl(
                ctrl, self.W, self.H, self.config.spline_mesh_samples_u, self.config.spline_mesh_samples_v))
            # update history window
            r = self.confirmUpdate(score=score, ctrl=ctrl)
            self._score_window.append(float(score))
            self._score_window_big.append(float(score))
            # isn, _ = self.nd.eval(score=score)
            # new: sliding-window stopping checks
            cut, var, var_big = self._should_stop()
            if cut:
                break

            if self.config.verbosity == "tiny":
                if var is not None:
                    if var_big is None:
                        text = f"internal opt it: {self.it} - sc: {score:.2f} - noisy: {self.noisy} - lr: {self.lr:.5f} - var: {var:.5f}"
                    else:
                        text = f"internal opt it: {self.it} - sc: {score:.2f} - noisy: {self.noisy} - lr: {self.lr:.5f} - var: {var:.5f} - var_big: {var_big:.4f}"
                else:
                    text = f"internal opt it: {self.it} - sc: {score:.2f} - noisy: {self.noisy} - lr: {self.lr:.5f}"

            if r:        
                print(text)
            else:
                print(strike(text))
        return self.ctrl[:, 2]

    def _should_stop(self):
        """
        Returns True if stopping criteria are met (relative to mean score in window):
        - If we don't yet have a full window, return False.
        - Compute relative average absolute score change:
            avg_rel_change = mean(abs(diff(scores))) / mean(scores)
        If avg_rel_change < self.config.tunning_avgChangeTol -> stop (converged / stagnation).
        - Compute relative variance:
            rel_var = var(scores) / mean(scores)
        If rel_var > self.config.tunning_varThresh -> stop (unstable oscillations).
        - Criteria combined with OR: if either triggers, stop.
        """
        if len(self._score_window) < self.config.tunning_window:
            return False, None, None

        scores = np.array(self._score_window, dtype=float)
        mean_score = float(np.mean(scores))

        scores_big = np.array(self._score_window_big, dtype=float)
        mean_score_big = float(np.mean(scores_big))


        ret = False
        if mean_score == 0.0:  # avoid division by zero
            return ret, None, None

        diffs = np.abs(np.diff(scores))
        avg_rel_change = (np.mean(diffs) / mean_score) if diffs.size > 0 else 0.0
        # rel_var = np.var(scores) / mean_score

        diffs_big = np.abs(np.diff(scores_big))
        avg_rel_change_big = (np.mean(diffs_big) / mean_score_big) if diffs_big.size > 0 else 0.0

        if self.config.verbosity == "full":
            print(f"_should_stop check: avg_rel_change={avg_rel_change} "
                f"avg_tol={self.config.tunning_avgChangeTol}, var_thresh={self.config.tunning_varThresh}")

        if avg_rel_change < self.config.tunning_avgChangeTol:
            if self.config.verbosity:
                print(f"Stopping because of 'avg_rel_change': {avg_rel_change} < {self.config.tunning_avgChangeTol}")
            ret = True

        if len(self._score_window_big) >= self._big_window_len:
            if avg_rel_change_big < self.avgChangeTolBig:
                if self.config.verbosity:
                    print(f"Stopping because of 'avg_rel_change_big': {avg_rel_change_big} < {self.avgChangeTolBig}")
                ret = True
        else:
            avg_rel_change_big = None

        # if rel_var > self.config.tunning_varThresh:
        #     if self.config.verbosity:
        #         print(f"Stopping because rel_var {rel_var} > {self.config.tunning_varThresh}")
        #     return True
        return ret, avg_rel_change, avg_rel_change_big

    def central_diff_grad(self, i):
        """Compute central-difference gradient of score wrt ctrl[i,2]."""
        # +eps
        ctrl_p = self.ctrl.copy()
        ctrl_p[i, 2] += self.config.tunning_eps
        mesh_p = bspline_surface_mesh_from_ctrl(
            ctrl_p, self.W, self.H, self.config.spline_mesh_samples_u, self.config.spline_mesh_samples_v
        )
        Jp, _ = self.scorer.score(mesh_p)
        ctrl_m = self.ctrl.copy()
        ctrl_m[i, 2] -= self.config.tunning_eps
        mesh_m = bspline_surface_mesh_from_ctrl(
            ctrl_m, self.W, self.H, self.config.spline_mesh_samples_u, self.config.spline_mesh_samples_v
        )
        Jm, _ = self.scorer.score(mesh_m)
        return (Jp - Jm) / (2.0 * self.config.tunning_eps)

    # ---- Per-point operations (GUI default: space / A / O) ----
    def step_point(self, i):
        """One per-point gradient step at index i."""
        oldz = float(self.ctrl[i, 2])
        if self.config.tunning_partialGrad:
            dJdZ = self.central_diff_partial_grad(i)
        else:
            dJdZ = self.central_diff_grad(i)
        step = self.lr * dJdZ
        newz = oldz + step
        self.ctrl[i, 2] = newz
        score, _ = self.scorer.score(bspline_surface_mesh_from_ctrl(
            self.ctrl, self.W, self.H, self.config.spline_mesh_samples_u, self.config.spline_mesh_samples_v))
        return i, oldz, newz, dJdZ, score

    def iterate_once(self):
        """Serial per-point update of all control points (0..N-1)."""
        for i in range(self.N):
            if self.config.tunning_partialGrad:
                dJdZ = self.central_diff_partial_grad(i)
            else:
                dJdZ = self.central_diff_grad(i)
            self.ctrl[i, 2] = float(self.ctrl[i, 2]) + self.lr * dJdZ
            if self.config.verbosity == "full":
                log_point_update(self.args, i, None, self.ctrl[i, 2], self.lr, dJdZ)
        score, _ = self.scorer.score(bspline_surface_mesh_from_ctrl(
            self.ctrl, self.W, self.H, self.config.spline_mesh_samples_u, self.config.spline_mesh_samples_v))
        return score, self.ctrl[:, 2]


