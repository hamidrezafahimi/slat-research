#!/usr/bin/env python3
"""
optim.py

Refactored: Optimizer class (numerical only) + external visualization interface.
Use:
  python3 optim.py --in pcd_file.pcd --spline_data best_ctrl.csv --grid_w 6 --grid_h 4 --viz
  python3 optim.py --in pcd_file.pcd --spline_data best_ctrl.csv --grid_w 6 --grid_h 4 --iters 10 --fast
"""
import argparse
import sys
import threading
import time
import numpy as np
import open3d as o3d

from helper import (
    Projection3DScorer,
    bspline_surface_mesh_from_ctrl,
    generate_spline,
    generate_xy_spline,
)

# ---------------- small utilities ----------------
def factor_pairs(n):
    pairs = []
    for a in range(1, int(np.sqrt(n)) + 1):
        if n % a == 0:
            pairs.append((a, n // a))
            if a != n // a:
                pairs.append((n // a, a))
    pairs.sort(key=lambda ab: (-ab[0], ab[1]))
    return pairs

def print_manual(args, N):
    if args.verbosity == "none":
        return
    print("\n=== Z-GD — Classic Viewer (no overlays) ===")
    print(" SPACE : single-step for current control point (updates one index)")
    print(" A     : run one full serial iteration (update all control points once), then stop")
    print(" O     : toggle continuous optimization loop (start/stop). One full iteration per animation frame.")
    print(" J / K : prev / next control point")
    print(" R     : reset control points")
    print(" Q     : quit")
    print("-------------------------------------------")
    print(f" grid: {args.grid_w} x {args.grid_h}   N={N}")
    print(f" alpha (lr): {args.alpha}")
    print(f" samples_u,v: {args.samples_u}, {args.samples_v}")
    print(f" show_ext: {args.show_ext}   show_pp: {args.show_pp}")
    print(f" continuous --max_iters: {args.max_iters}  --tol: {args.tol}")
    print(f" verbosity: {args.verbosity}   fast: {args.fast}")
    print("===========================================\n")

# ---------------- logging helpers ----------------
def log_point_update(args, i, oldz, newz, alpha, dJdZ):
    if args.verbosity == "full":
        print(f"[idx={i}] new val = old val + alpha * dJdZ  -->  {newz:.6f} = {oldz:.6f} + {alpha:.3e} * {dJdZ:.3e}")

def log_iter_summary(args, mode, iter_num, score):
    if args.verbosity == "none":
        return
    if args.verbosity == "tiny":
        print(f"[{mode} iter={iter_num}] score = {score:.6f}")
    else:
        print(f"[{mode} iter={iter_num}] score = {score:.6f}")

def log_z_values(args, prefix, zvals):
    if args.verbosity == "none":
        return
    if args.verbosity == "full":
        print(f"{prefix} z values:\n", np.array2string(zvals, precision=6, floatmode='fixed', separator=', '))

# ---------------- optimization core - Optimizer class ----------------
class Optimizer:
    """
    Numerical optimizer for z-values of control net.
    - Does NOT touch visualization.
    - Owns the ctrl ndarray (view into user's array or copy).
    - Methods:
        step_point(i) -> (i, oldz, newz, dJdZ, score)
        iterate_once() -> (score, z_vals)
        run_loop(max_iters, tol, callback=None, stop_event=None)
    """
    def __init__(self, ctrl, scorer, args):
        """
        ctrl: ndarray (N,3) - will be modified in place
        scorer: Projection3DScorer (initialized with base mesh)
        args: namespace with grid_w, grid_h, samples_u, samples_v, alpha, eps, tol, verbosity, fast...
        """
        self.ctrl = ctrl
        self.scorer = scorer
        self.args = args
        self.N = ctrl.shape[0]
        self.alpha = float(args.alpha)
        self.eps = float(args.eps)
        # Precompute nothing expensive in fast mode
        self.fast = bool(getattr(args, "fast", False))

    def central_diff_grad(self, i):
        """Compute central-difference gradient of score wrt ctrl[i,2]."""
        # +eps
        ctrl_p = self.ctrl.copy()
        ctrl_p[i, 2] += self.eps
        mesh_p = bspline_surface_mesh_from_ctrl(ctrl_p, self.args.grid_w, self.args.grid_h, self.args.samples_u, self.args.samples_v)
        Jp, _ = self.scorer.score(mesh_p)
        # -eps
        ctrl_m = self.ctrl.copy()
        ctrl_m[i, 2] -= self.eps
        mesh_m = bspline_surface_mesh_from_ctrl(ctrl_m, self.args.grid_w, self.args.grid_h, self.args.samples_u, self.args.samples_v)
        Jm, _ = self.scorer.score(mesh_m)
        return (Jp - Jm) / (2.0 * self.eps)

    def step_point(self, i):
        """
        Update ctrl[i,2] by one gradient step.
        Returns (i, oldz, newz, dJdZ, score)
        """
        oldz = float(self.ctrl[i, 2])
        dJdZ = self.central_diff_grad(i)
        step = self.alpha * dJdZ
        newz = oldz + step
        self.ctrl[i, 2] = newz
        # compute score of candidate mesh
        score, _ = self.scorer.score(bspline_surface_mesh_from_ctrl(
            self.ctrl, self.args.grid_w, self.args.grid_h, self.args.samples_u, self.args.samples_v))
        return i, oldz, newz, dJdZ, score

    def iterate_once(self):
        """
        Serial update of all control points (0..N-1).
        Returns (score_after, z_vals_view)
        """
        # Do all updates without returning visuals; only compute score at end
        for i in range(self.N):
            dJdZ = self.central_diff_grad(i)
            self.ctrl[i, 2] = float(self.ctrl[i, 2]) + self.alpha * dJdZ
            if self.args.verbosity == "full":
                # provide detailed per-point line if full verbosity requested
                log_point_update(self.args, i, None, self.ctrl[i,2], self.alpha, dJdZ)
        # final score
        score, _ = self.scorer.score(bspline_surface_mesh_from_ctrl(
            self.ctrl, self.args.grid_w, self.args.grid_h, self.args.samples_u, self.args.samples_v))
        return score, self.ctrl[:, 2]

    def run_loop(self, max_iters=100, tol=1e-8, callback=None, stop_event: threading.Event = None):
        """
        Run optimize iterations up to max_iters or until convergence (max absolute step < tol).
        callback(iter_idx, score, z_vals) is called after each iteration if provided.
        stop_event: optional threading.Event to break early when it's set.
        Returns (iter_count, final_score, final_z_vals)
        """
        iter_count = 0
        for it in range(max_iters):
            if stop_event is not None and stop_event.is_set():
                break
            max_step = 0.0
            # one full serial iteration
            for i in range(self.N):
                dJdZ = self.central_diff_grad(i)
                step = self.alpha * dJdZ
                self.ctrl[i, 2] = float(self.ctrl[i, 2]) + step
                if abs(step) > max_step:
                    max_step = abs(step)
            # compute score
            score, _ = self.scorer.score(bspline_surface_mesh_from_ctrl(
                self.ctrl, self.args.grid_w, self.args.grid_h, self.args.samples_u, self.args.samples_v))
            iter_count += 1
            if callback is not None:
                try:
                    callback(iter_count, score, self.ctrl[:,2])
                except Exception:
                    pass
            if max_step < tol:
                break
            print(f"run internal optimization loop iteration {it} - score: {score}")
        return iter_count, score, self.ctrl[:,2]

# ---------------- ClassicViewer / external UI ----------------
class ClassicViewer:
    def __init__(self, args, cloud_pts, ctrl_init, scorer, W, H):
        self.args = args
        self.cloud_pts = cloud_pts
        self.ctrl = ctrl_init.copy()      # the optimizer will operate on this
        self.ctrl0 = ctrl_init.copy()
        self.N = self.ctrl.shape[0]
        self.W, self.H = W, H

        self.alpha = float(args.alpha)
        self.idx = 0

        # optimization backend
        self.optimizer = Optimizer(self.ctrl, scorer, args)

        # continuous-mode
        self.running = False
        self.iter_count = 0
        self._in_full_iter = False

        # initial candidate mesh/score
        score0, mesh0, pj0 = apply_update_and_score(self.ctrl, scorer, args)
        self.score = score0
        self.mesh = mesh0
        self.pj = pj0
        self.scorer = scorer

        # initial draw geoms
        ext_mode = "black" if args.show_ext else None
        pp_mode = "score" if args.show_pp else None
        ge_p = self.scorer.draw(pj_pts=self.pj, ext_mode=ext_mode, pp_mode=pp_mode)
        self.pcd_ext = ge_p[0]
        self.pcd_pp = ge_p[1]

        # persistent geometry objects (mesh_geom and ctrl_pcd)
        self.mesh_geom = self.mesh
        try:
            if not args.fast:
                self.mesh_geom.compute_vertex_normals()
        except Exception:
            pass

        self.ctrl_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.ctrl))
        self.ctrl_pcd.paint_uniform_color([0.9, 0.2, 0.2])
        self.sel_sphere = self._make_sel_sphere(self._sel_xyz())

        # visualizer
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window("Spline Z-GD — classic", width=1280, height=900)

        # Add geometries once. Use reset_bounding_box only on first add to fit camera.
        added_any = False
        if len(self.pcd_ext.points) > 0:
            self.vis.add_geometry(self.pcd_ext, reset_bounding_box=True)
            added_any = True
        if len(self.pcd_pp.points) > 0:
            self.vis.add_geometry(self.pcd_pp, reset_bounding_box=(not added_any))
            added_any = True

        self.vis.add_geometry(self.mesh_geom, reset_bounding_box=(not added_any))
        added_any = True
        self.vis.add_geometry(self.ctrl_pcd, reset_bounding_box=False)
        self.vis.add_geometry(self.sel_sphere, reset_bounding_box=False)

        # key callbacks - bind to external optimizer backend
        self.vis.register_key_callback(ord(" "), self._cb_space)   # single-step index
        self.vis.register_key_callback(ord("A"), self._cb_full_iter)  # one full iteration
        self.vis.register_key_callback(ord("a"), self._cb_full_iter)
        self.vis.register_key_callback(ord("O"), self._cb_toggle_run)  # start/stop continuous
        self.vis.register_key_callback(ord("o"), self._cb_toggle_run)
        self.vis.register_key_callback(ord("J"), self._cb_prev)
        self.vis.register_key_callback(ord("K"), self._cb_next)
        self.vis.register_key_callback(ord("R"), self._cb_reset)
        self.vis.register_key_callback(ord("Q"), self._cb_quit)
        self.vis.register_key_callback(ord("q"), self._cb_quit)

        # animation callback: when running, perform one full iteration per call using optimizer
        self.vis.register_animation_callback(self._anim_cb)

        # camera fit once
        self._fit_camera_once()

        # run
        self.vis.run()
        self.vis.destroy_window()

    # --- callbacks ---
    def _cb_space(self, vis):
        i = self.idx
        i, oldz, newz, dJdZ, score = self.optimizer.step_point(i)
        # logging
        if self.args.verbosity != "none":
            log_point_update(self.args, i, oldz, newz, self.alpha, dJdZ)
            if self.args.verbosity == "tiny":
                print(f"[step] score = {score:.6f}")
            else:
                print(f"      score = {score:.6f}")
        # update mesh/pj for visuals
        self.score, self.mesh, self.pj = apply_update_and_score(self.ctrl, self.scorer, self.args)
        self._refresh_visuals()
        # advance index if not fixed
        self.idx = (i + 1) % self.N
        self._update_sel_marker()
        return False

    def _cb_full_iter(self, vis):
        # Prevent double-running
        if self._in_full_iter:
            if self.args.verbosity == "full":
                print("[A] Full-iteration already running; ignoring duplicate key press.")
            return False
        self._in_full_iter = True
        try:
            if self.args.verbosity != "none":
                print("[A] Single full iteration: updating ALL control points (serial order).")
            score, zvals = self.optimizer.iterate_once()
            log_iter_summary(self.args, "A", 1, score)
            log_z_values(self.args, "[A]", zvals)
            # refresh visuals once
            self.score, self.mesh, self.pj = apply_update_and_score(self.ctrl, self.scorer, self.args)
            self._refresh_visuals()
        finally:
            self._in_full_iter = False
        return False

    def _cb_toggle_run(self, vis):
        self.running = not self.running
        if self.running:
            self.iter_count = 0
            if self.args.verbosity != "none":
                print("[O] Continuous optimization STARTED (toggle O to stop). Running full iterations; one redraw per iteration.")
        else:
            if self.args.verbosity != "none":
                print("[O] Continuous optimization STOPPED by user.")
        return False

    def _cb_prev(self, vis):
        self.idx = (self.idx - 1) % self.N
        self._update_sel_marker()
        return False

    def _cb_next(self, vis):
        self.idx = (self.idx + 1) % self.N
        self._update_sel_marker()
        return False

    def _cb_reset(self, vis):
        if self.args.verbosity != "none":
            print("[R] Reset control net to initial values.")
        self.ctrl[:] = self.ctrl0
        self.optimizer.ctrl = self.ctrl  # keep optimizer view consistent
        self.score, self.mesh, self.pj = apply_update_and_score(self.ctrl, self.scorer, self.args)
        self._refresh_visuals()
        return False

    def _cb_quit(self, vis):
        self.running = False
        self.vis.close()
        return False

    # animation callback: runs once per frame; when running, do one full iteration and redraw once
    def _anim_cb(self, vis):
        if not self.running:
            return True

        if self.iter_count >= self.args.max_iters:
            self.running = False
            if self.args.verbosity != "none":
                print(f"[O] Reached max_iters ({self.args.max_iters}) iterations. Stopping.")
            return True

        # call optimizer.iterate_once() (one full serial iteration)
        score, zvals = self.optimizer.iterate_once()
        self.iter_count += 1
        log_iter_summary(self.args, "O", self.iter_count, score)
        log_z_values(self.args, f"[O iter={self.iter_count}]", zvals)
        # refresh visuals once
        self.score, self.mesh, self.pj = apply_update_and_score(self.ctrl, self.scorer, self.args)
        self._refresh_visuals()

        # convergence check based on approximate max step (we didn't return max_step, so use tol only indirectly)
        # For strict check you'd capture max step inside optimizer.iterate_once; for now rely on optimizer's internal stopping
        return True

    # visuals helpers
    def _sel_xyz(self):
        sel = self.idx
        return self.ctrl[sel].tolist()

    def _make_sel_sphere(self, xyz):
        s = o3d.geometry.TriangleMesh.create_sphere(radius=max(1e-6, 0.01 * min(self.W, self.H)))
        s.translate(xyz)
        s.compute_vertex_normals()
        s.paint_uniform_color([1.0, 0.9, 0.1])
        return s

    def _update_sel_marker(self):
        try:
            self.vis.remove_geometry(self.sel_sphere, reset_bounding_box=False)
        except Exception:
            pass
        self.sel_sphere = self._make_sel_sphere(self._sel_xyz())
        self.vis.add_geometry(self.sel_sphere, reset_bounding_box=False)

    def _refresh_visuals(self):
        # update mesh_geom in-place (preferred)
        try:
            self.mesh_geom.vertices = self.mesh.vertices
            self.mesh_geom.triangles = self.mesh.triangles
            try:
                self.mesh_geom.vertex_normals = self.mesh.vertex_normals
            except Exception:
                if not self.args.fast:
                    self.mesh_geom.compute_vertex_normals()
            self.vis.update_geometry(self.mesh_geom)
        except Exception:
            try:
                self.vis.remove_geometry(self.mesh_geom, reset_bounding_box=False)
            except Exception:
                pass
            self.mesh_geom = self.mesh
            self.vis.add_geometry(self.mesh_geom, reset_bounding_box=False)

        # control points update in place
        self.ctrl_pcd.points = o3d.utility.Vector3dVector(self.ctrl)
        self.ctrl_pcd.paint_uniform_color([0.9, 0.2, 0.2])
        try:
            self.vis.update_geometry(self.ctrl_pcd)
        except Exception:
            try:
                self.vis.remove_geometry(self.ctrl_pcd, reset_bounding_box=False)
            except Exception:
                pass
            self.vis.add_geometry(self.ctrl_pcd, reset_bounding_box=False)

        # update external & projected point clouds according to flags (in-place where possible)
        ext_mode = "black" if self.args.show_ext else None
        pp_mode = "score" if self.args.show_pp else None
        ge_p = self.scorer.draw(pj_pts=self.pj, ext_mode=ext_mode, pp_mode=pp_mode)
        new_ext, new_pp = ge_p[0], ge_p[1]

        # external
        if len(new_ext.points) > 0:
            if getattr(self, "pcd_ext", None) is None or len(self.pcd_ext.points) == 0:
                self.pcd_ext = new_ext
                self.vis.add_geometry(self.pcd_ext, reset_bounding_box=False)
            else:
                self.pcd_ext.points = new_ext.points
                self.pcd_ext.colors = new_ext.colors
                try:
                    self.vis.update_geometry(self.pcd_ext)
                except Exception:
                    try:
                        self.vis.remove_geometry(self.pcd_ext, reset_bounding_box=False)
                    except Exception:
                        pass
                    self.vis.add_geometry(self.pcd_ext, reset_bounding_box=False)
        else:
            try:
                if getattr(self, "pcd_ext", None) is not None and len(self.pcd_ext.points) > 0:
                    self.vis.remove_geometry(self.pcd_ext, reset_bounding_box=False)
                    self.pcd_ext = new_ext
            except Exception:
                self.pcd_ext = new_ext

        # projected points
        if len(new_pp.points) > 0:
            if getattr(self, "pcd_pp", None) is None or len(self.pcd_pp.points) == 0:
                self.pcd_pp = new_pp
                self.vis.add_geometry(self.pcd_pp, reset_bounding_box=False)
            else:
                self.pcd_pp.points = new_pp.points
                self.pcd_pp.colors = new_pp.colors
                try:
                    self.vis.update_geometry(self.pcd_pp)
                except Exception:
                    try:
                        self.vis.remove_geometry(self.pcd_pp, reset_bounding_box=False)
                    except Exception:
                        pass
                    self.vis.add_geometry(self.pcd_pp, reset_bounding_box=False)
        else:
            try:
                if getattr(self, "pcd_pp", None) is not None and len(self.pcd_pp.points) > 0:
                    self.vis.remove_geometry(self.pcd_pp, reset_bounding_box=False)
                    self.pcd_pp = new_pp
            except Exception:
                self.pcd_pp = new_pp

        # render (camera unchanged)
        self.vis.poll_events()
        self.vis.update_renderer()

    def _fit_camera_once(self):
        aabbs = []
        if len(self.pcd_ext.points) > 0:
            aabbs.append(self.pcd_ext.get_axis_aligned_bounding_box())
        if len(self.pcd_pp.points) > 0:
            aabbs.append(self.pcd_pp.get_axis_aligned_bounding_box())
        aabbs.append(self.mesh.get_axis_aligned_bounding_box())
        aabbs.append(self.ctrl_pcd.get_axis_aligned_bounding_box())

        mins = np.min([a.min_bound for a in aabbs], axis=0)
        maxs = np.max([a.max_bound for a in aabbs], axis=0)
        center = 0.5 * (mins + maxs)
        extent = float(np.max(maxs - mins))
        ctr = self.vis.get_view_control()
        eye = center + np.array([0.0, -3.0 * extent, 1.8 * extent])
        up = np.array([0.0, 0.0, 1.0])
        ctr.set_lookat(center)
        front = (center - eye)
        front = front / np.linalg.norm(front)
        ctr.set_front(front)
        ctr.set_up(up / np.linalg.norm(up))
        ctr.set_zoom(0.8)

# ---------------- small wrappers reused earlier ----------------
def central_diff_grad_for_z(i, ctrl, scorer, args, eps=1e-3):
    # kept for backward compatibility in some code paths (unused by new Optimizer)
    ctrl_p = ctrl.copy(); ctrl_p[i, 2] += eps
    mesh_p = bspline_surface_mesh_from_ctrl(ctrl_p, args.grid_w, args.grid_h, args.samples_u, args.samples_v)
    Jp, _ = scorer.score(mesh_p)
    ctrl_m = ctrl.copy(); ctrl_m[i, 2] -= eps
    mesh_m = bspline_surface_mesh_from_ctrl(ctrl_m, args.grid_w, args.grid_h, args.samples_u, args.samples_v)
    Jm, _ = scorer.score(mesh_m)
    return (Jp - Jm) / (2.0 * eps)

def apply_update_and_score(ctrl, scorer, args):
    mesh = bspline_surface_mesh_from_ctrl(ctrl, args.grid_w, args.grid_h, args.samples_u, args.samples_v)
    score, pj = scorer.score(mesh)
    return score, mesh, pj

# ---------------- CLI & main ----------------
def main():
    ap = argparse.ArgumentParser(description="Classic Open3D viewer + terminal-logged Z-only optimization of control points.")
    ap.add_argument("--in", dest="inp", required=True, help="Input point cloud (.pcd/.ply/.xyz/.npy/.npz)")
    ap.add_argument("--spline_data", required=True, help="CSV of control points (x,y,z)")
    ap.add_argument("--samples_u", type=int, default=40)
    ap.add_argument("--samples_v", type=int, default=40)
    ap.add_argument("--max_dz", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--alpha", type=float, default=1e-6, help="Learning rate used in: new = old + alpha * dJdZ")
    ap.add_argument("--eps", type=float, default=1e-3, help="Epsilon for central diff grad wrt z.")
    ap.add_argument("--iters", type=int, default=100, help="Headless: run N full iterations without viewer.")
    ap.add_argument("--show_ext", action="store_true", help="Draw external points (black). Default: off.")
    ap.add_argument("--show_pp", action="store_true", help="Draw projected points (score heatmap). Default: off.")
    ap.add_argument("--max_iters", type=int, default=100, help="Max full iterations in continuous mode.")
    ap.add_argument("--tol", type=float, default=1e-8, help="Stopping threshold for continuous mode on max |alpha*dJdZ|.")
    ap.add_argument("--verbosity", choices=["full", "tiny", "none"], default="full",
                    help="Terminal verbosity: full = detailed, tiny = only score+iter, none = silent")
    ap.add_argument("--fast", action="store_true", help="Fast headless mode: no viz, minimal logging, skip heavy ops.")
    args = ap.parse_args()

    # load cloud
    cloud_pts = None
    orig_cols = None
    if args.inp.endswith(".npy"):
        arr = np.load(args.inp)
        cloud_pts = np.asarray(arr, dtype=float)
    elif args.inp.endswith(".npz"):
        d = np.load(args.inp)
        cloud_pts = np.asarray(d["points"], dtype=float)
        if "colors" in d:
            orig_cols = np.asarray(d["colors"], dtype=float)
    else:
        pcd = o3d.io.read_point_cloud(args.inp)
        if pcd.is_empty():
            raise SystemExit("[ERROR] Empty point cloud.")
        cloud_pts = np.asarray(pcd.points, dtype=float)
        if pcd.has_colors():
            orig_cols = np.asarray(pcd.colors, dtype=float)

    # load control net
    ctrl_pts = np.loadtxt(args.spline_data, delimiter=",", dtype=float)
    if ctrl_pts.ndim != 2 or ctrl_pts.shape[1] != 3:
        raise SystemExit("[ERROR] --spline_data must be (N,3) CSV of x,y,z.")
    # maybe_autoset_grid(args, ctrl_pts)
    # automatically infer grid size from x,y of ctrl points
    N = ctrl_pts.shape[0]
    def infer_grid(ctrl_pts):
        # sort unique x and y values
        xs = np.unique(np.round(ctrl_pts[:,0], 8))
        ys = np.unique(np.round(ctrl_pts[:,1], 8))
        return len(xs), len(ys)
    args.grid_w, args.grid_h = infer_grid(ctrl_pts)
    if args.verbosity != "none":
        print(f"[auto] grid inferred from spline CSV: grid_w={args.grid_w} grid_h={args.grid_h} N={N}")

    # create centered-XY base mesh for scorer (never reset later)
    base_mesh, base_ctrl_grid, W, H, center_xy, z0 = generate_xy_spline(
        cloud_pts, args.grid_w, args.grid_h, args.samples_u, args.samples_v, margin=0.02
    )

    scorer = Projection3DScorer(
        cloud_pts, base_mesh,
        kmin_neighbors=8, neighbor_cap=64,
        max_delta_z=args.max_dz, tau=args.tau,
        original_colors=orig_cols
    )

    if args.fast:
        optimizer = Optimizer(ctrl_pts.copy(), scorer, args)
        iters_done, final_score, final_z = optimizer.run_loop(max_iters=args.iters, tol=args.tol, callback=None)
        if args.verbosity != "none":
            print(f"[fast] done iters={iters_done} score={final_score:.6f}")
        print("Z vals for given ctrl: ")
        print(final_z)
        return
    else:
        print_manual(args, len(ctrl_pts))
        ClassicViewer(args, cloud_pts, ctrl_pts, scorer, W, H)


if __name__ == "__main__":
    main()
