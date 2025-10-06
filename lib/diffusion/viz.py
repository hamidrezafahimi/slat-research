from .tunning import Optimizer, log_point_update
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/..")
from geom.surfaces import bspline_surface_mesh_from_ctrl
import open3d as o3d
import numpy as np

def apply_update_and_score(ctrl, scorer, W, H, cfg):
    mesh = bspline_surface_mesh_from_ctrl(ctrl, W, H, cfg.spline_mesh_samples_u, 
                                          cfg.spline_mesh_samples_v)
    score, pj = scorer.score(mesh)
    return score, mesh, pj

def log_iter_summary(cfg, mode, iter_num, score):
    if cfg.verbosity == "none":
        return
    if cfg.verbosity == "tiny":
        print(f"[{mode} iter={iter_num}] score = {score:.6f}")
    else:
        print(f"[{mode} iter={iter_num}] score = {score:.6f}")

def log_z_values(cfg, prefix, zvals):
    if cfg.verbosity == "none":
        return
    if cfg.verbosity == "full":
        print(f"{prefix} z values:\n", np.array2string(zvals, precision=6, floatmode='fixed', separator=', '))


class ClassicViewer:
    def __init__(self, config, cloud_pts, ctrl_init, scorer, W, H, _alpha, iters):
        self.config = config
        self.cloud_pts = cloud_pts
        self.ctrl = ctrl_init.copy()      # the optimizer will operate on this
        self.ctrl0 = ctrl_init.copy()
        self.N = self.ctrl.shape[0]
        self.W = W
        self.H = H
        self.idx = 0
        self.iters = iters

        # optimization backend
        self.optimizer = Optimizer(_ctrl=self.ctrl, _scorer=scorer, config=config, _alpha=_alpha)

        # continuous-mode
        self.running = False
        self.iter_count = 0
        self._in_full_iter = False

        # initial candidate mesh/score
        score0, mesh0, pj0 = apply_update_and_score(self.ctrl, scorer, self.W, self.H, self.config)
        self.score = score0
        self.mesh = mesh0
        self.pj = pj0
        self.scorer = scorer

        # initial draw geoms
        ext_mode = "smoothness"
        pp_mode = "score"
        ge_p = self.scorer.draw(pj_pts=self.pj, ext_mode=ext_mode, pp_mode=pp_mode)
        self.pcd_ext = ge_p[0]
        self.pcd_pp = ge_p[1]

        # persistent geometry objects (mesh_geom and ctrl_pcd)
        self.mesh_geom = self.mesh
        try:
            if not self.config.fast:
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

        # key callbacks - GUI behavior is independent of --move_all
        self.vis.register_key_callback(ord(" "), self._cb_space)       # single per-point step (current index)
        self.vis.register_key_callback(ord("A"), self._cb_full_iter)   # one full per-point iteration
        self.vis.register_key_callback(ord("a"), self._cb_full_iter)
        self.vis.register_key_callback(ord("M"), self._cb_move_all)    # one 'move-all' iteration (uniform shift)
        self.vis.register_key_callback(ord("m"), self._cb_move_all)
        self.vis.register_key_callback(ord("O"), self._cb_toggle_run)  # start/stop continuous per-point iterations
        self.vis.register_key_callback(ord("o"), self._cb_toggle_run)
        self.vis.register_key_callback(ord("J"), self._cb_prev)
        self.vis.register_key_callback(ord("K"), self._cb_next)
        self.vis.register_key_callback(ord("R"), self._cb_reset)
        self.vis.register_key_callback(ord("Q"), self._cb_quit)
        self.vis.register_key_callback(ord("q"), self._cb_quit)

        # animation callback: when running, perform one full per-point iteration per call
        self.vis.register_animation_callback(self._anim_cb)

        # camera fit once
        self._fit_camera_once()

        # run
        self.vis.run()
        self.vis.destroy_window()

    # --- callbacks ---
    def _cb_space(self, vis):
        i = self.idx
        i, oldz, newz, dJd_val, score = self.optimizer.step_point(i)
        if self.config.verbosity != "none":
            log_point_update(self.config, i, oldz, newz, self.optimizer.lr, dJd_val)
            if self.config.verbosity == "tiny":
                print(f"[step] score = {score:.6f}")
            else:
                print(f"      score = {score:.6f}")
        self.score, self.mesh, self.pj = apply_update_and_score(self.ctrl, self.scorer, self.W, 
                                                                self.H, self.config)
        self._refresh_visuals()
        self.idx = (i + 1) % self.N
        self._update_sel_marker()
        return False

    def _cb_full_iter(self, vis):
        if self._in_full_iter:
            if self.config.verbosity == "full":
                print("[A] Full-iteration already running; ignoring duplicate key press.")
            return False
        self._in_full_iter = True
        try:
            if self.config.verbosity != "none":
                print("[A] Single full iteration: updating ALL control points (per-point serial order).")
            score, zvals = self.optimizer.iterate_once()
            log_iter_summary(self.config, "A", 1, score)
            log_z_values(self.config, "[A]", zvals)
            self.score, self.mesh, self.pj = apply_update_and_score(self.ctrl, self.scorer, self.W, 
                                                                    self.H, self.config)
            self._refresh_visuals()
        finally:
            self._in_full_iter = False
        return False

    def _cb_move_all(self, vis):
        """One uniform 'move-all' iteration (independent of --move_all flag)."""
        if self.config.verbosity != "none":
            print("[M] Single iteration: applying a uniform Z shift to ALL control points (shape preserved).")
        score, zvals = self.optimizer.iterate_once_move_all()
        log_iter_summary(self.config, "M", 1, score)
        log_z_values(self.config, "[M]", zvals)
        self.score, self.mesh, self.pj = apply_update_and_score(self.ctrl, self.scorer, self.W, 
                                                                self.H, self.config)
        self._refresh_visuals()
        return False

    def _cb_toggle_run(self, vis):
        self.running = not self.running
        if self.running:
            self.iter_count = 0
            if self.config.verbosity != "none":
                print("[O] Continuous optimization STARTED (toggle O to stop). Running full per-point iterations; one redraw per iteration.")
        else:
            if self.config.verbosity != "none":
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
        if self.config.verbosity != "none":
            print("[R] Reset control net to initial values.")
        self.ctrl[:] = self.ctrl0
        self.optimizer.ctrl = self.ctrl  # keep optimizer view consistent
        self.score, self.mesh, self.pj = apply_update_and_score(self.ctrl, self.scorer, self.W, self.H, self.config)
        self._refresh_visuals()
        return False

    def _cb_quit(self, vis):
        self.running = False
        self.vis.close()
        return False

    # animation callback: runs once per frame; when running, do one full per-point iteration and redraw once
    def _anim_cb(self, vis):
        if not self.running:
            return True

        if self.iter_count >= self.iters:
            self.running = False
            if self.config.verbosity != "none":
                print(f"[O] Reached max_iters ({self.iters}) iterations. Stopping.")
            return True

        score, zvals = self.optimizer.iterate_once()  # per-point sweep
        self.iter_count += 1
        log_iter_summary(self.config, "O", self.iter_count, score)
        log_z_values(self.config, f"[O iter={self.iter_count}]", zvals)
        self.score, self.mesh, self.pj = apply_update_and_score(self.ctrl, self.scorer, self.W, self.H, self.config)
        self._refresh_visuals()
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
                if not self.self.config.fast:
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
        ext_mode = "smoothness" 
        pp_mode = "score" 
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
