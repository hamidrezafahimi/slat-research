#!/usr/bin/env python3
import argparse, sys
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from helper import (
    build_surface_mesh, make_o3d_pcd, make_grid_points,
    load_ctrl_points_csv, infer_grid_wh_from_points, extent_wh_from_points_xy,
    project_external_to_surface_idw, calc_loss, project_mode_dedup, project_mode_indirect,
    project_mode_original
)

class PointsApp:
    def __init__(self, ctrl_pcd, grid_w, grid_h, extra_cloud_path=None,
                 save_path="spline_ctrl.csv", step=0.2,
                 samples_u=40, samples_v=40, loss_thresh=0.2, k=3, eps=1e-9,
                 proj_method="original", xy_cell=0.02):
        self.window = gui.Application.instance.create_window("Spline Editor", 1000, 750)
        self.scene = gui.SceneWidget(); self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.window.add_child(self.scene)

        self.grid_w, self.grid_h = int(grid_w), int(grid_h)
        self.save_path = save_path
        self.step = float(step)
        self.samples_u, self.samples_v = int(samples_u), int(samples_v)
        self.loss_thresh = float(loss_thresh)
        self.k = int(k); self.eps = float(eps)

        # Geometries
        self.ctrl_pcd = ctrl_pcd
        self.points = np.asarray(self.ctrl_pcd.points)
        self.scene.scene.add_geometry("points", self.ctrl_pcd, self._mat_points(10.0))
        self.surf_mesh = build_surface_mesh(self.points, self.grid_w, self.grid_h, self.samples_u, self.samples_v)
        self.scene.scene.add_geometry("spline_surf", self.surf_mesh, self._mat_mesh())

        self.external_pcd = None
        if extra_cloud_path:
            self.external_pcd = o3d.io.read_point_cloud(extra_cloud_path)
            self.scene.scene.add_geometry("external_pcd", self.external_pcd, self._mat_points(4.0))

        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0,0,0])
        self.scene.scene.add_geometry("axis", axis, rendering.MaterialRecord())

        self.cur_r, self.cur_c = 0, 0
        self._colors = np.asarray(self.ctrl_pcd.colors)
        if self._colors.size == 0:
            self._colors = np.tile(np.array([[0.2, 0.8, 1.0]]), (self.points.shape[0], 1))
            self.ctrl_pcd.colors = o3d.utility.Vector3dVector(self._colors)
        self._highlight((self.cur_r, self.cur_c))

        self.spline_pcd = None
        self.window.set_on_key(self.on_key)
        self.k = int(k); self.eps = float(eps)
        self.proj_method = str(proj_method)
        self.xy_cell = float(xy_cell)

    # ---- Materials
    def _mat_points(self, size=4.0):
        m = rendering.MaterialRecord(); m.shader = "defaultUnlit"; m.point_size = float(size); return m
    def _mat_mesh(self):
        m = rendering.MaterialRecord(); m.shader = "defaultLit"; m.base_color=(0.7,0.7,0.9,1.0); m.base_roughness=0.8; return m

    # ---- Helpers
    def rc2idx(self, r, c): return r*self.grid_w + c
    def _refresh_points_geom(self):
        self.ctrl_pcd.points = o3d.utility.Vector3dVector(self.points)
        self.ctrl_pcd.colors = o3d.utility.Vector3dVector(self._colors)
        self.scene.scene.remove_geometry("points"); self.scene.scene.add_geometry("points", self.ctrl_pcd, self._mat_points(10.0))
    def _highlight(self, rc):
        r,c = rc
        self._colors[:] = [0.2,0.8,1.0]; self._colors[self.rc2idx(r,c)] = [1.0,0.0,0.0]
        self._refresh_points_geom()
    def _update_surface(self):
        if self.scene.scene.has_geometry("spline_surf"):
            self.scene.scene.remove_geometry("spline_surf")
        self.surf_mesh = build_surface_mesh(self.points, self.grid_w, self.grid_h, self.samples_u, self.samples_v)
        self.scene.scene.add_geometry("spline_surf", self.surf_mesh, self._mat_mesh())
        if self.scene.scene.has_geometry("spline_pcd"):
            self.scene.scene.remove_geometry("spline_pcd")
        self.spline_pcd = None

    def _project_and_render(self):
        if self.external_pcd is None:
            print("[INFO] Need external cloud to project."); return
        ext = np.asarray(self.external_pcd.points)
        if ext.size == 0:
            print("[INFO] External cloud is empty."); return

        if self.proj_method == "original":
            spline_pts, colors, loss_val = \
                project_mode_original(ext, self.surf_mesh, self.k, self.eps, self.loss_thresh)
            msg = f"[LOSS] {loss_val} (|Δz|>{self.loss_thresh:g})"

        elif self.proj_method == "indirect":
            spline_pts, colors, loss_val = \
                project_mode_indirect(ext, self.surf_mesh, self.k, self.eps, self.xy_cell, self.loss_thresh)
            msg = (f"[LOSS] {loss_val} "
                f"(indirect: {spline_pts.shape[0]} cell-centers, xy_cell={self.xy_cell:g}, "
                f"|Δz|>{self.loss_thresh:g})")

        elif self.proj_method == "dedup":
            spline_pts, colors, loss_val = \
                project_mode_dedup(ext, self.surf_mesh, self.samples_u, self.samples_v,
                                self.k, self.eps, self.loss_thresh)
            msg = f"[LOSS] {loss_val} (dedup by {self.samples_u}×{self.samples_v} XY cells)"

        else:
            print(f"[ERROR] Unknown proj_method: {self.proj_method}")
            return

        sp = o3d.geometry.PointCloud()
        sp.points = o3d.utility.Vector3dVector(spline_pts)
        if colors is None or colors.size == 0:
            colors = np.tile(np.array([[0.2, 0.9, 1.0]], dtype=float), (spline_pts.shape[0], 1))
        sp.colors = o3d.utility.Vector3dVector(colors)
        self.spline_pcd = sp

        name = "spline_pcd"
        if self.scene.scene.has_geometry(name):
            self.scene.scene.remove_geometry(name)
        self.scene.scene.add_geometry(name, self.spline_pcd, self._mat_points(2.0))
        print(msg)

    def calcLoss(self):
        if self.external_pcd is None or self.spline_pcd is None:
            return None
        ext = np.asarray(self.external_pcd.points)
        spl = np.asarray(self.spline_pcd.points)
        loss_val, mask, _ = calc_loss(ext, spl, self.loss_thresh)
        return loss_val

    # ---- Key handling (parity with old app)
    def on_key(self, e):
        if e.type != gui.KeyEvent.Type.DOWN: return False
        moved = False; r,c = self.cur_r, self.cur_c; idx = self.rc2idx(r,c)

        if e.key == gui.KeyName.LEFT:  self.cur_c = max(0, c-1); moved=True
        elif e.key == gui.KeyName.RIGHT:self.cur_c = min(self.grid_w-1, c+1); moved=True
        elif e.key == gui.KeyName.UP:   self.cur_r = min(self.grid_h-1, r+1); moved=True
        elif e.key == gui.KeyName.DOWN: self.cur_r = max(0, r-1); moved=True

        elif e.key == gui.KeyName.U: self.points[idx,2]+=self.step; self._update_surface(); self._highlight((self.cur_r,self.cur_c)); return True
        elif e.key == gui.KeyName.I: self.points[idx,2]-=self.step; self._update_surface(); self._highlight((self.cur_r,self.cur_c)); return True

        elif e.key == gui.KeyName.O:  # sync row z to active z
            z = float(self.points[idx,2]); s = self.cur_r*self.grid_w; self.points[s:s+self.grid_w,2] = z
            self._update_surface(); self._highlight((self.cur_r,self.cur_c)); return True
        elif e.key == gui.KeyName.P:  # sync col z to active z
            z = float(self.points[idx,2]); col_idx = [self.rc2idx(rr,self.cur_c) for rr in range(self.grid_h)]
            self.points[col_idx,2] = z; self._update_surface(); self._highlight((self.cur_r,self.cur_c)); return True

        elif e.key == gui.KeyName.W: self.points[:,1]+=self.step; self._update_surface(); self._highlight((self.cur_r,self.cur_c)); return True
        elif e.key == gui.KeyName.S: self.points[:,1]-=self.step; self._update_surface(); self._highlight((self.cur_r,self.cur_c)); return True
        elif e.key == gui.KeyName.A: self.points[:,0]-=self.step; self._update_surface(); self._highlight((self.cur_r,self.cur_c)); return True
        elif e.key == gui.KeyName.D: self.points[:,0]+=self.step; self._update_surface(); self._highlight((self.cur_r,self.cur_c)); return True
        elif e.key == gui.KeyName.C: self.points[:,2]-=self.step; self._update_surface(); self._highlight((self.cur_r,self.cur_c)); return True
        elif e.key == gui.KeyName.V: self.points[:,2]+=self.step; self._update_surface(); self._highlight((self.cur_r,self.cur_c)); return True

        elif e.key == gui.KeyName.M: self._project_and_render(); return True
        elif e.key == gui.KeyName.N:
            val = self.calcLoss()
            print("[INFO] Loss unavailable; need projection first." if val is None else f"[LOSS] {val}")
            return True
        elif e.key == gui.KeyName.Y:
            np.savetxt(self.save_path, self.points, fmt="%.6f", delimiter=",")
            print(f"[OK] Saved control points → {self.save_path}")
            if self.spline_pcd is not None:
                out = self.save_path.replace(".csv", "_splinepcd.pcd")
                o3d.io.write_point_cloud(out, self.spline_pcd)
                print(f"[OK] Saved spline_pcd → {out}")
            return True
        elif e.key == gui.KeyName.Q:
            gui.Application.instance.quit(); return True
        else:
            return False

        if moved: self._highlight((self.cur_r, self.cur_c))
        return True

    def run(self): gui.Application.instance.run()

def print_keymap(save_path, step, proj_method):
    s = float(step)
    print("\n=== Keymap ===")
    print("Arrows    : move selection (row/col)")
    print(f"U / I     : raise / lower selected point (±{s:.3f} z)")
    print(f"W/A/S/D   : translate grid in y/x (±{s:.3f})")
    print(f"V / C     : lift / drop grid in z (±{s:.3f})")
    print("O / P     : sync row / col z with active point")
    print(f"M         : project (mode={proj_method})")
    print("N         : print LOSS for last projection")
    print(f"Y         : save control points to '{save_path}' (also saves spline_pcd if exists)")
    print("Q         : quit\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive B-spline editor (IDW projection + loss).")
    parser.add_argument("--cloud", type=str, required=True, help="External point cloud (.pcd/.ply/.xyz)")
    parser.add_argument("--out", type=str, default="spline_ctrl.csv")
    parser.add_argument("--step", type=float, default=0.2)
    parser.add_argument("--samples_u", type=int, default=40)
    parser.add_argument("--samples_v", type=int, default=40)
    parser.add_argument("--loss_thresh", type=float, default=0.2)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--eps", type=float, default=1e-9)

    # Option A: load control points
    parser.add_argument("--spline_data", type=str, default=None, help="CSV Nx3 control points")
    # Option B: generate flat grid if no CSV
    parser.add_argument("--grid_w", type=int)
    parser.add_argument("--grid_h", type=int)
    parser.add_argument("--metric_w", type=float)
    parser.add_argument("--metric_h", type=float)
    parser.add_argument("--proj_method", type=str, default="indirect",
                    choices=["original", "indirect", "dedup"],
                    help="Projection mode for M: original | indirect | dedup")
    parser.add_argument("--xy_cell", type=float, default=0.02,
                    help="(indirect) XY grid cell size (meters) for occupancy sampling on z=0")

    args = parser.parse_args()

    if args.spline_data is not None:
        pts = load_ctrl_points_csv(args.spline_data)
        gw, gh = infer_grid_wh_from_points(pts)
        ctrl_pcd = make_o3d_pcd(pts)
        GRID_W, GRID_H = gw, gh
    else:
        missing = [n for n in ("grid_w","grid_h","metric_w","metric_h") if getattr(args, n) is None]
        if missing:
            print(f"[ERROR] Missing {missing}. Provide all or use --spline_data."); sys.exit(1)
        GRID_W, GRID_H = args.grid_w, args.grid_h
        pts = make_grid_points(GRID_W, GRID_H, args.metric_w, args.metric_h)
        ctrl_pcd = make_o3d_pcd(pts)

    gui.Application.instance.initialize()
    print_keymap(args.out, args.step, args.proj_method)
    PointsApp(ctrl_pcd, GRID_W, GRID_H,
            extra_cloud_path=args.cloud,
            save_path=args.out, step=args.step,
            samples_u=args.samples_u, samples_v=args.samples_v,
            loss_thresh=args.loss_thresh, k=args.k, eps=args.eps,
            proj_method=args.proj_method, xy_cell=args.xy_cell).run()
