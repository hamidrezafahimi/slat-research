#!/usr/bin/env python3
import argparse
import sys
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

# ---------- B-spline helpers (clamped uniform cubic) ----------
def clamped_uniform_knot_vector(n_ctrl: int, degree: int):
    p = degree
    m = n_ctrl + p + 1              # length of knot vector
    kv = np.zeros(m, dtype=float)
    kv[:p+1] = 0.0
    kv[-(p+1):] = 1.0
    interior_count = n_ctrl - p - 1
    if interior_count > 0:
        interior = np.linspace(0.0, 1.0, interior_count + 2)[1:-1]
        kv[p+1 : m-(p+1)] = interior
    return kv

def bspline_basis_all(n_ctrl: int, degree: int, kv: np.ndarray, t: float):
    p = degree
    lo, hi = kv[p], kv[-p-1]
    if t <= lo: t = lo + 1e-12
    if t >= hi: t = hi - 1e-12
    t = np.clip(t, kv[p], kv[-p-1] - 1e-12 if p > 0 else kv[-1])

    N = np.zeros(n_ctrl)
    tmp = np.zeros(len(kv) - 1, dtype=float)
    for j in range(len(tmp)):
        tmp[j] = 1.0 if (kv[j] <= t < kv[j+1]) else 0.0
    for d in range(1, p+1):
        for j in range(len(tmp) - d):
            left = 0.0
            right = 0.0
            denom_left = kv[j+d] - kv[j]
            denom_right = kv[j+d+1] - kv[j+1]
            if denom_left > 0:
                left = (t - kv[j]) / denom_left * tmp[j]
            if denom_right > 0:
                right = (kv[j+d+1] - t) / denom_right * tmp[j+1]
            tmp[j] = left + right
    N[:n_ctrl] = tmp[:n_ctrl]
    return N

def sample_bspline_surface(ctrl_pts, gw, gh, samples_u=40, samples_v=40):
    p = 3  # cubic
    q = 3
    U = clamped_uniform_knot_vector(gw, p)
    V = clamped_uniform_knot_vector(gh, q)

    us = np.linspace(0, 1, samples_u)
    vs = np.linspace(0, 1, samples_v)
    Bu = np.stack([bspline_basis_all(gw, p, U, u) for u in us], axis=0)  # (Mu, gw)
    Bv = np.stack([bspline_basis_all(gh, q, V, v) for v in vs], axis=0)  # (Mv, gh)

    P = ctrl_pts.reshape(gh, gw, 3)  # (r,c,3)
    S = np.zeros((samples_v, samples_u, 3), dtype=float)
    for k in range(3):
        Gk = P[..., k]
        inner_u = np.tensordot(Bu, Gk.transpose(0,1), axes=(1,1))      # (Mu, gh)
        S[..., k] = np.tensordot(Bv, inner_u.transpose(1,0), axes=(1,0))  # (Mv, Mu)

    verts = S.reshape(-1, 3)
    tris = []
    for j in range(samples_v - 1):
        for i in range(samples_u - 1):
            a = j * samples_u + i
            b = a + 1
            c = a + samples_u
            d = c + 1
            tris.append([a, c, b])
            tris.append([b, c, d])
    tris = np.asarray(tris, dtype=np.int32)
    return verts, tris

# ---------- Grid factory ----------
def make_grid(grid_w, grid_h, metric_w, metric_h):
    xs = np.linspace(-metric_w/2, metric_w/2, grid_w)
    ys = np.linspace(-metric_h/2, metric_h/2, grid_h)
    xx, yy = np.meshgrid(xs, ys)
    zz = np.zeros_like(xx)
    pts = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    colors = np.tile(np.array([[0.2, 0.8, 1.0]]), (len(pts), 1))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

# ---------- Utilities for CSV IO ----------
def load_ctrl_points_csv(path: str) -> np.ndarray:
    try:
        arr = np.loadtxt(path, delimiter=",")
    except Exception:
        # try whitespace
        arr = np.loadtxt(path)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 3)
    if arr.shape[1] != 3:
        raise ValueError(f"Expected Nx3 CSV, got shape {arr.shape}")
    return arr.astype(float)

def infer_grid_wh_from_points(pts: np.ndarray, tol=1e-8):
    """Infer (grid_w, grid_h) by counting unique X and Y (with tolerance)."""
    xs = pts[:, 0]
    ys = pts[:, 1]
    # Round to kill tiny noise
    xr = np.round(xs / max(np.max(np.abs(xs)), 1.0) * 1e8) if np.max(np.abs(xs)) > 0 else xs
    yr = np.round(ys / max(np.max(np.abs(ys)), 1.0) * 1e8) if np.max(np.abs(ys)) > 0 else ys
    uniq_x = np.unique(xr)
    uniq_y = np.unique(yr)
    gw = len(uniq_x)
    gh = len(uniq_y)
    if gw * gh != pts.shape[0]:
        # fallback: try to guess a reasonable rectangular factorization
        n = pts.shape[0]
        facs = [(w, n // w) for w in range(2, n + 1) if n % w == 0]
        # Prefer shapes with w close to number of unique x values
        facs.sort(key=lambda ab: abs(ab[0] - gw))
        if facs:
            gw, gh = facs[0]
        else:
            raise ValueError("Cannot infer a rectangular grid from the CSV points.")
    return int(gw), int(gh)

def extent_wh_from_points(pts: np.ndarray):
    mins = pts[:, :2].min(axis=0)
    maxs = pts[:, :2].max(axis=0)
    metric_w = float(maxs[0] - mins[0])
    metric_h = float(maxs[1] - mins[1])
    return metric_w, metric_h

# ---------- App ----------
class PointsApp:
    def __init__(self, p, grid_w, grid_h, extra_cloud_path=None, save_path="spline_ctrl.csv", step=0.2):
        self.window = gui.Application.instance.create_window("Pick Points Demo", 1000, 750)
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.window.add_child(self.scene)

        self.grid_w = grid_w
        self.grid_h = grid_h
        self.save_path = save_path
        self.step = float(step)

        # Materials
        self.points_mat = rendering.MaterialRecord()
        self.points_mat.shader = "defaultUnlit"
        self.points_mat.point_size = 10.0

        # SOLID surface material
        self.surf_mat = rendering.MaterialRecord()
        self.surf_mat.shader = "defaultLit"
        self.surf_mat.base_color = (0.7, 0.7, 0.9, 1.0)
        self.surf_mat.base_metallic = 0.0
        self.surf_mat.base_roughness = 0.8
        self.surf_mat.base_reflectance = 0.5

        # External cloud material
        self.ext_mat = rendering.MaterialRecord()
        self.ext_mat.shader = "defaultUnlit"
        self.ext_mat.point_size = 4.0

        # Geometry: control points
        self.pcd = p
        self.scene.scene.add_geometry("points", self.pcd, self.points_mat)

        # Colors
        self.base_color = np.array([0.2, 0.8, 1.0])
        self.highlight_color = np.array([1.0, 0.0, 0.0])
        self.colors = np.asarray(self.pcd.colors).copy()

        # Editable control points
        self.points = np.asarray(self.pcd.points)

        # Selection
        self.cur_r, self.cur_c = 0, 0
        self._apply_highlight(None, (self.cur_r, self.cur_c))

        # Axes
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])
        axis_mat = rendering.MaterialRecord()
        axis_mat.shader = "defaultLit"
        self.scene.scene.add_geometry("axis", axis, axis_mat)

        # Initial B-spline surface
        self.surf_mesh = self._build_surface_mesh(self.points)
        self.scene.scene.add_geometry("spline_surf", self.surf_mesh, self.surf_mat)

        # Optional: load extra point cloud
        self.external_pcd = None
        if extra_cloud_path is not None:
            self.add_external_cloud(extra_cloud_path)

        # Camera — fit to all geometry present
        self._fit_camera_to_all()

        self.window.set_on_key(self.on_key)
        self.scene.set_on_mouse(self.on_mouse)

    # --- External cloud loader ---
    def add_external_cloud(self, path: str):
        p = o3d.io.read_point_cloud(path)
        if len(p.points) == 0:
            print(f"[WARN] External cloud at '{path}' has 0 points or failed to load.")
            return
        if not p.has_colors():
            colors = np.tile(np.array([[0.5, 0.5, 0.5]]), (len(p.points), 1))
            p.colors = o3d.utility.Vector3dVector(colors)
        self.external_pcd = p
        try:
            self.scene.scene.remove_geometry("external_pcd")
        except Exception:
            pass
        self.scene.scene.add_geometry("external_pcd", self.external_pcd, self.ext_mat)
        print(f"[OK] Loaded external point cloud: {path}  (#pts={len(p.points)})")

    # --- Utilities ---
    def rc2idx(self, r, c):
        return r * self.grid_w + c

    def _apply_highlight(self, prev_rc, new_rc):
        if prev_rc is not None:
            pr, pc = prev_rc
            self.colors[self.rc2idx(pr, pc)] = self.base_color
        nr, nc = new_rc
        self.colors[self.rc2idx(nr, nc)] = self.highlight_color
        self.pcd.colors = o3d.utility.Vector3dVector(self.colors)
        self.pcd.points = o3d.utility.Vector3dVector(self.points)
        self.scene.scene.remove_geometry("points")
        self.scene.scene.add_geometry("points", self.pcd, self.points_mat)
        print(f"Selected ctrl (row, col): ({nr}, {nc})  idx={self.rc2idx(nr, nc)}  pos={self.points[self.rc2idx(nr, nc)]}")

    def _build_surface_mesh(self, ctrl_points):
        verts, tris = sample_bspline_surface(
            ctrl_points, self.grid_w, self.grid_h,
            samples_u=40, samples_v=40
        )
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(tris)
        mesh.compute_vertex_normals()
        return mesh

    def _update_surface(self):
        self.scene.scene.remove_geometry("spline_surf")
        self.surf_mesh = self._build_surface_mesh(self.points)
        self.scene.scene.add_geometry("spline_surf", self.surf_mesh, self.surf_mat)

    def _fit_camera_to_all(self):
        geoms = [self.pcd, self.surf_mesh]
        if self.external_pcd is not None:
            geoms.append(self.external_pcd)
        mins = []
        maxs = []
        for g in geoms:
            aabb = g.get_axis_aligned_bounding_box()
            mins.append(np.asarray(aabb.min_bound))
            maxs.append(np.asarray(aabb.max_bound))
        mins = np.vstack(mins).min(axis=0)
        maxs = np.vstack(maxs).max(axis=0)
        big = o3d.geometry.AxisAlignedBoundingBox(mins, maxs)
        self.scene.setup_camera(60, big, big.get_center())

    def _translate_all(self, dx=0.0, dy=0.0, dz=0.0):
        self.points[:, 0] += dx
        self.points[:, 1] += dy
        self.points[:, 2] += dz
        self._apply_highlight((self.cur_r, self.cur_c), (self.cur_r, self.cur_c))
        self._update_surface()
        print(f"Translated grid by ({dx:.3f}, {dy:.3f}, {dz:.3f})")

    def _save_ctrl_points(self):
        np.savetxt(self.save_path, self.points, fmt="%.6f", delimiter=",")
        print(f"[OK] Saved {len(self.points)} control points to '{self.save_path}'")

    # --- New: sync row/column z-values to active ctrl ---
    def _sync_row_to_active(self):
        z = float(self.points[self.rc2idx(self.cur_r, self.cur_c), 2])
        r = self.cur_r
        start = r * self.grid_w
        end = start + self.grid_w
        self.points[start:end, 2] = z
        print(f"[ROW] Set row {r} z -> {z:.6f}")
        self._apply_highlight((r, self.cur_c), (r, self.cur_c))
        self._update_surface()

    def _sync_col_to_active(self):
        z = float(self.points[self.rc2idx(self.cur_r, self.cur_c), 2])
        c = self.cur_c
        idxs = [self.rc2idx(r, c) for r in range(self.grid_h)]
        self.points[idxs, 2] = z
        print(f"[COL] Set col {c} z -> {z:.6f}")
        self._apply_highlight((self.cur_r, c), (self.cur_r, c))
        self._update_surface()

    def on_key(self, event):
        if event.type != gui.KeyEvent.Type.DOWN:
            return False

        prev = (self.cur_r, self.cur_c)
        moved = False
        idx = self.rc2idx(self.cur_r, self.cur_c)

        if event.key == gui.KeyName.LEFT:
            self.cur_c = max(0, self.cur_c - 1); moved = True
        elif event.key == gui.KeyName.RIGHT:
            self.cur_c = min(self.grid_w - 1, self.cur_c + 1); moved = True
        elif event.key == gui.KeyName.UP:
            self.cur_r = min(self.grid_h - 1, self.cur_r + 1); moved = True
        elif event.key == gui.KeyName.DOWN:
            self.cur_r = max(0, self.cur_r - 1); moved = True

        # Raise/lower only selected ctrl (uses --step)
        elif event.key == gui.KeyName.U:
            self.points[idx, 2] += self.step
            print(f"Raised ctrl {idx} to z={self.points[idx,2]:.6f}")
            self._apply_highlight((self.cur_r, self.cur_c), (self.cur_r, self.cur_c))
            self._update_surface()
            return True
        elif event.key == gui.KeyName.I:
            self.points[idx, 2] -= self.step
            print(f"Lowered ctrl {idx} to z={self.points[idx,2]:.6f}")
            self._apply_highlight((self.cur_r, self.cur_c), (self.cur_r, self.cur_c))
            self._update_surface()
            return True

        # New: sync same row/column z-values to active
        elif event.key == gui.KeyName.O:
            self._sync_row_to_active(); return True
        elif event.key == gui.KeyName.P:
            self._sync_col_to_active(); return True

        # Global move whole spline+ctrls (uses --step)
        elif event.key == gui.KeyName.W:   # up
            self._translate_all(dy=self.step);   return True
        elif event.key == gui.KeyName.S:   # down
            self._translate_all(dy=-self.step);  return True
        elif event.key == gui.KeyName.A:   # left
            self._translate_all(dx=-self.step);  return True
        elif event.key == gui.KeyName.D:   # right
            self._translate_all(dx=self.step);   return True
        elif event.key == gui.KeyName.C:   # drop
            self._translate_all(dz=-self.step);  return True
        elif event.key == gui.KeyName.V:   # lift
            self._translate_all(dz=self.step);   return True

        # Save
        elif event.key == gui.KeyName.Y:
            self._save_ctrl_points()
            return True
        elif event.key == gui.KeyName.Q:
            print("[INFO] Quit requested (Q)")
            gui.Application.instance.quit()
            return True
        else:
            return False

        if moved:
            self._apply_highlight(prev, (self.cur_r, self.cur_c))
        return True

    def on_mouse(self, event):
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN:
            return True
        return False

    def run(self):
        gui.Application.instance.run()

def print_keymap(save_path, step):
    s = float(step)
    print("\n=== Keymap ===")
    print("Arrow keys : move selection (row/col)")
    print(f"U / I      : raise / lower selected control point (±{s:.3f} in z)")
    print(f"W A S D    : move whole grid up/left/down/right (±{s:.3f} in x/y)")
    print(f"V / C      : lift / drop whole grid (±{s:.3f} in z)")
    print("O          : set entire row z to the active point's z")
    print("P          : set entire column z to the active point's z")
    print(f"Y          : save control points to '{save_path}'")
    print("Q          : quit\n")

# ---------- Main ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive B-spline surface editor with optional external cloud.")
    parser.add_argument("--cloud", type=str, default=None, help="Path to extra point cloud (.ply/.pcd/.xyz/...)")
    parser.add_argument("--out", type=str, default="spline_ctrl.csv", help="Path to save control points when pressing 'Y'")
    parser.add_argument("--step", type=float, default=0.2, help="Movement step size for raise/lower and translations (world units)")

    # Grid/metric arguments (used only when NOT loading from file)
    parser.add_argument("--grid_w", type=int, help="Grid width (number of control points along X)")
    parser.add_argument("--grid_h", type=int, help="Grid height (number of control points along Y)")
    parser.add_argument("--metric_w", type=float, help="Metric width (extent in X)")
    parser.add_argument("--metric_h", type=float, help="Metric height (extent in Y)")

    # Load from CSV
    parser.add_argument("--spline_data", type=str, default=None, help="CSV file with Nx3 control points. If set, do NOT pass grid/metric sizes.")

    args = parser.parse_args()

    # Validate argument combinations
    if args.spline_data is not None:
        forbidden = [a for a in ("grid_w","grid_h","metric_w","metric_h") if getattr(args, a) is not None]
        if forbidden:
            print(f"[ERROR] When --spline_data is used, do NOT pass {forbidden}. The app infers grid/metric size from data.")
            sys.exit(1)
        # Load points and infer grid sizes and metrics
        pts = load_ctrl_points_csv(args.spline_data)
        gw, gh = infer_grid_wh_from_points(pts)
        mw, mh = extent_wh_from_points(pts)
        print(f"[OK] Loaded {pts.shape[0]} points from '{args.spline_data}'. Inferred grid: {gw}×{gh}, extents: W={mw:.3f}, H={mh:.3f}")
        # Build point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        base_color = np.array([[0.2, 0.8, 1.0]])
        pcd.colors = o3d.utility.Vector3dVector(np.tile(base_color, (len(pts), 1)))
        GRID_W, GRID_H = gw, gh
    else:
        # Require all four sizes if not loading from file
        missing = [name for name in ("grid_w","grid_h","metric_w","metric_h") if getattr(args, name) is None]
        if missing:
            print(f"[ERROR] Missing required args: {missing}. Either provide all four OR use --spline_data.")
            print("Example (no file): --grid_w 10 --grid_h 10 --metric_w 10 --metric_h 10")
            sys.exit(1)
        GRID_W, GRID_H = args.grid_w, args.grid_h
        pcd = make_grid(GRID_W, GRID_H, args.metric_w, args.metric_h)

    gui.Application.instance.initialize()
    print_keymap(args.out, args.step)
    app = PointsApp(pcd, GRID_W, GRID_H, extra_cloud_path=args.cloud, save_path=args.out, step=args.step)
    app.run()
