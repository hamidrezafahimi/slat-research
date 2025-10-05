# camera.py  ────────────────────────────────────────────────────────────────
"""
Ideal pin-hole camera utilities + depth rasterizer.

What's new
----------
• project_3dTo2d_pc(pts[, rpy])   – point-cloud → pixels  (old behavior)
  - If `rpy` is provided, points are assumed in NWU and are transformed to
    the camera frame using your `transformations.py`, then projected.

• project_3dTo2d_sp(ctrl_pts, rpy) – spline control-grid → per-pixel depth
  - ctrl_pts is a rectangular grid of (x,y,z) in NWU (order can be arbitrary).
  - `rpy` is MANDATORY. The grid is transformed NWU→camera using your
    `transformations.py`, triangulated, and rasterized to a depth image
    where each pixel stores the metric range from the camera.

Compatibility
-------------
• project_3dTo2d = project_3dTo2d_pc  (backward compatible)
"""

from __future__ import annotations

import math
from typing import Dict, Tuple, Iterable, Optional

import cv2
import numpy as np

import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, ".."))
from kinematics.transformations import transform_nwu_to_camera

# ====================================================================== #
#  simple rotation helpers (kept for project_2dTo3d compatibility)
# ====================================================================== #
def rotation_matrix_x(phi):
    c = np.cos(phi)
    s = np.sin(phi)
    return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])

def rotation_matrix_y(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])

def rotation_matrix_z(psi):
    c = np.cos(psi)
    s = np.sin(psi)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])

# Add near the top of camera.py (helpers) or co-locate with the class:
def _kv(n_ctrl: int, degree: int):
    p = degree
    m = n_ctrl + p + 1
    kv = np.zeros(m, dtype=float)
    kv[-(p + 1):] = 1.0
    interior = n_ctrl - p - 1
    if interior > 0:
        kv[p + 1:-p - 1] = np.linspace(1, interior, interior, dtype=float) / (interior + 1)
    return kv

def _find_span(n: int, p: int, U: np.ndarray, u: float) -> int:
    if u >= U[n]:
        return n - 1
    if u <= U[p]:
        return p
    low, high = p, n
    mid = (low + high) // 2
    while u < U[mid] or u >= U[mid + 1]:
        if u < U[mid]:
            high = mid
        else:
            low = mid
        mid = (low + high) // 2
    return mid

def _basis_funs(span: int, u: float, p: int, U: np.ndarray) -> np.ndarray:
    N = np.zeros(p + 1, dtype=float)
    left = np.zeros(p + 1, dtype=float)
    right = np.zeros(p + 1, dtype=float)
    N[0] = 1.0
    for j in range(1, p + 1):
        left[j] = u - U[span + 1 - j]
        right[j] = U[span + j] - u
        saved = 0.0
        for r in range(j):
            denom = right[r + 1] + left[j - r]
            term = 0.0 if denom == 0.0 else N[r] / denom
            temp = term * right[r + 1]
            N[r] = saved + temp
            saved = term * left[j - r]
        N[j] = saved
    return N

def _basis_matrix(n_ctrl: int, degree: int, U: np.ndarray, params: np.ndarray) -> np.ndarray:
    B = np.zeros((params.size, n_ctrl), dtype=float)
    p = degree
    for r, u in enumerate(params):
        span = _find_span(n_ctrl, p, U, float(u))
        vals = _basis_funs(span, float(u), p, U)
        B[r, span - p : span + 1] = vals
    return B

def _sample_bspline_surface(ctrl_pts: np.ndarray, gw: int, gh: int,
                            samples_u: int, samples_v: int):
    p = q = 3
    U = _kv(gw, p)
    V = _kv(gh, q)
    us = np.linspace(0.0, 1.0, samples_u)
    vs = np.linspace(0.0, 1.0, samples_v)
    Bu = _basis_matrix(gw, p, U, us)  # (Mu, gw)
    Bv = _basis_matrix(gh, q, V, vs)  # (Mv, gh)
    P = ctrl_pts.reshape(gh, gw, 3)

    S = np.zeros((samples_v, samples_u, 3), dtype=float)
    for k in range(3):
        S[..., k] = Bv @ P[..., k] @ Bu.T

    # topology (two tris per param cell)
    verts = S.reshape(-1, 3)
    tris = []
    for j in range(samples_v - 1):
        base = j * samples_u
        for i in range(samples_u - 1):
            a = base + i
            b = a + 1
            c = a + samples_u
            d = c + 1
            tris.append([a, c, b])
            tris.append([b, c, d])
    tris = np.asarray(tris, dtype=np.int32)
    return verts, tris

# ====================================================================== #
#  core class
# ====================================================================== #
class PinholeCamera:
    """
    Ideal pin-hole camera (no lens distortion).

    Parameters
    ----------
    fx, fy : float
        Focal lengths (px).
    cx, cy : float
        Principal point (px).
    image_shape : (W, H)
        Canvas for FOV checks and previews.
    show, log, ambif, noise_std, show_noise, report_disp, rng :
        Same behavior as before.
    """

    def __init__(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        *,
        image_shape: Tuple[int, int] | None = None,
        show: bool = False,
        log: bool = False,
        ambif: bool = True,
        noise_std: float | Tuple[float, float] = 0.0,
        show_noise: bool = False,
        report_disp: bool = False,
        rng: int | np.random.Generator | None = None,
    ) -> None:
        self.fx, self.fy = float(fx), float(fy)
        self.cx, self.cy = float(cx), float(cy)

        self.show = bool(show)
        self.dolog = bool(log)
        self.allMustBeInFOV = bool(ambif)
        self.show_noise = bool(show_noise)
        self.report_disp = bool(report_disp)

        w = int(2 * self.cx) if image_shape is None else int(image_shape[0])
        h = int(2 * self.cy) if image_shape is None else int(image_shape[1])
        self.image_shape = (w, h)               # (W, H)
        self.image_diag = math.hypot(w, h)

        if isinstance(noise_std, (tuple, list, np.ndarray)):
            if len(noise_std) != 2:
                raise ValueError("noise_std tuple must be (σu_pct, σv_pct)")
            self.noise_u_pct, self.noise_v_pct = map(float, noise_std)
        else:
            self.noise_u_pct = self.noise_v_pct = float(noise_std)

        self.noise_u = (self.noise_u_pct / 100.0) * self.image_diag
        self.noise_v = (self.noise_v_pct / 100.0) * self.image_diag

        self._rng = (
            rng
            if isinstance(rng, np.random.Generator)
            else np.random.default_rng(rng)
        )

        self._prev_uv: Dict[int, Tuple[float, float]] | None = None
        self.mean_pct = 0

    # ----------------------- intrinsics ---------------------------------
    def getK(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0, self.cx],
             [0, self.fy, self.cy],
             [0,     0,     1]],
            float,
        )

    # ----------------------- back-projection ----------------------------
    def project_2dTo3d(self, uvs: np.ndarray, rpy=None) -> np.ndarray:
        """
        Back-project *pixel* coordinates to *unit* rays.

        If `rpy` is None → rays are in **camera frame** (Right-Down-Forward).
        If `rpy` is provided → rays are expressed in **NWU** (to match your
        previous tests).
        """
        uv = np.asarray(uvs, dtype=float)
        if uv.shape[-1] != 2:
            raise ValueError("`uvs` must have …×2 shape (u, v)")

        orig_shape = uv.shape[:-1]
        u, v = uv.reshape(-1, 2).T

        x_cam = (u - self.cx) / self.fx
        y_cam = (v - self.cy) / self.fy
        z_cam = np.ones_like(x_cam)

        dirs_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)
        dirs_cam /= np.linalg.norm(dirs_cam, axis=-1, keepdims=True)

        if rpy is None:
            return dirs_cam.reshape(*orig_shape, 3)

        roll, pitch, yaw = map(float, rpy)
        if max(abs(roll), abs(pitch), abs(yaw)) > 2 * math.pi:
            roll, pitch, yaw = map(math.radians, (roll, pitch, yaw))

        # Map camera (RDF) → FRD (body) → NED (via 3-2-1) → NWU
        Ry = np.array([[0, 0, 1],
                       [0, 1, 0],
                       [-1, 0, 0]])
        Rx = np.array([[1, 0, 0],
                       [0, 0, -1],
                       [0, 1, 0]])
        cam_to_frd = Rx @ Ry

        Rphi   = rotation_matrix_x(-roll)
        Rtheta = rotation_matrix_y(-pitch)
        Rpsi   = rotation_matrix_z(-yaw)
        Rnwu_ned = rotation_matrix_x(math.pi)

        R_total = Rnwu_ned @ Rpsi @ Rtheta @ Rphi @ cam_to_frd
        dirs_nwu = dirs_cam @ R_total.T
        dirs_nwu /= np.linalg.norm(dirs_nwu, axis=-1, keepdims=True)

        return dirs_nwu.reshape(*orig_shape, 3)

    # ----------------------- NEW: 3D→2D (point cloud) -------------------
    def project_3dTo2d_pc(
        self,
        pts: np.ndarray,
        rpy: Optional[Iterable[float]] = None,
        cam_pose_xyz: Iterable[float] = (0.0, 0.0, 0.0),
    ) -> np.ndarray:
        """
        Project **(id, X, Y, Z)** to **(id, u, v)** (noisy), like before.

        If `rpy` is provided, input points are assumed in **NWU** and are
        first transformed to the camera frame using your `transformations.py`.
        """
        arr = np.asarray(pts, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.shape[1] != 4:
            raise ValueError("Input must have shape (N,4): (id,x,y,z)")

        ids = arr[:, 0].astype(int)
        XYZ = arr[:, 1:4]

        if rpy is not None:
            if transform_nwu_to_camera is None:
                raise ImportError(
                    "transformations.py not found; required for rpy-handling "
                    "in project_3dTo2d_pc()."
                )
            roll, pitch, yaw = map(float, rpy)
            if max(abs(roll), abs(pitch), abs(yaw)) > 2 * math.pi:
                roll, pitch, yaw = map(math.radians, (roll, pitch, yaw))
            cam_pos = np.array(list(cam_pose_xyz), dtype=float)
            XYZ_cam = transform_nwu_to_camera(
                XYZ, cam_pose=cam_pos, roll=roll, pitch=pitch, yaw=yaw
            )
        else:
            XYZ_cam = XYZ

        X, Y, Z = XYZ_cam.T
        if np.any(Z <= 0):
            raise ValueError("All Z must be positive (points in front of camera)")

        u_ideal = self.fx * (X / Z) + self.cx
        v_ideal = self.fy * (Y / Z) + self.cy

        u_noisy = u_ideal.copy()
        v_noisy = v_ideal.copy()
        if self.noise_u > 0 or self.noise_v > 0:
            u_noisy += self._rng.normal(0.0, self.noise_u, size=u_noisy.shape)
            v_noisy += self._rng.normal(0.0, self.noise_v, size=v_noisy.shape)

        width, height = self.image_shape
        oob = ((u_ideal < 0) | (u_ideal >= width) |
               (v_ideal < 0) | (v_ideal >= height))
        if np.any(oob):
            msg_ids = ", ".join(map(str, ids[oob]))
            msg = f"Point(s) {msg_ids} outside image {self.image_shape}"
            if self.allMustBeInFOV:
                raise ValueError(msg)
            elif self.dolog:
                print("[WARN]", msg)

        if self.show:
            self._show_on_canvas(ids, u_ideal, v_ideal, u_noisy, v_noisy)

        if self.report_disp and self._prev_uv is not None:
            common = np.intersect1d(ids, list(self._prev_uv.keys()))
            if common.size:
                disp_px = []
                for pid in common:
                    prev_u, prev_v = self._prev_uv[pid]
                    idx = np.where(ids == pid)[0][0]
                    du = u_noisy[idx] - prev_u
                    dv = v_noisy[idx] - prev_v
                    disp_px.append(math.hypot(du, dv))
                mean_px = float(np.mean(disp_px))
                self.mean_pct = (mean_px / self.image_diag) * 100.0
                print(f"[camera] Average displacement: {self.mean_pct:.3f}% "
                      f"of diag ({mean_px:.2f} px, {len(disp_px)} pts)")

        self._prev_uv = {pid: (u, v) for pid, u, v in zip(ids, u_noisy, v_noisy)}
        return np.column_stack((ids, u_noisy, v_noisy))

    # And replace the method in class PinholeCamera:
    def project_3dTo2d_sp(
        self,
        ctrl_pts: np.ndarray,
        *,
        rpy,
        cam_pose_xyz=(0.0, 0.0, 0.0),
        samples_u: int = 200,
        samples_v: int = 200,
        return_uvmask: bool = False,
    ):
        """
        Rasterize a TRUE bicubic B-spline surface to a per-pixel depth map.

        ctrl_pts : (gh*gw, 3) control net in NWU (rectangular grid).
        rpy      : required (roll,pitch,yaw). Degrees or radians accepted.
        samples_u, samples_v : parametric sampling density for the spline surface.
                            Increase for a smoother surface; beware memory/time.
        """
        if transform_nwu_to_camera is None:
            raise ImportError("transformations.py required for project_3dTo2d_sp().")

        pts = np.asarray(ctrl_pts, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("ctrl_pts must have shape (N,3)")

        # Infer grid (gh, gw) from unique x/y counts (stable ordering recommended)
        xs = np.unique(pts[:, 0])
        ys = np.unique(pts[:, 1])
        gw, gh = len(xs), len(ys)
        if gw * gh != pts.shape[0]:
            raise ValueError("Control points must form a full rectangular grid")

        # Reorder to (gh, gw, 3) grid by ascending y then ascending x
        def _find(val, arr):
            i = np.argmin(np.abs(arr - val))
            if abs(arr[i] - val) > 1e-8:
                raise ValueError("Grid mismatch/tolerance failure")
            return i

        grid = np.empty((gh, gw, 3), dtype=float)
        for x, y, z in pts:
            j = _find(x, xs)  # column (u)
            i = _find(y, ys)  # row    (v)
            grid[i, j] = (x, y, z)

        # Sample the B-spline surface densely in NWU space
        verts_nwu, tris = _sample_bspline_surface(grid.reshape(-1, 3), gw, gh, samples_u, samples_v)

        # Transform surface vertices NWU → camera
        roll, pitch, yaw = map(float, rpy)
        if max(abs(roll), abs(pitch), abs(yaw)) > 2 * math.pi:
            roll, pitch, yaw = map(math.radians, (roll, pitch, yaw))
        cam_pos = np.array(list(cam_pose_xyz), dtype=float)

        verts_cam = transform_nwu_to_camera(
            verts_nwu, cam_pose=cam_pos, roll=roll, pitch=pitch, yaw=yaw
        )

        # Prepare per-pixel rays (unit, camera frame)
        W, H = self.image_shape
        fx, fy, cx, cy = self.fx, self.fy, self.cx, self.cy
        uu, vv = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
        x_cam = (uu - cx) / fx
        y_cam = (vv - cy) / fy
        z_cam = np.ones_like(x_cam)
        dirs = np.stack([x_cam, y_cam, z_cam], axis=-1)
        dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)

        def _proj_uv(XYZ):
            X, Y, Z = XYZ[..., 0], XYZ[..., 1], XYZ[..., 2]
            return np.stack([fx * (X / Z) + cx, fy * (Y / Z) + cy], axis=-1)

        # Build triangles and their screen-space boxes
        depth = np.full((H, W), np.inf, dtype=np.float32)
        hit = np.zeros((H, W), dtype=bool)

        def _pin(px, py, A, B, C):
            (ax, ay), (bx, by), (cx_, cy_) = A, B, C
            denom = (by - cy_) * (ax - cx_) + (cx_ - bx) * (ay - cy_)
            if abs(denom) < 1e-12:
                return False
            w1 = ((by - cy_) * (px - cx_) + (cx_ - bx) * (py - cy_)) / denom
            w2 = ((cy_ - ay) * (px - cx_) + (ax - cx_) * (py - cy_)) / denom
            w3 = 1.0 - w1 - w2
            return (w1 >= -1e-6) and (w2 >= -1e-6) and (w3 >= -1e-6)

        # Iterate triangles (two per param cell)
        V = verts_cam
        for a, c, b in tris:  # (note the winding we used earlier)
            A, C, B = V[a], V[c], V[b]
            if (A[2] <= 0) or (B[2] <= 0) or (C[2] <= 0):
                continue
            uva, uvb, uvc = _proj_uv(A), _proj_uv(B), _proj_uv(C)
            n = np.cross(B - A, C - A)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-12:
                continue
            num = float(np.dot(n, A))

            ua, va = uva
            ub, vb = uvb
            uc, vc = uvc
            umin = max(0, int(np.floor(min(ua, ub, uc))))
            umax = min(W - 1, int(np.ceil(max(ua, ub, uc))))
            vmin = max(0, int(np.floor(min(va, vb, vc))))
            vmax = min(H - 1, int(np.ceil(max(va, vb, vc))))
            if (umin > umax) or (vmin > vmax):
                continue

            for v in range(vmin, vmax + 1):
                for u in range(umin, umax + 1):
                    if not _pin(u + 0.5, v + 0.5, (ua, va), (ub, vb), (uc, vc)):
                        continue
                    d = dirs[v, u]
                    den = float(np.dot(n, d))
                    if abs(den) < 1e-12:
                        continue
                    t = num / den
                    if t <= 0.0 or not np.isfinite(t):
                        continue
                    if t < depth[v, u]:
                        depth[v, u] = t
                        hit[v, u] = True

        depth[np.isinf(depth)] = np.nan
        return (depth, hit) if return_uvmask else depth


    # ----------------------- (legacy preview helper) --------------------
    def _show_on_canvas(
        self,
        ids: np.ndarray,
        u_ideal: np.ndarray,
        v_ideal: np.ndarray,
        u_noisy: np.ndarray,
        v_noisy: np.ndarray,
    ) -> None:
        width, height = self.image_shape
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thickness = 0.35, 1

        for pid, ui, vi, un, vn in zip(ids, u_ideal, v_ideal, u_noisy, v_noisy):
            ui_i, vi_i = int(round(ui)), int(round(vi))
            un_i, vn_i = int(round(un)), int(round(vn))
            clr_ideal = (200, 200, 200)
            clr_noisy = (0, 255, 0)
            clr_line  = (120, 120, 120)

            cv2.circle(canvas, (ui_i, vi_i), 3, clr_ideal, -1)
            if self.show_noise:
                cv2.circle(canvas, (un_i, vn_i), 3, clr_noisy, -1)
                cv2.line(canvas, (ui_i, vi_i), (un_i, vn_i), clr_line, 1)
                mid_x = int(round((ui_i + un_i) / 2))
                mid_y = int(round((vi_i + vn_i) / 2))
                cv2.putText(canvas, str(pid), (mid_x + 3, mid_y - 3),
                            font, scale, clr_noisy, thickness)
            else:
                cv2.circle(canvas, (un_i, vn_i), 3, clr_noisy, -1)
                cv2.putText(canvas, str(pid), (un_i + 3, vn_i - 3),
                            font, scale, clr_noisy, thickness)

        cv2.imshow("Pinhole Projection", canvas)
        cv2.waitKey()

    def __repr__(self) -> str:  # pragma: no cover
        flags = []
        if self.show: flags.append("show")
        if self.show_noise: flags.append("show_noise")
        if self.report_disp: flags.append("report_disp")
        if not self.allMustBeInFOV: flags.append("ambif=False")
        flag_str = ", ".join(flags)
        return (f"{self.__class__.__name__}(image_shape={self.image_shape}, "
                f"noise_std_pct=({self.noise_u_pct:.2f}, {self.noise_v_pct:.2f})"
                + (", " + flag_str if flag_str else "") + ")")


# ====================================================================== #
#  convenience subclass – square pixels, centred principal point
# ====================================================================== #
class SquaredPixelFocalCenteredPinholeCamera(PinholeCamera):
    """Square pixels & centred principal point; configure via image_shape + hfov."""
    def __init__(
        self,
        image_shape: Tuple[int, int],
        hfov_deg: float,
        *,
        show: bool = False,
        log: bool = False,
        ambif: bool = True,
        noise_std: float | Tuple[float, float] = 0.0,
        show_noise: bool = False,
        report_disp: bool = False,
        rng: int | np.random.Generator | None = None,
    ) -> None:
        W, H = map(int, image_shape)
        hfov_rad = math.radians(float(hfov_deg))
        if not 0 < hfov_rad < math.pi:
            raise ValueError("hfov_deg must be in (0, 180)")

        f = (W / 2) / math.tan(hfov_rad / 2)
        cx, cy = W / 2, H / 2
        self.f = f
        self.hfov_deg = float(hfov_deg)
        self.vfov_deg = math.degrees(2 * math.atan((H / 2) / f))
        self.image_shape = (W, H)

        super().__init__(
            fx=f, fy=f, cx=cx, cy=cy,
            image_shape=(W, H),
            show=show, log=log, ambif=ambif,
            noise_std=noise_std, show_noise=show_noise,
            report_disp=report_disp, rng=rng,
        )

    def __repr__(self) -> str:  # pragma: no cover
        base = super().__repr__()
        base = base.replace(self.__class__.__name__, "")
        return (f"{self.__class__.__name__}(image_shape={self.image_shape}, "
                f"hfov_deg={self.hfov_deg:.2f}, f={self.f:.2f}px, "
                f"vfov_deg={self.vfov_deg:.2f}{base}")


# convenient alias
SimpleCamera = SquaredPixelFocalCenteredPinholeCamera

# Back-compat alias (old name)
def project_3dTo2d(self, *args, **kwargs):
    """Backward-compatible alias → project_3dTo2d_pc."""
    return self.project_3dTo2d_pc(*args, **kwargs)

__all__ = [
    "PinholeCamera",
    "SquaredPixelFocalCenteredPinholeCamera",
    "SimpleCamera",
]
