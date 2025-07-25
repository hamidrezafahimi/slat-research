# camera.py  ────────────────────────────────────────────────────────────────
"""
Ideal pin-hole camera with

  • *Gaussian pixel noise* set **as a % of the image diagonal**,  
  • dual preview of ideal vs noisy projections (`show_noise=True`), and  
  • per-call report of the **average key-point displacement** expressed both
    in pixels *and* as a % of the image diagonal.

The normalisation follows

    normalized [%] = (value / image_diagonal) × 100
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import cv2
import numpy as np


def rotation_matrix_x(phi):
    """Rotation about x-axis."""
    c = np.cos(phi)
    s = np.sin(phi)
    return np.array([
        [1, 0, 0],
        [0, c, s],
        [0, -s, c]
    ])

def rotation_matrix_y(theta):
    """Rotation about y-axis."""
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        [c, 0, -s],
        [0, 1, 0],
        [s, 0, c]
    ])

def rotation_matrix_z(psi):
    """Rotation about z-axis."""
    c = np.cos(psi)
    s = np.sin(psi)
    return np.array([
        [c, s, 0],
        [-s, c, 0],
        [0, 0, 1]
    ])

# ========================================================================= #
#  core class
# ========================================================================= #
class PinholeCamera:
    """
    Ideal pin-hole camera (no lens distortion).

    Parameters
    ----------
    fx, fy : float
        Focal lengths in **pixels**.
    cx, cy : float
        Principal point in **pixels**.
    image_shape : (W, H), default ``(2*cx, 2*cy)``
        Canvas for preview and FOV checks.
    show : bool, default ``False``
        Show a live OpenCV window (“Pinhole Projection”).
    log : bool, default ``False``
        Print warnings (instead of raising) when points leave the FOV while
        ``ambif`` is *False*.
    ambif : bool, default ``True``
        If *True* (“all must be in FOV”) ⇒ raise on out-of-FOV;  
        if *False* ⇒ warn or stay silent.
    noise_std : float | (float, float), default ``0.0``
        **Percentage of the image diagonal** (0 – 100) used as the *σ* of the
        zero-mean Gaussian noise on *(u,v)*.  
        Scalar ⇒ isotropic noise; pair ⇒ anisotropic (σu, σv).  
        ``0`` ➜ deterministic projection.
    show_noise : bool, default ``False``
        If *True* (and ``show`` is *True*) draw ideal & noisy points with a
        connecting line.  If *False*, draw only the noisy points.
    report_disp : bool, default ``False``
        After the first call, print  

            Average displacement : d_norm [%] (d_px px, N pts)

        where ``d_norm`` is normalised to the image diagonal.
    rng : None | int | np.random.Generator, default ``None``
        RNG for sampling the noise.
    """

    # ------------------------------------------------------------------ #
    #  constructor
    # ------------------------------------------------------------------ #
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
        # — intrinsics —
        self.fx, self.fy = float(fx), float(fy)
        self.cx, self.cy = float(cx), float(cy)

        # — behaviour flags —
        self.show = bool(show)
        self.dolog = bool(log)
        self.allMustBeInFOV = bool(ambif)
        self.show_noise = bool(show_noise)
        self.report_disp = bool(report_disp)

        # — preview canvas —
        w = int(2 * self.cx) if image_shape is None else int(image_shape[0])
        h = int(2 * self.cy) if image_shape is None else int(image_shape[1])
        self.image_shape = (w, h)               # (width, height)
        self.image_diag = math.hypot(w, h)      # for normalisation

        # — noise model: input given in % of diag ⇒ convert to px —
        if isinstance(noise_std, (tuple, list, np.ndarray)):
            if len(noise_std) != 2:
                raise ValueError("noise_std tuple must be (σ_u_pct, σ_v_pct)")
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

        # — storage for displacement reporting —
        self._prev_uv: Dict[int, Tuple[float, float]] | None = None
        self.mean_pct = 0

    # ------------------------------------------------------------------ #
    #  public helpers
    # ------------------------------------------------------------------ #
    def getK(self) -> np.ndarray:
        """Return the 3 × 3 intrinsic matrix."""
        return np.array(
            [[self.fx, 0, self.cx],
             [0, self.fy, self.cy],
             [0,     0,     1]],
            float,
        )

    def project_2dTo3d(self, uvs: np.ndarray, rpy=None) -> np.ndarray:
        """
        Back-project *pixel* coordinates to *unit* 3-D rays.

        Parameters
        ----------
        uvs : ndarray[..., 2]
            Any shape whose *last* dimension is ``(u, v)``.
        rpy : (roll, pitch, yaw) | None, optional
            Aircraft attitude (NED → FRD).  Values may be **degrees** or
            **radians** – a simple 2 π heuristic decides.  If *None*, the
            rays are returned in the **camera frame** (Right-Down-Forward).
            Otherwise they are expressed in the **earth NWU frame**.

        Returns
        -------
        ndarray[..., 3]   (unit-length)
        """

        # ---------- 1. pixel → camera (Right-Down-Forward)  -------------
        uv = np.asarray(uvs, dtype=float)
        if uv.shape[-1] != 2:
            raise ValueError("`uvs` must have …×2 shape (u, v)")

        orig_shape = uv.shape[:-1]
        uv_flat = uv.reshape(-1, 2)
        u, v = uv_flat.T

        print(self.cx, self.cy, self.fx, self.fy)
        x_cam = (u - self.cx) / self.fx
        y_cam = (v - self.cy) / self.fy
        z_cam = np.ones_like(x_cam)

        dirs_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)
        dirs_cam /= np.linalg.norm(dirs_cam, axis=-1, keepdims=True)

        # ---------- 2. early exit (camera frame requested) --------------
        if rpy is None:
            return dirs_cam.reshape(*orig_shape, 3)

        # ---------- 3. camera → body (FRD) ------------------------------
        Ry = np.array([[0, 0, 1],
                       [0, 1, 0],
                       [-1, 0, 0]])
        Rx = np.array([[1, 0, 0],
                       [0, 0, -1],
                       [0, 1, 0]])
        cam_to_frd = Rx @ Ry                              # fixed mapping
        dirs_frd = dirs_cam @ cam_to_frd.T

        # ---------- 4. body → NED (Euler 3-2-1, NED → FRD) --------------
        roll, pitch, yaw = map(float, rpy)
        if max(abs(roll), abs(pitch), abs(yaw)) > 2 * math.pi:
            roll, pitch, yaw = map(math.radians, (roll, pitch, yaw))

        Rphi   = rotation_matrix_x(-roll)     # note the *minus*
        Rtheta = rotation_matrix_y(-pitch)
        Rpsi   = rotation_matrix_z(-yaw)

        # ---------- 5. NED → NWU (earth frame used in the demo) ---------
        Rnwu_ned = rotation_matrix_x(math.pi)  # 180° about X (North)

        # Total transformation: cam → NWU
        R_total = Rnwu_ned @ Rpsi @ Rtheta @ Rphi @ cam_to_frd
        dirs_nwu = dirs_cam @ R_total.T
        dirs_nwu /= np.linalg.norm(dirs_nwu, axis=-1, keepdims=True)

        return dirs_nwu.reshape(*orig_shape, 3)


    def project_3dTo2d(self, pts_cam: np.ndarray) -> np.ndarray:
        """
        Project **(id, X, Y, Z)** → **(id, u, v)** (noisy coords).

        The Gaussian noise σ is derived from ``noise_std`` % of the
        image diagonal.  The displacement report is likewise normalised.
        """
        pts = np.asarray(pts_cam, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts[None, :]
        if pts.shape[1] != 4:
            raise ValueError("Input must have shape (N,4): (id,x,y,z)")

        ids = pts[:, 0].astype(int)
        X, Y, Z = pts[:, 1], pts[:, 2], pts[:, 3]

        if np.any(Z <= 0):
            raise ValueError("All Z must be positive (point in front of camera)")

        # — ideal pin-hole projection —
        u_ideal = self.fx * (X / Z) + self.cx
        v_ideal = self.fy * (Y / Z) + self.cy

        # — add Gaussian noise (σ in px) —
        u_noisy = u_ideal.copy()
        v_noisy = v_ideal.copy()
        if self.noise_u > 0 or self.noise_v > 0:
            u_noisy += self._rng.normal(0.0, self.noise_u, size=u_noisy.shape)
            v_noisy += self._rng.normal(0.0, self.noise_v, size=v_noisy.shape)

        # — FOV check (ideal coords) —
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

        # — optional preview —
        if self.show:
            self._show_on_canvas(ids, u_ideal, v_ideal, u_noisy, v_noisy)

        # — average-displacement report (after preview) —
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

        # — store coords for next call —
        self._prev_uv = {pid: (u, v) for pid, u, v in zip(ids, u_noisy, v_noisy)}

        return np.column_stack((ids, u_noisy, v_noisy))

    # ------------------------------------------------------------------ #
    #  private helpers
    # ------------------------------------------------------------------ #
    def _show_on_canvas(
        self,
        ids: np.ndarray,
        u_ideal: np.ndarray,
        v_ideal: np.ndarray,
        u_noisy: np.ndarray,
        v_noisy: np.ndarray,
    ) -> None:
        """Draw projections in a window called “Pinhole Projection”."""
        width, height = self.image_shape
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.35  # small font to reduce clutter
        thickness = 1

        for pid, ui, vi, un, vn in zip(ids, u_ideal, v_ideal, u_noisy, v_noisy):
            ui_i, vi_i = int(round(ui)), int(round(vi))
            un_i, vn_i = int(round(un)), int(round(vn))

            clr_ideal = (200, 200, 200)      # light-grey
            clr_noisy = (0, 255, 0)          # green
            clr_line  = (120, 120, 120)      # mid-grey

            # ideal point
            cv2.circle(canvas, (ui_i, vi_i), 3, clr_ideal, -1)

            if self.show_noise:
                # noisy point + connection
                cv2.circle(canvas, (un_i, vn_i), 3, clr_noisy, -1)
                cv2.line(canvas, (ui_i, vi_i), (un_i, vn_i), clr_line, 1)

                mid_x = int(round((ui_i + un_i) / 2))
                mid_y = int(round((vi_i + vn_i) / 2))
                cv2.putText(canvas, str(pid), (mid_x + 3, mid_y - 3),
                            font, scale, clr_noisy, thickness)
            else:
                # noisy point only
                cv2.circle(canvas, (un_i, vn_i), 3, clr_noisy, -1)
                cv2.putText(canvas, str(pid), (un_i + 3, vn_i - 3),
                            font, scale, clr_noisy, thickness)

        cv2.imshow("Pinhole Projection", canvas)
        cv2.waitKey(1)

    # ------------------------------------------------------------------ #
    #  repr
    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:  # pragma: no cover
        flags = []
        if self.show:
            flags.append("show")
        if self.show_noise:
            flags.append("show_noise")
        if self.report_disp:
            flags.append("report_disp")
        if not self.allMustBeInFOV:
            flags.append("ambif=False")
        flag_str = ", ".join(flags)
        return (f"{self.__class__.__name__}(image_shape={self.image_shape}, "
                f"noise_std_pct=({self.noise_u_pct:.2f}, {self.noise_v_pct:.2f})"
                + (", " + flag_str if flag_str else "") + ")")


# ========================================================================= #
#  convenience subclass – square pixels, centred principal point
# ========================================================================= #
class SquaredPixelFocalCenteredPinholeCamera(PinholeCamera):
    """
    Convenience subclass: square pixels & centred principal point.
    Needs only ``image_shape`` and ``hfov_deg``.
    """

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
            fx=f,
            fy=f,
            cx=cx,
            cy=cy,
            image_shape=(W, H),
            show=show,
            log=log,
            ambif=ambif,
            noise_std=noise_std,
            show_noise=show_noise,
            report_disp=report_disp,
            rng=rng,
        )

    def __repr__(self) -> str:  # pragma: no cover
        base = super().__repr__()
        base = base.replace(self.__class__.__name__, "")  # drop duplicate
        return (f"{self.__class__.__name__}(image_shape={self.image_shape}, "
                f"hfov_deg={self.hfov_deg:.2f}, f={self.f:.2f}px, "
                f"vfov_deg={self.vfov_deg:.2f}{base}")


# convenient alias
SimpleCamera = SquaredPixelFocalCenteredPinholeCamera

__all__ = [
    "PinholeCamera",
    "SquaredPixelFocalCenteredPinholeCamera",
    "SimpleCamera",
]
