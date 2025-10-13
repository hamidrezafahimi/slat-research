
import numpy as np

def _gap_top_bottom(xy_flat: np.ndarray, H: int, W: int) -> float:
    """Perpendicular separation between top (TL–TR) and bottom (BL–BR) edges."""
    TL, TR, BL, BR = 0, W - 1, (H - 1) * W, H * W - 1
    TL2, TR2, BL2, BR2 = xy_flat[TL], xy_flat[TR], xy_flat[BL], xy_flat[BR]
    u_top = TR2 - TL2
    nrm = np.linalg.norm(u_top)
    if nrm < 1e-12:
        return 0.0
    n = np.array([-u_top[1], u_top[0]], float) / nrm
    c_top = 0.5 * (TL2 + TR2); c_bot = 0.5 * (BL2 + BR2)
    return float(abs(np.dot(c_bot - c_top, n)))


def _gap_left_right(xy_flat: np.ndarray, H: int, W: int) -> float:
    """Perpendicular separation between left (TL–BL) and right (TR–BR) edges."""
    TL, TR, BL, BR = 0, W - 1, (H - 1) * W, H * W - 1
    TL2, TR2, BL2, BR2 = xy_flat[TL], xy_flat[TR], xy_flat[BL], xy_flat[BR]
    u_left = BL2 - TL2
    nrm = np.linalg.norm(u_left)
    if nrm < 1e-12:
        return 0.0
    n = np.array([-u_left[1], u_left[0]], float) / nrm
    c_left = 0.5 * (TL2 + BL2); c_right = 0.5 * (TR2 + BR2)
    return float(abs(np.dot(c_right - c_left, n)))


def rectify_xy_proj(pcm: np.ndarray) -> np.ndarray:
    """
    Rectify the XY projection by equalizing adjacent spacings in both directions.

    Parameters
    ----------
    pcm : np.ndarray
        H×W×3 grid of 3D points (XYZ), row-major order.

    Returns
    -------
    pcm_rect : np.ndarray
        H×W×3 grid of rectified 3D points:
          - XY equalized:
              * columns: spacing = l1 / H along each column direction, middle row fixed
              * rows:    spacing = w1 / W along each row direction, middle column fixed
          - Z inherited point-wise from input pcm.
    """
    if pcm.ndim != 3 or pcm.shape[2] != 3:
        raise ValueError("pcm must be H×W×3 (XYZ grid).")

    H, W = pcm.shape[:2]
    pts_flat = pcm.reshape(-1, 3)
    xy_base = pts_flat[:, :2].copy()      # (N,2) baseline XY
    z_vals  = pts_flat[:, 2].copy()       # (N,)

    # Global gaps (baseline; robust to rotation)
    l1 = _gap_top_bottom(xy_base, H, W)
    w1 = _gap_left_right(xy_base, H, W)
    d_col = l1 / float(H)   # per spec (H, not H-1)
    d_row = w1 / float(W)   # per spec (W, not W-1)

    # Work in (H,W,2)
    xy = xy_base.reshape(H, W, 2).copy()
    base = xy_base.reshape(H, W, 2)       # baseline for directions

    # ----- Columns pass (vectorized) -----
    r_mid = H // 2
    top_xy = base[0, :, :]                 # (W,2)
    bot_xy = base[-1, :, :]                # (W,2)
    dir_c  = bot_xy - top_xy               # (W,2)
    nrm_c  = np.linalg.norm(dir_c, axis=1, keepdims=True)  # (W,1)
    safe_c = np.where(nrm_c > 1e-12, nrm_c, 1.0)
    e_c    = dir_c / safe_c                # (W,2)
    e_c[nrm_c[:, 0] <= 1e-12] = 0.0

    Pm_c   = xy[r_mid, :, :]               # (W,2) current mid-row anchors
    offs_r = (np.arange(H, dtype=float) - r_mid)[:, None] * d_col  # (H,1)
    xy = Pm_c[None, :, :] + offs_r[:, None, :] * e_c[None, :, :]

    # ----- Rows pass (vectorized) -----
    c_mid = W // 2
    left_xy  = base[:, 0, :]               # (H,2)
    right_xy = base[:, -1, :]              # (H,2)
    dir_r  = right_xy - left_xy            # (H,2)
    nrm_r  = np.linalg.norm(dir_r, axis=1, keepdims=True)  # (H,1)
    safe_r = np.where(nrm_r > 1e-12, nrm_r, 1.0)
    e_r    = dir_r / safe_r                # (H,2)
    e_r[nrm_r[:, 0] <= 1e-12] = 0.0

    Pm_r   = xy[:, c_mid, :]               # (H,2) current mid-col anchors
    offs_c = (np.arange(W, dtype=float) - c_mid)[None, :] * d_row  # (1,W)
    xy = Pm_r[:, None, :] + offs_c[:, :, None] * e_r[:, None, :]

    # Reattach Z (unchanged) and return as H×W×3
    rect_xy = xy.reshape(-1, 2)
    rect_pts = np.column_stack([rect_xy, z_vals]).reshape(H, W, 3)
    return rect_pts