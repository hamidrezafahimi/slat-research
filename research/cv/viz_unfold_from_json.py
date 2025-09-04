import json
import argparse
from pathlib import Path
import numpy as np
import cv2

# --------------------------- helpers ---------------------------------
def load_unfold_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Prefer explicit dense_red if present; otherwise derive from splines[*].Y_dense
    if "dense_red" in data and all(k in data["dense_red"] for k in ("top","right","bottom","left")):
        dense_red = {k: np.asarray(v, dtype=np.float32) for k, v in data["dense_red"].items()}
    else:
        spl = data["splines"]
        dense_red = {k: np.asarray(spl[k]["Y_dense"], dtype=np.float32) for k in ("top","right","bottom","left")}
    # Optional: spline polylines (for drawing)
    splines = {}
    if "splines" in data:
        for side in ("top","right","bottom","left"):
            s = data["splines"][side]
            splines[side] = {
                "P": np.asarray(s.get("P", []), dtype=np.float32),
                "Y_dense": np.asarray(s.get("Y_dense", []), dtype=np.float32),
            }
    else:
        splines = {k: {"P": np.empty((0,3), np.float32), "Y_dense": dense_red[k]} for k in dense_red.keys()}
    # Optional: sparse orange samples (not required for interpolation)
    unfolded_sample = np.asarray(data.get("unfolded_sample", []), dtype=np.float32)
    return dense_red, splines, unfolded_sample

def unfold_surface_from_dense_red(dense_red):
    """
    Coons-patch interpolation over 4 border splines with chord-length parameterization.
    Ensures interior (green) points lie inside the red boundary and iso-lines are
    well aligned with the opposite border splines.

    dense_red: dict with 'top','right','bottom','left' arrays of shape (N,3).
               Expected directions:
                 top:    left -> right
                 right:  top  -> bottom
                 bottom: right-> left   (we'll reverse)
                 left:   bottom-> top   (we'll reverse)
    Returns:
      Q: (H,W,3) unfolded surface
      sides: dict of aligned borders (top/right/bottom/left)
    """
    import numpy as np

    # ---- fetch & align directions ----
    top    = np.asarray(dense_red["top"],    dtype=np.float32)      # (W,3)  left->right
    right  = np.asarray(dense_red["right"],  dtype=np.float32)      # (H,3)  top->bottom
    bottom = np.asarray(dense_red["bottom"], dtype=np.float32)[::-1]  # make left->right
    left   = np.asarray(dense_red["left"],   dtype=np.float32)[::-1]  # make top->bottom

    W = top.shape[0]
    H = right.shape[0]
    assert bottom.shape[0] == W and left.shape[0] == H, "Inconsistent dense_red lengths."

    # ---- chord-length parameterization helper ----
    def _arcparam(P):
        d = np.linalg.norm(np.diff(P[:, :3], axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(d)])
        s /= max(s[-1], 1e-9)
        return s

    def _sample_curve(P, sP, t_vec):
        """Linear interpolation on chord-length parameter t∈[0,1]."""
        t = np.clip(np.asarray(t_vec, np.float32), 0.0, 1.0)
        # indices of the segment just below t
        idx = np.searchsorted(sP, t, side='right') - 1
        idx = np.clip(idx, 0, len(sP) - 2)
        t0 = sP[idx]; t1 = sP[idx + 1]
        w  = (t - t0) / np.maximum(t1 - t0, 1e-9)
        P0 = P[idx]
        P1 = P[idx + 1]
        return P0 + (w[:, None] * (P1 - P0))

    # precompute arc-params
    s_top    = _arcparam(top)
    s_bottom = _arcparam(bottom)
    s_left   = _arcparam(left)
    s_right  = _arcparam(right)

    # normalized grid params
    u = (np.arange(W, dtype=np.float32) / max(W - 1.0, 1.0))  # left->right
    v = (np.arange(H, dtype=np.float32) / max(H - 1.0, 1.0))  # top->bottom

    # sample border curves at those parameters
    T = _sample_curve(top,    s_top,    u)         # (W,3)
    B = _sample_curve(bottom, s_bottom, u)         # (W,3)
    L = _sample_curve(left,   s_left,   v)         # (H,3)
    R = _sample_curve(right,  s_right,  v)         # (H,3)

    # corners (consistent with directions)
    TL, TR = T[0],  T[-1]
    BL, BR = B[0],  B[-1]

    # expand to grid
    uu = u[None, :][:, :, None]    # (1,W,1)
    vv = v[:, None][:, :, None]    # (H,1,1)

    Tg = T[None, :, :]             # (1,W,3)
    Bg = B[None, :, :]             # (1,W,3)
    Lg = L[:, None, :]             # (H,1,3)
    Rg = R[:, None, :]             # (H,1,3)

    # Coons patch:
    # S(u,v) = (1-v)T(u) + vB(u) + (1-u)L(v) + uR(v)
    #          - [(1-u)(1-v)TL + u(1-v)TR + (1-u)v BL + u v BR]
    S = (1 - vv) * Tg + vv * Bg + (1 - uu) * Lg + uu * Rg \
        - ((1 - uu) * (1 - vv) * TL
           + uu * (1 - vv) * TR
           + (1 - uu) * vv * BL
           + uu * vv * BR)

    sides = dict(top=T, right=R, bottom=B, left=L)
    return S.astype(np.float32), sides

def fit_view(points, width, height, margin=0.05):
    """
    Map XY points to image pixels:
      - keep aspect ratio
      - add margin ratio
      - y-up in world -> y-down in image
    Returns a function world_to_px(xy) -> (u,v) ints and the used bounds.
    """
    pts = np.asarray(points, dtype=np.float32)
    if pts.size == 0:
        # Avoid degenerate bounds
        return lambda xy: (np.array([], int), np.array([], int)), ((0,1),(0,1))
    xs, ys = pts[:,0], pts[:,1]
    minx, maxx = float(xs.min()), float(xs.max())
    miny, maxy = float(ys.min()), float(ys.max())
    rangex = max(maxx - minx, 1e-6)
    rangey = max(maxy - miny, 1e-6)

    # compute scale with margins
    mx = margin * rangex
    my = margin * rangey
    minx_m, maxx_m = minx - mx, maxx + mx
    miny_m, maxy_m = miny - my, maxy + my
    rangex_m = maxx_m - minx_m
    rangey_m = maxy_m - miny_m

    sx = (width - 1)  / rangex_m
    sy = (height - 1) / rangey_m
    s  = min(sx, sy)

    # center fit
    cx_world = 0.5 * (minx_m + maxx_m)
    cy_world = 0.5 * (miny_m + maxy_m)
    cx_img   = 0.5 * (width - 1)
    cy_img   = 0.5 * (height - 1)

    def world_to_px(xy):
        xy = np.asarray(xy, dtype=np.float32)
        u = cx_img + (xy[:,0] - cx_world) * s
        v = cy_img - (xy[:,1] - cy_world) * s  # flip Y for image coords
        return u.astype(int), v.astype(int)

    return world_to_px, ((minx_m, maxx_m), (miny_m, maxy_m))

def draw_polyline(img, pts_xy, world2px, thickness=1, color=(0,128,255)):
    if pts_xy.shape[0] < 2:
        return
    u, v = world2px(pts_xy[:, :2])
    for k in range(len(u)-1):
        cv2.line(img, (u[k], v[k]), (u[k+1], v[k+1]), color, thickness, cv2.LINE_AA)

def draw_points(img, pts_xy, world2px, radius=1, color=(0,0,255)):
    if pts_xy.shape[0] == 0:
        return
    u, v = world2px(pts_xy[:, :2])
    for x, y in zip(u, v):
        cv2.circle(img, (int(x), int(y)), radius, color, -1, lineType=cv2.LINE_AA)

# --------------------------- main CLI --------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Visualize unfold_image_borders JSON + unfolded surface (XY plane) with OpenCV."
    )
    ap.add_argument("--json", required=True, type=Path, help="Path to unfold_image_borders_output.json")
    ap.add_argument("--out", type=Path, default=Path("unfold_xy.png"), help="Output PNG path")
    ap.add_argument("--width", type=int, default=1200, help="Output image width in pixels")
    ap.add_argument("--height", type=int, default=1200, help="Output image height in pixels")
    ap.add_argument("--pt_rad", type=int, default=1, help="Dot radius for dense points")
    args = ap.parse_args()

    dense_red, splines, unfolded_sample = load_unfold_json(args.json)
    unfolded_surface, sides = unfold_surface_from_dense_red(dense_red)

    # Gather everything we want to render in XY
    reds_xy = np.concatenate([
        sides["top"][:, :2],
        sides["right"][:, :2],
        sides["bottom"][:, :2],
        sides["left"][:, :2]
    ], axis=0)

    surface_xy = unfolded_surface.reshape(-1, 3)[:, :2]  # (H*W, 2)
    # Use both reds + surface to set view bounds (so all are visible)
    all_xy_for_view = np.concatenate([reds_xy, surface_xy], axis=0)

    world2px, _ = fit_view(np.column_stack([all_xy_for_view, np.zeros(len(all_xy_for_view))]),
                           args.width, args.height, margin=0.05)

    # Create canvas
    img = np.full((args.height, args.width, 3), 255, np.uint8)

    # 1) Draw splines (orange polylines)
    # ORANGE = (0, 165, 255)
    # for side in ("top","right","bottom","left"):
    #     yd = splines[side]["Y_dense"]
    #     if yd.size:
    #         draw_polyline(img, yd[:, :2], world2px, thickness=2, color=ORANGE)

    # 2) Draw dense reds (tiny red dots)
    RED = (0, 0, 255)
    draw_points(img, sides["top"][:, :2], world2px, radius=args.pt_rad, color=RED)
    draw_points(img, sides["right"][:, :2], world2px, radius=args.pt_rad, color=RED)
    draw_points(img, sides["bottom"][:, :2], world2px, radius=args.pt_rad, color=RED)
    draw_points(img, sides["left"][:, :2], world2px, radius=args.pt_rad, color=RED)

    # 3) Draw unfolded surface (light green dots, denser)
    # LIGHT_GREEN = (144, 238, 144)
    # draw_points(img, surface_xy, world2px, radius=1, color=LIGHT_GREEN)

    # 4) Optional: draw sparse orange samples if present
    # if unfolded_sample.size:
    #     # thicker / larger to distinguish
    #     ORANGE_DARK = (0, 128, 255)
    #     draw_points(img, unfolded_sample[:, :2], world2px, radius=3, color=ORANGE_DARK)

    # 5) Legend
    # cv2.putText(img, "Splines (corner->corner): ORANGE", (20, 30),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, ORANGE, 2, cv2.LINE_AA)
    # cv2.putText(img, "Dense border reds: RED", (20, 60),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2, cv2.LINE_AA)
    # cv2.putText(img, "Unfolded surface (XY): LIGHT-GREEN", (20, 90),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, LIGHT_GREEN, 2, cv2.LINE_AA)

    # Save and also show (press a key to close)
    cv2.imwrite(str(args.out), img)
    cv2.imshow("Unfold XY", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
