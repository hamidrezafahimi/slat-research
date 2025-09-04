import json, cv2, numpy as np, argparse, time

def orient_sides(data):
    """
    Normalize side directions so indexing matches the schematic:
      top:    left -> right
      bottom: left -> right
      left:   top  -> bottom
      right:  top  -> bottom
    Returns dict of np.ndarray sides with shape (N, 2) in XY.
    """
    top    = np.array(data["top"],    dtype=float)[:, :2]
    right  = np.array(data["right"],  dtype=float)[:, :2]
    bottom = np.array(data["bottom"], dtype=float)[:, :2]
    left   = np.array(data["left"],   dtype=float)[:, :2]

    # Top: left→right
    if np.linalg.norm(top[0] - left[0]) > np.linalg.norm(top[-1] - left[0]):
        top = top[::-1]
    # Bottom: left→right
    if np.linalg.norm(bottom[0] - left[-1]) > np.linalg.norm(bottom[-1] - left[-1]):
        bottom = bottom[::-1]
    # Left: top→bottom
    if np.linalg.norm(left[0] - top[0]) > np.linalg.norm(left[-1] - top[0]):
        left = left[::-1]
    # Right: top→bottom
    if np.linalg.norm(right[0] - top[-1]) > np.linalg.norm(right[-1] - top[-1]):
        right = right[::-1]

    return {"top": top, "right": right, "bottom": bottom, "left": left}

def clamp01(x):
    return np.minimum(1.0, np.maximum(0.0, x))

def main():
    ap = argparse.ArgumentParser(description="Compute pose P(I,J) for all I,J and plot tiny black dots (fast).")
    ap.add_argument("--json", default="metric_pattern_dense_red.json",
                    help="Path to JSON with 'top','right','bottom','left' arrays")
    ap.add_argument("--out", default="xy_dense_red_fullgrid.png", help="Output image path")
    ap.add_argument("--max_w", type=int, default=1400, help="Max canvas width (px)")
    ap.add_argument("--max_h", type=int, default=1000, help="Max canvas height (px)")
    ap.add_argument("--margin", type=int, default=40, help="Canvas margin (px)")
    args = ap.parse_args()

    t0 = time.time()
    with open(args.json, "r") as f:
        data = json.load(f)

    sides = orient_sides(data)
    top, right, bottom, left = sides["top"], sides["right"], sides["bottom"], sides["left"]

    # Logical sizes
    W = len(top)    # columns (expected 1241)
    H = len(left)   # rows    (expected 176)

    # Precompute canvas transform
    all_xy = np.vstack([top, right, bottom, left])
    min_xy, max_xy = all_xy.min(0), all_xy.max(0)
    rng = np.maximum(max_xy - min_xy, 1e-9)
    sx = (args.max_w - 2 * args.margin) / rng[0]
    sy = (args.max_h - 2 * args.margin) / rng[1]
    scale = float(min(sx, sy))
    canvas_W = int(rng[0] * scale + 2 * args.margin)
    canvas_H = int(rng[1] * scale + 2 * args.margin)

    def to_img_xy(xy):
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        norm = (xy - min_xy) * scale + args.margin
        norm[:, 1] = canvas_H - norm[:, 1]
        return norm

    # Prepare background (white)
    img = np.full((canvas_H, canvas_W, 3), 255, np.uint8)

    # Draw the four borders for context (thin)
    colors = {"top": (0, 0, 255), "right": (0, 200, 0), "bottom": (255, 0, 0), "left": (200, 0, 200)}
    for name, arr in sides.items():
        pts = to_img_xy(arr)
        pts_i = np.round(pts).astype(int)
        pts_i[:, 0] = np.clip(pts_i[:, 0], 0, canvas_W - 1)
        pts_i[:, 1] = np.clip(pts_i[:, 1], 0, canvas_H - 1)
        if len(pts_i) >= 2:
            cv2.polylines(img, [pts_i.reshape(-1, 1, 2)], isClosed=False, color=colors[name], thickness=1, lineType=cv2.LINE_AA)

    # ---------- FAST full-grid P(I,J) computation ----------
    # Corners
    TL, TR, BL, BR = top[0], top[-1], bottom[0], bottom[-1]

    # E components depend only on I (row), via L[I] and R[I]
    TL_BL_len = np.linalg.norm(TL - BL)
    TR_BR_len = np.linalg.norm(TR - BR)
    # Avoid division by zero
    TL_BL_len = TL_BL_len if TL_BL_len > 0 else 1.0
    TR_BR_len = TR_BR_len if TR_BR_len > 0 else 1.0

    # Distances for all I at once
    E1 = np.linalg.norm(left - TL, axis=1) / TL_BL_len          # shape (H,)
    E2 = np.linalg.norm(right - TR, axis=1) / TR_BR_len         # shape (H,)
    E  = clamp01(0.5 * (E1 + E2))                                # shape (H,)

    # Line A per column J: T(J)→B(J)
    # Compute P(I,J) = T(J) + E(I) * (B(J) - T(J)) in vectorized form
    T2 = top[None, :, :]              # (1, W, 2)
    dTB = (bottom - top)[None, :, :]  # (1, W, 2)
    Ecol = E[:, None, None]           # (H, 1, 2) broadcast over W,2
    P = T2 + Ecol * dTB               # (H, W, 2)

    # Map all P to image pixels and paint tiny black dots
    Pimg = to_img_xy(P.reshape(-1, 2))
    Pij = np.round(Pimg).astype(int)
    # Clip
    np.clip(Pij[:, 0], 0, canvas_W - 1, out=Pij[:, 0])
    np.clip(Pij[:, 1], 0, canvas_H - 1, out=Pij[:, 1])
    # Paint: advanced indexing (fast)
    img[Pij[:, 1], Pij[:, 0]] = (0, 0, 0)

    # Rotate for your preferred viewing convention
    rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Save & show
    cv2.imwrite(args.out, rotated)

    t1 = time.time()
    print(f"Saved: {args.out}")
    print(f"Grid size: H={H}, W={W} -> {H*W} points")
    print(f"Runtime: {t1 - t0:.3f} s")

    cv2.imshow("dense_red: full grid P(I,J)", rotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
