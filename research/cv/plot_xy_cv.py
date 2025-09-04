import json, cv2, numpy as np, argparse

def main():
    ap = argparse.ArgumentParser(description="Plot XY projection of dense_red (OpenCV).")
    ap.add_argument("--json", default="metric_pattern_dense_red.json", help="Path to JSON with top/right/bottom/left arrays")
    ap.add_argument("--out", default="xy_dense_red_opencv.png", help="Output image path")
    ap.add_argument("--max_w", type=int, default=1400)
    ap.add_argument("--max_h", type=int, default=1000)
    ap.add_argument("--margin", type=int, default=40)
    args = ap.parse_args()

    with open(args.json, "r") as f:
        data = json.load(f)

    sides = ["top", "right", "bottom", "left"]
    pts_dict = {k: np.array(data[k], dtype=float)[:, :2] for k in sides}
    all_xy = np.vstack([pts_dict[k] for k in sides])

    min_xy = all_xy.min(axis=0)
    max_xy = all_xy.max(axis=0)
    rng = np.maximum(max_xy - min_xy, 1e-9)

    sx = (args.max_w - 2*args.margin) / rng[0]
    sy = (args.max_h - 2*args.margin) / rng[1]
    scale = min(sx, sy)

    W = int(rng[0] * scale + 2*args.margin)
    H = int(rng[1] * scale + 2*args.margin)
    img = np.full((H, W, 3), 255, np.uint8)

    colors = {
        "top":    (0, 0, 255),   # red
        "right":  (0, 200, 0),   # green
        "bottom": (255, 0, 0),   # blue
        "left":   (200, 0, 200)  # magenta
    }

    def to_img_xy(xy):
        norm = (xy - min_xy) * scale + args.margin
        norm[:, 1] = H - norm[:, 1]   # flip Y for image coordinates
        return norm.astype(int)

    for k in sides:
        img_pts = to_img_xy(pts_dict[k].copy())
        for p in img_pts:
            cv2.circle(img, tuple(p), 2, colors[k], -1, lineType=cv2.LINE_AA)
        if len(img_pts) >= 2:
            cv2.polylines(img, [img_pts.reshape(-1,1,2)], False, colors[k], 1, cv2.LINE_AA)

    # Simple legend
    cv2.rectangle(img, (10,10), (490, 80), (255,255,255), -1)
    cv2.rectangle(img, (10,10), (490, 80), (0,0,0), 1)
    cv2.putText(img, "XY Projection of dense_red borders", (20,35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
    cv2.putText(img, "top/red | right/green | bottom/blue | left/magenta", (20,65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

    rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imwrite(args.out, rotated)
    print(f"Saved: {args.out} ({W}x{H})")

if __name__ == "__main__":
    main()
