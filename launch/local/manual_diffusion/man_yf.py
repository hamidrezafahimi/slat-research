#!/usr/bin/env python3
"""
Spline editor (800 × 600 canvas) with coloured control points and
pattern‑vs‑image overlay.

Changes
=======
* **Endpoints always red** – and **no horizontal stripe** is drawn for them.
* Interior control points get **unique random colours** (never repeat, never
  red). Horizontal stripes in the Pattern window use exactly the control
  point’s colour and are 4 px thick.
* **s** key still saves:
  * **pattern.png** – greyscale threshold resized to reference‐image size.
  * **mask.png** – 0/255 binary mask where `ref_gray > pattern_gray`.
"""
from __future__ import annotations
import cv2
import numpy as np
import argparse
import sys
from pathlib import Path

# ── SciPy cubic spline (optional) ───────────────────────────────────────────
try:
    from scipy.interpolate import CubicSpline
    HAS_SCIPY = True
except ModuleNotFoundError:
    HAS_SCIPY = False

# ── constants ────────────────────────────────────────────────────────────────
CANVAS_W, CANVAS_H = 800, 600
RED          = (0, 0, 255)  # BGR red
CP_RADIUS    = 6
HIT_THRESH   = 10
MERGE_THRESH = 3
SAVE_PATTERN_PATH = "pattern_y.png"
SAVE_MASK_PATH    = "mask_y.png"

# ── helper functions ─────────────────────────────────────────────────────────

def rand_unique_color(existing: set[tuple[int, int, int]]) -> tuple[int, int, int]:
    """Return a random BGR colour not in *existing* and not red."""
    while True:
        c = tuple(int(x) for x in np.random.randint(0, 256, size=3))
        if c != RED and c not in existing:
            return c


def evaluate_curve(ctrl_pts, width):
    """Return integer xs, ys of spline/polyline across columns 0‥width‑1."""
    ctrl_sorted = sorted(ctrl_pts, key=lambda d: d['x'])
    xs_ctrl = np.array([p['x'] for p in ctrl_sorted])
    ys_ctrl = np.array([p['y'] for p in ctrl_sorted])
    xs = np.arange(width)
    if HAS_SCIPY and len(ctrl_sorted) >= 3:
        ys = CubicSpline(xs_ctrl, ys_ctrl, bc_type="natural")(xs)
    else:
        ys = np.interp(xs, xs_ctrl, ys_ctrl)
    ys = np.clip(ys, 0, CANVAS_H - 1)
    return xs.astype(np.int32), ys.astype(np.int32)


def make_pattern_gray(ys):
    h_from_bottom = (CANVAS_H - 1 - ys).astype(np.float32)
    intens = (h_from_bottom / (CANVAS_H - 1) * 255).astype(np.float32)
    return intens  # length == CANVAS_W

# ── main ─────────────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser("Spline editor with unique coloured points")
    parser.add_argument("reference", help="reference (parsed) image – mandatory")
    parser.add_argument("pattern", help="reference pattern")
    args = parser.parse_args(argv)

    # ↓ reference image ---------------------------------------------------
    ref_path = Path(args.reference)
    if not ref_path.is_file():
        sys.exit(f"File '{ref_path}' not found.")
    ref_img_orig_ = np.loadtxt(str(ref_path), delimiter=',')
    ref_img_orig = ref_img_orig_.T

    loaded_pattern = cv2.imread(str(args.pattern), cv2.IMREAD_GRAYSCALE).T

    if ref_img_orig is None:
        sys.exit("Failed to load reference image.")
    ref_h, ref_w = ref_img_orig.shape[:2]

    # ↓ spline state ------------------------------------------------------
    ctrl_pts = [
        {'x': 0,            'y': CANVAS_H // 2, 'color': RED},
        {'x': CANVAS_W - 1, 'y': CANVAS_H // 2, 'color': RED},
    ]
    selected_idx: int | None = None
    offset_xy = (0, 0)

    cv2.namedWindow("Spline Editor", cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
    cv2.namedWindow("Pattern", cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)

    def on_mouse(event, x, y, flags, _):
        nonlocal selected_idx, offset_xy, ctrl_pts
        if event == cv2.EVENT_RBUTTONDOWN:
            _, ys_curve = evaluate_curve(ctrl_pts, CANVAS_W)
            if abs(y - ys_curve[x]) <= HIT_THRESH:
                colours_in_use = {cp['color'] for cp in ctrl_pts}
                new_col = rand_unique_color(colours_in_use)
                ctrl_pts.append({'x': x, 'y': int(ys_curve[x]), 'color': new_col})
                ctrl_pts.sort(key=lambda d: d['x'])
        elif event == cv2.EVENT_LBUTTONDOWN:
            for i, cp in enumerate(ctrl_pts):
                if abs(x - cp['x']) <= HIT_THRESH and abs(y - cp['y']) <= HIT_THRESH:
                    selected_idx = i
                    offset_xy = (x - cp['x'], y - cp['y'])
                    break
        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
            if selected_idx is None:
                return
            new_x = x - offset_xy[0]
            new_y = y - offset_xy[1]
            new_y = int(np.clip(new_y, 0, CANVAS_H - 1))
            cp = ctrl_pts[selected_idx]
            if selected_idx in (0, len(ctrl_pts) - 1):  # endpoints: x locked
                cp['y'] = new_y
                return
            x_prev = ctrl_pts[selected_idx - 1]['x'] + 1
            x_next = ctrl_pts[selected_idx + 1]['x'] - 1
            new_x = int(np.clip(new_x, x_prev, x_next))
            cp['x'], cp['y'] = new_x, new_y
            # merge if neighbouring points too close
            if new_x - ctrl_pts[selected_idx - 1]['x'] < MERGE_THRESH:
                ctrl_pts.pop(selected_idx)
                selected_idx = None
            elif ctrl_pts[selected_idx + 1]['x'] - new_x < MERGE_THRESH:
                ctrl_pts.pop(selected_idx + 1)
                selected_idx = None
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            selected_idx = None

    cv2.setMouseCallback("Spline Editor", on_mouse)

    # ↓ main loop ---------------------------------------------------------
    canvas = np.full((CANVAS_H, CANVAS_W, 3), 255, np.uint8)
    while True:
        xs, ys = evaluate_curve(ctrl_pts, CANVAS_W)
        patt_gray_cols = make_pattern_gray(ys)  # length 800

        # ── draw canvas ─────────────────────────────────────────────────
        canvas[:] = 255
        cv2.polylines(canvas, [np.column_stack((xs, ys)).reshape(-1, 1, 2)], False, (0, 0, 0), 2)
        for cp in ctrl_pts:
            cv2.circle(canvas, (cp['x'], cp['y']), CP_RADIUS, cp['color'], -1)
        cv2.putText(canvas, f"degree = {max(len(ctrl_pts)-1,1)}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)

        # ── build pattern display image (overlay) ───────────────────────
        # resize pattern gray to reference size
        patt_gray_img = cv2.resize(patt_gray_cols[:, None].repeat(CANVAS_H, axis=1), (ref_w, ref_h), interpolation=cv2.INTER_LINEAR)
        arr = patt_gray_img + loaded_pattern.astype(np.float32) - 127.0
        patt_gray_img = np.where(arr <= 0, 0, arr)
        patt_gray_img = np.where(patt_gray_img >= 255, 255, patt_gray_img)
        # patt_gray_img = patt_gray_img.astype(np.uint8)
        cv2.imshow("pattern", patt_gray_img.astype(np.uint8))
        
        mask_bool = ref_img_orig > patt_gray_img
        overlay = cv2.cvtColor(ref_img_orig.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        overlay[mask_bool] = RED
        # add coloured horizontal stripes for interior points
        for cp in ctrl_pts[1:-1]:  # skip endpoints
            y_line = int(cp['x'] * ref_h / CANVAS_W)
            cv2.line(overlay, (0, y_line), (ref_w - 1, y_line), cp['color'], thickness=4, lineType=cv2.LINE_AA)

        cv2.imshow("Spline Editor", canvas)
        # print((0.25 * overlay.shape[1], 0.25 * overlay.shape[0]))
        cv2.imshow("Pattern", overlay)
        # cv2.imshow("Pattern", cv2.resize(overlay,
        #                                  (int(0.25 * overlay.shape[1]), int(0.25 * overlay.shape[0])),
        #                                  interpolation=cv2.INTER_LINEAR))

        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord('q')):
            break
        if key == ord('r'):
            ctrl_pts = [
                {'x': 0,            'y': CANVAS_H // 2, 'color': RED},
                {'x': CANVAS_W - 1, 'y': CANVAS_H // 2, 'color': RED},
            ]
        if key == ord('s'):
            # save pattern and mask (no stripes) at reference resolution
            patt_gray_ref = cv2.resize(patt_gray_cols[:, None].repeat(CANVAS_H, axis=1), (ref_w, ref_h), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(SAVE_PATTERN_PATH, patt_gray_img.T)
            cv2.imwrite(SAVE_MASK_PATH, (mask_bool.astype(np.uint8).T * 255))
            print(f"Saved pattern to {SAVE_PATTERN_PATH}, mask to {SAVE_MASK_PATH}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
