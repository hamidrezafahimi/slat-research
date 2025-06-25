#!/usr/bin/env python3
"""
kp_editor.py (updated)
=====================

Manual key‑point annotation + *auto‑ID* helper.

•  Loads an *image directory* and an **existing** JSON annotation file
   (same format the previous scripts produced).

•  Opens **two fixed windows**:
       “Current Frame”   – the frame you edit
       “Previous Frame”  – the last‑saved frame

•  Controls
   ---------
   Mouse (on *Current Frame* window)
     ◼  **Right‑click**         → add *new* key‑point  (auto‑generated ID)
     ◼  **Left‑click + digits↵**→ correlate to an existing ID.
         –  If the typed ID never appeared before, the point is discarded.

   Keyboard
     s    save current frame & go to next
     n    next frame *without* saving changes
     c↵id↵  clear the given ID from *current* frame
     a    *auto‑ID* mode – click key‑points, then **Enter** to automatically
          assign IDs from the previous frame using nearest‑neighbour search.
          Press **d** or **Esc** to cancel.
     d    (normal mode) delete *all* points of *current* frame
     q    quit (JSON always saved before exit)

•  While you edit a frame, the Previous‑Frame window *live‑updates*:
   any ID that exists in both frames is highlighted **yellow**.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import cv2
import numpy as np

# ── colours (BGR) ────────────────────────────────────────────────────────────
GREEN   = (  0,255,  0)
BLUE    = (255,  0,  0)
YELLOW  = (  0,255,255)
RED     = (  0,  0,255)

RADIUS  = 4
THICK   = -1   # filled circle

WIN_CUR = "Current Frame"
WIN_PRE = "Previous Frame"

# ── helper: assign IDs by nearest neighbour ─────────────────────────────────-

def assign_ids(uvs1: np.ndarray, uvs2: np.ndarray) -> np.ndarray:
    """Assign IDs from *uvs1* to *uvs2* (no ID column) by nearest (u,v).

    Parameters
    ----------
    uvs1 : (N,3) – ``[id, u, v]``
    uvs2 : (M,2) – ``[u, v]`` ( *M* need not equal *N* )

    Returns
    -------
    (M,3)  – ``[id, u, v]`` with IDs borrowed from *uvs1*.
    """
    if uvs1.ndim != 2 or uvs1.shape[1] != 3:
        raise ValueError(f"uvs1 must be (N,3), got {uvs1.shape}")
    if uvs2.ndim != 2 or uvs2.shape[1] != 2:
        raise ValueError(f"uvs2 must be (M,2), got {uvs2.shape}")

    ids1    = uvs1[:,0].astype(int)
    coords1 = uvs1[:,1:3].astype(float)
    coords2 = uvs2.astype(float)

    assigned_ids = np.empty(coords2.shape[0], dtype=int)
    for i, xy in enumerate(coords2):
        diff     = coords1 - xy
        sq_dists = diff[:,0]**2 + diff[:,1]**2
        assigned_ids[i] = ids1[np.argmin(sq_dists)]

    annotated = np.zeros((coords2.shape[0], 3), dtype=float)
    annotated[:,0] = assigned_ids
    annotated[:,1:] = coords2
    return annotated

# ── misc helpers ─────────────────────────────────────────────────────────────

def sorted_images(img_dir: Path) -> list[Path]:
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    imgs = [p for p in img_dir.iterdir() if p.suffix.lower() in exts]
    if not imgs:
        sys.exit(f"[ERR] no images found in {img_dir}")
    return sorted(imgs, key=lambda p: p.name)

def load_annotations(path: Path, n_frames: int) -> dict[int, list[list[float]]]:
    if not path.exists():
        return {}
    with open(path, 'r') as f:
        raw = json.load(f)
    by_frame: dict[int, list[list[float]]] = {}
    for frame_block in raw:
        idx = int(frame_block["frame"])
        if idx < n_frames:
            by_frame[idx] = frame_block["keypoints"]
    return by_frame

def save_annotations(path: Path, data: dict[int, list[list[float]]]) -> None:
    out = [{"frame": k, "keypoints": v} for k,v in sorted(data.items())]
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)

# ── drawing helpers ─────────────────────────────────────────────────────────-

def draw_keypoints(img: np.ndarray,
                   kps: list[list[float]],
                   *,
                   highlight: set[int] | None = None,
                   typing_hint: str | None = None) -> np.ndarray:
    """Return a *copy* with circles & IDs drawn."""
    canvas = img.copy()
    for pid, u, v in kps:
        pid = int(pid)
        colour = GREEN if not (highlight and pid in highlight) else YELLOW
        cv2.circle(canvas, (int(round(u)), int(round(v))), RADIUS, colour, THICK)
        cv2.putText(canvas, str(pid), (int(u)+6, int(v)-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1, cv2.LINE_AA)
    if typing_hint:
        cv2.putText(canvas, typing_hint, (10,25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2, cv2.LINE_AA)
    return canvas

# ── main class holding state ─────────────────────────────────────────────────

class KPTool:
    def __init__(self, imgs: list[Path], ann_in: Path, ann_out: Path) -> None:
        self.imgs      = imgs
        self.n_frames  = len(imgs)
        self.ann_path  = ann_out

        # frame_idx -> [[id,u,v], ...]
        self.data      = load_annotations(ann_in, self.n_frames)
        self.curr_idx  = 0
        self.curr_pts  = [p[:] for p in self.data.get(0, [])]

        # typing / mode state
        self.assign_idx: int | None = None   # index awaiting id typing
        self.mode: str | None = None         # None | "assign" | "clear" | "auto"
        self.buffer: str = ""

        # auto‑mode buffer
        self.auto_pts: list[tuple[float,float]] = []

        # next fresh pid
        self.next_pid = (max((kp[0] for kps in self.data.values() for kp in kps),
                             default=-1) + 1)

        # GUI windows
        flags = cv2.WINDOW_NORMAL
        if hasattr(cv2, "WINDOW_GUI_NORMAL"):
            flags |= cv2.WINDOW_GUI_NORMAL
        cv2.namedWindow(WIN_CUR, flags)
        cv2.namedWindow(WIN_PRE, flags)
        cv2.setMouseCallback(WIN_CUR, self.on_click)

    # ── convenience properties ───────────────────────────────────────
    @property
    def prev_pts(self) -> list[list[float]]:
        return self.data.get(self.curr_idx-1, []) if self.curr_idx else []

    @property
    def prev_curr_overlap(self) -> set[int]:
        return {int(pid) for pid, *_ in self.curr_pts} & \
               {int(pid) for pid, *_ in self.prev_pts}

    # ── mouse callback ───────────────────────────────────────────────
    def on_click(self, event, x, y, flags, param) -> None:
        if self.mode in ("assign", "clear"):   # ignore clicks while typing
            return
        if self.mode == "auto":  # collect points for auto‑ID
            if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
                self.auto_pts.append((float(x), float(y)))
                self.refresh()
                return

        # normal modes ------------------------------------------------
        if event == cv2.EVENT_RBUTTONDOWN:
            # new key‑point with fresh ID
            self.curr_pts.append([int(self.next_pid), float(x), float(y)])
            self.next_pid += 1
            self.refresh()
        elif event == cv2.EVENT_LBUTTONDOWN:
            # add a temp point, then wait for ID typing
            self.curr_pts.append([-1, float(x), float(y)])
            self.assign_idx = len(self.curr_pts)-1
            self.mode, self.buffer = "assign", ""
            self.refresh()

    # ── typing handlers ──────────────────────────────────────────────
    def handle_typing(self, key: int):
        """Typing while in *assign* or *clear* mode."""
        if key in (13,10):      # Enter
            if self.buffer == "":
                # cancel
                if self.mode == "assign" and self.assign_idx is not None:
                    self.curr_pts.pop(self.assign_idx)
            else:
                pid = int(self.buffer)
                if self.mode == "assign":
                    seen_before = any(pid in (int(k[0]) for k in self.data.get(f, []))
                                       for f in range(self.curr_idx))
                    if seen_before and all(pid != int(k[0]) for k in self.curr_pts[:-1]):
                        self.curr_pts[self.assign_idx][0] = pid
                    else:
                        # invalid ➜ discard placeholder
                        self.curr_pts.pop(self.assign_idx)
                elif self.mode == "clear":
                    self.curr_pts = [k for k in self.curr_pts if int(k[0]) != pid]
            # reset
            self.mode, self.assign_idx, self.buffer = None, None, ""
            self.refresh(); return

        if key in (27,):        # Esc
            if self.mode == "assign" and self.assign_idx is not None:
                self.curr_pts.pop(self.assign_idx)
            self.mode, self.assign_idx, self.buffer = None, None, ""
            self.refresh(); return

        # digits / backspace
        if 48 <= key <= 57:
            self.buffer += chr(key); self.refresh()
        elif key in (8,127):
            self.buffer = self.buffer[:-1]; self.refresh()

    # ── auto‑mode key handler ────────────────────────────────────────
    def handle_auto(self, key: int):
        if key in (13,10):          # Enter → run auto assignment
            self.run_auto_assign()
        elif key in (27, ord('d')): # Esc or d → cancel
            self.mode = None; self.auto_pts.clear(); self.refresh()

    def run_auto_assign(self):
        if not self.auto_pts or not self.prev_pts:
            # nothing to do
            self.mode = None; self.auto_pts.clear(); self.refresh(); return

        uvs1 = np.array(self.prev_pts, dtype=float)          # (P,3)
        uvs2 = np.array(self.auto_pts, dtype=float)          # (K,2)
        try:
            annotated = assign_ids(uvs1, uvs2)               # (K,3)
        except Exception as e:
            print(f"\n[WARN] auto‑ID failed: {e}")
            self.mode = None; self.auto_pts.clear(); self.refresh(); return

        # merge results, avoiding duplicate IDs within current frame
        existing_ids = {int(k[0]) for k in self.curr_pts}
        for pid,u,v in annotated:
            if int(pid) not in existing_ids:
                self.curr_pts.append([int(pid), float(u), float(v)])
        self.auto_pts.clear(); self.mode = None; self.refresh()

    # ── render windows ───────────────────────────────────────────────
    def refresh(self):
        img_cur = cv2.imread(str(self.imgs[self.curr_idx]))
        img_pre = cv2.imread(str(self.imgs[self.curr_idx-1])) if self.curr_idx else np.zeros_like(img_cur)

        hint = None
        if self.mode == "assign":
            hint = f"ID: {self.buffer}"
        elif self.mode == "clear":
            hint = f"Clear ID: {self.buffer}"
        elif self.mode == "auto":
            hint = "AUTO – click pts, Enter=go, d/Esc=cancel"

        img_cur_d = draw_keypoints(img_cur, self.curr_pts, typing_hint=hint)
        # draw auto‑mode clicks in blue
        if self.mode == "auto":
            for u,v in self.auto_pts:
                cv2.circle(img_cur_d, (int(round(u)), int(round(v))), RADIUS, BLUE, THICK)

        img_pre_d = draw_keypoints(img_pre, self.prev_pts, highlight=self.prev_curr_overlap)

        cv2.imshow(WIN_CUR, img_cur_d)
        cv2.imshow(WIN_PRE, img_pre_d)
        print(f"\rFrame {self.curr_idx+1}/{self.n_frames} – {len(self.curr_pts)} pts   ", end="")

    # ── main loop ────────────────────────────────────────────────────
    def loop(self):
        self.refresh()
        while True:
            key = cv2.waitKey(20) & 0xFF

            if self.mode == "auto":
                self.handle_auto(key); continue
            if self.mode in ("assign", "clear"):
                self.handle_typing(key); continue

            # normal key map ---------------------------------------------------
            if key == ord('q'):
                self.data[self.curr_idx] = [p[:] for p in self.curr_pts]
                save_annotations(self.ann_path, self.data); break
            elif key == ord('s'):
                self.data[self.curr_idx] = [p[:] for p in self.curr_pts]
                self.advance()
            elif key == ord('n'):
                self.advance()
            elif key == ord('c'):
                self.mode = "clear"; self.buffer = ""; self.refresh()
            elif key == ord('d'):
                # delete all pts this frame
                self.curr_pts = []; self.refresh()
            elif key == ord('a'):
                # enter auto mode
                self.mode = "auto"; self.auto_pts.clear(); self.refresh()

        cv2.destroyAllWindows()

    def advance(self):
        if self.curr_idx < self.n_frames-1:
            self.curr_idx += 1
            self.curr_pts = [p[:] for p in self.data.get(self.curr_idx, [])]
            self.mode = None
            self.refresh()
        else:
            save_annotations(self.ann_path, self.data)
            print("\nAll frames processed. Annotations saved.")
            cv2.waitKey(500); cv2.destroyAllWindows(); sys.exit(0)

# ── entry‑point ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("img_dir",  type=Path, help="directory with image sequence")
    ap.add_argument("json_in",  type=Path, help="existing JSON key‑points file")
    ap.add_argument("json_out", type=Path, nargs='?', default=None,
                    help="output (merged) JSON")
    args = ap.parse_args()

    imgs = sorted_images(args.img_dir)
    json_out = Path(args.json_out) if args.json_out else args.json_in

    KPTool(imgs, args.json_in, json_out).loop()

if __name__ == "__main__":
    main()
