#!/usr/bin/env python3
"""Key‑point tracker with periodic key‑frame corrections.

Given a directory of image frames whose filenames are *zero‑padded* frame
numbers (e.g. ``00000087.jpg``) **and** a JSON file that stores *ground‑truth*
key‑points at six sparsely‑sampled frames, this script tracks every point
through *all* intermediate images with LK optical flow.  When the current
frame number reaches one of the annotated key‑frames, the tracked points are
*snapped* back to the ground‑truth locations (ids are matched), and any new
ids that appear in the JSON are inserted into the active set.  The resulting
per‑frame list of ``[id, u, v]`` tuples is written back to a new JSON file
whose structure mirrors the input example:

```
[
  {"frame": 0, "keypoints": [[id, u, v], …]},
  {"frame": 1, "keypoints": …},
  …
]
```

Usage (example) ::

    python keypoint_tracker.py \
        --frames_dir /media/d/Workspace/viot3/frames \
        --gt_json    /media/d/Workspace/viot3/out5.json \
        --out_json   /media/d/Workspace/viot3/full_track.json

Notes
-----
* The hard‑coded list ``CORRECTION_FRAMES`` describes **which** absolute
  frame numbers the six JSON entries refer to.  Adjust it if your data uses a
  different mapping.
* The very first frame processed **must** be the first element of
  ``CORRECTION_FRAMES`` (87 in the sample data).
* Lost tracks (LK status == 0) are silently dropped.
"""

import argparse
import json
import os
import re
import glob
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Mapping between the six entries in the JSON file and their absolute frame
# numbers on disk (edit if necessary)
CORRECTION_FRAMES = [87, 105, 114, 123, 127, 130]  # → JSON frames 0‥5
FIRST_FRAME = CORRECTION_FRAMES[0]

# LK optical‑flow parameters (reasonable defaults for 480 p imagery)
LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)

# ──────────────────────────────────────────────────────────────────────────────

def natural_key(p: Path) -> int:
    """Return the *numeric* part of a zero‑padded filename for sorting."""
    m = re.search(r"(\d+)", p.stem)
    if not m:
        raise ValueError(f"No digits found in filename: {p.name}")
    return int(m.group(1))


def load_corrections(gt_json: Path) -> Dict[int, Dict[int, Tuple[float, float]]]:
    """Read the six key‑frames from *gt_json* and map ➜ ``frame_no → {id: (u,v)}``."""
    with open(gt_json, "r", encoding="utf‑8") as f:
        data = json.load(f)

    if len(data) != len(CORRECTION_FRAMES):
        raise ValueError(
            f"Expected {len(CORRECTION_FRAMES)} correction frames, "
            f"found {len(data)} in {gt_json}")

    corrections: Dict[int, Dict[int, Tuple[float, float]]] = {}
    for entry in data:
        local_idx = entry["frame"]               # 0‥5 in sample JSON
        frame_no  = CORRECTION_FRAMES[local_idx]  # absolute frame number
        kps = {int(kp[0]): (float(kp[1]), float(kp[2])) for kp in entry["keypoints"]}
        corrections[frame_no] = kps
    return corrections


def main():
    parser = argparse.ArgumentParser(description="Track key‑points with periodic corrections.")
    parser.add_argument("--frames_dir", required=True, type=Path,
                        help="Directory with JPEG frames named ######.jpg")
    parser.add_argument("--gt_json",    required=True, type=Path,
                        help="Ground‑truth JSON with six key‑frames")
    parser.add_argument("--out_json",   required=True, type=Path,
                        help="Destination JSON for the full track")
    args = parser.parse_args()

    # ── gather and sort all image paths ────────────────────────────────────
    img_paths = sorted(Path(args.frames_dir).glob("*.jpg"), key=natural_key)
    frame_numbers = [natural_key(p) for p in img_paths]

    if FIRST_FRAME not in frame_numbers:
        raise RuntimeError(f"First required frame {FIRST_FRAME} not found in {args.frames_dir}")

    start_idx = frame_numbers.index(FIRST_FRAME)
    img_paths   = img_paths[start_idx:]      # discard earlier frames, if any
    frame_numbers = frame_numbers[start_idx:]

    # ── load the six correction snapshots ───────────────────────────────────
    corrections = load_corrections(args.gt_json)

    # ── tracking state ─────────────────────────────────────────────────────
    active_pts: Dict[int, Tuple[float, float]] = {}
    track_out: List[Dict] = []

    prev_gray = None  # previous grayscale frame

    for frame_idx, (frame_no, img_path) in enumerate(zip(frame_numbers, img_paths)):
        frame_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise RuntimeError(f"Failed to load image: {img_path}")
        curr_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # ── initialise or propagate key‑points ────────────────────────────
        if frame_idx == 0:
            # Initial frame *must* have correction data
            active_pts = corrections[frame_no].copy()
        else:
            if not active_pts:
                raise RuntimeError("No active key‑points left to track → nothing to do")

            # Prepare data for LK: shape (N,1,2)
            ids  = np.fromiter(active_pts.keys(), dtype=np.int32)
            prev = np.array(list(active_pts.values()), dtype=np.float32).reshape(-1, 1, 2)

            nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev, None, **LK_PARAMS)
            st = st.reshape(-1)

            # Update only the successfully tracked points
            active_pts = {int(i): tuple(pt) for i, pt, ok in zip(ids, nxt.reshape(-1, 2), st) if ok}

        # ── snap back to ground‑truth on correction frames ─────────────────
        if frame_no in corrections:
            for pid, (u_gt, v_gt) in corrections[frame_no].items():
                active_pts[pid] = (u_gt, v_gt)  # insert or overwrite

        # ── store results for current frame ───────────────────────────────
        frame_entry = {
            "frame": frame_idx,  # zero‑based relative index (like the example JSON)
            "keypoints": [[pid, float(u), float(v)] for pid, (u, v) in active_pts.items()],
        }
        track_out.append(frame_entry)

        # ── prepare for next iteration ────────────────────────────────────
        prev_gray = curr_gray

    # ── write output JSON ─────────────────────────────────────────────────
    with open(args.out_json, "w", encoding="utf‑8") as f:
        json.dump(track_out, f, indent=2)
    print(f"✓ Wrote {len(track_out)} frames to {args.out_json}")


if __name__ == "__main__":
    main()
