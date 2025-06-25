#!/usr/bin/env python3
"""
compute_dense_flow.py

Compute dense optical-flow magnitude images (grayscale) between sequential frames in a directory,
using OpenCV’s Dual TV-L1 algorithm (one of the most precise dense-flow methods available).

Usage:
    python compute_dense_flow.py /path/to/input_frames_dir /path/to/output_flow_dir

For each pair of consecutive frames (lexicographically sorted by filename), this script computes
the dense optical flow, converts the flow vectors to a magnitude image (normalized to 0–255),
and saves it as a monochrome PNG in the specified output directory. Brighter (whiter) pixels
correspond to larger flow magnitudes (faster motion).
"""

import os
import sys
import argparse

import cv2
import numpy as np

import os, sys, argparse
import cv2
import numpy as np

def parse_args():
    p = argparse.ArgumentParser( 
        description="Compute dense optical‐flow magnitude images (grayscale) between sequential frames." 
    )
    p.add_argument("input_dir",  help="Directory containing sorted input frames.")
    p.add_argument("output_dir", help="Directory where flow images will be written.")
    return p.parse_args()

def get_sorted_image_files(input_dir):
    all_files = sorted(os.listdir(input_dir))
    imgs = []
    for fname in all_files:
        full = os.path.join(input_dir, fname)
        if not os.path.isfile(full):
            continue
        gray = cv2.imread(full, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        imgs.append(fname)
    return imgs

def main():
    args = parse_args()
    input_dir  = args.input_dir
    output_dir = args.output_dir

    if not os.path.isdir(input_dir):
        print(f"Error: '{input_dir}' is not a directory.", file=sys.stderr)
        sys.exit(1)
    os.makedirs(output_dir, exist_ok=True)

    filenames = get_sorted_image_files(input_dir)
    if len(filenames) < 2:
        print("Error: Need at least two images to compute flow.", file=sys.stderr)
        sys.exit(1)

    # Create the TV-L1 solver
    tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()

    # Read the very first grayscale frame
    prev_name = filenames[0]
    prev_im   = cv2.imread(os.path.join(input_dir, prev_name), cv2.IMREAD_GRAYSCALE)
    if prev_im is None:
        print(f"Error: Could not read '{prev_name}'.", file=sys.stderr)
        sys.exit(1)

    # You can tweak this: any flow ≥ MAX_FLOW px/frame → white (255)
    MAX_FLOW = 100.0  

    for idx in range(1, len(filenames)):
        curr_name = filenames[idx]
        curr_im   = cv2.imread(os.path.join(input_dir, curr_name), cv2.IMREAD_GRAYSCALE)
        if curr_im is None:
            print(f"Warning: Skipping unreadable '{curr_name}'.")
            continue

        # Compute TV-L1 flow
        flow = tvl1.calc(prev_im, curr_im, None)
        fx, fy = flow[...,0], flow[...,1]
        mag    = np.sqrt(fx*fx + fy*fy)

        # DEBUG: show raw statistics
        raw_min, raw_max = float(mag.min()), float(mag.max())
        print(f"[DEBUG] {prev_name} → {curr_name}: mag.min={raw_min:.3f}, mag.max={raw_max:.3f}")

        # If truly zero everywhere, leave black; otherwise do a fixed‐max normalization
        if raw_max > 0.0:
            normalized = np.minimum(mag / MAX_FLOW, 1.0)  # clamp to [0,1]
            mag_u8 = (normalized * 255).astype(np.uint8)
        else:
            mag_u8 = np.zeros_like(mag, dtype=np.uint8)

        # Save: e.g. "frame001_flow.png" for prev_name="frame001.png"
        base = os.path.splitext(prev_name)[0]
        out_fname = f"{base}_flow.png"
        cv2.imwrite(os.path.join(output_dir, out_fname), mag_u8)

        # Advance
        prev_im   = curr_im.copy()
        prev_name = curr_name

    print(f"Done. Flow images in '{output_dir}'.")

if __name__ == "__main__":
    main()
