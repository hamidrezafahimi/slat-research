import sys
import json
import numpy as np
from matplotlib import pyplot as plt
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../lib")
from utils.plotting import plot_3d_kps, get_new_ax3d
import sys
import json
import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Plot the last-frame VO & GT points (camera-frame) from a VO JSON file."
    )
    parser.add_argument(
        'json_path',
        help="Path to the VO output JSON (containing 'vo_results' and 'noise_disp_ratio')."
    )
    parser.add_argument(
        '--title',
        type=str,
        default=None,
        help="Optional custom title to prepend (overrides filename-based title)."
    )
    args = parser.parse_args()

    # Load JSON data
    with open(args.json_path, 'r') as f:
        data = json.load(f)

    # Print noise_disp_ratio if present
    ratio = data.get("noise_disp_ratio", None)
    if ratio is not None:
        print(f"noise_std / avg_disp ratio: {ratio}")
    else:
        print("No 'noise_disp_ratio' found in JSON.")

    vo_list = data.get("vo_results", [])
    if not vo_list:
        print(f"No 'vo_results' found or list is empty in {args.json_path}.")
        sys.exit(1)

    last_entry = vo_list[-1]
    frame_idx = last_entry["frame"]

    # Convert to NumPy arrays
    pts_vo = np.array(last_entry["vo_points"], dtype=np.float32)
    pts_gt = np.array(last_entry["gt_points"], dtype=np.float32)

    # Build a default title from the filename if not provided
    if args.title:
        plot_title = args.title
    else:
        # e.g. vo_ts3_ns1.0.json → "TS=3, NS=1.0, frame=F"
        base = args.json_path.rstrip('/').split('/')[-1].replace('.json','')
        parts = base.split('_')
        ts_str = next((p.replace('ts','') for p in parts if p.startswith('ts')), 'N/A')
        ns_str = next((p.replace('ns','') for p in parts if p.startswith('ns')), 'N/A')
        plot_title = f"TS={ts_str}, NS={ns_str}, frame={frame_idx}"

    # Create a new 3D axis
    ax = get_new_ax3d()

    # 1) Plot ground truth camera-frame points (blue), clearing the axes
    plot_3d_kps(
        pts_gt,
        title=plot_title,
        ax=ax,
        pts_color='blue',
        write_id=True,
        label="GT",
        camera_trajectory=None,
        plt_pause=0,
        plt_show=False,  # wait until after overlay
        cla=True
    )

    # 2) Overlay VO points (green) on the same axes
    plot_3d_kps(
        pts_vo,
        ax=ax,
        title=plot_title,
        pts_color='green',
        write_id=True,
        label="VO",
        camera_trajectory=None,
        plt_pause=0,
        plt_show=False,
        cla=False
    )

    # Register a key-press handler so that pressing 'q' closes the figure
    fig = ax.get_figure()
    def on_key(event):
        if event.key == 'q':
            plt.close(fig)
    fig.canvas.mpl_connect('key_press_event', on_key)

    # Finally, show the combined plot. Blocks until window closes.
    plt.show()


if __name__ == "__main__":
    main()


