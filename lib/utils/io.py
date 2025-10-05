import os
import sys
import argparse
import yaml
from pathlib import Path
import pandas as pd
from kinematics.pose import Pose
import cv2
import numpy as np

def _find_config(dataset_dir: Path) -> Path:
    """Return the path to the first existing YAML config file in dataset_dir."""
    for name in ("config.yaml", "config.yml"):
        candidate = dataset_dir / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No config.yaml or config.yml found in {dataset_dir}")

class IOHandler:
    def __init__(self, read_mode):
        if len(sys.argv) < 2:
            print("Usage: python script.py --dataset <dataset.csv> [--index N]")
            sys.exit(1)

        parser = argparse.ArgumentParser()
        parser.add_argument("--index", type=int, 
                            help="Row index to use from dataset (matches 'index' column)")
        parser.add_argument("--dataset", required=True,
                            help="Path to the dataset directory (e.g., ../../data/e)")
        parser.add_argument("--do_fuse", action="store_true")
        
        self.args = parser.parse_args()

        # Resolve the dataset path relative to the current working directory
        dataset_dir = Path(self.args.dataset).expanduser().resolve()
        if not dataset_dir.is_dir():
            print(f"Dataset directory not found: {dataset_dir}", file=sys.stderr)
            sys.exit(2)

        cfg_path = _find_config(dataset_dir)

        # Attempt to load the config.yaml file
        try:
            with open(cfg_path, 'r') as file:
                self.yaml_data = yaml.safe_load(file)
        except FileNotFoundError:
            cfg_path, "not found"

        self.df = pd.read_csv(os.path.join(self.args.dataset, "data.csv"))

        bgdir = os.path.join(dataset_dir, "bgpat")
        if read_mode and os.path.isdir(bgdir):
            self.bgDir = bgdir
        else:
            self.bgDir = None

    def getDoFuse(self):
        return self.args.do_fuse
    
    def getDataRootDir(self):
        return self.args.dataset
    
    def load_row(self, row):
        p = Pose(x=row["x"], y=row["y"], z=row["z"], roll=row["phi"], pitch=row["theta"], yaw=row["psi"])
        metric_depth_path = os.path.join(self.args.dataset, row["metric_depth"])
        color_path = os.path.join(self.args.dataset, row["color_img"])
        
        # Load data
        metric_depth = np.loadtxt(metric_depth_path, delimiter=',', dtype=np.float32)
        color_img = cv2.imread(color_path)

        if self.bgDir is not None:
            d = os.path.join(self.bgDir, f"bg_{row['index']}.csv")
            bg = np.loadtxt(d, delimiter=',', dtype=np.float32)
        else:
            bg = None
        return p, metric_depth, color_img, bg

    def load(self):
        """
        A generator function that yields Pose, metric depth, and color image for each row in the DataFrame.
        
        Args:
            df (pandas.DataFrame): The dataframe containing the data.
            dataset_path (str): The path to the dataset directory.
        
        Yields:
            Pose, metric_depth (np.array), color_img (np.array)
        """
        if self.args.index:
            while True:
                row = self.df.loc[self.df["index"] == self.args.index].iloc[0]
                p, metric_depth, color_img, bg = self.load_row(row) 
                yield p, metric_depth, color_img, row['index'], bg

        else:
            for _, row in self.df.iterrows():
                p, metric_depth, color_img, bg = self.load_row(row) 
                yield p, metric_depth, color_img, row['index'], bg
