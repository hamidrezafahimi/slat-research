import os, sys
import open3d as o3d
import cv2

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../lib")
from kinematics.pose import Point6, Pose
from projection.helper import project3DAndScale
from utils.conversion import pcm2pcd
from diffusion.helper import downsample_pcd

import numpy as np

from PIL import Image

def load_rgb_png(image_path: str) -> np.ndarray:
    # Returns HxWx3 in [0,1]
    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)
    return (arr.astype(np.float32) / 255.0)

def load_depth_csv(depth_path: str) -> np.ndarray:    
    """Load a depth map from a CSV file (HxW numeric values)."""    
    depth = np.loadtxt(depth_path, delimiter=",")    
    if depth.ndim != 2:        
        raise ValueError(f"Depth CSV must be HxW, got shape {depth.shape}")    
    return depth.astype(np.float32)

def color_from_image(rgb: np.ndarray, vu_idx: np.ndarray) -> np.ndarray:
    """
    Sample RGB colors at pixel indices (v,u). rgb is HxWx3 float32 in [0,1].
    Returns Nx3.
    """
    v = vu_idx[:, 0]
    u = vu_idx[:, 1]
    return rgb[v, u, :].astype(np.float32)

def make_pcd(points: np.ndarray, colors: np.ndarray | None = None) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors is not None:
        # Clamp to [0,1]
        colors = np.clip(colors, 0.0, 1.0)
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    return pcd

def depth_to_points(depth: np.ndarray, fx: float, fy: float, cx: float, cy: float):
    """
    Unproject to camera frame. Returns Nx3 points and Nx2 integer (v,u) pixel indices.
    """
    H, W = depth.shape
    vs, us = np.meshgrid(np.arange(H, dtype=np.float32),
                         np.arange(W, dtype=np.float32),
                         indexing="ij")

    z = depth  # meters
    valid = np.isfinite(z) & (z > 0.0)

    # x = (u - cx)/fx * z ; y = (v - cy)/fy * z
    x = (us - cx) / fx * z
    y = (vs - cy) / fy * z

    Xc = np.stack((x, y, z), axis=-1)  # HxWx3
    pts = Xc[valid]                     # Nx3

    # Keep the pixel indices for color sampling
    vu_idx = np.stack((np.nonzero(valid)[0], np.nonzero(valid)[1]), axis=1)  # Nx2 (v,u)

    return pts, vu_idx

if __name__ == '__main__':
    if len(sys.argv) > 1:
        s = sys.argv[1]
        h = float(sys.argv[2])
        f = sys.argv[3]
        print("Got:", s)
    else:
        print("No input string provided.")

    import os
    if f == "csv":
        p = os.path.join(s, "000000.csv")
        arr = load_depth_csv(p)
        # p = "000000.npy"
        assert os.path.isfile(p), f"Missing file: {p} (check path)"
    elif f == "npy":
        p = os.path.join(s, "000000.npy")
        arr = np.load(p, allow_pickle=True)
        d = os.path.join(s, "000000.csv")
        arr2 = load_depth_csv(d)

    H, W = arr.shape
    center_y, center_x = H // 2, W // 2

    center_value = arr[center_y, center_x]
    print(f"Array shape: {arr.shape}")
    print(f"Center pixel index: ({center_y}, {center_x})")
    print(f"Center depth value: {center_value:.6f} meters")

    print("type:", type(arr))
    try:
        print("shape:", getattr(arr, "shape", None), "dtype:", getattr(arr, "dtype", None))
    except Exception as e:
        print("shape/dtype check failed:", e)

    # If it's a dict-like (pickled), list keys
    if isinstance(arr, dict):
        print("dict keys:", list(arr.keys()))
    elif hasattr(arr, "item") and arr.dtype == object:
        # sometimes saved as a 0-d object array containing a dict
        maybe = arr.item()
        if isinstance(maybe, dict):
            print("object array contained dict keys:", list(maybe.keys()))

    # If it's a numeric array, show quick stats
    if isinstance(arr, np.ndarray) and np.issubdtype(arr.dtype, np.number) and arr.size > 0:
        print("min:", np.nanmin(arr), "max:", np.nanmax(arr))
        print("nonzero:", np.count_nonzero(arr))

    metric_depth = arr

    p6 = Point6(x=0, y=0,z=40,roll=0,pitch=-1.57, yaw=0)
    pose = Pose(p6 = p6)

    cols = cv2.imread(os.path.join(s, "000000.png"))
    fresh_pc, _ = project3DAndScale(metric_depth, pose, h)
    pcd = pcm2pcd(fresh_pc, cols)
    
    asd, _ = project3DAndScale(arr2, pose, h)
    pcd2 = pcm2pcd(asd, np.zeros_like(cols))

    # rgb = load_rgb_png("000000.png")
    # # fx, fy, cx, cy = 4548.913814319164, 4548.913814319164, 2647.233923204192, 1964.0013181957042
    # fx, fy, cx, cy = 4572.614131981836, 4572.614131981836, 2640.0, 1978.0
    # # 4572.614131981836 2640.0 1978.0

    # pts_cam, vu_idx = depth_to_points(metric_depth, fx, fy, cx, cy)
    # cols = color_from_image(rgb, vu_idx)

    # # Transform to world
    # # pts_world = transform_points(pts_cam, T_wc)

    # pcd = make_pcd(pts_cam, cols)




    # import open3d as o3d
    import open3d.visualization.gui as gui


    gui.Application.instance.initialize()
    window = gui.Application.instance.create_window("3D Mapper", 800, 600)
    scene = gui.SceneWidget()
    scene.scene = o3d.visualization.rendering.Open3DScene(window.renderer)
    window.add_child(scene)
    # Create the coordinate frame (three perpendicular arrows)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])
    # Add the coordinate frame to the scene
    scene.scene.add_geometry("axis", axis, o3d.visualization.rendering.MaterialRecord())
    # Set the key event to trigger the 'project' method on 'A'

    def _mat_points(size=4.0):
        m = o3d.visualization.rendering.MaterialRecord()
        m.shader = "defaultUnlit"
        m.point_size = float(size)
        return m

    scene.scene.add_geometry("name", downsample_pcd(pcd, 100000), _mat_points(5.0))
    scene.scene.add_geometry("name2", downsample_pcd(pcd2, 100000), _mat_points(5.0))
    gui.Application.instance.run()