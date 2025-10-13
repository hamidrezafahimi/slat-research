import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/..")
import numpy as np
import open3d as o3d
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy as np
from .config import Scaling
import cv2

def calc_scale_factor(desired_altitude, scaling, pc_to_be_rescaled=None, bgz=None):
    if scaling == Scaling.NULL:
        assert pc_to_be_rescaled is not None
        min_z = np.nanmin(pc_to_be_rescaled[:,:,2])
        return -30 / min_z
    elif scaling == Scaling.MIN_Z:
        assert pc_to_be_rescaled is not None
        min_z = np.nanmin(pc_to_be_rescaled[:,:,2])
        return desired_altitude / min_z
    elif scaling == Scaling.MEAN_Z:
        assert bgz is not None, "MEAN_Z Scaling requires bgz provided"
        mean_z = np.mean(bgz)
        return desired_altitude / mean_z
    elif scaling == Scaling.RESHAPE_BG_Z:
        assert bgz is not None, "RESHAPE_BG_Z Scaling requires bgz provided" 

def project3D(depth_img, pose, hfov_deg, scaling, bg=None, move=False, pyramidProj=False):
    pc, dirs = depthImage2pointCloud(depth_img, hfov_deg, pose, pyramidProj=pyramidProj)
    scale_factor = calc_scale_factor(-abs(pose.p6.z), scaling, bgz=bg, pc_to_be_rescaled=pc)
    pc, dirs = depthImage2pointCloud(depth_img,hfov_deg, pose, scale_factor=scale_factor, 
                                     pyramidProj=pyramidProj)
    if move:
        p = np.array([[pose.p6.x], [pose.p6.y], [pose.p6.z]])
        move_const = p.T
        pc += move_const
    else:
        move_const = None
    return pc, move_const

def depthImage2pointCloud(D, horizontal_fov, p, scale_factor = 1, pyramidProj=False):
    """
    Computes a point cloud from a depth image 
    """
    H, W = D.shape
    hfov_rad = np.radians(horizontal_fov)
    focal_length = (W / 2) / np.tan(hfov_rad / 2)
    cx, cy = W / 2, H / 2
    # Generate direction vectors in camera frame
    x_idxs = np.arange(W)
    y_idxs = np.arange(H)
    x_grid, y_grid = np.meshgrid(x_idxs, y_idxs)
    X = (x_grid - cx) / focal_length
    Y = (y_grid - cy) / focal_length
    Z = np.ones_like(X)
    if not pyramidProj:
        norm = np.sqrt(X**2 + Y**2 + Z**2)
        X /= norm; Y /= norm; Z /= norm
    dirs = np.stack((X, Y, Z), axis=-1)  # (H, W, 3)
    # Rotate direction vectors into NWU frame
    dirs_nwu = dirs @ p.getCAM2NWU().T
    # Scale by depth and altitude
    D2 = D * scale_factor
    pc1 = dirs_nwu * (D2[..., np.newaxis])
    return pc1, dirs_nwu

def transform_depth(depth_image, bg_image, gep_image):
    assert depth_image.dtype == np.float32 and bg_image.dtype == np.float32 and \
        gep_image.dtype == np.float32
    # Transformation
    trns = gep_image / np.where(bg_image != 0, bg_image, 0.01)
    final_f = trns * depth_image
    final_f = np.minimum(final_f, gep_image)
    return final_f

def arg_min_2d(arr):
    amin = np.argmin(arr)
    w = arr.shape[1]
    return (int(np.floor(amin / w)), (amin + 1) % w - 1)

def arg_max_2d(arr):
    am = np.argmax(arr)
    w = arr.shape[1]
    return (int(np.floor(am / w)), (am + 1) % w - 1)

def unified_scale(foreground: np.ndarray, background: np.ndarray):
    min_fg = np.min(foreground)
    max_bg = np.max(background)
    assert min_fg <= np.min(background), "Foreground must contain the global minimum"
    assert max_bg >= np.max(foreground), "Background must contain the global maximum"
    fg_flat = foreground.flatten()
    bg_flat = background.flatten()
    combined = np.concatenate([fg_flat, bg_flat])
    max_val = np.max(combined)
    if max_val == 0:
        raise ValueError("Maximum value is zero; cannot scale.")
    scaled_combined = combined * 255.0 / max_val
    # Split back
    fg_scaled = scaled_combined[:fg_flat.size].reshape(foreground.shape)
    bg_scaled = scaled_combined[fg_flat.size:].reshape(background.shape)
    return fg_scaled, bg_scaled

def interp_2d(metric_depth, mask, plot=False):
    mask_bool = np.where(mask > 127, False, True)
    Z_masked = np.where(mask_bool, metric_depth, np.nan) # Replace masked values with NaN
    x_coords = np.linspace(0, mask.shape[1], mask.shape[1])
    y_coords = np.linspace(0, mask.shape[0], mask.shape[0])
    X_orig, Y_orig = np.meshgrid(x_coords, y_coords)
    valids = ~np.isnan(Z_masked)
    x_s = X_orig[valids]
    y_s = Y_orig[valids]
    z_s = metric_depth[valids]
    xi, yi = np.meshgrid(np.linspace(0, mask.shape[1], mask.shape[1]), np.linspace(0, mask.shape[0], mask.shape[0]))
    # # 'kind' can be 'linear', 'cubic', or 'quintic'
    zi = griddata((x_s,y_s), z_s, (xi, yi), method='linear')
    if plot:
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(111, projection='3d')
        ax1.plot_surface(xi, yi, zi, cmap='viridis')
        # ax1.plot_surface(X_orig, Y_orig, Z_masked, cmap='viridis')
        # print(x_coords.shape, y_coords.shape, Z_masked.shape)
        plt.tight_layout()
        plt.show()
    return zi

def resize_keep_ar(img, desired_width):
    """
    Resize an image to a desired width while preserving aspect ratio (AR).
    
    Args:
        img (np.ndarray): Input image (OpenCV format).
        desired_width (int): Desired width in pixels.
        
    Returns:
        np.ndarray: Resized image with preserved AR.
    """
    h, w = img.shape[:2]
    aspect_ratio = h / w
    new_height = int(desired_width * aspect_ratio)
    resized = cv2.resize(img, (desired_width, new_height), interpolation=cv2.INTER_AREA)
    return resized



# def project3DAndScale2(depth_img, pose, hfov_deg, shape):
#     pc, dirs = depthImage2pointCloud(depth_img, roll_rad=pose._roll_rad, 
#                                      pitch_rad=pose._pitch_rad, yaw_rad=pose._yaw_rad,
#                                      horizontal_fov=hfov_deg)
#     gpc = np.ones(shape).astype(np.float32) * (-abs(pose.z))
#     scale_factor = calc_scale_factor(gpc, pc)
#     pc_scaled, dirs = depthImage2pointCloud(depth_img, roll_rad=pose._roll_rad, 
#                                             pitch_rad=pose._pitch_rad, yaw_rad=pose._yaw_rad,
#                                     horizontal_fov=hfov_deg, scale_factor=scale_factor)
#     return pc_scaled


