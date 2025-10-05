import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/..")
import numpy as np
import open3d as o3d
from scipy.interpolate import griddata, LinearNDInterpolator
import matplotlib.pyplot as plt
import numpy as np
from kinematics.transformations import rotation_matrix_x, rotation_matrix_y, rotation_matrix_z

def calc_scale_factor(desired_altitude, pc_to_be_rescaled):
    min_z = np.nanmin(pc_to_be_rescaled[:,:,2])
    return desired_altitude / min_z

def project3DAndScale(depth_img, pose, hfov_deg, move=False):
    pc, dirs = depthImage2pointCloud(depth_img, roll_rad=pose._roll_rad, 
                                     pitch_rad=pose._pitch_rad, yaw_rad=pose._yaw_rad,
                                     horizontal_fov=hfov_deg)
    scale_factor = calc_scale_factor(-abs(pose.z), pc)
    pc_scaled, dirs = depthImage2pointCloud(depth_img, roll_rad=pose._roll_rad, 
                                            pitch_rad=pose._pitch_rad, yaw_rad=pose._yaw_rad,
                                    horizontal_fov=hfov_deg, scale_factor=scale_factor)
    if move:
        p = np.array([[pose.x], [pose.y], [pose.z]])
        move_const = p.T
        pc_scaled += move_const
    else:
        move_const = None

    return pc_scaled, move_const

def depthImage2pointCloud(D,
                          horizontal_fov,
                          roll_rad,
                          pitch_rad,
                          yaw_rad,
                          scale_factor = None,
                          abs_alt=0,
                          mask=None):
    """
    Computes a point cloud from a depth image with optional masking.

    Parameters:
    - D (H, W): Depth image (in meters).
    - horizontal_fov (float): Horizontal field of view in degrees.
    - roll_rad, pitch_rad, yaw_rad (float): Orientation angles in radians.
    - abs_alt (float): Altitude offset to add to the Z coordinate.
    - mask (H, W) optional: Binary mask (0 or 255). If a pixel's mask value is 0, its Z coordinate is forced to 0.

    Returns:
    - pc (H, W, 3): Point cloud in NWU coordinates with masking applied.
    """
    # # Prepare mask: default all valid
    if mask is None:
        mask = np.full((D.shape[0], D.shape[1]), 255, dtype=np.uint8)
    else:
        mask = mask.astype(np.uint8)
        unique_vals = np.unique(mask)
        assert set(unique_vals).issubset({0, 255}), "binary mask should only have 0 or 255"

    dirs = _img2dirVecsCam(D.shape, horizontal_fov)

    # Camera-to-body and body-to-earth rotations
    Ry = np.array([[ 0,  0, 1], [0, 1, 0], [-1, 0, 0]])
    Rx = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    Rphi   = rotation_matrix_x(-roll_rad)
    Rtheta = rotation_matrix_y(-pitch_rad)
    Rpsi   = rotation_matrix_z(-yaw_rad)
    Rnwu   = rotation_matrix_x(np.pi)
    R = Rnwu @ Rpsi @ Rtheta @ Rphi @ Rx @ Ry

    # Rotate direction vectors into NWU frame
    dirs_nwu = dirs @ R.T

    # Scale by depth and altitude
    pc1 = dirs_nwu * (D[..., np.newaxis])

    if scale_factor is None:
        scale_factor = 1

    D2 = D * scale_factor
    pc1 = dirs_nwu * (D2[..., np.newaxis])
    return pc1, dirs_nwu

def _img2dirVecsCam(output_shape, hfov_degs):
    H, W = output_shape
    hfov_rad = np.radians(hfov_degs)
    focal_length = (W / 2) / np.tan(hfov_rad / 2)
    cx, cy = W / 2, H / 2

    # Generate direction vectors in camera frame
    x_idxs = np.arange(W)
    y_idxs = np.arange(H)
    x_grid, y_grid = np.meshgrid(x_idxs, y_idxs)
    X = (x_grid - cx) / focal_length
    Y = (y_grid - cy) / focal_length
    # Z = np.sqrt(1-X**2-Y**2)
    Z = np.ones_like(X)
    norm = np.sqrt(X**2 + Y**2 + Z**2)
    X /= norm; Y /= norm; Z /= norm
    return np.stack((X, Y, Z), axis=-1)  # (H, W, 3)


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






# def img2dirVecsCam(output_shape, hfov_degs):
#     H, W = output_shape
#     aspect_ratio = H / W
#     # Horizontal FOV in radians
#     hfov_rad = np.deg2rad(hfov_degs)
#     # Vertical FOV based on aspect ratio
#     vfov_rad = 2 * np.arctan(np.tan(hfov_rad / 2) * aspect_ratio)
#     # Generate angles for each pixel
#     x_angles = np.linspace(-hfov_rad / 2, hfov_rad / 2, W)
#     y_angles = np.linspace(-vfov_rad / 2, vfov_rad / 2, H)
#     # Create meshgrid of angles
#     theta, phi = np.meshgrid(x_angles, y_angles)
#     # Compute direction vectors in camera frame
#     # Assuming forward is Z, right is X, down is Y (OpenCV style)
#     # For each pixel, the ray direction in 3D is computed from angles:
#     # x = tan(theta), y = tan(phi), z = 1, then normalize
#     x = np.sin(theta)
#     y = np.sin(phi)
#     # z = np.ones_like(x)
#     z = np.sqrt(1-x**2-y**2)
#     # Stack and normalize the vectors
#     dirs1 = np.stack((x, y, z), axis=-1)
#     norms = np.linalg.norm(dirs1, axis=-1, keepdims=True)
#     return dirs1 / norms  # shape (H, W, 3)





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


