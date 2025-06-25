import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp2d, griddata
import matplotlib.pyplot as plt

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


def move_depth(depth_image, bg_image, gep_image):
    assert depth_image.dtype == np.float32 and bg_image.dtype == np.float32 and \
        gep_image.dtype == np.float32
    relative_depth_before = depth_image / bg_image

    # Convert depth data into proximity data
    proximity_image = 255.0 - depth_image
    bg_prox = 255.0 - bg_image
    gep_prox = 255.0 - gep_image

    # Extract foreground
    fg_prox = proximity_image - bg_prox
    assert np.all(fg_prox >= 0)

    # Rescale foreground objects
    # 1. No rescaling
    # r = 1
    # 2. Rescaling based on relative depth
    # r = gep_image[min_depth_index] * (1 - relative_depth_before[min_depth_index]) / fg_prox[min_depth_index]
    # 3. Rescaling based on relative proximity
    # r = (gep_prox[min_depth_index] * (1 - relative_prox_before[min_depth_index])) / \
    #       (fg_prox[min_depth_index] * relative_prox_before[min_depth_index])
    # fg_prox_scaled = fg_prox * r
    r = np.where(fg_prox != 0, gep_image * (1 - relative_depth_before) / fg_prox, 0.0)
    fg_prox_scaled = fg_prox * r

    # Replace old background with new background
    # 1. main functionality
    moved_prox = fg_prox_scaled + gep_prox
    # 2. No background
    # moved_prox = fg_prox_scaled
    # 3. No rescaling
    # moved_prox = fg_prox_scaled + bg_prox

    # Check
    # relative_prox_before = bg_prox / proximity_image
    # relative_prox_after = gep_prox / moved_prox
    # relative_depth_before = depth_image / gep_image
    # relative_depth_after = moved_depth / gep_image
    # print(relative_depth_before[min_depth_index], relative_depth_after[min_depth_index])
    # print(relative_prox_before[min_depth_index], relative_prox_after[min_depth_index])

    assert np.max(moved_prox) < 255.0 and np.min(moved_prox) >= 0.0,\
        f"min: {np.min(moved_prox):.2f}, max: {np.max(moved_prox):.2f}"
    moved_depth = 255.0 - moved_prox
    return moved_depth


def rotation_matrix_x(phi):
    """Rotation about x-axis."""
    c = np.cos(phi)
    s = np.sin(phi)
    return np.array([
        [1, 0, 0],
        [0, c, s],
        [0, -s, c]
    ])

def rotation_matrix_y(theta):
    """Rotation about y-axis."""
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        [c, 0, -s],
        [0, 1, 0],
        [s, 0, c]
    ])

def rotation_matrix_z(psi):
    """Rotation about z-axis."""
    c = np.cos(psi)
    s = np.sin(psi)
    return np.array([
        [c, s, 0],
        [-s, c, 0],
        [0, 0, 1]
    ])

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

    # Convert horizontal FOV to radians and compute focal length
    hfov_rad = np.radians(horizontal_fov)
    H, W = D.shape
    focal_length = (W / 2) / np.tan(hfov_rad / 2)
    cx, cy = W / 2, H / 2

    # Generate direction vectors in camera frame
    x_idxs = np.arange(W)
    y_idxs = np.arange(H)
    x_grid, y_grid = np.meshgrid(x_idxs, y_idxs)
    X = (x_grid - cx) / focal_length
    Y = (y_grid - cy) / focal_length
    Z = np.ones_like(X)
    norm = np.sqrt(X**2 + Y**2 + Z**2)
    X /= norm; Y /= norm; Z /= norm
    dirs = np.stack((X, Y, Z), axis=-1)  # (H, W, 3)

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
        scale_factor = np.ones_like(D)

    D2 = D * scale_factor
    pc1 = dirs_nwu * (D2[..., np.newaxis])
    return pc1


def calc_ground_depth(hfov_degs,
                      pitch_rad,
                      output_shape,
                      fixed_alt=10.0,
                      horizon_pitch_rad=-0.034):
    """
    Returns a float32 image whose pixel intensities encode
        C · metric_distance_to_ground    0 ≤ value ≤ 255

    C is chosen so that the farthest *ground* distance visible in
    the image is mapped to exactly 255.  Rays that never intersect
    the ground, or whose pitch > horizon_pitch_rad, are forced to 255.
    """
    H, W = output_shape

    # --- 1) intrinsics ------------------------------------------------
    f  = (W / 2) / np.tan(np.radians(hfov_degs) / 2)
    cx, cy = W / 2.0, H / 2.0

    # --- 2) unit rays in camera coords --------------------------------
    xg, yg = np.meshgrid(np.arange(W), np.arange(H))
    X = (xg - cx) / f
    Y = (yg - cy) / f
    Z = np.ones_like(X)
    norm = np.sqrt(X**2 + Y**2 + Z**2)
    dirs_cam = np.stack([X / norm, Y / norm, Z / norm], axis=-1)

    # --- 3) camera  ->  NED -------------------------------------------
    Ry90neg = np.array([[ 0, 0, 1],
                        [ 0, 1, 0],
                        [-1, 0, 0]])
    Rx90neg = np.array([[1, 0,  0],
                        [0, 0, -1],
                        [0, 1,  0]])
    dirs_ned = dirs_cam @ (Rx90neg @ Ry90neg).T

    # --- 4) apply camera pitch about +Y in NED ------------------------
    Ry = np.array([[ np.cos(-pitch_rad), 0, -np.sin(-pitch_rad)],
                   [ 0,                  1,  0               ],
                   [ np.sin(-pitch_rad), 0,  np.cos(-pitch_rad)]])
    dirs = dirs_ned @ Ry.T                     # final unit directions

    # --- 5) ray pitch angle & ground intersection --------------------
    horiz_len     = np.linalg.norm(dirs[..., :2], axis=-1)
    pitch_angle   = np.arctan2(-dirs[..., 2], horiz_len)   # +ve up, –ve down
    dir_z         = dirs[..., 2]                           # downward component

    eps           = 1e-12
    distance_raw  = np.where(dir_z > eps,                 # t = alt / dir_z
                             fixed_alt / dir_z,
                             np.inf).astype(np.float32)

    # --- 6) classify pixels ------------------------------------------
    sky_mask      = (pitch_angle > horizon_pitch_rad) | ~np.isfinite(distance_raw)
    ground_mask   = ~sky_mask                            # valid finite hits

    if not np.any(ground_mask):
        raise RuntimeError("Camera sees no ground below the horizon!")

    d_max         = distance_raw[ground_mask].max()      # farthest ground distance
    scale         = 255.0 / d_max                        # C  so that d_max→255

    # --- 7) build output image ---------------------------------------
    depth_img     = np.full((H, W), 255.0, dtype=np.float32)    # start with sky
    depth_img[ground_mask] = distance_raw[ground_mask] * scale
    depth_img     = np.minimum(depth_img, 255.0)                # clamp just in case
    return depth_img


def rescale_depth(depth_image, bg_image, pitch):
    gep_f = calc_ground_depth(66.0, pitch_rad=pitch, output_shape=depth_image.shape)
    # Convert to float32
    depth_f = depth_image.astype(np.float32)
    bg_f = bg_image.astype(np.float32)
    # gep_f = gep_depth.astype(np.float32)
    # Compute how much closer foreground is, in original image
    fg_raw_f = depth_f - bg_f
    np.clip(fg_raw_f, 0, 255, out=fg_raw_f)
    dist_to_camera_raw = np.abs(255.0 - bg_f)
    # Avoid divide-by-zero
    dist_to_camera_raw[dist_to_camera_raw == 0] = 1.0
    # Ratio of foreground proximity
    ratio = fg_raw_f / dist_to_camera_raw
    if np.min(ratio) < 0 or np.max(ratio) > 1:
        raise ValueError("Non-logical ratio calculation")

    dist_to_camera_new = np.abs(255.0 - gep_f)
    dist_to_camera_new[dist_to_camera_new == 0] = 1.0
    # Apply same ratio to refined background to get final proximity
    fg_new_f = ratio * dist_to_camera_new
    final_f = gep_f + fg_new_f
    if np.min(final_f) < 0 or np.max(final_f) > 255.0:
        raise ValueError("Non-logical ratio calculation")
    return final_f, gep_f


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
