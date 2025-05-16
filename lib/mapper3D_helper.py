import numpy as np

def transform_depth(depth_image, bg_image, pitch):
    # Compute ground elevation profile
    gep_f = _calc_ground_depth(66.0, pitch_rad=pitch, output_shape=depth_image.shape)

    # Convert images to float32
    depth_f = depth_image.astype(np.float32)
    bg_f = bg_image.astype(np.float32)

    # Transformation
    trns = gep_f / np.where(bg_f != 0, bg_f, 0.01)
    final_f = trns * depth_f
    final_f = np.minimum(final_f, gep_f)
    return final_f, gep_f

def move_depth(depth_image, bg_image, pitch):
    gep_f = _calc_ground_depth(66.0, pitch_rad=pitch, output_shape=depth_image.shape)
    # Convert to float32
    depth_f = depth_image.astype(np.float32)
    bg_f = bg_image.astype(np.float32)
    # gep_f = gep_depth.astype(np.float32)
    # Compute how much closer foreground is, in original image
    fg_f = depth_f - bg_f
    # np.clip(fg_f, 0, 255, out=fg_f)

    # Apply same ratio to refined background to get final proximity
    final_f = gep_f + fg_f
    np.clip(final_f, 0, 255, out=final_f)
    # maxval = np.max(final_f)
    # if maxval > 255.0:
    #     final_f *= (255.0 / maxval)
    # fg_i = depth_image - bg_image
    # final_i = gep_f.astype(np.uint8) + np.where(fg_i > 0, fg_i, 0)
    if np.min(final_f) < 0:
        print(np.min(final_f))
        raise ValueError("Non-logical ratio calculation")

    final_f = np.minimum(final_f, gep_f)
    # final_image = final_f.astype(np.uint8)
    return final_f, gep_f

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

def depthImage2pointCloud(D, horizontal_fov, roll_rad, pitch_rad, yaw_rad, abs_alt=0, alt_scale=1):
    """
    Computes the unit direction vectors for each pixel in the image coordinate system
    
    Parameters:
    - image_width (int): Width of the image in pixels.
    - image_height (int): Height of the image in pixels.
    - horizontal_fov (float): Horizontal field of view of the camera in degrees.
    
    Returns:
    - np.ndarray: A (H, W, 3) array where each pixel contains its unit direction vector (X, Y, Z).
    """
    # Convert angles to radians
    horizontal_fov_rad = np.radians(horizontal_fov)
    # Compute focal length using the horizontal FOV
    image_height, image_width = D.shape
    focal_length = (image_width / 2) / np.tan(horizontal_fov_rad / 2)
    
    # Compute the intrinsic center
    cx, cy = image_width / 2, image_height / 2
    
    # Create a grid of pixel coordinates
    x_indices = np.arange(image_width)
    y_indices = np.arange(image_height)
    
    # Create meshgrid for pixel coordinates
    x_grid, y_grid = np.meshgrid(x_indices, y_indices)
    
    X = (x_grid - cx) / focal_length
    Y = (y_grid - cy) / focal_length
    Z = np.ones_like(X)
    
    # Normalize to unit vectors
    norm = np.sqrt(X**2 + Y**2 + Z**2)
    X /= norm
    Y /= norm
    Z /= norm

    direction_vectors = np.stack((X, Y, Z), axis=-1)

    # from camera to body
    Ry = np.array([
        [ 0,  0, 1],
        [ 0,  1, 0],
        [ -1,  0, 0]
    ])
    Rx = np.array([
        [1,  0,  0],
        [0,  0,  -1],
        [0, 1,  0]
    ])
    # from body to earth
    Rphi = rotation_matrix_x(-roll_rad)
    Rtheta = rotation_matrix_y(-pitch_rad)
    Rpsi = rotation_matrix_z(-yaw_rad)
    Rnwu = rotation_matrix_x(np.pi)

    R = Rnwu @ Rpsi @ Rtheta @ Rphi @ Rx @ Ry
    # Apply rotation to all direction vectors
    direction_vectors_nwu = direction_vectors @ R.T # shape remains (H, W, 3)
    return (direction_vectors_nwu * D[..., np.newaxis] * alt_scale) + np.array([0,0,abs_alt])

def _calc_ground_depth(hfov_degs, 
                      pitch_rad, 
                      fixed_alt=10.0, 
                      output_shape=(480, 640), 
                      max_dist=255.0):
    """
    Computes a synthetic "ground depth" image for a camera at (0,0,-fixed_alt) in NED
    with a pitch of `pitch_rad`, such that passing this depth image into `depthImage2pointCloud`
    yields 3D points all on the z=0 plane.

    The returned depth image is float-valued in [0..255], with 0 corresponding to the
    maximum distance (max_dist) and 255 corresponding to zero distance from the camera.

    Parameters
    ----------
    hfov_degs : float
        Horizontal field of view of the camera in degrees.
    pitch_rad : float
        The camera's pitch (rotation about Y), in radians. Negative pitch typically means
        looking downward, depending on your convention.
    fixed_alt : float, optional
        The altitude of the camera above the z=0 plane, defaults to 1.
    output_shape : (int, int), optional
        (height, width) of the output depth image.
    max_dist : float, optional
        The maximum distance we allow in the synthetic depth map. Default is 256.

    Returns
    -------
    D_float : np.ndarray of shape (H, W), dtype float32
        A 2D array whose values lie in [0.0 .. 255.0]. 
        0.0 => farthest visible ground point (max_dist).
        255.0 => zero distance from camera (directly under the camera if looking down).
        If you feed this into `depthImage2pointCloud(D_float, hfov_degs, pitch_rad, max_dist)`,
        you get points on z=0.
    """
    image_height, image_width = output_shape

    # Convert horizontal FOV to radians
    hfov_rad = np.radians(hfov_degs)

    # Match the focal length logic from depthImage2pointCloud
    focal_length = (image_width / 2) / np.tan(hfov_rad / 2)

    # Intrinsic center
    cx, cy = image_width / 2.0, image_height / 2.0

    # Create meshgrid for pixel coordinates
    x_indices = np.arange(image_width)
    y_indices = np.arange(image_height)
    x_grid, y_grid = np.meshgrid(x_indices, y_indices)

    # -- 1) Compute camera-frame unit directions (same as depthImage2pointCloud) --
    X = (x_grid - cx) / focal_length
    Y = (y_grid - cy) / focal_length
    Z = np.ones_like(X)

    # Normalize
    norm = np.sqrt(X**2 + Y**2 + Z**2)
    X /= norm
    Y /= norm
    Z /= norm

    direction_vectors = np.stack([X, Y, Z], axis=-1)

    # -- 2) Apply the same camera->NED rotation used in depthImage2pointCloud --
    # R_cam_to_ned = Rx(-90°) @ Rz(-90°) in your original code
    Ry_90neg = np.array([
        [ 0,  0,  1],
        [ 0,  1,  0],
        [-1,  0,  0]
    ])
    Rx_90neg = np.array([
        [ 1,  0,  0],
        [ 0,  0, -1],
        [ 0,  1,  0]
    ])
    R_cam_to_ned = Rx_90neg @ Ry_90neg

    direction_vectors_ned = direction_vectors @ R_cam_to_ned.T

    # -- 3) Apply the -pitch rotation about Y (as in depthImage2pointCloud) --
    Ry_pitch = np.array([
        [ np.cos(-pitch_rad),  0, -np.sin(-pitch_rad)],
        [ 0,                   1,  0],
        [ np.sin(-pitch_rad),  0,  np.cos(-pitch_rad)]
    ])
    final_dirs = direction_vectors_ned @ Ry_pitch.T

    # -- 4) Intersect each ray with ground plane z=0, from camera at (0,0,-fixed_alt) --
    dir_z = final_dirs[..., 2]
    eps = 1e-12
    # t is how far along the ray we go to hit z=0
    # camera_pos_z + t*dir_z = 0 => -fixed_alt + t*dir_z = 0 => t = fixed_alt / dir_z
    # only valid if dir_z>0
    t = np.where(dir_z > eps, fixed_alt / dir_z, np.inf)
    distances = t.astype(np.float32)

    # Check basic statistics
    min_dist = np.min(distances)
    max_dist = np.max(distances)

    # Map distances (0 to max_dist) -> (255.0 -> 0.0) in float
    # Formula: mapped_value = 255.0 * (1 - distance / max_dist)
    mapped_distances = 255.0 * (1.0 - distances / max_dist)

    # Clip to make sure no values go outside 0-255
    D_float = np.clip(mapped_distances, 0.0, 255.0)
    D_int = D_float.astype(np.uint8)
    # cv2.imshow("gep depth", D_int)
    return 255 - D_float

def rescale_depth(depth_image, bg_image, pitch):
    gep_f = _calc_ground_depth(66.0, pitch_rad=pitch, output_shape=depth_image.shape)
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

