import os
import sys
sys.path.append("/media/hamid/Workspace/thing/lib")
from lines import calcFlatGroundDepth, expand_segments
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting
from matplotlib.patches import Patch

def plot_bgs(bg1, bg2):
    # Validate input types and shapes
    if not (isinstance(bg1, np.ndarray) and isinstance(bg2, np.ndarray)):
        raise TypeError("Both inputs must be numpy arrays.")
    if bg1.dtype != np.uint8 or bg2.dtype != np.uint8:
        raise TypeError("Both images must be of dtype uint8.")
    if bg1.ndim != 2 or bg2.ndim != 2:
        raise ValueError("Both images must be 2D (grayscale).")
    if bg1.shape != bg2.shape:
        raise ValueError("Both images must have the same shape.")

    # Prepare meshgrid
    x = np.arange(0, bg1.shape[1])
    y = np.arange(0, bg1.shape[0])
    X, Y = np.meshgrid(x, y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Normalize for better 3D coloring (optional)
    bg1_norm = bg1.astype(np.float32)
    bg2_norm = bg2.astype(np.float32)

    # Plot both surfaces with distinct colors
    surf1 = ax.plot_surface(X, Y, bg1_norm, color='red', alpha=0.7, label='Background 1')
    surf2 = ax.plot_surface(X, Y, bg2_norm, color='blue', alpha=0.7, label='Background 2')

    # Add custom legend using proxy artists
    legend_patches = [
        Patch(facecolor='red', label='Background 1'),
        Patch(facecolor='blue', label='Background 2')
    ]
    ax.legend(handles=legend_patches)

    ax.set_title("Background Comparison (3D)")
    plt.show()

def calc_ground_depth(hfov_degs, 
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
    np.savetxt("dists.csv", distances, fmt="%d", delimiter=",")

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
    return D_float


def depthImage2pointCloud(D, horizontal_fov, pitch_rad, max_dist=255):
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

    # Combined rotation: first -90° around Z, then -90° around X
    R = Rx @ Ry

    # Apply rotation to all direction vectors
    direction_vectors_ned = direction_vectors @ R.T # shape remains (H, W, 3)

    # pitch_rad is the rotation of camera. so the point cloud must rotate in reverse angle
    Ry = np.array([
        [ np.cos(-pitch_rad),  0, -np.sin(-pitch_rad)],
        [ 0,  1, 0],
        [ np.sin(-pitch_rad),  0, np.cos(-pitch_rad)]
    ])
        
    unit_vectors = direction_vectors_ned @ Ry.T

    # Compute point cloud
    return unit_vectors * D[..., np.newaxis]

def get_euc_dist_3d(pt1, pt2):
    return np.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2 + (pt1[2]-pt2[2])**2)

def push_points_behind_min_distance(_start, min_possible_distance_to_camera, ground_end_points, points):
    # Compute the original distances from the camera to each point.
    nearest_distance = 1e9
    farthest_distance = -1
    dists = np.zeros((points.shape[0], points.shape[1]))
    for i in range(points.shape[0]):
        for j in range(points.shape[1]):
            d = get_euc_dist_3d(points[i, j, :], _start)
            dists[i, j] = d
            if d < nearest_distance:
                nearest_distance = d
            if d > farthest_distance:
                farthest_distance = d

    # Prepare arrays for the new points and their distances.
    new_points = np.zeros((points.shape[0], points.shape[1], 3))
    new_dists = np.zeros((points.shape[0], points.shape[1]))

    # Remap each original distance so that:
    # - A point with distance equal to nearest_distance gets mapped to min_possible_distance_to_camera.
    # - A point with distance equal to farthest_distance remains unchanged.
    # - Points in between are moved proportionally.
    for i in range(points.shape[0]):
        for j in range(points.shape[1]):
            old_d = dists[i, j]
            if farthest_distance != nearest_distance:
                new_distance = min_possible_distance_to_camera + ((old_d - nearest_distance) / (farthest_distance - nearest_distance)) * (farthest_distance - min_possible_distance_to_camera)
            else:
                new_distance = old_d  # Edge case: all points are at the same distance.

            # Compute the direction from the camera to the corresponding ground end point.
            direction = ground_end_points[i, j] - _start
            norm_dir = np.linalg.norm(direction)
            
            # Check saturation: if new_distance exceeds the distance to the ground endpoint, set the new point to be the ground endpoint.
            if norm_dir > 0:
                if new_distance > norm_dir:
                    new_distance = norm_dir
                    new_point = ground_end_points[i, j]
                else:
                    factor = new_distance / norm_dir
                    new_point = _start + factor * direction
            else:
                new_point = _start

            new_points[i, j] = new_point
            new_dists[i, j] = new_distance

    return new_points, new_dists

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

    # final_image = final_f.astype(np.uint8)
    return final_f, gep_f

def transform_depth(depth_image, bg_image, pitch):
    gep_f = calc_ground_depth(66.0, pitch_rad=pitch, output_shape=depth_image.shape)
    # Convert to float32
    depth_f = depth_image.astype(np.float32)
    bg_f = bg_image.astype(np.float32)

    trns = gep_f / np.where(bg_f > 0.01, bg_f, 0.01)
    final_f = trns * depth_f

    return final_f, gep_f

def move_depth(depth_image, bg_image, pitch):
    gep_f = calc_ground_depth(66.0, pitch_rad=pitch, output_shape=depth_image.shape)
    # Convert to float32
    depth_f = depth_image.astype(np.float32)
    bg_f = bg_image.astype(np.float32)
    # gep_f = gep_depth.astype(np.float32)
    # Compute how much closer foreground is, in original image
    fg_f = depth_f - bg_f
    # np.clip(fg_f, 0, 255, out=fg_f)

    # Apply same ratio to refined background to get final proximity
    final_f = gep_f + fg_f
    # fg_i = depth_image - bg_image
    # final_i = gep_f.astype(np.uint8) + np.where(fg_i > 0, fg_i, 0)
    if np.min(final_f) < 0 or np.max(final_f) > 255.0:
        print(np.min(final_f))
        print(np.max(final_f))
        raise ValueError("Non-logical ratio calculation")

    # final_image = final_f.astype(np.uint8)
    return final_f, gep_f
    # return final_i, gep_f
    # plot_bgs(gep_depth, final_image)
    # cv2.imshow("refined", final_image)
    # cv2.waitKey()
    # cv2.imwrite("gepdep.jpg", gep_depth)

def plot_point_cloud(point_cloud, step=20):
    # Sample points for visualization
    X_sample = point_cloud[::step, ::step, 0].flatten()
    Y_sample = point_cloud[::step, ::step, 1].flatten()
    Z_sample = point_cloud[::step, ::step, 2].flatten()

    # Stack into (N, 3) array of points
    sampled_points = np.vstack((X_sample, Y_sample, Z_sample)).T

    # Compute distances to the origin (0, 0, 0)
    distances = np.linalg.norm(sampled_points, axis=1)

    # Plot the point cloud
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot origin
    ax.scatter(0, 0, 0, color='red', label='Origin')

    # Scatter plot colored by distance
    sc = ax.scatter(X_sample, Y_sample, Z_sample, c=distances, cmap='coolwarm', s=1)
    plt.colorbar(sc, label='Distance to Origin')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=-180, azim=0)
    ax.legend()
    plt.show()

def plot_point_cloud_2(point_cloud, step=20):
    # If point_cloud is a tuple (points, dists), extract the points.
    if isinstance(point_cloud, tuple):
        point_cloud = point_cloud[0]
    
    # Sample points for visualization
    X_sample = point_cloud[::step, ::step, 0]
    Y_sample = point_cloud[::step, ::step, 1]
    Z_sample = point_cloud[::step, ::step, 2]

    # Plot the point cloud
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(0, 0, 0, color='red')
    ax.plot(X_sample.flatten(), Y_sample.flatten(), Z_sample.flatten(), 'b.', markersize=1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=-180, azim=0)
    plt.show()


def main():
    depth_path = sys.argv[1]
    # Extract filename only (e.g., 00000001.jpg)
    depth_image = np.loadtxt(depth_path, delimiter=',', dtype=np.float32)
    max_val = np.max(depth_image)
    if max_val > 0:
        depth_image_scaled = depth_image / max_val * 255.0
    else:
        depth_image_scaled = depth_image  # or np.zeros_like(depth_image)

    pitch = -0.785398
    depth_pc = depthImage2pointCloud(depth_image_scaled, pitch_rad=pitch, horizontal_fov=66)
    plot_point_cloud(depth_pc, step=10)


if __name__ == "__main__":
    main()
