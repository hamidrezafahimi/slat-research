import matplotlib.pyplot as plt
import numpy as np
import time

import math

def scale_d(d, f):
    d_max = np.max(d)
    # Ensure all values of d are above f
    scaled_d = np.copy(d).astype(float)  # Use float for scaling calculations
    min_f_d = np.maximum(f, np.min(d))
    
    # Rescale the values while keeping d_max the same
    scaled_d = (scaled_d - np.min(d)) / (d_max - np.min(d)) * (d_max - min_f_d) + min_f_d
    scaled_d = np.clip(scaled_d, 0, 255)  # Ensure values stay in [0, 255]
    return scaled_d#.astype(np.uint8)


def calc_gep(angle, y_value, center=(0, 0)):
    """
    Calculate the intersection points of radial lines with a horizontal line (y = y_value).

    Parameters:
        angles (array): Array of angles (in radians) representing the directions.
        y_value (float): The y-coordinate of the horizontal line.
        center (tuple): The (x, z) coordinates of the center point.

    Returns:
        np.ndarray: A numpy array of intersection points [(x1, z1), (x2, z2), ...].
    """
    cx, cz = center  # Center point coordinates

    # for angle in angles:
    # Calculate direction vector components
    dx = np.cos(angle)
    dz = np.sin(angle)

    # Scale factor to intersect at y = y_value
    scale = (y_value-center[1]) / dz if dz != 0 else np.inf

    # Calculate intersection point in the x-z plane
    x_intersect = cx + dx * scale
    z_intersect = cz + dz * scale

    return np.array([x_intersect, z_intersect])



def direction_from_angle_y(angle):
    """
    Calculate a 2D unit direction vector in the x-z plane 
    based on an angle around the Y-axis.
    
    Parameters:
        angle (float): Angle in radians.
    
    Returns:
        tuple: A tuple representing the (x, z) direction vector.
    """
    x = math.cos(angle)
    z = math.sin(angle)
    return np.array([x, z])

def unit_direction_vector(p1, p2):
    """
    Calculate the unit direction vector between two points in the x-z plane.
    
    Parameters:
        p1 (tuple): The first point (x1, z1).
        p2 (tuple): The second point (x2, z2).
    
    Returns:
        tuple: A tuple representing the unit direction vector (dx, dz).
    """
    x1, z1 = p1
    x2, z2 = p2
    dx = x2 - x1
    dz = z2 - z1
    magnitude = math.sqrt(dx**2 + dz**2)
    if magnitude == 0:
        raise ValueError("The two points are identical; direction is undefined.")
    return np.array([dx / magnitude, dz / magnitude])

def get_unit_vec_of_vec(vec):
    # print(vec)
    ret = vec / np.linalg.norm(vec)
    print(vec)
    assert (not np.any(np.isnan(ret)))
    return ret

# def get_euc_dist(pt1, pt2):
#     return np.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2 + (pt1[2]-pt2[2])**2)

def get_euc_dist_2d(pt1, pt2):
    return np.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)

def get_euc_dist_3d(pt1, pt2):
    return np.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2 + (pt1[2]-pt2[2])**2)

def npimshow(data, text):
    plt.figure()
    # Normalize the array to 0-255 range
    normalized_data = 255 * (data - np.min(data)) / (np.max(data) - np.min(data))

    # Display the array using imshow
    plt.imshow(normalized_data, cmap='gray', vmin=0, vmax=255)

    plt.colorbar()  # Adds a color scale bar
    plt.title(text)

def get_euc_dists_2d(start, points):
    return [get_euc_dist_2d(p, start) for p in points]

def get_euc_dists_3d(start, points):
    dists = np.zeros((points.shape[0], points.shape[1]))
    for i in range(points.shape[0]):
        for j in range(points.shape[1]):
            dists[i,j] = get_euc_dist_3d(points[i,j], start)
    return dists

def perform_tilt_correction_2d(start, raw_depth_points, gep_dists):
    max_gep_dist = np.max(gep_dists)
    tilt_corrected_depth_points = []
    for rdp, gepd in zip(raw_depth_points, gep_dists):
        print(rdp)
        data = list(np.array(rdp) + (max_gep_dist - gepd)*(get_unit_vec_of_vec(np.array(rdp)-start)))
        # print(get_unit_vec_of_vec(np.array(rdp)-start), np.array(rdp), start)
        tilt_corrected_depth_points.append(data)
    return tilt_corrected_depth_points

from scipy.ndimage import zoom
def resized_imread(path, r):
    # Read the image using matplotlib
    data = plt.imread(path)  # Replace with your image file path
    # Resize the image to 1/4 of its original size using scipy.ndimage.zoom
    return zoom(data, (r, r))  # (0.25, 0.25) for spatial dimensions, 1 for color channels

def normalize_array(arr):
    """
    Normalizes a 2D NumPy array so that its values are between 0 and 1.
    The maximum value will be 1 and the minimum value will be 0.
    
    Parameters:
    arr (numpy.ndarray): 2D array to normalize

    Returns:
    numpy.ndarray: Normalized 2D array with values between 0 and 1
    """
    min_val = np.min(arr)
    max_val = np.max(arr)
    normalized_arr = (arr - min_val) / (max_val - min_val)
    return normalized_arr

def normalize_under_roof(_start, roof_z, ground_end_points, points):
    nearest_distance = 1e9
    farthest_distance = -1
    dists = np.zeros((points.shape[0], points.shape[1]))
    for i in range(points.shape[0]):
        for j in range(points.shape[1]):
            dist = get_euc_dist_3d(points[i,j,:], _start)
            dists[i,j] = dist
            if dist < nearest_distance:
                nearest_distance = dist
            if dist > farthest_distance:
                farthest_distance = dist

    distance_range = farthest_distance - nearest_distance
    irn_points = np.zeros((dists.shape[0], dists.shape[1], 3)) # in roof normalized points
    irn_dists = np.zeros(dists.shape)
    for i in range(dists.shape[0]):
        for j in range(dists.shape[0]):
            # K_saturation_to_nearest = rep_dists[i,j] / nearest_distance
            # p2 = start + K_saturation_to_nearest * (tilt_corrected_depth_points[i,j,:] - start)
            # K_saturation_to_farthest = gep_dists[i,j] / farthest_distance
            # irn_point = start + K_saturation_to_farthest * (p2 - start)
            # irn_points[i,j,:] = irn_point
            z = roof_z - (roof_z * (dists[i,j]-nearest_distance) / distance_range)
            ratio = (z - _start[2]) / (ground_end_points[i,j][2] - _start[2])  # Interpolation ratio for x
            x = _start[0] + ratio * (ground_end_points[i,j][0] - _start[0])
            y = _start[1] + ratio * (ground_end_points[i,j][1] - _start[1])
            irn_points[i,j,:] = [x,y,z]
            irn_dists[i,j] = get_euc_dist_3d((x,y,z), _start)
    
    return irn_points, irn_dists

def normalize_under_roof_2d(_start, roof_z, ground_end_points, points):
    nearest_distance = 1e9
    farthest_distance = -1
    dists = []
    for p in points:
        dist = get_euc_dist_2d(p, _start)
        dists.append(dist)
        if dist < nearest_distance:
            nearest_distance = dist
        if dist > farthest_distance:
            farthest_distance = dist

    distance_range = farthest_distance - nearest_distance
    irn_points = []
    irn_dists = []
    for d, g in zip(dists, ground_end_points):
        z = roof_z - (roof_z * (d-nearest_distance) / distance_range)
        ratio = (z - _start[1]) / (g[1] - _start[1])  # Interpolation ratio for x
        x = _start[0] + ratio * (g[0] - _start[0])
        irn_points.append((x,z))
        irn_dists.append(get_euc_dist_2d((x,z), _start))
    
    return irn_points, irn_dists


def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

        # # Normalize the z-coordinate for the irn point between z=roof_z and z=0
        # z = roof_z - (roof_z * (tcdp_dists[i,j] - nearest_distance) / distance_range)
        # # Interpolate the x-coordinate to stay on the line
        # x = start[0] + ratio * (ground_end_points[i,j][0] - start[0])
        # y = start[1] + ratio * (ground_end_points[i,j][1] - start[1])
        # irn_points[i,j,0] = x
        # irn_points[i,j,1] = y
        # irn_points[i,j,2] = z

def calc_derivative(xs, ys, ret_array=False):
    assert(len(xs) == len(ys))
    dys = []
    for k in range(1, len(ys)):
        dys.append((ys[k] - ys[k-1]) / (xs[k] - xs[k-1]))
    if ret_array:
        return np.array(dys)
    else:
        return dys

# def calc_integration(xs, ys, initial_value=0):
#     assert len(xs) == len(ys)
#     sys = [initial_value]
#     sum = initial_value
#     for k in range(1, len(ys)):
#         sum += ((ys[k] + ys[k-1]) / 2) * (xs[k] - xs[k-1])
#         sys.append(sum)
#     return np.array(sys)

def calc_integration(xs, ys, initial_value=0):
    assert len(xs) == len(ys) + 1  # xs should be one element longer than ys
    sys = [initial_value]
    sum = initial_value
    for k in range(len(ys)):
        sum += ((ys[k] + ys[k-1]) / 2) * (xs[k+1] - xs[k])
        sys.append(sum)
    return np.array(sys)

def get_relative_midpoint(start, end, ratio):
    return (start[0] + ratio * (end[0] - start[0]), start[1] + ratio * (end[1] - start[1]))

def get_absolute_midpoint(start, end, dist):
    return np.array(start) + dist * get_unit_vec_of_vec(np.array(end) - np.array(start))

def remove_derivative_and_integrate_1d(xs, set1, set2):
    d1 = calc_derivative(xs, set1, ret_array=True)
    d2 = calc_derivative(xs, set2, ret_array=True)
    corrected_d1 = d1 - d2
    out = calc_integration(xs, corrected_d1, initial_value=set2[0])
    return out

def get_all_from_dists(start, ground_end_points, dists):
    return [list(get_absolute_midpoint(start, point, dist)) for dist, point in zip(dists, ground_end_points)]

def get_all_from_dists_2d(start, geps, dists):
    out = np.zeros(geps.shape)
    for i in range(geps.shape[0]):
        for j in range(geps.shape[1]):
            out[i,j,:] = get_absolute_midpoint(start, geps[i,j,:], dists[i,j])
    return out

def get_relative_midpoint_3d(start, end, ratio):
    # print(start)
    # print(end)
    # print(ratio)
    # print(np.array((start[0] + ratio * (end[0] - start[0]),  start[1] + ratio * (end[1] - start[1]), 
    #         start[2] + ratio * (end[2] - start[2]))))
    # time.sleep(1000)
    return np.array([start[0] + ratio * (end[0] - start[0]),  start[1] + ratio * (end[1] - start[1]), 
            start[2] + ratio * (end[2] - start[2])])

def perform_tilt_correction_3d(start, raw_depth_points, max_gep_dist, gep_dists):
    tilt_corrected_depth_points = np.zeros(raw_depth_points.shape)
    for i in range(raw_depth_points.shape[0]):
        for j in range(raw_depth_points.shape[1]):
            tilt_corrected_depth_points[i,j,:] = raw_depth_points[i,j,:] + \
                (max_gep_dist - gep_dists[i,j]) * (get_unit_vec_of_vec(raw_depth_points[i,j,:]-start))
    return tilt_corrected_depth_points

def remove_derivative_and_integrate_2d(xs, set1, set2):
    assert(set1.shape == set2.shape)
    corrected_data = np.zeros(set1.shape)
    for j in range(set1.shape[1]):
        corrected_data[:,j] = remove_derivative_and_integrate_1d(xs, set1[:,j], set2[:,j])
    return corrected_data
