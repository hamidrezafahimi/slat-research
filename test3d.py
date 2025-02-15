import matplotlib.pyplot as plt
import numpy as np
from utils import *

qdata = resized_imread('00000004.png', 0.0625) 
qdata = normalize_array(qdata)

# mesh2d = np.zeros(qdata.shape)

# mesh2d[5:10, 6:11] = 1

# for i in range(qdata.shape[0]):
#     for j in range(qdata.shape[1]):
#         mesh[i,j,]

print(qdata.shape)
npimshow(qdata, "an image")

start = np.array([0, 0, 10])
ground_z = 0
roof_z = 3
max_raw_depth = 27

# Number of radial lines
n_lines_y_img = qdata.shape[0]
n_lines_x_img = qdata.shape[1]
# Calculate angles_y for each line between 3.75 and 5.5 o'clock in radians
angles_y = np.linspace(-0.75 * np.pi / 6, -2.25 * np.pi / 6, n_lines_y_img)
angles_z = np.linspace(-1 * np.pi / 6, 1 * np.pi / 6, n_lines_x_img)

max_gep_dist = -1
ground_end_points = np.zeros((n_lines_y_img, n_lines_x_img, 3))
mesh = np.zeros((n_lines_y_img, n_lines_x_img, 3))
gep_dists = np.zeros((n_lines_y_img, n_lines_x_img))
for i, angle_y in enumerate(angles_y):
    x_diff = (ground_z - start[2]) * np.tan(angle_y)
    # x_radial_tick = start[0] + x_diff
    # line_length = np.sqrt((x_diff)**2 + (ground_z - start[2])**2)
    for j, angle_z in enumerate(angles_z):
        x_diff_project_x = x_diff*np.cos(angle_z)
        x_diff_project_y = x_diff*np.sin(angle_z)
        gepx = start[0]+x_diff_project_x
        gepy = start[1]+x_diff_project_y
        gepz = ground_z
        ground_end_points[i,j,0] = gepx
        ground_end_points[i,j,1] = gepy
        ground_end_points[i,j,2] = gepz
        gep_dists[i,j] = get_euc_dist_3d((gepx, gepy, gepz), start)
        if max_gep_dist < gep_dists[i,j]:
            max_gep_dist = gep_dists[i,j]
        mesh[i,j,0] = gepx
        mesh[i,j,1] = gepy
        if i < 17 and i > 12 and j > 10 and j < 17:
            mesh[i,j,2] = 2
        else:
            mesh[i,j,2] = 0

# ^ z axis
# |
# Center
#   \
#    \
#     \     ^
#      \     \
#       \ line_length
#        \     \
#         \     v
#          \
#  _________\
#  <-xdiff-> ^
#            |
#      x_radial_tick
# --------------------------> x axis
#

# Generate random values between 0 and 1, each associated with a pixel
# random_values = np.random.rand(n_lines_y_img*n_lines_x_img)
k = 0
normalized_gep_dists = np.zeros((n_lines_y_img, n_lines_x_img, 3))
raw_depth_points = np.zeros((n_lines_y_img, n_lines_x_img, 3))
tilt_corrected_depth_points = np.zeros((n_lines_y_img, n_lines_x_img, 3))
rep_dists = np.zeros((n_lines_y_img, n_lines_x_img))
tcdp_dists = np.zeros((n_lines_y_img, n_lines_x_img))
# ground_end_point_dists = np.zeros((n_lines_y_img, n_lines_x_img))
# ground_end_point_dists = []
# raw_end_points = np.zeros((n_lines_y_img, n_lines_x_img))
raw_end_points = np.zeros((n_lines_y_img, n_lines_x_img, 3))
for i, angle_y in enumerate(angles_y):
    x_diff_roof = (roof_z - start[2]) * np.tan(angle_y)
    # x_radial_tick = start[0] + x_diff
    # line_length = np.sqrt((x_diff)**2 + (ground_z - start[2])**2)
    for j, angle_z in enumerate(angles_z):
        x_diff_project_x = x_diff*np.cos(angle_z)
        x_diff_project_y = x_diff*np.sin(angle_z)
        x_diff_roof_project_x = x_diff_roof*np.cos(angle_z)
        x_diff_roof_project_y = x_diff_roof*np.sin(angle_z)
        repx = start[0]+x_diff_roof_project_x
        repy = start[1]+x_diff_roof_project_y
        repz = roof_z
        # normalized_gep_dists[i,j] = get_euc_dist((gepx, gepy, gepz), start)
        rep_dists[i,j] = get_euc_dist_3d((repx, repy, repz), start)

        # ground_end_point_dists[i,j] = np.sqrt(x_diff_project_x**2 + x_diff_project_y**2 + (start[2]-ground_z)**2)
        gepx = ground_end_points[i,j,0]
        gepy = ground_end_points[i,j,1]
        gepz = ground_end_points[i,j,2]
        dist = np.sqrt(x_diff_project_x**2 + x_diff_project_y**2 + (start[2]-ground_z)**2)
        raw_end_pointx = start[0]+(max_raw_depth/dist)*(gepx-start[0])
        raw_end_pointy = start[1]+(max_raw_depth/dist)*(gepy-start[1])
        raw_end_pointz = start[2]+(max_raw_depth/dist)*(gepz-start[2])
        raw_end_points[i,j,0] = raw_end_pointx
        raw_end_points[i,j,1] = raw_end_pointy
        raw_end_points[i,j,2] = raw_end_pointz
        r = qdata[i,j]
        k += 1
        # Calculate raw depth points based on random values (interpolation between start and end points)
        rdpx = start[0] + r * (raw_end_pointx - start[0])
        rdpy = start[1] + r * (raw_end_pointy - start[1])
        rdpz = start[2] + r * (raw_end_pointz - start[2])
        raw_depth_points[i,j,0] = rdpx
        raw_depth_points[i,j,1] = rdpy
        raw_depth_points[i,j,2] = rdpz
        rdp_dist = get_euc_dist_3d((rdpx, rdpy, rdpz), start)
        
# min_ngep_dist = np.min(normalized_gep_dists)
# max_ngep_dist = np.max(normalized_gep_dists)
# normalized_gep_dists -= min_ngep_dist
# Adjust irn points so the nearest point is at z=roof_z and the farthest is at z=ground_z


for i in range(raw_depth_points.shape[0]):
    for j in range(raw_depth_points.shape[1]):
        # ngep_dist = gep_dists[i,j] - (gep_dists[i,j]-max_gep_dist)
        tilt_corrected_depth_points[i,j,:] = raw_depth_points[i,j,:] + (max_gep_dist - gep_dists[i,j])*(get_unit_vec(raw_depth_points[i,j,:]-start))

irn_points_1, irn_dists_1 = normalize_under_roof(start, roof_z, ground_end_points, tilt_corrected_depth_points)
irn_points_2, irn_dists_2 = normalize_under_roof(start, roof_z, ground_end_points, raw_depth_points)


# Plot setup
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 

fig = plt.figure()
ax = fig.gca(projection='3d')

set_axes_equal(ax)
ax.set_xlim([-1, 25])
ax.set_ylim([-15, 15])
ax.set_zlim([-10, 10])

all_rdp_xs = []
all_rdp_ys = []
all_rdp_zs = []
all_irnp_xs = []
all_irnp_ys = []
all_irnp_zs = []
# for i in range(ground_end_points.shape[0]):
#     for j in range(ground_end_points.shape[1]):
#         ax.plot([start[0], ground_end_points[i,j,0]], [start[1], ground_end_points[i,j,1]], [start[2], ground_end_points[i,j,2]], color='blue')
#         # ax.plot([start[0], raw_end_points[i,j,0]], [start[1], raw_end_points[i,j,1]], [start[2], raw_end_points[i,j,2]], color='yellow')

ax.scatter(raw_depth_points[:,:,0], raw_depth_points[:,:,1], raw_depth_points[:,:,2], color = 'red')
ax.scatter(irn_points_1[:,:,0], irn_points_1[:,:,1], irn_points_1[:,:,2], color = 'black')
ax.scatter(irn_points_2[:,:,0], irn_points_2[:,:,1], irn_points_2[:,:,2], color = 'purple')
ax.scatter(tilt_corrected_depth_points[:,:,0], tilt_corrected_depth_points[:,:,1], tilt_corrected_depth_points[:,:,2], color='green')
# ax.scatter(mesh[:,:,0], mesh[:,:,1], mesh[:,:,2], color='green')
# ax.scatter(mesh[:,:])

# npimshow(tcdp_dists, "tcdp")
# npimshow(irn_dists, "irnp")

plt.show()