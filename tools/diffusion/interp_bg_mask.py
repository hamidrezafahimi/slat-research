import numpy as np
import cv2
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../../lib")
from mapper3D_helper import interp_2d


    # print(zi.shape)
metric_depth = np.loadtxt('/home/hamid/w/DATA/272/asfd.csv', delimiter=',', dtype=np.float32)
mask = 255 - cv2.imread('/home/hamid/w/DATA/272/mask.png', cv2.IMREAD_GRAYSCALE)

zi = interp_2d(metric_depth, mask, plot=True)

np.savetxt('/home/hamid/zi.csv', zi, delimiter=',')
# # Create the interpolation function

# # Define new points for interpolation
# x_new = np.linspace(0, 2, 10)
# y_new = np.linspace(0, 3, 10)

# # Interpolate the z values at the new points


# Plotting the results (optional)
# X_new, Y_new = np.meshgrid(x_new, y_new)

# f = interp2d(x_coords, y_coords, Z_masked, kind='linear')
# z_interpolated = f(x_coords, y_coords)

# np.savetxt('/home/hamid/Z_masked.csv', Z_masked, delimiter=',')
# print(X_orig)
