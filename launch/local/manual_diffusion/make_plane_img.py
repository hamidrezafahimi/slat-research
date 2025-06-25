import numpy as np
import cv2
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../../lib")
from mapper3D_helper import interp_2d


w = 1241
h = 376

corners = np.array([[0.0,    0.0,   35],
                     [1240.0, 0.0,   35],
                     [0.0,    375.0, 4],
                     [1240.0, 375.0, 4]])

img = np.zeros((h,w))
img[0,0] = corners[0,2]
img[0,w-1] = corners[1,2]
img[h-1,0] = corners[2,2]
img[h-1,w-1] = corners[3,2]

mask = np.ones_like(img) * 255.0
mask[0,0] = 0
mask[0,w-1] = 0
mask[h-1,0] = 0
mask[h-1,w-1] = 0

mask = mask.astype(np.uint8)

zi = interp_2d(img, mask, plot=True)

zi = np.where(zi > 255, 255, zi)
zi = np.where(zi < 0,   0,   zi)

np.savetxt('/home/hamid/w/DATA/273/zi.csv', zi, delimiter=',')
# cv2.imwrite('/home/hamid/img.png', zi.astype(np.uint8))


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
