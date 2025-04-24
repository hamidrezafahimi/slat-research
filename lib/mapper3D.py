import numpy as np
from mapper3D_helper import *
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting

class Mapper3D:
    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

    def process(self, depth_image, bg_image, color_img, pitch, altitude, vis=False, plot=False):
        max_val = np.max(depth_image)
        depth_image_scaled = depth_image / max_val * 255.0 if max_val > 0 else depth_image

        bg_scaled = bg_image.astype(np.float32) * ((np.max(depth_image_scaled) - np.min(depth_image_scaled)) / 255.0) + np.min(depth_image_scaled)

        fimg, gep_depth = transform_depth(depth_image_scaled, bg_scaled, pitch)
        gep_pc = depthImage2pointCloud(gep_depth, pitch_rad=pitch, horizontal_fov=66)
        fimg_pc = depthImage2pointCloud(fimg, pitch_rad=pitch, horizontal_fov=66,
                                        alt_scale=altitude / np.mean(gep_pc[:, :, 2]))

        if plot:
            self.ax.clear()
            self.plot_point_cloud(fimg_pc, color_img, step=10)

        if vis:
            cv2.imshow("Color Image", color_img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                return False  # Stop loop

        return True
    
    def plot_point_cloud(self, point_cloud, visualization_image, step=20):
        self.ax.view_init(elev=-170, azim=-85)
        # Ensure visualization_image is of the same size as the point cloud in the X and Y dimensions
        height, width, _ = visualization_image.shape
        assert point_cloud.shape[0] == height and point_cloud.shape[1] == width, \
            "Point cloud and visualization image must have the same dimensions (height, width)"

        # Sample points for visualization (downsampling by step)
        X_sample = point_cloud[::step, ::step, 0].flatten()
        Y_sample = point_cloud[::step, ::step, 1].flatten()
        Z_sample = point_cloud[::step, ::step, 2].flatten()

        # Get corresponding colors from the visualization image
        colors = visualization_image[::step, ::step].reshape(-1, 3) / 255.0  # Normalize to [0, 1] range

        self.ax.scatter(0, 0, 0, color='red', label='Origin')
        # Plot the points with colors from the visualization image
        self.ax.scatter(X_sample, Y_sample, Z_sample, c=colors, s=1)

        # Set labels and colorbar
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
