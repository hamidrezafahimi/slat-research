import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mapper3D_helper import *


class Mapper3D:
    def __init__(self, vis=False, plot=False, color_mode='proximity', backend='matplotlib', block_plot=True):
        """
        color_mode: 'image' or 'proximity'
        backend: 'open3d' or 'matplotlib'
        block_plot: if True, use plt.show() or open3d blocking; else, use non-blocking update
        """
        assert color_mode in ['image', 'proximity'], "color_mode must be 'image' or 'proximity'"
        assert backend in ['open3d', 'matplotlib'], "backend must be 'open3d' or 'matplotlib'"
        self.color_mode = color_mode
        self.backend = backend
        self.block_plot = block_plot
        self.pcd = o3d.geometry.PointCloud()
        self.vis = vis
        self.plot = plot
        if backend == 'matplotlib':
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.ax.view_init(elev=-170, azim=-85)

    def process(self, depth_image, bg_image, color_img, roll, pitch, yaw, altitude):
        max_val = np.max(depth_image)
        depth_image_scaled = depth_image / max_val * 255.0 if max_val > 0 else depth_image

        bg_scaled = bg_image.astype(np.float32) * ((np.max(depth_image_scaled) - np.min(depth_image_scaled)) / 255.0) + np.min(depth_image_scaled)

        fimg, gep_depth = transform_depth(depth_image_scaled, bg_scaled, pitch)
        # fimg, gep_depth = rescale_depth(depth_image_scaled, bg_scaled, pitch)
        # fimg, gep_depth = move_depth(depth_image_scaled, bg_scaled, pitch)
        gep_pc = depthImage2pointCloud(gep_depth, roll_rad=roll, pitch_rad=pitch, yaw_rad=yaw,
                                       horizontal_fov=66)
        # fimg_pc = None
        fimg_pc = depthImage2pointCloud(fimg, roll_rad=roll, pitch_rad=pitch, yaw_rad=yaw, 
                                        horizontal_fov=66,
                                        abs_alt=abs(altitude),
                                        alt_scale=abs(altitude / np.mean(gep_pc[:, :, 2])))
        # print(altitude / np.mean(gep_pc[:, :, 2]))
        # rd_pc = depthImage2pointCloud(depth_image_scaled, pitch_rad=pitch, horizontal_fov=66,
        #                               alt_scale=altitude / np.mean(gep_pc[:, :, 2]))

        if self.plot:
            self.plot_point_cloud(fimg_pc, color_img)
            # self.plot_point_cloud(gep_pc, color_img)

        if self.vis:
            cv2.imshow("Color Image", color_img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                return None, False
        return fimg_pc, True

    def plot_point_cloud(self, point_cloud, visualization_image):
        h, w, _ = visualization_image.shape
        assert point_cloud.shape[:2] == (h, w), "Image and point cloud dimensions mismatch."

        points = point_cloud.reshape(-1, 3)

        if self.color_mode == 'image':
            colors = visualization_image.reshape(-1, 3).astype(np.float32) / 255.0
        elif self.color_mode == 'proximity':
            dists = np.linalg.norm(points, axis=1)
            norm_dists = (dists - dists.min()) / (dists.max() - dists.min() + 1e-6)
            colors = plt.cm.coolwarm(norm_dists)[:, :3]  # Red-blue

        # Add one red point at origin (or any specific point)
        red_point = np.array([[0.0, 0.0, 0.0]])
        red_color = np.array([[1.0, 0.0, 0.0]])

        points = np.vstack([points, red_point])
        colors = np.vstack([colors, red_color])

        if self.backend == 'open3d':
            self.pcd.points = o3d.utility.Vector3dVector(points)
            self.pcd.colors = o3d.utility.Vector3dVector(colors)
            if self.block_plot:
                o3d.visualization.draw_geometries([self.pcd])
            else:
                vis = o3d.visualization.Visualizer()
                vis.create_window(window_name="Open3D Viewer", width=960, height=720, visible=True)
                vis.add_geometry(self.pcd)
                vis.poll_events()
                vis.update_renderer()
                vis.run()  # Starts non-blocking, auto closes
                vis.destroy_window()
        else:
            # Downsample to plot every n-th point
            n = 10  # Adjustable sampling rate
            points_sampled = points[::n]
            colors_sampled = colors[::n]

            self.ax.clear()
            self.ax.view_init(elev=-180, azim=-90)
            self.ax.scatter(points_sampled[:, 0], points_sampled[:, 1], points_sampled[:, 2], c=colors_sampled, s=1)

            # Set axis labels
            self.ax.set_xlabel("X")
            self.ax.set_ylabel("Y")
            self.ax.set_zlabel("Z")

            # Equal aspect ratio
            max_range = np.ptp(points_sampled, axis=0).max()  # peak-to-peak (range) of largest axis
            mid = points_sampled.mean(axis=0)
            self.ax.set_xlim(mid[0] - max_range / 2, mid[0] + max_range / 2)
            self.ax.set_ylim(mid[1] - max_range / 2, mid[1] + max_range / 2)
            # self.ax.set_zlim(mid[2] - max_range / 2, mid[2] + max_range / 2)
            if self.block_plot:
                plt.show()
            else:
                plt.pause(0.01)
