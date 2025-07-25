import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mapper3D_helper import *


class Mapper3D:
    def __init__(self, vis=False, plot=False, color_mode='constant', backend='open3d'):
        """
        color_mode: 'image', 'proximity', or 'constant'
        backend: 'open3d' or 'matplotlib'
        """
        assert color_mode in ['image', 'proximity', 'constant', 'none'], "color_mode must be 'image', 'proximity', or 'constant'"
        assert backend in ['open3d', 'matplotlib'], "backend must be 'open3d' or 'matplotlib'"
        self.color_mode = color_mode
        self.backend = backend
        self.pcd = o3d.geometry.PointCloud()
        self.vis = vis
        self.plot = plot

        if color_mode == 'constant':
            # Generate a random RGB color for all points
            self.constant_color = np.random.rand(3)
        else:
            self.constant_color = None

        if backend == 'matplotlib':
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection='3d')
            # Set initial view orientation
            self.ax.view_init(elev=-170, azim=-85)

    def process(self, metric_depth, metric_bg, color_img, mask_image, roll, pitch, yaw, 
                altitude, hfov):

        ## Scale depth data and background image to lie under 255.0 as float numbers
        # Metric depth data, such that the farthest point becomes 255.0 meters away
        metric_depth_scaled = metric_depth / np.max(metric_depth) * 255.0
        metric_bg_scaled = metric_bg / np.max(metric_depth) * 255.0

        # Setup background image ---------------------
        bg_scaled = np.where(metric_bg_scaled < metric_depth_scaled, metric_depth_scaled, metric_bg_scaled)

        # before = np.where(metric_depth < metric_bg, 255, 0).astype(np.uint8)
        # cv2.imshow("mask before rescaling", before)
        # after = np.where(metric_depth_scaled < metric_bg_scaled, 255, 0).astype(np.uint8)
        # cv2.imshow("mask after rescaling", after)
        # cv2.waitKey()
        
        assert np.max(metric_depth_scaled) <= 255.0 and np.min(metric_depth_scaled) >= 0.0, \
            f"max: {np.max(metric_depth_scaled)}, min: {np.min(metric_depth_scaled)}"
        assert np.max(bg_scaled) <= 255.0 and np.min(bg_scaled) >= 0.0, \
            f"max: {np.max(bg_scaled)}, min: {np.min(bg_scaled)}"

        # Compute ground elevation profile - What the true background must be, with 255.0 as max val
        gep_depth = calc_ground_depth(hfov, pitch_rad=pitch, output_shape=metric_depth.shape)

        # Replace visual background pattern with ground elevation pattern
        fimg = move_depth(metric_depth_scaled, bg_scaled, gep_depth)

        ## Generate point-clouds and scale them
        gep_pc, dirs = depthImage2pointCloud(gep_depth, roll_rad=roll, pitch_rad=pitch, yaw_rad=yaw,
                                       horizontal_fov=hfov)#, abs_alt=abs(altitude), geppc=gpc, mask=mask_image)#,
        gpc = np.zeros_like(color_img).astype(np.float32)
        gpc[:,:,2] = -abs(altitude)
        scale_factor = gpc[:,:,2] / gep_pc[:,:,2]
        gep_pc_scaled, dirs = depthImage2pointCloud(gep_depth, roll_rad=roll, pitch_rad=pitch, yaw_rad=yaw,
                                       horizontal_fov=hfov, scale_factor=scale_factor)
        rd_pc, dirs = depthImage2pointCloud(metric_depth_scaled, roll_rad=roll, pitch_rad=pitch,
                                      yaw_rad=yaw, horizontal_fov=hfov)

        # mean_bg_z = np.nanmean(np.where(mask_image>127, np.nan, rd_pc[:,:,2]))
        scale_factor = calc_scale_factor(gpc, rd_pc)
        rd_pc_scaled, dirs = depthImage2pointCloud(metric_depth_scaled, roll_rad=roll, pitch_rad=pitch,
                                             yaw_rad=yaw, horizontal_fov=hfov, 
                                             scale_factor=scale_factor)
        bg_pc_scaled, _ = depthImage2pointCloud(bg_scaled, roll_rad=roll, pitch_rad=pitch,
                                             yaw_rad=yaw, horizontal_fov=hfov, 
                                             scale_factor=scale_factor)
        fimg_pc, _ = depthImage2pointCloud(fimg, roll_rad=roll, pitch_rad=pitch, yaw_rad=yaw,
                                        horizontal_fov=hfov)
        scale_factor = calc_scale_factor(gpc, fimg_pc)
        fimg_pc_scaled, _ = depthImage2pointCloud(fimg, roll_rad=roll, pitch_rad=pitch, yaw_rad=yaw,
                                        horizontal_fov=hfov, scale_factor=scale_factor)
        
        reshaped_dropped = drop_depth(rd_pc_scaled, bg_pc_scaled, gep_pc_scaled)

        if self.plot:
            # self.plot_point_cloud(gep_pc_scaled, color_img, aug_points=gep_pc_scaled2, aug_img=np.zeros_like(color_img))
            # self.plot_point_cloud(gep_pc_scaled, color_img)
            # self.plot_point_cloud(gep_pc, color_img, dirs=dirs)
            # self.plot_point_cloud(rd_pc_scaled, color_img)
            # self.plot_point_cloud(rd_pc_scaled, color_img, save=True)
            # self.plot_point_cloud(rd_pc_scaled, color_img, save=True, dirs=dirs)
            # self.plot_point_cloud(rd_pc_scaled, color_img, aug_points=gep_pc_scaled, aug_img=np.zeros_like(color_img))
            # self.plot_point_cloud(rd_pc_scaled, color_img, aug_points=bg_pc_scaled, aug_img=np.zeros_like(color_img))
            # self.plot_point_cloud(bg_pc, color_img, aug_points=gep_pc, aug_img=color_img)
            # self.plot_point_cloud(rd_pc, color_img, aug_points=bg_pc, aug_img=np.zeros_like(color_img))
            # self.plot_point_cloud(rd_pc, color_img, aug_points=gep_pc_scaled,
            #                       aug_img=np.zeros_like(color_img), save=True)
            # self.plot_point_cloud(rd_pc_scaled, color_img, aug_points=gep_pc_scaled,
            #                       aug_img=np.zeros_like(color_img), save=True)
            # self.plot_point_cloud(fimg_pc, color_img, aug_points=gep_pc, aug_img=np.zeros_like(color_img))
            # self.plot_point_cloud(fimg_pc_scaled, color_img, save=True)
            self.plot_point_cloud(reshaped_dropped, color_img, save=True)
            # self.plot_point_cloud(bg_pc, np.zeros_like(color_img))

        if self.vis:
            cv2.imshow("Color Image", color_img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                return None, False
        return fimg_pc, True

    def plot_point_cloud(self, point_cloud, visualization_image=None, constant_color=None,
                     block_plot=True, aug_points=None, aug_img=None, save=False, dirs=None):
        points = point_cloud.reshape(-1, 3)
        geometries = []

        if self.color_mode == 'image':
            assert visualization_image is not None
            h, w, _ = visualization_image.shape
            assert point_cloud.shape[:2] == (h, w), "Image and point cloud dimensions mismatch."
            colors = visualization_image.reshape(-1, 3).astype(np.float32) / 255.0
            if aug_img is not None:
                aug_colors = aug_img.reshape(-1, 3).astype(np.float32) / 255.0
        elif self.color_mode == 'proximity':
            dists = np.linalg.norm(points, axis=1)
            norm_dists = (dists - dists.min()) / (dists.max() - dists.min() + 1e-6)
            colors = plt.cm.coolwarm(norm_dists)[:, :3]  # Red-blue
        elif self.color_mode == 'constant':
            assert constant_color is not None
            colors = np.tile(constant_color, (points.shape[0], 1))
        elif self.color_mode == 'none':
            pass
        else:
            raise ValueError(f"Unknown color_mode: {self.color_mode}")

        # Add one red point at origin
        red_point = np.array([[0.0, 0.0, 0.0]])
        red_color = np.array([[1.0, 0.0, 0.0]])
        points = np.vstack([points, red_point])
        if self.color_mode != 'none':
            colors = np.vstack([colors, red_color])

        if aug_points is not None:
            points = np.vstack([points, aug_points.reshape(-1, 3)])
            colors = np.vstack([colors, aug_colors])

        # Point cloud geometry
        self.pcd.points = o3d.utility.Vector3dVector(points)
        if self.color_mode != 'none':
            self.pcd.colors = o3d.utility.Vector3dVector(colors)
        geometries.append(self.pcd)

        # Add ray visualization if dirs is given
        if dirs is not None:
            H, W, _ = dirs.shape
            origin = np.array([[0, 0, 0]])
            ray_points = []
            ray_lines = []
            ray_colors = []

            idx = 0
            scale = 100  # Length of each ray
            ray_stride = 25  # <-- Downsampling factor (e.g. 2 for every second pixel)

            for i in range(0, H, ray_stride):
                for j in range(0, W, ray_stride):
                    d = dirs[i, j] * scale
                    ray_points.append(origin[0])
                    ray_points.append(d)
                    ray_lines.append([idx, idx + 1])
                    ray_colors.append([0, 0, 1])  # Blue
                    idx += 2

            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(ray_points)
            line_set.lines = o3d.utility.Vector2iVector(ray_lines)
            line_set.colors = o3d.utility.Vector3dVector(ray_colors)
            geometries.append(line_set)

        # Display
        if self.backend == 'open3d':
            if block_plot:
                o3d.visualization.draw_geometries(geometries)
            else:
                vis = o3d.visualization.Visualizer()
                vis.create_window(window_name="Open3D Viewer", width=960, height=720, visible=True)
                for g in geometries:
                    vis.add_geometry(g)
                vis.poll_events()
                vis.update_renderer()
                vis.run()
                vis.destroy_window()

            if save:
                o3d.io.write_point_cloud("output_cloud.pcd", self.pcd)
        else:
            # Downsample to plot every n-th point
            n = 500  # Adjustable sampling rate
            points_sampled = points[::n]
            colors_sampled = colors[::n]

            # Plot without clearing to allow overlay of multiple datasets
            self.ax.scatter(points_sampled[:, 0], points_sampled[:, 1], points_sampled[:, 2],
                            c=colors_sampled, s=1)
            # for k in range(points_sampled.shape[0]):
            #     self.ax.plot([0, points_sampled[k,0]], [0, points_sampled[k,1]], [abs(2), points_sampled[k,2]])

            # Set axis labels
            self.ax.set_xlabel("X")
            self.ax.set_ylabel("Y")
            self.ax.set_zlabel("Z")

            # Equal aspect ratio
            max_range = np.ptp(points_sampled, axis=0).max()
            mid = points_sampled.mean(axis=0)
            self.ax.set_xlim(mid[0] - max_range / 2, mid[0] + max_range / 2)
            self.ax.set_ylim(mid[1] - max_range / 2, mid[1] + max_range / 2)
            self.ax.set_zlim(mid[2] - max_range / 2, mid[2] + max_range / 2)

            if block_plot:
                plt.show()
            else:
                plt.pause(0.01)
