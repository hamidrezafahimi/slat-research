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

    def process(self, metric_depth, metric_bg, bg_image, color_img, mask_image, roll, pitch, yaw, 
                altitude, hfov):

        ## Scale depth data and background image to lie under 255.0 as float numbers
        # Metric depth data, such that the farthest point becomes 255.0 meters away
        metric_depth_scaled = metric_depth / np.max(metric_depth) * 255.0
        # bg_image = bg_image / np.max(metric_depth) * 255.0
        metric_bg = metric_bg / np.max(metric_depth) * 255.0

        ## Setup background image ---------------------
        # 1. metric background
        bg_scaled = np.where(metric_bg < metric_depth_scaled, metric_depth_scaled, metric_bg)

        # 2. Visual background 
        # bg_scaled = np.where(bg_image < metric_depth_scaled, metric_depth_scaled, bg_image)

        # cv2.imwrite('/home/hamid/depth.png', metric_depth_scaled)
        before = np.where(metric_depth_scaled < metric_bg, 255, 0).astype(np.uint8)
        cv2.imshow("mask before rescaling", before)
        # after = np.where(metric_depth_scaled < metric_bg_scaled, 255, 0).astype(np.uint8)
        # cv2.imshow("mask after rescaling", after)
        # cv2.imshow("bg pat", bg_scaled.astype(np.uint8))
        # bg_scaled = np.where(bg_scaled < metric_depth_scaled, metric_depth_scaled, bg_scaled)
        # print(np.min(depth_image))

        # cv2.imshow('s', bg_image)
        # cv2.imshow('d', depth_image)
        # bg_scaled = np.min(metric_depth_scaled) + \
        #     ((np.max(metric_depth_scaled) - np.min(metric_depth_scaled)) / 255.0) * (bg_image.astype(np.float32) - np.min(bg_image.astype(np.float32)))
        # min_val = np.min(metric_depth_scaled)
        # bg_scaled = min_val + ((1.0 - (min_val / 255.0)) * bg_image.astype(np.float32))
        # bg_scaled = metric_depth_scaled - (depth_image.astype(np.float32) - bg_image.astype(np.float32))
        # metric_depth_scaled, bg_scaled = unified_scale(metric_depth, bg_image.astype(np.float32))
        # bg_scaled = np.where(mask_image > 127, )
        cv2.waitKey()
        
        assert np.max(metric_depth_scaled) <= 255.0 and np.min(metric_depth_scaled) >= 0.0, \
            f"max: {np.max(metric_depth_scaled)}, min: {np.min(metric_depth_scaled)}"
        assert np.max(bg_scaled) <= 255.0 and np.min(bg_scaled) >= 0.0, \
            f"max: {np.max(bg_scaled)}, min: {np.min(bg_scaled)}"

        # Compute ground elevation profile - What the true background must be, with 255.0 as max val
        gep_depth = calc_ground_depth(hfov, pitch_rad=pitch, output_shape=metric_depth.shape)
        # print("---", np.min(gep_depth), np.max(gep_depth), np.min(bg_scaled), np.max(bg_scaled))
        # if (np.min(gep_depth) > np.min(bg_scaled)):
        #     gep_depth -= (np.min(gep_depth) - np.min(bg_scaled))
        # print(f"max: {np.max(gep_depth)}, min: {np.min(gep_depth)}")
        # print(f"max: {np.max(depth_image)}, min: {np.min(depth_image)}")
        # print(f"max: {np.max(bg_image)}, min: {np.min(bg_image)}")
        # print(f"max: {np.max(metric_depth_scaled)}, min: {np.min(metric_depth_scaled)}")
        # print(f"max: {np.max(bg_scaled)}, min: {np.min(bg_scaled)}")
        
        # cv2.imshow("gep depth", gep_depth.astype(np.uint8))
        # cv2.imwrite('/home/hamid/gep.png', gep_depth)
        # cv2.imshow("bg_scaled", bg_scaled.astype(np.uint8))
        # fimg = transform_depth(metric_depth_scaled, bg_scaled, gep_depth)
        fimg = move_depth(metric_depth_scaled, bg_scaled, gep_depth)
        # np.savetxt('/home/hamid/fimg.csv', fimg, delimiter=',')
        # np.savetxt('/home/hamid/metric_depth_scaled.csv', metric_depth_scaled, delimiter=',')
        # np.savetxt('/home/hamid/mask.csv', mask_image, delimiter=',')
        # fimg = np.where(mask_image > 127, metric_depth_scaled, gep_depth)

        # index_mask = np.where(mask_image < 127 and gep_depth <= 254.0, True, False)
        # index_mask = np.zeros_like(mask_image)
        # for i in range(index_mask.shape[0]):
        #     for j in range(index_mask.shape[1]):
        #         index_mask[i,j] = (gep_depth[i,j] <= 100 and mask_image[i,j] < 127)
        # print(bg_scaled[262,416])
        # mean_raw_depth_pc_z = np.nanmean(np.where(index_mask, rd_pc[:,:,2], np.nan))
        # rd_pc = depthImage2pointCloud(metric_depth_scaled, roll_rad=roll, pitch_rad=pitch,
        #                               yaw_rad=yaw, horizontal_fov=hfov, abs_alt=abs(altitude),
        #                               mean_z_gep_pc = mean_raw_depth_pc_z*3.5)
        gep_pc = depthImage2pointCloud(gep_depth, roll_rad=roll, pitch_rad=pitch, yaw_rad=yaw,
                                       horizontal_fov=hfov)#, abs_alt=abs(altitude), geppc=gpc, mask=mask_image)#,
        gpc = np.zeros_like(color_img).astype(np.float32)
        gpc[:,:,2] = -abs(altitude)
        scale_factor = gpc[:,:,2] / gep_pc[:,:,2]
        gep_pc_scaled = depthImage2pointCloud(gep_depth, roll_rad=roll, pitch_rad=pitch, yaw_rad=yaw,
                                       horizontal_fov=hfov, scale_factor=scale_factor)#, abs_alt=abs(altitude), geppc=gpc, mask=mask_image)#,
        # print(gpc[0,0,2], gep_pc[0,0,2])
        # bg_pc = depthImage2pointCloud(bg_scaled, roll_rad=roll, pitch_rad=pitch, yaw_rad=yaw,
        #                               horizontal_fov=hfov)#,
        rd_pc = depthImage2pointCloud(metric_depth_scaled, roll_rad=roll, pitch_rad=pitch,
                                      yaw_rad=yaw, horizontal_fov=hfov)

        # mean_bg_z = np.nanmean(np.where(mask_image>127, np.nan, rd_pc[:,:,2]))
        mean_bg_z = np.min(rd_pc[:,:,2])
        scale_factor = gpc[:,:,2] / mean_bg_z
        rd_pc_scaled = depthImage2pointCloud(metric_depth_scaled, roll_rad=roll, pitch_rad=pitch,
                                             yaw_rad=yaw, horizontal_fov=hfov, 
                                             scale_factor=scale_factor)#, abs_alt=abs(altitude),
        
        fimg_pc = depthImage2pointCloud(fimg, roll_rad=roll, pitch_rad=pitch, yaw_rad=yaw,
                                        horizontal_fov=hfov)

        # np.savetxt('/home/hamid/rebef.csv', metric_depth_scaled/bg_scaled, delimiter=',')
        # np.savetxt('/home/hamid/reaf.csv', fimg/gep_depth, delimiter=',')
        # np.savetxt('/home/hamid/gep_d.csv', gep_depth, delimiter=',')
        # np.savetxt('/home/hamid/fimg.csv', fimg_pc[:,:,2], delimiter=',')
        # np.savetxt('/home/hamid/metric_depth_scaled.csv', metric_depth_scaled, delimiter=',')
        # np.savetxt('/home/hamid/bg_s.csv', bg_scaled, delimiter=',')

        if self.plot:
            # self.plot_point_cloud(gep_pc_scaled, color_img)
            self.plot_point_cloud(rd_pc_scaled, color_img, save=True)
            # self.plot_point_cloud(rd_pc_scaled, color_img, aug_points=gep_pc_scaled, aug_img=np.zeros_like(color_img))
            # self.plot_point_cloud(bg_pc, color_img, aug_points=gep_pc, aug_img=color_img)
            # self.plot_point_cloud(rd_pc, color_img, aug_points=bg_pc, aug_img=np.zeros_like(color_img))
            # self.plot_point_cloud(rd_pc, color_img, aug_points=gep_pc_scaled, 
            #                       aug_img=np.zeros_like(color_img), save=True)
            # self.plot_point_cloud(rd_pc_scaled, color_img, aug_points=gep_pc_scaled, 
            #                       aug_img=np.zeros_like(color_img), save=True)
            self.plot_point_cloud(fimg_pc, color_img, aug_points=gep_pc, aug_img=np.zeros_like(color_img))
            # self.plot_point_cloud(bg_pc, np.zeros_like(color_img))

        if self.vis:
            cv2.imshow("Color Image", color_img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                return None, False
        return fimg_pc, True

    def plot_point_cloud(self, point_cloud, visualization_image=None, constant_color=None,
                         block_plot=True, aug_points=None, aug_img=None, save=False):
        points = point_cloud.reshape(-1, 3)
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
            # Use the constant color for all points
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

        if self.backend == 'open3d':
            self.pcd.points = o3d.utility.Vector3dVector(points)
            if self.color_mode != 'none':
                self.pcd.colors = o3d.utility.Vector3dVector(colors)
            if block_plot:
                o3d.visualization.draw_geometries([self.pcd])
            else:
                vis = o3d.visualization.Visualizer()
                vis.create_window(window_name="Open3D Viewer", width=960, height=720, visible=True)
                vis.add_geometry(self.pcd)
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
