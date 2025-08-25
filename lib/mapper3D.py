import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mapper3D_helper import *
from enum import Enum, auto
from kinematics.pose import Pose
from dataclasses import dataclass

class RefusionMode(Enum):
    Replace_25D = auto()
    Drop = auto()
    Unfold = auto()

@dataclass
class Mapper3DConfig:
    ref_mode: RefusionMode
    hfov_deg: float
    shape: tuple = None
    vis: bool = False
    plot: bool = False
    color_mode: str = 'constant'   # Options: 'image', 'proximity', 'constant', 'none'
    backend: str = 'open3d'        # Options: 'open3d', 'matplotlib'

class Mapper3D:
    def __init__(self, config: Mapper3DConfig):
        assert config.color_mode in ['image', 'proximity', 'constant', 'none'], \
            "color_mode must be 'image', 'proximity', 'constant', or 'none'"
        assert config.backend in ['open3d', 'matplotlib'], \
            "backend must be 'open3d' or 'matplotlib'"

        self.config = config
        self.pcd = o3d.geometry.PointCloud()

        # Constant color if needed
        self.constant_color = np.random.rand(3) if self.config.color_mode == 'constant' else None

        # Setup matplotlib backend if selected
        if self.config.backend == 'matplotlib':
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.ax.view_init(elev=-170, azim=-85)

    def generate_gep_data(self, pose: Pose):
        # Compute ground elevation profile - What the true background must be, with 255.0 as max val
        gep_depth = calc_ground_depth(self.config.hfov_deg, pitch_rad=pose._pitch_rad, 
                                      output_shape=self.config.shape)

        gep_pc_scaled = project3DAndScale(gep_depth, pose, self.config.hfov_deg, self.config.shape)
        return gep_pc_scaled, gep_depth

    def process(self, metric_depth, metric_bg, pose, color_img):
        assert metric_depth.shape == metric_bg.shape == color_img.shape[:2], \
            f"shapes: depth: {metric_depth.shape}, bg: {metric_bg.shape}, color: {color_img.shape}"
        if self.config.shape is None:
            self.config.shape = metric_depth.shape
        
        ## Scale depth data and background image to lie under 255.0 as float numbers
        # Metric depth data, such that the farthest point becomes 255.0 meters away
        metric_depth_s = metric_depth / np.max(metric_depth) * 255.0
        metric_bg_s = metric_bg / np.max(metric_depth) * 255.0

        gep_pc_scaled, gep_depth = self.generate_gep_data(pose=pose)

        rd_pc_scaled = project3DAndScale(metric_depth_s, pose, self.config.hfov_deg, 
                                         self.config.shape)
        
        bg_pc_scaled = project3DAndScale(metric_bg_s, pose, self.config.hfov_deg, 
                                         self.config.shape)
        # print(metric_bg_s.shape)
        # np.savetxt('/home/hamid/sdfa.csv', metric_bg, delimiter=',')

        # Select processing strategy
        if self.config.ref_mode == RefusionMode.Replace_25D:
            bg_scaled = np.where(metric_bg_s < metric_depth_s, metric_depth_s, metric_bg_s)
            fimg = move_depth(metric_depth_s, bg_scaled, gep_depth)
            refused_pc = project3DAndScale(fimg, pose, self.config.hfov_deg, self.config.shape)
            
        elif self.config.ref_mode == RefusionMode.Drop:
            refused_pc = drop_depth(rd_pc_scaled, bg_pc_scaled, gep_pc_scaled)

        elif self.config.ref_mode == RefusionMode.Unfold:
            refused_pc = unfold_depth(rd_pc_scaled, bg_pc_scaled, gep_pc_scaled)

        else:
            raise ValueError("Unknown refusion mode")

        if self.config.plot:
            # pass
            self.plot_point_cloud(refused_pc, color_img, aug_points=rd_pc_scaled, 
                                  aug_img=np.zeros_like(color_img))
            # self.plot_point_cloud(rd_pc_scaled, color_img, aug_points=gep_pc_scaled, 
            #                       aug_img=np.zeros_like(color_img))
            # self.plot_point_cloud(refused_pc, color_img, save=True, filename="refused_pc")
            # self.plot_point_cloud(gep_pc_scaled, color_img, save=True, filename="gep_pc_scaled")
            # self.plot_point_cloud(rd_pc_scaled, color_img, save=True, filename="rd_pc_scaled")
            # plot_point_cloud(bg_pc_scaled, color_img)

        if self.config.vis:
            cv2.imshow("Color Image", color_img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                return None, False
        return refused_pc
    
    def plot_point_cloud(self, point_cloud, visualization_image=None, constant_color=None,
                     block_plot=True, aug_points=None, aug_img=None, save=False, dirs=None, filename = "output"):
        points = point_cloud.reshape(-1, 3)
        geometries = []
        if self.config.color_mode == 'image':
            assert visualization_image is not None
            h, w, _ = visualization_image.shape
            assert point_cloud.shape[:2] == (h, w), "Image and point cloud dimensions mismatch."
            colors = visualization_image.reshape(-1, 3).astype(np.float32) / 255.0
            if aug_img is not None:
                aug_colors = aug_img.reshape(-1, 3).astype(np.float32) / 255.0
        elif self.config.color_mode == 'proximity':
            dists = np.linalg.norm(points, axis=1)
            norm_dists = (dists - dists.min()) / (dists.max() - dists.min() + 1e-6)
            colors = plt.cm.coolwarm(norm_dists)[:, :3]  # Red-blue
        elif self.config.color_mode == 'constant':
            assert constant_color is not None
            colors = np.tile(constant_color, (points.shape[0], 1))
        elif self.config.color_mode == 'none':
            pass
        else:
            raise ValueError(f"Unknown color_mode: {self.config.color_mode}")

        # Add one red point at origin
        red_point = np.array([[0.0, 0.0, 0.0]])
        red_color = np.array([[1.0, 0.0, 0.0]])
        points = np.vstack([points, red_point])
        if self.config.color_mode != 'none':
            colors = np.vstack([colors, red_color])

        if aug_points is not None:
            points = np.vstack([points, aug_points.reshape(-1, 3)])
            colors = np.vstack([colors, aug_colors])

        # Point cloud geometry
        self.pcd.points = o3d.utility.Vector3dVector(points)
        if self.config.color_mode != 'none':
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
        if self.config.backend == 'open3d':
            if block_plot:
                print(111)
                # o3d.visualization.draw_geometries(geometries)
                o3d.visualization.draw_geometries([self.pcd])
                print(113)
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
                output_filename = str(filename) + ".pcd"
                o3d.io.write_point_cloud(output_filename, self.pcd)
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
