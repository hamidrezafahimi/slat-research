import open3d as o3d
import numpy as np
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../lib")
from geom.surfaces import cg_centeric_xy_spline, bspline_surface_mesh_from_ctrl
from projection.helper import project3DAndScale
from utils.typeConversion import pcm2pcd, pcd2pcdArr
# from utils.o3dviz import visualize_spline_mesh
from .tunning import Optimizer
from .scoring import Projection3DScorer
from .config import BGPatternDiffuserConfig
from .helper import downsample_pcd, RandomSurfacer, compute_shifted_ctrl_points, \
    create_grid_on_surface
from utils.o3dviz import visualize_spline_mesh
import time


class BGPatternDiffuser:
    def __init__(self, config: BGPatternDiffuserConfig):
        self.config = config
        self.scorer = Projection3DScorer(self.config)
        self.tunner = Optimizer(self.config)
        os.makedirs(self.config.output_dir, exist_ok=True)

    def diffuse(self, metric_depth, color_image, p, idx):
        print(f" =============== [BGPatternDiffuser] Diffusing on index {idx}")
        t0 = time.time()
        # if pose is None:
        #     # TODO: Write autopose func
        #     p = autopose(metric_depth)
        # else:
            # p = pose
        
        rd_pcm, _ = project3DAndScale(metric_depth, p, self.config.hfov_deg)
        rd_pcd = pcm2pcd(rd_pcm, color_image)
        if self.config.downsample_ratio != 1.0:
            rd_pcd = downsample_pcd(rd_pcd, self.config.downsample_ratio)

        rd_pcd_arr = pcd2pcdArr(rd_pcd)

        base_mesh, ctrl_pts, W, H, center_xy, z0 = cg_centeric_xy_spline(
            rd_pcd_arr, self.config.coarsetune_grid_w, self.config.coarsetune_grid_h, 
            self.config.spline_mesh_samples_u, self.config.spline_mesh_samples_v, 
            self.config.spline_mesh_marginal_ratio)
        
        if self.config.viz: # Pre-coarse-tune viz
            ctrl_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ctrl_pts))
            ctrl_pcd.paint_uniform_color([1.0, 0.2, 0.2])
            visualize_spline_mesh(ctrl_pcd, base_mesh, rd_pcd, name="pre coarse tune model")

        # -------------- Coarse-Tune the Back-Ground Model
        print(f"[BGPatternDiffuser] Start coarse tunning on index {idx}")
        rs = RandomSurfacer(cloud=rd_pcd_arr, grid_w=self.config.coarsetune_grid_w, 
                            grid_h=self.config.coarsetune_grid_h, 
                            samples_u=self.config.spline_mesh_samples_u,
                            samples_v=self.config.spline_mesh_samples_v,
                            margin=self.config.spline_mesh_marginal_ratio)
        max_dz = rs.OF
        self.scorer.reset(rd_pcd_arr, smoothness_base_mesh=base_mesh, max_dz=max_dz)
        _, _, coarse_tunned_z = self.tunner.tune(ctrl_pts.copy(), self.scorer, 
                                                 iters=self.config.coarsetune_iters,
                                                 alpha=self.config.coarsetunning_alpha)
        coarse_tunned = ctrl_pts.copy()
        coarse_tunned[:,2] = coarse_tunned_z

        print(f"[BGPatternDiffuser] Coarse tunning done on index {idx}")
        # ------------- Prepare for Fine-Tunning
        _, nonshifted_mesh, _, shifted_mesh = compute_shifted_ctrl_points(rd_pcd_arr, coarse_tunned, 
                                                    self.config.spline_mesh_samples_u,
                                                    self.config.spline_mesh_samples_v,
                                                    self.config.shift_k)
        if self.config.viz: # Post-coarse-tune viz
            ctrl_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(coarse_tunned))
            ctrl_pcd.paint_uniform_color([1.0, 0.2, 0.2])
            visualize_spline_mesh(ctrl_pcd, nonshifted_mesh, rd_pcd, name="post coarse tune model")

        upsampled_ctrl, _ = create_grid_on_surface(shifted_mesh, self.config.finetune_grid_w,
                                                   self.config.finetune_grid_h)

        if self.config.viz: # Pre-fine-tune viz
            ctrl_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(upsampled_ctrl))
            ctrl_pcd.paint_uniform_color([1.0, 0.2, 0.2])
            visualize_spline_mesh(ctrl_pcd, shifted_mesh, rd_pcd, name="pre fine tune model")

        # ------------- Fine-Tune the Back-Ground Model
        print(f"[BGPatternDiffuser] Start fine tunning on index {idx}")
        max_dz = rs.OF / 8.0
        self.scorer.reset(rd_pcd_arr, smoothness_base_mesh=shifted_mesh, max_dz=max_dz, fine_tune=True)
        _, _, fine_tunned_z = self.tunner.tune(upsampled_ctrl, self.scorer,
                                               iters=self.config.finetune_iters,
                                               alpha=self.config.finetunning_alpha)
        fine_tunned = upsampled_ctrl.copy()
        fine_tunned[:,2] = fine_tunned_z
        if self.config.viz: # Post-fine-tune viz
            ctrl_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(fine_tunned))
            ctrl_pcd.paint_uniform_color([1.0, 0.2, 0.2])
            mesh = bspline_surface_mesh_from_ctrl(fine_tunned, self.config.finetune_grid_w, 
                                            self.config.finetune_grid_h, 
                                            self.config.spline_mesh_samples_u, 
                                            self.config.spline_mesh_samples_v)
            visualize_spline_mesh(ctrl_pcd, mesh, rd_pcd, name="pre fine tune model")

        filename = f"bg_{idx}.csv"
        outname = os.path.join(self.config.output_dir, filename)
        np.savetxt(outname, fine_tunned, delimiter=',')
        print(f"[BGPatternDiffuser] Fine tunning done on index {idx}, data written to {outname}. Time: {(time.time() - t0):.2f} sec")
