import os
import numpy as np
from .config import BGPatternFuserConfig, FlatFusionMode
from .helper import unfold_depth, drop_depth, NDFDrop_depth, calc_ground_depth, \
    depyramidize_pointCloud, downsample_pcm
from projection.helper import project3D, calc_scale_factor, NULL_SCALE_MIN_Z
from geom.rectification import rectify_xy_proj
from utils.conversion import pcm2pcdArr, pcdArr2pcm
from projection.config import Scaling


class BGPatternFuser:
    def __init__(self, config: BGPatternFuserConfig):
        self.config = config
        self.shape = None
        # os.makedirs(self.config.output_dir, exist_ok=True)

    # def fuse_flat_ground(self, pose, rd_pc_scaled, bg_pcd_scaled):
    def fuse_flat_ground(self, pose, metric_depth, bg_pcd_scaled):
        if self.config.flat_mode == FlatFusionMode.Depyramidize:
            # 'Depyramidize' fusion method requires pyramid projection math
            rd_pc_scaled, _ = project3D(metric_depth, pose, self.config.hfov_deg, move=False, 
                                        scaling=Scaling.NULL, do_rotate=True, pyramidProj=True)
        else:
            # Typical radial projection
            rd_pc_scaled, _ = project3D(metric_depth, pose, self.config.hfov_deg, move=False, 
                                        scaling=Scaling.NULL, do_rotate=True)
        
        rd_pc_scaled = downsample_pcm(rd_pc_scaled, self.config.downsample_dstW)
        
        if self.shape is None:
            self.shape = rd_pc_scaled.shape[:2]
        else:
            assert self.shape == rd_pc_scaled.shape[:2], "Inconsistent image shape detected in fuser"

        gep_depth = calc_ground_depth(self.config.hfov_deg, p = pose, 
                                output_shape=self.shape)
        gep_pc_scaled, _ = project3D(gep_depth, pose, self.config.hfov_deg, 
                                             scaling=Scaling.NULL, move=False)
        # bg_pc_scaled = pcd2pcm(bg_pcd_scaled, 1000, 100)
        rdpcarr = pcm2pcdArr(rd_pc_scaled)
        # bg_pc_scaled = project_external_along_normals_noreject(rdpcarr, bg_mesh)

        if self.config.flat_mode == FlatFusionMode.Replace_25D:
            assert False, "Currently no support on this legacy task"
            
        elif self.config.flat_mode == FlatFusionMode.Drop:
            assert bg_pcd_scaled is not None, "'Drop' fusion mode requires background data"
            fused_pc = drop_depth(rd_pc_scaled, bg_pcd_scaled, gep_pc_scaled)

        elif self.config.flat_mode == FlatFusionMode.NDFDrop:
            assert bg_pcd_scaled is not None, "'NDFDrop' fusion mode requires background data"
            fused_pc, base_elevs = NDFDrop_depth(rd_pc_scaled, bg_pcd_scaled, gep_pc_scaled)
            fused_pc_cam = fused_pc @ pose.getNWU2CAM().T
            fused_pc_cam = rectify_xy_proj(fused_pc_cam)
            fused_pc = fused_pc_cam @ pose.getCAM2NWU().T
            # fused_pc *= calc_scale_factor(-abs(pose.p6.z), Scaling.MEAN_Z, bgz=base_elevs)
            fused_pc *= calc_scale_factor(-abs(pose.p6.z), Scaling.MEAN_Z, 
                                          bgz=np.ones_like(rd_pc_scaled[:,:,2])*NULL_SCALE_MIN_Z)

        elif self.config.flat_mode == FlatFusionMode.Unfold:
            assert bg_pcd_scaled is not None, "'Unfold' fusion mode requires background data"
            fused_pc = unfold_depth(rd_pc_scaled, bg_pcd_scaled, gep_pc_scaled)
        
        elif self.config.flat_mode == FlatFusionMode.Depyramidize:
            rd_pc_scaled_cam = rd_pc_scaled @ pose.getNWU2CAM().T
            rdpcarr_cam = pcm2pcdArr(rd_pc_scaled_cam)
            fus_pc_arr_cam, _, _ = depyramidize_pointCloud(rdpcarr_cam)
            fused_pc_cam = pcdArr2pcm(fus_pc_arr_cam, rd_pc_scaled.shape[0], rd_pc_scaled.shape[1])
            fused_pc = fused_pc_cam @ pose.getCAM2NWU().T
            fused_pc *= calc_scale_factor(-abs(pose.p6.z), Scaling.MEAN_Z, 
                                          bgz=np.ones_like(rd_pc_scaled[:,:,2])*NULL_SCALE_MIN_Z)

        else:
            raise ValueError("Unknown refusion mode")
        return fused_pc

    def fuse_elevation(self):
        assert False, "Not yet developed"