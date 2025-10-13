import numpy as np
from .config import BGPatternFuserConfig, FlatFusionMode
from .helper import unfold_depth, drop_depth, ndfDrop_depth, calc_ground_depth
from projection.helper import project3D, calc_scale_factor
from geom.rectification import rectify_xy_proj
from utils.typeConversion import pcm2pcdArr
from projection.config import Scaling


class BGPatternFuser:
    def __init__(self, config: BGPatternFuserConfig):
        self.config = config
        self.shape = None

    def fuse_flat_ground(self, pose, rd_pc_scaled, bg_pcd_scaled):
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
            fused_pc = drop_depth(rd_pc_scaled, bg_pcd_scaled, gep_pc_scaled)

        elif self.config.flat_mode == FlatFusionMode.ndfDrop:
            fused_pc, base_elevs = ndfDrop_depth(rd_pc_scaled, bg_pcd_scaled, gep_pc_scaled)
            fused_pc_cam = fused_pc @ pose.getNWU2CAM().T
            fused_pc_cam = rectify_xy_proj(fused_pc_cam)
            fused_pc = fused_pc_cam @ pose.getCAM2NWU().T
            fused_pc *= calc_scale_factor(-abs(pose.p6.z), Scaling.MEAN_Z, bgz=base_elevs)

        elif self.config.flat_mode == FlatFusionMode.Unfold:
            fused_pc = unfold_depth(rd_pc_scaled, bg_pcd_scaled, gep_pc_scaled)

        else:
            raise ValueError("Unknown refusion mode")
        return fused_pc

    def fuse_elevation(self):
        assert False, "Not yet developed"