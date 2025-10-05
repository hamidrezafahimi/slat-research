import numpy as np
from .config import BGPatternFuserConfig, FlatFusionMode
from .helper import unfold_depth, drop_depth, ndfDrop_depth, calc_ground_depth
from projection.helper import project3DAndScale
from geom.surfaces import project_external_along_normals_noreject
from utils.typeConversion import pcm2pcdArr

class BGPatternFuser:
    def __init__(self, config: BGPatternFuserConfig):
        self.config = config
        self.shape = None

    def fuse_flat_ground(self, pose, rd_pc_scaled, bg_mesh):
        if self.shape is None:
            self.shape = rd_pc_scaled.shape[:2]
        else:
            assert self.shape == rd_pc_scaled.shape[:2], "Inconsistent image shape detected in fuser"

        gep_depth = calc_ground_depth(self.config.hfov_deg, pitch_rad=pose._pitch_rad, 
                                output_shape=self.shape)
        gep_pc_scaled, _ = project3DAndScale(gep_depth, pose, self.config.hfov_deg, self.shape)

        rdpcarr = pcm2pcdArr(rd_pc_scaled)
        bg_pc_scaled = project_external_along_normals_noreject(rdpcarr, bg_mesh)

        if self.config.flat_mode == FlatFusionMode.Replace_25D:
            assert False, "Currently no support on this legacy task"
            
        elif self.config.flat_mode == FlatFusionMode.Drop:
            fused_pc = drop_depth(rd_pc_scaled, bg_pc_scaled, gep_pc_scaled)

        elif self.config.flat_mode == FlatFusionMode.Unfold:
            fused_pc = unfold_depth(rd_pc_scaled, bg_pc_scaled, gep_pc_scaled)

        elif self.config.flat_mode == FlatFusionMode.ndfDrop:
            fused_pc = ndfDrop_depth(rd_pc_scaled, bg_pc_scaled, gep_pc_scaled)

        else:
            raise ValueError("Unknown refusion mode")
        return fused_pc

    def fuse_elevation(self):
        assert False, "Not yet developed"