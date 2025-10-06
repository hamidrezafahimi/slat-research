from dataclasses import dataclass
import os

@dataclass
class BGPatternDiffuserConfig:
    hfov_deg: float 
    output_dir: str
    coarsetune_iters: int = 400
    finetune_iters: int = 800
    viz: bool = False
    coarsetune_grid_w: int = 3
    coarsetune_grid_h: int = 3
    shift_k: float = 1.2
    finetune_grid_w: int = 6
    finetune_grid_h: int = 6
    coarsetunning_alpha: float = 0.15
    finetunning_alpha: float = 0.15
    fast: bool = False
    downsample_dstNum: int = 1e5
    scorebased_downsample_target: float = 0.1
    verbosity: str = "tiny"
    spline_mesh_samples_u: int = 40
    spline_mesh_samples_v: int = 40
    spline_mesh_marginal_ratio: float = 0.02
    scoring_smoothness_k: int = 10
    scoring_smoothness_kmin_neighbors: int = 8
    scoring_smoothness_neighbors_cap: int = 64
    tunning_eps: float = 1e-3
    tunning_moveAll: bool = False
    tunning_partialGrad: bool = False
    tunning_avgChangeTol: float = 0.0001
    tunning_varThresh: float = 100
    tunning_window: int = 10
    ct1iters: int = 30

    def __post_init__(self):
        # Enforce valid verbosity value
        allowed_verbosity = ["tiny", "none", "full"]
        if self.verbosity not in allowed_verbosity:
            raise ValueError(f"Invalid verbosity value: {self.verbosity}. Allowed values are {allowed_verbosity}.")
            
        self.output_dir = os.path.join(self.output_dir, "bgpat")


