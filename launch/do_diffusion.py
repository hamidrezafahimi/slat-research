import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../lib")
from diffusion.BGPatternDiffuser import BGPatternDiffuser, BGPatternDiffuserConfig
from utils.io import IOHandler

def main():
    io = IOHandler(False)

    cfg = BGPatternDiffuserConfig(
        hfov_deg = 90.0,
        output_dir = io.getDataRootDir()
    )

    b = BGPatternDiffuser(cfg)
    
    for p, metric_depth, color_img, idx, *_ in io.load():
        b.diffuse(metric_depth, color_img, p, idx)

if __name__ == '__main__':
    main()