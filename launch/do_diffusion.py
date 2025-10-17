import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../lib")
from diffusion.diffusion import BGPatternDiffuser, BGPatternDiffuserConfig
from utils.io import IOHandler

def main():
    io = IOHandler(False)

    cfg = BGPatternDiffuserConfig(
        hfov_deg = io.cfg["hfov_deg"],
        output_dir = io.getDataRootDir()
    )

    b = BGPatternDiffuser(cfg)
    
    for dict in io.load():
        b.diffuse(dict["metric_depth"], 
                  dict["color_img"], 
                  dict["pose"], 
                  dict["idx"])

if __name__ == '__main__':
    main()