import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../lib")
from projection.mapper3D import Mapper3D, Mapper3DConfig
from utils.io import IOHandler
from fusion.BGPatternFuser import BGPatternFuser
from fusion.config import BGPatternFuserConfig

def main():
    global mapper
    io = IOHandler(True)

    # Config
    cfg = Mapper3DConfig(color_mode='image',   # Options: 'image', 'proximity', 'constant', 'none'
                         hfov_deg=io.cfg["hfov_deg"],
                         output_dir = io.getDataRootDir())

    if io.getDoFuse():
        fcfg = BGPatternFuserConfig(
            hfov_deg=io.cfg["hfov_deg"],
            output_dir = io.getDataRootDir()
        )
        f = BGPatternFuser(config=fcfg)
        mapper = Mapper3D(cfg, fuser=f)
    else:
        mapper = Mapper3D(cfg)

    # Main thread work (e.g., check advance flag and update point cloud)
    for dict in io.load():
        # Wait until the flag is set
        mapper.advance.wait()
        print(f"[info] projecting the new frame: {dict['idx']} ...")
        # Generate a random point cloud and update the app
        mapper.project(dict["metric_depth"], dict["pose"], dict["color_img"], 
                       dict["bg"])
        print(f"[info] projection done and scene updated for index {dict['idx']}.")
        # Reset the advance flag
        mapper.advance.clear()
        if io.getDoSave():
            mapper.saveDepth(dict['idx'])


if __name__ == "__main__":
    main()
