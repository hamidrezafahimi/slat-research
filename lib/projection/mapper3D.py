import threading
import cv2
import open3d as o3d
import open3d.visualization.gui as gui
import numpy as np
import matplotlib.pyplot as plt
from .helper import *
from utils.o3dviz import mat_mesh, fit_camera
from .config import *
from utils.io import save_pcd
from utils.conversion import pcm2pcd

class O3DGUI:
    def __init__(self, visMode):
        self.advance = threading.Event()
        self.scene_lock = threading.Lock()  
        self.visMode = visMode
        self.gui_thread = threading.Thread(target=self.do_gui, name="the-thread")
        self.gui_thread.start()

    def do_gui(self):
        if self.visMode == VisMode.Null:
            return  # No visualization, exit the method early

        gui.Application.instance.initialize()
        self.window = gui.Application.instance.create_window("3D Mapper", 800, 600)
        self.scene = gui.SceneWidget()
        self.scene.scene = o3d.visualization.rendering.Open3DScene(self.window.renderer)
        self.window.add_child(self.scene)
        # Create the coordinate frame (three perpendicular arrows)
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])
        # Add the coordinate frame to the scene
        self.scene.scene.add_geometry("axis", axis, o3d.visualization.rendering.MaterialRecord())
        # Set the key event to trigger the 'project' method on 'A'
        self.window.set_on_key(self.on_key)
        gui.Application.instance.run()

    def _mat_points(self, size=4.0):
        m = o3d.visualization.rendering.MaterialRecord()
        m.shader = "defaultUnlit"
        m.point_size = float(size)
        return m

    def on_key(self, e):
        if e.type != gui.KeyEvent.Type.DOWN:
            return False
        if e.key == gui.KeyName.A:
            print("user request: advance one frame")
            self.advance.set()
            return True
        elif e.key == gui.KeyName.Q:
            gui.Application.instance.quit(); 
            return True
        return False


class Evaluator3D(O3DGUI):
    def __init__(self):
        super().__init__()



class Mapper3D(O3DGUI):
    def __init__(self, config: Mapper3DConfig, fuser = None):
        assert config.color_mode in ['image', 'proximity', 'constant', 'none'], \
            "color_mode must be 'image', 'proximity', 'constant', or 'none'"
        self.config = config
        self.geomName = "points"
        self.it = 0
        self.projected_pcd = None
        self.fuser = fuser
        if (fuser is not None):
            self.config.output_dir = fuser.config.output_dir
            assert not self.config.in_camera, "When requesting fusion, the in_camera must be False (i.e. cam2nwu rotation required)"
        os.makedirs(self.config.output_dir, exist_ok=True)
        super().__init__(self.config.visMode)

    def project(self, metric_depth, pose, cimg, idx, bgpcd=None):
        """
        Projects the 3D point cloud based on the provided metric depth and pose, and shows the visualization.
        """
        # ======== Projection ========
        if self.fuser is None:
            # Projecting the raw depth, requires 'pyramid-projection'. The point cloud is translated
            # as well as being projected
            projected_pc, mc = project3D(metric_depth, pose, self.config.hfov_deg, move=self.config.on_video,
                                    pyramidProj=True, scaling=self.config.scaling, 
                                    do_rotate=(not self.config.in_camera))
            if bgpcd is not None and mc is not None:
                # The case: When 'raw-depth-projection' with provided bg - then move bg as well as above
                bgpcd.points = o3d.utility.Vector3dVector(np.asarray(bgpcd.points) + mc)
            filename = os.path.join(self.config.output_dir, f"rdproj_{idx}.pcd")
            
        else:
            cimg = resize_keep_ar(cimg, self.fuser.config.downsample_dstW)
            # Perform pattern fusion - the fuser scales the point cloud internally
            projected_pc = self.fuser.fuse_flat_ground(pose, metric_depth, bgpcd)
            if self.config.on_video:
                # Translate to current pose
                projected_pc += np.array([[pose.p6.x], [pose.p6.y], [pose.p6.z]]).T
            # No need to show the bg used for pattern fusion - the output is enough in viz
            bgpcd = None
            filename = os.path.join(self.config.output_dir, f"fusedproj_{idx}.pcd")

        _pcd = pcm2pcd(projected_pc, cimg)
        # ======== Write to disk ========
        if self.config.do_save:
            save_pcd(np.asarray(_pcd.points), np.asarray(_pcd.colors), filepath=filename)
            print(f"Wrote pcd file on {filename}")

        # ======== Visualize ========
        with self.scene_lock:
            if self.config.visMode == VisMode.MSingle:
                name = f"points_{self.it}"
                mname = f"bgpcd_{self.it}"
                self.scene.scene.remove_geometry(name)
                self.scene.scene.remove_geometry(mname)
            elif self.config.visMode == VisMode.MAccum:
                pass
            self.it += 1
            name = f"points_{self.it}"
            mname = f"bgpcd_{self.it}"
            self.scene.scene.add_geometry(name, _pcd, self._mat_points(5.0))
            if bgpcd is not None:
                self.scene.scene.add_geometry(mname, bgpcd, mat_mesh())
            fit_camera(self.scene.scene, [_pcd])


