import threading
import cv2
import open3d as o3d
import open3d.visualization.gui as gui
import numpy as np
import matplotlib.pyplot as plt
from .helper import *
from utils.o3dviz import mat_mesh, fit_camera
from .config import *


class O3DGUI:
    def __init__(self, visMode):
        self.advance = threading.Event()
        self.scene_lock = threading.Lock()  
        self.visMode = visMode;
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
        os.makedirs(self.config.output_dir, exist_ok=True)
        super().__init__(self.config.visMode)

    def project(self, metric_depth, pose, cimg, bgpcd=None):
        """
        Projects the 3D point cloud based on the provided metric depth and pose, and shows the visualization.
        """
        # Generate the point cloud from metric depth and pose
        if self.fuser is None:
            # Projecting the raw depth, requires 'pyramid-projection'. The point cloud is translated
            # as well as being projected
            fresh_pc, mc = project3D(metric_depth, pose, self.config.hfov_deg, move=True,
                                    pyramidProj=True, scaling=self.config.scaling)
        else:
            # Projecting to be fused later, requires 'spherical-projection'. The point cloud is not 
            # translated yet
            fresh_pc, mc = project3D(metric_depth, pose, self.config.hfov_deg, move=False, 
                                    scaling=Scaling.NULL)

        if bgpcd is not None and mc is not None:
            # The case: When 'raw-depth-projection' with provided bg - then move bg as well as above
            bgpcd.points = o3d.utility.Vector3dVector(np.asarray(bgpcd.points) + mc)
        
        projected_pc = self.downsample_pcm(fresh_pc, self.config.downsample_dstW)
        cimg = resize_keep_ar(cimg, self.config.downsample_dstW)
        if self.fuser is not None:
            # Perform pattern fusion - the fuser scales the point cloud internally
            assert bgpcd is not None
            projected_pc = self.fuser.fuse_flat_ground(pose, projected_pc, bgpcd)
            # Translate to current pose
            projected_pc += np.array([[pose.p6.x], [pose.p6.y], [pose.p6.z]]).T
            # No need to show the bg used for pattern fusion - the output is enough in viz
            bgpcd = None

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
            pcd = self.pcm2pcd(projected_pc, cimg)
            self.projected_pcd = pcd
            self.scene.scene.add_geometry(name, pcd, self._mat_points(5.0))
            if bgpcd is not None:
                self.scene.scene.add_geometry(mname, bgpcd, mat_mesh())
            fit_camera(self.scene.scene, [pcd])

    def pcm2pcd(self, pcm, visualization_image):
        points = pcm.reshape(-1, 3)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if self.fuser is not None:
            visualization_image = self.downsample_pcm(visualization_image, self.config.downsample_dstW)
        if self.config.color_mode == 'image':
            assert visualization_image is not None
            h, w, _ = visualization_image.shape
            assert pcm.shape[:2] == (h, w), "Image and point cloud dimensions mismatch."
            colors = visualization_image.reshape(-1, 3).astype(np.float32) / 255.0
        elif self.config.color_mode == 'proximity':
            dists = np.linalg.norm(points, axis=1)
            norm_dists = (dists - dists.min()) / (dists.max() - dists.min() + 1e-6)
            colors = plt.cm.coolwarm(norm_dists)[:, :3]  # Red-blue
        elif self.config.color_mode == 'none':
            pass
        else:
            raise ValueError(f"Unknown color_mode: {self.config.color_mode}")
        pcd.colors = o3d.utility.Vector3dVector(colors[:, [2,1,0]])
        return pcd
    
    def downsample_pcm(self, pcm: np.ndarray, W_final: int) -> np.ndarray:
        """
        Downsample a PCM (H, W, 3) numpy array to a target width, keeping aspect ratio.
        
        Args:
            pcm (np.ndarray): Input PCM array of shape (H, W, 3)
            W_final (int): Desired final width after downsampling
            
        Returns:
            np.ndarray: Downsampled PCM array (H_new, W_final, 3)
        """
        # --- Validation ---
        if not isinstance(pcm, np.ndarray):
            raise TypeError("pcm must be a numpy array")
        if pcm.ndim != 3 or pcm.shape[2] != 3:
            raise ValueError("pcm must have shape (H, W, 3)")
        if W_final <= 0:
            raise ValueError("W_final must be a positive integer")

        H, W, _ = pcm.shape

        # --- Compute new size while keeping aspect ratio ---
        ratio = W_final / W
        H_final = int(H * ratio)

        # --- Downsample using OpenCV with area interpolation ---
        pcm_down = cv2.resize(pcm, (W_final, H_final), interpolation=cv2.INTER_AREA)

        return pcm_down
    
    def saveDepth(self, idx) -> np.ndarray:
        filename = f"projected_{idx}.pcd"
        path = os.path.join(self.config.output_dir, filename)
        success = o3d.io.write_point_cloud(path, self.projected_pcd)
        if success:
            print(f"Projected point cloud saved successfully to: {path}")
        else:
            print(f"Failed to save point cloud to: {path}")


