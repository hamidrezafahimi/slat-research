import threading
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui

class PointsApp:
    def __init__(self):
        self.window = None
        self.scene = None
        self.pcd = None
        self.advance_flag = threading.Event()  # Event to signal point cloud generation
        self.pcd_lock = threading.Lock()  # Lock to protect point cloud access
        self.scene_lock = threading.Lock()  # Lock to protect scene updates

        self.gui_thread = threading.Thread(target=self.do_gui, name="the-thread")

    def do_gui(self):
        # GUI initialization
        gui.Application.instance.initialize()
        self.window = gui.Application.instance.create_window("Random Point Cloud", 800, 600)
        self.scene = gui.SceneWidget()
        self.scene.scene = o3d.visualization.rendering.Open3DScene(self.window.renderer)
        self.window.add_child(self.scene)

        # Initially generate a random point cloud
        self.pcd = self.generate_random_point_cloud()
        self.scene.scene.add_geometry("points", self.pcd, self._mat_points(5.0))

        # Set the key event to change the point cloud on 'a'
        self.window.set_on_key(self.on_key)

        gui.Application.instance.run()

    def generate_random_point_cloud(self):
        # Generate a random point cloud
        num_points = 1000
        points = np.random.rand(num_points, 3)  # Random 3D points
        colors = np.random.rand(num_points, 3)  # Random colors
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd

    def _mat_points(self, size=4.0):
        m = o3d.visualization.rendering.MaterialRecord()
        m.shader = "defaultUnlit"
        m.point_size = float(size)
        return m

    def on_key(self, e):
        if e.type != gui.KeyEvent.Type.DOWN:
            return False
        if e.key == gui.KeyName.A:
            # Set the advance flag when 'A' is pressed
            self.advance_flag.set()
            return True
        return False

    def update_point_cloud(self):
        # Generate a random point cloud and set the related attribute in the app class
        new_pcd = self.generate_random_point_cloud()
        # Lock access to the point cloud while updating
        with self.pcd_lock:
            self.pcd = new_pcd
        # Lock the scene before updating it
        with self.scene_lock:
            # Update the scene with the new point cloud
            self.scene.scene.remove_geometry("points")
            self.scene.scene.add_geometry("points", new_pcd, self._mat_points(5.0))

    def start(self):
        # Start the GUI in a separate thread
        self.gui_thread.start()

    def do_post_gui_work(self):
        # Main thread work (e.g., check advance flag and update point cloud)
        while True:
            # Wait until the flag is set
            self.advance_flag.wait()

            # Generate a random point cloud and update the app
            self.update_point_cloud()

            # Reset the advance flag
            self.advance_flag.clear()

            # Simulate some main thread work
            print("Main thread: Point cloud updated!")

if __name__ == "__main__":
    app = PointsApp()

    # Start the GUI in a separate thread
    app.start()
    # Now, perform additional work in the main thread
    app.do_post_gui_work()
    # If you want to wait for the GUI thread to complete (e.g., GUI closes), join the thread:
    app.gui_thread.join()
    print("Main thread: GUI thread has completed.")
