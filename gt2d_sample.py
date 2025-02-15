import numpy as np
import matplotlib.pyplot as plt
from GroundTruth2D import GroundTruth2D
from config import Config

# Script to use the class
if __name__ == "__main__":
    # cfg = Config("config.yaml")

    # some_value = cfg.get("size_percentage")
    # print(f"Some value: {some_value}")
    # print(type(some_value))

    # Camera parameters
    camera_position = (0, 9)
    theta = -45  # Central angle in radians
    fov_deg = 50       # Field of view in degrees
    num_of_pix = 330

    # Create the camera radial lines object
    gt2d = GroundTruth2D(camera_position, theta, fov_deg, num_of_pix)
    gt2d.main()
    depth_data = gt2d.get_depth_data()
    depth_data = np.array(depth_data, dtype=float)
    gt2d.plot()
