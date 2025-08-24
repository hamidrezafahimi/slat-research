import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../lib")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <pcd_file>")
        sys.exit(1)

    _file = sys.argv[1]
    
    import open3d as o3d
    p = o3d.io.read_point_cloud(_file)
    # Create a coordinate frame
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])

    # Draw both the point cloud and the axis
    o3d.visualization.draw_geometries([p, axis])