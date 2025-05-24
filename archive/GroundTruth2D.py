import numpy as np
import matplotlib.pyplot as plt
from config import Config

class GroundTruth2D:
    def __init__(self, camera_position, theta_deg, fov_deg, num_of_pix):
        self.camera_position = camera_position
        self.theta_deg = theta_deg
        self.fov_deg = fov_deg
        self.num_of_pix = num_of_pix
        self.radial_lines = []
        self.terrain_points = np.zeros((self.num_of_pix + 2, 2))
        self.intersection_points = []  # all intersection points for plotting
        self.closest_points = []  # closest points for highlighting
        self.distances = np.zeros((self.num_of_pix, 1))
        self.normalized_distances = np.zeros((self.num_of_pix, 1))
        self.max_gep = 0
        self.cfg = Config("config.yaml")
    
    def main(self):
        self.compute_radial_lines()
        self.terrain_generator()
        self.find_intersections()
        self.get_distances()
        self.normalize_and_map()

    def compute_radial_lines(self):
        fov_rad = np.deg2rad(self.fov_deg)
        theta_rad = np.deg2rad(self.theta_deg)
        angles = np.linspace(theta_rad - (fov_rad/2), theta_rad + (fov_rad/2), self.num_of_pix)
        x_camera, y_camera = self.camera_position
        
        for angle in angles:
            if np.tan(angle) == 0:
                continue
            # Calculate intersection with the ground (y = 0)
            # y - y_cam = m (x - x_cam) --> y = 0
            # x_intersect = x_cam - y_cam/m
            x_intersect = x_camera - y_camera / np.tan(angle)
            
            # Add the radial line endpoint
            self.radial_lines.append(((x_camera, y_camera), (x_intersect, 0)))

        # maximum ground end point length
        self.max_gep = np.sqrt((x_camera - self.radial_lines[-1][1][0])**2 + 
                               (y_camera - self.radial_lines[-1][1][1])**2)
        
        self.observed_ground_distance = self.radial_lines[-1][1][0] - self.radial_lines[0][1][0]
        self.max_ground_distance = y_camera

    def terrain_generator(self):
        y_terrain = np.zeros(self.num_of_pix)
        size_percentage = self.cfg.get("size_percentage")
        height_percentage = self.cfg.get("height_percentage")
        obstacle_position_percentage = self.cfg.get("obstacle_position_percentage")
        mid_start = int(np.floor(obstacle_position_percentage*self.num_of_pix))
        mid_end = int(mid_start + size_percentage*self.num_of_pix)
        
        y_terrain[mid_start:mid_end] = height_percentage * self.max_ground_distance

        stx, _ = self.radial_lines[0][1]
        enx, _ = self.radial_lines[-1][1]
        x_points = np.linspace(stx, enx+0.1, self.num_of_pix)
        self.terrain_points[0][0] = x_points[0]
        self.terrain_points[0][1] = 0
        self.terrain_points[-1][0] = x_points[-1]
        self.terrain_points[-1][1] = 0
        for i in range(self.num_of_pix):
            self.terrain_points[i+1][0] = x_points[i]
            self.terrain_points[i+1][1] = y_terrain[i]
    def _line_intersection(self, p1, p2, q1, q2):
        """
        Check the intersection of two line segments (p1-p2 and q1-q2).
        Returns the intersection point if it exists, otherwise None.
        """
        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        r = (p2[0] - p1[0], p2[1] - p1[1])
        s = (q2[0] - q1[0], q2[1] - q1[1])
        rs_cross = det(r, s)
        qp = (q1[0] - p1[0], q1[1] - p1[1])

        if rs_cross == 0:  # Lines are parallel or collinear
            return None

        t = det(qp, s) / rs_cross
        u = det(qp, r) / rs_cross

        if 0 <= t <= 1 and 0 <= u <= 1:  # Intersection within both segments
            intersection = (p1[0] + t * r[0], p1[1] + t * r[1])
            return intersection

        return None

    def find_intersections(self):
        """
        Find intersections of radial lines with terrain segments.

        Returns:
        - A list of tuples, each containing:
          (radial_line_index, intersection_point, distance_from_camera)
        - If no intersection exists for a radial line, the tuple contains None.
        """
        intersections = []
        x_camera, y_camera = self.camera_position

        # for i, radial_line in enumerate(self.radial_lines):
        for i, radial_line in enumerate(reversed(self.radial_lines)):
            x1, y1 = radial_line[0]
            x2, y2 = radial_line[1]
            closest_distance = None
            closest_point = None

            # Iterate over terrain segments
            for j in range(len(self.terrain_points) - 1):
                t1 = self.terrain_points[j]
                t2 = self.terrain_points[j + 1]
                
                # Check intersection with this terrain segment
                intersection = self._line_intersection((x1, y1), (x2, y2), t1, t2)

                if intersection:
                    # print(f"Radial Line {i} intersects Terrain Segment {j}-{j+1} at {intersection}")
                    self.intersection_points.append(intersection)  # Store for plotting

                    # Calculate distance to camera
                    distance = np.sqrt((x_camera - intersection[0]) ** 2 + (y_camera - intersection[1]) ** 2)
                    if closest_distance is None or distance < closest_distance:
                        closest_distance = distance
                        closest_point = intersection

            if closest_point is not None:
                intersections.append((i, closest_point, closest_distance))
                self.closest_points.append(closest_point)  # Store closest point for highlighting
            else:
                intersections.append((i, None, None))
        
        return intersections

    def normalize_and_map(self):
        
        # max_gep --> 255
        # dist --> x
        # x = dist*255 / max_gep
        for i in enumerate(self.distances):
            self.normalized_distances[i[0]] = int(255 - ((i[1]*255) / self.max_gep))
            # print (i[0], i[1], self.normalized_distances[i[0]])
    
    def get_distances(self):
        for i in enumerate(self.closest_points):
            camera_x, camera_y = self.camera_position
            self.distances[i[0]] = np.sqrt((camera_x - i[1][0])**2 + (camera_y - i[1][1])**2)
        return self.distances
    
    def get_closest_points(self):
        return np.array([list(cp) for cp in self.closest_points]).reshape((len(self.closest_points), 2))
    
    def get_ground_truth(self):
        return self.terrain_points
    
    def get_seen_points(self):
        return self.closest_points

    def get_depth_data(self):
        temp_array = []
        for i in self.normalized_distances:
            temp_array.append(int(i))
        return temp_array
    
    def plot(self):
        plt.figure(figsize=(8, 8))

        # Plot camera position
        plt.plot(self.camera_position[0], self.camera_position[1], 'bo', label="Camera Position")

        # Plot radial lines
        for line in self.radial_lines:
            x_vals = [line[0][0], line[1][0]]
            y_vals = [line[0][1], line[1][1]]
            plt.plot(x_vals, y_vals, 'b-', alpha=0.3)

        # Plot terrain points
        terrain_x = [point[0] for point in self.terrain_points]
        terrain_y = [point[1] for point in self.terrain_points]
        plt.plot(terrain_x, terrain_y, color='black', label="Terrain Points", markersize=3)

        # Plot intersection points
        if self.intersection_points:
            inter_x = [p[0] for p in self.intersection_points]
            inter_y = [p[1] for p in self.intersection_points]
            plt.plot(inter_x, inter_y, 'gx', label="Intersection Points")

        # Highlight closest points in red
        if self.closest_points:
            closest_x = [p[0] for p in self.closest_points]
            closest_y = [p[1] for p in self.closest_points]
            plt.plot(closest_x, closest_y, 'ro', label="Closest Points", markersize=5)

        # Ground line at y = 0
        plt.axhline(0, color='gray', linestyle='--', label="Ground (y=0)")

        # Configure plot
        plt.xlim(-1, 27)
        plt.ylim(-1, 12)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Camera, Radial Lines, Terrain, and Intersections")
        plt.legend()
        plt.grid()
        plt.show()