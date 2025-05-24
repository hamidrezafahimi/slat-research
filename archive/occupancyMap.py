import numpy as np
import matplotlib.pyplot as plt

# Generate random point clouds (x, y, z)
def generate_point_clouds(num_points):
    return np.random.rand(num_points, 3) * 10  # Points in range [0, 10]

# Process point clouds
def process_point_clouds(all_point_clouds, step):
    # Extract x, y, z coordinates
    x_coords = all_point_clouds[:, 0]
    y_coords = all_point_clouds[:, 1]
    z_coords = all_point_clouds[:, 2]

    # Find maximum x and y values
    max_x = np.max(x_coords)
    max_y = np.max(y_coords)

    print(f"Step {step}: Max x: {max_x}, Max y: {max_y}")

    # Divide the 2D space into 1x1 grid cells
    grid_size = 1  # Grid size
    num_x_cells = int(np.ceil(max_x / grid_size))
    num_y_cells = int(np.ceil(max_y / grid_size))

    # Initialize grid counts and z comparison results
    grid_counts = np.zeros((num_x_cells, num_y_cells), dtype=int)
    grid_z_check = np.zeros((num_x_cells, num_y_cells), dtype=bool)

    # Count points in each grid cell and evaluate z conditions
    for x, y, z in zip(x_coords, y_coords, z_coords):
        cell_x = int(x // grid_size)
        cell_y = int(y // grid_size)
        grid_counts[cell_x, cell_y] += 1

        # Count z values in each grid cell
        if z >= 5:
            grid_z_check[cell_x, cell_y] += 1
        else:
            grid_z_check[cell_x, cell_y] -= 1

    # Print grid results
    for i in range(num_x_cells):
        for j in range(num_y_cells):
            points_ge5 = max(0, grid_z_check[i, j])  # Points with z >= 5
            points_lt5 = grid_counts[i, j] - points_ge5  # Points with z < 5
            result = points_ge5 > points_lt5
            print(f"Grid cell ({i}, {j}): z>=5: {points_ge5}, z<5: {points_lt5}, Result: {result}")

    # Visualize the points and grid
    fig, ax = plt.subplots()
    ax.scatter(x_coords, y_coords, c=z_coords, cmap='coolwarm', label='Points (color by z)', s=10)

    # Draw grid
    for i in range(num_x_cells + 1):
        ax.axvline(i * grid_size, color='gray', linestyle='--', linewidth=0.5)
    for j in range(num_y_cells + 1):
        ax.axhline(j * grid_size, color='gray', linestyle='--', linewidth=0.5)

    # Set plot limits
    ax.set_xlim(0, num_x_cells * grid_size)
    ax.set_ylim(0, num_y_cells * grid_size)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"Point Clouds with Grid (Step {step})")
    ax.legend()
    plt.show()

# Main program
if __name__ == "__main__":
    num_steps = 15  # Total number of steps
    points_per_step = 50  # Points to add at each step

    # Initialize an empty list to store all point clouds
    all_point_clouds = np.empty((0, 3))

    for step in range(1, num_steps + 1):
        # Generate new point clouds
        new_point_clouds = generate_point_clouds(points_per_step)

        # Append new points to the existing point clouds
        all_point_clouds = np.vstack((all_point_clouds, new_point_clouds))

        # Process and visualize the combined point clouds
        process_point_clouds(all_point_clouds, step)
