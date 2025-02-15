import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# # Create example 2D surfaces with safe scaling to avoid uint8 overflows
# rows, cols = 100, 100
# x = np.linspace(0, 1000, cols)
# y = np.linspace(0, 1000, rows)
# xs, ys = np.meshgrid(x, y)

# # Ensure no overloading happens by pre-scaling the calculations
# f_surface = 2 * np.sqrt(xs + ys)
# f_surface = np.clip(f_surface, 0, 255).astype(np.uint8)

# d_surface = 10 + 0.0002 * (xs + ys) ** 2
# d_surface = np.clip(d_surface, 0, 255).astype(np.uint8)

# d_max_surface = np.max(d_surface)

# Function to scale d surface
def scale_d_surface(d, f):
    """
    Scale the d surface so that it is saturated between the f surface and d_max.
    Keeps d_max unchanged and ensures all values are above the corresponding f values.

    Args:
        d (np.ndarray): Input surface (2D array).
        f (np.ndarray): Reference surface (2D array).
        d_max (int): Maximum value of d to remain unchanged.

    Returns:
        np.ndarray: Scaled version of d.
    """
    d_max = np.max(d)
    # Ensure inputs are floats for scaling calculations
    d = d.astype(float)
    f = f.astype(float)

    # Calculate element-wise scaling
    min_d = np.min(d)  # Minimum of d
    min_f_d = np.maximum(f, min_d)  # Ensure d values are above f

    # Rescale d
    scaled_d = (d - min_d) / (d_max - min_d) * (d_max - min_f_d) + min_f_d

    # Clip values to [0, 255] and convert back to uint8
    scaled_d = np.clip(scaled_d, 0, 255).astype(np.uint8)
    return scaled_d

if __name__ == '__main__':

    # Create example 2D surfaces with safe scaling
    rows, cols = 100, 100
    x = np.linspace(0, 1000, cols)
    y = np.linspace(0, 1000, rows)
    xs, ys = np.meshgrid(x, y)

    # Ensure no overloading happens by pre-scaling the calculations
    f_surface = 2 * np.sqrt(xs + ys)
    f_surface = np.clip(f_surface, 0, 255).astype(np.uint8)

    d_surface = 10 + 0.0002 * (xs + ys) ** 2
    d_surface = np.clip(d_surface, 0, 255).astype(np.uint8)

    d_max_surface = np.max(d_surface)

    # Apply the updated scaling function
    scaled_d_surface = scale_d_surface(d_surface, f_surface, d_max_surface)

    # Plot results in 3D
    fig = plt.figure(figsize=(18, 6))

    # Original f surface
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xs, ys, f_surface, cmap='viridis', edgecolor='none')
    # ax.set_title('f Surface (Reference)')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Value')

    # Original d surface
    # ax = fig.add_subplot(132, projection='3d')
    ax.plot_surface(xs, ys, d_surface, cmap='plasma', edgecolor='none')
    # ax.set_title('d Surface (Original)')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Value')

    # Scaled d surface
    # ax = fig.add_subplot(133, projection='3d')
    ax.plot_surface(xs, ys, scaled_d_surface, cmap='inferno', edgecolor='none')
    ax.set_title('Scaled d Surface (Transformed)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Value')

    plt.tight_layout()
    plt.show()

