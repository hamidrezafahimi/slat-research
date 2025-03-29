import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import LSQBivariateSpline

class VisualSplineFit:
    def __init__(self, x_coeffs=4, degree_x=3, degree_y=3):
        """
        A simple 2D spline-fitting helper for a grayscale image of shape (H, W).
        
        :param img_shape: (H, W) from the image
        :param degree_x, degree_y: B-spline degrees in x and y directions.
        
        We set:
            num_coeffs_x = W/40
            num_coeffs_y = H/40
        internally, but you can tweak that as needed.
        """
        self.degree_x = degree_x
        self.degree_y = degree_y
        self.num_coeffs_x = x_coeffs
        self.params_init = False

    def initParams(self, img_shape):
        # For the spline-fitting step, we need "input" points x, y in [0..1].
        # We'll build a mesh for the knots. We'll store them in raveled form
        # just for LSQBivariateSpline input:
        self.H, self.W = img_shape  # shape of the image
        self.num_coeffs_y = int(np.round((float(self.H)/self.W) * self.num_coeffs_x))
        print("cx", self.num_coeffs_x)
        print("cy", self.num_coeffs_y)

        X, Y = np.meshgrid(
            np.linspace(0, 1, self.W),  # length W
            np.linspace(0, 1, self.H)   # length H
        )
        self.x = X.ravel()  # shape => (H*W,)
        self.y = Y.ravel()  # shape => (H*W,)

        # Derived values for the knots
        self.num_knots_x = self.num_coeffs_x + self.degree_x + 1
        self.num_knots_y = self.num_coeffs_y + self.degree_y + 1
        self.num_inner_knots_x = self.num_knots_x - 2 * self.degree_x
        self.num_inner_knots_y = self.num_knots_y - 2 * self.degree_y

        # Build the interior knots (exclude the very edges [0,1])
        self.inner_knots_x = np.linspace(0, 1, self.num_inner_knots_x + 2)[1:-1]
        self.inner_knots_y = np.linspace(0, 1, self.num_inner_knots_y + 2)[1:-1]
        self.params_init = True

    def fit(self, img):
        """
        Fits a 2D B-spline to the image data Z (shape HxW) using LSQBivariateSpline.
        
        Returns the fitted surface (H x W) as a NumPy array (float64).
        """
        if not self.params_init:
            self.initParams(img.shape)

        # Convert to float (so we can fit a continuous surface to the intensities)
        # Flatten the original image intensities
        z = img.astype(np.float64).ravel()

        # 1) Fit the spline to the (x, y, z) data
        spline = LSQBivariateSpline(
            self.x,          # sorted or unsorted is okay for fitting
            self.y,
            z,
            tx=self.inner_knots_x,
            ty=self.inner_knots_y,
            kx=self.degree_x,
            ky=self.degree_y
        )

        # 2) Evaluate on strictly increasing 1D arrays in [0..1].
        #    By default, LSQBivariateSpline.__call__ uses grid=True
        #    and returns an array of shape (len(new_y), len(new_x)).
        x_sorted = np.linspace(0, 1, self.W)  # strictly increasing
        y_sorted = np.linspace(0, 1, self.H)  # strictly increasing

        fitted_2d = spline(x_sorted, y_sorted)  # shape => (H, W)

        if fitted_2d.shape[1] == self.H and fitted_2d.shape[0] == self.W:
            fitted_2d = fitted_2d.T

        return fitted_2d


def main():
    # --------------------------------------------------------------------
    # 1) Read a monochrome image (grayscale) from disk
    # --------------------------------------------------------------------
    image_path = "00000001.jpg"
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        print(f"Could not read image: {image_path}")
        return

    # --------------------------------------------------------------------
    # 2) Instantiate our spline-fitting helper and fit the surface
    # --------------------------------------------------------------------
    spline_fit = VisualSplineFit()

    # Returns the fitted surface as shape (H, W)
    fitted_surface = spline_fit.fit(img_gray)

    # --------------------------------------------------------------------
    # 3) Show the comparison in 3D (Matplotlib)
    # --------------------------------------------------------------------
    # Create a coordinate mesh for plotting
    H, W = img_gray.shape
    x_plot = np.linspace(0, 1, W)
    y_plot = np.linspace(0, 1, H)
    X_plot, Y_plot = np.meshgrid(x_plot, y_plot)

    fig = plt.figure(figsize=(12, 5))

    # Original
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf1 = ax1.plot_surface(X_plot, Y_plot, img_gray, cmap='viridis', edgecolor='none', alpha=0.8)
    ax1.set_title("Original Image Surface")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Intensity")
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)

    # Fitted
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    surf2 = ax2.plot_surface(X_plot, Y_plot, fitted_surface, cmap='viridis', edgecolor='none', alpha=0.8)
    ax2.set_title("Fitted Spline Surface")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Intensity")
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)

    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------------------
    # 4) Also show the images in 2D (OpenCV)
    # --------------------------------------------------------------------
    # Convert the fitted surface to 8-bit for display
    fitted_8u = np.clip(fitted_surface, 0, 255).astype(np.uint8)

    # Show original image
    cv2.imshow("Original Image", img_gray)
    # Show fitted "image" (which is the spline approximation)
    cv2.imshow("Fitted Image", fitted_8u)
    cv2.imwrite("fds.jpg", fitted_8u)

    print("Press any key in the OpenCV window to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
