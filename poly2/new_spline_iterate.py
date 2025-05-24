import cv2
import numpy as np
from scipy.interpolate import LSQBivariateSpline

def apply_threshold(image, threshold_pattern):
    return np.where(image > threshold_pattern, 255, 0).astype(np.uint8)

class VisualSplineFit:
    def __init__(self, x_coeffs=4, degree_x=3, degree_y=3):
        self.degree_x = degree_x
        self.degree_y = degree_y
        self.num_coeffs_x = x_coeffs
        self.params_init = False

    def initParams(self, img_shape):
        self.H, self.W = img_shape
        # Compute how many coefficients we want in the y-direction
        self.num_coeffs_y = int(np.round((float(self.H) / self.W) * self.num_coeffs_x))
        
        # Create [0..1] grids
        X, Y = np.meshgrid(
            np.linspace(0, 1, self.W),
            np.linspace(0, 1, self.H)
        )
        self.x = X.ravel()
        self.y = Y.ravel()

        # Setup knots
        self.num_knots_x = self.num_coeffs_x + self.degree_x + 1
        self.num_knots_y = self.num_coeffs_y + self.degree_y + 1
        self.num_inner_knots_x = self.num_knots_x - 2 * self.degree_x
        self.num_inner_knots_y = self.num_knots_y - 2 * self.degree_y

        self.inner_knots_x = np.linspace(0, 1, self.num_inner_knots_x + 2)[1:-1]
        self.inner_knots_y = np.linspace(0, 1, self.num_inner_knots_y + 2)[1:-1]

        self.params_init = True

    def fit(self, img_gray, bin_img):
        """
        Fits a 2D B-spline to `img_gray` only where `bin_img == 0`.
        
        Returns the fitted surface (H x W) as a float64 NumPy array.
        """
        if not self.params_init:
            self.initParams(img_gray.shape)

        # Flatten grayscale intensities
        z_full = img_gray.astype(np.float64).ravel()

        # Build the mask for valid pixels (binary == 0)
        # Make sure to flatten the binary image so it matches x,y,z shape
        mask = (bin_img.ravel() == 0)
        
        # Keep only masked pixels
        x_masked = self.x[mask]
        y_masked = self.y[mask]
        z_masked = z_full[mask]

        # Fit the spline using only the masked pixel values
        spline = LSQBivariateSpline(
            x_masked, 
            y_masked, 
            z_masked,
            tx=self.inner_knots_x,
            ty=self.inner_knots_y,
            kx=self.degree_x,
            ky=self.degree_y
        )

        # Evaluate on the full grid (0..1) in x and y
        x_eval = np.linspace(0, 1, self.W)
        y_eval = np.linspace(0, 1, self.H)
        fitted_2d = spline(x_eval, y_eval)  # shape => (len(y_eval), len(x_eval))

        # LSQBivariateSpline.__call__ returns (len(y_eval), len(x_eval)).
        # For consistency, reorder if needed so fitted_2d is (H, W):
        if fitted_2d.shape[0] != self.H:
            fitted_2d = fitted_2d.T

        return fitted_2d


import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def nothing(x):
    pass

def main():
    # 1) Load images
    org_img = "depth_image.jpg"
    edg_path = "edge_image.jpg"
    gray_path = "depth_image_bgsMasked.jpg"
    bin_path =  "depth_image_bgsBin.jpg"

    img_org = cv2.imread(org_img, cv2.IMREAD_GRAYSCALE)
    img_gray = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)
    img_bin = cv2.imread(bin_path, cv2.IMREAD_GRAYSCALE)
    img_edg = cv2.imread(edg_path, cv2.IMREAD_GRAYSCALE)

    if img_gray is None or img_bin is None or img_org is None or img_edg is None:
        print("Could not load input images.")
        return

    # Binarize img_edg
    img_edg = np.where(img_edg >= 25, 255, 0).astype(np.uint8)

    # 2) Fit spline surface
    spline_fit = VisualSplineFit(x_coeffs=4, degree_x=2, degree_y=2)
    fitted_surface = spline_fit.fit(img_gray, img_bin)
    fitted_8u = np.clip(fitted_surface, 0, 255).astype(np.uint8)

    best_k = None
    max_overlap = -1
    best_masked = None

    for K in range(1, 20):
        m = apply_threshold(img_org, fitted_8u + K)
        detected_edges = cv2.Canny(m, 0, 255)

        # Optional: dilate detected_edges to increase overlap chance
        kernel = np.ones((5, 5), np.uint8)
        detected_edges = cv2.dilate(detected_edges, kernel, iterations=1)

        # Calculate overlap between detected_edges and img_edg
        overlap = np.logical_and(detected_edges == 255, img_edg == 255).sum()

        if overlap > max_overlap:
            max_overlap = overlap
            best_k = K
            best_masked = np.where(m == 0, img_org, 0).astype(np.uint8)

    print(f"Best K: {best_k} with overlap: {max_overlap}")
    cv2.imshow("Best new_masked", best_masked)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
