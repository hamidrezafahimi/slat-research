import cv2
import numpy as np
from scipy.interpolate import LSQBivariateSpline
from scipy.spatial import Delaunay

def apply_threshold(image, threshold_pattern):
    return np.where(image > threshold_pattern, 255, 0).astype(np.uint8)

class VisualSplineFit:
    def __init__(self, x_coeffs=4, degree_x=3, degree_y=3):
        """
        Same constructor signature as before.
        """
        self.degree_x = degree_x
        self.degree_y = degree_y
        self.num_coeffs_x = x_coeffs
        self.params_init = False

    def initParams(self, img_shape):
        """
        Prepare the (x, y) grids and knot vectors.
        """
        self.H, self.W = img_shape
        self.num_coeffs_y = int(np.round((float(self.H)/self.W) * self.num_coeffs_x))

        # Build x,y in [0..1] for each pixel
        X, Y = np.meshgrid(
            np.linspace(0, 1, self.W),
            np.linspace(0, 1, self.H)
        )
        self.x = X.ravel()  # shape => (H*W,)
        self.y = Y.ravel()  # shape => (H*W,)

        # Knots
        self.num_knots_x = self.num_coeffs_x + self.degree_x + 1
        self.num_knots_y = self.num_coeffs_y + self.degree_y + 1
        self.num_inner_knots_x = self.num_knots_x - 2 * self.degree_x
        self.num_inner_knots_y = self.num_knots_y - 2 * self.degree_y

        self.inner_knots_x = np.linspace(0, 1, self.num_inner_knots_x + 2)[1:-1]
        self.inner_knots_y = np.linspace(0, 1, self.num_inner_knots_y + 2)[1:-1]

        self.params_init = True

    def fit(self, img_gray, bin_img):
        """
        Fit a cubic spline to the known region (where bin_img == 0),
        and use a linear plane for extrapolation (points outside the
        smallest convex boundary).
        
        Returns the fitted surface (H x W) as float64.
        """
        if not self.params_init:
            self.initParams(img_gray.shape)

        # 1) Flatten the grayscale image
        z_full = img_gray.astype(np.float64).ravel()
        # Build mask for known data (e.g. bin_img == 0)
        mask = (bin_img.ravel() == 0)

        # x_masked, y_masked, z_masked => known points
        x_masked = self.x[mask]
        y_masked = self.y[mask]
        z_masked = z_full[mask]

        if len(z_masked) < 3:
            raise ValueError("Not enough known data points to fit a spline or plane!")

        # 2) Fit a cubic B-spline for the *interpolation* region
        spline = LSQBivariateSpline(
            x_masked,
            y_masked,
            z_masked,
            tx=self.inner_knots_x,
            ty=self.inner_knots_y,
            kx=self.degree_x,
            ky=self.degree_y
        )

        # 3) Fit a linear plane z = a + b*x + c*y as a fallback for extrapolation
        #    We'll do a simple least-squares fit:
        ones = np.ones_like(x_masked)
        A = np.column_stack((ones, x_masked, y_masked))  # shape: (N, 3)
        coefs, _, _, _ = np.linalg.lstsq(A, z_masked, rcond=None)
        # coefs => [a, b, c]
        a, b, c = coefs

        # 4) Identify the smallest convex region (the convex hull) of the known data
        #    We'll use a 2D Delaunay triangulation to check membership inside the hull.
        points_known = np.column_stack((x_masked, y_masked))
        tri = Delaunay(points_known)

        # 5) Evaluate the spline and linear plane at every pixel in [0..1]
        x_eval = np.linspace(0, 1, self.W)
        y_eval = np.linspace(0, 1, self.H)

        # Evaluate the spline on the full grid => shape (H, W)
        z_spline_2d = spline(x_eval, y_eval)
        # If shape mismatch: transpose to (H, W)
        if z_spline_2d.shape != (self.H, self.W):
            z_spline_2d = z_spline_2d.T

        # Evaluate the linear plane on the full grid
        # We'll build the mesh again so we can do a direct broadcast.
        X_eval, Y_eval = np.meshgrid(x_eval, y_eval)
        Z_linear_2d = a + b * X_eval + c * Y_eval  # shape (H, W)

        # 6) For each pixel, determine if (x,y) is INSIDE the convex hull:
        #    tri.find_simplex(...) >= 0 => INSIDE;  -1 => OUTSIDE
        #    We'll flatten X_eval, Y_eval, then reshape the membership check
        flat_eval = np.column_stack((X_eval.ravel(), Y_eval.ravel()))
        inside_idx = tri.find_simplex(flat_eval) >= 0
        inside_mask = inside_idx.reshape((self.H, self.W))  # shape (H, W)

        # 7) Combine the two surfaces:
        #    Use the spline for interpolation (inside), linear model for extrapolation (outside).
        fitted_2d = np.where(inside_mask, z_spline_2d, Z_linear_2d)

        return fitted_2d

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

    if img_gray is None or img_bin is None or img_org is None:
        print("Could not load input images.")
        return

    # 2) Instantiate spline-fitting helper
    spline_fit = VisualSplineFit(x_coeffs=2, degree_x=2, degree_y=2)

    # 3) Fit using only pixels where the binary image is 0
    fitted_surface = spline_fit.fit(img_gray, img_bin)

    # 4) Convert fitted surface to 8-bit image
    fitted_8u = np.clip(fitted_surface, 0, 255).astype(np.uint8)

    # 5) Create window and trackbar
    cv2.namedWindow("fitted_only_where_bin0.png")
    cv2.createTrackbar("K", "fitted_only_where_bin0.png", 3, 50, nothing)

    while True:
        # Get current value of K from trackbar
        K = cv2.getTrackbarPos("K", "fitted_only_where_bin0.png")

        m = apply_threshold(img_org, fitted_8u + K)
        new_masked = np.where(m == 0, img_org, 0).astype(np.uint8)

        cv2.imshow("fitted_only_where_bin0.png", new_masked)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to exit
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
