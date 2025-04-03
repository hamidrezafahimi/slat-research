import numpy as np
from visualSplineFit_helper import *
from scipy.interpolate import LSQBivariateSpline
from scipy.spatial import ConvexHull  # for hull boundary checks

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
        Prepare (x, y) grids and knot vectors for the spline.
        """
        self.H, self.W = img_shape

        # Rough guess for how many coeffs in y direction
        self.num_coeffs_y = int(np.round((float(self.H) / self.W) * self.num_coeffs_x))

        # x, y in [0..1] for each pixel
        X, Y = np.meshgrid(
            np.linspace(0, 1, self.W),  # shape => (H, W)
            np.linspace(0, 1, self.H)
        )
        self.x_flat = X.ravel()  # shape => (H*W,)
        self.y_flat = Y.ravel()  # shape => (H*W,)

        # Prepare knots
        self.num_knots_x = self.num_coeffs_x + self.degree_x + 1
        self.num_knots_y = self.num_coeffs_y + self.degree_y + 1
        self.num_inner_knots_x = self.num_knots_x - 2 * self.degree_x
        self.num_inner_knots_y = self.num_knots_y - 2 * self.degree_y

        self.inner_knots_x = np.linspace(0, 1, self.num_inner_knots_x + 2)[1:-1]
        self.inner_knots_y = np.linspace(0, 1, self.num_inner_knots_y + 2)[1:-1]

        self.params_init = True

    def fit(self, img_gray, bin_img, do_clip=True):
        """
        For interpolation (inside the convex hull of known data) => use spline.
        For extrapolation (outside that hull) => do row-by-row linear extension.
        
        Additionally, after building the final fitted surface:
         - Saturate values to [0..255].
         - Compute the error as abs(fitted - original) for pixels where bin_img==0,
           and 0 where bin_img!=0.
         - Return that error map as a uint8 image (0 means exact fit, 255 means
           abs diff of 255).
        """
        if not self.params_init:
            self.initParams(img_gray.shape)

        # Flatten grayscale intensities
        z_full = img_gray.astype(np.float64).ravel()

        # Known data: e.g. bin == 0
        mask_known = (bin_img.ravel() == 0)
        x_masked = self.x_flat[mask_known]
        y_masked = self.y_flat[mask_known]
        z_masked = z_full[mask_known]

        if len(z_masked) < 2:
            raise ValueError("Not enough known points to fit anything!")

        # 1) Fit Cubic Spline inside the region
        spline = LSQBivariateSpline(
            x_masked,
            y_masked,
            z_masked,
            tx=self.inner_knots_x,
            ty=self.inner_knots_y,
            kx=self.degree_x,
            ky=self.degree_y
        )

        # Evaluate the spline over the full grid => shape (H, W)
        x_eval = np.linspace(0, 1, self.W)
        y_eval = np.linspace(0, 1, self.H)

        z_spline_2d = spline(x_eval, y_eval)  # => shape (len(y_eval), len(x_eval))
        if z_spline_2d.shape != (self.H, self.W):
            z_spline_2d = z_spline_2d.T  # ensure (H, W)

        # 2) Build the convex hull of known points for inside/outside tests
        points_known = np.column_stack([x_masked, y_masked])
        if len(points_known) < 3:
            # With <3 points, we can't form a polygon. Treat everything as outside.
            hull_mask = np.zeros((self.H, self.W), dtype=bool)
        else:
            hull = ConvexHull(points_known)  # hull.vertices => indices
            hull_pts = points_known[hull.vertices]  # (N_hull, 2)

            # We'll do a standard 'point in polygon' test for each pixel
            X_eval, Y_eval = np.meshgrid(x_eval, y_eval)
            coords_eval = np.column_stack((X_eval.ravel(), Y_eval.ravel()))

            inside = points_in_poly(coords_eval, hull_pts)
            hull_mask = inside.reshape((self.H, self.W))

        # 3) Combine spline + row-based extrapolation
        fitted_2d = np.copy(z_spline_2d)  # start with spline predictions
        outside_mask = ~hull_mask

        # Precompute the row-based boundary for each row: 
        row_boundaries = [None]*self.H
        for row_i in range(self.H):
            inside_cols = np.where(hull_mask[row_i, :])[0]
            if inside_cols.size == 0:
                row_boundaries[row_i] = None
            else:
                min_col = inside_cols.min()
                max_col = inside_cols.max()
                row_boundaries[row_i] = (min_col, max_col)

        # Handle each row's outside pixels
        for row_i in range(self.H):
            minmax = row_boundaries[row_i]
            if minmax is None:
                # Entire row is outside => fallback
                fitted_2d[row_i, :] = fallback_extrapolate_entire_row(
                    row_i, fitted_2d, row_boundaries
                )
                continue

            min_col, max_col = minmax
            z_left  = fitted_2d[row_i, min_col]
            z_right = fitted_2d[row_i, max_col]

            # For columns < min_col => do linear extrapolation from left boundary
            for col_j in range(0, min_col):
                if outside_mask[row_i, col_j]:
                    slope = z_left - fitted_2d[row_i, min_col - 1]
                    dist = (col_j - (min_col - 1))
                    fitted_2d[row_i, col_j] = (
                        fitted_2d[row_i, min_col - 1] + slope * dist
                    )

            # For columns > max_col => do linear extrapolation from right boundary
            for col_j in range(max_col + 1, self.W):
                if outside_mask[row_i, col_j]:
                    # Simple zero slope in this example
                    fitted_2d[row_i, col_j] = z_right

        # 4) Saturate the final fitted array to [0..255]
        if do_clip:
            np.clip(fitted_2d, 0, 255, out=fitted_2d)

        # 5) Compute the error map (abs difference where bin_img == 0, else 0)
        error_2d = np.zeros((self.H, self.W), dtype=np.float64)
        # Indices of known data in 2D
        known_rows, known_cols = np.where(bin_img == 0)
        
        # For known pixels, error = |fitted - original|
        error_2d[known_rows, known_cols] = np.abs(
            fitted_2d[known_rows, known_cols] - img_gray[known_rows, known_cols]
        )

        # Clip the error range to [0..255]
        np.clip(error_2d, 0, 255, out=error_2d)

        # Convert error to uint8
        error_8u = error_2d.astype(np.uint8)

        # Return the error image
        return fitted_2d, error_8u