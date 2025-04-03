import numpy as np
import cv2

from enum import Enum
from visualSplineFit import VisualSplineFit  # your existing spline class

class FitType(Enum):
    COMPOUND_FIT   = 1 # least err on 0's - average on 1's
    AVERAGE_FIT    = 2 # average on all
    SELECTIVE_FIT  = 3 # select the one with least 'curve + difference'
    MIX_FIT        = 4 # least err on 0's - least curvature on 1's --> finally fit a spline to make the result continuous

class VisSpFitsWrapper:
    def __init__(self, fitType=FitType.MIX_FIT, bias=8):
        """
        :param fitType: One of the FitType enums
        :param bias: Constant added at the end of the final fit, then clamped to [0..255]
        """
        # Example: two different spline configurations. Add more as needed.
        self.fitters = [
            VisualSplineFit(x_coeffs=1, degree_x=2, degree_y=2),
            VisualSplineFit(x_coeffs=2, degree_x=2, degree_y=2),
            # ...
        ]
        self.fitType = fitType
        self.bias = bias

    def fit(self, img_gray, img_bin):
        """
        1) Uses each spline fitter in self.fitters to produce:
           - fitted surfaces
           - error maps
           - curvature maps
        2) Combines them according to self.fitType
        3) Optionally does another pass of spline fitting on the combined result (MIX_FIT case)
        4) Adds self.bias and clips to [0..255]
        5) Returns the final fitted result as a uint8 image
        """

        H, W = img_gray.shape
        num_fitters = len(self.fitters)

        # We'll store surfaces, error maps, curvature maps from each fitter
        surfaces = []
        errors   = []
        curv_map = []

        # 1) Run each fitter. We assume each VisualSplineFit returns
        #    both the final fitted image and a local error map.
        #    If not, you can compute error yourself as: abs(surface - img_gray) where bin==0.
        #    For curvature, we can do: curvature[i] = abs(Laplacian of surface).
        for f in self.fitters:
            # We'll assume your VisualSplineFit has a method returning (surface, error_map).
            # If not, adapt accordingly. For curvature, we compute it now:
            surface, error_map = f.fit(img_gray, img_bin)

            # Compute local curvature: use Laplacian
            # We'll convert the surface to float32 first for Laplacian
            lap = cv2.Laplacian(surface.astype(np.float32), cv2.CV_32F, ksize=3)
            curvature = np.abs(lap)  # shape (H, W) float32

            surfaces.append(surface)
            errors.append(error_map.astype(np.float32))   # keep a float version
            curv_map.append(curvature)

        # We'll build our final result as float64
        final_fit = np.zeros((H, W), dtype=np.float64)

        # Known / unknown masks
        known_mask   = (img_bin == 0)
        unknown_mask = ~known_mask  # or (bin != 0)

        # 2) Different combine logic
        if self.fitType == FitType.COMPOUND_FIT:
            final_fit = self._do_compound_fit(img_bin, surfaces, errors)

        elif self.fitType == FitType.AVERAGE_FIT:
            final_fit = self._do_average_fit(surfaces)

        elif self.fitType == FitType.SELECTIVE_FIT:
            final_fit = self._do_selective_fit(surfaces, errors, img_bin)

        elif self.fitType == FitType.MIX_FIT:
            # MIX_FIT logic:
            # For each pixel:
            #  - if bin == 0 => pick the spline with min error
            #  - if bin == 1 => pick the spline with min curvature
            # Then do one more "smooth" spline fit on that piecewise result

            # 2A) Build a local "winner" surface pixel by pixel
            #     We'll do a loop for clarity, though you could do a fancy argmin across dimension.
            final_fit = np.zeros((H, W), dtype=np.float32)

            # convert surfaces -> shape (N,H,W)
            surf_stack = np.array(surfaces)     # (num_fitters, H, W)
            err_stack  = np.array(errors)       # (num_fitters, H, W)
            curv_stack = np.array(curv_map)     # (num_fitters, H, W)

            for row in range(H):
                for col in range(W):
                    if img_bin[row, col] == 0:
                        # Choose min error
                        best_idx = np.argmin(err_stack[:, row, col])
                    else:
                        # Choose min curvature
                        best_idx = np.argmin(curv_stack[:, row, col])
                    final_fit[row, col] = surf_stack[best_idx, row, col]

            # 2B) We want to ensure continuity, so we run a second spline pass
            #     We'll treat the entire final_fit as "known" => bin=0
            secondPassBin = np.zeros_like(img_bin)
            # We'll create a new VisualSplineFit with x_coeffs=2, deg=2
            vs_final = VisualSplineFit(x_coeffs=2, degree_x=2, degree_y=2)

            # Because the .fit(...) expects a gray image in [0..255], we clamp:
            np.clip(final_fit, 0, 255, out=final_fit)
            piecewise_8u = final_fit.astype(np.uint8)

            # Now fit it. We'll assume vs_final has a standard .fit(...) that returns
            # the final surface in float64. This depends on how your VisualSplineFit is defined!
            # (If it only returns an error map, adapt accordingly.)
            final_continuous, _ = vs_final.fit(piecewise_8u, secondPassBin)

            # Overwrite final_fit with the new continuous surface
            final_fit = final_continuous

        else:
            raise ValueError(f"Unknown fitType: {self.fitType}")

        # 3) Add bias & saturate
        final_fit += self.bias
        np.clip(final_fit, 0, 255, out=final_fit)

        return final_fit.astype(np.uint8)

    def _do_compound_fit(self, img_bin, surfaces, errors):
        """
        For known pixels (bin==0), pick the local min-error spline;
        for unknown pixels (bin!=0), average them.
        """
        H, W = img_bin.shape
        num_fitters = len(surfaces)

        final_fit = np.zeros((H, W), dtype=np.float64)

        # Stack surfaces, errors
        stacked_surfaces = np.array(surfaces)  # (N, H, W)
        stacked_errors = np.array(errors)       # (N, H, W)
        argmin_error = np.argmin(stacked_errors, axis=0)  # shape => (H, W)

        known_mask = (img_bin == 0)
        unknown_mask = ~known_mask

        # Known => pick the local min error
        for row in range(H):
            for col in range(W):
                if known_mask[row, col]:
                    best_idx = argmin_error[row, col]
                    final_fit[row, col] = stacked_surfaces[best_idx, row, col]

        # Unknown => average
        mean_surface = np.mean(stacked_surfaces, axis=0)
        final_fit[unknown_mask] = mean_surface[unknown_mask]

        return final_fit

    def _do_average_fit(self, surfaces):
        """
        AVERAGE_FIT: average all surfaces for all pixels.
        """
        stacked_surfaces = np.array(surfaces, dtype=np.float64)  # shape => (N, H, W)
        mean_surface = np.mean(stacked_surfaces, axis=0)         # (H, W)
        return mean_surface

    def _do_selective_fit(self, surfaces, errors, img_bin):
        """
        SELECTIVE_FIT: pick exactly one spline with the best global metric.
        For simplicity, let's do: metric = sum of error for known pixels.
        """
        known_mask = (img_bin == 0)

        best_index = None
        best_metric = 1e15

        for i, (surf, err) in enumerate(zip(surfaces, errors)):
            # sum of error in known region
            sum_error = np.sum(err[known_mask])

            if sum_error < best_metric:
                best_metric = sum_error
                best_index = i

        final_fit = surfaces[best_index].astype(np.float64)
        return final_fit
