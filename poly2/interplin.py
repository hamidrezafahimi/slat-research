import cv2
import numpy as np

class VisualLinearInterpolate:
    def __init__(self, x_coeffs=4, degree_x=3, degree_y=3):
        """
        Matches the same constructor signature as VisualSplineFit
        """
        self.x_coeffs = x_coeffs
        self.degree_x = degree_x
        self.degree_y = degree_y
        self.params_init = False

    def initParams(self, img_shape):
        """
        Matches the same signature/pattern as VisualSplineFit.initParams
        """
        self.H, self.W = img_shape
        self.params_init = True

    def fit(self, img_gray, bin_img):
        """
        Interpolate (fill) img_gray where bin_img == 255,
        returning a float64 array of shape (H, W).
        """
        if not self.params_init:
            self.initParams(img_gray.shape)

        # Create a mask of pixels to fill (1-channel, 8-bit)
        mask = np.where(bin_img == 255, 255, 0).astype(np.uint8)

        # Inpaint with a chosen radius and method. (TELEA or NS)
        # This “fills” the masked region by interpolating from surrounding pixels.
        inpainted = cv2.inpaint(img_gray, mask, 3, cv2.INPAINT_TELEA)

        # Convert to float64 for consistency with spline’s output
        return inpainted.astype(np.float64)

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Suppose you already have VisualSplineFit in your codebase:
# from spline_fit import VisualSplineFit
# from your_code import VisualLinearInterpolate  # The new class above

def main():
    # 1) Load input images
    gray_path = "depth_image_bgsMasked.jpg"
    bin_path =  "depth_image_bgsBin.jpg"

    img_gray = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)
    img_bin  = cv2.imread(bin_path, cv2.IMREAD_GRAYSCALE)

    if img_gray is None or img_bin is None:
        print("Could not load images.")
        return

    # # 2) Use the spline-based fitter (ignores pixels where bin_img != 0)
    # spline_fit = VisualSplineFit(x_coeffs=6, degree_x=3, degree_y=3)
    # fitted_spline = spline_fit.fit(img_gray, img_bin)

    # 3) Use the linear interpolation class (fills where bin_img == 255)
    linear_interp = VisualLinearInterpolate(x_coeffs=6, degree_x=3, degree_y=3)
    fitted_linear = linear_interp.fit(img_gray, img_bin)

    # 4) Display or save the results
    # Convert to 8-bit for showing in standard image formats
    # fitted_spline_8u = np.clip(fitted_spline, 0, 255).astype(np.uint8)
    fitted_linear_8u = np.clip(fitted_linear, 0, 255).astype(np.uint8)

    # cv2.imwrite("spline_fitted.png", fitted_spline_8u)
    cv2.imwrite("linear_interpolated.png", fitted_linear_8u)
    cv2.imshow("sdf", fitted_linear_8u)
    cv2.waitKey()

    # Show with matplotlib (optional)
    # plt.subplot(1,3,1); plt.title("Original Gray");  plt.imshow(img_gray, cmap='gray')
    # # plt.subplot(1,3,2); plt.title("Spline Fitted");  plt.imshow(fitted_spline_8u, cmap='gray')
    # plt.subplot(1,3,3); plt.title("Linear Filled");   plt.imshow(fitted_linear_8u, cmap='gray')
    plt.show()

if __name__ == "__main__":
    main()
