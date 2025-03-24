import cv2
import numpy as np
from scipy.optimize import curve_fit

def compute_residual_variance(img):
    """
    Computes the residual variance (mean squared error) between a 2D
    polynomial surface fitted on the unmasked pixels of 'img'
    and the actual unmasked pixel values.
    
    Args:
        img (np.ndarray): 2D array representing the masked depth image.
                          Pixel value 0 is treated as masked.
    
    Returns:
        float: The residual variance (MSE) where higher values indicate
               more deviation between the fitted surface and the actual data.
    """
    def poly2D(coords, a, b, c, d, e, f):
        # 2D polynomial: z = a + b*x + c*y + d*x^2 + e*x*y + f*y^2
        x, y = coords
        return a + b*x + c*y + d*x**2 + e*x*y + f*y**2

    # Create a mask for unmasked (non-zero) pixels.
    mask = (img != 0)
    
    # Get coordinates of unmasked pixels.
    x_coords, y_coords = np.where(mask)
    
    # Get actual depth values at unmasked pixels.
    z_vals = img[x_coords, y_coords]
    
    # Initial guess for the polynomial coefficients.
    initial_guess = np.zeros(6, dtype=np.float64)
    
    # Fit the polynomial surface on the unmasked pixel data.
    popt, _ = curve_fit(poly2D, (x_coords, y_coords), z_vals, p0=initial_guess)
    
    # Compute the fitted values on the unmasked pixels.
    fitted_vals = poly2D((x_coords, y_coords), *popt)
    
    # Compute residuals and then the residual variance (mean squared error).
    residuals = z_vals - fitted_vals
    residual_variance = np.mean(residuals**2)
    
    return residual_variance

def main():
    # Input path for the masked depth image.
    input_path = "masked_bt250_slope145.jpg"
    
    # Read the image in grayscale (8-bit) mode. Adjust flags if your image
    # has a different depth (e.g., 16-bit).
    img_in = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img_in is None:
        raise FileNotFoundError(f"Could not read image from {input_path}")
    
    # Convert image to float for computation.
    img_in = img_in.astype(np.float64)
    
    # Compute the residual variance.
    res_var = compute_residual_variance(img_in)
    print(f"Residual variance (MSE): {res_var:.4f}")

if __name__ == "__main__":
    main()
