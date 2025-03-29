from scipy.interpolate import RBFInterpolator
import cv2

def post_process_combined_thresh(combined_thresh, out_fname="fitted_combined_thresh.jpg"):
    """
    Replace white (255) areas in combined_thresh with a 'smooth' spline interpolation
    over the non-white cells. Uses RBF interpolation from SciPy as an example.

    :param combined_thresh: 2D numpy array (uint8) of shape (H, W).
    :param out_fname: filename to save the fitted version.
    :return: fitted_combined_thresh (2D numpy array).
    """
    h, w = combined_thresh.shape[:2]

    # Collect known points (x, y) and their intensities z
    known_xy = []
    known_z = []
    for y in range(h):
        for x in range(w):
            val = combined_thresh[y, x]
            if val < 255:  # Non-white => known data
                known_xy.append([x, y])
                known_z.append(float(val))

    if len(known_xy) == 0:
        # Corner case: everything is white -> just return the same image
        cv2.imwrite(out_fname, combined_thresh)
        return combined_thresh

    known_xy = np.array(known_xy, dtype=np.float64)
    known_z = np.array(known_z, dtype=np.float64)

    # Fit radial basis function. You can tweak smoothing or kernel as needed.
    # Smoothing > 0 => more smoothing, tries not to interpolate all data exactly.
    rbf = RBFInterpolator(known_xy, known_z, smoothing=5.0)

    # Create a copy so we don't overwrite combined_thresh
    fitted_combined_thresh = combined_thresh.copy()

    # For each white pixel, evaluate the RBF at (x, y)
    for y in range(h):
        for x in range(w):
            if fitted_combined_thresh[y, x] == 255:
                pred_z = rbf([[float(x), float(y)]])[0]
                # Clip/round
                pred_z = max(0, min(255, pred_z))
                fitted_combined_thresh[y, x] = int(round(pred_z))

    # Save fitted image
    cv2.imwrite(out_fname, fitted_combined_thresh)
    return fitted_combined_thresh

