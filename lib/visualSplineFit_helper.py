import numpy as np

# -------------------------------------------------------------------------
# HELPER: Test if a set of points is inside a convex polygon (x,y).
def points_in_poly(points, hull_pts):
    """
    points: shape (N, 2)
    hull_pts: shape (M, 2) in some order (the hull's vertices).
    Returns: boolean array of length N, True if inside
    """
    if len(hull_pts) < 3:
        return np.zeros(len(points), dtype=bool)
    return _ray_casting(points, hull_pts)


def _ray_casting(points, polygon):
    """ Minimal 'ray casting' approach to test inside vs outside. """
    x = points[:, 0]
    y = points[:, 1]
    inside = np.zeros(len(points), dtype=bool)

    xs = polygon[:, 0]
    ys = polygon[:, 1]
    n = len(polygon)

    for ipt, (px, py) in enumerate(points):
        count_intersect = 0
        for i in range(n):
            x1, y1 = xs[i], ys[i]
            x2, y2 = xs[(i + 1) % n], ys[(i + 1) % n]
            if y1 > y2:
                x1, x2 = x2, x1
                y1, y2 = y2, y1
            if (py > y1) and (py <= y2):
                if y2 == y1:
                    continue
                intersect_x = x1 + (py - y1)*(x2 - x1)/(y2 - y1)
                if intersect_x >= px:
                    count_intersect += 1
        if (count_intersect % 2) == 1:
            inside[ipt] = True
    return inside


def fallback_extrapolate_entire_row(row_i, fitted_2d, row_boundaries):
    """
    If the entire row row_i is outside the hull, fallback approach:
    - Copy from the nearest row above/below that intersects the hull
      or fill zeros if none found.
    """
    H, W = fitted_2d.shape
    above, below = None, None
    # search up
    for r in range(row_i - 1, -1, -1):
        if row_boundaries[r] is not None:
            above = r
            break
    # search down
    for r in range(row_i + 1, H):
        if row_boundaries[r] is not None:
            below = r
            break

    if above is None and below is None:
        # no rows in the entire image intersect => fallback: fill zeros
        return np.zeros(W, dtype=np.float64)

    # If we only found 'above':
    if below is None:
        return fitted_2d[above, :].copy()

    # If we only found 'below':
    if above is None:
        return fitted_2d[below, :].copy()

    # If we have both, do vertical interpolation
    dist_above = row_i - above
    dist_below = below - row_i
    total = dist_above + dist_below
    w_above = dist_below / total
    w_below = dist_above / total
    return w_above * fitted_2d[above, :] + w_below * fitted_2d[below, :]
