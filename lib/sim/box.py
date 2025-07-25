import numpy as np

class Box:
    def __init__(self, anchor: tuple[float, float, float], dims: tuple[float, float, float]) -> None:
        self.anchor = np.array(anchor, dtype=float)
        self.dims = np.array(dims, dtype=float)
        self.corners = self._compute_corners()

    def _compute_corners(self) -> np.ndarray:
        x0, y0, z0 = self.anchor
        dx, dy, dz = self.dims
        return np.array([
            [x0,     y0,     z0],
            [x0+dx,  y0,     z0],
            [x0+dx,  y0+dy,  z0],
            [x0,     y0+dy,  z0],
            [x0,     y0,     z0+dz],
            [x0+dx,  y0,     z0+dz],
            [x0+dx,  y0+dy,  z0+dz],
            [x0,     y0+dy,  z0+dz],
        ], dtype=float)

    def get_all_points(self, num_points_per_dim=5) -> np.ndarray:
        xs = np.linspace(self.anchor[0], self.anchor[0] + self.dims[0], num_points_per_dim)
        ys = np.linspace(self.anchor[1], self.anchor[1] + self.dims[1], num_points_per_dim)
        zs = np.linspace(self.anchor[2], self.anchor[2] + self.dims[2], num_points_per_dim)
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
        pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
        return pts

    def get_random_points(self, N: int) -> np.ndarray:
        random_offsets = np.random.rand(N, 3) * self.dims
        points_xyz = self.anchor + random_offsets
        ids = np.arange(N).reshape(-1, 1)
        return np.hstack((ids, points_xyz))

    def get_random_on_nodes(self) -> np.ndarray:
        """
        Returns points on integer (x, y) grid covering the box base,
        with random float z between min and max z of the box.
        
        Returns
        -------
        points : ndarray of shape (N, 4)
            Each row is [id, x, y, z], where x and y are integers,
            and z is a random float within box height range.
        """
        x_min, y_min, z_min = self.anchor
        x_max, y_max, z_max = self.anchor + self.dims

        x_vals = np.arange(np.ceil(x_min), np.floor(x_max) + 1).astype(int)
        y_vals = np.arange(np.ceil(y_min), np.floor(y_max) + 1).astype(int)

        X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
        X = X.ravel()
        Y = Y.ravel()

        Z = np.random.uniform(z_min, z_max, size=X.shape)
        ids = np.arange(len(X)).reshape(-1, 1)
        points = np.stack([X, Y, Z], axis=-1)
        return np.hstack((ids, points))
