# occupancy_grid3d.py – **re‑vamped**
"""Sparse 3‑D occupancy grid with *Mapper3D‑style* live visualisation.

Major changes requested by the user (Apr 26 2025)
-------------------------------------------------
* **Visualisation lives inside ``update``** – no manual calls needed.
* **Blue → Red colour ramp** (``coolwarm``) where *blue = few hits*, *red = many*.
* **Visibility switch**: pass ``live_vis=False`` to the constructor and the grid
  just logs each iteration; call ``show_final()`` once at the end to see the
  cumulative map.
* Interface closely mirrors *Mapper3D*:

    ```python
    og = OccupancyGrid3D(cell_size=0.10, backend='open3d', block_plot=False,
                         live_vis=True, min_count_vis=1)
    ```

    * ``backend`` – ``'open3d'`` (default) or ``'matplotlib'``
    * ``block_plot`` – behaviour of the viewer window (non‑blocking by default)
    * ``live_vis`` – toggle per‑frame visualisation
    * ``min_count_vis`` – voxels with fewer hits are hidden
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa – register 3‑D proj

__all__ = ["OccupancyGrid3D", "rpy_to_rot"]

# ---------------------------------------------------------------------------
# Maths helpers – unchanged
# ---------------------------------------------------------------------------

def rpy_to_rot(roll: float, pitch: float, yaw: float, /, *, degrees: bool = False) -> np.ndarray:
    """Return body‑to‑world rotation **R = Rz(yaw) @ Ry(pitch) @ Rx(roll)**."""
    if degrees:
        roll, pitch, yaw = map(np.radians, (roll, pitch, yaw))
    cr, sr = np.cos(roll),  np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw),   np.sin(yaw)
    return np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [  -sp,             cp*sr,             cp*cr],
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Core occupancy‑grid class
# ---------------------------------------------------------------------------
@dataclass
class OccupancyGrid3D:
    cell_size: float = 0.10                     # voxel edge in metres
    backend: str = "open3d"                     # 'open3d' | 'matplotlib'
    block_plot: bool = False                    # mimic Mapper3D semantics
    live_vis: bool = False                       # per‑update visualisation?
    min_count_vis: int = 1                      # hide weak voxels

    # internal state -----------------------------------------------------------------
    _counts: dict[tuple[int,int,int], int] = field(default_factory=lambda: defaultdict(int), init=False)
    _last_seen: dict[tuple[int,int,int], int] = field(default_factory=dict, init=False)
    _vis: o3d.visualization.Visualizer | None = field(default=None, init=False)
    _pcd: o3d.geometry.PointCloud | None = field(default=None, init=False)
    _mpl_ax: Axes3D | None = field(default=None, init=False)

    # ---------------------------------------------------------------------
    # Public API -----------------------------------------------------------
    # ---------------------------------------------------------------------
    def update(self, pts_world: np.ndarray, frame_idx: int) -> None:
        """Insert *pts_world* (**N×3**) and optionally refresh the viewer."""
        if pts_world.ndim != 2 or pts_world.shape[1] != 3:
            raise ValueError("pts_world must be (N,3)")
        voxels = np.unique(np.floor(pts_world / self.cell_size).astype(int), axis=0)
        for vx, vy, vz in map(tuple, voxels):
            key = (vx, vy, vz)
            self._counts[key] += 1
            self._last_seen[key] = frame_idx

        if self.live_vis:
            self._draw_live()
        else:
            print(f"[OccupancyGrid3D] iter {frame_idx:>5} – grid now holds {len(self._counts)} voxels.")

    # call once at the very end when live_vis=False ----------------------------------
    def show_final(self):
        if self.backend == "open3d":
            self._draw_open3d(blocking=True)
        else:
            self._draw_matplotlib(blocking=True)

    # ---------------------------------------------------------------------
    # Internal visualisation helpers --------------------------------------
    # ---------------------------------------------------------------------
    def _draw_live(self):
        if self.backend == "open3d":
            self._draw_open3d(blocking=self.block_plot)
        else:
            self._draw_matplotlib(blocking=self.block_plot)

    def _draw_open3d(self, *, blocking: bool):
        coords, counts = self._extract(self.min_count_vis)
        if coords.size == 0:
            return
        # blue→red ramp (coolwarm)
        cmap_vals = (counts - counts.min()) / (counts.ptp() + 1e-8)
        colors = plt.cm.coolwarm(cmap_vals)[:, :3]

        if self._vis is None:
            self._vis = o3d.visualization.Visualizer()
            self._vis.create_window("Occupancy grid", 960, 720, visible=True)
            self._pcd = o3d.geometry.PointCloud()
            self._vis.add_geometry(self._pcd)
            self._vis.get_render_option().point_size = 5.0

        self._pcd.points = o3d.utility.Vector3dVector(coords)
        self._pcd.colors = o3d.utility.Vector3dVector(colors)
        self._vis.update_geometry(self._pcd)
        self._vis.poll_events()
        self._vis.update_renderer()
        if blocking:
            self._vis.run()
            self._vis.destroy_window()
            self._vis = self._pcd = None

    def _draw_matplotlib(self, *, blocking: bool):
        coords, counts = self._extract(self.min_count_vis)
        if coords.size == 0:
            return
        cmap_vals = (counts - counts.min()) / (counts.ptp() + 1e-8)
        if self._mpl_ax is None:
            fig = plt.figure()
            self._mpl_ax = fig.add_subplot(111, projection="3d")
            self._mpl_ax.set_xlabel("X [m]")
            self._mpl_ax.set_ylabel("Y [m]")
            self._mpl_ax.set_zlabel("Z [m]")
        self._mpl_ax.cla()
        self._mpl_ax.scatter(coords[:,0], coords[:,1], coords[:,2],
                             c=plt.cm.coolwarm(cmap_vals), s=1)
        self._mpl_ax.view_init(elev=30, azim=-60)
        if blocking:
            plt.show(block=True)
        else:
            plt.pause(0.01)

    # ---------------------------------------------------------------------
    # Data extraction utilities -------------------------------------------
    # ---------------------------------------------------------------------
    def _extract(self, min_count: int):
        coords, counts = [], []
        for (vx,vy,vz), c in self._counts.items():
            if c >= min_count:
                coords.append((vx+0.5, vy+0.5, vz+0.5))
                counts.append(c)
        if coords:
            return (np.array(coords, dtype=float) * self.cell_size,
                    np.array(counts, dtype=int))
        return np.empty((0,3)), np.empty(0)

    # ---------------------------------------------------------------------
    # Optional dense export -----------------------------------------------
    # ---------------------------------------------------------------------
    def to_dense(self):
        if not self._counts:
            return np.zeros((0,0,0), dtype=np.uint16), np.zeros((0,0,0), dtype=np.int32)
        keys = np.array(list(self._counts))
        mins, maxs = keys.min(axis=0), keys.max(axis=0)
        shape = (maxs - mins + 1)
        counts = np.zeros(shape, dtype=np.uint16)
        last   = -np.ones(shape, dtype=np.int32)
        for (vx,vy,vz), c in self._counts.items():
            ix,iy,iz = (vx-mins[0], vy-mins[1], vz-mins[2])
            counts[ix,iy,iz] = c
            last  [ix,iy,iz] = self._last_seen[(vx,vy,vz)]
        return counts, last


# ---------------------------------------------------------------------------
# Quick CLI smoke‑test ------------------------------------------------------
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    og = OccupancyGrid3D(cell_size=0.25, backend="matplotlib", live_vis=False)
    for f in range(50):
        t = np.linspace(0, 2*np.pi, 400)
        z = 0.2*f
        r = 5
        pts = np.column_stack([r*np.cos(t), r*np.sin(t), z+np.zeros_like(t)])
        og.update(pts, f)
    og.show_final()
