# occupancy_grid3d.py – **adds offline PCD export & live‑vis flag**
"""Sparse 3‑D occupancy grid.

* **live_vis** (bool) – if *True* every ``update`` call refreshes the viewer; if *False* nothing is drawn during the run and you call ``save_pcd()`` once at the end.
* **save_pcd(path, min_count=1)** – writes the voxel centres (filtered by hit‑count) to an ASCII *.pcd* compatible with Open3D/PCL.

Usage
-----
```python
og = OccupancyGrid3D(cell_size=0.10, live_vis=False)  # no live plotting
...
for k, pts in enumerate(stream_of_pointclouds):
    og.update(pts, k)

og.save_pcd("occ_final.pcd", min_count=3)  # after the loop
```

Read & plot later:
```python
import open3d as o3d
pcd = o3d.io.read_point_cloud("occ_final.pcd")
o3d.visualization.draw_geometries([pcd])
```
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

__all__ = ["OccupancyGrid3D", "rpy_to_rot"]

# ---------------------------------------------------------------------------
# Maths helper
# ---------------------------------------------------------------------------
def rpy_to_rot(roll: float, pitch: float, yaw: float, *, degrees: bool = False) -> np.ndarray:
    if degrees:
        roll, pitch, yaw = map(np.radians, (roll, pitch, yaw))
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw),   np.sin(yaw)
    return np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [  -sp,            cp*sr,            cp*cr],
    ], dtype=np.float32)

# ---------------------------------------------------------------------------
# Occupancy grid
# ---------------------------------------------------------------------------
@dataclass
class OccupancyGrid3D:
    cell_size: float = 0.10                       # voxel edge [m]
    live_vis: bool = False                        # draw inside update?
    backend: str = "matplotlib"                       # 'open3d' | 'matplotlib'
    block_plot: bool = False                      # viewer modality (if live_vis)
    min_count_vis: int = 1                        # filter threshold for drawing
    vis_open3d = None  # persistent visualizer instance
    pcd_open3d = None

    _counts: dict[tuple[int,int,int], int] = field(default_factory=lambda: defaultdict(int), init=False)
    _last_seen: dict[tuple[int,int,int], int] = field(default_factory=dict, init=False)
    _vis: o3d.visualization.Visualizer | None = field(default=None, init=False)
    _pcd: o3d.geometry.PointCloud | None = field(default=None, init=False)
    _mpl_ax: Axes3D | None = field(default=None, init=False)

    # ---------------------------------------------------------------
    def update(self, pts_world: np.ndarray, frame_idx: int) -> None:
        if pts_world.ndim != 2 or pts_world.shape[1] != 3:
            raise ValueError("pts_world must be (N,3)")
        vox = np.unique(np.floor(pts_world / self.cell_size).astype(int), axis=0)
        for vx,vy,vz in map(tuple, vox):
            key = (vx,vy,vz)
            self._counts[key] += 1
            self._last_seen[key] = frame_idx

        if self.live_vis:
            self._draw_live()
        else:
            print(f"[OccupancyGrid3D] iter {frame_idx:>5} – grid now holds {len(self._counts)} voxels.")

    # ---------------------------------------------------------------
    # Offline export ------------------------------------------------
    # ---------------------------------------------------------------
    def save_pcd(self, path: str, *, min_count: int = 1) -> None:
        """Write occupied voxel centres to *path* in ASCII PCD (XYZRGB)."""
        coords, counts = self._extract(min_count)
        if coords.size == 0:
            print("[OccupancyGrid3D] nothing to save – no voxels >= min_count")
            return
        # colour map blue→red
        cmap_vals = (counts - counts.min()) / (counts.ptp() + 1e-8)
        colors = (plt.cm.coolwarm(cmap_vals)[:, :3] * 255).astype(np.uint8)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords)
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
        o3d.io.write_point_cloud(path, pcd, write_ascii=True)
        print(f"[OccupancyGrid3D] wrote {len(coords)} points -> {path}")

    # ---------------------------------------------------------------
    # Internal drawing helpers (unchanged except cmap)
    # ---------------------------------------------------------------
    def _draw_live(self):
        if self.backend == "open3d":
            self._draw_open3d(blocking=self.block_plot)
        else:
            self._draw_matplotlib(blocking=self.block_plot)

    def _draw_open3d(self, *, blocking: bool, min_count: int = 1, point_size: float = 5.0) -> None:
        coords, counts = self._extract(min_count)
        if coords.size == 0:
            print("[OccupancyGrid3D] No occupied voxels meet criteria – nothing to show.")
            return

        cmap_vals = (counts - counts.min()) / (counts.ptp() + 1e-8)
        colors = plt.cm.viridis(cmap_vals)[:, :3]
        if self.vis_open3d is None:
            # First time: create visualizer and add real point cloud
            self.vis_open3d = o3d.visualization.Visualizer()
            self.vis_open3d.create_window(window_name="Occupancy grid", width=960, height=720)
            render_option = self.vis_open3d.get_render_option()
            render_option.point_size = point_size

            self.pcd_open3d = o3d.geometry.PointCloud()
            self.pcd_open3d.points = o3d.utility.Vector3dVector(coords)
            self.pcd_open3d.colors = o3d.utility.Vector3dVector(colors)

            self.vis_open3d.add_geometry(self.pcd_open3d)
        else:
            # Update existing geometry
            self.pcd_open3d.points = o3d.utility.Vector3dVector(coords)
            self.pcd_open3d.colors = o3d.utility.Vector3dVector(colors)
            self.vis_open3d.update_geometry(self.pcd_open3d)

        self.vis_open3d.poll_events()
        self.vis_open3d.update_renderer()

    def _draw_matplotlib(self, *, blocking: bool):
        coords, counts = self._extract(self.min_count_vis)
        if coords.size == 0:
            return
        cmap = plt.cm.coolwarm((counts - counts.min()) / (counts.ptp() + 1e-8))[:, :3]
        if self._mpl_ax is None:
            fig = plt.figure()
            self._mpl_ax = fig.add_subplot(111, projection="3d")
            self._mpl_ax.set_xlabel("X [m]")
            self._mpl_ax.set_ylabel("Y [m]")
            self._mpl_ax.set_zlabel("Z [m]")
        self._mpl_ax.cla()
        self._mpl_ax.scatter(coords[:,0], coords[:,1], coords[:,2], c=cmap, s=1)
        self._mpl_ax.view_init(elev=30, azim=-60)
        if blocking:
            plt.show(block=True)
        else:
            plt.pause(0.01)
        
    def close_visualizer(self):
        if self.vis_open3d is not None:
            self.vis_open3d.destroy_window()
            self.vis_open3d = None
            self.pcd_open3d = None

    # ---------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------
    def _extract(self, min_count: int):
        coords, counts = [], []
        for (vx,vy,vz), c in self._counts.items():
            if c >= min_count:
                coords.append(((vx+0.5)*self.cell_size,
                               (vy+0.5)*self.cell_size,
                               (vz+0.5)*self.cell_size))
                counts.append(c)
        if coords:
            return np.asarray(coords, float), np.asarray(counts, int)
        return np.empty((0,3)), np.empty(0)

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    og = OccupancyGrid3D(cell_size=0.25, live_vis=False)
    for f in range(100):
        theta = np.linspace(0, 2*np.pi, 500)
        z = 0.1 * f
        r = 5
        pts = np.column_stack([r*np.cos(theta), r*np.sin(theta), z*np.ones_like(theta)])
        og.update(pts, f)
    og.save_pcd("demo_occ.pcd", min_count=1)
    # quick preview
    import open3d as o3d
    p = o3d.io.read_point_cloud("demo_occ.pcd")
    o3d.visualization.draw_geometries([p])
