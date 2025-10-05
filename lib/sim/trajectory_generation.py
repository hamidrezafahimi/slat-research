import math
import numpy as np
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/..")
from kinematics.transformations import transform_kps_nwu2camera    # already in your repo
from scipy.interpolate import CubicSpline   # SciPy ≥1.4
from sim.camera import SimpleCamera
from sim.box import Box

def get_cam_pose(cam, box, *, D: float | None = None,
                 max_trials: int = 10_000,
                 rng: int | np.random.Generator | None = None):
    """
    Random camera pose (NWU coords) such that the whole `box`
    is visible to `cam`, with RPY defined as:

        yaw   ψ — rot about +Zᴺᴱᴰ  (down)   Xᴮ = Rz(ψ) · …
        pitch θ — rot about +Yᴮ     (right)
        roll  ϕ — rot about +Xᴮ     (front)

    Returns
    -------
    cam_pose : (3,) float  (NWU)
    roll, pitch, yaw : floats [rad]   following the above convention
    """
    # ─────────────────── preliminaries ───────────────────
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    W, H = cam.image_shape
    hfov = getattr(cam, "hfov_deg",
                   math.degrees(2*math.atan((W/2)/cam.fx)))
    vfov = getattr(cam, "vfov_deg",
                   math.degrees(2*math.atan((H/2)/cam.fy)))
    half_h, half_v = map(lambda a: math.radians(a)/2, (hfov, vfov))

    anchor = np.asarray(box.anchor, float)
    dims   = np.asarray(box.dims,   float)
    centre = anchor + 0.5*dims
    half   = 0.5*dims
    radius = np.linalg.norm(half)

    d_min = radius / max(math.tan(half_h), math.tan(half_v)) * 1.05
    d_max = D if D is not None else d_min * 5.0
    if d_max < d_min - 1e-9:
        raise ValueError(f"D={D} < d_min={d_min:.3f}; box cannot fit.")

    # eight corners (id, x, y, z) in NWU
    offsets = np.array([[sx, sy, sz]
                        for sx in (-half[0], half[0])
                        for sy in (-half[1], half[1])
                        for sz in (-half[2], half[2])])
    verts_nwu = np.hstack([np.arange(8)[:, None], centre + offsets])

    # ─────────────────── main loop ───────────────────
    xy_span = d_max * 2
    z_hi    = centre[2] + d_max

    for _ in range(max_trials):
        # 1. random position (NWU)  ───────────────
        x = rng.uniform(centre[0] - d_max, centre[0] + d_max)
        y = rng.uniform(centre[1] - d_max, centre[1] + d_max)
        z = rng.uniform(0.0,          z_hi)          # z ≥ 0 (NWU up)
        cam_pose = np.array([x, y, z], float)

        dist = np.linalg.norm(centre - cam_pose)
        if not (d_min <= dist <= d_max):
            continue

        # 2. vector to centre and convert to NED ───
        v_nwu = centre - cam_pose
        v_ned = np.array([ v_nwu[0],      # X north
                          -v_nwu[1],      # Y east
                          -v_nwu[2]])     # Z down

        # 3. Euler angles (NED → body)  ───────────
        yaw   = math.atan2(v_ned[1], v_ned[0])               # ψ
        horiz = math.hypot(v_ned[0], v_ned[1])
        pitch = math.atan2(-v_ned[2], horiz)                 # θ  (nose-up +)
        roll  = rng.uniform(-math.pi/6, math.pi/6)           # ϕ

        # 4. visibility check  ─────────────────────
        try:
            pts_cam = transform_kps_nwu2camera(
                verts_nwu,
                cam_pose=cam_pose,
                roll=roll, pitch=pitch, yaw=yaw
            )
            if np.any(pts_cam[:, 3] <= 0):
                continue
            uv = cam.project(pts_cam)
        except ValueError:
            continue

        if np.all((0 <= uv[:, 1]) & (uv[:, 1] < W) &
                  (0 <= uv[:, 2]) & (uv[:, 2] < H)):
            return cam_pose, roll, pitch, yaw

    # ─────────────────── failure ───────────────────
    raise RuntimeError(
        f"No valid pose found after {max_trials} trials "
        f"(d_min={d_min:.3f}, d_max={d_max:.3f})."
    )


# ──────────────────────────────────────────────────────────────────────────
# Trajectory generation utilities
# ──────────────────────────────────────────────────────────────────────────
def _greedy_nearest_order(poses_xyz: np.ndarray) -> list[int]:
    """
    Return an index ordering that starts at element 0 and always continues to
    the *nearest yet-unvisited* point.  Classic “nearest-neighbour” TSP
    heuristic – O(N²) is fine for the small N we have here.
    """
    N = len(poses_xyz)
    order   = [0]
    remain  = set(range(1, N))
    while remain:
        last    = poses_xyz[order[-1]]
        nxt_idx = min(remain, key=lambda i: np.linalg.norm(poses_xyz[i] - last))
        order.append(nxt_idx)
        remain.remove(nxt_idx)
    return order

def get_cam_trajectory(
        cam: "SimpleCamera",
        box: "Box",
        *,
        N: int,
        D: float | None = None,
        step_size: float | None = None,
        max_trials: int = 2_000,
        log_every: int | None = None,
        smooth: bool = False,
        smooth_upsample: int | None = None,
    ) -> np.ndarray:
    """
    Generate **N** camera poses that all keep the whole ``box`` inside the
    field-of-view, *never* go **closer** to the box than the very first pose,
    and move at most ``step_size`` between successive samples.  The final list
    is re-ordered with a greedy nearest-neighbour pass to form a visually
    smooth trajectory.

    Returns
    -------
    ndarray,  shape (N, 6)   – ordered as columns  
    ``[x, y, z, roll, pitch, yaw]``
    """
    rng = np.random.default_rng()
    # ── first pose (acts as anchor for distance constraint) ────────────────
    cam_pose, roll, pitch, yaw = get_cam_pose(   # uses the helper we already have
        cam, box, D=D, max_trials=max_trials
    )  # :contentReference[oaicite:0]{index=0}
    centre = np.asarray(box.anchor) + 0.5 * np.asarray(box.dims)
    base_dist = np.linalg.norm(cam_pose - centre)

    poses = [np.hstack([cam_pose, [roll, pitch, yaw]])]

    # Sensible default if the caller did not specify one
    if step_size is None:
        # quarter of the maximum allowed radius, but never smaller than a box edge
        step_size = max(np.max(box.dims), (D or base_dist * 1.5) / 4.0)

    # ── sample until we have N valid poses ────────────────────────────────
    attempts = 0
    while len(poses) < N and attempts < max_trials * N:
        attempts += 1
        try:
            p_xyz, r, pth, yw = get_cam_pose(cam, box, D=D, max_trials=max_trials)
        except RuntimeError:       # extremely unlucky – just resample
            continue

        # distance to box centre must not shrink compared to the first pose
        if np.linalg.norm(p_xyz - centre) < base_dist - 1e-6:
            continue

        # must stay within step_size of the *last accepted* pose
        if np.linalg.norm(p_xyz - poses[-1][:3]) > step_size:
            continue

        poses.append(np.hstack([p_xyz, [r, pth, yw]]))
        if log_every and len(poses) % log_every == 0:
            print(f"  ✓  {len(poses):>3}/{N} poses accepted")

    if len(poses) < N:
        raise RuntimeError(
            f"Could only generate {len(poses)} poses after {attempts} attempts "
            f"(need {N}).  Try increasing ``step_size`` or ``D``."
        )

    # ── connect them with a greedy nearest-neighbour pass ─────────────────
    order = _greedy_nearest_order(np.asarray([p[:3] for p in poses]))
    trajectory = np.vstack([poses[i] for i in order])

    # ------------------------------------------------------------------
    #  Smooth / up-sample the path if requested
    # ------------------------------------------------------------------
    if smooth:
        # how densely to resample?  roughly `upsample_factor` points between
        # every pair of key-frames
        upsample_factor = smooth_upsample if smooth_upsample else 10
        t_key   = np.arange(len(trajectory))            # 0,1,2,…,N-1
        t_fine  = np.linspace(0, len(trajectory) - 1,
                              (len(trajectory) - 1) * upsample_factor + 1)

        # spline-fit *positions* (C2-continuous, natural cubic spline)
        cs_x = CubicSpline(t_key, trajectory[:, 0], bc_type="natural")
        cs_y = CubicSpline(t_key, trajectory[:, 1], bc_type="natural")
        cs_z = CubicSpline(t_key, trajectory[:, 2], bc_type="natural")
        x_f  = cs_x(t_fine)
        y_f  = cs_y(t_fine)
        z_f  = cs_z(t_fine)

        # simple linear interpolation for each Euler angle
        # (unwrap yaw to avoid 2π jumps)
        yaw_unwrapped = np.unwrap(trajectory[:, 5])
        r_f = np.interp(t_fine, t_key, trajectory[:, 3])
        p_f = np.interp(t_fine, t_key, trajectory[:, 4])
        y_fa = np.interp(t_fine, t_key, yaw_unwrapped)

        trajectory = np.column_stack([x_f, y_f, z_f, r_f, p_f, y_fa])
    # ------------------------------------------------------------------
    return trajectory