import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def set_axes_equal(ax):
    """Set 3D plot axes to equal scale so that shapes are preserved."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    max_range = max(x_range, y_range, z_range)

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    ax.set_xlim3d([x_middle - max_range / 2, x_middle + max_range / 2])
    ax.set_ylim3d([y_middle - max_range / 2, y_middle + max_range / 2])
    ax.set_zlim3d([z_middle - max_range / 2, z_middle + max_range / 2])


def plot_3d_kps(pts_nwu, title="", ax=None, pts_color = 'red', 
    camera_trajectory=None, plt_pause=0, plt_show=False, cla=False, write_id=False, label=''):
    if ax is None:
        ax1 = get_new_ax3d()
    else:
        ax1 = ax

    if cla:
        ax1.cla()
    ax1.scatter(pts_nwu[:, 1], pts_nwu[:, 2], pts_nwu[:, 3], c=pts_color, 
        label=label)
    if write_id:
        for pt in pts_nwu:
            ax1.text(*pt[1:], f'{int(pt[0])}', color=pts_color)

    if not camera_trajectory is None:
        ax1.plot(camera_trajectory[:, 0], camera_trajectory[:, 1], camera_trajectory[:, 2], 
                'g-o', label='Camera Trajectory')

    ax1.set_title(title)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.legend()
    ax1.view_init(elev=30, azim=-45)
    set_axes_equal(ax1)

    if plt_pause != 0:
        plt.pause(plt_pause)
    if plt_show:
        plt.show()


def get_new_ax3d():
    fig1 = plt.figure()
    return fig1.add_subplot(111, projection='3d')


def rpy_to_rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Convert roll, pitch, yaw (in radians) into a 3×3 rotation matrix
    using a ZYX (yaw–pitch–roll) Euler sequence.  The resulting matrix
    maps a vector in the camera (body) frame into the world frame.
    """
    cr, sr = np.cos(roll),  np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw),   np.sin(yaw)

    # Rotation about X (roll)
    R_x = np.array([
        [1,  0,   0],
        [0, cr, -sr],
        [0, sr,  cr],
    ])
    # Rotation about Y (pitch)
    R_y = np.array([
        [ cp, 0, sp],
        [  0, 1,  0],
        [-sp, 0, cp],
    ])
    # Rotation about Z (yaw)
    R_z = np.array([
        [cy, -sy, 0],
        [sy,  cy, 0],
        [ 0,   0, 1],
    ])

    # Combined ZYX: first roll, then pitch, then yaw
    return R_z @ R_y @ R_x


def visualize_trajectory_and_points(
        pts_nwu: np.ndarray,
        trajectory: np.ndarray,
        arrow_length: float = 1.0
    ) -> None:
    """
    Draw 3D points, camera path, and at each pose:
      • a red arrow along the camera's forward (+Z) axis,
      • a red label with the pose index.

    Parameters
    ----------
    pts_nwu     : (M,4) array of [id, x, y, z] in NWU
    trajectory  : (N,6) array of [x, y, z, roll, pitch, yaw] (rad)
    arrow_length: length of the forward‐axis arrows
    """
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')
    set_axes_equal(ax)

    # — plot the 3D keypoints
    xs, ys, zs = pts_nwu[:,1], pts_nwu[:,2], pts_nwu[:,3]
    ax.scatter(xs, ys, zs, marker='o', label='3D points')
    for pid, x, y, z in pts_nwu:
        ax.text(x, y, z, f'N{int(pid)}', color='k')

    # — plot the camera trajectory
    cam_xyz = trajectory[:, :3]
    ax.plot(cam_xyz[:,0], cam_xyz[:,1], cam_xyz[:,2],
            '-o', label='Camera path')

    # — add forward‐axis arrows and pose indices
    for idx, (x, y, z, roll, pitch, yaw) in enumerate(trajectory):
        # # 1) build body→world rotation via ZYX yaw–pitch–roll
        # R_c2w = rpy_to_rot(roll, pitch, yaw)

        # # 2) camera's forward axis is +Z in camera frame
        # forward_cam   = np.array([1.0, 0.0, 0.0])
        # forward_world = R_c2w @ forward_cam

        # ax.quiver(
        #     x, y, z,
        #     forward_world[0], forward_world[1], forward_world[2],
        #     length=arrow_length,
        #     normalize=True,
        #     color='r'
        # )

        # 3) label the pose index in red
        ax.text(x, y, z, str(idx),
                color='r',
                fontsize=10,
                horizontalalignment='left',
                verticalalignment='bottom')

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title('Keypoints, Camera Path & Orientations')
    ax.legend()
    plt.tight_layout()
    plt.show()