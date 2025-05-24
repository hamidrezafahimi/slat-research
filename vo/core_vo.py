# ----------------------------------------------------------------------
#  core_vo.py   (Python ≥ 3.9, OpenCV ≥ 4.5, SciPy ≥ 1.10)
# ----------------------------------------------------------------------

from __future__ import annotations
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as Rscipy
from scipy.optimize import least_squares
from .base_mono_vo import BaseMonoVO

# ========  helper conversions  ========================================

def rmat_from_rpy(roll: float, pitch: float, yaw: float,
                  degrees: bool = True) -> np.ndarray:
    """NED → body (ZYX)   —  returns a 3×3 rotation matrix."""
    seq = "ZYX"            # yaw, pitch, roll (aircraft convention)
    rot = Rscipy.from_euler(seq, [yaw, pitch, roll], degrees=degrees)
    return rot.as_matrix()            # world→body


def rod_to_rmat(r: np.ndarray) -> np.ndarray:
    """Rodrigues(3,) → R(3,3)."""
    return cv2.Rodrigues(r.reshape(3, 1))[0]


def rmat_to_rod(R: np.ndarray) -> np.ndarray:
    """R(3,3) → Rodrigues(3,)."""
    return cv2.Rodrigues(R)[0].ravel()


# =====================================================================
#  1.  VIO EXTENSION  (IMU priors + occasional range factors)
# =====================================================================

class VIO_MonoVO(BaseMonoVO):
    """
    Adds:
      • orientation prior (roll-pitch-yaw in NED → body)
      • optional key-point range constraints  [[id, dist], …]
    """

    def __init__(self,
                 K: np.ndarray,
                 ba_window: int = 30,
                 ori_sigma_deg: float = 1.0,
                 range_sigma: float = 0.10):
        super().__init__(K, ba_window)
        self.ori_sigma = np.deg2rad(ori_sigma_deg)   # rad
        self.range_sigma = range_sigma               # metres (internal)

        # cache the last orientation prior (for bootstrap)
        self._imu_R_wb: list[np.ndarray] = []

        # store sparse range hints
        self._range_hints: dict[int, float] = {}     # id → distance (m)

    # ------------------------------------------------------------------
    # override bootstrap & PnP to use IMU orientation when provided ----
    # ------------------------------------------------------------------

    def _bootstrap_with_first_two_frames(self, ids, uv, rpy):
        """
        First TWO frames – keep IMU orientation fixed, solve only t.
        """
        if len(self.poses) == 0:
            # store IMU R (if given) else eye — for frame-0
            R_wb0 = rmat_from_rpy(*rpy) if rpy is not None else np.eye(3)
            T0 = np.eye(4);  T0[:3, :3] = R_wb0
            self.poses.append(T0)
            self.frame0_ids, self.frame0_uv = ids, uv
            self._imu_R_wb.append(R_wb0)
            return

        # ---------- second frame ----------
        if rpy is None:
            raise ValueError("VIO needs RPY for every frame")

        R_wb1 = rmat_from_rpy(*rpy)      # store for later BA prior
        self._imu_R_wb.append(R_wb1)

        ids0, uv0 = self.frame0_ids, self.frame0_uv
        mask = np.isin(ids0, ids)
        ids0c, uv0c = ids0[mask], uv0[mask]
        idx1        = np.nonzero(np.isin(ids, ids0c))[0]
        uv1c        = uv[idx1]

        # --- essential matrix (just like BaseMonoVO) -----------------
        E, inl = cv2.findEssentialMat(
            uv0c, uv1c, self.K,
            method=cv2.RANSAC, prob=0.999, threshold=1.0)
        inl = inl.ravel().astype(bool)
        _, R_rel, t_hat, _ = cv2.recoverPose(E,
                                            uv0c[inl], uv1c[inl], self.K)
        t = t_hat.ravel() / np.linalg.norm(t_hat)

        # Pose-0 already stored (world frame)
        T1 = np.eye(4)
        T1[:3, :3] = R_rel             # relative to cam-0, fine
        T1[:3, 3]  = t
        self.poses.append(T1)
        self.initialised = True

        # --- triangulate seed landmarks ---------------------------------
        common_ids = ids0c[inl]
        self._triangulate_new(common_ids, uv0c[inl],
                            common_ids, uv1c[inl],
                            T0=self.poses[0], T1=T1)

    # -------------------------------------------------------------
    def _track_and_pose(self, ids, uv, f_idx, rpy):
        """
        For frames ≥2: pose = known IMU R ⊕ solvePnP(translation).
        """
        R_wb = rmat_from_rpy(*rpy)
        self._imu_R_wb.append(R_wb)

        obj, img = [], []
        for pid, xy in zip(ids, uv):
            if pid in self.landmarks:
                obj.append(self.landmarks[pid])
                img.append(xy)
        if len(obj) < 4:
            # Fallback: treat this as a new baseline initialisation
            self._bootstrap_with_first_two_frames(ids, uv, rpy)
            raise RuntimeError("Not enough 3-D points for PnP")

        obj = np.asarray(obj, np.float32)
        img = np.asarray(img, np.float32)

        retval, rvec, tvec = cv2.solvePnP(
            obj, img, self.K, None,
            rvec=rmat_to_rod(R_wb), useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_EPNP
        )
        if not retval:
            raise RuntimeError("solvePnP failed with IMU guess")

        T = np.eye(4);  T[:3, :3] = R_wb;  T[:3, 3] = tvec.ravel()
        self.poses.append(T)

        # triangulate new points with previous frame (same as Base class)
        # ---- triangulate *only* new points ----------------------------
        prev_idx = f_idx - 1
        prev_ids, prev_uv = [], []
        for pid, xy in zip(ids, uv):
            if pid not in self.landmarks:
                for pidx, puv in self.obs.get(pid, []):
                    if pidx == prev_idx:
                        prev_ids.append(pid)
                        prev_uv.append(puv)
        if prev_ids:
            self._triangulate_new(
                np.array(prev_ids), np.array(prev_uv),
                np.array(prev_ids),
                np.array([uv[np.where(ids == pid)[0][0]]
                          for pid in prev_ids]),
                T0=self.poses[prev_idx], T1=T
            )

    # -------------------------------------------------------------
    # override BA: add orientation & range residuals  --------------
    # -------------------------------------------------------------
    def _bundle_adjust(self):
        """
        Extends the normal BA:
          • adds an orientation prior  (R_imu × R_est ≈ I)
          • adds range residual  (‖Xc‖ − r_meas)
        """
        # ===== gather data exactly like BaseMonoVO =================
        last = len(self.poses) - 1
        first = max(0, last - self.Nba + 1)
        frame_ids = list(range(first, last + 1))
        f2idx = {f: i for i, f in enumerate(frame_ids)}

        pose_vec = []
        for f in frame_ids:
            T = self.poses[f]
            pose_vec.extend(rmat_to_rod(T[:3, :3]))
            pose_vec.extend(T[:3, 3])
        pose_vec = np.array(pose_vec)

        lm_ids = [pid for pid, obs in self.obs.items()
                  if any(first <= f <= last for f, _ in obs)]
        lm_vec = np.concatenate([self.landmarks[pid] for pid in lm_ids])

        observations = []
        for j, pid in enumerate(lm_ids):
            for f, uv in self.obs[pid]:
                if first <= f <= last:
                    observations.append((f2idx[f], j, uv))

        def unpack(vec):
            p, l = vec[:pose_vec.size], vec[pose_vec.size:]
            return p, l.reshape((-1, 3))

        # ===== residuals ==========================================
        def residuals(x):
            pvec, lms = unpack(x)
            res = []

            # poses
            Rm, tm = [], []
            for i in range(len(frame_ids)):
                r = pvec[6*i:6*i+3]
                t = pvec[6*i+3:6*i+6]
                Rm.append(rod_to_rmat(r))
                tm.append(t)

            # 1) pixel reprojection
            for fidx, j, uv in observations:
                X = lms[j]
                x_cam = Rm[fidx] @ X + tm[fidx]
                u_proj = self.K[0, 0] * x_cam[0] / x_cam[2] + self.K[0, 2]
                v_proj = self.K[1, 1] * x_cam[1] / x_cam[2] + self.K[1, 2]
                res.extend([u_proj - uv[0], v_proj - uv[1]])

            # 2) orientation prior  (only for frames that have IMU)
            for i, f in enumerate(frame_ids):
                R_imu = self._imu_R_wb[f]
                dR = R_imu @ Rm[i].T
                rot_err = rmat_to_rod(dR)
                res.extend(rot_err / self.ori_sigma)

            # 3) range hints  (‖Xc‖ − r_meas)
            for pid, d_meas in self._range_hints.items():
                if pid in lm_ids:
                    j = lm_ids.index(pid)
                    # use the most recent pose to express X in camera frame
                    Xc = Rm[-1] @ lms[j] + tm[-1]
                    res.append((np.linalg.norm(Xc) - d_meas) / self.range_sigma)

            return np.array(res)

        x0 = np.concatenate([pose_vec, lm_vec])
        if not np.isfinite(x0).all():
            print("returning BA due to nan/infs")
            return          # skip this BA iteration
        least_squares(residuals, x0, verbose=0, max_nfev=15)

    # -------------------------------------------------------------
    # PUBLIC  override to accept rpy & ranges ---------------------
    def do_vo(self,
              frame_uv: np.ndarray,
              rpy: tuple[float, float, float] | None = None,
              keyed_ranges: np.ndarray | None = None):
        """
        Same signature as Base, but *requires* rpy and stores range hints.
        """
        if keyed_ranges is not None:
            for pid, dist in keyed_ranges:
                self._range_hints[int(pid)] = float(dist)

        f_idx = len(self.poses)
        if not self.initialised:
            self._bootstrap_with_first_two_frames(
                frame_uv[:, 0].astype(int), frame_uv[:, 1:3], rpy)
        else:
            self._track_and_pose(frame_uv[:, 0].astype(int),
                                 frame_uv[:, 1:3], f_idx, rpy)

        # store obs (unchanged)
        for pid, uv in zip(frame_uv[:, 0].astype(int), frame_uv[:, 1:3]):
            self.obs.setdefault(pid, []).append((f_idx, uv))

        if self.initialised and len(self.poses) >= 3:
            self._bundle_adjust()

        pts_cam = self._points_in_current_cam()
        # return pts_cam, *self.current_pose()
        return pts_cam
